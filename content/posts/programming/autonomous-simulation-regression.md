---
title: "시뮬레이션 회귀 테스트 설계: 시나리오를 데이터로, 파이프라인을 계산으로"
date: 2026-04-29T16:00:00+09:00
draft: false
tags: ["함수형 프로그래밍", "Rust", "설계", "자율주행", "로봇", "테스트", "회귀 테스트", "시뮬레이션"]
categories: ["프로그래밍", "자율주행"]
description: "주행 시나리오를 데이터로, 판단 파이프라인을 순수 계산으로 만들면 시나리오 파일 하나만 추가해도 테스트가 자동으로 늘어납니다. 시뮬레이터 없이 CI에서 수백 가지 엣지 케이스를 검증하는 구조를 설명합니다."
---

## 이 글을 읽고 나면

- 주행 시나리오를 데이터로 표현하는 방법을 이해합니다.
- 판단 파이프라인을 순수 계산으로 만들어 시나리오만 추가하면 테스트가 늘어나는 구조를 봅니다.
- 실차 주행 로그를 회귀 테스트 케이스로 자동 변환하는 패턴을 이해합니다.

이전 글 [액션/계산/데이터](/posts/programming/functional-actions-calculations-data/)와 [자율주행 센서 파이프라인](/posts/programming/autonomous-sensor-pipeline/)을 먼저 읽으면 더 자연스럽게 이어집니다.

---

## 문제: 시뮬레이션에 의존하는 테스트

자율주행 소프트웨어의 전형적인 테스트 방법은 이렇습니다.

1. CARLA, LGSVL 같은 시뮬레이터를 띄운다
2. 시나리오를 설정한다 (날씨, 교통, 도로 형태)
3. 차량을 주행시키며 결과를 관찰한다

문제가 있습니다.

- 시뮬레이터 실행에 수십 초 ~ 수 분이 걸린다
- CI 서버에 GPU가 필요하다
- 시나리오 하나가 실패해도 원인이 시뮬레이터인지 알고리즘인지 알기 어렵다
- 재현이 보장되지 않는다 (시뮬레이터의 물리 엔진 타이밍에 따라 결과가 다를 수 있음)

핵심 질문: **판단 알고리즘 자체가 맞는지 확인하는 데 정말 시뮬레이터가 필요한가?**

대부분의 경우 필요하지 않습니다. 알고리즘이 순수 계산이라면, 입력 데이터만 있으면 충분합니다.

---

## 설계 원칙

```
[시나리오 데이터]      [판단 파이프라인]     [기대 출력]
  장애물 위치,    →     (순수 계산)      →   제어 명령,
  자차 속도,                               안전 판단
  목표 경로...
```

- **시나리오**: 입력값의 집합. JSON/TOML 파일로 표현 가능
- **파이프라인**: 시나리오 → 출력을 계산하는 순수 함수
- **기대 출력**: 어설션. "이 상황에서는 비상정지여야 한다"

시나리오 파일을 추가하면 테스트가 늘어납니다. 파이프라인 코드를 수정하면 모든 시나리오가 자동으로 재검증됩니다.

---

## 데이터: 시나리오 구조체

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Obstacle {
    pub id: u32,
    pub x: f32,         // 자차 기준 전방이 양수 (미터)
    pub y: f32,         // 자차 기준 좌측이 양수 (미터)
    pub vx: f32,        // 종방향 속도 (m/s)
    pub vy: f32,        // 횡방향 속도 (m/s)
    pub width: f32,
    pub length: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EgoState {
    pub speed_mps: f32,         // 자차 속도 (m/s)
    pub heading_rad: f32,       // 자차 헤딩 (라디안)
    pub lateral_offset_m: f32,  // 차선 중앙으로부터의 횡방향 오프셋
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioInput {
    pub name: String,
    pub ego: EgoState,
    pub obstacles: Vec<Obstacle>,
    pub target_speed_mps: f32,
    pub speed_limit_mps: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SafetyVerdict {
    Safe,
    CautionSlowDown { recommended_speed: f32 },
    EmergencyBrake { reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioExpected {
    pub verdict: SafetyVerdict,
    pub max_allowed_speed: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scenario {
    pub input: ScenarioInput,
    pub expected: ScenarioExpected,
}
```

`Scenario` 구조체 하나가 테스트 케이스 하나입니다. JSON으로 직렬화할 수 있으므로 파일로 관리합니다.

---

## 계산: 판단 파이프라인

```rust
#[derive(Debug, Clone)]
pub struct PipelineOutput {
    pub verdict: SafetyVerdict,
    pub applied_speed: f32,
    pub closest_obstacle_m: f32,
}

// 전방 장애물까지 거리 계산 (계산)
pub fn find_closest_forward_obstacle(obstacles: &[Obstacle]) -> Option<f32> {
    obstacles.iter()
        .filter(|o| o.x > 0.0 && o.y.abs() < 2.0) // 전방 2m 폭 안에 있는 것만
        .map(|o| o.x - o.length / 2.0)             // 장애물 앞면까지 거리
        .filter(|d| *d > 0.0)
        .reduce(f32::min)
}

// 안전 거리 계산: 속도에 따라 필요 거리가 달라짐 (계산)
pub fn required_safety_distance(speed_mps: f32) -> f32 {
    let reaction_distance = speed_mps * 0.5; // 반응 시간 0.5초
    let braking_distance  = speed_mps.powi(2) / (2.0 * 4.0); // μg = 4m/s²
    reaction_distance + braking_distance + 2.0 // 여유 2m
}

// 안전 판정 (계산)
pub fn assess_safety(ego: &EgoState, obstacles: &[Obstacle]) -> SafetyVerdict {
    let closest = find_closest_forward_obstacle(obstacles);

    match closest {
        None => SafetyVerdict::Safe,
        Some(dist) => {
            let required = required_safety_distance(ego.speed_mps);
            if dist < 5.0 {
                SafetyVerdict::EmergencyBrake {
                    reason: format!("장애물까지 {:.1}m — 안전 거리 미확보", dist),
                }
            } else if dist < required {
                let safe_speed = ((2.0 * 4.0 * (dist - 2.0)).max(0.0)).sqrt();
                SafetyVerdict::CautionSlowDown {
                    recommended_speed: safe_speed.min(ego.speed_mps),
                }
            } else {
                SafetyVerdict::Safe
            }
        }
    }
}

// 속도 결정 (계산)
pub fn decide_speed(
    ego: &EgoState,
    verdict: &SafetyVerdict,
    target_speed: f32,
    speed_limit: f32,
) -> f32 {
    let base = target_speed.min(speed_limit);
    match verdict {
        SafetyVerdict::Safe                           => base,
        SafetyVerdict::CautionSlowDown { recommended_speed } => {
            base.min(*recommended_speed)
        }
        SafetyVerdict::EmergencyBrake { .. }          => 0.0,
    }
}

// 전체 파이프라인 (계산들의 조합)
pub fn run_pipeline(input: &ScenarioInput) -> PipelineOutput {
    let verdict = assess_safety(&input.ego, &input.obstacles);
    let applied_speed = decide_speed(
        &input.ego, &verdict,
        input.target_speed_mps,
        input.speed_limit_mps,
    );
    let closest_obstacle_m = find_closest_forward_obstacle(&input.obstacles)
        .unwrap_or(f32::INFINITY);

    PipelineOutput { verdict, applied_speed, closest_obstacle_m }
}
```

`run_pipeline`은 `ScenarioInput`만 받아서 `PipelineOutput`을 돌려줍니다. 시뮬레이터도, ROS2도, 하드웨어도 없습니다.

---

## 시나리오 파일 관리

시나리오를 TOML 파일로 관리합니다.

```toml
# tests/scenarios/emergency_brake_close_obstacle.toml
[input]
name = "긴급정지: 전방 3m 장애물"
target_speed_mps = 10.0
speed_limit_mps = 13.9

[input.ego]
speed_mps = 10.0
heading_rad = 0.0
lateral_offset_m = 0.0

[[input.obstacles]]
id = 1
x = 3.0    # 전방 3m
y = 0.0
vx = 0.0
vy = 0.0
width = 1.8
length = 4.5

[expected]
max_allowed_speed = 0.0

[expected.verdict]
type = "EmergencyBrake"
```

```toml
# tests/scenarios/slow_down_medium_distance.toml
[input]
name = "감속: 전방 20m 장애물, 속도 30km/h"
target_speed_mps = 8.3
speed_limit_mps = 13.9

[input.ego]
speed_mps = 8.3
heading_rad = 0.0
lateral_offset_m = 0.0

[[input.obstacles]]
id = 1
x = 20.0
y = 0.3
vx = 3.0
vy = 0.0
width = 1.8
length = 4.5

[expected.verdict]
type = "CautionSlowDown"
```

```toml
# tests/scenarios/clear_road.toml
[input]
name = "장애물 없음: 정상 주행"
target_speed_mps = 13.9
speed_limit_mps = 13.9

[input.ego]
speed_mps = 13.9
heading_rad = 0.0
lateral_offset_m = 0.0

[expected.verdict]
type = "Safe"
```

---

## 테스트 러너: 파일을 읽어 자동 실행

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn load_scenario(path: &str) -> Scenario {
        let content = fs::read_to_string(path)
            .unwrap_or_else(|_| panic!("시나리오 파일 없음: {}", path));
        toml::from_str(&content)
            .unwrap_or_else(|e| panic!("파싱 실패 {}: {}", path, e))
    }

    fn run_scenario(scenario: &Scenario) {
        let output = run_pipeline(&scenario.input);

        // verdict 타입 검사
        let verdict_matches = match (&output.verdict, &scenario.expected.verdict) {
            (SafetyVerdict::Safe, SafetyVerdict::Safe) => true,
            (SafetyVerdict::EmergencyBrake { .. }, SafetyVerdict::EmergencyBrake { .. }) => true,
            (SafetyVerdict::CautionSlowDown { .. }, SafetyVerdict::CautionSlowDown { .. }) => true,
            _ => false,
        };

        assert!(
            verdict_matches,
            "시나리오 '{}': 예상={:?}, 실제={:?}",
            scenario.input.name, scenario.expected.verdict, output.verdict
        );

        // 속도 상한 검사
        if let Some(max_speed) = scenario.expected.max_allowed_speed {
            assert!(
                output.applied_speed <= max_speed + 0.01,
                "시나리오 '{}': 속도 {:.2} > 허용 상한 {:.2}",
                scenario.input.name, output.applied_speed, max_speed
            );
        }
    }

    // 시나리오 디렉토리 전체를 스캔해서 실행
    #[test]
    fn run_all_scenarios() {
        let scenario_dir = "tests/scenarios";

        // 디렉토리가 없으면 스킵 (CI 환경 대비)
        if !std::path::Path::new(scenario_dir).exists() {
            return;
        }

        let entries = fs::read_dir(scenario_dir).unwrap();
        let mut count = 0;

        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("toml") {
                let scenario = load_scenario(path.to_str().unwrap());
                run_scenario(&scenario);
                count += 1;
            }
        }

        println!("시나리오 {}개 통과", count);
    }

    // 개별 시나리오 테스트 (빠른 피드백용)
    #[test]
    fn test_emergency_brake() {
        let scenario = Scenario {
            input: ScenarioInput {
                name: "인라인 긴급정지 테스트".into(),
                ego: EgoState { speed_mps: 10.0, heading_rad: 0.0, lateral_offset_m: 0.0 },
                obstacles: vec![Obstacle {
                    id: 1, x: 3.0, y: 0.0, vx: 0.0, vy: 0.0,
                    width: 1.8, length: 4.5,
                }],
                target_speed_mps: 10.0,
                speed_limit_mps: 13.9,
            },
            expected: ScenarioExpected {
                verdict: SafetyVerdict::EmergencyBrake { reason: String::new() },
                max_allowed_speed: Some(0.0),
            },
        };
        run_scenario(&scenario);
    }

    #[test]
    fn test_clear_road() {
        let scenario = Scenario {
            input: ScenarioInput {
                name: "인라인 클리어 로드".into(),
                ego: EgoState { speed_mps: 13.9, heading_rad: 0.0, lateral_offset_m: 0.0 },
                obstacles: vec![],
                target_speed_mps: 13.9,
                speed_limit_mps: 13.9,
            },
            expected: ScenarioExpected {
                verdict: SafetyVerdict::Safe,
                max_allowed_speed: None,
            },
        };
        run_scenario(&scenario);
    }
}
```

`tests/scenarios/` 디렉토리에 TOML 파일을 추가하면 `cargo test`가 자동으로 실행합니다. 코드 수정 없이 테스트 케이스가 늘어납니다.

---

## 실차 로그를 시나리오로 변환

실차 주행 중 아슬아슬했던 상황을 로그에서 뽑아 회귀 테스트로 고정합니다.

```rust
#[derive(Debug, Deserialize)]
struct RawDriveLog {
    timestamp_ns: u64,
    ego_speed: f32,
    ego_heading: f32,
    obstacles: Vec<RawObstacle>,
    actual_verdict: String,
}

fn log_to_scenario(log: &RawDriveLog, expected_verdict: SafetyVerdict) -> Scenario {
    Scenario {
        input: ScenarioInput {
            name: format!("로그 재현 ts={}", log.timestamp_ns),
            ego: EgoState {
                speed_mps: log.ego_speed,
                heading_rad: log.ego_heading,
                lateral_offset_m: 0.0,
            },
            obstacles: log.obstacles.iter().map(|o| Obstacle {
                id: o.id, x: o.x, y: o.y,
                vx: o.vx, vy: o.vy,
                width: o.width, length: o.length,
            }).collect(),
            target_speed_mps: log.ego_speed,
            speed_limit_mps: 13.9,
        },
        expected: ScenarioExpected {
            verdict: expected_verdict,
            max_allowed_speed: None,
        },
    }
}
```

필드 테스트에서 발생한 엣지 케이스를 로그에서 추출하면 그 상황이 영구적인 회귀 테스트로 고정됩니다. 알고리즘을 수정해도 그 케이스가 깨지는 순간 CI가 즉시 알려줍니다.

---

## 전체 흐름

```
시나리오 파일 (TOML)
    │
    ▼
ScenarioInput 구조체 (데이터)
    │
    ▼
run_pipeline() (계산 — 시뮬레이터 없음)
    │
    ▼
PipelineOutput 구조체 (데이터)
    │
    ▼
어설션: expected vs actual
```

시뮬레이터는 이 흐름에 없습니다. 시뮬레이터는 시각적 확인, 전체 스택 통합 테스트, 물리 현상 검증에 씁니다. 판단 알고리즘의 정확성은 이 파이프라인으로 충분합니다.

---

## 정리

| | 시뮬레이터 기반 테스트 | 시나리오 기반 단위 테스트 |
|---|---|---|
| 실행 속도 | 수십 초 ~ 수 분 | 수 밀리초 |
| CI 비용 | GPU 서버 필요 | 일반 서버 |
| 재현성 | 물리 엔진 타이밍에 따라 다름 | 완전 결정론적 |
| 엣지 케이스 추가 | 시나리오 스크립트 작성 | TOML 파일 추가 |
| 실패 원인 파악 | 시뮬레이터인지 알고리즘인지 불명확 | 알고리즘만 검증 |

두 가지는 대체 관계가 아닙니다. 시나리오 기반 단위 테스트가 빠른 피드백을 담당하고, 시뮬레이터가 전체 통합을 담당합니다. 알고리즘 변경의 90%는 시뮬레이터 없이 검증할 수 있습니다.

---

*관련 글: [액션/계산/데이터](/posts/programming/functional-actions-calculations-data/), [자율주행 센서 파이프라인](/posts/programming/autonomous-sensor-pipeline/), [함수형 PID 제어기](/posts/programming/autonomous-pid-controller/), [ROS2 콜백을 함수형으로](/posts/programming/autonomous-ros2-functional/)*
