---
title: "자율주행 모드 전이를 타입으로 만들기: 상태 기계 패턴"
date: 2026-04-29T10:00:00+09:00
draft: false
tags: ["함수형 프로그래밍", "Rust", "설계", "자율주행", "로봇", "상태 기계", "타입스테이트"]
categories: ["프로그래밍", "자율주행"]
description: "자율주행 차량의 운행 모드 전이를 enum과 타입스테이트 패턴으로 표현하면, 잘못된 모드 전환을 컴파일러가 원천 차단합니다."
---

## 이 글을 읽고 나면

- 자율주행 모드 전이를 `enum`으로 안전하게 표현하는 방법을 이해합니다.
- 컴파일러가 불가능한 상태 전환을 막는 타입스테이트 패턴을 봅니다.
- 긴급정지 같은 안전 임계 전이를 설계 수준에서 보장하는 방법을 압니다.

이전 글 [함수형으로 상태 기계 만들기](/posts/programming/functional-state-machine/)를 먼저 읽으면 더 자연스럽게 이어집니다.

---

## 자율주행 모드란

자율주행 차량은 여러 운행 모드를 오갑니다. 대표적인 구성은 이렇습니다.

```
        [운전자 개입]           [자율주행 활성화]
Manual ──────────────→ Supervised ──────────────→ Autonomous
  ↑                        ↑                          |
  |                        └──────── [조건 미충족] ────┘
  |
  └──────────────────── [긴급정지] ──────────────────────
                     EmergencyStop (어느 모드에서든)
```

이 전이 규칙이 소프트웨어에서 제대로 지켜지지 않으면 안전 사고로 이어집니다. 코드 리뷰에만 의존하는 대신, 컴파일러가 잘못된 전이를 거부하도록 설계할 수 있습니다.

---

## 나쁜 방법: 문자열 상태

```rust
struct Vehicle {
    mode: String, // "manual", "supervised", "autonomous", "emergency"
    speed: f32,
}

fn set_mode(vehicle: &mut Vehicle, new_mode: &str) {
    // 전이 규칙? 검증? 없음.
    vehicle.mode = new_mode.to_string();
}
```

`set_mode(vehicle, "autonomus")` 처럼 오타를 내도 컴파일이 됩니다. `"emergency"` 상태에서 `"autonomous"`로 전환해도 아무도 막지 않습니다.

---

## 좋은 방법: `enum`으로 상태 표현

```rust
#[derive(Debug, Clone, PartialEq)]
enum DrivingMode {
    Manual,
    Supervised { override_available: bool },
    Autonomous { confidence: f32 },
    EmergencyStop { reason: String },
}
```

각 상태가 필요한 데이터를 직접 들고 다닙니다. `Autonomous` 상태는 인지 신뢰도를, `EmergencyStop`은 원인을 포함합니다.

---

## 이벤트와 전이 함수

```rust
#[derive(Debug)]
enum ModeEvent {
    DriverActivates,                    // 운전자가 자율주행 활성화 요청
    DriverTakesOver,                    // 운전자가 수동으로 전환
    PerceptionConfident { score: f32 }, // 인지 시스템이 신뢰도 보고
    PerceptionDegraded,                 // 인지 신뢰도 하락
    EmergencyDetected { reason: String }, // 긴급 상황 감지
    SystemCleared,                      // 긴급 상황 해소, 재시작
}

fn transition(mode: DrivingMode, event: ModeEvent) -> Result<DrivingMode, String> {
    match (mode, event) {
        // Manual → Supervised: 운전자가 자율주행 활성화
        (DrivingMode::Manual, ModeEvent::DriverActivates) => {
            Ok(DrivingMode::Supervised { override_available: true })
        }

        // Supervised → Autonomous: 인지 신뢰도가 충분할 때
        (DrivingMode::Supervised { .. }, ModeEvent::PerceptionConfident { score })
            if score >= 0.85 =>
        {
            Ok(DrivingMode::Autonomous { confidence: score })
        }

        // Autonomous → Supervised: 인지 신뢰도 하락
        (DrivingMode::Autonomous { .. }, ModeEvent::PerceptionDegraded) => {
            Ok(DrivingMode::Supervised { override_available: true })
        }

        // 어느 모드에서든 → EmergencyStop
        (_, ModeEvent::EmergencyDetected { reason }) => {
            Ok(DrivingMode::EmergencyStop { reason })
        }

        // EmergencyStop → Manual: 시스템 점검 후 복귀
        (DrivingMode::EmergencyStop { .. }, ModeEvent::SystemCleared) => {
            Ok(DrivingMode::Manual)
        }

        // 수동 전환은 항상 허용
        (_, ModeEvent::DriverTakesOver) => {
            Ok(DrivingMode::Manual)
        }

        // 그 외 모든 조합은 허용하지 않음
        (mode, event) => Err(format!(
            "{:?} 모드에서 {:?} 이벤트는 허용되지 않습니다", mode, event
        )),
    }
}
```

`transition`은 순수한 계산입니다. 차량 제어도, 알림도 없습니다. 현재 모드와 이벤트만 받아서 다음 모드를 돌려줍니다.

---

## 테스트

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manual_to_supervised() {
        let next = transition(DrivingMode::Manual, ModeEvent::DriverActivates).unwrap();
        assert_eq!(next, DrivingMode::Supervised { override_available: true });
    }

    #[test]
    fn test_supervised_to_autonomous_needs_high_confidence() {
        // 신뢰도 부족: 전이 불가
        let result = transition(
            DrivingMode::Supervised { override_available: true },
            ModeEvent::PerceptionConfident { score: 0.70 },
        );
        assert!(result.is_err());

        // 신뢰도 충분: 전이 가능
        let next = transition(
            DrivingMode::Supervised { override_available: true },
            ModeEvent::PerceptionConfident { score: 0.90 },
        ).unwrap();
        assert_eq!(next, DrivingMode::Autonomous { confidence: 0.90 });
    }

    #[test]
    fn test_emergency_from_any_mode() {
        for mode in [
            DrivingMode::Manual,
            DrivingMode::Supervised { override_available: true },
            DrivingMode::Autonomous { confidence: 0.95 },
        ] {
            let next = transition(
                mode,
                ModeEvent::EmergencyDetected { reason: "장애물 회피 불가".into() },
            ).unwrap();
            assert!(matches!(next, DrivingMode::EmergencyStop { .. }));
        }
    }

    #[test]
    fn test_cannot_go_autonomous_from_emergency() {
        let result = transition(
            DrivingMode::EmergencyStop { reason: "테스트".into() },
            ModeEvent::PerceptionConfident { score: 0.99 },
        );
        assert!(result.is_err());
    }
}
```

차량 없이, 시뮬레이터 없이 모드 전이 로직 전체를 검증합니다.

---

## 부수효과는 바깥에서

모드가 바뀔 때 경고음, 대시보드 업데이트, 로그 기록 같은 일이 필요합니다. 이것은 액션입니다. 전이 함수 안에 넣으면 계산이 오염됩니다.

```rust
fn apply_mode_event(
    current: DrivingMode,
    event: ModeEvent,
    dashboard: &Dashboard,
    logger: &Logger,
) -> Result<DrivingMode, String> {
    let next = transition(current, event)?; // 계산: 다음 모드 결정

    // 액션: 모드 진입 시 부수효과
    match &next {
        DrivingMode::Autonomous { confidence } => {
            dashboard.show_autonomous_indicator(*confidence);
            logger.info(&format!("자율주행 활성화 (신뢰도: {:.2})", confidence));
        }
        DrivingMode::EmergencyStop { reason } => {
            dashboard.alert_emergency(reason);
            logger.error(&format!("긴급정지: {}", reason));
        }
        DrivingMode::Manual => {
            dashboard.show_manual_indicator();
        }
        _ => {}
    }

    Ok(next)
}
```

---

## 타입스테이트 패턴: 컴파일러가 전이를 막게 하기

`enum` 기반은 잘못된 전이를 런타임 `Err`로 반환합니다. 한 걸음 더 나아가면 컴파일 타임에 막을 수 있습니다.

각 모드를 별도 타입으로 만들고, 가능한 전이만 메서드로 구현합니다.

```rust
struct ManualMode;
struct SupervisedMode { override_available: bool }
struct AutonomousMode { confidence: f32 }
struct EmergencyMode  { reason: String }

impl ManualMode {
    fn activate(self) -> SupervisedMode {
        SupervisedMode { override_available: true }
    }
    fn emergency(self, reason: String) -> EmergencyMode {
        EmergencyMode { reason }
    }
}

impl SupervisedMode {
    fn engage(self, confidence: f32) -> Result<AutonomousMode, SupervisedMode> {
        if confidence >= 0.85 {
            Ok(AutonomousMode { confidence })
        } else {
            Err(self) // 신뢰도 부족 시 Supervised로 돌아옴
        }
    }
    fn take_over(self) -> ManualMode { ManualMode }
    fn emergency(self, reason: String) -> EmergencyMode { EmergencyMode { reason } }
}

impl AutonomousMode {
    fn degrade(self) -> SupervisedMode {
        SupervisedMode { override_available: true }
    }
    fn take_over(self) -> ManualMode { ManualMode }
    fn emergency(self, reason: String) -> EmergencyMode { EmergencyMode { reason } }
}

impl EmergencyMode {
    fn clear(self) -> ManualMode { ManualMode }
    // emergency에서 autonomous()는 메서드 자체가 없음 → 컴파일 불가
}
```

이제 이 코드는 **컴파일 자체가 되지 않습니다.**

```rust
fn bad_sequence() {
    let emergency = EmergencyMode { reason: "충돌".into() };
    // emergency.engage(0.99); // 컴파일 에러: EmergencyMode에 engage() 없음
}
```

가능한 전이만 타입 시스템에 새겨져 있으므로, 실수로 잘못된 전이를 넣으면 컴파일러가 즉시 알려줍니다.

---

## 두 방식 비교

| | `enum` 기반 | 타입스테이트 |
|---|---|---|
| 잘못된 전이 감지 | 런타임 (`Result::Err`) | 컴파일 타임 |
| 여러 모드를 컬렉션에 담기 | 가능 (`Vec<DrivingMode>`) | 어려움 |
| 구현 복잡도 | 낮음 | 높음 |
| 적합한 경우 | 모드가 많고 동적인 경우 | 안전 임계 전이가 있는 경우 |

자율주행처럼 안전이 중요한 시스템에서는 타입스테이트가 특히 가치 있습니다. `EmergencyStop`에서 `Autonomous`로 가는 코드는 작성 자체가 불가능해야 합니다.

---

## 이벤트 이력으로 상태 재현하기

전이 함수가 순수한 계산이면, 이벤트 목록을 재생해서 언제든 상태를 복원할 수 있습니다.

```rust
fn replay(events: &[ModeEvent]) -> Result<DrivingMode, String> {
    let events_clone: Vec<ModeEvent> = events.to_vec();
    events_clone.into_iter().try_fold(
        DrivingMode::Manual,
        |mode, event| transition(mode, event),
    )
}
```

주행 로그에서 이벤트 목록을 읽어 오면, 사고 당시의 모드 상태를 정확히 재현할 수 있습니다. 디버깅과 사후 분석이 자연스럽게 따라옵니다.

---

## 정리

1. **상태와 이벤트를 `enum`으로** — 오타는 컴파일 에러, 빠뜨린 분기는 컴파일러가 경고
2. **전이 함수는 순수 계산으로** — 차량 없이 단위 테스트, 이벤트 소싱과 자연스럽게 연결
3. **부수효과는 전이 바깥에서** — 경고음·로그·대시보드는 다음 모드를 알고 나서 결정
4. **안전 임계 전이는 타입스테이트로** — `EmergencyStop → Autonomous`는 메서드 자체가 없어야

---

*관련 글: [함수형으로 상태 기계 만들기](/posts/programming/functional-state-machine/), [자율주행 센서 파이프라인](/posts/programming/autonomous-sensor-pipeline/), [함수형 센서 퓨전](/posts/programming/autonomous-sensor-fusion/), [함수형 PID 제어기](/posts/programming/autonomous-pid-controller/), [시뮬레이션 회귀 테스트 설계](/posts/programming/autonomous-simulation-regression/)*
