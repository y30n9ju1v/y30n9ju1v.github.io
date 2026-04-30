---
title: "자율주행 센서 파이프라인: 액션/계산/데이터로 설계하기"
date: 2026-04-29T09:00:00+09:00
draft: false
tags: ["함수형 프로그래밍", "Rust", "설계", "자율주행", "로봇", "센서", "액션/계산/데이터"]
categories: ["프로그래밍", "자율주행"]
description: "LiDAR와 카메라 데이터를 처리하는 자율주행 파이프라인을 액션/계산/데이터로 나누면 테스트와 시뮬레이션이 얼마나 쉬워지는지 설명합니다."
---

## 이 글을 읽고 나면

- 자율주행 센서 파이프라인을 액션/계산/데이터로 분류할 수 있습니다.
- 하드웨어 없이도 파이프라인 로직을 단위 테스트하는 방법을 이해합니다.
- 시뮬레이터와 실차를 같은 계산 코드로 구동하는 설계를 볼 수 있습니다.

이전 글 [액션/계산/데이터](/posts/programming/functional/functional-actions-calculations-data/)를 먼저 읽으면 더 자연스럽게 이어집니다.

---

## 문제: 센서 코드는 왜 테스트하기 어려운가

자율주행 소프트웨어를 처음 짜면 이런 함수가 나오기 쉽습니다.

```rust
fn process_lidar(sensor: &LidarDevice) {
    let raw = sensor.read_points(); // 하드웨어에서 읽음
    let filtered = raw.iter()
        .filter(|p| p.distance < 80.0 && p.intensity > 10)
        .collect::<Vec<_>>();
    let clusters = dbscan_cluster(&filtered);

    if clusters.iter().any(|c| c.distance_to_ego < 5.0) {
        vehicle_controller::emergency_brake(); // 차량 제어
    }

    telemetry::send("obstacles", &clusters); // 원격 전송
}
```

이 함수 하나에 세 가지가 섞여 있습니다.

- `sensor.read_points()` — 하드웨어 I/O (액션)
- `filter`, `dbscan_cluster` — 포인트 클라우드 처리 (계산)
- `emergency_brake`, `telemetry::send` — 차량 제어와 네트워크 (액션)

테스트하려면 LiDAR 하드웨어가 있어야 합니다. 시뮬레이터로 바꾸려면 함수 전체를 손대야 합니다. 알고리즘만 바꿔도 브레이크 로직까지 영향을 받습니다.

---

## 세 가지로 나누기

### 데이터 먼저

센서가 만들어 내는 것, 알고리즘이 주고받는 것을 데이터로 정의합니다.

```rust
#[derive(Debug, Clone)]
struct LidarPoint {
    x: f32,
    y: f32,
    z: f32,
    intensity: f32,
    distance: f32, // 원점으로부터의 거리 (미터)
}

#[derive(Debug, Clone)]
struct Obstacle {
    center_x: f32,
    center_y: f32,
    distance_to_ego: f32, // 자차로부터의 거리 (미터)
    point_count: usize,
}

#[derive(Debug, Clone)]
struct ControlCommand {
    target_velocity: f32,    // m/s
    target_steering: f32,    // rad
    emergency_brake: bool,
}
```

이 구조체들은 그 자체로는 아무 일도 하지 않습니다. 하드웨어도 없고, 네트워크도 없습니다. 어디서든 만들고 복사하고 비교할 수 있습니다.

---

### 계산: 순수한 처리 로직

데이터를 받아서 데이터를 돌려주는 함수들입니다. 하드웨어 의존이 전혀 없습니다.

```rust
// 노이즈 필터링: 거리와 반사강도 기준
fn filter_points(points: &[LidarPoint]) -> Vec<LidarPoint> {
    points.iter()
        .filter(|p| p.distance < 80.0 && p.intensity > 10.0)
        .cloned()
        .collect()
}

// 간단한 격자 기반 클러스터링
fn cluster_obstacles(points: &[LidarPoint]) -> Vec<Obstacle> {
    // 실제 구현은 DBSCAN 등을 쓰지만, 여기서는 구조를 보여주는 것이 목적
    let mut clusters: Vec<Obstacle> = Vec::new();

    for point in points {
        let merged = clusters.iter_mut().find(|c| {
            let dx = c.center_x - point.x;
            let dy = c.center_y - point.y;
            (dx * dx + dy * dy).sqrt() < 1.5 // 1.5m 이내면 같은 클러스터
        });

        if let Some(cluster) = merged {
            let n = cluster.point_count as f32;
            cluster.center_x = (cluster.center_x * n + point.x) / (n + 1.0);
            cluster.center_y = (cluster.center_y * n + point.y) / (n + 1.0);
            cluster.distance_to_ego = (cluster.center_x.powi(2) + cluster.center_y.powi(2)).sqrt();
            cluster.point_count += 1;
        } else {
            clusters.push(Obstacle {
                center_x: point.x,
                center_y: point.y,
                distance_to_ego: point.distance,
                point_count: 1,
            });
        }
    }
    clusters
}

// 장애물 목록을 보고 제어 명령 결정
fn decide_command(obstacles: &[Obstacle], desired_velocity: f32) -> ControlCommand {
    let closest = obstacles.iter()
        .map(|o| o.distance_to_ego)
        .fold(f32::INFINITY, f32::min);

    if closest < 5.0 {
        ControlCommand { target_velocity: 0.0, target_steering: 0.0, emergency_brake: true }
    } else if closest < 15.0 {
        let slow_velocity = desired_velocity * (closest / 15.0);
        ControlCommand { target_velocity: slow_velocity, target_steering: 0.0, emergency_brake: false }
    } else {
        ControlCommand { target_velocity: desired_velocity, target_steering: 0.0, emergency_brake: false }
    }
}
```

`filter_points`, `cluster_obstacles`, `decide_command` — 세 함수 모두 입력만 있으면 실행됩니다. 하드웨어도, 네트워크도, 전역 상태도 없습니다.

---

### 액션: 경계에서만 I/O

I/O를 다루는 코드는 파이프라인의 시작과 끝에만 있습니다.

```rust
// 액션: 하드웨어에서 읽기
fn read_lidar(device: &LidarDevice) -> Vec<LidarPoint> {
    device.read_raw_points()
}

// 액션: 차량 제어기에 명령 전송
fn send_command(controller: &VehicleController, cmd: &ControlCommand) {
    controller.apply(cmd);
}

// 액션: 텔레메트리 전송
fn send_telemetry(client: &TelemetryClient, obstacles: &[Obstacle]) {
    client.send("obstacles", obstacles);
}

// 오케스트레이션: 액션이 계산을 감싼다
fn run_pipeline(
    device: &LidarDevice,
    controller: &VehicleController,
    telemetry: &TelemetryClient,
    desired_velocity: f32,
) {
    let raw     = read_lidar(device);                   // 액션
    let filtered = filter_points(&raw);                 // 계산
    let obstacles = cluster_obstacles(&filtered);       // 계산
    let command  = decide_command(&obstacles, desired_velocity); // 계산

    send_command(controller, &command);                 // 액션
    send_telemetry(telemetry, &obstacles);              // 액션
}
```

구조가 명확해졌습니다.

```
[액션] read_lidar
    ↓
[계산] filter_points → cluster_obstacles → decide_command
    ↓
[액션] send_command, send_telemetry
```

---

## 하드웨어 없이 테스트하기

계산 함수들은 데이터만 넘기면 테스트됩니다.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn make_point(x: f32, y: f32, dist: f32) -> LidarPoint {
        LidarPoint { x, y, z: 0.0, intensity: 50.0, distance: dist }
    }

    #[test]
    fn test_filter_removes_far_and_weak_points() {
        let points = vec![
            make_point(1.0, 0.0, 1.0),    // 통과
            make_point(0.0, 0.0, 90.0),   // 너무 멀어서 제거
        ];
        let filtered = filter_points(&points);
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn test_emergency_brake_within_5m() {
        let obstacles = vec![Obstacle {
            center_x: 3.0, center_y: 0.0,
            distance_to_ego: 3.0,
            point_count: 10,
        }];
        let cmd = decide_command(&obstacles, 10.0);
        assert!(cmd.emergency_brake);
        assert_eq!(cmd.target_velocity, 0.0);
    }

    #[test]
    fn test_slow_down_between_5_and_15m() {
        let obstacles = vec![Obstacle {
            center_x: 10.0, center_y: 0.0,
            distance_to_ego: 10.0,
            point_count: 5,
        }];
        let cmd = decide_command(&obstacles, 20.0);
        assert!(!cmd.emergency_brake);
        assert!(cmd.target_velocity < 20.0);
    }

    #[test]
    fn test_full_speed_when_clear() {
        let cmd = decide_command(&[], 20.0);
        assert_eq!(cmd.target_velocity, 20.0);
        assert!(!cmd.emergency_brake);
    }
}
```

LiDAR 장치도, ROS도, 시뮬레이터도 없이 알고리즘 로직 전체를 검증합니다.

---

## 시뮬레이터와 실차를 같은 코드로

이 설계의 가장 큰 장점은 **계산 코드가 실차와 시뮬레이터에서 동일하게 작동**한다는 것입니다.

```rust
trait LidarSource {
    fn read_points(&self) -> Vec<LidarPoint>;
}

struct RealLidar { /* 하드웨어 드라이버 */ }
struct SimLidar  { scenario: Vec<LidarPoint> }

impl LidarSource for RealLidar {
    fn read_points(&self) -> Vec<LidarPoint> {
        // 실제 드라이버 호출
        hardware_driver::read()
    }
}

impl LidarSource for SimLidar {
    fn read_points(&self) -> Vec<LidarPoint> {
        self.scenario.clone() // 미리 준비한 데이터 반환
    }
}

fn run_pipeline_generic(source: &dyn LidarSource, desired_velocity: f32) -> ControlCommand {
    let raw       = source.read_points();      // 액션 — 소스만 교체
    let filtered  = filter_points(&raw);       // 계산 — 동일
    let obstacles = cluster_obstacles(&filtered); // 계산 — 동일
    decide_command(&obstacles, desired_velocity)  // 계산 — 동일
}
```

알고리즘을 건드리지 않고 데이터 소스만 바꿉니다. 실차 테스트를 거치기 전에 수천 가지 시나리오를 시뮬레이터로 검증할 수 있습니다. 회귀 테스트도 CI에서 자동으로 돌립니다.

---

## 정리

| 분류 | 예시 | 특징 |
|------|------|------|
| 데이터 | `LidarPoint`, `Obstacle`, `ControlCommand` | 하드웨어 의존 없음, 어디서든 만들 수 있음 |
| 계산 | `filter_points`, `cluster_obstacles`, `decide_command` | 순수 함수, 단위 테스트 trivial |
| 액션 | `read_lidar`, `send_command`, `send_telemetry` | 파이프라인의 시작과 끝에만 존재 |

자율주행 파이프라인에서 버그가 가장 많이 나는 곳은 센서 처리 로직입니다. 그 로직을 순수한 계산으로 만들면 하드웨어 없이도 수백 개의 시나리오를 단위 테스트로 검증할 수 있습니다. 실차 주행 시간이 줄어들고, 엣지 케이스를 더 빨리 찾아냅니다.

---

*관련 글: [액션/계산/데이터](/posts/programming/functional/functional-actions-calculations-data/), [함수 컴포지션](/posts/programming/functional/functional-composition/), [자율주행 모드 전이를 타입으로 만들기](/posts/programming/functional/autonomous-state-machine/), [함수형 센서 퓨전](/posts/programming/functional/autonomous-sensor-fusion/), [로봇 경로 계획과 불변 데이터](/posts/programming/functional/autonomous-path-planning/), [ROS2 콜백을 함수형으로](/posts/programming/functional/autonomous-ros2-functional/), [시뮬레이션 회귀 테스트 설계](/posts/programming/functional/autonomous-simulation-regression/)*
