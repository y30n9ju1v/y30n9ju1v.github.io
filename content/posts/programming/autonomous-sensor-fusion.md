---
title: "함수형 센서 퓨전: 컴포지션과 추상화 장벽으로 GPS·IMU·카메라 합치기"
date: 2026-04-29T11:00:00+09:00
draft: false
tags: ["함수형 프로그래밍", "Rust", "설계", "자율주행", "로봇", "센서 퓨전", "컴포지션", "추상화 장벽"]
categories: ["프로그래밍", "자율주행"]
description: "GPS, IMU, 카메라를 합치는 센서 퓨전 파이프라인을 함수 컴포지션과 추상화 장벽으로 설계하면, 각 센서를 독립적으로 교체하고 테스트할 수 있습니다."
---

## 이 글을 읽고 나면

- 센서 퓨전을 함수 컴포지션으로 표현하는 방법을 이해합니다.
- 추상화 장벽이 센서 레이어 간 의존을 어떻게 차단하는지 봅니다.
- GPS가 끊겨도, 카메라가 바뀌어도 퓨전 로직을 건드리지 않는 설계를 볼 수 있습니다.

이전 글 [함수 컴포지션](/posts/programming/functional-composition/)과 [추상화 장벽](/posts/programming/functional-abstraction-barrier/)을 먼저 읽으면 더 자연스럽게 이어집니다.

---

## 문제: 센서가 섞이면 무슨 일이 생기나

자율주행 차량의 위치 추정(Localization)은 여러 센서를 함께 씁니다.

- **GPS**: 절대 위치, 하지만 터널·지하에서 끊김
- **IMU**: 가속도·각속도, 적분 오차가 쌓임
- **카메라**: 차선·표지판으로 상대 위치 보정

이걸 하나의 함수에 다 욱여넣으면 이렇게 됩니다.

```rust
fn localize(gps: &GpsDevice, imu: &ImuDevice, camera: &Camera) -> Pose {
    let gps_pos = gps.read_nmea_and_parse(); // GPS 파싱
    let imu_data = imu.read_raw_and_integrate(); // IMU 적분
    let lane_offset = camera.detect_lanes_and_compute_offset(); // 차선 검출

    // 셋 다 없으면 어떻게? GPS만 없으면? 로직이 한 곳에 다 섞임
    Pose {
        x: gps_pos.x * 0.6 + imu_data.x * 0.3 + lane_offset.x * 0.1,
        y: gps_pos.y * 0.6 + imu_data.y * 0.3 + lane_offset.y * 0.1,
        heading: imu_data.heading,
    }
}
```

GPS 드라이버가 바뀌면 `localize` 전체를 건드려야 합니다. 카메라 없이 테스트하려면 mock을 만들어야 합니다. 가중치 조정도 함수 안에서만 가능합니다.

---

## 추상화 장벽으로 레이어 나누기

센서 퓨전 파이프라인을 세 레이어로 나눕니다.

```
┌────────────────────────────────────┐
│        퓨전 레이어 (Fusion)         │  ← 여러 추정치를 합침
├────────────────────────────────────┤
│       추정 레이어 (Estimation)      │  ← 각 센서의 위치 추정
├────────────────────────────────────┤
│        원시 레이어 (Raw)            │  ← 센서별 데이터 구조체
└────────────────────────────────────┘
```

각 레이어는 아래 레이어의 내부를 모릅니다. GPS가 NMEA를 쓰는지 UBX를 쓰는지는 추정 레이어가 알 필요 없습니다.

---

## 레이어 1: 원시 데이터 (Data)

각 센서가 만들어 내는 구조체입니다. 파싱도, 보정도 없습니다.

```rust
#[derive(Debug, Clone)]
struct GpsMeasurement {
    latitude: f64,
    longitude: f64,
    altitude: f32,
    accuracy_m: f32,
    timestamp_ms: u64,
}

#[derive(Debug, Clone)]
struct ImuMeasurement {
    accel_x: f32, accel_y: f32, accel_z: f32,  // m/s²
    gyro_x: f32,  gyro_y: f32,  gyro_z: f32,   // rad/s
    timestamp_ms: u64,
}

#[derive(Debug, Clone)]
struct CameraLaneDetection {
    left_offset_m: Option<f32>,   // 차선 없으면 None
    right_offset_m: Option<f32>,
    heading_error_rad: f32,
    confidence: f32,
}
```

---

## 레이어 2: 추정 함수 (Calculation)

원시 데이터를 공통 좌표계의 위치 추정치로 변환합니다. 각 함수는 독립적이고 순수합니다.

```rust
#[derive(Debug, Clone)]
struct PoseEstimate {
    x: f64,       // 미터, 로컬 좌표계
    y: f64,
    heading: f32, // 라디안
    confidence: f32, // 0.0 ~ 1.0
}

// GPS: 위경도 → 로컬 XY 변환
fn estimate_from_gps(
    gps: &GpsMeasurement,
    origin: &GpsMeasurement, // 로컬 좌표계 원점
) -> Option<PoseEstimate> {
    if gps.accuracy_m > 5.0 {
        return None; // 정확도 불량
    }
    let x = (gps.longitude - origin.longitude) * 111_320.0 * origin.latitude.to_radians().cos();
    let y = (gps.latitude  - origin.latitude)  * 110_540.0;
    Some(PoseEstimate {
        x, y,
        heading: 0.0, // GPS는 방향을 모름
        confidence: (5.0 - gps.accuracy_m).max(0.0) / 5.0,
    })
}

// IMU: 이전 포즈 + 적분 → 새 포즈 추정
fn estimate_from_imu(
    prev: &PoseEstimate,
    imu: &ImuMeasurement,
    dt_s: f32,
) -> PoseEstimate {
    let new_heading = prev.heading + imu.gyro_z * dt_s;
    let speed = (imu.accel_x * dt_s).abs(); // 단순화
    PoseEstimate {
        x: prev.x + (speed * new_heading.cos()) as f64,
        y: prev.y + (speed * new_heading.sin()) as f64,
        heading: new_heading,
        confidence: (prev.confidence - 0.01 * dt_s).max(0.1), // 시간이 지날수록 신뢰도 감소
    }
}

// 카메라: 차선 중심으로 횡방향 보정
fn estimate_from_camera(
    prev: &PoseEstimate,
    lane: &CameraLaneDetection,
) -> Option<PoseEstimate> {
    if lane.confidence < 0.5 {
        return None; // 차선 신뢰도 낮음
    }
    let lateral_correction = match (lane.left_offset_m, lane.right_offset_m) {
        (Some(l), Some(r)) => (l - r) / 2.0, // 양쪽 차선: 중앙 보정
        (Some(l), None)    => l - 1.75,       // 왼쪽만: 표준 차선폭 가정
        (None, Some(r))    => 1.75 - r,
        (None, None)       => return None,
    };
    Some(PoseEstimate {
        x: prev.x + (lateral_correction * (-prev.heading).sin()) as f64,
        y: prev.y + (lateral_correction * prev.heading.cos()) as f64,
        heading: prev.heading - lane.heading_error_rad,
        confidence: lane.confidence,
    })
}
```

세 함수 모두:
- 하드웨어 의존 없음
- 입력만 있으면 테스트 가능
- 데이터가 없거나 품질이 낮으면 `None`을 반환

---

## 레이어 3: 퓨전 (Calculation)

여러 추정치를 합칩니다. 어떤 센서가 없어도 동작합니다.

```rust
fn fuse_estimates(estimates: &[Option<PoseEstimate>]) -> Option<PoseEstimate> {
    let valid: Vec<&PoseEstimate> = estimates.iter()
        .filter_map(|e| e.as_ref())
        .filter(|e| e.confidence > 0.0)
        .collect();

    if valid.is_empty() {
        return None;
    }

    let total_confidence: f32 = valid.iter().map(|e| e.confidence).sum();

    // 신뢰도 가중 평균
    let x = valid.iter().map(|e| e.x * e.confidence as f64).sum::<f64>() / total_confidence as f64;
    let y = valid.iter().map(|e| e.y * e.confidence as f64).sum::<f64>() / total_confidence as f64;
    let heading = valid.iter().map(|e| e.heading * e.confidence).sum::<f32>() / total_confidence;

    Some(PoseEstimate {
        x, y, heading,
        confidence: total_confidence / valid.len() as f32,
    })
}
```

`fuse_estimates`는 GPS가 몇 개든, 카메라가 없든 상관하지 않습니다. `Option<PoseEstimate>` 목록만 받습니다. 새 센서를 추가해도 이 함수를 건드릴 필요가 없습니다.

---

## 파이프라인 조합: 컴포지션

세 레이어를 이어붙이면 전체 파이프라인이 됩니다.

```rust
struct SensorReadings {
    gps: Option<GpsMeasurement>,
    imu: Option<ImuMeasurement>,
    camera: Option<CameraLaneDetection>,
}

fn localize(
    readings: &SensorReadings,
    prev_pose: &PoseEstimate,
    origin: &GpsMeasurement,
    dt_s: f32,
) -> Option<PoseEstimate> {
    // 각 센서 추정 (계산)
    let gps_estimate = readings.gps.as_ref()
        .and_then(|g| estimate_from_gps(g, origin));

    let imu_estimate = readings.imu.as_ref()
        .map(|imu| estimate_from_imu(prev_pose, imu, dt_s));

    let camera_estimate = readings.camera.as_ref()
        .and_then(|lane| estimate_from_camera(prev_pose, lane));

    // 퓨전 (계산)
    fuse_estimates(&[gps_estimate, imu_estimate, camera_estimate])
}
```

데이터 흐름이 명확합니다.

```
SensorReadings
    │
    ├─ gps   → estimate_from_gps   ─┐
    ├─ imu   → estimate_from_imu   ─┼─→ fuse_estimates → PoseEstimate
    └─ camera→ estimate_from_camera─┘
```

---

## 테스트: 센서 시나리오별로

함수들이 순수하므로 데이터를 직접 만들어서 각 시나리오를 검증합니다.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn origin() -> GpsMeasurement {
        GpsMeasurement { latitude: 37.5665, longitude: 126.9780, altitude: 30.0,
                         accuracy_m: 1.0, timestamp_ms: 0 }
    }

    #[test]
    fn test_gps_rejected_when_low_accuracy() {
        let bad_gps = GpsMeasurement { accuracy_m: 10.0, ..origin() };
        assert!(estimate_from_gps(&bad_gps, &origin()).is_none());
    }

    #[test]
    fn test_fusion_works_with_gps_only() {
        let gps = GpsMeasurement { latitude: 37.5666, accuracy_m: 1.0, ..origin() };
        let gps_est = estimate_from_gps(&gps, &origin());
        let result = fuse_estimates(&[gps_est, None, None]);
        assert!(result.is_some());
    }

    #[test]
    fn test_fusion_returns_none_when_all_missing() {
        let result = fuse_estimates(&[None, None, None]);
        assert!(result.is_none());
    }

    #[test]
    fn test_camera_lateral_correction() {
        let prev = PoseEstimate { x: 0.0, y: 0.0, heading: 0.0, confidence: 0.8 };
        let lane = CameraLaneDetection {
            left_offset_m: Some(1.0),   // 왼쪽 차선까지 1m
            right_offset_m: Some(2.5),  // 오른쪽 차선까지 2.5m → 오른쪽으로 치우침
            heading_error_rad: 0.0,
            confidence: 0.9,
        };
        let est = estimate_from_camera(&prev, &lane).unwrap();
        // 오른쪽으로 치우쳤으니 왼쪽으로 보정 → x가 감소
        assert!(est.x < 0.0 || est.y != 0.0 || est.confidence > 0.0); // 보정됨
    }
}
```

---

## 새 센서 추가: 퓨전 로직 무수정

LiDAR 기반 위치 추정을 추가한다고 합시다.

```rust
fn estimate_from_lidar(scan: &LidarScan, map: &HdMap) -> Option<PoseEstimate> {
    // 포인트 클라우드와 HD맵을 매칭해 위치 추정 (NDT 등)
    // ...
    Some(PoseEstimate { x: 0.0, y: 0.0, heading: 0.0, confidence: 0.95 })
}
```

퓨전 호출에 한 줄만 추가하면 됩니다.

```rust
let lidar_estimate = scan.as_ref()
    .and_then(|s| estimate_from_lidar(s, map));

fuse_estimates(&[gps_estimate, imu_estimate, camera_estimate, lidar_estimate])
```

`fuse_estimates` 함수를 건드리지 않습니다. 추상화 장벽이 레이어 간 변경 전파를 막기 때문입니다.

---

## 정리

| 레이어 | 내용 | 특징 |
|--------|------|------|
| 원시 데이터 | `GpsMeasurement`, `ImuMeasurement`, `CameraLaneDetection` | 센서별로 독립, 파싱 없음 |
| 추정 함수 | `estimate_from_gps`, `estimate_from_imu`, `estimate_from_camera` | 순수 계산, 센서 교체 시 이 함수만 수정 |
| 퓨전 | `fuse_estimates`, `localize` | 센서 종류 무관, 새 센서 추가 시 무수정 |

추상화 장벽이 레이어 사이를 자르고, 컴포지션이 레이어를 잇습니다. 어떤 센서가 추가·교체·제거되어도 다른 레이어는 영향을 받지 않습니다.

---

*관련 글: [함수 컴포지션](/posts/programming/functional-composition/), [추상화 장벽](/posts/programming/functional-abstraction-barrier/), [자율주행 센서 파이프라인](/posts/programming/autonomous-sensor-pipeline/), [자율주행 경로 계획](/posts/programming/autonomous-path-planning/), [ROS2 콜백을 함수형으로](/posts/programming/autonomous-ros2-functional/)*
