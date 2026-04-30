---
title: "함수형 칼만 필터: predict와 update를 순수 함수로"
date: 2026-04-30T14:00:00+09:00
draft: false
tags: ["함수형 프로그래밍", "Rust", "설계", "자율주행", "로봇", "칼만 필터", "다중 물체 추적", "액션/계산/데이터"]
categories: ["프로그래밍", "자율주행"]
description: "칼만 필터의 predict/update를 불변 값 전달 순수 함수로 구현하면, 다중 물체 추적(MOT)까지 센서 없이 테스트하고 로그만으로 임의 프레임을 재현할 수 있습니다."
---

## 이 글을 읽고 나면

- 칼만 필터의 predict/update 구조를 이해합니다.
- 필터 상태를 전역 변수 대신 불변 값으로 전달하면 무엇이 달라지는지 봅니다.
- 다중 물체 추적(MOT)에서 여러 트래커를 독립적으로 운용하는 방법을 봅니다.
- 하드웨어 없이 추적 로직을 단위 테스트하고 로그로 재현하는 방법을 압니다.

이전 글 [액션/계산/데이터](/posts/programming/functional/functional-actions-calculations-data/), [불변 데이터와 구조적 공유](/posts/programming/functional/functional-immutable-data/), [함수형 센서 퓨전](/posts/programming/functional/autonomous-sensor-fusion/)을 먼저 읽으면 더 자연스럽게 이어집니다.

---

## 칼만 필터란

칼만 필터(Kalman Filter)는 노이즈가 있는 측정값에서 실제 상태를 추정하는 알고리즘입니다. 두 단계를 반복합니다.

1. **predict**: 이전 상태와 운동 모델로 현재 상태를 예측
2. **update**: 실제 측정값이 오면 예측을 보정

자율주행에서는 이런 곳에 씁니다.

- **물체 추적**: 카메라·LiDAR 검출 박스에서 차량·보행자의 실제 위치·속도 추정
- **차선 추적**: 프레임마다 달라지는 차선 검출을 부드럽게 추적
- **자기 위치**: GPS + IMU 퓨전으로 위치 노이즈 제거

선형 칼만 필터(LKF)의 수식은 이렇습니다.

```
predict:
  x̂⁻  = F·x̂ + B·u       (상태 예측)
  P⁻   = F·P·Fᵀ + Q       (공분산 예측)

update:
  K    = P⁻·Hᵀ·(H·P⁻·Hᵀ + R)⁻¹   (칼만 게인)
  x̂   = x̂⁻ + K·(z - H·x̂⁻)        (상태 보정)
  P    = (I - K·H)·P⁻              (공분산 보정)
```

수식이 복잡해 보이지만, 구조는 단순합니다. 이전 상태와 측정값을 받아 다음 상태를 돌려주는 함수 두 개입니다.

---

## 나쁜 방법: 상태를 객체 안에 묻기

```rust
struct KalmanTracker {
    state: Vec<f64>,      // 상태 벡터 [x, y, vx, vy]
    covariance: Vec<Vec<f64>>,  // 공분산 행렬
    track_id: u32,
}

impl KalmanTracker {
    fn predict(&mut self, dt: f64) {
        // self.state를 직접 수정
    }
    fn update(&mut self, measurement: (f64, f64)) {
        // self.state를 직접 수정
    }
}
```

여러 물체를 추적할 때 문제가 됩니다.

- 100개 물체 → 100개 `KalmanTracker`, 각자 가변 상태
- 특정 물체의 프레임 10에서의 상태를 재현하려면 처음부터 다시 돌려야 함
- 물체가 사라졌다가 다시 나타났을 때 상태를 어떻게 리셋하는지 불명확
- 물체 추적 히스토리를 저장하고 싶어도 가변 참조가 충돌

---

## 데이터: 추적 상태를 값으로

```rust
/// 추적 대상의 운동 상태
/// 상태 벡터: [x, y, vx, vy] — 위치와 속도
#[derive(Debug, Clone, PartialEq)]
pub struct TrackState {
    pub x: f64,   // 위치 x (픽셀 또는 미터)
    pub y: f64,   // 위치 y
    pub vx: f64,  // 속도 x
    pub vy: f64,  // 속도 y
}

/// 4×4 공분산 행렬 ([x, y, vx, vy])
#[derive(Debug, Clone)]
pub struct TrackCovariance(pub [[f64; 4]; 4]);

/// 칼만 필터가 스텝마다 들고 다니는 불변 상태
#[derive(Debug, Clone)]
pub struct KfState {
    pub track: TrackState,
    pub cov: TrackCovariance,
    pub age: u32,        // 추적 시작 이후 프레임 수
    pub miss_count: u32, // 연속 미검출 프레임 수
}

/// 검출기가 내놓는 박스 (2D 바운딩 박스 중심)
#[derive(Debug, Clone)]
pub struct Detection {
    pub cx: f64,
    pub cy: f64,
    pub score: f32,
}

/// 노이즈 파라미터
#[derive(Debug, Clone)]
pub struct KfParams {
    pub process_noise_pos: f64,  // 위치 프로세스 노이즈
    pub process_noise_vel: f64,  // 속도 프로세스 노이즈
    pub measurement_noise: f64,  // 측정 노이즈
}

impl KfParams {
    pub fn default_vehicle() -> Self {
        Self { process_noise_pos: 1.0, process_noise_vel: 10.0, measurement_noise: 1.0 }
    }
    pub fn default_pedestrian() -> Self {
        Self { process_noise_pos: 2.0, process_noise_vel: 20.0, measurement_noise: 2.0 }
    }
}
```

`KfState`는 특정 프레임의 추적 상태를 완전히 표현합니다. 이 값 하나만 있으면 다음 프레임을 계산할 수 있습니다.

---

## 계산: predict

```rust
pub fn kf_predict(state: &KfState, dt: f64, params: &KfParams) -> KfState {
    let t = &state.track;

    // 등속 운동 모델: x' = x + vx·dt
    let new_track = TrackState {
        x:  t.x  + t.vx * dt,
        y:  t.y  + t.vy * dt,
        vx: t.vx,
        vy: t.vy,
    };

    // 상태 전이 행렬 F (4×4)
    let f = [
        [1.0, 0.0, dt,  0.0],
        [0.0, 1.0, 0.0, dt ],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    // 프로세스 노이즈 Q
    let dt2 = dt * dt;
    let dt3 = dt2 * dt;
    let dt4 = dt3 * dt;
    let qp = params.process_noise_pos;
    let qv = params.process_noise_vel;
    let q = [
        [qp * dt4 / 4.0, 0.0,            qp * dt3 / 2.0, 0.0           ],
        [0.0,            qp * dt4 / 4.0, 0.0,            qp * dt3 / 2.0],
        [qp * dt3 / 2.0, 0.0,            qv * dt2,       0.0           ],
        [0.0,            qp * dt3 / 2.0, 0.0,            qv * dt2      ],
    ];

    // P' = F·P·Fᵀ + Q
    let new_cov = mat4_add(
        &mat4_mul_transpose(&mat4_mul(&f, &state.cov.0), &f),
        &q,
    );

    KfState {
        track: new_track,
        cov: TrackCovariance(new_cov),
        age: state.age + 1,
        miss_count: state.miss_count + 1, // 검출 전까지 미스로 가정
    }
}
```

`kf_predict`는 `KfState → KfState`입니다. `dt` 하나만 달라져도 다른 결과가 나오고, 항상 새 값을 돌려줍니다.

---

## 계산: update

```rust
pub fn kf_update(state: &KfState, det: &Detection, params: &KfParams) -> KfState {
    let t = &state.track;

    // 관측 행렬 H: 위치만 관측 (2×4)
    let h = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ];

    // 측정 노이즈 R (2×2)
    let r = [
        [params.measurement_noise, 0.0],
        [0.0, params.measurement_noise],
    ];

    // 혁신: z - H·x̂
    let innovation = [det.cx - t.x, det.cy - t.y];

    // 혁신 공분산 S = H·P·Hᵀ + R (2×2)
    let hp = mat2x4_mul_mat4(&h, &state.cov.0);
    let s = mat2_add(&mat2x4_mul_transpose(&hp, &h), &r);

    // 칼만 게인 K = P·Hᵀ·S⁻¹ (4×2)
    let pht = mat4x2_mul_from_pht(&state.cov.0, &h);
    let k = mat4x2_mul_mat2(&pht, &mat2_inv(&s));

    // 상태 보정: x̂ = x̂⁻ + K·innovation
    let new_track = TrackState {
        x:  t.x  + k[0][0] * innovation[0] + k[0][1] * innovation[1],
        y:  t.y  + k[1][0] * innovation[0] + k[1][1] * innovation[1],
        vx: t.vx + k[2][0] * innovation[0] + k[2][1] * innovation[1],
        vy: t.vy + k[3][0] * innovation[0] + k[3][1] * innovation[1],
    };

    // 공분산 보정: P = (I - K·H)·P
    let kh = mat4_from_kh(&k, &h);
    let i_minus_kh = mat4_sub(&identity4(), &kh);
    let new_cov = mat4_mul(&i_minus_kh, &state.cov.0);

    KfState {
        track: new_track,
        cov: TrackCovariance(new_cov),
        age: state.age,
        miss_count: 0, // 검출됐으므로 리셋
    }
}
```

---

## 다중 물체 추적 (MOT)

칼만 필터가 순수 함수이므로 여러 물체를 독립적으로 추적할 수 있습니다.

```rust
/// 한 프레임의 추적 결과
#[derive(Debug, Clone)]
pub struct TrackerState {
    pub tracks: Vec<(u32, KfState)>, // (track_id, 필터 상태)
    pub next_id: u32,
}

/// 검출-트랙 매칭 결과
struct MatchResult {
    matched: Vec<(u32, Detection)>,    // (track_id, 대응 검출)
    unmatched_tracks: Vec<u32>,        // 검출 없는 트랙
    new_detections: Vec<Detection>,    // 새 물체 후보
}

/// 한 프레임 처리 — 순수 계산
pub fn update_tracks(
    state: &TrackerState,
    detections: &[Detection],
    dt: f64,
    params: &KfParams,
    max_miss: u32,
) -> TrackerState {
    // 1. 모든 트랙 predict
    let predicted: Vec<(u32, KfState)> = state.tracks.iter()
        .map(|(id, kf)| (*id, kf_predict(kf, dt, params)))
        .collect();

    // 2. 헝가리안 매칭 (IoU 기반)
    let matches = hungarian_match(&predicted, detections);

    // 3. 매칭된 트랙은 update, 미매칭 트랙은 predict 결과 유지
    let mut new_tracks: Vec<(u32, KfState)> = Vec::new();

    for (id, kf) in &predicted {
        if let Some(det) = matches.matched.iter().find(|(tid, _)| tid == id).map(|(_, d)| d) {
            new_tracks.push((*id, kf_update(kf, det, params)));
        } else if kf.miss_count < max_miss {
            new_tracks.push((*id, kf.clone())); // 잠시 안 보여도 유지
        }
        // miss_count >= max_miss → 트랙 삭제 (그냥 추가 안 함)
    }

    // 4. 새 검출 → 새 트랙
    let mut next_id = state.next_id;
    for det in &matches.new_detections {
        let init_state = KfState {
            track: TrackState { x: det.cx, y: det.cy, vx: 0.0, vy: 0.0 },
            cov: TrackCovariance(identity4_scaled(10.0)),
            age: 0,
            miss_count: 0,
        };
        new_tracks.push((next_id, init_state));
        next_id += 1;
    }

    TrackerState { tracks: new_tracks, next_id }
}
```

`update_tracks`는 `TrackerState → TrackerState`입니다. 100개 트랙이든 1개든 같은 함수 하나로 처리합니다.

---

## 액션: 카메라·LiDAR 루프

```rust
fn tracking_loop(
    detector: &dyn ObjectDetector, // 액션: 검출기 호출
    display: &dyn TrackDisplay,    // 액션: 결과 출력
    params: KfParams,
) {
    let mut tracker = TrackerState { tracks: vec![], next_id: 0 };
    let mut last_time = detector.now_ms();

    loop {
        let now = detector.now_ms();                  // 액션
        let dt = (now - last_time) as f64 / 1000.0;
        last_time = now;

        let detections = detector.detect_objects();   // 액션

        // 계산: 순수 함수
        tracker = update_tracks(&tracker, &detections, dt, &params, 3);

        display.show_tracks(&tracker.tracks);         // 액션
    }
}
```

---

## 테스트: 시나리오별 검증

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> KfParams { KfParams::default_vehicle() }

    fn init_track(x: f64, y: f64) -> KfState {
        KfState {
            track: TrackState { x, y, vx: 0.0, vy: 0.0 },
            cov: TrackCovariance(identity4_scaled(1.0)),
            age: 0,
            miss_count: 0,
        }
    }

    #[test]
    fn test_predict_moves_by_velocity() {
        let state = KfState {
            track: TrackState { x: 0.0, y: 0.0, vx: 2.0, vy: 1.0 },
            cov: TrackCovariance(identity4_scaled(1.0)),
            age: 0,
            miss_count: 0,
        };
        let next = kf_predict(&state, 1.0, &params());
        assert!((next.track.x - 2.0).abs() < 1e-9);
        assert!((next.track.y - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_update_moves_toward_detection() {
        let state = init_track(0.0, 0.0);
        let det = Detection { cx: 10.0, cy: 0.0, score: 0.9 };
        let updated = kf_update(&state, &det, &params());
        // 검출 방향으로 이동해야 함
        assert!(updated.track.x > 0.0);
        assert!(updated.track.x < 10.0);
    }

    #[test]
    fn test_update_reduces_covariance() {
        let state = init_track(5.0, 5.0);
        let det = Detection { cx: 5.0, cy: 5.0, score: 0.9 };
        let updated = kf_update(&state, &det, &params());
        // 위치 불확실성이 줄어야 함
        assert!(updated.cov.0[0][0] < state.cov.0[0][0]);
    }

    #[test]
    fn test_miss_count_increments_on_predict() {
        let state = init_track(0.0, 0.0);
        let predicted = kf_predict(&state, 0.1, &params());
        assert_eq!(predicted.miss_count, 1);
    }

    #[test]
    fn test_miss_count_resets_on_update() {
        let mut state = init_track(0.0, 0.0);
        state.miss_count = 2;
        let det = Detection { cx: 0.1, cy: 0.0, score: 0.9 };
        let updated = kf_update(&state, &det, &params());
        assert_eq!(updated.miss_count, 0);
    }

    #[test]
    fn test_track_deleted_after_max_miss() {
        let state = TrackerState {
            tracks: vec![(0, init_track(100.0, 100.0))],
            next_id: 1,
        };
        // 3프레임 동안 검출 없음
        let s1 = update_tracks(&state, &[], 0.1, &params(), 2);
        let s2 = update_tracks(&s1,    &[], 0.1, &params(), 2);
        let s3 = update_tracks(&s2,    &[], 0.1, &params(), 2);
        assert!(s3.tracks.is_empty());
    }

    #[test]
    fn test_new_detection_creates_track() {
        let state = TrackerState { tracks: vec![], next_id: 0 };
        let dets = vec![Detection { cx: 5.0, cy: 3.0, score: 0.8 }];
        let next = update_tracks(&state, &dets, 0.1, &params(), 3);
        assert_eq!(next.tracks.len(), 1);
    }

    #[test]
    fn test_same_inputs_give_same_outputs() {
        let state = TrackerState {
            tracks: vec![(0, init_track(0.0, 0.0))],
            next_id: 1,
        };
        let dets = vec![Detection { cx: 1.0, cy: 0.5, score: 0.9 }];

        let r1 = update_tracks(&state, &dets, 0.1, &params(), 3);
        let r2 = update_tracks(&state, &dets, 0.1, &params(), 3);

        let (_, kf1) = &r1.tracks[0];
        let (_, kf2) = &r2.tracks[0];
        assert!((kf1.track.x - kf2.track.x).abs() < 1e-12);
    }
}
```

---

## 로그에서 프레임 재현

```rust
#[derive(Debug, Clone)]
struct FrameLog {
    timestamp_ms: u64,
    dt: f64,
    detections: Vec<Detection>,
}

fn replay_tracking(
    logs: &[FrameLog],
    params: &KfParams,
    max_miss: u32,
) -> Vec<(u64, TrackerState)> {
    let mut tracker = TrackerState { tracks: vec![], next_id: 0 };
    logs.iter().map(|frame| {
        tracker = update_tracks(&tracker, &frame.detections, frame.dt, params, max_miss);
        (frame.timestamp_ms, tracker.clone())
    }).collect()
}
```

검출 로그만 저장해두면 어느 프레임이든 추적 결과를 재현할 수 있습니다. 알고리즘을 교체하거나 파라미터를 바꿔서 같은 로그로 비교할 수도 있습니다.

---

## 센서 퓨전 글과의 차이

[함수형 센서 퓨전](/posts/programming/functional/autonomous-sensor-fusion/) 글은 같은 시점의 여러 센서를 합치는 문제였습니다. 이 글의 칼만 필터는 시간축 위에서 이전 상태와 현재 측정을 합치는 문제입니다.

| | 센서 퓨전 | 칼만 필터 |
|---|---|---|
| 합치는 대상 | 같은 시점의 여러 센서 | 이전 상태 + 현재 측정 |
| 핵심 함수 | `fuse_estimates(estimates)` | `kf_predict` + `kf_update` |
| 상태 전파 | 없음 (매 프레임 독립) | 있음 (이전 `KfState` → 다음 `KfState`) |
| 주된 문제 | 센서 이질성 | 노이즈와 시간 연속성 |

두 패턴 모두 핵심은 같습니다. 외부 의존 없이 입력에서 출력이 완전히 결정되는 순수 함수로 만드는 것입니다.

---

## 정리

| 구성 요소 | 분류 | 특징 |
|---|---|---|
| `TrackState`, `KfState`, `Detection`, `KfParams` | 데이터 | 불변, 직렬화 가능, 로그 저장 가능 |
| `kf_predict`, `kf_update`, `update_tracks` | 계산 | 순수 함수, 하드웨어 의존 없음 |
| `tracking_loop`, 검출기 호출, 결과 출력 | 액션 | 루프 바깥에만 모임 |

칼만 필터는 "노이즈 속에서 상태를 추정하는 알고리즘"이지만, 함수형으로 구현하면 단순히 `이전 상태 → 다음 상태` 변환 두 개가 됩니다. 이 구조는 LKF에서 EKF, UKF, 파티클 필터로 알고리즘을 교체해도 바깥 루프를 건드리지 않습니다.

---

*관련 글: [액션/계산/데이터](/posts/programming/functional/functional-actions-calculations-data/), [불변 데이터와 구조적 공유](/posts/programming/functional/functional-immutable-data/), [함수형 센서 퓨전](/posts/programming/functional/autonomous-sensor-fusion/), [함수형 PID 제어기](/posts/programming/functional/autonomous-pid-controller/), [함수형 포인트 클라우드 처리](/posts/programming/functional/functional-point-cloud/), [시뮬레이션 회귀 테스트 설계](/posts/programming/functional/autonomous-simulation-regression/)*
