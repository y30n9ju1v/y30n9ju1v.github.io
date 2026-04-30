---
title: "함수형 오도메트리와 SLAM: 위치 추정을 순수 계산으로"
date: 2026-04-30T13:00:00+09:00
draft: false
tags: ["함수형 프로그래밍", "Rust", "설계", "자율주행", "로봇", "SLAM", "오도메트리", "칼만 필터", "액션/계산/데이터"]
categories: ["프로그래밍", "자율주행"]
description: "로봇의 위치 추정(EKF, UKF)을 액션/계산/데이터로 분리하면, 센서 없이 추정 알고리즘 전체를 단위 테스트하고 주행 로그만으로 임의 시점을 재현할 수 있습니다."
---

## 이 글을 읽고 나면

- 오도메트리와 SLAM의 핵심 구조를 이해합니다.
- EKF(확장 칼만 필터)의 predict/update를 순수 계산으로 구현하는 방법을 봅니다.
- `State` 불변 값 전달이 PID 제어기와 어떻게 같은 패턴인지 확인합니다.
- 하드웨어 없이 위치 추정 로직 전체를 단위 테스트하는 방법을 압니다.

이전 글 [액션/계산/데이터](/posts/programming/functional-actions-calculations-data/), [함수형 PID 제어기](/posts/programming/autonomous-pid-controller/), [함수형 센서 퓨전](/posts/programming/autonomous-sensor-fusion/)을 먼저 읽으면 더 자연스럽게 이어집니다.

---

## 오도메트리란

오도메트리(Odometry)는 "지금까지 얼마나, 어느 방향으로 이동했는가"를 추정하는 기법입니다. 바퀴 회전수, IMU 적분, 비주얼 피처 추적 등을 이용합니다.

SLAM(Simultaneous Localization and Mapping)은 오도메트리를 확장해서 이동하면서 동시에 지도도 만드는 문제입니다. 여기서는 위치 추정(Localization) 부분, 즉 알려진 랜드마크를 이용한 EKF Localization에 집중합니다.

자율주행·로봇에서 오도메트리가 중요한 이유는 이렇습니다.

- GPS는 실내·터널에서 끊긴다
- IMU만으론 적분 오차가 쌓인다
- 바퀴 인코더도 슬립이 생기면 오차가 생긴다

여러 소스를 합쳐서 오차를 보정하는 것이 EKF의 역할입니다.

---

## 나쁜 방법: 상태를 필드에 묻어두기

```rust
struct EkfLocalizer {
    x: f64,       // 위치 x
    y: f64,       // 위치 y
    theta: f64,   // 방향각
    p: [[f64; 3]; 3], // 공분산 행렬 (3×3)
}

impl EkfLocalizer {
    fn predict(&mut self, v: f64, omega: f64, dt: f64) {
        // 상태를 직접 변경
        self.x += v * self.theta.cos() * dt;
        self.y += v * self.theta.sin() * dt;
        self.theta += omega * dt;
        // 공분산 업데이트도 self를 직접 바꿈 ...
    }

    fn update(&mut self, landmark: (f64, f64), measurement: (f64, f64)) {
        // GPS 보정도 self를 직접 바꿈 ...
    }
}
```

이렇게 짜면 생기는 문제들:

- 같은 입력을 주어도 이전에 `predict`가 몇 번 불렸냐에 따라 결과가 달라짐
- 특정 스텝의 상태를 재현하려면 처음부터 다시 재생해야 함
- 두 경로를 동시에 비교하려면 구조체를 직접 복제해야 함
- 단위 테스트에서 내부 상태를 수동으로 맞춰야 함

---

## 데이터: 추정 상태를 값으로 표현

EKF가 스텝마다 들고 다녀야 하는 것을 구조체로 만듭니다.

```rust
/// 로봇의 현재 포즈 추정치
#[derive(Debug, Clone, PartialEq)]
pub struct Pose {
    pub x: f64,     // 미터
    pub y: f64,     // 미터
    pub theta: f64, // 라디안
}

/// 불확실성 (3×3 공분산 행렬, [x, y, θ])
#[derive(Debug, Clone)]
pub struct Covariance([[f64; 3]; 3]);

/// EKF가 한 스텝에서 다음 스텝으로 넘기는 불변 상태
#[derive(Debug, Clone)]
pub struct EkfState {
    pub pose: Pose,
    pub cov: Covariance,
}

/// 운동 모델 입력 (바퀴 인코더 또는 IMU)
#[derive(Debug, Clone)]
pub struct MotionInput {
    pub linear_velocity: f64,  // m/s
    pub angular_velocity: f64, // rad/s
    pub dt: f64,               // 초
}

/// 랜드마크 관측값
#[derive(Debug, Clone)]
pub struct LandmarkObservation {
    pub landmark_id: usize,
    pub range: f64,   // 측정 거리 (m)
    pub bearing: f64, // 측정 방위각 (rad)
}

/// 지도 위 랜드마크의 알려진 위치
#[derive(Debug, Clone)]
pub struct Landmark {
    pub id: usize,
    pub x: f64,
    pub y: f64,
}
```

`EkfState`는 "이 시점의 로봇 위치 추정"을 완전히 담습니다. 이 값 하나만 있으면 다음 스텝 계산이 가능합니다.

---

## 계산: predict — 운동 모델로 상태 전진

이전 상태와 제어 입력으로 다음 상태를 예측합니다. 센서도, 외부 호출도 없는 순수 계산입니다.

```rust
/// 프로세스 노이즈 파라미터
pub struct MotionNoise {
    pub alpha1: f64, // 회전 대비 회전 노이즈
    pub alpha2: f64, // 선속도 대비 회전 노이즈
    pub alpha3: f64, // 선속도 대비 선속도 노이즈
    pub alpha4: f64, // 회전 대비 선속도 노이즈
}

pub fn ekf_predict(state: &EkfState, motion: &MotionInput, noise: &MotionNoise) -> EkfState {
    let Pose { x, y, theta } = state.pose;
    let v = motion.linear_velocity;
    let w = motion.angular_velocity;
    let dt = motion.dt;

    // 선형화된 운동 모델 (속도 모션 모델)
    let new_pose = if w.abs() < 1e-6 {
        // 직선 운동
        Pose {
            x: x + v * theta.cos() * dt,
            y: y + v * theta.sin() * dt,
            theta,
        }
    } else {
        // 곡선 운동
        let r = v / w;
        Pose {
            x: x - r * theta.sin() + r * (theta + w * dt).sin(),
            y: y + r * theta.cos() - r * (theta + w * dt).cos(),
            theta: normalize_angle(theta + w * dt),
        }
    };

    // 야코비안 G (선형화 행렬, 3×3)
    let g = jacobian_motion(&state.pose, motion);

    // 프로세스 노이즈 공분산 M
    let m = motion_noise_cov(motion, noise);

    // V: 제어 입력에 대한 야코비안 (3×2)
    let v_jac = jacobian_control(&state.pose, motion);

    // 공분산 전파: P' = G·P·Gᵀ + V·M·Vᵀ
    let new_cov = mat3_add(
        mat3_mul_transpose(&mat3_mul(&g, &state.cov.0), &g),
        mat3_mul_transpose(&mat2x3_mul(&v_jac, &m), &v_jac),
    );

    EkfState {
        pose: new_pose,
        cov: Covariance(new_cov),
    }
}
```

`ekf_predict`는 `EkfState → EkfState` 변환입니다. 항상 새 값을 돌려주고 입력을 바꾸지 않습니다.

---

## 계산: update — 랜드마크 관측으로 보정

랜드마크가 보이면 예측 오차를 줄입니다. 이것도 순수 계산입니다.

```rust
pub struct ObservationNoise {
    pub sigma_range: f64,   // 거리 측정 표준편차 (m)
    pub sigma_bearing: f64, // 방위각 측정 표준편차 (rad)
}

pub fn ekf_update(
    state: &EkfState,
    obs: &LandmarkObservation,
    landmark: &Landmark,
    noise: &ObservationNoise,
) -> EkfState {
    let Pose { x, y, theta } = state.pose;

    // 랜드마크까지 예측 거리·방위각
    let dx = landmark.x - x;
    let dy = landmark.y - y;
    let q = dx * dx + dy * dy;
    let expected_range = q.sqrt();
    let expected_bearing = normalize_angle(dy.atan2(dx) - theta);

    // 혁신 (실제 - 예측)
    let innovation_range = obs.range - expected_range;
    let innovation_bearing = normalize_angle(obs.bearing - expected_bearing);

    // 관측 야코비안 H (2×3)
    let h = jacobian_observation(dx, dy, q);

    // 관측 노이즈 공분산 Q (2×2)
    let q_noise = [
        [noise.sigma_range.powi(2), 0.0],
        [0.0, noise.sigma_bearing.powi(2)],
    ];

    // 혁신 공분산 S = H·P·Hᵀ + Q
    let s = mat2_add(
        mat2x3_mul_transpose(&mat2x3_mul_mat3(&h, &state.cov.0), &h),
        q_noise,
    );

    // 칼만 게인 K = P·Hᵀ·S⁻¹ (3×2)
    let k = mat3x2_mul(
        &mat3x2_from_mat3_and_ht(&state.cov.0, &h),
        &mat2_inv(&s),
    );

    // 상태 업데이트
    let new_x = x + k[0][0] * innovation_range + k[0][1] * innovation_bearing;
    let new_y = y + k[1][0] * innovation_range + k[1][1] * innovation_bearing;
    let new_theta = normalize_angle(
        theta + k[2][0] * innovation_range + k[2][1] * innovation_bearing,
    );

    // 공분산 업데이트: P' = (I - K·H)·P
    let kh = mat3_from_mat3x2_and_mat2x3(&k, &h);
    let i_minus_kh = mat3_sub(&identity3(), &kh);
    let new_cov = mat3_mul(&i_minus_kh, &state.cov.0);

    EkfState {
        pose: Pose { x: new_x, y: new_y, theta: new_theta },
        cov: Covariance(new_cov),
    }
}
```

`ekf_update`도 `EkfState → EkfState`입니다. 랜드마크가 보일 때만 호출하고, 안 보이면 `predict` 결과를 그대로 씁니다.

---

## 액션: 센서 읽기와 이벤트 루프

순수 계산은 안쪽에, 센서 I/O는 바깥에 둡니다.

```rust
enum LocalizationEvent {
    Motion(MotionInput),
    Observation(LandmarkObservation),
}

/// 이벤트 하나를 처리하는 순수 계산
fn process_event(
    state: &EkfState,
    event: &LocalizationEvent,
    landmarks: &[Landmark],
    motion_noise: &MotionNoise,
    obs_noise: &ObservationNoise,
) -> EkfState {
    match event {
        LocalizationEvent::Motion(motion) => {
            ekf_predict(state, motion, motion_noise)
        }
        LocalizationEvent::Observation(obs) => {
            if let Some(lm) = landmarks.iter().find(|l| l.id == obs.landmark_id) {
                ekf_update(state, obs, lm, obs_noise)
            } else {
                state.clone() // 모르는 랜드마크 → 상태 유지
            }
        }
    }
}

/// 실제 로봇에서 실행되는 루프 (액션이 여기에만 모임)
fn localization_loop(
    sensor: &dyn LocalizationSensor, // 액션: 센서 읽기
    display: &dyn PoseDisplay,        // 액션: 결과 출력
    initial_state: EkfState,
    landmarks: Vec<Landmark>,
    motion_noise: MotionNoise,
    obs_noise: ObservationNoise,
) {
    let mut state = initial_state;

    loop {
        let event = sensor.next_event(); // 액션

        // 계산: 순수 함수
        state = process_event(&state, &event, &landmarks, &motion_noise, &obs_noise);

        display.show(&state.pose); // 액션
    }
}
```

루프 구조가 명확합니다.

```
[액션] 센서 이벤트 읽기
    ↓
[계산] ekf_predict 또는 ekf_update → 새 EkfState
    ↓
[액션] 포즈 출력
```

---

## 여러 경로 후보를 병렬 추정

불변 상태의 강점은 복사가 자유롭다는 점입니다. 같은 출발 상태에서 여러 경로를 동시에 탐색할 수 있습니다. 파티클 필터(Particle Filter)가 이 패턴입니다.

```rust
/// 파티클 = (가설 상태, 가중치)
#[derive(Debug, Clone)]
struct Particle {
    state: EkfState,
    weight: f64,
}

/// 모션 업데이트: 모든 파티클을 독립적으로 전진
fn particle_predict(
    particles: &[Particle],
    motion: &MotionInput,
    noise: &MotionNoise,
) -> Vec<Particle> {
    particles.iter().map(|p| Particle {
        state: ekf_predict(&p.state, motion, noise),
        weight: p.weight,
    }).collect()
}

/// 관측 업데이트: 랜드마크에 가까울수록 가중치 증가
fn particle_update(
    particles: &[Particle],
    obs: &LandmarkObservation,
    landmark: &Landmark,
    obs_noise: &ObservationNoise,
) -> Vec<Particle> {
    let updated: Vec<Particle> = particles.iter().map(|p| {
        let Pose { x, y, theta } = p.state.pose;
        let dx = landmark.x - x;
        let dy = landmark.y - y;
        let expected_range = (dx * dx + dy * dy).sqrt();
        let expected_bearing = normalize_angle(dy.atan2(dx) - theta);

        let range_err = obs.range - expected_range;
        let bearing_err = normalize_angle(obs.bearing - expected_bearing);

        // 가우시안 가중치
        let w = gaussian_likelihood(range_err, obs_noise.sigma_range)
              * gaussian_likelihood(bearing_err, obs_noise.sigma_bearing);

        Particle { state: p.state.clone(), weight: p.weight * w }
    }).collect();

    normalize_weights(updated)
}
```

`ekf_predict`가 순수 함수이기 때문에 파티클마다 독립적으로 호출할 수 있습니다. 상태가 가변이었다면 불가능한 패턴입니다.

---

## 테스트: 하드웨어 없이 알고리즘 검증

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn default_motion_noise() -> MotionNoise {
        MotionNoise { alpha1: 0.1, alpha2: 0.01, alpha3: 0.01, alpha4: 0.1 }
    }

    fn default_obs_noise() -> ObservationNoise {
        ObservationNoise { sigma_range: 0.1, sigma_bearing: 0.05 }
    }

    fn initial_state() -> EkfState {
        EkfState {
            pose: Pose { x: 0.0, y: 0.0, theta: 0.0 },
            cov: Covariance([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.05]]),
        }
    }

    #[test]
    fn test_predict_moves_forward() {
        let state = initial_state();
        let motion = MotionInput { linear_velocity: 1.0, angular_velocity: 0.0, dt: 1.0 };
        let next = ekf_predict(&state, &motion, &default_motion_noise());
        assert!((next.pose.x - 1.0).abs() < 1e-6);
        assert!(next.pose.y.abs() < 1e-6);
    }

    #[test]
    fn test_predict_turns() {
        let state = initial_state();
        // π/2 회전 (제자리 회전)
        let motion = MotionInput {
            linear_velocity: 0.0,
            angular_velocity: std::f64::consts::PI / 2.0,
            dt: 1.0,
        };
        let next = ekf_predict(&state, &motion, &default_motion_noise());
        assert!((next.pose.theta - std::f64::consts::PI / 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_update_reduces_uncertainty() {
        let state = initial_state();
        let landmark = Landmark { id: 0, x: 1.0, y: 0.0 };
        let obs = LandmarkObservation { landmark_id: 0, range: 1.0, bearing: 0.0 };

        let updated = ekf_update(&state, &obs, &landmark, &default_obs_noise());

        // 관측 후 위치 불확실성(공분산 대각 원소)이 줄어야 함
        assert!(updated.cov.0[0][0] < state.cov.0[0][0]);
    }

    #[test]
    fn test_same_events_give_same_result() {
        let events = vec![
            LocalizationEvent::Motion(MotionInput {
                linear_velocity: 0.5, angular_velocity: 0.1, dt: 0.1
            }),
            LocalizationEvent::Observation(LandmarkObservation {
                landmark_id: 0, range: 1.4, bearing: 0.3
            }),
        ];
        let landmarks = vec![Landmark { id: 0, x: 1.0, y: 1.0 }];

        let run = || {
            let mut s = initial_state();
            for e in &events {
                s = process_event(&s, e, &landmarks, &default_motion_noise(), &default_obs_noise());
            }
            s
        };

        let r1 = run();
        let r2 = run();
        assert!((r1.pose.x - r2.pose.x).abs() < 1e-12);
        assert!((r1.pose.y - r2.pose.y).abs() < 1e-12);
    }

    #[test]
    fn test_unknown_landmark_leaves_state_unchanged() {
        let state = initial_state();
        let obs = LandmarkObservation { landmark_id: 99, range: 1.0, bearing: 0.0 };
        let result = process_event(
            &state,
            &LocalizationEvent::Observation(obs),
            &[], // 빈 맵
            &default_motion_noise(),
            &default_obs_noise(),
        );
        assert_eq!(result.pose, state.pose);
    }
}
```

---

## 로그에서 임의 시점 재현

`EkfState`가 불변 값이므로 이벤트 로그만 있으면 어느 시점이든 재현됩니다.

```rust
#[derive(Debug, Clone)]
struct EventLog {
    timestamp_ms: u64,
    event: LocalizationEvent,
}

fn replay(
    logs: &[EventLog],
    initial: EkfState,
    landmarks: &[Landmark],
    motion_noise: &MotionNoise,
    obs_noise: &ObservationNoise,
) -> Vec<(u64, EkfState)> {
    let mut state = initial;
    logs.iter().map(|log| {
        state = process_event(&state, &log.event, landmarks, motion_noise, obs_noise);
        (log.timestamp_ms, state.clone())
    }).collect()
}
```

사고 분석, 알고리즘 교체 전후 비교, 오프라인 파라미터 튜닝 모두 로그 파일 하나면 충분합니다.

---

## PID 제어기와 같은 패턴

[함수형 PID 제어기](/posts/programming/autonomous-pid-controller/) 글에서 `PidState`를 값으로 전달한 것과 완전히 같은 구조입니다.

| | PID 제어기 | EKF Localizer |
|---|---|---|
| 스텝 간 상태 | `PidState { integral, prev_error }` | `EkfState { pose, cov }` |
| 계산 함수 | `pid_step(config, state, ...) → PidOutput` | `ekf_predict / ekf_update(state, ...) → EkfState` |
| 액션 위치 | 루프 바깥 (센서 읽기, 액추에이터 쓰기) | 루프 바깥 (센서 이벤트 읽기, 포즈 출력) |
| 재현 방법 | 제어 입력 로그 | 이벤트 로그 |
| 다중 인스턴스 | `VehicleController` 여러 개 | 파티클 여러 개 |

패턴의 이름은 다르지만 구조는 같습니다. **시간축 위의 상태 전이를 불변 값 전달로 표현**하면, 어떤 알고리즘이든 재현 가능한 순수 계산이 됩니다.

---

## 정리

| 구성 요소 | 분류 | 특징 |
|---|---|---|
| `Pose`, `EkfState`, `MotionInput`, `LandmarkObservation` | 데이터 | 불변, 직렬화 가능, 로그에 저장 가능 |
| `ekf_predict`, `ekf_update`, `process_event` | 계산 | 순수 함수, 하드웨어 의존 없음, 테스트 가능 |
| `localization_loop`, 센서 읽기, 포즈 출력 | 액션 | 루프 바깥에만 모임 |

상태가 가변 필드 대신 불변 값이 되면 EKF는 그냥 `이전 상태 → 다음 상태` 변환 함수가 됩니다. 이 구조는 파티클 필터, 그래프 SLAM, 어떤 위치 추정 알고리즘으로 교체해도 바깥 구조를 건드리지 않습니다.

---

*관련 글: [액션/계산/데이터](/posts/programming/functional-actions-calculations-data/), [함수형 PID 제어기](/posts/programming/autonomous-pid-controller/), [함수형 센서 퓨전](/posts/programming/autonomous-sensor-fusion/), [불변 데이터와 구조적 공유](/posts/programming/functional-immutable-data/), [자율주행 센서 파이프라인](/posts/programming/autonomous-sensor-pipeline/), [시뮬레이션 회귀 테스트 설계](/posts/programming/autonomous-simulation-regression/)*
