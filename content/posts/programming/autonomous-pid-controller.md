---
title: "함수형 PID 제어기: 상태를 값으로 전달하기"
date: 2026-04-29T14:00:00+09:00
draft: false
tags: ["함수형 프로그래밍", "Rust", "설계", "자율주행", "로봇", "PID", "제어"]
categories: ["프로그래밍", "자율주행"]
description: "PID 제어기를 순수 함수로 구현하면 이전 상태를 불변 값으로 전달하게 됩니다. 같은 입력이면 항상 같은 출력이 나오므로 단위 테스트와 시뮬레이션 재현이 쉬워집니다."
---

## 이 글을 읽고 나면

- PID 제어기를 순수 계산으로 구현하는 방법을 이해합니다.
- 제어기 상태를 전역 변수 대신 불변 값으로 전달하면 무엇이 달라지는지 봅니다.
- 하드웨어 없이 PID 튜닝 로직을 단위 테스트하는 방법을 압니다.

이전 글 [액션/계산/데이터](/posts/programming/functional-actions-calculations-data/)와 [불변 데이터와 구조적 공유](/posts/programming/functional-immutable-data/)를 먼저 읽으면 더 자연스럽게 이어집니다.

---

## PID 제어기란

PID는 **P**roportional(비례), **I**ntegral(적분), **D**erivative(미분)의 약자입니다. 목표값과 현재값의 차이(오차)를 보고 출력을 조정하는 가장 널리 쓰이는 제어 알고리즘입니다.

자율주행에서는 이런 곳에 씁니다.

- **속도 제어**: 목표 속도 30km/h → 현재 속도 25km/h → 가속 페달 얼마나?
- **조향 제어**: 목표 헤딩 0° → 현재 헤딩 5° → 핸들 얼마나?
- **차간 거리 유지**: 목표 거리 15m → 현재 거리 10m → 브레이크 얼마나?

수식은 이렇습니다.

```
output = Kp * e(t)
       + Ki * ∫e(t)dt
       + Kd * de(t)/dt

e(t)       = 목표값 - 현재값  (오차)
∫e(t)dt    = 오차의 누적합    (적분항)
de(t)/dt   = 오차의 변화율    (미분항)
```

---

## 나쁜 방법: 전역 상태로 구현

전형적인 임베디드 PID 구현은 이렇게 생겼습니다.

```rust
static mut INTEGRAL: f32 = 0.0;
static mut PREV_ERROR: f32 = 0.0;

fn pid_update(target: f32, current: f32, dt: f32) -> f32 {
    let kp = 1.0_f32;
    let ki = 0.1_f32;
    let kd = 0.05_f32;

    let error = target - current;

    unsafe {
        INTEGRAL += error * dt;
        let derivative = (error - PREV_ERROR) / dt;
        PREV_ERROR = error;

        kp * error + ki * INTEGRAL + kd * derivative
    }
}
```

문제가 여럿 있습니다.

- `unsafe` 전역 변수 — 멀티스레드에서 데이터 경쟁 발생
- 테스트하려면 `INTEGRAL`과 `PREV_ERROR`를 수동으로 리셋해야 함
- 속도 PID와 조향 PID를 동시에 쓰려면 변수를 두 벌 복제해야 함
- 로그에서 특정 시점의 상태를 재현하는 것이 불가능

---

## 좋은 방법: 상태를 값으로 만들기

### 데이터 정의

PID가 다음 스텝에 넘겨야 할 상태를 구조체로 만듭니다.

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct PidConfig {
    pub kp: f32,
    pub ki: f32,
    pub kd: f32,
    pub output_min: f32, // 출력 하한 (클램핑)
    pub output_max: f32, // 출력 상한
}

#[derive(Debug, Clone, PartialEq)]
pub struct PidState {
    pub integral: f32,
    pub prev_error: f32,
}

#[derive(Debug, Clone)]
pub struct PidOutput {
    pub value: f32,       // 최종 출력
    pub p_term: f32,      // 디버깅용 P항
    pub i_term: f32,      // 디버깅용 I항
    pub d_term: f32,      // 디버깅용 D항
    pub next_state: PidState, // 다음 스텝에 넘길 상태
}
```

`PidState`는 스텝 간에 전달해야 하는 최소한의 기억입니다. `PidConfig`는 튜닝 파라미터로, 런타임 중 바뀌지 않습니다.

---

### 계산: 순수 함수로 구현

```rust
impl PidState {
    pub fn initial() -> Self {
        Self { integral: 0.0, prev_error: 0.0 }
    }
}

pub fn pid_step(
    config: &PidConfig,
    state: &PidState,
    target: f32,
    current: f32,
    dt: f32,
) -> PidOutput {
    let error = target - current;

    let p_term = config.kp * error;

    let new_integral = state.integral + error * dt;
    let i_term = config.ki * new_integral;

    let derivative = (error - state.prev_error) / dt;
    let d_term = config.kd * derivative;

    let raw_output = p_term + i_term + d_term;
    let clamped = raw_output.clamp(config.output_min, config.output_max);

    // 와인드업 방지: 출력이 클램핑된 경우 적분항을 더 이상 쌓지 않음
    let next_integral = if raw_output != clamped {
        state.integral // 클램핑 중일 때 적분 동결
    } else {
        new_integral
    };

    PidOutput {
        value: clamped,
        p_term,
        i_term,
        d_term,
        next_state: PidState {
            integral: next_integral,
            prev_error: error,
        },
    }
}
```

`pid_step`은 순수한 계산입니다.

- 전역 변수 없음
- 같은 `(config, state, target, current, dt)` 조합이면 항상 같은 결과
- 호출 횟수, 순서, 시점이 출력에 영향을 주지 않음

---

### 여러 제어기를 독립적으로 운용

전역 상태가 없으므로 속도 PID와 조향 PID가 완전히 독립적입니다.

```rust
struct VehicleController {
    speed_config:   PidConfig,
    heading_config: PidConfig,
    speed_state:    PidState,
    heading_state:  PidState,
}

impl VehicleController {
    fn new(speed_cfg: PidConfig, heading_cfg: PidConfig) -> Self {
        Self {
            speed_config:   speed_cfg,
            heading_config: heading_cfg,
            speed_state:    PidState::initial(),
            heading_state:  PidState::initial(),
        }
    }

    fn update(
        &self,
        target_speed: f32, current_speed: f32,
        target_heading: f32, current_heading: f32,
        dt: f32,
    ) -> (PidOutput, PidOutput, VehicleController) {
        let speed_out = pid_step(
            &self.speed_config, &self.speed_state,
            target_speed, current_speed, dt,
        );
        let heading_out = pid_step(
            &self.heading_config, &self.heading_state,
            target_heading, current_heading, dt,
        );

        let next = VehicleController {
            speed_config:   self.speed_config.clone(),
            heading_config: self.heading_config.clone(),
            speed_state:    speed_out.next_state.clone(),
            heading_state:  heading_out.next_state.clone(),
        };

        (speed_out, heading_out, next)
    }
}
```

`update`는 새 `VehicleController`를 반환합니다. 이전 상태를 변경하지 않습니다.

---

## 액션은 루프 바깥에서

실제 차량에서 실행할 때는 센서 읽기와 액추에이터 출력(액션)을 계산 바깥에 둡니다.

```rust
fn control_loop(
    vehicle: &dyn VehicleSensor,   // 액션: 센서 읽기
    actuator: &dyn VehicleActuator, // 액션: 출력 쓰기
    speed_cfg: PidConfig,
    heading_cfg: PidConfig,
    target_speed: f32,
    target_heading: f32,
) {
    let mut controller = VehicleController::new(speed_cfg, heading_cfg);
    let mut last_time = vehicle.now_ms();

    loop {
        let now = vehicle.now_ms();           // 액션
        let dt = (now - last_time) as f32 / 1000.0;
        last_time = now;

        let current_speed   = vehicle.speed_mps();   // 액션
        let current_heading = vehicle.heading_rad();  // 액션

        // 계산: 상태 전혀 없는 순수 함수
        let (speed_out, heading_out, next_controller) = controller.update(
            target_speed, current_speed,
            target_heading, current_heading,
            dt,
        );

        actuator.set_throttle(speed_out.value);       // 액션
        actuator.set_steering(heading_out.value);     // 액션

        controller = next_controller;
    }
}
```

루프 구조가 명확합니다.

```
[액션] 센서 읽기
    ↓
[계산] pid_step × 2 → 출력값 + 다음 상태
    ↓
[액션] 액추에이터 쓰기
```

---

## 테스트: 시뮬레이션 없이 튜닝 로직 검증

`pid_step`이 순수 함수이므로 수치를 직접 넣어 검증합니다.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> PidConfig {
        PidConfig { kp: 1.0, ki: 0.1, kd: 0.05,
                    output_min: -100.0, output_max: 100.0 }
    }

    #[test]
    fn test_zero_error_gives_zero_output() {
        let out = pid_step(&default_config(), &PidState::initial(), 50.0, 50.0, 0.1);
        assert_eq!(out.value, 0.0);
        assert_eq!(out.next_state.prev_error, 0.0);
    }

    #[test]
    fn test_positive_error_gives_positive_output() {
        // 목표가 현재보다 크면 양의 출력
        let out = pid_step(&default_config(), &PidState::initial(), 60.0, 50.0, 0.1);
        assert!(out.value > 0.0);
    }

    #[test]
    fn test_integral_accumulates_over_steps() {
        let config = default_config();
        let state0 = PidState::initial();

        let out1 = pid_step(&config, &state0, 60.0, 50.0, 0.1);
        let out2 = pid_step(&config, &out1.next_state, 60.0, 50.0, 0.1);

        // 오차가 같아도 적분항이 쌓여 두 번째 출력이 더 큰지 확인
        assert!(out2.i_term > out1.i_term);
    }

    #[test]
    fn test_output_is_clamped() {
        let config = PidConfig { kp: 100.0, output_min: -10.0, output_max: 10.0,
                                 ..default_config() };
        let out = pid_step(&config, &PidState::initial(), 100.0, 0.0, 0.1);
        assert_eq!(out.value, 10.0); // 상한에 클램핑
    }

    #[test]
    fn test_windup_prevention() {
        // 클램핑 중엔 적분이 동결돼야 함
        let config = PidConfig { kp: 100.0, ki: 1.0, output_min: -10.0, output_max: 10.0,
                                 ..default_config() };
        let state0 = PidState::initial();
        let out1 = pid_step(&config, &state0, 100.0, 0.0, 0.1);
        let out2 = pid_step(&config, &out1.next_state, 100.0, 0.0, 0.1);

        // 와인드업 방지로 integral이 동결됐으면 두 스텝의 integral이 같아야 함
        assert_eq!(out1.next_state.integral, out2.next_state.integral);
    }

    #[test]
    fn test_step_by_step_reproducibility() {
        // 같은 입력 시퀀스는 항상 같은 출력 시퀀스를 만들어야 함
        let config = default_config();
        let inputs = vec![(60.0_f32, 50.0_f32), (60.0, 52.0), (60.0, 58.0)];

        let replay = |inputs: &[(f32, f32)]| {
            let mut state = PidState::initial();
            inputs.iter().map(|(t, c)| {
                let out = pid_step(&config, &state, *t, *c, 0.1);
                state = out.next_state.clone();
                out.value
            }).collect::<Vec<_>>()
        };

        assert_eq!(replay(&inputs), replay(&inputs));
    }
}
```

하드웨어도, 시뮬레이터도, 타이밍 의존도 없습니다. PID 게인 튜닝 로직을 CI에서 자동으로 검증합니다.

---

## 로그에서 상태 재현하기

순수 함수의 가장 큰 이점은 **재현 가능성**입니다. 로그에 입력값만 기록해두면 어떤 타임스텝이든 그 시점의 출력을 정확히 재현할 수 있습니다.

```rust
#[derive(Debug, Clone)]
struct ControlLog {
    target_speed: f32,
    current_speed: f32,
    target_heading: f32,
    current_heading: f32,
    dt: f32,
}

fn replay_logs(
    config: &PidConfig,
    logs: &[ControlLog],
) -> Vec<PidOutput> {
    let mut state = PidState::initial();
    logs.iter().map(|log| {
        let out = pid_step(config, &state, log.target_speed, log.current_speed, log.dt);
        state = out.next_state.clone();
        out
    }).collect()
}
```

사고 분석, A/B 게인 비교, 오프라인 튜닝 모두 로그 파일 하나면 충분합니다.

---

## 정리

| | 전역 상태 방식 | 값 전달 방식 |
|---|---|---|
| 테스트 | 전역 변수 수동 리셋 필요 | 입력만 넣으면 됨 |
| 여러 인스턴스 | 변수 별도 복제 필요 | 구조체 여러 개 생성 |
| 재현성 | 실행 순서에 따라 달라짐 | 같은 입력 → 같은 출력 |
| 로그 재현 | 불가능 | 입력 로그만 있으면 가능 |
| 멀티스레드 | `unsafe` 또는 락 필요 | 불변 값이므로 안전 |

PID 제어기는 단순해 보이지만 전역 상태를 암묵적으로 들고 다니는 구조입니다. 상태를 값으로 만들면 제어 로직이 순수 계산이 되고, 자율주행 소프트웨어에서 가장 어려운 문제 중 하나인 **재현 가능한 테스트**가 자연스럽게 해결됩니다.

---

*관련 글: [액션/계산/데이터](/posts/programming/functional-actions-calculations-data/), [불변 데이터와 구조적 공유](/posts/programming/functional-immutable-data/), [자율주행 센서 파이프라인](/posts/programming/autonomous-sensor-pipeline/), [자율주행 모드 전이를 타입으로 만들기](/posts/programming/autonomous-state-machine/), [ROS2 콜백을 함수형으로](/posts/programming/autonomous-ros2-functional/), [시뮬레이션 회귀 테스트 설계](/posts/programming/autonomous-simulation-regression/)*
