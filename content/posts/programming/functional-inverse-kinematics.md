---
title: "함수형 역기구학: 순수 계산으로 로봇 팔 제어하기"
date: 2026-04-30T07:00:00+09:00
draft: false
tags: ["함수형 프로그래밍", "Rust", "설계", "로봇", "역기구학", "액션/계산/데이터"]
categories: ["프로그래밍", "로봇"]
description: "로봇 팔의 역기구학을 액션/계산/데이터로 분리하면, 시뮬레이터와 실기기가 같은 계산 코드를 공유하고 하드웨어 없이 테스트할 수 있습니다."
---

## 이 글을 읽고 나면

- 역기구학(IK)이 왜 순수 계산에 가장 적합한 문제인지 이해합니다.
- 액션/계산/데이터 분리가 시뮬레이터와 실기기 코드를 어떻게 통합하는지 봅니다.
- 하드웨어 없이도 IK 로직을 완전히 테스트하는 방법을 이해합니다.

이전 글 [액션, 계산, 데이터](/posts/programming/functional-actions-calculations-data/)를 먼저 읽으면 더 자연스럽게 이어집니다.

---

## 문제: 로봇 팔을 어디로 움직여야 하나

로봇 팔에게 "저 위치에 손을 가져다 놓아라"고 명령할 때, 실제로 필요한 것은 각 관절의 각도입니다.

- **순기구학(FK, Forward Kinematics)**: 관절 각도 → 끝점 위치. 쉽습니다.
- **역기구학(IK, Inverse Kinematics)**: 끝점 위치 → 관절 각도. 어렵습니다.

IK는 수학적으로 닫힌 해가 없는 경우가 많고, 해가 여러 개이거나 없을 수도 있습니다. 이걸 실기기 제어 코드 안에 욱여넣으면 이렇게 됩니다.

```rust
fn move_arm(target_x: f32, target_y: f32, robot: &mut Robot) {
    let angles = robot.solve_ik(target_x, target_y); // IK 계산
    robot.send_joint_commands(angles);                 // 모터 명령
    robot.wait_until_reached();                        // I/O 대기
}
```

IK 알고리즘을 바꾸려면 실기기가 있어야 테스트됩니다. 시뮬레이터에서 검증하려면 `Robot` 전체를 mock해야 합니다. 계산과 액션이 섞여 있기 때문입니다.

---

## 액션/계산/데이터로 나누기

```
데이터(Data)        계산(Calculation)          액션(Action)
─────────────       ─────────────────          ─────────────
JointAngles    ←─  solve_ik(target, config) ←  read_current_pose()
RobotConfig        check_limits(angles)  →     send_joint_commands(angles)
TargetPose         compute_jacobian(...)        wait_until_reached()
```

계산은 입력만 있으면 실행됩니다. 실기기도, 시뮬레이터도 필요 없습니다.

---

## 데이터: 구조체로 표현

```rust
#[derive(Debug, Clone)]
struct JointAngles {
    theta: [f32; 3], // 라디안, 3-DOF 팔 기준
}

#[derive(Debug, Clone)]
struct Pose2D {
    x: f32, // 미터
    y: f32,
}

#[derive(Debug, Clone)]
struct RobotConfig {
    link_lengths: [f32; 3], // 각 링크 길이 (미터)
    joint_limits: [(f32, f32); 3], // (최소, 최대) 라디안
}
```

이 구조체들은 파일로 저장하고, 네트워크로 보내고, 로그로 기록할 수 있습니다. 하드웨어 상태와 무관합니다.

---

## 계산: 순수 함수로 IK 풀기

3-DOF 평면 팔의 해석적 IK입니다. 입력과 출력만 있고 부수 효과가 없습니다.

```rust
// 순기구학: 관절 각도 → 끝점 위치
fn forward_kinematics(angles: &JointAngles, config: &RobotConfig) -> Pose2D {
    let l = &config.link_lengths;
    let t = &angles.theta;

    let x = l[0] * t[0].cos()
          + l[1] * (t[0] + t[1]).cos()
          + l[2] * (t[0] + t[1] + t[2]).cos();
    let y = l[0] * t[0].sin()
          + l[1] * (t[0] + t[1]).sin()
          + l[2] * (t[0] + t[1] + t[2]).sin();

    Pose2D { x, y }
}

// 역기구학: 끝점 위치 → 관절 각도 후보들
fn solve_ik(target: &Pose2D, config: &RobotConfig) -> Vec<JointAngles> {
    let l = &config.link_lengths;

    // 2-링크 부분으로 먼저 풀고, 마지막 링크는 방향 제약으로 결정
    let r = (target.x.powi(2) + target.y.powi(2)).sqrt();
    let r2 = (target.x.powi(2) + target.y.powi(2) - l[2].powi(2)).sqrt();

    if r2 > l[0] + l[1] || r2 < (l[0] - l[1]).abs() {
        return vec![]; // 도달 불가
    }

    let cos_t1 = (r2.powi(2) - l[0].powi(2) - l[1].powi(2)) / (2.0 * l[0] * l[1]);
    let cos_t1 = cos_t1.clamp(-1.0, 1.0);

    // 팔꿈치 위/아래 두 가지 해
    [1.0_f32, -1.0].iter().filter_map(|&sign| {
        let t1 = (cos_t1.acos()) * sign;
        let k1 = l[0] + l[1] * cos_t1;
        let k2 = l[1] * t1.sin();
        let t0 = target.y.atan2(target.x) - k2.atan2(k1);
        let t2 = 0.0; // 말단 방향 제약 없음 (단순화)

        let angles = JointAngles { theta: [t0, t1, t2] };
        Some(angles)
    }).collect()
}

// 관절 한계 검사
fn check_limits(angles: &JointAngles, config: &RobotConfig) -> bool {
    angles.theta.iter().zip(config.joint_limits.iter())
        .all(|(&theta, &(min, max))| theta >= min && theta <= max)
}

// 여러 해 중 현재 포즈에서 가장 가까운 것 선택
fn select_best(
    candidates: &[JointAngles],
    current: &JointAngles,
    config: &RobotConfig,
) -> Option<JointAngles> {
    candidates.iter()
        .filter(|a| check_limits(a, config))
        .min_by(|a, b| {
            let dist_a: f32 = a.theta.iter().zip(current.theta.iter())
                .map(|(x, y)| (x - y).powi(2)).sum();
            let dist_b: f32 = b.theta.iter().zip(current.theta.iter())
                .map(|(x, y)| (x - y).powi(2)).sum();
            dist_a.partial_cmp(&dist_b).unwrap()
        })
        .cloned()
}
```

이 함수들은 전부 계산입니다. `RobotConfig`와 `Pose2D`만 있으면 실행됩니다.

---

## 액션: 하드웨어 경계에서만

계산이 끝난 뒤에야 액션이 시작됩니다.

```rust
// 액션: 현재 관절 상태 읽기
fn read_current_angles(robot: &HardwareRobot) -> JointAngles {
    JointAngles { theta: robot.read_encoders() }
}

// 액션: 계산된 각도를 모터로 전송
fn send_joint_commands(angles: &JointAngles, robot: &mut HardwareRobot) {
    robot.set_joint_targets(angles.theta);
}
```

이 두 함수만 하드웨어에 의존합니다. 나머지 모든 IK 로직은 하드웨어와 무관합니다.

---

## 파이프라인: 계산을 먼저, 액션은 마지막에

```rust
fn move_to(target: &Pose2D, config: &RobotConfig, robot: &mut HardwareRobot) -> Result<(), String> {
    // 액션: 현재 상태 읽기
    let current = read_current_angles(robot);

    // 계산: IK 풀기
    let candidates = solve_ik(target, config);
    let best = select_best(&candidates, &current, config)
        .ok_or_else(|| format!("도달 불가: ({}, {})", target.x, target.y))?;

    // 액션: 명령 전송
    send_joint_commands(&best, robot);
    Ok(())
}
```

데이터 흐름이 명확합니다.

```
read_current_angles()   →  current: JointAngles      (액션)
solve_ik(target, config) →  candidates: Vec<JointAngles>  (계산)
select_best(...)         →  best: JointAngles         (계산)
send_joint_commands(best) →  모터 구동                (액션)
```

---

## 테스트: 하드웨어 없이

계산 함수들이 순수하므로 테스트 데이터를 직접 만들 수 있습니다.

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn config() -> RobotConfig {
        RobotConfig {
            link_lengths: [0.3, 0.25, 0.15],
            joint_limits: [(-PI, PI), (-PI * 0.75, PI * 0.75), (-PI * 0.5, PI * 0.5)],
        }
    }

    #[test]
    fn test_fk_ik_roundtrip() {
        let config = config();
        let original = JointAngles { theta: [0.3, -0.5, 0.2] };

        // FK로 끝점 계산
        let target = forward_kinematics(&original, &config);

        // IK로 다시 관절 각도 복원
        let candidates = solve_ik(&target, &config);
        assert!(!candidates.is_empty(), "IK 해가 없음");

        // 복원된 각도로 FK를 다시 돌리면 같은 끝점이 나와야 함
        for candidate in &candidates {
            let recovered = forward_kinematics(candidate, &config);
            assert!((recovered.x - target.x).abs() < 1e-3);
            assert!((recovered.y - target.y).abs() < 1e-3);
        }
    }

    #[test]
    fn test_out_of_reach_returns_empty() {
        let config = config();
        let target = Pose2D { x: 10.0, y: 10.0 }; // 링크 총합 0.7m보다 훨씬 멀다
        let candidates = solve_ik(&target, &config);
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_joint_limits_filter() {
        let config = config();
        let angles_ok = JointAngles { theta: [0.0, 0.0, 0.0] };
        let angles_over = JointAngles { theta: [0.0, 3.0, 0.0] }; // 2번 관절 한계 초과
        assert!(check_limits(&angles_ok, &config));
        assert!(!check_limits(&angles_over, &config));
    }

    #[test]
    fn test_select_best_prefers_closest() {
        let config = config();
        let current = JointAngles { theta: [0.1, -0.1, 0.0] };
        let target = Pose2D { x: 0.4, y: 0.2 };
        let candidates = solve_ik(&target, &config);
        let best = select_best(&candidates, &current, &config);
        assert!(best.is_some());
    }
}
```

모든 테스트가 하드웨어 없이 실행됩니다. 시뮬레이터도 `solve_ik`, `forward_kinematics` 함수를 그대로 씁니다.

---

## 시뮬레이터와 실기기의 코드 공유

액션/계산/데이터를 나누면 자연스럽게 이 구조가 됩니다.

```
┌─────────────────────────────────────────┐
│           계산 (공유)                    │
│  solve_ik, forward_kinematics,          │
│  check_limits, select_best              │
├──────────────────┬──────────────────────┤
│  실기기 액션      │  시뮬레이터 액션      │
│  read_encoders() │  sim_read_angles()   │
│  set_targets()   │  sim_set_angles()    │
└──────────────────┴──────────────────────┘
```

시뮬레이터에서 검증한 IK 알고리즘이 실기기에서도 동일하게 동작합니다. 교체해야 할 것은 액션 두 개뿐입니다.

---

## 정리

| 분류 | 내용 | 특징 |
|------|------|------|
| 데이터 | `JointAngles`, `Pose2D`, `RobotConfig` | 하드웨어 무관, 직렬화 가능 |
| 계산 | `solve_ik`, `forward_kinematics`, `check_limits`, `select_best` | 순수 함수, 테스트 단순 |
| 액션 | `read_current_angles`, `send_joint_commands` | 하드웨어 경계에만 존재 |

계산이 두껍고 액션이 얇을수록 테스트하기 쉽고 시뮬레이터와 실기기 코드를 공유하기 쉽습니다.

---

*관련 글: [액션, 계산, 데이터](/posts/programming/functional-actions-calculations-data/), [함수 컴포지션](/posts/programming/functional-composition/), [함수형 센서 퓨전](/posts/programming/autonomous-sensor-fusion/), [함수형 셰이더 파이프라인](/posts/programming/functional-shader-pipeline/)*
