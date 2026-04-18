---
title: "Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-To-End Autonomous Driving"
date: 2026-04-18T10:40:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving", "Benchmark & Dataset"]
tags: ["Autonomous Driving", "Benchmark", "End-to-End Planning", "Closed-Loop", "Simulation"]
---

## 개요
- **저자**: Xiaosong Jia, Zhenjie Yang, Qifeng Li, Zhiyuan Zhang, Junchi Yan
- **소속**: Shanghai Jiao Tong University
- **발행년도**: 2024 (NeurIPS 2024 Datasets and Benchmarks Track)
- **주요 내용**: E2E 자율주행(E2E-AD) 시스템의 다중 능력(Multi-Ability)을 폐쇄 루프(Closed-Loop) 방식으로 평가하는 최초의 종합 벤치마크 Bench2Drive를 제안

## 한계 극복

이 논문이 기존 연구의 어떤 한계를 극복하기 위해 작성되었는지 설명합니다.

- **기존 한계 1 - 개방 루프 평가의 부적합성**: nuScenes 같은 데이터셋에서 L2 오류와 충돌률을 메트릭으로 사용하는 개방 루프(open-loop) 방식은 분포 이동(distribution shift)과 인과 혼동(causal confusion) 문제로 실제 주행 성능을 제대로 반영하지 못함
- **기존 한계 2 - 폐쇄 루프 평가의 분산 문제**: CARLA의 Town05Long, Longest6 같은 기존 폐쇄 루프 벤치마크는 7~10km에 달하는 긴 경로를 사용하여, 지수 감쇠 기반 Driving Score가 매우 높은 분산을 보이고 현재 리더보드에서 대부분의 방법이 10점 미만을 기록
- **기존 한계 3 - 공정한 비교 불가**: 각 팀이 자체 데이터를 수집하여 훈련하므로 알고리즘 수준의 공정한 비교가 불가능
- **이 논문의 접근 방식**: 44개 상호작용 시나리오 × 5개 경로 = 220개 짧은 경로(약 150m)를 통해 개별 스킬을 독립적으로 평가하고, Think2Drive RL 전문가가 수집한 200만 프레임의 공식 학습 데이터셋을 제공하여 알고리즘 간 공정 비교를 가능하게 함

## 목차
- Section 1: Introduction
- Section 2: Related Work (Planning Benchmarks, End-to-End Autonomous Driving)
- Section 3: Bench2Drive (Data Collection Agent, Expert Dataset, Multi-Ability Evaluation)
- Section 4: Experiments (Baselines & Datasets, Results, Case Analysis)
- Section 5: Conclusion

---

## Section 1: Introduction

**요약**

파운데이션 모델의 급격한 발전과 함께 E2E-AD 시스템에 대한 관심이 높아지고 있습니다. 그러나 기존 평가 방법들은 두 가지 큰 문제를 가지고 있습니다. 첫째, nuScenes 같은 개방 루프 평가는 실제 주행 능력을 충분히 반영하지 못하며, 심지어 센서 입력 없이 자아 상태(ego state)만 인코딩해도 복잡한 방법과 비슷한 L2 오류를 달성할 수 있습니다. 둘째, CARLA의 기존 폐쇄 루프 벤치마크(Town05Long, Longest6, Leaderboard V2)는 경로가 너무 길어 Driving Score의 분산이 크고 방법 간 구분이 어렵습니다.

Bench2Drive는 이를 해결하기 위해 (1) 강력한 RL 전문가(Think2Drive)가 수집한 200만 프레임의 통일된 공식 훈련 세트, (2) 44개 시나리오 × 5개 경로 = 220개 단거리 폐쇄 루프 평가 경로, (3) Merging/Overtaking/Emergency Brake/Give Way/Traffic Sign의 5가지 능력에 대한 다차원 평가를 제공합니다.

**핵심 개념**
- **End-to-End Autonomous Driving (E2E-AD)**: 원시 센서 입력(카메라, LiDAR 등)을 받아 직접 제어 신호나 경로를 출력하는 통합 자율주행 시스템. 모듈식 인식 파이프라인과 달리 데이터 기반으로 스케일 가능
- **Open-Loop Evaluation**: 녹화된 전문가 경로를 재생하며 예측된 경로와의 L2 오류를 측정. 에고 차량의 행동이 환경에 영향을 미치지 않아 실제 주행 능력 평가에 한계
- **Closed-Loop Evaluation**: 에고 차량이 실제로 환경에서 주행하며 행동이 다음 상태에 영향을 미치는 현실적 평가 방식
- **Distribution Shift**: 훈련 데이터(전문가 경로)와 실제 배포 시 분포가 달라 성능이 저하되는 현상. 개방 루프 평가는 이를 감지하지 못함

---

## Section 2: Related Work

**요약**

기존 자율주행 계획 벤치마크들의 특성과 한계를 정리합니다. nuScenes는 개방 루프 메트릭만 제공하고 데이터의 75%가 직선 주행이라 의사결정 능력 평가에 부적합합니다. nuPlan과 Waymax는 폐쇄 루프를 지원하지만 바운딩 박스 수준 평가에 그쳐 E2E-AD에 적합하지 않습니다. CARLA Leaderboard V2는 공식 훈련 데이터가 없어 시스템 수준 비교만 가능합니다.

**핵심 개념**
- **nuScenes**: 인기 있는 자율주행 데이터셋. 개방 루프 메트릭(L2, 충돌률) 제공. 75%가 직진 시나리오로 복잡한 상호작용 평가에 불충분
- **CARLA Leaderboard V2**: 39개의 도전적 시나리오로 구성된 CARLA 기반 벤치마크. 공식 훈련 데이터 부재와 긴 경로로 인한 높은 분산이 문제
- **Imitation Learning vs RL**: 대부분의 E2E-AD는 전문가 시연을 모방하는 imitation learning 방식. 상호작용 행동(합류, 추월 등)의 게임적 사고 과정을 학습하기 어려운 한계가 있음

---

## Section 3: Bench2Drive

**요약**

Bench2Drive는 세 가지 핵심 구성 요소로 이루어집니다.

### 3.1 데이터 수집 에이전트 (Think2Drive)

시뮬레이션의 데이터 수집에는 특권 정보(주변 에이전트의 위치, 상태, 의도, 신호등 상태 등)를 사용할 수 있는 교사 모델이 활용됩니다. Bench2Drive는 CARLA에서 44개 시나리오를 모두 해결할 수 있는 유일한 전문가 모델인 **Think2Drive** (세계 모델 기반 RL)를 사용합니다.

### 3.2 전문가 데이터셋

| 센서 | 사양 |
|------|------|
| LiDAR | 64채널, 85m 범위, 초당 600,000 포인트 |
| Camera | 6개 서라운드, 900×1600 해상도 |
| Radar | 5개, 100m 범위, 수평/수직 30° FoV |
| IMU/GNSS | 위치, 속도, 가속도, 각속도 |
| BEV Camera | 디버깅 및 원격 감지용 |

어노테이션: 3D 바운딩 박스, 깊이, 의미론적/인스턴스 분할, HD-Map, RL 가치 추정 및 특징

데이터 규모:
- **Full**: 13,638개 클립, 200만 프레임
- **Base**: 1,000개 클립 (8×RTX3090 서버 적합)
- **Mini**: 10개 클립 (디버깅/시각화용)

각 클립은 약 150m, 단일 특정 시나리오를 포함하며, 44개 시나리오 × 23개 날씨 × 12개 도시에 걸쳐 균등 분포

### 3.3 다중 능력 평가

기존 벤치마크가 모든 경로의 평균 점수를 사용하는 것과 달리, Bench2Drive는 5가지 운전 능력별로 분리된 평가를 제공합니다:

| 능력 | 주요 시나리오 예시 |
|------|------------------|
| Merging | CrossingBicycleFlow, LaneChange, SignalizedJunctionRightTurn 등 |
| Overtaking | ConstructionObstacle, ParkedObstacle, VehicleOpenDoorTwoWays 등 |
| Emergency Brake | DynamicObjectCrossing, PedestrianCrossing, StaticCutIn 등 |
| Give Way | InvadingTurn, YieldToEmergencyVehicle |
| Traffic Sign | TJunction, VanillaSignalizedTurnEncounterRedLight 등 |

**핵심 개념**
- **Curriculum Learning**: 개별 스킬을 독립적으로 학습하는 방식. 150m 단거리 클립 설계가 이를 지원
- **Long-tail Distribution**: 자율주행에서 드문 위험 시나리오. nuScenes의 75% 직진 문제처럼 학습 데이터가 편향될 때 발생
- **Privileged Information**: 시뮬레이션에서만 접근 가능한 정보(신호등 실제 상태, 주변 차량 의도 등). 실제 도로에서는 사용 불가

**수식 - 평가 메트릭**

$$\text{Success Rate} = \frac{n_\text{success}}{n_\text{total}}$$

**수식 설명**
- **$n_\text{success}$**: 교통 위반 없이 목적지에 도달한 경로 수
- **$n_\text{total}$**: 전체 평가 경로 수 (220개)
- 경로는 제한 시간 내 교통 위반 없이 목적지 도달 시 성공으로 간주

$$\text{Driving Score} = \frac{1}{n_\text{total}} \sum_{i=1}^{n_\text{total}} \text{Route-Completion}_i * \prod_{j=1}^{n_{i,\text{penalty}}} p_{i,j} \tag{1}$$

**수식 설명**
이 수식은 각 경로의 완주율에 위반 패널티를 곱해 평균을 낸 종합 주행 점수입니다:
- **$\text{Route-Completion}_i$**: i번째 경로에서 완주한 거리 비율 (0~1)
- **$p_{i,j}$**: i번째 경로의 j번째 위반에 대한 패널티 (0~1, 위반 심각도에 따라 감소)
- **$\prod$**: 모든 위반 패널티를 곱하는 연산. 위반이 많을수록 점수가 기하급수적으로 감소
- 예: 90% 완주 + 빨간불 위반 1회(패널티 0.7) → 0.9 × 0.7 = 0.63

$$\text{Speed Percentage} = \frac{\text{Ego Vehicle's Speed}}{\text{Average Speed of Nearby Vehicles}} \tag{2}$$

**수식 설명**
- 에고 차량의 속도를 주변 차량 평균 속도와 비교
- 100% 이상이면 주변 차량보다 빠르게 주행 (효율적)
- 너무 느리면 패널티로 반영됨
- Bench2Drive는 체크포인트를 4개 → 20개로 늘려 경로의 5%마다 속도 체크 (더 세밀한 평가)

$$\text{Driving Efficiency} = \frac{\sum_i \text{Speed Percentage}_i}{\text{Speed Check Times}} \tag{3}$$

$$\text{Frame Variable Smoothness (FVS)} = \begin{cases} \text{True} & \text{if lower bound} \leq p_i \leq \text{upper bound} \\ \text{False} & \text{otherwise} \end{cases} \tag{4}$$

**수식 설명 (주행 부드러움)**
인간 전문가 주행 데이터에서 도출한 임계값을 기준으로 각 프레임의 운동 변수가 안전 범위 내에 있는지 판단합니다:
- **종방향 가속도**: [-4.05, 2.40] m/s²
- **횡방향 가속도**: [-4.89, 4.89] m/s²
- **Yaw rate**: [-0.95, 0.95] rad/s
- **Jerk (종방향)**: [-4.13, 4.13] m/s³

$$\text{Smoothness} = \frac{\text{Number of Smooth Segments}}{\text{Total Segments}}$$

**수식 설명**
- nuPlan과 달리 전체 궤적이 아닌 20프레임 세그먼트 단위로 부드러움을 평가
- 앞차의 급정거로 인한 급제동처럼 불가피한 상황에서의 급격한 동작이 전체 점수를 왜곡하는 문제를 해결

---

## Section 4: Experiments

**요약**

6가지 베이스라인 E2E-AD 방법을 Base 훈련 세트(1,000 클립)로 학습하고 Bench2Drive에서 평가했습니다.

### 베이스라인 모델

| 방법 | 특징 |
|------|------|
| **UniAD** | Transformer Query로 인식-예측-계획 통합, BEVFormer 사용 |
| **VAD** | Transformer Query + 벡터화 장면 표현으로 효율성 향상 |
| **AD-MLP** | 에고 히스토리만 MLP에 입력하는 단순 기준 모델 |
| **TCP** | 전방 카메라 + 에고 상태 → 궤적 및 제어 신호 예측 |
| **ThinkTwice** | 계획 경로를 레이어별로 정제하는 coarse-to-fine 방식 |
| **DriveAdapter** | 전문가 인식과 계획을 어댑터 모듈로 연결 |

`*` 표시: 전문가 특징 증류(expert feature distillation) 적용

### 주요 실험 결과

**개방 루프 vs 폐쇄 루프 결과 (Base 훈련 세트)**

| 방법 | Avg L2↓ | Driving Score↑ | Success Rate(%)↑ |
|------|---------|---------------|-----------------|
| AD-MLP | 3.64 | 18.05 | 0.00 |
| UniAD-Base | **0.73** | 45.81 | **16.36** |
| VAD | 0.91 | 42.35 | 15.00 |
| ThinkTwice* | 0.95 | 62.44 | 31.23 |
| DriveAdapter* | 1.01 | **64.22** | **33.08** |

**다중 능력 결과 (Base 훈련 세트)**

| 방법 | Merging | Overtaking | Emergency Brake | Give Way | Traffic Sign | Mean |
|------|---------|-----------|----------------|----------|-------------|------|
| AD-MLP | 0.00 | 0.00 | 0.00 | 0.00 | 4.35 | 0.87 |
| UniAD-Base | 14.10 | 17.78 | 21.67 | 10.00 | 14.21 | 15.55 |
| VAD | 8.11 | 24.44 | 18.64 | 20.00 | 19.15 | **18.07** |
| DriveAdapter* | **28.82** | **26.38** | 48.76 | **50.00** | **56.43** | **42.08** |

**핵심 개념**
- **Expert Feature Distillation**: 전문가 모델(Think2Drive)의 특징을 학생 모델에 증류하는 기법. 고차원 센서 입력의 과적합 문제를 완화하며, 적용 시 Driving Score와 Success Rate가 크게 향상됨
- **Causal Confusion**: 모델이 올바른 이유가 아닌 잘못된 상관관계를 학습하는 현상. 예: 브레이크등이 켜진 차량 앞에서 감속하는 법을 배우는 대신, 특정 도로 패턴에서 감속을 학습

### 주요 발견

1. **개방 루프 메트릭은 모델 수렴은 확인하지만 심화 비교는 불가**: AD-MLP는 L2 오류가 높고 폐쇄 루프에서 완전 실패(SR=0%). UniAD-Base는 낮은 L2이지만 VAD보다 폐쇄 루프 성능이 낮음. 개방 루프는 분포 이동과 인과 혼동 문제를 무시

2. **전문가 특징 증류가 중요한 가이드 역할**: E2E-AD의 고차원 입력 공간(카메라, 포인트 클라우드)은 과적합을 유발. 전문가 특징 증류 적용 시 모든 메트릭에서 큰 성능 향상

3. **상호작용 행동 학습이 어려움**: Merging, Overtaking, Emergency Brake 능력 점수가 전반적으로 낮음. 원인 두 가지: (1) 클립 내 상호작용 프레임 비율이 낮은 장기 분포 문제, (2) 모방 학습이 게임적 사고 과정을 학습하기 어려운 패러다임 한계

---

## Section 5: Conclusion

**요약**

Bench2Drive는 E2E-AD 방법의 폐쇄 루프 평가를 위한 새로운 벤치마크로, 다음을 공개 제공합니다:
- 200만 프레임의 완전 어노테이션된 공식 훈련 데이터셋
- 다중 능력 평가 툴킷 (44개 시나리오 × 5개 경로 = 220개 경로)
- 여러 E2E-AD 베이스라인 구현체

**한계**: CARLA 시뮬레이션과 실제 세계 사이의 렌더링 갭이 존재합니다. 실제 세계 데이터셋은 현실적이지만 반응적이지 않고, 시뮬레이션은 반응적이지만 카툰 스타일입니다. 디퓨전 모델 기반의 현실적이고 반응적인 렌더링이 미래 방향으로 제시됩니다.

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **E2E-AD (End-to-End Autonomous Driving)** | 센서 → 제어까지 하나의 통합 신경망으로 처리하는 자율주행 방식 |
| **Closed-Loop Evaluation** | 에고 차량의 행동이 환경에 실시간으로 영향을 미치는 현실적 평가 |
| **Driving Score** | 경로 완주율 × 위반 패널티의 곱. 지수 감쇠로 인해 긴 경로에서 분산이 큼 |
| **Success Rate (SR)** | 교통 위반 없이 목적지까지 도달한 경로 비율 |
| **Expert Feature Distillation** | 전문가 모델의 내부 특징을 학생 모델에 전달하여 학습 효율 향상 |
| **Think2Drive** | CARLA에서 44개 시나리오를 모두 해결하는 세계 모델 기반 RL 전문가 |
| **Multi-Ability Evaluation** | Merging, Overtaking, Emergency Brake, Give Way, Traffic Sign 5가지 능력별 독립 평가 |
| **Simulation-Reality Gap** | 시뮬레이션 환경과 실제 도로 주행 간의 시각적·물리적 차이 |

## 결론 및 시사점

Bench2Drive는 E2E 자율주행 시스템 평가의 세 가지 근본적인 문제를 해결합니다:

1. **개방 루프의 한계**: L2 오류는 모델 수렴 확인에만 유효하며 실제 주행 능력을 반영하지 못함을 실험적으로 입증. AD-MLP(L2 높음 → 폐쇄 루프 완전 실패), UniAD-Base(L2 낮음 → 폐쇄 루프 VAD보다 낮음)가 이를 보여줌

2. **공정 비교 기반 마련**: 통일된 공식 훈련 데이터셋 제공으로 알고리즘 수준의 공정 비교 가능. 이전에는 각 팀이 자체 데이터로 학습하여 시스템 수준 비교만 가능했음

3. **세분화된 능력 평가**: 220개 단거리 경로로 5가지 능력을 독립 평가함으로써 기존 단일 점수 방식보다 구체적인 강점/약점 파악 가능

**실무적 시사점**: 합성 데이터 생성 및 회귀 테스트 관점에서, Bench2Drive의 시나리오 분류 체계(44개 × 5가지 능력)는 테스트 커버리지를 구조화하는 데 참고할 수 있으며, 단거리 클립 기반 평가 방식은 특정 시나리오에 대한 회귀 테스트 설계에 적합합니다.
