---
title: "VAD: Vectorized Scene Representation for Efficient Autonomous Driving"
date: 2026-04-10T08:30:00+09:00
draft: false
categories: ["Papers"]
tags: ["Autonomous Driving", "End-to-End Planning", "Vectorized Representation", "BEV"]
---

## 개요

- **저자**: Bo Jiang, Shaoyu Chen, Qing Xu, Bencheng Liao, Jiajie Chen, Helong Zhou, Qian Zhang, Wenyu Liu, Chang Huang, Xinggang Wang
- **소속**: Huazhong University of Science & Technology, Horizon Robotics
- **발행년도**: 2023 (arXiv:2303.12077)
- **주요 내용**: 자율주행을 위한 완전 벡터화된 장면 표현(VAD) 프레임워크 제안. 래스터화된 밀집 맵 대신 벡터화된 에이전트 모션과 맵 요소를 사용하여 인스턴스 수준의 계획 제약을 명시적으로 모델링함으로써 안전성과 효율성을 동시에 향상시킴.

## 목차

- 1. Introduction: 자율주행에서 벡터화 표현의 필요성
- 2. Related Work: 인식, 모션 예측, 계획 관련 선행 연구
- 3. Method: VAD 아키텍처 및 벡터화 계획 제약
- 4. Experiments: nuScenes 및 CARLA 벤치마크 실험 결과
- 5. Conclusion: 벡터화 패러다임의 가능성

---

## 1. Introduction

**요약**

자율주행은 안전하고 효율적인 궤적 계획을 위해 주변 환경에 대한 포괄적인 이해가 필요합니다. 기존의 방법들은 시맨틱 맵, 점유 맵, 비용 맵 등의 **래스터화된(rasterized) 장면 표현**에 의존했습니다. 하지만 이 방식은 다음과 같은 한계가 있습니다:

1. **계산 비용이 큼**: 밀집된 픽셀 단위 표현으로 인해 처리 부담이 큼
2. **인스턴스 수준 정보 부재**: 개별 에이전트나 맵 요소를 구분하지 못해 정밀한 계획 제약을 세우기 어려움

이 논문은 장면 전체를 **벡터화된 표현(Vectorized Scene Representation)**으로 모델링하는 **VAD(Vectorized Autonomous Driving)**를 제안합니다. VAD는 에이전트의 모션 벡터와 맵 요소를 벡터 형태로 직접 활용하여 계획 제약을 인스턴스 단위로 정의합니다.

**핵심 개념**

- **래스터화 표현의 한계**: 픽셀 기반 밀집 맵은 연산 집약적이고 수작업 후처리가 필요하며, 인스턴스별 정보 활용이 어려움
- **벡터화 표현의 장점**: 경계 벡터, 차선 벡터, 모션 벡터 형태로 장면을 표현하여 경량화 및 인스턴스 수준 제약 가능
- **End-to-End 학습**: 인식부터 계획까지 하나의 모델로 통합 학습 가능

---

## 2. Related Work

**요약**

세 가지 분야의 선행 연구를 검토합니다.

**핵심 개념**

- **Perception (인식)**: BEVFormer, FIERY 등은 멀티카메라 이미지에서 BEV(Bird's Eye View) 특징을 추출. VAD도 BEV 인코더를 백본으로 사용
- **Motion Prediction (모션 예측)**: VectorNet, MapTR 등은 벡터화된 표현으로 에이전트 미래 궤적을 예측. VAD는 이를 계획에 통합
- **Planning (계획)**: UniAD, ST-P3 등의 E2E 자율주행 방법은 래스터화 맵 기반. VAD는 벡터화 맵으로 이를 대체하여 계산 효율 개선

---

## 3. Method

### 3.1 전체 아키텍처 개요

**요약**

VAD의 전체 파이프라인은 4단계로 구성됩니다:

1. **Backbone**: 멀티카메라 이미지에서 특징 추출 후 BEV 특징 생성
2. **Vectorized Scene Learning**: 에이전트 쿼리와 맵 쿼리로 모션 벡터 및 맵 벡터 학습
3. **Planning (Inferring Phase)**: ego 쿼리가 에이전트/맵과 상호작용하여 planning trajectory 출력
4. **Vectorized Planning Constraints (Training Phase)**: 벡터화된 제약으로 계획 궤적 정규화

**핵심 개념**

- **Agent Query ($Q_a$)**: 도로 위 다른 차량 등 동적 에이전트를 나타내는 학습 가능한 쿼리
- **Map Query ($Q_m$)**: 도로 경계, 차선 등 정적 지도 요소를 나타내는 쿼리
- **Ego Query ($Q_{ego}$)**: 자차(ego vehicle)의 계획을 위한 쿼리. 에이전트 및 맵 쿼리와 상호작용하여 계획 정보 획득
- **BEV Encoder**: 6개 카메라 이미지를 $1280 \times 720$ 해상도로 입력받아 BEV 특징 생성

---

### 3.2 Ego-Agent 및 Ego-Map 상호작용

**요약**

Ego 쿼리는 Transformer Decoder를 통해 에이전트 및 맵 쿼리와 상호작용합니다.

**Ego-Agent 상호작용** 수식:

$$Q_{ego} = \text{TransformerDecoder}(q, k, v, q_{pos}, k_{pos})$$

$$q = Q_{ego},\ k = v = Q_a,$$

$$q_{pos} = \text{PE}_1(p_{ego}),\ k_{pos} = \text{PE}_1(p_a)$$

**수식 설명**
- **$Q_{ego}$**: ego 차량의 계획 쿼리 (업데이트됨)
- **$q, k, v$**: Transformer의 query, key, value. ego가 query로서 에이전트 정보를 주목(attention)함
- **$p_{ego}$**: ego 차량의 예측 위치
- **$p_a$**: 각 에이전트의 위치
- **$\text{PE}_1$**: 위치를 임베딩 벡터로 변환하는 단일 레이어 MLP. 상대적 위치 관계를 인코딩

**Ego-Map 상호작용** 수식:

$$Q''_{ego} = \text{TransformerDecoder}(q, k, v, q_{pos}, k_{pos})$$

$$q = Q'_{ego},\ k = v = Q_m,$$

$$q_{pos} = \text{PE}_2(p_{ego}),\ k_{pos} = \text{PE}_2(p_m)$$

**수식 설명**
- **$Q'_{ego}$**: 에이전트와 상호작용 후 업데이트된 ego 쿼리
- **$Q_m$**: 맵 쿼리 (차선, 경계 등 정적 정보)
- **$p_m$**: 맵 요소의 위치 포인트
- **$\text{PE}_2$**: 맵 위치용 MLP 임베딩. 에이전트 상호작용과 구분된 별도의 위치 인코더 사용

---

### 3.3 Planning Head

**요약**

Planning Head는 ego 쿼리와 driving command를 입력받아 미래 궤적을 출력합니다.

$$\hat{V}_{ego} = \text{PlanHead}(\text{ft} = f_{ego},\ \text{cmd} = c)$$

$$f_{ego} = [Q_{ego},\ Q'_{ego},\ s_{ego}]$$

**수식 설명**
- **$\hat{V}_{ego} \in \mathbb{R}^{T_f \times 2}$**: 예측된 미래 궤적. $T_f$개의 미래 타임스텝에서 2D 위치 좌표 시퀀스
- **$f_{ego}$**: ego 특징 벡터. 에이전트 상호작용 전후의 ego 쿼리와 ego 상태($s_{ego}$)를 concatenate
- **$c$**: 고수준 주행 명령 (turn left / turn right / go straight)
- **$[\cdot]$**: concatenation 연산

---

### 3.4 Vectorized Planning Constraint

**요약**

VAD는 세 가지 벡터화된 계획 제약을 통해 학습 시 궤적을 정규화합니다.

#### (1) Ego-Agent 충돌 제약 (Collision Constraint)

자차 계획 궤적과 다른 에이전트의 미래 궤적 사이의 안전 거리를 유지하도록 강제합니다.

$$\mathcal{L}_{col} = \frac{1}{T_f} \sum_{t=1}^{T_f} \sum_{i} \mathcal{L}^{it}_{col},\ i \in \{X, Y\}$$

$$\mathcal{L}^{it}_{col} = \begin{cases} \delta_i - d^{it}_n, & \text{if } d^{it}_n < \delta_i \\ 0, & \text{if } d^{it}_n \geq \delta_i \end{cases}$$

**수식 설명**
- **$\mathcal{L}_{col}$**: 충돌 제약 손실. 모든 미래 타임스텝과 방향에 대해 평균
- **$d^{it}_n$**: 시간 $t$에서 방향 $i$로의 가장 가까운 에이전트까지의 거리
- **$\delta_i$**: 방향별 안전 거리 임계값 ($\delta_X$: 종방향, $\delta_Y$: 횡방향). 나란히 달리는 경우 횡방향($\delta_Y$)보다 종방향($\delta_X$)에서 더 긴 안전 거리 필요
- 거리가 임계값보다 가까울 때만 패널티 부과 (마진 기반 손실)

#### (2) Ego-Boundary 이탈 제약 (Boundary Overstepping Constraint)

계획 궤적이 도로 경계를 벗어나지 않도록 합니다.

$$\mathcal{L}_{bd} = \frac{1}{T_f} \sum_{t=1}^{T_f} \mathcal{L}^t_{bd}$$

$$\mathcal{L}^t_{bd} = \begin{cases} \delta_{bd} - d^t_{bd}, & \text{if } d^t_{bd} < \delta_{bd} \\ 0, & \text{if } d^t_{bd} \geq \delta_{bd} \end{cases}$$

**수식 설명**
- **$d^t_{bd}$**: 시간 $t$의 계획 위치에서 가장 가까운 맵 경계선까지의 거리
- **$\delta_{bd}$**: 경계 안전 임계값 (기본값 1.0m)
- 낮은 신뢰도의 맵 예측은 임계값 $\epsilon_m$으로 필터링 후 사용

#### (3) Ego-Lane 방향 제약 (Lane Directional Constraint)

자차의 이동 방향이 현재 차선 방향과 일치하도록 합니다.

$$\mathcal{L}_{dir} = \frac{1}{T_f} \sum_{t=1}^{T_f} F_{ang}(\hat{v}^t_m,\ \hat{v}^t_{ego})$$

**수식 설명**
- **$\hat{v}^t_m \in \mathbb{R}^{T_f \times 2 \times 2}$**: 가장 가까운 차선 중앙선의 방향 벡터
- **$\hat{v}^t_{ego}$**: 계획 시작점에서의 ego 이동 방향 벡터
- **$F_{ang}(v_1, v_2)$**: 두 벡터 간의 각도 차이. 이 값이 작을수록 차선 방향을 잘 따름
- 신뢰도 $\epsilon_{dir}$ 이하의 차선은 제외하고, 거리 $\delta_{dir}$ 이내의 차선만 사용

---

### 3.5 전체 학습 손실

$$\mathcal{L} = \omega_1 \mathcal{L}_{map} + \omega_2 \mathcal{L}_{mot} + \omega_3 \mathcal{L}_{col} + \omega_4 \mathcal{L}_{bd} + \omega_5 \mathcal{L}_{dir} + \omega_6 \mathcal{L}_{imi}$$

**수식 설명**
- **$\mathcal{L}_{map}$**: 벡터화 맵 학습 손실 (Manhattan distance + focal loss)
- **$\mathcal{L}_{mot}$**: 에이전트 모션 예측 손실 ($l_1$ regression + classification)
- **$\mathcal{L}_{col}, \mathcal{L}_{bd}, \mathcal{L}_{dir}$**: 세 가지 벡터화 계획 제약 손실
- **$\mathcal{L}_{imi}$**: Imitation learning 손실. 전문가 주행 궤적을 모방하도록 학습

$$\mathcal{L}_{imi} = \frac{1}{T_f} \sum_{t=1}^{T_f} ||\hat{V}^t_{ego} - \tilde{V}^t_{ego}||_1$$

- **$\hat{V}^t_{ego}$**: 예측 궤적, **$\tilde{V}^t_{ego}$**: ground truth 전문가 궤적

---

## 4. Experiments

### 4.1 Open-loop Planning (nuScenes)

**요약**

nuScenes validation 데이터셋에서 VAD는 SOTA 성능을 달성했습니다.

| Method | L2 Avg (m) ↓ | Collision Avg (%) ↓ | FPS |
|--------|-------------|---------------------|-----|
| ST-P3 | 2.11 | 0.71 | 1.6 |
| UniAD | 1.03 | 0.31 | 1.8 |
| **VAD-Tiny** | **0.78** | **0.38** | **16.8** |
| **VAD-Base** | **0.72** | **0.22** | **4.5** |

**핵심 개념**

- **L2 (m)**: 예측 궤적과 실제 궤적의 평균 거리 오차. 낮을수록 좋음
- **Collision Rate (%)**: 다른 에이전트와의 충돌 비율. 낮을수록 안전
- **VAD-Base**: 평균 충돌률을 29.0% 감소, L2 오차도 크게 개선
- **VAD-Tiny**: UniAD 대비 9.3배 빠른 추론 속도 (16.8 FPS vs 1.8 FPS)

### 4.2 Closed-loop Simulation (CARLA)

**요약**

CARLA 시뮬레이터에서 VAD-Base는 비전 전용 E2E 방법 중 최고 성능을 달성했습니다.

| Method | Town05 Short DS↑ | Town05 Short RC↑ | Town05 Long DS↑ | Town05 Long RC↑ |
|--------|-----------------|-----------------|----------------|----------------|
| ST-P3 | 55.14 | 86.74 | 11.45 | 83.15 |
| **VAD-Base** | **64.29** | **87.26** | **30.31** | **75.20** |

- **DS (Driving Score)**: 경로 완성률과 안전 이벤트를 종합한 점수
- **RC (Route Completion)**: 목표 경로 완주율

### 4.3 Ablation Study

**요약**

설계 선택의 유효성을 검증하는 절제 실험 결과:

| ID | Agent Inter. | Map Inter. | Overstep. | Dir. | Col. | L2 Avg ↓ | Collision Avg ↓ |
|----|-------------|-----------|-----------|------|------|----------|----------------|
| 1 | ✓ | - | ✓ | ✓ | ✓ | 0.86 | 0.29 |
| 3 | ✓ | ✓ | ✓ | - | - | 0.76 | 0.28 |
| 7 (Full) | ✓ | ✓ | ✓ | ✓ | ✓ | **0.72** | **0.22** |

- 맵 상호작용 추가 시 충돌률 크게 감소
- 세 가지 제약 모두 사용할 때 최고 성능
- 래스터화 맵 대비 벡터화 맵의 충돌률이 현저히 낮음

### 4.4 Module Runtime (VAD-Tiny)

| Module | Latency (ms) | Proportion |
|--------|-------------|-----------|
| Backbone | 23.2 | 39.0% |
| BEV Encoder | 12.3 | 20.7% |
| Motion Module | 11.5 | 19.3% |
| Map Module | 9.1 | 15.3% |
| Planning Module | 3.4 | 5.7% |
| **Total** | **59.5** | **100%** |

- Planning 모듈이 전체의 단 5.7%만 차지 → 벡터화 표현으로 인한 경량 계획 가능
- Backbone + BEV Encoder가 가장 큰 비중 (59.7%)

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **VAD** | Vectorized Autonomous Driving. 장면 전체를 벡터 형태로 표현하는 E2E 자율주행 프레임워크 |
| **Vectorized Scene Representation** | 경계 벡터, 차선 벡터, 모션 벡터로 장면을 표현. 래스터 맵보다 경량화되고 인스턴스 단위 정보 포함 |
| **BEV (Bird's Eye View)** | 하늘에서 내려다보는 시점. 멀티카메라 이미지를 BEV 공간으로 변환하여 공간적 관계 파악 |
| **Ego Query** | 자차의 계획 정보를 담는 쿼리 벡터. 에이전트/맵 쿼리와 Attention을 통해 상호작용 |
| **Vectorized Planning Constraint** | 충돌 제약, 경계 이탈 제약, 차선 방향 제약 세 가지로 구성. 인스턴스 수준에서 안전 계획 강제 |
| **Imitation Learning** | 전문가(사람) 운전 궤적을 모방하도록 학습하는 방식 |
| **HD Map-free Planning** | 사전 구축된 HD 맵 없이 실시간으로 맵을 예측하면서 계획 수행 |
| **minFDE** | minimum Final Displacement Error. 다중 모달 예측 중 최종 위치 오차가 가장 작은 예측 선택 |

---

## 결론 및 시사점

VAD는 자율주행에서 **벡터화된 장면 표현**의 잠재력을 입증했습니다:

1. **성능**: nuScenes에서 이전 SOTA 대비 충돌률 29% 감소, L2 오차 대폭 감소
2. **효율성**: VAD-Tiny는 9.3배 빠른 추론 속도로 실시간 배포 가능성 제시
3. **안전성**: 인스턴스 수준의 벡터화 제약으로 더 정밀하고 해석 가능한 계획 달성
4. **확장성**: 교통 신호, 속도 제한 등 추가 정보를 벡터 쿼리로 통합 가능

**한계 및 미래 연구 방향**:
- 다중 모달 모션 예측의 계획 활용 방법 추가 연구 필요
- 차선 그래프, 도로 표지판, 교통 신호 등 추가 교통 정보 통합 탐색 필요
