---
title: "CARLA: An Open Urban Driving Simulator"
date: 2026-04-10T08:30:00+09:00
draft: false
categories: ["Papers"]
tags: ["autonomous-driving", "simulation", "deep-learning", "sensor-simulation"]
---

## 개요
- **저자**: Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio López, Vladlen Koltun
- **기관**: Intel Labs, Toyota Research Institute, Computer Vision Center Barcelona
- **발행년도**: 2017 (CoRL 2017)
- **주요 내용**: 자율주행 연구를 위한 오픈소스 시뮬레이터 CARLA(Car Learning to Act) 소개. 도시 환경에서 모듈식 파이프라인, 모방 학습, 강화 학습 등 세 가지 자율주행 방식 비교 평가

## 목차
1. 소개 (Introduction)
2. 관련 연구 (Related Work)
3. 방법론 (Method)
   - 벡터화된 장면 학습 (Vectorized Scene Learning)
   - 상호작용을 통한 계획 (Planning via Interaction)
   - 벡터화된 계획 제약 (Vectorized Planning Constraint)
   - 엔드-투-엔드 학습 (End-to-End Learning)
4. 실험 결과 (Experiments)
5. 결론 (Conclusion)

---

## 1. 소개 (Introduction)

### 배경
자율주행은 주변 환경을 정확히 이해하면서도 효율적이어야 합니다. 기존의 모듈식 자율주행 시스템은 다음과 같은 문제점을 가지고 있습니다:

- **모듈 간 정보 손실**: 인식(perception)과 계획(planning) 모듈이 분리되어 있어, 원본 센서 데이터의 정밀한 정보가 손실됨
- **계획 모듈의 제약**: 계획 모듈은 사전 처리된 인식 결과에만 접근 가능하므로, 인식 오류가 계획에 영향을 미침
- **해석 가능성 부족**: 최근의 엔드-투-엔드 방법은 해석 가능성이 떨어지고 최적화가 어려움

### VAD의 핵심 아이디어
**VAD(Vectorized Autonomous Driving)**는 완전한 벡터화된 패러다임을 제안합니다:

- **벡터 기반 장면 표현**: 경계(boundary), 차선(lane), 운동 정보(motion)를 모두 벡터로 표현
- **명시적 제약 조건**: 벡터화된 계획 제약(planning constraints)으로 안전성 보장
- **효율성 극대화**: 래스터 표현과 후처리 단계를 제거하여 계산 효율 향상

---

## 2. 관련 연구 (Related Work)

### 인식(Perception) 분야
자율주행의 기본은 정확한 장면 인식입니다:

- **3D 객체 탐지**: DETR3D, BEVFormer 등이 조류 눈 뷰(BEV: Bird's Eye View) 표현에서 객체를 탐지
- **조류 눈 뷰의 확산**: BEV 표현이 공간 정보를 효과적으로 인코딩하면서 자율주행 분야에서 표준화
- **벡터 기반 지도**: VectorMapNet, MapTR 등이 지도 요소를 벡터로 예측

### 운동 예측(Motion Prediction)
차량과 보행자의 미래 궤적 예측:

- **궤적 기반**: 과거 궤적과 HD 지도를 입력으로 미래 궤적 예측
- **CNN/RNN 기반**: 조류 눈 뷰 이미지와 CNN을 활용한 예측
- **최근 발전**: Transformer 기반 방법들이 다중 모달(multi-modal) 궤적 예측 가능

### 계획(Planning) 분야
자율주행 의사결정의 핵심:

- **전통적 계획**: 학습 기반이 아닌 규칙 기반의 비용 지도(cost map) 활용
- **학습 기반 계획**: 강화학습을 통한 계획 학습이 주목받고 있음
- **해석 가능성의 중요성**: 안전성을 위해 계획 과정의 투명성 필요

---

## 3. 방법론 (Method)

### 3.1 개요 (Overview)

VAD는 4개 단계로 구성됩니다:

1. **백본(Backbone)**: 다중 프레임, 다중 뷰 이미지에서 BEV 특성 추출
2. **벡터화된 장면 학습**: BEV 특성으로부터 벡터화된 지도와 에이전트 운동 벡터 학습
3. **계획 추론(Planning Inference)**: 에이전트와 지도 쿼리를 통한 상호작용으로 계획 궤적 생성
4. **계획 학습(Planning Training)**: 벡터화된 제약 조건으로 계획 궤적 정규화

### 3.2 벡터화된 장면 학습 (Vectorized Scene Learning)

#### 벡터화된 지도(Vectorized Map)
지도의 기하학적 정보를 명시적으로 표현합니다:

**주요 요소**:
- **차선 분할선(Lane Divider)**: 도로 방향 정보 제공
- **도로 경계(Road Boundary)**: 주행 가능 영역 정의
- **보행자 횡단보도(Pedestrian Crossing)**: 보행자 관련 정보

**수식**:
$$V_{m} \in \mathbb{R}^{N_m \times d_m}$$

**수식 설명**:
- **$V_m$**: 예측된 지도 벡터들의 집합
- **$N_m$**: 지도 벡터의 개수 (예: 100개)
- **$d_m$**: 각 벡터의 차원 (예: 128차원)
- **의미**: 지도의 모든 기하학적 요소를 고정 크기의 벡터 집합으로 인코딩

#### 벡터화된 에이전트 운동(Vectorized Agent Motion)
다른 차량과 보행자의 미래 움직임을 다중 모달로 예측합니다:

**수식**:
$$V_a \in \mathbb{R}^{N_a \times T \times 2}$$

**수식 설명**:
- **$V_a$**: 모든 에이전트의 예측된 궤적 모음
- **$N_a$**: 장면에 있는 에이전트의 개수
- **$T$**: 예측할 미래 시간 스텝 개수 (예: 6초를 0.5초 간격으로 12스텝)
- **$2$**: 각 스텝에서 x, y 좌표 (평면 운동)
- **$\mathbb{R}^{...}$**: 실수 공간을 의미
- **의미**: 각 에이전트가 미래에 어디로 이동할지를 예측

**학습 방식**:
- BEV 특성 맵에서 **에이전트 쿼리(Agent Query)** $Q_a$를 무작위로 초기화
- Transformer 디코더를 통해 BEV 특성과 상호작용하여 에이전트 속성(위치, 방향, 크기 등) 학습
- MLP 기반 디코더로 에이전트 속성을 운동 벡터로 변환

### 3.3 상호작용을 통한 계획 (Planning via Interaction)

#### Ego-에이전트 상호작용
자신의 차량(ego vehicle)과 다른 에이전트들 간의 관계를 학습합니다:

**수식**:
$$Q^t_{ego} = \text{TransformerDecoder}(q, k, v, q_{pos}, k_{pos})$$

$$q = Q_{ego}, \quad k = v = Q_a,$$

$$q_{pos} = \text{PE}_1(q_{ego}), \quad k_{pos} = \text{PE}_1(p_a)$$

**수식 설명**:
- **TransformerDecoder**: 자기 자신과 다른 차량들의 관계를 학습하는 신경망
- **$q$ (Query)**: Ego 차량의 현재 위치/상태
- **$k, v$ (Key, Value)**: 주변 에이전트들의 정보
- **$\text{PE}_1$**: Positional Encoding - 절대 위치를 신경망이 이해할 수 있도록 변환
- **의미**: "Ego 차량 주변에 어떤 다른 차량이 있고, 어떻게 상호작용하는가?"를 학습

#### Ego-지도 상호작용
자신의 차량과 도로 지도 간의 관계를 학습합니다:

**수식**:
$$Q^t_{ego} = \text{MLP}(Q^t_{ego}, Q^t_m, \text{MLP}(\text{PE}_2(p_{ego})))$$

**수식 설명**:
- **MLP (Multi-Layer Perceptron)**: 간단한 신경망 (fully connected layers)
- **$\text{PE}_2$**: Ego 차량의 상대 위치를 지도 쿼리와 융합
- **의미**: "도로 지도에서 ego 차량이 어디에 위치하고, 주변 도로 정보는 무엇인가?"를 결합

#### 계획 헤드(Planning Head)
학습된 정보로부터 최종 계획 명령을 생성합니다:

**수식**:
$$V_{ego} = \text{PlanHead}(f = f_{ego}, \text{cmd} = c)$$

$$f_{ego} = [Q_{ego}, Q'_{ego}, s_{ego}]$$

**수식 설명**:
- **PlanHead**: Ego 차량의 특성 및 네비게이션 명령으로부터 궤적을 생성하는 신경망
- **$f_{ego}$**: Ego 차량의 모든 특성을 연결(concatenate)한 것
  - $Q_{ego}$: Ego 차량 쿼리
  - $Q'_{ego}$: 에이전트와의 상호작용으로 업데이트된 쿼리
  - $s_{ego}$: Ego 상태 (속도, 가속도 등)
- **$c$**: 네비게이션 명령 (turn left, turn right, go straight)
- **$V_{ego}$**: 최종 계획된 궤적 (다음 6초간의 경로)
- **의미**: "현재 상황에서 네비게이션 명령에 따라 어떤 경로를 그려야 할까?"

### 3.4 벡터화된 계획 제약 (Vectorized Planning Constraint)

안전한 주행을 보장하기 위해 3가지 명시적 제약을 도입합니다:

#### 1️⃣ Ego-에이전트 충돌 회피 제약(Ego-Agent Collision Constraint)
다른 차량과의 충돌을 피합니다:

**수식**:
$$\mathcal{L}_{col} = \frac{1}{T_f} \sum_{t=1}^{T_f} \sum_{i \in \{X, Y\}} L^{it}_{col}, \quad i \in \{X, Y\}$$

$$L^{it}_{col} = \begin{cases} \delta_i - d^it_u, & \text{if } d^it_u < \delta_i \\ 0, & \text{if } d^it_u \geq \delta_i \end{cases}$$

**수식 설명**:
- **$L_{col}$**: 충돌 손실(loss) - 충돌의 위험도를 수치화
- **$T_f$**: 미래 예측 시간 스텝 개수 (예: 12)
- **$d^it_u$**: 시간 $t$에서 Ego 차량과 가장 가까운 다른 차량 $u$ 사이의 거리 (X 또는 Y 방향)
- **$\delta_i$**: 안전 거리 임계값 (X 방향 3m, Y 방향 1m)
- **의미**: 계획된 궤적이 다른 차량과의 거리 기준을 만족하도록 강제

**직관**: 
- 만약 Ego 차량이 옆 차량에 너무 가까워지려 하면, 그 손실 값이 커짐
- 신경망은 이 손실을 최소화하려고 궤적을 조정하여 안전 거리 유지

#### 2️⃣ Ego-경계 오버스테핑 제약(Ego-Boundary Overstepping Constraint)
도로 경계를 넘어가지 않습니다:

**수식**:
$$\mathcal{L}_{bd} = \frac{1}{T_f} \sum^{T_f}_{t=1} L^t_{bd}$$

$$L^t_{bd} = \begin{cases} \delta_{bd} - d^t_{bd}, & \text{if } d^t_{bd} < \delta_{bd} \\ 0, & \text{if } d^t_{bd} \geq \delta_{bd} \end{cases}$$

**수식 설명**:
- **$d^t_{bd}$**: 시간 $t$에서 계획 궤적과 도로 경계선 사이의 거리
- **$\delta_{bd}$**: 경계와의 최소 안전 거리 (예: 0.5m)
- **의미**: 계획된 경로가 도로 경계 안에 있도록 강제

#### 3️⃣ Ego-차선 방향 제약(Ego-Lane Directional Constraint)
주행 방향이 차선 방향과 일치하도록 합니다:

**수식**:
$$\mathcal{L}_{dir} = \frac{1}{T_f} \sum^{T_f}_{t=1} \mathcal{L}_{ang}(\vec{v}^t_m, \vec{v}^t_{ego})$$

$$\mathcal{L}_{ang}(\vec{v}_m, \vec{v}') = \arccos\left(\frac{\vec{v}_m \cdot \vec{v}'}{|\vec{v}_m| |\vec{v}'|}\right)$$

**수식 설명**:
- **$\vec{v}^t_m$**: 시간 $t$에서 가장 가까운 차선의 방향 벡터
- **$\vec{v}^t_{ego}$**: 시간 $t$에서 Ego 차량의 주행 방향 벡터
- **$\arccos$**: 두 벡터 사이의 각도를 계산 (코사인 역함수)
- **의미**: 주행 방향이 차선과 일치할수록 손실이 작아짐

**직관**:
- 차선이 북쪽으로 향하고 Ego 차량이 남쪽으로 향하면 각도가 크므로 손실 값이 큼
- 신경망은 이 손실을 최소화하여 차선 방향과 일치하는 방향으로 주행

### 3.5 엔드-투-엔드 학습 (End-to-End Learning)

#### 벡터화된 장면 학습 손실(Vectorized Scene Learning Loss)
지도와 에이전트 운동 예측을 감독합니다:

**지도 학습 손실**:
- **Manhattan 거리**: 예측된 지도 포인트와 실제 포인트 사이의 거리 계산
- **분류 손실**: 각 지도 벡터의 클래스 분류

**운동 예측 손실**:
$$\mathcal{L}_{mot} = \text{l1 loss (regression)} + \text{focal loss (classification)}$$

- **회귀 손실(l1 loss)**: 궤적의 위치 오차
- **초점 손실(focal loss)**: 에이전트 클래스 분류

#### 벡터화된 계획 제약 손실(Vectorized Constraint Loss)
앞서 정의한 3가지 제약을 손실함수로 표현:

$$\mathcal{L}_{con} = \omega_1 \mathcal{L}_{col} + \omega_2 \mathcal{L}_{bd} + \omega_3 \mathcal{L}_{dir}$$

**수식 설명**:
- **$\omega_1, \omega_2, \omega_3$**: 각 제약의 중요도를 조절하는 가중치
- **의미**: 세 제약을 균형있게 조합하여 안전한 계획 생성

#### 모방 학습 손실(Imitation Learning Loss)
전문가 운전 데이터로부터 학습합니다:

$$\mathcal{L}_{imi} = \frac{1}{T_f} \sum_{t=1}^{T_f} ||V^t_{ego} - V^{t*}_{ego}||_1$$

**수식 설명**:
- **$V^t_{ego}$**: 신경망이 예측한 Ego 궤적
- **$V^{t*}_{ego}$**: 전문 운전자의 실제 궤적 (ground truth)
- **$||..||_1$**: L1 거리 (절댓값의 합)
- **의미**: 신경망의 예측이 전문가의 운전과 얼마나 비슷한지 측정

#### 전체 손실 함수(Overall Loss)
모든 손실을 결합합니다:

$$\mathcal{L} = \omega_1 \mathcal{L}_{map} + \omega_2 \mathcal{L}_{mot} + \omega_3 \mathcal{L}_{col} + \omega_4 \mathcal{L}_{bd} + \omega_5 \mathcal{L}_{dir} + \omega_6 \mathcal{L}_{imi}$$

**수식 설명**:
- 6가지 손실을 가중합으로 결합
- 각 항의 가중치를 조절하여 모든 목표 간의 균형 유지
- **의미**: 장면 이해, 계획, 안전성을 모두 고려하는 종합적 학습

---

## 4. 실험 결과 (Experiments)

### 4.1 데이터셋 및 평가 지표

**데이터셋**: nuScenes
- 1,000개의 주행 장면
- 각 장면 약 20초
- 6개 카메라로 360도 촬영
- 주석: 2Hz 샘플링

**평가 지표**:
- **L2 Displacement Error (L2 m ↓)**: 예측된 경로와 실제 경로 간의 거리 오차 (낮을수록 좋음)
- **Collision Rate (%) ↓**: 충돌 사건의 비율 (낮을수록 좋음)
- **Latency (ms)**: 계획 생성 시간 (낮을수록 좋음)
- **FPS**: 초당 처리 프레임 (높을수록 좋음)

### 4.2 주요 결과

#### 개방 루프(Open-loop) 계획 성능

| 방법 | L2 (m) ↓ | Collision (%) ↓ | Latency (ms) |
|------|----------|-----------------|--------------|
| NMP† | 2.31 | 1.92 | - |
| SA-NMP† | 2.05 | 1.59 | - |
| FF† | 2.54 | 1.07 | - |
| EOI† | 2.78 | 0.88 | - |
| ST-P3† | 2.90 | 1.27 | 628.3 |
| UniAD | 1.65 | 0.71 | 555.6 |
| VAD-Tiny | **1.12** | **0.58** | **59.5** |
| VAD-Base | **1.05** | **0.41** | **224.3** |

**핵심 결과**:
- **VAD-Base**: L2 오차 30% 감소, 충돌율 29% 감소
- **VAD-Tiny**: UniAD 대비 **2.5배 빠른** 추론 속도 (9.3 FPS → 16.8 FPS)
- **우수한 균형**: 가장 빠르면서도 경쟁력 있는 정확도 유지

#### 폐쇄 루프(Closed-loop) 시뮬레이션

| 방법 | Town05 Short | Town05 Long |
|------|--------------|------------|
| CILRS | DS ↑ 7.47 | DS ↑ 3.68 |
| LBC | DS ↑ 30.97 | DS ↑ 7.05 |
| Transfuser | DS ↑ 54.52 | DS ↑ 33.15 |
| ST-P3 | DS ↑ 55.14 | DS ↑ 11.45 |
| VAD-Base | **DS ↑ 64.29** | **DS ↑ 30.31** |

**의미**:
- **DS (Driving Score)**: 주행 성공도 (높을수록 좋음)
- VAD가 모든 벤치마크에서 최고 성능 달성

### 4.3 어블레이션 연구 (Ablation Study)

#### 설계 선택의 효과성

| ID | Agent Inter. | Map Inter. | Overstep. | Dir. | Col. | L2 (m) | Collision (%) |
|----|--------------|-----------|----------|------|------|--------|---------------|
| 1 | ✓ | - | ✓ | ✓ | ✓ | 0.52 | 0.29 |
| 2 | - | ✓ | ✓ | ✓ | ✓ | 0.49 | 0.26 |
| 3 | ✓ | ✓ | - | ✓ | ✓ | 0.43 | 0.28 |
| 4 | ✓ | ✓ | ✓ | - | - | 0.46 | 0.24 |
| 5 | ✓ | ✓ | ✓ | ✓ | - | 0.42 | 0.25 |
| 6 | ✓ | ✓ | - | - | ✓ | 0.44 | 0.26 |
| 7 | ✓ | ✓ | ✓ | ✓ | ✓ | **0.41** | **0.22** |

**발견**:
- **지도 쿼리의 중요성**: 지도 상호작용 없을 때 L2 오차 증가
- **에이전트 상호작용**: 충돌율에 큰 영향
- **모든 제약의 기여**: 전체 제약을 함께 사용할 때만 최적 성능

#### 지도 표현 비교 (래스터 vs 벡터)

| 표현 방식 | Vectorized Map | L2 (m) | Collision (%) |
|---------|---|--------|---------------|
| Rasterized | ✓ | - | 0.43 | 0.39 |
| Vectorized | - | 0.44 | 0.26 |
| Vectorized | ✓ | **0.41** | **0.22** |

**의미**:
- **래스터 방식**: 높은 계산 비용으로 성능 저하
- **벡터 방식**: 계산 효율성과 성능 동시 달성

### 4.4 모듈별 런타임 분석

VAD-Tiny 기준 (NVIDIA GeForce RTX 3090):

| 모듈 | 지연시간 (ms) | 비율 |
|------|------------|------|
| Backbone | 23.2 | 39.0% |
| BEV Encoder | 12.3 | 20.7% |
| Motion Module | 11.5 | 19.3% |
| Map Module | 9.1 | 15.3% |
| Planning Module | 3.4 | 5.7% |
| **합계** | **59.5** | 100% |

**분석**:
- 백본 네트워크가 전체 시간의 40% 차지
- **계획 모듈은 단 5.7%**: 벡터화된 계획이 매우 효율적
- 전체 시스템이 16.8 FPS 달성 가능 (실시간 요구사항 충족)

---

## 5. 핵심 개념 정리

### 벡터 표현의 장점

| 특성 | 래스터 | 벡터 |
|-----|-------|------|
| **표현** | 고해상도 2D 그리드 | 기하학적 포인트/선 |
| **정밀도** | 그리드 해상도 제약 | 무제한 정밀도 |
| **계산량** | 높음 | 낮음 |
| **해석 가능성** | 낮음 | 높음 |
| **점진적 처리** | 어려움 | 용이 |

### Transformer 기반 상호작용

```
[Agent Queries]  ──┐
                   ├→ TransformerDecoder ──→ [Updated Queries]
[Map Features] ───┘

의미: 에이전트 정보와 지도 정보를 신경망이 상호작용하게 하여 
      더 나은 표현 학습
```

### 명시적 제약의 역할

**제약 없음**: 신경망이 임의로 궤적 생성 → 불안전
**제약 있음**: 안전 조건을 명시적으로 강제 → 신뢰할 수 있는 계획

---

## 6. 결론 및 시사점

### VAD의 기여

1. **새로운 패러다임**: 자율주행을 위한 완전한 벡터화된 표현 제안
2. **성능과 효율성**: 기존 방법 대비 더 정확하면서도 훨씬 빠름
3. **해석 가능성**: 벡터 표현과 명시적 제약으로 의사결정 과정 투명성 확보
4. **실용성**: 실시간 요구사항을 만족하는 추론 속도

### 실무적 시사점

- **산업 배포**: 계산 효율성이 자율주행 실제 배포의 핵심 요소
- **안전성 보장**: 신경망 기반 학습 + 명시적 제약의 조합이 신뢰성 확보
- **모듈식 설계의 재평가**: 완전 엔드-투-엔드보다 벡터 기반 구조화된 표현이 더 효과적
- **향후 방향**: 벡터 기반 방식이 자율주행의 표준 패러다임이 될 가능성

### 주요 성과 요약

| 지표 | 개선도 |
|-----|-------|
| 경로 오차 | 30% ↓ |
| 충돌율 | 29% ↓ |
| 추론 속도 | 2.5배 ↑ |
| 폐쇄 루프 성능 | 최고 달성 |

---

## 참고 자료

- **GitHub**: https://github.com/hustvl/VAD
- **arXiv**: 2303.12077
- **Conference**: ICCV 2023

## 관련 논문 및 기술

- **BEVFormer**: 조류 눈 뷰 기반 다중 작업 학습
- **MapTR**: 벡터 기반 지도 요소 예측
- **UniAD**: 통합 자율주행 프레임워크
- **Transformer**: 시퀀스 기반 상호작용 모델링
- **CARLA**: 자율주행 시뮬레이션 환경
