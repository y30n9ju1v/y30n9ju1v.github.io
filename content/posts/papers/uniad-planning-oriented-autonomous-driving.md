---
title: "UniAD: Planning-oriented Autonomous Driving"
date: 2026-04-09
draft: false
categories: ["Papers"]
---

## 개요
- **저자**: Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, Lewei Lu, Xiaosong Jia, Qiang Liu, Jifeng Dai, Yu Qiao, Hongyang Li
- **발행년도**: 2023
- **주요 내용**: 자동운전 시스템의 모든 핵심 작업(인식, 예측, 계획)을 하나의 통합 네트워크에서 처리하는 Planning-Oriented 프레임워크인 UniAD를 제안합니다. Query 기반 설계를 통해 모든 모듈을 연결하고, 계획을 최종 목표로 삼아 각 작업을 최적화합니다.

## 목차
- 1. Introduction: 자동운전 시스템 설계의 현황과 문제점
- 2. Methodology: UniAD의 구조와 각 모듈의 설계
  - 2.1. Perception: TrackFormer와 MapFormer
  - 2.2. Prediction: MotionFormer
  - 2.3. Prediction: OccFormer (점유도 예측)
  - 2.4. Planning
  - 2.5. Learning (학습 전략)
- 3. Experiments: nuScenes 벤치마크에서의 성능 평가
- 4. Ablation Studies: 각 모듈의 효과 검증

## 1. Introduction: 현대 자동운전의 프레임워크 설계

**요약**

현대 자동운전 시스템은 세 가지 핵심 작업을 순차적으로 수행합니다: 인식(perception), 예측(prediction), 계획(planning). 기존 방식들은 크게 세 가지로 나뉩니다:

1. **독립적 모델(Standalone Models)**: 각 작업마다 별도의 모델 배포
   - 장점: 팀 간 R&D 독립성
   - 단점: 모듈 간 정보 손실, 오류 누적, 기능 불일치

2. **다중작업 학습(Multi-Task Learning)**: 공유 백본에 작업별 헤드 추가
   - 장점: 계산 효율성, 기능 공유
   - 단점: "음의 전이(negative transfer)" 가능성

3. **End-to-End 학습**: 모든 모듈을 하나의 네트워크로 통합
   - 문제: 계획을 최종 목표로 삼지 않으면 비효율적

**핵심 개념**

- **Planning-Oriented Philosophy**: 최종 목표인 계획(planning)을 기준으로 모든 선행 작업(인식, 예측)을 설계해야 함
- **Query-Based Design**: 모든 모듈을 연결하는 통합 인터페이스로 작동하여 누적 오류 감소
- **Task Coordination**: 단순한 모듈 스택이 아닌 효과적인 작업 조율이 필요

UniAD는 5가지 핵심 작업을 포함합니다:
- 추적(Tracking), 맵핑(Mapping) - 인식
- 동작 예측(Motion Forecasting), 점유도 예측(Occupancy Prediction) - 예측
- 계획(Planning)

## 2. Methodology: UniAD의 통합 아키텍처

**요약**

UniAD는 4개의 Transformer 디코더 기반 모듈과 1개의 계획기로 구성됩니다. 모든 모듈은 Query를 통해 연결되어 있으며, 각 모듈의 출력이 다음 모듈의 입력으로 사용됩니다.

**아키텍처 흐름**:
1. 멀티카메라 이미지 → 특성 추출
2. 조류 뷰(BEV) 인코더 (BEVFormer) → 통합 BEV 특성 생성
3. **TrackFormer** → 에이전트(물체) 감지 및 추적
4. **MapFormer** → 맵 요소(차선, 구분선) 의미 분할
5. **MotionFormer** → 모든 에이전트의 미래 궤적 예측
6. **OccFormer** → 에이전트 정체성이 보존된 미래 점유도 예측
7. **Planner** → 자차(ego-vehicle)의 안전한 궤적 계획

### 2.1. Perception: TrackFormer와 MapFormer

**TrackFormer: 검출 및 추적**

동시에 다중 물체 추적을 수행하는 Transformer 기반 모듈입니다.

**핵심 개념**:
- **Detection Queries**: 처음 감지되는 새로운 에이전트 감지
- **Track Queries**: 이전 프레임에서 감지된 에이전트 추적
- **Temporal Aggregation**: 이전 프레임의 쿼리와 자체 주의(self-attention)를 통해 시간 정보 집계

$$Q_A^{t} = \text{MHSA}(Q_A^{t}, Q_A^{t-1})$$

**수식 설명**:
- **$Q_A^{t}$**: 시간 $t$에서의 에이전트 쿼리
- **$Q_A^{t-1}$**: 이전 시간 단계의 에이전트 쿼리
- **$\text{MHSA}$**: 다중 헤드 자체 주의(Multi-Head Self-Attention) - 현재 쿼리가 이전 쿼리의 정보를 수집하여 시간적 연속성 유지

**MapFormer: 맵 의미 분할**

도로 요소를 희소하게 표현하는 Query 기반 모듈입니다.

**핵심 개념**:
- **Map Queries**: 차선, 구분선, 횡단보도 등 도로 요소를 나타냄
- **Panoptic Segmentation**: 도로 요소를 "things"(물체)와 "stuff"(배경)으로 분류
- 다층 구조로 각 계층의 출력이 감독됨

### 2.2. Prediction: MotionFormer

**요약**

모든 에이전트의 상호작용을 모델링하여 미래 궤적(multimodal trajectories)을 예측합니다.

**핵심 개념**:

MotionFormer는 세 가지 유형의 상호작용을 모델링합니다:

1. **Agent-Agent Interaction** (에이전트 간 상호작용):

$$Q_{a} = \text{MHCA}(\text{MHSA}(Q), Q_A)$$

- **$Q$**: 대상 에이전트의 쿼리
- **$Q_A$**: 모든 에이전트의 쿼리 집합
- **$\text{MHSA}(Q)$**: 먼저 대상 에이전트 자신의 정보를 처리
- **$\text{MHCA}$**: 다른 에이전트들과의 상호작용을 모델링하여 사회적 기하학(social geometry) 파악

2. **Agent-Map Interaction** (에이전트-맵 상호작용):

$$Q_{m} = \text{MHCA}(\text{MHSA}(Q), Q_M)$$

- 도로 구조(차선, 경계)가 미래 궤적에 미치는 영향 모델링

3. **Agent-Goal Point Interaction** (에이전트-목표점 상호작용):

$$Q_{g} = \text{DeformAttn}(Q, \hat{x}_{T}^{l-1}, B)$$

**수식 설명**:
- **$\hat{x}_{T}^{l-1}$**: 이전 계층에서 예측한 목표점(시간 T에서의 위치)
- **$B$**: BEV 특성
- **$\text{DeformAttn}$**: 변형 가능한 주의(Deformable Attention) - 목표점 주변의 공간 특성에만 희소하게 주의를 집중하여 계산 효율성 향상

**Motion Query 설계**:

$$Q_{\text{pos}} = \text{MLP}(\text{PE}(I^s)) + \text{MLP}(\text{PE}(I^a)) + \text{MLP}(\text{PE}(\hat{x}_0)) + \text{MLP}(\text{PE}(\hat{x}_T^{l-1}))$$

**수식 설명**:
- **$I^s$**: 씬 레벨 앵커 - 전체 운전 장면에서 일반적인 움직임 패턴(e.g., 직진, 좌회전)
- **$I^a$**: 에이전트 레벨 앵커 - 각 에이전트가 취할 수 있는 지역적 의도
- **$\hat{x}_0$**: 현재 에이전트의 위치 - 예측의 시작점
- **$\hat{x}_T^{l-1}$**: 이전 계층의 예측 목표점 - 계층별로 점진적으로 정제(coarse-to-fine)
- 이들을 더함으로써 전역적 패턴, 지역적 의도, 현재 상태, 미래 목표를 모두 반영

**Non-Linear Optimization (비선형 최적화)**:

실제로는 정확한 감지 위치를 모를 수 있으므로, 학습 중에만 예측 목표 궤적을 최적화합니다:

$$\tilde{x}^* = \arg \min_{\mathbf{x}} c(\mathbf{x}, \tilde{\mathbf{x}})$$

$$c(\mathbf{x}, \tilde{\mathbf{x}}) = \lambda_{\text{xy}} \|\mathbf{x} - \tilde{\mathbf{x}}\|_2 + \lambda_{\text{goal}} \|\mathbf{x}_T - \tilde{\mathbf{x}}_T\|_2 + \sum_{\phi \in \Phi} \phi(\mathbf{x})$$

**수식 설명**:
- **$\tilde{\mathbf{x}}$**: 원본 그라운드 트루스 궤적 (정확한 감지 위치에서 시작)
- **$\mathbf{x}$**: Multiple-shooting 방법으로 생성된 물리적으로 가능한 궤적
- **$c(\cdot)$**: 비용 함수로 궤적을 부드럽게 만듦
- **$\lambda_{\text{xy}}$**: 전체 궤적의 부드러움 정도 제어
- **$\lambda_{\text{goal}}$**: 목표점 도달의 정확성 제어
- **$\Phi$**: 5가지 물리 제약 - 저크(jerk), 곡률(curvature), 곡률률(curvature rate), 가속도(acceleration), 횡방향 가속도(lateral acceleration)

예를 들어, 자동차가 급격하게 꺾인 궤적을 예측하면 높은 곡률 항이 이를 벌칙으로 주어 자연스러운 궤적으로 조정합니다.

### 2.3. Prediction: OccFormer

**요약**

미래의 점유도 맵(occupancy grid)을 예측하되, 각 에이전트의 정체성을 보존합니다. 기존 방식과 달리 복잡한 후처리 없이 에이전트별 점유도를 직접 생성합니다.

**핵심 개념**:

**Pixel-Agent Interaction** (픽셀-에이전트 상호작용):

$$D_{\text{ds}}^t = \text{MHCA}(\text{MHSA}(F_{\text{ds}}^t), G^t, \text{attn\_mask} = O_m^t)$$

**수식 설명**:
- **$D_{\text{ds}}^t$**: 시간 $t$에서 축소된(1/8) 밀도 특성이 업데이트된 것
- **$F_{\text{ds}}^t$**: 현재 시간 단계의 축소된 밀도 특성
- **$\text{MHSA}(F_{\text{ds}}^t)$**: 먼저 원거리 그리드 간의 관계를 모델링하여 씬 이해도 향상
- **$G^t$**: 시간 $t$의 에이전트 특성(에이전트의 현재 상태 정보)
- **$\text{MHCA}$**: 에이전트 특성과 각 픽셀의 관계를 모델링
- **$\text{attn\_mask} = O_m^t$**: 각 픽셀이 시간 $t$에서 그것을 차지하고 있는 에이전트만 살펴보도록 제한 - 오버헤드 감소 및 정확성 향상

**Agent Feature Fusion** (에이전트 특성 융합):

$$G^t = \text{MLP}_t([Q_A, P_A, Q_X])$$

**수식 설명**:
- **$Q_A$**: TrackFormer의 추적 쿼리 (에이전트의 현재 위치, 크기 등)
- **$P_A$**: 에이전트의 현재 위치 임베딩
- **$Q_X$**: MotionFormer의 동작 쿼리에서 양식 차원으로 최대 풀링한 것 (에이전트의 미래 의도)
- 이들을 시간별 특화된 MLP로 융합하여 동적 정보(현재) + 공간 정보(위치) + 의도(미래)를 통합

**Instance-Level Occupancy** (인스턴스 수준 점유도):

$$\hat{O}_A^t = U^t \cdot F_{\text{dec}}^t$$

**수식 설명**:
- **$U^t \in \mathbb{R}^{N_a \times C}$**: 에이전트 수($N_a$) × 채널(C) 크기의 점유도 특성 (각 에이전트마다 하나의 특성 벡터)
- **$F_{\text{dec}}^t \in \mathbb{R}^{C \times H \times W}$**: 채널 × 높이 × 너비의 씬 특성
- 행렬 곱셈으로 각 에이전트의 점유도 맵을 직접 생성
- 예: 에이전트 1이 점유도 벡터 $[0.1, 0.3, 0.8]$을 가지면, 씬 특성의 해당 채널들을 가중치로 곱하여 그 에이전트의 점유도 맵 생성

### 2.4. Planning

**요약**

자차(ego-vehicle)의 미래 경로를 계획합니다. 네비게이션 신호(좌회전, 우회전, 직진)를 활용하고, 예측된 점유도를 바탕으로 충돌을 피합니다.

**핵심 개념**:

**Command Embedding** (명령 임베딩):
- 네비게이션 신호(좌회전, 우회전, 직진)를 학습 가능한 벡터로 변환
- 자차 쿼리와 결합하여 "계획 쿼리(plan query)" 형성

**Planning Optimization** (계획 최적화):

$$\tau^* = \arg \min_{\tau} f(\tau, \hat{\tau}, \hat{O})$$

$$f(\tau, \hat{\tau}, \hat{O}) = \lambda_{\text{coord}} \|\tau - \hat{\tau}\|_2 + \lambda_{\text{obs}} \sum_{t} \mathcal{D}(\tau_t, \hat{O}^t)$$

$$\mathcal{D}(\tau_t, \hat{O}^t) = \sum_{(x,y) \in \mathcal{S}} \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{\|\tau_t - (x,y)\|_2^2}{2\sigma^2}\right)$$

**수식 설명**:
- **$\tau^*$**: 최적화된 최종 계획 궤적
- **$\hat{\tau}$**: 신경망이 예측한 원본 궤적
- **$\hat{O}^t$**: 시간 $t$의 예측된 점유도 맵
- **$\lambda_{\text{coord}}$**: 원본 예측과의 일치도 중요도 제어
- **$\lambda_{\text{obs}}$**: 충돌 회피의 중요도 제어
- **좌측 첫 번째 항** ($\|\tau - \hat{\tau}\|_2$): L2 거리로 원본 예측 궤적에 근접하도록 유지
- **$\mathcal{D}(\tau_t, \hat{O}^t)$**: 가우시안 거리 함수로 점유도가 높은 영역에서 멀어지도록 함
  - 예: 시간 $t$에서 궤적이 점유도 높은 영역에 가까워지면 높은 비용 발생
  - $\sigma$: 가우시안의 표준편차로 충돌 회피 범위 제어 (크면 멀리서부터 회피, 작으면 가까이서 회피)

### 2.5. Learning (학습)

**요약**

두 단계의 학습 과정:
1. **초기 학습(6 에포크)**: TrackFormer와 MapFormer(인식 모듈)만 훈련
2. **End-to-End 학습(20 에포크)**: 모든 모듈을 함께 훈련하여 상호작용 최적화

이 두 단계 접근법이 경험적으로 더 안정적임이 입증됨.

**Shared Matching** (공유 매칭):
- DETR의 이분 매칭(bipartite matching) 알고리즘 사용
- TrackFormer의 매칭 결과를 MotionFormer와 OccFormer에서 재사용하여 일관성 유지

## 3. Experiments & Results

**요약**

nuScenes 벤치마크에서 UniAD의 성능을 평가합니다.

**핵심 성능 지표**:
- **Tracking**: AMOTA (Average Multi-Object Tracking Accuracy), AMOTP (Average Multi-Object Tracking Precision)
- **Motion Prediction**: ADE (Average Displacement Error), FDE (Final Displacement Error)
- **Occupancy Prediction**: 정확도(Accuracy)
- **Planning**: L2 거리, 충돌 회피율

**실험 결과의 주요 발견**:

1. 모든 모듈(TrackFormer, MapFormer, MotionFormer, OccFormer)이 함께 작동할 때 최고 성능 달성
2. Planning-Oriented 설계가 기존 방식(독립적 모델, 단순 MTL)보다 우수
3. Query 기반 설계의 효과: 에이전트 간 상호작용 정확히 모델링
4. 각 모듈 제거 시 성능 감소로 모든 모듈의 필요성 입증

## 핵심 개념 정리

### 1. **Query-Based Interface (쿼리 기반 인터페이스)**

모든 모듈을 연결하는 통합 인터페이스 역할을 합니다. 전통적인 바운딩 박스 표현과 달리, Query는:
- 더 큰 수용 영역(receptive field)을 가져 상류 예측의 누적 오류 완화
- 다양한 상호작용 모델링에 유연성 제공
- 정보 손실 감소로 정확도 향상

### 2. **Multi-Agent Interaction (다중 에이전트 상호작용)**

운전 장면의 모든 에이전트(자차, 다른 차량, 보행자 등)의 행동은 상호 의존적입니다. UniAD는 세 가지 상호작용을 모델링합니다:
- **에이전트-에이전트**: 차량 간 상호작용 (예: 한 차량의 급정거가 뒤차에 영향)
- **에이전트-맵**: 도로 구조의 영향 (예: 차선이 진행 방향 제한)
- **에이전트-목표**: 의도 기반 행동 예측

### 3. **BEV (Bird's-Eye-View) Representation**

여러 카메라의 관점을 조류 뷰로 변환하여:
- 모든 에이전트의 상대적 위치 정확히 파악
- 위치 기반 충돌 감지 용이
- 자연스러운 점유도 맵 생성

### 4. **Coarse-to-Fine Refinement (단계적 정제)**

MotionFormer의 여러 계층을 통과하면서 예측이 점진적으로 정제됩니다:
- 초기 계층: 거친 방향 결정
- 중간 계층: 다른 에이전트와의 상호작용 고려
- 최종 계층: 미세한 경로 조정

### 5. **Non-Linear Physics Constraints (비선형 물리 제약)**

예측된 궤적이 실제 자동차의 물리 법칙을 따르도록 제약:
- 무한 곡률 불가능 (자동차는 원을 그릴 수 없음)
- 무한 가속도 불가능 (순간적인 속도 변화 불가)
- 저크 제약으로 부드러운 운동 보장

### 6. **Planning-Oriented Philosophy (계획-지향적 철학)**

전체 시스템이 최종 목표(안전한 경로 계획)를 달성하기 위해 설계됨:
- 인식(perception) 오류: 이후 예측의 오류 증폭 가능 → 점유도로 완화
- 예측(prediction) 오류: 계획의 충돌 회피로 최소화
- 결과: 각 모듈의 완벽성보다 **전체 시스템의 안전성 극대화**

## 결론 및 시사점

**주요 기여**

1. **새로운 프레임워크 철학**: Planning-Oriented 설계가 자동운전 시스템의 효과적인 통합 방식임을 입증
2. **포괄적 통합 모듈**: 인식, 예측, 계획을 모두 포함하는 첫 번째 포괄적 시스템
3. **Query 기반 설계의 효과**: 누적 오류 감소 및 다양한 상호작용 모델링 가능

**실무적 시사점**

1. **오류 누적 문제 해결**: 단순 모듈 스택 대신 에러를 보상하는 설계 필요
2. **전체 최적화**: 각 모듈의 개별 성능보다 시스템 전체의 안전성 우선
3. **유연한 표현**: Query 기반 설계로 다양한 상황 대응 가능
4. **실시간 최적화**: 추론 중 경로 최적화로 예측 오류 보정

**향후 연구 방향**

1. 더 다양한 시나리오(악천후, 야간 주행 등)에서의 성능 검증
2. 계산 효율성 개선으로 실제 자동차 탑재 가능성 향상
3. 다른 BEV 인코더와의 결합으로 확장성 검증
4. 실제 폐쇄 루프(closed-loop) 운전 테스트를 통한 안전성 확보
