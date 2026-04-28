---
title: "UniAD: Planning-oriented Autonomous Driving"
date: 2026-04-10T09:00:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving"]
tags: ["Autonomous Driving", "End-to-End Planning", "Transformer", "Multi-Task Learning"]
---

## 개요

- **저자**: Yihan Hu, Jiazhi Yang, Li Chen, Keyu Li, Chonghao Sima, Xizhou Zhu, Siqi Chai, Senyao Du, Tianwei Lin, Wenhai Wang, Lewei Lu, Xiaosong Jia, Qiang Liu, Jifeng Dai, Yu Qiao, Hongyang Li
- **소속**: OpenDriveLab & OpenGVLab, Shanghai AI Laboratory; Wuhan University; SenseTime Research
- **발행년도**: 2023 (CVPR 2023)
- **주요 내용**: 자율주행의 궁극적 목표인 **계획(Planning)**을 중심으로 모든 인식/예측 모듈을 통합한 엔드-투-엔드 프레임워크 UniAD를 제안. 객체 추적, 지도 매핑, 모션 예측, 점유 예측, 경로 계획 5가지 태스크를 단일 네트워크에서 수행하며, query 기반 인터페이스로 모듈 간 정보를 전달

## 목차

- 1. Introduction: 왜 계획 지향 설계가 필요한가
- 2. Methodology: UniAD 전체 파이프라인
  - 2.1 Perception: TrackFormer & MapFormer
  - 2.2 Prediction: MotionFormer
  - 2.3 Prediction: OccFormer
  - 2.4 Planning: Planner
  - 2.5 Learning
- 3. Experiments: nuScenes 벤치마크 결과 및 ablation study
- 4. Conclusion

---

## 1. Introduction: 왜 계획 지향 설계가 필요한가

**요약**

현대 자율주행 시스템은 크게 두 가지 방식으로 설계됩니다. 하나는 각 태스크(검출, 추적, 예측 등)를 독립된 모델로 처리하는 **독립 모델(Standalone Models)** 방식이고, 다른 하나는 하나의 백본을 공유하되 태스크별 헤드를 붙이는 **멀티-태스크 학습(MTL, Multi-Task Learning)** 방식입니다.

저자들은 두 방식 모두 문제가 있다고 주장합니다:
- **독립 모델**: 모듈 간 정보 손실, 오류 누적, 최적화 목표 불일치
- **MTL**: 태스크 간 부정적 전이(negative transfer) 발생 위험

대신, 이 논문은 **"최종 목표(계획)를 중심으로 모든 태스크를 설계하고 연결해야 한다"**는 **계획 지향(Planning-oriented)** 철학을 제안합니다. 각 태스크는 계획에 기여하는 정도에 맞게 중요도가 결정되고, query 기반 인터페이스를 통해 모듈 간 정보가 통합됩니다.

**핵심 개념**

- **계획 지향 설계(Planning-oriented Design)**: 자율주행의 최종 목표인 경로 계획을 기준으로 상위 태스크(인식, 예측)의 설계 방향을 결정하는 접근법
- **엔드-투-엔드(End-to-end)**: 카메라 영상 입력부터 미래 주행 경로 출력까지 하나의 네트워크가 처리
- **Query 기반 인터페이스**: 각 모듈이 학습 가능한 query 벡터를 통해 서로 정보를 주고받아, 그래디언트가 전체 파이프라인에 걸쳐 역전파 가능

---

## 2. Methodology: UniAD 전체 파이프라인

**요약**

UniAD는 아래 그림과 같이 5개 모듈이 순차적으로 연결된 구조입니다:

**멀티-카메라 이미지** → BEV 백본 → **TrackFormer** → **MapFormer** → **MotionFormer** → **OccFormer** → **Planner**

각 모듈의 출력 query가 다음 모듈의 입력으로 전달되며, 모든 모듈은 **Transformer Decoder** 구조로 설계됩니다.

---

### 2.1 Perception: TrackFormer & MapFormer

**요약**

**TrackFormer**는 멀티-카메라 이미지에서 추출한 BEV(Bird's Eye View) feature를 이용해 주변 에이전트(차량, 보행자 등)를 검출하고 추적합니다. DETR 스타일의 query 기반 검출을 사용하며, 매 프레임마다 **detection query**(새로운 객체 탐지)와 **track query**(이전 프레임에서 유지되는 객체)를 함께 사용합니다. 트랙이 사라질 때까지 query를 유지함으로써 별도의 비미분 후처리 없이 추적을 수행합니다.

**MapFormer**는 2D 파노픽 분할(Panoptic Segmentation) 방법에서 영감을 받아, 차선, 구분선, 교차로 등 도로 요소를 map query로 표현합니다. 위치 및 구조 정보를 인코딩하여 하위 모션 예측 모듈에 도로 맥락을 제공합니다.

**핵심 개념**

- **BEV(Bird's Eye View) Feature**: 여러 카메라 이미지를 하나의 조감도 형태의 2D 특징 맵으로 변환한 표현. 공간적 일관성이 높아 자율주행에 유리
- **Detection Query / Track Query**: Detection query는 새로운 객체를 탐지하기 위해 초기화되는 학습 가능한 벡터이고, track query는 이전 프레임에서 이어받아 동일 객체를 계속 추적하는 벡터
- **파노픽 분할(Panoptic Segmentation)**: 이미지의 모든 픽셀을 "어떤 클래스인가"(semantic) + "어떤 개별 인스턴스인가"(instance)로 동시에 분류하는 방법

---

### 2.2 Prediction: MotionFormer

**요약**

MotionFormer는 모든 에이전트의 미래 궤적을 장면 전체 관점에서 동시에 예측합니다. TrackFormer의 에이전트 query $Q_A$와 MapFormer의 지도 query $Q_M$을 입력받아 에이전트 간, 에이전트-지도 간 상호작용을 모델링합니다.

각 에이전트에 대해 **top-k 개의 가능한 궤적**을 다중 모달 예측으로 출력하며, **장면 중심(scene-centric)** 방식으로 단일 포워드 패스에서 모든 에이전트를 처리합니다.

자아 차량(ego-vehicle)을 위한 특별한 **ego-vehicle query**를 추가하여 자아 차량이 다른 에이전트와 상호작용하며 미래 의도를 표현할 수 있게 합니다.

**핵심 수식**

에이전트-지도 상호작용:

$$Q_{a/m} = \text{MHCA}(\text{MHSA}(Q), Q_A / Q_M)$$

**수식 설명**:
- **$Q$**: 현재 모션 query (에이전트의 현재 상태 표현)
- **$Q_A$**: 주변 에이전트 query (다른 차량, 보행자 등)
- **$Q_M$**: 지도 query (차선, 도로 구조 등)
- **$\text{MHSA}$**: Multi-Head Self-Attention — query들끼리 서로 정보를 교환 (에이전트 간 상호작용)
- **$\text{MHCA}$**: Multi-Head Cross-Attention — query가 외부 정보(다른 에이전트 또는 지도)를 참조 (에이전트-지도 상호작용)
- 직관: 에이전트가 "주변 차들은 어디 있지? 도로 구조는 어떻지?"를 보면서 자신의 미래 행동을 결정하는 과정

모션 query의 위치 임베딩:

$$Q_{pos} = \text{MLP}(\text{PE}(I^s)) + \text{MLP}(\text{PE}(I^a)) + \text{MLP}(\text{PE}(\hat{x}_0)) + \text{MLP}(\text{PE}(\hat{x}_{ep}^{l-1}))$$

**수식 설명**:
- **$I^s$**: 장면 수준 앵커 (전체 장면의 일반적인 이동 패턴)
- **$I^a$**: 에이전트 수준 앵커 (개별 에이전트의 예상 의도)
- **$\hat{x}_0$**: 에이전트의 현재 위치
- **$\hat{x}_{ep}^{l-1}$**: 이전 레이어에서 예측된 궤적의 끝점 (coarse-to-fine 방식으로 점진적 정밀화)
- 직관: 모션 예측은 "어떤 패턴으로 움직이나" + "지금 어디 있나" + "지금까지 예측한 도착지가 어딘가"를 모두 고려

**비선형 스무딩(Non-linear Optimization)**:

$$\tilde{x} = \arg\min_x c(\mathbf{x}, \hat{\mathbf{x}})$$

$$c(\mathbf{x}, \hat{\mathbf{x}}) = \lambda_{xy}\|\mathbf{x}, \hat{\mathbf{x}}\|_2^2 + \lambda_{goal}\|\mathbf{x}_T, \hat{\mathbf{x}}_T\|_2 + \sum_{\phi \in \Phi} \phi(\mathbf{x})$$

**수식 설명**:
- **$\hat{\mathbf{x}}$**: 네트워크가 직접 예측한 궤적
- **$\tilde{\mathbf{x}}$**: 물리적으로 타당하도록 보정된 최종 궤적
- **$\lambda_{xy}$**: 예측 궤적에 가깝게 유지하는 항의 가중치
- **$\lambda_{goal}$**: 목표 지점에 도달해야 한다는 제약의 가중치
- **$\Phi$**: 운동학적 제약 집합 (저크, 곡률, 가속도, 속도 제한 등)
- 직관: 신경망 예측 결과를 물리 법칙에 맞게 다듬는 후처리 과정 (급격한 방향 전환, 물리적으로 불가능한 가속 등을 제거)

**핵심 개념**

- **다중 모달 예측(Multi-modal Prediction)**: 미래 경로가 하나가 아니라 여러 가능성이 있음을 반영하여 top-k 후보 궤적을 함께 출력
- **장면 중심(Scene-centric) 방식**: 모든 에이전트를 공통 좌표계에서 한 번의 연산으로 처리 (각 에이전트 중심으로 좌표 변환하는 에이전트 중심 방식과 대비)
- **Coarse-to-fine 예측**: 레이어를 거치며 점점 더 정밀한 궤적을 예측하는 방식

---

### 2.3 Prediction: OccFormer

**요약**

OccFormer는 미래 시간에 따른 **BEV 점유 그리드(Occupancy Grid)**를 예측합니다. 각 셀이 언제, 어느 에이전트에 의해 점유될지를 예측하여, Planner가 충돌을 피할 수 있는 정보를 제공합니다.

OccFormer는 **장면 수준(scene-level)**과 **인스턴스 수준(instance-level)** 정보를 동시에 활용합니다. 장면 수준 BEV feature $F^t$는 전체 도로 상황을 파악하고, 에이전트 수준 feature는 개별 객체의 경계와 점유를 정밀하게 예측합니다.

**핵심 수식**

에이전트-장면 픽셀 상호작용:

$$D_{ds}^t = \text{MHCA}(\text{MHSA}(F_{ds}^t), G^t, \text{attn\_mask} = O_m^t)$$

**수식 설명**:
- **$F_{ds}^t$**: 다운샘플된 BEV dense feature (장면 전체 맥락)
- **$G^t$**: 에이전트 수준 feature (각 객체의 개별 특성)
- **$O_m^t$**: 점유 마스크 (attention 범위를 점유된 픽셀로 제한)
- **$\text{attn\_mask}$**: 각 픽셀이 점유한 에이전트만 참조하도록 강제하는 마스크
- 직관: 각 도로 위치가 "나를 점유하는 에이전트가 누구인지"를 attention으로 파악

인스턴스 점유 예측:

$$\hat{O}_A^t = U^t \cdot F_{doc}^A$$

**수식 설명**:
- **$U^t$**: 에이전트 feature에서 생성된 공간 가중치 행렬
- **$F_{doc}^A$**: 에이전트 수준 dense feature
- **$\hat{O}_A^t$**: 시간 $t$에서의 인스턴스별 점유 예측
- 직관: 행렬 곱셈만으로 각 에이전트가 어느 위치를 차지하는지 효율적으로 계산

**핵심 개념**

- **점유 그리드(Occupancy Grid)**: BEV 공간을 격자(grid)로 나누어 각 셀이 점유(occupied)인지 아닌지를 나타내는 표현. 충돌 회피 계획에 핵심적으로 사용
- **인스턴스 점유(Instance-wise Occupancy)**: 어떤 에이전트가 어느 위치를 점유하는지 개별 식별이 가능한 점유 예측
- **$T_o$ 블록 구조**: $T_o$개의 순차적 블록으로 구성되어 각 블록이 미래 시간 스텝 하나의 점유를 예측

---

### 2.4 Planning: Planner

**요약**

Planner는 UniAD의 최종 모듈로, ego-vehicle의 미래 주행 경로를 예측합니다. HD 맵이나 사전 정의된 경로 없이 동작하며, 내비게이션 명령(좌회전/우회전/직진)만을 입력으로 받습니다.

MotionFormer의 ego-vehicle query가 계획 query로 사용되어, 주변 에이전트 및 지도 정보가 이미 인코딩된 상태에서 경로를 계획합니다. OccFormer의 점유 예측 결과를 활용하여 점유된 영역을 회피하는 최적화를 수행합니다.

**핵심 수식**

최적 경로 탐색:

$$\tau^* = \arg\min_\tau f(\tau, \hat{\tau}, \hat{O})$$

**수식 설명**:
- **$\tau$**: 후보 경로 (multiple-shooting으로 생성)
- **$\hat{\tau}$**: 네트워크의 초기 계획 예측
- **$\hat{O}$**: OccFormer에서 예측한 점유 지도
- **$\tau^*$**: 비용 함수를 최소화하는 최적 경로
- 직관: 여러 경로 후보 중 안전하고 목적지에 잘 도달하는 경로를 선택

비용 함수:

$$f(\tau, \hat{\tau}, \hat{O}) = \lambda_{tmed} \|\tau, \hat{\tau}\|_2 + \lambda_{obs} \sum_t \mathcal{D}(\tau_t, \hat{O}^t)$$

$$\mathcal{D}(\tau_t, \hat{O}^t) = \sum_{(x,y) \in S} \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{\|\tau_t - (x,y)\|_2^2}{2\sigma^2}\right)$$

**수식 설명**:
- **$\lambda_{tmed}$**: 예측 경로 $\hat{\tau}$에 가깝게 유지하는 항의 가중치
- **$\lambda_{obs}$**: 장애물 회피 항의 가중치
- **$\mathcal{D}(\tau_t, \hat{O}^t)$**: 충돌 비용 — 점유된 위치에 가까울수록 높은 비용
- **$S = \{(x,y) \mid (x,y)\text{가 점유됨}, d < \bar{d}\}$**: 근처 점유 셀 집합
- **$\sigma$**: 가우시안 분포의 폭 (충돌 회피의 민감도 조절)
- 직관: "예측한 경로에 가깝게 가되, 점유된 곳(다른 차, 보행자)은 피해라"는 두 조건을 동시에 최적화

**핵심 개념**

- **Multiple-shooting**: 여러 초기값에서 출발하여 후보 경로들을 생성하고, 그 중 최적을 선택하는 수치 최적화 기법
- **내비게이션 커맨드 임베딩**: "좌회전", "우회전", "직진" 세 가지 명령을 학습 가능한 임베딩으로 변환하여 계획 query에 조건부로 사용
- **충돌 비용(Collision Cost)**: OccFormer가 예측한 점유 지도를 기반으로 각 후보 경로의 안전성을 수치로 평가하는 항

---

### 2.5 Learning

**요약**

UniAD는 두 단계로 학습됩니다:
1. **1단계**: 추적(TrackFormer)과 매핑(MapFormer) 모듈만 먼저 학습 (약 6 epoch)
2. **2단계**: 전체 모델(인식 + 예측 + 계획)을 함께 20 epoch 학습

**공유 매칭(Shared Matching)**: 인식과 예측 태스크에서 ground truth와 예측 쌍을 맞출 때 Hungarian 매칭 알고리즘을 공유합니다. 추적에서 생성된 매칭 결과를 모션 및 점유 예측에서도 재사용하여 전체 파이프라인의 일관성을 유지합니다.

---

## 3. Experiments: 벤치마크 결과 및 Ablation Study

**요약**

**nuScenes 벤치마크** 결과:

| 태스크 | 지표 | UniAD 결과 |
|--------|------|-----------|
| 다중 객체 추적 | AMOTA↑ | 0.359 (SOTA) |
| 온라인 매핑 | IoU (차선 분류기) | 30.6 / 17.2 |
| 모션 예측 | minADE↓ | 0.71 (38.3%↓ vs PnPNet) |
| 점유 예측 | IoU (near)↑ | 63.4 (vs FIERY 59.4) |
| 계획 | avg.L2↓ / Col.Rate↓ | 1.03 / 0.31 |

**주요 Ablation 결과**:

- **두 예측 모듈(모션 + 점유)이 모두 필요**: 둘 다 있을 때 계획 성능이 최고 (L2: 1.03, 충돌률: 0.31)
- **계획 지향 설계의 효과**: MTL 방식 대비 minADE -15.2%, minFDE -17.0%, 충돌률 -0.15m 향상
- **MotionFormer의 각 컴포넌트**:
  - 회전된 장면 수준 앵커(rotated scene-level anchor): -15.8% minADE 향상
  - 에이전트-목표 포인트 상호작용: -11.2% minFDE 향상
  - 비선형 최적화(NLO): 추가 -8.4% minFDE 향상
- **OccFormer의 attention mask**: IoU 62.6%↑, VPQ 53.2%↑ (mask 없는 기준 대비)
- **Planner의 점유 회피 최적화**: 충돌률 1s에서 0.44→0.40, 3s에서 1.76→1.05로 감소

**핵심 개념**

- **AMOTA(Average Multi-Object Tracking Accuracy)**: 다양한 신뢰도 임계값에서 평균화된 MOT 정확도 지표
- **minADE / minFDE**: 예측된 top-k 궤적 중 ground truth에 가장 가까운 것의 평균/최종 변위 오차
- **VPQ(Video Panoptic Quality)**: 점유 예측의 시간적 일관성과 정확도를 함께 평가하는 지표
- **avg.L2 / Col.Rate**: 계획 경로의 ground truth 대비 L2 거리 오차와 충돌 발생 비율

---

## 4. 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **계획 지향(Planning-oriented)** | 모든 태스크를 최종 목표인 계획에 기여하도록 설계하는 철학 |
| **Query 기반 인터페이스** | 학습 가능한 벡터(query)가 모듈 간 정보를 전달하고 gradient를 연결 |
| **BEV Feature** | 멀티-카메라 이미지를 조감도 형태의 2D 특징 맵으로 통합 |
| **TrackFormer** | DETR 스타일의 detection/track query로 NMS 없이 멀티-객체 추적 |
| **MapFormer** | 도로 요소를 파노픽 분할로 인식하여 지도 query 생성 |
| **MotionFormer** | 장면 중심 방식으로 모든 에이전트의 다중 모달 궤적 예측 |
| **OccFormer** | 미래 BEV 점유 그리드를 인스턴스 단위로 예측 |
| **Planner** | ego-vehicle query + 점유 회피 최적화로 안전한 경로 계획 |
| **Two-stage Training** | 인식 모듈 선학습 후 전체 모델 joint training으로 학습 안정화 |

---

## 5. 결론 및 시사점

**결론**

UniAD는 자율주행에서 **"계획을 위해 인식하고, 인식을 위해 예측한다"**는 계획 지향적 철학을 최초로 체계적으로 구현한 엔드-투-엔드 프레임워크입니다. 5개의 핵심 태스크를 단일 네트워크에서 처리하면서도 각 태스크에서 SOTA 수준의 성능을 달성했으며, 특히 안전성과 직결된 계획 태스크에서 기존 방법 대비 충돌률을 51.2%~56.3% 감소시켰습니다.

**실무적 시사점**

1. **모듈 선택의 기준**: 어떤 태스크를 포함할지는 "계획에 얼마나 기여하는가"로 결정해야 함. UniAD는 모션 예측과 점유 예측 모두가 필수임을 실험으로 검증
2. **Query 기반 설계의 강점**: gradient가 전체 파이프라인을 통과하여 각 모듈이 최종 계획 성능을 위해 최적화됨. 정보 손실과 오류 누적을 최소화
3. **안전성 우선 설계**: OccFormer의 점유 예측을 Planner에 직접 연결하여 실시간으로 충돌 가능 영역을 회피하는 구조는 실제 배포에서 중요한 안전장치
4. **한계점**: 여러 태스크를 동시에 학습하므로 높은 연산 자원이 필요하고, 시간적 이력(temporal history) 처리가 복잡함. 경량화 배포를 위한 추가 연구가 필요


---

*관련 논문: [BEVFormer](/posts/papers/BEVFormer/), [Attention Is All You Need](/posts/papers/attention-is-all-you-need/), [VAD](/posts/papers/VAD/), [TransFuser](/posts/papers/TransFuser/), [nuScenes](/posts/papers/nuscenes-multimodal-dataset-autonomous-driving/), [NAVSIM](/posts/papers/NAVSIM/)*
