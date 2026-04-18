---
title: "TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving"
date: 2026-04-14T00:00:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving"]
tags: ["Autonomous Driving", "Transformer", "Sensor Fusion", "Imitation Learning", "LiDAR"]
---

## 개요

- **저자**: Kashyap Chitta, Aditya Prakash, Bernhard Jaeger, Zehao Yu, Katrin Renz, Andreas Geiger
- **소속**: University of Tübingen, Max Planck Institute for Intelligent Systems
- **발행년도**: 2022 (arXiv:2205.15997, IEEE TPAMI)
- **주요 내용**: 카메라와 LiDAR를 Transformer의 self-attention 메커니즘으로 통합하는 **다중 모달 융합 Transformer(TransFuser)**를 제안. 기존 합성곱(CNN) 기반 융합 방식의 locality 한계를 극복하여 전역적(global) 장면 맥락을 포착. CARLA 자율주행 벤치마크에서 이전 최고 방법 대비 충돌을 48% 감소시키며 공식 리더보드 1위 달성. NAVSIM에서도 강력한 baseline으로 활용됨.

## 목차

- Chapter 1: Introduction — 카메라+LiDAR 융합의 필요성과 Transformer 접근
- Chapter 2: Related Work — 다중 모달 인식, E2E 주행, 센서 융합 방법 정리
- Chapter 3: TransFuser 구조 — 다중 모달 융합 Transformer, 웨이포인트 예측, 보조 태스크
- Chapter 4: Experiments — CARLA 벤치마크, Longestè6 평가, 소거 실험
- Chapter 5: Discussion & Conclusions

---

## Chapter 1: Introduction

**요약**

자율주행에서 LiDAR는 정확한 3D 거리 정보를 제공하고, 카메라는 풍부한 시각·의미 정보를 제공한다. 두 센서는 상호 보완적이므로 함께 활용하는 것이 이상적이다.

기존 연구들은 두 센서의 특징 맵을 단순히 채널 방향으로 합치거나(concatenation), 픽셀 단위로 더하는(summation) 방식으로 융합했다. 그러나 이런 합성곱(CNN) 기반 방식은 **locality 가정** — 즉, 한 위치의 특징이 주변 위치에만 의존한다는 가정 — 으로 인해 전역적 장면 맥락을 포착하기 어렵다.

예를 들어 Fig. 1처럼 교차로에서 안전하게 주행하려면 교통 신호등(멀리 있는 카메라 정보)과 접근하는 차량(LiDAR 포인트 클라우드)을 **동시에** 고려해야 한다. CNN은 이 두 정보를 공간적으로 멀리 떨어진 위치에서 관계를 맺기 어렵다.

TransFuser는 Transformer의 **self-attention**을 사용하여 이미지와 LiDAR 특징 맵의 모든 위치 쌍 사이의 상호 관계를 학습한다. 이를 통해 전역적 장면 이해가 가능해지고, 복잡한 교차로와 고밀도 교통 상황에서 주행 성능이 크게 향상된다.

**핵심 개념**

- **모방 학습(Imitation Learning, IL)**: 전문가(사람 또는 규칙 기반 플래너)의 주행 데이터를 모방하여 정책을 학습. 강화학습보다 샘플 효율이 높음
- **Sensor Fusion**: 서로 다른 센서(카메라, LiDAR)의 정보를 하나의 표현으로 통합하는 방법
- **Locality 한계**: CNN은 receptive field가 제한되어 공간적으로 멀리 떨어진 정보 간 직접 관계를 학습하기 어려움
- **Global context**: 장면 전체를 한 번에 이해하는 능력. 교차로에서 신호등과 차량을 동시에 고려하는 것이 대표적 예

---

## Chapter 2: Related Work

**요약**

**다중 모달 자율주행**: NEAT, LAV 등이 카메라와 LiDAR를 결합하여 E2E 주행을 수행. PointPainting은 이미지 시맨틱을 LiDAR 포인트에 투영. 대부분은 CNN 기반 융합이거나 두 모달리티를 독립적으로 처리.

**Transformer in Driving**: Attention 메커니즘이 NLP에서 성공 후 비전으로 확장(ViT). NEAT은 어텐션을 자율주행에 적용했으나 단일 모달리티. TransFuser는 **크로스 모달 어텐션**을 다중 스케일에서 수행하는 최초의 방법 중 하나.

**BEV 기반 접근**: Lift-Splat, BEVFusion 등이 카메라를 BEV로 변환 후 융합. TransFuser는 BEV LiDAR와 원근 이미지를 Transformer로 직접 융합하여 변환 오류를 피함.

---

## Chapter 3: TransFuser 구조

**요약**

### 3.1 입력 표현

- **카메라**: 세 방향(전방 60°, 좌 60°, 우 60°)의 RGB 이미지. 각 이미지는 $900 \times 256$ 해상도로 처리
- **LiDAR**: 5개 LiDAR 포인트 클라우드를 병합한 BEV 히스토그램. $256 \times 256$ 해상도, 2채널(높이 구간별 점 밀도)
- **Ego status**: 현재 속도(스칼라 값), 고속도로/교차로/일반 도로 구분 내비게이션 목표(3차원 one-hot 벡터)

### 3.2 다중 모달 융합 Transformer

TransFuser의 핵심은 **ResNet 백본의 중간 특징 맵에서 Transformer를 반복 적용**하는 것이다. 이미지와 LiDAR 각각에 ResNet 인코더를 적용하고, 4개 해상도($\frac{1}{2}, \frac{1}{4}, \frac{1}{8}, \frac{1}{16}$)에서 특징 맵을 추출한다.

각 해상도에서 두 모달리티의 특징 맵을 **토큰(token) 시퀀스**로 변환한 후, Transformer의 self-attention을 적용한다:

$$\mathbf{Q} = \mathbf{F}^{\text{img}} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{F}^{\text{lidar}} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{F}^{\text{all}} \mathbf{W}_V$$

$$\mathbf{A} = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{D_k}}\right)$$

$$\mathbf{F}^{\text{out}} = \text{MLP}(\mathbf{A} \cdot \mathbf{V})$$

**수식 설명**:
- **$\mathbf{F}^{\text{img}}$**: 이미지 특징 맵을 토큰 시퀀스로 펼친 것 (각 토큰 = $32 \times 32$ 패치)
- **$\mathbf{F}^{\text{lidar}}$**: LiDAR BEV 특징 맵의 토큰 시퀀스
- **$\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$**: Query, Key, Value 투영 행렬 (학습 파라미터)
- **$D_k$**: Key 벡터의 차원수 (스케일링으로 softmax 포화 방지)
- **$\mathbf{A}$**: Attention 가중치 행렬. 이미지 토큰과 LiDAR 토큰 사이의 상관도
- **$\mathbf{F}^{\text{out}}$**: 크로스 모달 정보가 융합된 출력 특징

실제로는 이미지와 LiDAR 토큰을 **하나의 시퀀스로 연결**한 뒤 self-attention을 수행하므로, 이미지→LiDAR, LiDAR→이미지 양방향 크로스 어텐션이 동시에 이루어진다. 각 Transformer는 4개 어텐션 헤드, 25개(이미지)~64개(LiDAR) 토큰을 처리한다.

Transformer 출력은 원래 특징 맵 크기로 복원된 후 element-wise summation으로 백본 특징 맵에 더해진다. 4개 해상도에서 이 과정을 반복하며 점진적으로 전역 정보를 통합한다.

최종적으로 512차원 글로벌 특징 벡터를 생성하고, GRU를 통해 웨이포인트를 순차적으로 예측한다.

### 3.3 웨이포인트 예측 네트워크

글로벌 특징 벡터를 GRU(Gated Recurrent Unit)에 입력하여 $T = 4$개의 미래 웨이포인트 $\{\hat{w}_t\}_{t=1}^T$를 자기회귀(auto-regressive) 방식으로 예측한다. 학습 목표는:

$$\mathcal{L}_1 = \sum_{t=1}^{T} \|\hat{w}_t - w_t^*\|_1$$

**수식 설명**:
- **$\hat{w}_t$**: t번째 예측 웨이포인트 (자차 좌표계 기준 2D 위치)
- **$w_t^*$**: 전문가의 t번째 실제 웨이포인트 (ground truth)
- **$\|\cdot\|_1$**: L1 거리 (절댓값 차이의 합). L2보다 이상치에 덜 민감
- GRU가 이전 예측 결과를 받아 다음 웨이포인트를 예측하는 자기회귀 구조

웨이포인트는 ego-vehicle 좌표계 기준으로 나타내며, 목표 위치(goal location)도 GPS 좌표로 주어진다.

### 3.4 저속 제어기(PID Controller)

예측된 웨이포인트를 실제 steering과 throttle/brake 명령으로 변환하기 위해 두 개의 PID 제어기를 사용한다:
- **종방향(Longitudinal) PID**: 목표 속도와 실제 속도 차이로 throttle/brake 결정
- **횡방향(Lateral) PID**: 다음 웨이포인트 방향과 현재 방향 차이로 steering 결정

**Creeping 동작**: 차량이 오랫동안(55초 이상) 정지해 있으면 목표 속도를 3.0 m/s로 설정하여 강제로 전진시킨다. 교차로에서 모델이 멈추는 관성(inertia) 문제를 완화하지만 충돌 가능성도 있다.

### 3.5 보조 태스크 (Auxiliary Tasks)

웨이포인트 예측만으로는 의미 있는 중간 표현 학습이 어려울 수 있어, 4가지 보조 태스크를 추가한다:

1. **깊이 예측 (Depth Estimation)**: 이미지 브랜치에서 픽셀별 깊이 예측. 카메라 특징이 3D 공간을 이해하도록 유도
2. **2D 시맨틱 분할 (2D Semantic Segmentation)**: 이미지에서 도로, 차량, 보행자 등 클래스 분류
3. **HD 맵 예측 (HD Map Prediction)**: BEV에서 주행 가능 영역, 차선 중심선, 도로 경계 예측
4. **차량 탐지 (Vehicle Detection)**: BEV LiDAR 브랜치에서 차량 위치와 방향 예측. CenterPoint 헤드 사용

$$\mathcal{L}_{\text{total}} = \mathcal{L}_1 + \lambda_{\text{depth}} \mathcal{L}_{\text{depth}} + \lambda_{\text{seg}} \mathcal{L}_{\text{seg}} + \lambda_{\text{map}} \mathcal{L}_{\text{map}} + \lambda_{\text{det}} \mathcal{L}_{\text{det}}$$

보조 태스크를 모두 제거하면 DS가 53.76 → 46.23으로 크게 하락 (Table 7). 특히 HD 맵과 차량 탐지가 중요.

---

## Chapter 4: Experiments

**요약**

### 4.1 평가 환경

**CARLA 시뮬레이터**: 8개 마을(Town01~Town07, Town10)에서 평가. 학습에 사용하지 않은 새로운 마을과 날씨 조건(Night, Rain, Fog 등 6가지)에서 테스트.

**Longestè6 벤치마크**: 76개 사전 정의 경로로 구성된 공식 CARLA 평가. 평균 경로 길이 1.5km. 4가지 주요 지표:

| 지표 | 설명 |
|------|------|
| **DS (Driving Score)** | RC × IS의 곱. 종합 주행 점수 (높을수록 좋음) |
| **RC (Route Completion)** | 전체 경로 중 완료한 비율 (%) |
| **IS (Infraction Score)** | 위반 발생 시마다 패널티 곱. 충돌 0.65×, 신호 위반 0.7× 등 |
| **Collisions/km** | 킬로미터당 충돌 횟수 |

### 4.2 주요 결과

**Longestè6 벤치마크** (단일 모델):

| Method | DS↑ | RC↑ | IS↑ |
|--------|-----|-----|-----|
| NEAT | 44.82 | 31.94 | 0.65 |
| LAV | 38.42 | 43.62 | 0.60 |
| Late Fusion (GF) | 34.79 | 41.70 | 0.55 |
| Geometric Fusion (GF) | 34.22 | 47.91 | 0.47 |
| **TransFuser (Ours)** | **54.97** | **47.25** | **0.77** |

- TransFuser가 DS 기준 약 **10포인트** 차이로 1위
- 특히 IS(위반 점수)에서 큰 차이 → 교차로 등 복잡 상황에서 충돌 대폭 감소

**CARLA 공식 리더보드**:

| Method | DS↑ | RC↑ | IS↑ |
|--------|-----|-----|-----|
| NEAT | 21.83 | 41.71 | 0.65 |
| WOR | 31.37 | 57.65 | 0.56 |
| LAW | 61.85 | 94.46 | 0.64 |
| **TransFuser (Ours)** | **47.13** | **72.84** | **0.71** |
| **Latent TransFuser (Ours)** | **45.20** | **66.31** | **0.72** |

### 4.3 소거 실험

**보조 태스크 영향** (Table 7):

| 설정 | DS↑ | RC↑ | IS↑ |
|------|-----|-----|-----|
| No Aux Tasks | 46.23 | 56.88 | 0.58 |
| No Semantics | 53.76 | 56.82 | 0.66 |
| No HD Map | 50.96 | 49.52 | 0.58 |
| No Vehicle Detection | 53.43 | 54.49 | 0.67 |
| **All Losses (Default)** | **54.97** | **49.06** | **0.77** |

**센서 융합 방식 비교** (Table 5):
- **Global attention**: 메모리 과다로 $100 \times 100$ 해상도로 축소 필요, 성능↓
- **Points attention** (reference point만): DS 0.423
- **Local (Deformable)**: DS 최고

→ TransFuser의 크로스 모달 어텐션이 단순 Late/Early Fusion 대비 일관되게 우수

**입력 설정 소거** (Table 9):

| Camera FOV | DS↑ |
|-----------|-----|
| 120° (1 cam) | 47.90 |
| 180° (3 cam, default) | 47.35 |
| 240° (5 cam) | 46.37 |

- LiDAR 범위 확장이 오히려 성능↓ → LiDAR 좌표계 보조 태스크와 불일치
- 카메라 FOV 140°가 최적 (Table 9에서 NAVSIM 실험과 일치)

### 4.4 Attention 시각화

Fig. 6에서 LiDAR 쿼리 토큰이 교차로 주변의 이미지 토큰(교통 신호등, 차량)에 강하게 어텐션함을 확인. Table 4에서 초기 Transformer(T1, T2)는 LiDAR→Image 크로스 어텐션이 낮고, 후기(T3, T4)에서 높아짐 → 처음에는 각 모달리티 내 처리 후 점차 크로스 모달 통합.

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **다중 모달 융합 Transformer** | 이미지와 LiDAR 특징 맵을 토큰 시퀀스로 변환 후 self-attention으로 전역적 크로스 모달 관계 학습 |
| **자기회귀 웨이포인트 예측** | GRU가 이전 예측 웨이포인트를 입력받아 다음 웨이포인트를 순차적으로 예측 |
| **PID 제어기** | 예측된 웨이포인트를 실제 조향각과 가속/제동 명령으로 변환하는 고전적 제어 알고리즘 |
| **보조 태스크** | 깊이, 시맨틱 분할, HD 맵, 차량 탐지를 동시에 학습하여 의미 있는 중간 표현 유도 |
| **Creeping** | 장시간 정지 시 강제 전진하는 안전 휴리스틱. 교차로 관성 문제를 완화하나 충돌 리스크 존재 |
| **Driving Score (DS)** | 경로 완료율(RC)과 위반 점수(IS)의 곱. 자율주행 종합 성능 지표 |
| **Infraction Score (IS)** | 충돌·신호 위반 등 발생 시마다 일정 비율로 감점되는 패널티 지표. 1에 가까울수록 안전 |
| **모방 학습 (Imitation Learning)** | 전문가 데이터(상태→행동 쌍)를 supervised learning으로 모방. DAgger 등 확장으로 분포 이동 문제 완화 |

---

## 결론 및 시사점

TransFuser는 카메라와 LiDAR의 전역적 융합을 Transformer self-attention으로 실현하여 자율주행 성능을 크게 향상시켰다:

1. **전역 맥락 포착**: 교차로에서 신호등(이미지)과 접근 차량(LiDAR)을 동시에 고려하는 능력 확보
2. **보조 태스크의 중요성**: 웨이포인트 예측 외에 HD 맵, 차량 탐지를 함께 학습하면 표현의 질이 크게 향상
3. **단순함의 효과**: NAVSIM 결과에서 확인되듯, 간단한 아키텍처와 1 GPU 1일 학습으로도 UniAD, PARA-Drive 등 대형 모델과 대등한 성능

**한계**:
- 단일 타임스텝 입력만 처리하여 시간적 정보(차량 속도 추정 등) 활용 불가
- Creeping 휴리스틱이 일부 충돌을 유발
- 카메라-LiDAR 좌표 불일치로 LiDAR 범위 조절이 직관적이지 않은 결과를 낳기도 함

**실무적 시사점**: TransFuser는 CARLA 기반 E2E 자율주행의 새로운 baseline으로 자리잡았다. NAVSIM, Longestè6 등 여러 벤치마크에서 꾸준히 강력한 성능을 보이며, 간단하고 재현 가능한 코드베이스를 공개하여 후속 연구의 출발점으로 널리 사용된다.
