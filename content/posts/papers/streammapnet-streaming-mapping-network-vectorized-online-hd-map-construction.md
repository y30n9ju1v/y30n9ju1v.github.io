---
title: "StreamMapNet: Streaming Mapping Network for Vectorized Online HD Map Construction"
date: 2026-04-20T20:00:00+09:00
draft: false
categories: ["Papers"]
tags: ["Autonomous Driving", "HD Map", "BEV", "Temporal Fusion", "Transformer", "Streaming"]
---

## 개요

- **저자**: Tianyuan Yuan, Yicheng Liu, Yue Wang, Yilun Wang, Hang Zhao
- **소속**: Tsinghua University, University of Southern California
- **arXiv**: 2308.12570 (2023)
- **주요 내용**: 멀티프레임 비디오의 시간 정보를 streaming 방식으로 통합해 넓은 인식 범위(100×50m)에서도 안정적인 벡터 HD 맵을 온라인으로 구성하는 모델. 기존 벤치마크의 데이터 누출 문제를 발견하고 공정한 새 split을 제안

## 한계 극복

- **기존 한계 1 — 단일 프레임 입력**: VectorMapNet, MapTR 등 기존 방법은 단일 프레임만 사용해 폐색·역광 등 어려운 환경에서 오류 발생, 프레임 간 시간적 비일관성 문제
- **기존 한계 2 — 좁은 인식 범위**: 기존 방법은 60×30m 범위에선 잘 작동하나 100×50m로 넓히면 성능이 급락. 표준 Deformable Attention이 기다란 맵 요소의 비국소(non-local) 형태를 포착하지 못하기 때문
- **기존 한계 3 — 벤치마크 공정성 문제**: nuScenes 공식 split에서 검증 위치의 85%가 학습 데이터와 겹치고, Argoverse2도 54% 겹침 → 모델이 위치를 암기해 과적합, 실제 일반화 능력 측정 불가
- **이 논문의 접근 방식**: (1) Multi-Point Attention으로 넓은 범위의 비국소 어텐션 구현, (2) Streaming 시간 융합으로 메모리 효율적인 장기 이력 통합, (3) 지리적 겹침 없는 공정한 새 데이터 split 제안

## 목차

- Chapter 1: Introduction
- Chapter 2: Related Works
- Chapter 3: StreamMapNet Model
  - 3.1 Overall Architecture
  - 3.2 BEV Feature Encoder
  - 3.3 Decoder Transformer (Multi-Point Attention)
  - 3.4 Temporal Fusion (Query Propagation + BEV Fusion)
  - 3.5 Matching Cost and Training Loss
- Chapter 4: Experiments
  - 4.1 Rethinking on Datasets (벤치마크 공정성 분석)
  - 4.2 Implementation Details
  - 4.4 Comparison with Baselines
  - 4.5 Ablation Studies
  - 4.6 Qualitative Analysis
- Chapter 5: Conclusion

---

## Chapter 1: Introduction

**요약**

HD 맵은 자율주행 차량이 차선 구분선, 보행자 횡단보도, 도로 경계를 인식해 안전하게 주행하는 데 필수적입니다. 기존 오프라인 SLAM 기반 방식은 고비용이고 업데이트가 느린 반면, 최근 딥러닝 기반 온라인 방식(VectorMapNet, MapTR)은 실시간 생성이 가능하지만 두 가지 핵심 문제를 갖고 있습니다.

첫째, **단일 프레임 입력 한계**: 트럭 같은 큰 물체에 가려진 도로 구조를 한 프레임으로는 재구성하기 어렵고, 타임스탬프마다 맵이 달라지는 시간 불일관성 문제가 발생합니다. 둘째, **좁은 인식 범위**: 기다랗고 비규칙적인 맵 요소를 포착하려면 비국소(non-local) 어텐션이 필요한데, 표준 Deformable DETR은 이를 지원하지 못합니다.

StreamMapNet은 이를 해결하기 위해 **Multi-Point Attention**(넓은 범위 어텐션)과 **Streaming 시간 융합**(장기 이력 메모리)을 결합한 end-to-end 파이프라인을 제안합니다.

**핵심 개념**

- **Stacking vs Streaming**: Stacking은 여러 과거 프레임을 한 번에 concat해 처리 → 메모리·지연 비용이 프레임 수에 비례해 증가. Streaming은 한 번에 한 프레임씩 처리하되 압축된 메모리 피처를 다음 프레임으로 전달 → 상수 비용으로 장기 이력 통합 가능
- **Non-local attention**: 물체 탐지와 달리 차선·도로 경계는 BEV 공간에서 기다랗고 불규칙한 형태. 중심점 근방의 작은 영역에만 집중하는 standard deformable attention은 이러한 요소를 포착하기 어려움

---

## Chapter 2: Related Works

**요약**

관련 연구는 세 축으로 분류됩니다.

1. **온라인 벡터 HD 맵 구성**: HDMapNet(래스터 세그 + 후처리) → VectorMapNet(DETR 기반 end-to-end) → MapTR(계층적 쿼리) → BeMapNet(Bézier 곡선 기반). 모두 단일 프레임, 좁은 인식 범위라는 공통 한계.

2. **BEV 인식**: Lift-Splat-Shoot(깊이 분포), BEVFormer(deformable cross-attention), SimpleBEV(IPM 기반). StreamMapNet은 BEVFormer를 BEV 인코더로 채택.

3. **카메라 기반 3D 탐지의 시간 모델링**: BEVDet4D, BEVFormer v2 등은 stacking 전략으로 단기 시간 융합. VideoBEV, StreamPETR, Sparse4D v2는 streaming 전략으로 메모리 피처를 전파. StreamMapNet은 이 streaming 아이디어를 HD 맵 구성에 최초로 적용.

---

## Chapter 3: StreamMapNet Model

### 3.1 전체 아키텍처

**요약**

StreamMapNet은 세 주요 컴포넌트로 구성됩니다:
1. **BEV Feature Encoder**: 멀티뷰 카메라 이미지 → BEV 피처 맵
2. **Decoder Transformer with Multi-Point Attention**: BEV 피처에서 맵 인스턴스를 예측
3. **Memory Buffer**: 이전 프레임의 BEV 피처와 쿼리를 저장·전달하는 시간 융합 모듈

출력은 각 맵 인스턴스의 클래스 레이블과 $N_p$개의 점 좌표로 구성된 폴리라인 $\boldsymbol{P} = \{(x_i, y_i)\}_{i=1}^{N_p}$입니다.

### 3.2 BEV Feature Encoder

**요약**

공유 CNN 백본(ResNet50)으로 멀티뷰 이미지의 2D 피처를 추출하고, FPN(Feature Pyramid Network)으로 멀티스케일 피처를 구성합니다. 이후 BEV 인코더(BEVFormer의 단일 인코더 레이어)가 2D 피처를 BEV 공간으로 리프팅해 $\mathcal{F}_{\text{BEV}} \in \mathbb{R}^{C \times H \times W}$ 피처 맵을 생성합니다.

**핵심 개념**

- **FPN (Feature Pyramid Network)**: 다양한 해상도의 피처 맵을 계층적으로 합쳐 크고 작은 객체를 모두 잘 탐지하는 멀티스케일 피처 구조
- **BEV Lifting**: 카메라의 2D 원근 이미지를 BEV 조감도 공간의 3D 피처로 변환하는 과정

### 3.3 Decoder Transformer — Multi-Point Attention

**요약**

표준 Deformable DETR의 cross-attention은 각 쿼리에 하나의 참조점(reference point)을 할당하고, 이 점 주변에서만 피처를 수집합니다. 이 방식은 작고 국소적인 객체(차량, 보행자)에는 적합하지만, BEV 공간에서 기다랗고 비국소적인 맵 요소(도로 경계, 차선 구분선)에는 맞지 않습니다.

StreamMapNet은 이를 해결하기 위해 **Multi-Point Attention**을 제안합니다: 단일 중심 참조점 대신, 이전 레이어에서 예측한 **$N_p$개의 폴리라인 점들**을 참조점으로 사용해 맵 요소 전체 영역에서 피처를 수집합니다.

**Standard Deformable DETR (i번째 레이어)**

$$Q_i = \sum_{j=1}^{N_{\text{off}}} W_i^j \cdot \text{DA}(Q_{i-1}, R_i + O_i^j, \mathcal{F}_{\text{BEV}}) \tag{3}$$

$$R_{i+1} = \text{sigmoid}(\text{sigmoid}^{-1}(R_i) + \text{Reg}_i(Q_i)) \tag{4}$$

**수식 설명 (Standard)**
- **$Q_i$**: $i$번째 레이어의 쿼리 피처
- **$R_i$**: 현재 레이어의 단일 참조점 (중심점 좌표). 레이어마다 잔차로 갱신됨
- **$O_i^j$**: 참조점으로부터의 $j$번째 샘플링 오프셋 (참조점 주변 어디를 볼지)
- **$W_i^j$**: $j$번째 샘플의 어텐션 가중치
- **$\text{DA}(\cdot)$**: Deformable Attention 연산 — 지정 위치의 피처를 BEV 맵에서 샘플링

**Multi-Point Attention (StreamMapNet)**

$$O_i = \text{Offset\_Embed}(Q_{i-1}) \tag{5}$$

$$W_i = \text{Weight\_Embed}(Q_{i-1}) \tag{6}$$

$$Q_i = \sum_{j=1}^{N_p} \sum_{k=1}^{N_{\text{off}}} W_i^{(j-1) \cdot N_{\text{off}}+k} \cdot \text{DA}(Q_{i-1}, P_i^j + O_i^{(j-1) \cdot N_{\text{off}}+k}, \mathcal{F}_{\text{BEV}}) \tag{7}$$

$$P_{i+1} = \text{sigmoid}(\text{Reg}(Q_i)) \tag{8}$$

**수식 설명 (Multi-Point)**
- **$P_i^j$**: $i$번째 레이어에서 예측한 폴리라인의 $j$번째 점 좌표 (참조점 역할)
- **$N_p$**: 폴리라인을 구성하는 점의 수. 이 $N_p$개의 점 모두를 참조점으로 사용
- **$N_{\text{off}}$**: 각 참조점 주변의 추가 샘플링 오프셋 수
- **$O_i^{(j-1) \cdot N_{\text{off}}+k}$**: $j$번째 폴리라인 점 주변 $k$번째 오프셋
- **핵심 차이**: 기존은 중심점 1개 → 주변 국소 샘플링. Multi-Point는 폴리라인 전체 $N_p$개 점 → 요소 전체 범위에서 피처 수집. 연산 복잡도는 $O(N_p)$로 전역 어텐션 $O(HW)$보다 효율적

**핵심 개념**

- **Absolute vs Relative Prediction**: 기존 Deformable DETR은 참조점으로부터 잔차(relative)로 위치를 예측. Multi-Point Attention은 공유 MLP로 절대 좌표(absolute)를 직접 예측해 폴리라인 점들의 전역 위치를 안정적으로 잡음
- **Non-local Attention Region**: 기존은 쿼리의 중심 근처만 보는 "local" 어텐션. Multi-Point는 폴리라인 전체를 커버하는 "non-local" 어텐션

### 3.4 Temporal Fusion

**요약**

시간 정보 통합은 두 가지 보완적인 모듈로 구성됩니다.

#### Query Propagation (희소 쿼리 레벨)

맵 요소(차선, 경계 등)는 프레임 간에 거의 변하지 않습니다. 따라서 이전 프레임에서 높은 신뢰도로 예측된 상위 $k$개의 쿼리를 다음 프레임으로 전달(propagate)합니다. 이 쿼리들은 자차 좌표계가 달라졌으므로 변환 행렬 $\boldsymbol{T}$로 좌표계를 맞춥니다.

$$Q_t = \phi_t(\text{Concat}(Q_{t-1}, \text{flatten}(\boldsymbol{T}))) + Q_{t-1} \tag{9}$$

$$\boldsymbol{P}_t = \boldsymbol{T} \cdot \text{homogeneous}(\boldsymbol{P}_{t-1})_{:,0:2} \tag{10}$$

**수식 설명**

- **$Q_{t-1}$**: 이전 프레임의 전파된 쿼리 임베딩 (위치 및 의미 정보를 담은 벡터)
- **$\boldsymbol{T}$**: 두 프레임 좌표계 사이의 4×4 변환 행렬 (자차가 이동한 만큼 좌표를 보정)
- **$\phi_t$**: 변환 정보를 쿼리에 통합하는 MLP (잔차 연결 포함)
- **$\boldsymbol{P}_{t-1}$**: 이전 프레임의 폴리라인 점 좌표. 동차 좌표(homogeneous)로 변환 후 $\boldsymbol{T}$를 곱해 현재 좌표계로 이동
- **직관**: 이전 프레임에서 발견한 맵 요소들의 위치 기억을 현재 프레임에 "힌트"로 줌

#### BEV Fusion (밀집 BEV 피처 레벨)

희소 쿼리 전파와 달리, 밀집 BEV 피처도 이전 프레임 정보를 담을 수 있습니다. 이전 프레임의 BEV 피처를 자차 이동에 맞게 워핑(warp)한 뒤, GRU로 현재 BEV 피처와 융합합니다.

$$\tilde{\mathcal{F}}_{\text{BEV}}^{t-1} = \text{Warp}(\mathcal{F}_{\text{BEV}}^{t-1}, \boldsymbol{T}) \tag{13}$$

$$\mathcal{F}_{\text{BEV}}^t = \text{LayerNorm}\left(\text{GRU}\left(\tilde{\mathcal{F}}_{\text{BEV}}^{t-1}, \mathcal{F}_{\text{BEV}}^t\right)\right) \tag{14}$$

**수식 설명**

- **$\text{Warp}(\mathcal{F}_{\text{BEV}}^{t-1}, \boldsymbol{T})$**: 이전 프레임의 BEV 피처 맵을 변환 행렬 $\boldsymbol{T}$로 공간 변환 — 자차가 5m 전진했다면 BEV 피처 맵도 5m 앞으로 이동
- **$\text{GRU}(\cdot)$**: Gated Recurrent Unit. 이전 상태(워핑된 과거 BEV)와 현재 입력(현재 BEV)을 게이팅 메커니즘으로 선택적으로 통합
- **$\text{LayerNorm}$**: 학습 안정성을 위한 정규화
- **직관**: 과거의 BEV "지도"를 현재 위치로 당겨와 현재 관찰과 융합 → 폐색 영역도 과거 기억으로 복원 가능

**핵심 개념**

- **Streaming 전략의 장점**: (1) 전파된 hidden state가 모든 과거 이력을 압축 → 단기에 국한되지 않는 장기 연관 가능. (2) 처리 비용이 프레임 수와 무관하게 일정 → 메모리·지연 효율적
- **Auxiliary Transformation Loss**: 쿼리 전파 시 변환 학습을 보조하는 추가 손실. 첫 번째 디코더 레이어 출력 $\hat{P} = \text{Reg}(Q_t)$와 정답의 SmoothL1 거리를 최소화

$$\mathcal{L}_{\text{trans}} = \sum_{j=1}^{N_p} \mathcal{L}_{\text{SmoothL1}}(\hat{P}^j, P_l^j) \tag{12}$$

### 3.5 Matching Cost and Training Loss

**요약**

DETR 방식과 동일하게 이분 매칭(bipartite matching)으로 예측과 정답을 최적 대응시킵니다. 폴리라인 매칭 비용은 MapTR의 순열 그룹 $\Gamma$를 따라 정점 순서의 다양한 방향성을 고려합니다.

**Polyline Matching Cost**

$$\mathcal{L}_{\text{line}}(\hat{\boldsymbol{P}}, \boldsymbol{P}) = \min_{\gamma \in \Gamma} \frac{1}{N_p} \sum_{j=1}^{N_p} \mathcal{L}_{\text{SmoothL1}}(\hat{p}_j, p_{\gamma(j)}) \tag{15}$$

**수식 설명**

- **$\hat{\boldsymbol{P}}$**: 예측된 폴리라인 점 집합
- **$\boldsymbol{P}$**: 정답 폴리라인 점 집합
- **$\Gamma$**: 허용되는 정점 순서의 집합 (폴리라인은 방향이 바뀌어도 같은 요소이므로 순방향·역방향 모두 고려)
- **$\gamma(j)$**: 순열 $\gamma$에 따라 $j$번째 예측 점과 매칭되는 정답 점의 인덱스
- **$\min_{\gamma \in \Gamma}$**: 가장 비용이 낮은 순열 선택 → 최적 매칭 정렬

**최종 학습 손실**

$$\mathcal{L}_{\text{train}} = \lambda_1 \mathcal{L}_{\text{line}} + \lambda_2 \mathcal{L}_{\text{Focal}} + \lambda_3 \mathcal{L}_{\text{trans}} \tag{17}$$

- **$\mathcal{L}_{\text{line}}$**: 폴리라인 좌표 회귀 손실 ($\lambda_1 = 50$)
- **$\mathcal{L}_{\text{Focal}}$**: 클래스 분류 Focal Loss ($\lambda_2 = 5$)
- **$\mathcal{L}_{\text{trans}}$**: 쿼리 변환 보조 손실 ($\lambda_3 = 5$)

---

## Chapter 4: Experiments

### 4.1 벤치마크 공정성 문제 (핵심 기여)

**요약**

이 논문의 중요한 부수적 기여는 기존 벤치마크의 심각한 공정성 문제를 발견한 것입니다.

| 데이터셋 | Split | 학습-검증 위치 겹침 비율 |
|---------|-------|----------------------|
| nuScenes | 기존(공식) | **85%** |
| nuScenes | 새 Split | 11% |
| Argoverse2 | 기존 | **54%** |
| Argoverse2 | 새 Split | **0%** |

맵 데이터는 동일 위치에서 시간이 지나도 거의 변하지 않으므로, 검증 위치의 85%가 학습에 포함되면 모델이 맵을 "암기"해 높은 mAP를 기록하지만 실제로 새로운 환경에서는 동작하지 않습니다. 논문에서는 지리적 겹침이 없는(0%) 새 분할을 제안하고 공개했습니다.

**영향**: 기존 split → 새 split으로 바꾸면 모든 방법의 성능이 약 50% 하락. 이는 기존 벤치마크 결과가 실제 일반화 능력을 반영하지 못했음을 의미합니다.

### 4.4 비교 실험 결과

**Argoverse2 새 Split (Table 1)**

| 범위 | 방법 | AP_ped | AP_div | AP_bound | mAP | FPS |
|------|------|--------|--------|----------|-----|-----|
| 60×30m | VectorMapNet | 35.6 | 34.9 | 37.8 | 36.1 | 5.5 |
| 60×30m | MapTR | 48.1 | 50.4 | 55.0 | 51.1 | **18.0** |
| 60×30m | **StreamMapNet** | **56.9** | **55.9** | **61.4** | **58.1** | 14.2 |
| 100×50m | VectorMapNet | 32.4 | 20.6 | 24.3 | 25.7 | 5.5 |
| 100×50m | MapTR | 46.3 | 36.3 | 38.0 | 40.2 | **18.0** |
| 100×50m | **StreamMapNet** | **60.5** | **44.4** | **48.6** | **51.2** | 14.2 |

- **60m 범위**: StreamMapNet이 MapTR 대비 +7.0 mAP 우세
- **100m 범위**: StreamMapNet이 MapTR 대비 **+11.0 mAP** — 인식 범위가 넓어질수록 격차 확대. Multi-Point Attention + 시간 융합의 효과

**nuScenes 원본 Split (Table 4, 30m 범위)**

| 방법 | mAP |
|------|-----|
| VectorMapNet | 40.9 |
| MapTR | 48.7 |
| BeMapNet | 59.8 |
| **StreamMapNet** | **62.9** |

### 4.5 Ablation Studies (Table 5)

각 컴포넌트를 순차적으로 추가하며 효과 측정 (Argoverse2 새 Split, 100×50m):

| 설정 | mAP |
|------|-----|
| (a) 단일 프레임 baseline (relative predict) | 33.7 |
| (b) − Multi-Point Attention (standard deformable) | 측정 불가 (수렴 실패) |
| (c) + Direct predict (Multi-Point Attention 적용) | 41.7 |
| (d) + Query Propagation (변환 손실 없이) | 42.8 |
| (e) + Transformation loss | 43.7 |
| (f) + BEV Fusion | 46.1 |
| (g) + 이미지 크기 608×608 | 51.2 |

- **Multi-Point Attention 없이 wide range는 수렴 자체가 안 됨** → 가장 중요한 컴포넌트
- Query Propagation과 BEV Fusion 각각 +1.1, +2.4 mAP 기여 → 두 시간 융합이 상호 보완적

### 4.6 정성적 분석

**폐색 시나리오 (Figure 6)**: 시간 $t$에서 흰 트럭이 교차로 왼쪽 시야를 가림. 단일 프레임 모델은 교차로 구조 복원 실패. StreamMapNet은 $t-1$ 프레임 기억을 활용해 올바른 도로 구조를 재구성. 이는 자율주행 안전성에 직결됩니다.

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **Streaming Strategy** | 각 프레임을 개별 처리하면서 압축 메모리(쿼리/BEV 피처)를 다음 프레임으로 전달. Stacking 대비 메모리·지연 비용 일정 |
| **Multi-Point Attention** | 단일 중심 참조점 대신 폴리라인 전체 $N_p$개 점을 참조점으로 사용. 기다란 맵 요소의 비국소 형태를 효율적으로 포착 |
| **Query Propagation** | 이전 프레임의 상위 $k$개 쿼리를 좌표 변환 후 다음 프레임에 전달. 맵 요소의 시간적 지속성을 활용 |
| **BEV Fusion (GRU)** | 이전 BEV 피처를 워핑 후 GRU로 현재 BEV와 융합. 밀집 공간 정보의 시간 연속성 보장 |
| **Geographic Split** | 위치 겹침 없는 학습/검증 분할. 모델의 실제 일반화 능력을 공정하게 측정하기 위해 필수 |
| **Bipartite Matching** | 예측 집합과 정답 집합을 헝가리안 알고리즘으로 최적 1:1 매칭. NMS 불필요한 DETR 방식 학습 |
| **Focal Loss** | 쉬운 샘플보다 어려운 샘플에 더 높은 가중치를 부여하는 분류 손실. 클래스 불균형 완화 |
| **SmoothL1 Loss** | L1과 L2의 장점을 결합한 회귀 손실. 작은 오차에는 L2처럼 부드럽고, 큰 오차에는 L1처럼 이상값에 강건 |

---

## 결론 및 시사점

StreamMapNet은 온라인 HD 맵 구성에 **시간 정보**를 효율적으로 통합한 첫 번째 end-to-end 벡터 맵 학습 모델입니다.

**핵심 성과**:
- Argoverse2 새 Split 100m 범위에서 MapTR 대비 +11.0 mAP
- nuScenes 원본 Split에서 SOTA 62.9 mAP (BeMapNet +3.1)
- 14.2 FPS의 실시간 추론 속도 유지

**실무적 시사점**:
- **합성 데이터 생성**: 시간 연속적 맵 예측이 가능하므로, 프레임 간 맵 일관성이 필요한 합성 시나리오 생성에 유리. 폐색 상황에서도 이전 프레임 기억으로 완전한 맵 복원
- **회귀 테스트**: 기존 nuScenes 공식 split을 사용한 벤치마크 결과는 과적합 가능성이 높음. 지리적 겹침 없는 새 split으로 평가해야 실제 성능 측정 가능
- **VectorMapNet 대비**: VectorMapNet의 두 가지 핵심 한계(시간 정보 없음, 좁은 범위)를 모두 해결. HD Map 연구의 실질적 진보
- 향후 연구 방향: LiDAR 센서 통합, 더 긴 시간 범위의 메모리, 맵 요소 간 위상(topology) 모델링
