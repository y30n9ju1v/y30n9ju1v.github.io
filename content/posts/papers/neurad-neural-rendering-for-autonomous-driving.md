---
title: "NeuRAD: Neural Rendering for Autonomous Driving"
date: 2026-04-19T11:00:00+09:00
draft: false
categories: ["Papers"]
tags: ["NeRF", "Neural Rendering", "Autonomous Driving", "Novel View Synthesis", "LiDAR Simulation", "Sensor Simulation"]
---

## 개요
- **저자**: Adam Tonderski, Carl Lindström, Georg Hess, William Ljungbergh, Lennart Svensson, Christoffer Petersson (Zenseact, Chalmers University, Linköping University, Lund University)
- **발행년도**: 2024 (arXiv:2311.15260v3, 18 Apr 2024)
- **주요 내용**: 자율주행(AD) 동적 장면에 특화된 Neural Radiance Field 기반 Novel View Synthesis 방법론. 카메라와 LiDAR를 통합 모델링하여 센서 현실적인 합성 데이터를 생성하고, 에고 차량 및 액터의 포즈를 변경하여 새로운 시나리오를 생성할 수 있음.

## 한계 극복

- **기존 한계 1 — 긴 학습 시간**: 기존 NeRF 기반 AD 방법(S-NeRF 등)은 17시간 이상 학습이 필요해 실용성이 낮음
- **기존 한계 2 — 데이터셋 특화**: MARS, UniSim 등 기존 방법들은 특정 데이터셋 가정(밀집 깊이 맵, 시맨틱 레이블 등)에 의존해 범용성이 부족함
- **기존 한계 3 — LiDAR 모델링 부실**: 이전 연구들은 ray drop(레이저 빔이 반환 없이 사라지는 현상), beam divergence(빔 발산) 등 센서 특성을 제대로 모델링하지 못함
- **기존 한계 4 — rolling shutter 미고려**: 고속 이동 차량의 카메라 및 LiDAR에서 발생하는 rolling shutter 효과를 대부분 무시함
- **이 논문의 접근 방식**: iNGP 기반의 단일 통합 네트워크로 정적/동적 요소를 actor-aware hash encoding으로 구분하고, 센서별 물리적 특성을 명시적으로 모델링하여 5개 공개 데이터셋에서 dataset-specific 튜닝 없이 SoTA 달성

## 목차
- Section 1: Introduction
- Section 2: Related Work
- Section 3: Method
  - 3.1 Scene Representation and Sensor Modeling
  - 3.2 Extending Neural Feature Fields
  - 3.3 Automotive Data Modeling
  - 3.4 Losses
  - 3.5 Implementation Details
- Section 4: Experiments
- Section 5: Conclusions

---

## Section 1: Introduction

**요약**

Neural Radiance Fields(NeRF)는 3D 장면으로부터 새로운 시점의 이미지를 합성하는 기술로, 자율주행 분야에서 큰 잠재력을 가집니다. 편집 가능한 디지털 클론을 만들어 안전-임계 시나리오를 물리적 위험 없이 탐색하거나, closed-loop 시뮬레이터의 corner case 데이터를 생성하는 데 활용할 수 있습니다. 그러나 기존 방법들은 긴 학습 시간, 밀집 시맨틱 감독 필요성, 또는 범용성 부족 문제를 가집니다. NeuRAD는 이를 해결하기 위해 대규모 자율주행 장면을 처리하고 여러 데이터셋에서 즉시(out-of-the-box) 동작하도록 설계된 편집 가능한 오픈소스 NVS 방법론입니다.

**핵심 개념**
- **Neural Radiance Field (NeRF)**: 신경망이 3D 장면의 암묵적 표현을 학습하여 임의의 시점에서 렌더링 가능한 기법
- **Novel View Synthesis (NVS)**: 학습에 사용되지 않은 새로운 카메라 포즈에서 이미지를 생성하는 것
- **Closed-loop simulation**: 자율주행 시스템의 출력이 다시 시뮬레이터에 피드백되어 연속적으로 시나리오가 전개되는 시뮬레이션
- **Actor**: 도로 위의 동적 물체(차량, 보행자 등), bounding box와 SO(3) 포즈 집합으로 정의됨

---

## Section 2: Related Work

**요약**

관련 연구는 세 갈래로 분류됩니다. (1) NeRF 자체의 발전 계보: Instant-NGP의 multiresolution hash grid, Zip-NeRF의 anti-aliasing 기법 등. (2) 자동차 데이터에 NeRF를 적용한 연구: NSG, PNF, S-NeRF, Block-NeRF, SUDS 등. (3) Closed-loop 시뮬레이션을 위한 연구: MARS, UniSim. NeuRAD는 UniSim과 가장 유사하지만, LiDAR를 직접 포인트 클라우드로 지원하고, upsampling CNN 기반 anti-aliasing 전략을 도입하여 차별화됩니다.

**핵심 개념**
- **Instant-NGP (iNGP)**: 학습 가능한 multiresolution hash grid로 NeRF의 학습/추론 시간을 대폭 단축한 방법
- **MARS**: 정적 배경과 동적 액터를 위한 독립 NeRF 모듈을 혼합하는 모듈식 AD 시뮬레이터
- **UniSim**: PandaSet 전면 카메라와 360° LiDAR에 특화된 Neural closed-loop sensor simulator
- **S-NeRF**: mip-NeRF 360을 자동차 street view에 확장, 각 액터를 별도 MLP로 모델링하나 학습에 하루 이상 소요

---

## Section 3: Method

**요약**

NeuRAD의 핵심은 **Neural Feature Field(NFF)**를 기반으로 정적 배경과 동적 액터를 단일 네트워크로 통합하는 것입니다. Actor-aware hash encoding을 통해 두 요소를 구분하고, 카메라와 LiDAR 각각의 물리적 특성(rolling shutter, beam divergence, ray drop)을 명시적으로 모델링합니다.

### 3.1 Scene Representation and Sensor Modeling

**핵심 개념**
- **Neural Feature Field (NFF)**: $(s, \mathbf{f}) = \text{NFF}(\mathbf{x}, t, \mathbf{d})$ — 위치 $\mathbf{x}$, 시간 $t$, 시선 방향 $\mathbf{d}$를 입력으로 암묵적 기하학(SDF $s$)과 특징 벡터 $\mathbf{f}$를 출력하는 함수
- **Volume rendering**: 레이를 따라 샘플을 적분하여 최종 특징/색상/깊이를 계산

**수식: Alpha Compositing (볼륨 렌더링)**

$$\mathbf{f}(\mathbf{r}) = \sum_{i=1}^{N_r} w_i \mathbf{f}_i, \quad w_i = \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j)$$

**수식 설명**

레이 $\mathbf{r}$ 위의 $N_r$개 샘플을 alpha compositing으로 합산하여 최종 특징 벡터를 구합니다:
- **$\mathbf{f}(\mathbf{r})$**: 레이의 최종 렌더링 특징 벡터 (카메라면 색상, LiDAR면 반사도 등으로 디코딩됨)
- **$\mathbf{f}_i$**: $i$번째 샘플 위치의 특징
- **$w_i$**: $i$번째 샘플의 기여 가중치 (앞 샘플들의 불투명도를 통과한 빛의 비율 × 현재 불투명도)
- **$\alpha_i$**: $i$번째 샘플의 불투명도. SDF 값 $s_i$로부터 $\alpha_i = 1/(1 + e^{\beta s_i})$로 계산 ($\beta$는 학습 파라미터)
- **$\prod_{j=1}^{i-1}(1-\alpha_j)$**: $i$번째 샘플에 도달하기까지 앞 샘플들을 통과한 투과율

**카메라 모델링**: 레이들을 볼륨 렌더링하여 저해상도 특징 맵 $\mathcal{F} \in \mathbb{R}^{H_f \times W_f \times N_f}$를 만들고, CNN으로 업샘플링하여 최종 이미지 $\mathcal{I} \in \mathbb{R}^{H_I \times W_I \times 3}$ 생성. 이를 통해 쿼리 레이 수를 대폭 줄임.

**LiDAR 모델링**: 각 LiDAR 포인트마다 레이를 쏘아 기대 깊이 $\mathbb{E}[D_l(\mathbf{r})] = \sum_{i=1}^{N_r} w_i \tau_i$로 깊이를 예측하고, 특징을 MLP에 통과시켜 반사도(intensity)를 예측.

### 3.2 Extending Neural Feature Fields — Scene Composition

**핵심 개념**
- **Actor-aware hash encoding**: 샘플 $(\mathbf{x}, t)$이 액터의 bounding box 내부에 있으면 액터 좌표계로 변환 후 4D hash grid(4번째 차원 = 액터 인덱스)를 조회하고, 외부면 정적 multiresolution hash grid를 조회. 하나의 네트워크로 모든 액터를 병렬 처리 가능.
- **Unbounded static scene**: 정적 배경은 MipNeRF-360의 contraction 기법으로 무한 공간을 단일 그리드에 표현

### 3.3 Automotive Data Modeling

**수식: Frustum Volume (스케일 인식 다운웨이팅)**

$$V_i = \frac{\tau_{i+1} - \tau_i}{3}\left(A_i + \sqrt{A_i A_{i+1}} + A_{i+1}\right)$$

**수식 설명**

레이의 각 구간을 원뿔 절두체(frustum)로 모델링하여 스케일에 따른 다운웨이팅에 사용합니다:
- **$V_i$**: $i$번째 구간의 절두체 부피
- **$\tau_i, \tau_{i+1}$**: 구간의 시작/끝 깊이
- **$A_i, A_{i+1}$**: 구간 끝점에서의 단면적 (카메라는 pixel size × 깊이², LiDAR는 beam divergence로 결정)
- 부피 $V_i$가 클수록 ($= $ 원거리 샘플) hash grid 특징의 기여를 줄여 aliasing 방지

**핵심 개념**
- **Multiscale anti-aliasing**: Zip-NeRF에서 영감을 받아, hash grid 해상도 $l$에서 가중치 $\omega_{i,l} = \min(1, \frac{1}{n_l V_i^{1/3}})$로 downweighting — 셀 크기 대비 절두체 크기가 클수록 더 억제
- **Two-round proposal sampling**: 가벼운 NFF로 weight 분포를 먼저 예측한 후, 중요 위치에 집중 샘플링하여 효율적 렌더링
- **Rolling shutter 모델링**: 카메라/LiDAR의 각 레이에 개별 시간을 할당하고, 그 시간의 추정 모션에 따라 레이 원점을 선형 보간. 고속 주행 시 수 미터 오차 발생을 방지
- **Ray drop 모델링**: 반환 없는 LiDAR 빔(투명 표면, 거울 등)을 데이터에서 학습하는 확률 예측기로 모델링. 기존 물리 기반 방법의 한계를 데이터 기반으로 극복
- **Sensor embeddings**: 센서별(카메라 단위) appearance embedding을 학습하여 서로 다른 노출/파라미터를 가진 카메라를 통합 처리

### 3.4 Losses

**수식: Image Loss**

$$\mathcal{L}^{\text{image}} = \frac{1}{N_p} \sum_{i=1}^{N_p} \lambda^{\text{rgb}} \mathcal{L}_i^{\text{rgb}} + \lambda^{\text{vgg}} \mathcal{L}_i^{\text{vgg}}$$

- **$\mathcal{L}^{\text{rgb}}$**: 예측/실제 픽셀 값의 제곱 오차
- **$\mathcal{L}^{\text{vgg}}$**: VGG 특징 공간의 거리 (perceptual loss) — 고주파 디테일 보존
- **$N_p$**: 패치 수 (patch-wise 학습)

**수식: LiDAR Loss**

$$\mathcal{L}^{\text{lidar}} = \frac{1}{N} \sum_{i=1}^{N} (\lambda^d \mathcal{L}_i^d + \lambda^{\text{int}} \mathcal{L}_i^{\text{int}} + \lambda^{p_d} \mathcal{L}_i^{p_d} + \lambda^w \mathcal{L}_i^w)$$

- **$\mathcal{L}^d$**: 깊이 예측 오차 (squared error)
- **$\mathcal{L}^{\text{int}}$**: 반사도(intensity) 예측 오차
- **$\mathcal{L}^{p_d}$**: ray drop 확률 예측을 위한 binary cross-entropy
- **$\mathcal{L}^w$**: 빈 공간 밀도 패널티 — 센서 관측 거리 밖 샘플에 weight decay 적용

---

## Section 4: Experiments

**요약**

5개 공개 AD 데이터셋(nuScenes, PandaSet, Argoverse 2, KITTI, ZOD)에서 동일한 하이퍼파라미터로 평가. 단일 Nvidia A100으로 약 1시간 학습. 학습 속도(1시간 vs S-NeRF의 17시간)와 성능을 동시에 개선.

**Novel View Synthesis (카메라) 결과**

| 데이터셋 | 방법 | PSNR↑ | SSIM↑ | LPIPS↓ |
|---------|------|-------|-------|--------|
| PandaSet FC | NeuRAD (ours) | 26.58 | 0.778 | 0.190 |
| PandaSet FC | NeuRAD-2x | **26.84** | **0.801** | **0.148** |
| KITTI | NeuRAD | 27.00 | 0.795 | 0.082 |
| KITTI | NeuRAD-2x | **27.91** | **0.822** | **0.066** |

**LiDAR Novel View Synthesis 결과** (PandaSet FC 기준, UniSim 대비)

| 방법 | Depth↓ (m) | Intensity↓ | Drop acc.↑ | Chamfer↓ |
|-----|-----------|-----------|-----------|---------|
| UniSim* | 0.07 | 0.085 | 91.0 | 11.2 |
| NeuRAD | **0.01** | **0.062** | **96.2** | **1.6** |

**Novel scenario generation (FID)**: NeuRAD는 Lane 2m/3m shift 및 Actor shift에서 UniSim 대비 낮은 FID(더 현실적인 이미지)를 달성, 특히 최적화 포즈 사용 시 추가 향상.

**Ablation 주요 결과**:
- CNN decoder 제거 시 PSNR 1.0 감소, 처리 속도 0.1 MP/s로 급감
- Rolling shutter 제거 시 PSNR 0.5 감소, 빠른 장면에서 심각한 블러 발생
- Ray drop 모델링 제거 시 Chamfer 거리 대폭 증가 (1.6 → 100.6)
- SDF를 NeRF-like density로 교체해도 성능 거의 동일 → 필요에 따라 선택 가능

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **Actor-aware hash encoding** | 정적/동적 요소를 단일 네트워크에서 처리; bounding box 내부면 actor 좌표계로 변환 후 4D hash grid 조회 |
| **Ray drop** | LiDAR 빔이 반환 신호 없이 사라지는 현상; 투명 표면, 거울 등에서 발생. 데이터 기반으로 확률 예측 |
| **Rolling shutter** | 카메라/LiDAR가 행별로 순차 캡처할 때 고속 이동으로 생기는 왜곡. 레이별 시간을 달리 부여하여 보정 |
| **Beam divergence** | LiDAR 레이저 빔이 거리에 따라 퍼지는 현상. frustum 볼륨으로 모델링하여 anti-aliasing에 활용 |
| **Sensor embedding** | 센서(카메라)별 appearance 표현; 다른 노출/ISO 등 캡처 조건 차이를 흡수 |
| **Proposal sampling** | 가벼운 NFF로 weight 분포 예측 → 중요 위치 집중 샘플링으로 효율적 렌더링 |
| **SDF vs density** | SDF(부호 거리 함수)는 명확한 표면 추출 가능; density는 fog·투명 표면에 강건. NeuRAD는 교체 가능하도록 설계 |

---

## 결론 및 시사점

NeuRAD는 자율주행을 위한 실용적인 Neural Simulation 파운데이션을 제공합니다. 주요 시사점:

1. **범용성**: 5개 데이터셋에서 동일 하이퍼파라미터로 SoTA — 실제 운영 환경 적용 가능성이 높음
2. **센서 현실성**: LiDAR ray drop, beam divergence, rolling shutter를 통합 모델링하여 sim-to-real gap 감소
3. **편집 가능성**: 액터 위치/방향 변경, 에고 차량 레인 변경 등으로 다양한 시나리오 생성 → 회귀 테스트 및 코너 케이스 데이터 증강에 직접 활용 가능
4. **오픈소스**: Nerfstudio 기반으로 공개되어 후속 연구 확장 용이
5. **한계**: 액터를 rigid body로 가정(보행자 변형 불가), 악천후(비/눈) 조건 모델링 미지원
