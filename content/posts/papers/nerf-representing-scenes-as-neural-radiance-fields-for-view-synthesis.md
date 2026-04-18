---
title: "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"
date: 2026-04-10T09:00:00+09:00
draft: false
categories: ["Papers", "Novel View Synthesis"]
tags: ["NeRF", "Novel View Synthesis", "Neural Rendering", "3D Reconstruction"]
---

## 개요

- **저자**: Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng
- **소속**: UC Berkeley, Google Research, UC San Diego
- **발행년도**: 2020 (arXiv:2003.08934)
- **주요 내용**: 희소한 입력 이미지로부터 새로운 시점의 사진처럼 사실적인 이미지를 합성하기 위해, 장면 전체를 5D 연속 함수(Neural Radiance Field)로 표현하고 체적 렌더링(volume rendering)으로 학습하는 방법론

---

## 목차

- Chapter 1: Introduction — 문제 정의와 핵심 아이디어
- Chapter 2: Related Work — 기존 방법론과의 비교
- Chapter 3: Neural Radiance Field Scene Representation — 장면 표현 방식
- Chapter 4: Volume Rendering with Radiance Fields — 체적 렌더링
- Chapter 5: Optimizing a Neural Radiance Field — 최적화 기법
- Chapter 6: Results — 실험 결과
- Chapter 7: Conclusion — 결론

---

## Chapter 1: Introduction

**요약**

NeRF는 카메라 포즈가 알려진 여러 장의 이미지만으로 장면을 5D 연속 함수로 표현하고, 이를 활용해 임의의 새로운 시점에서 사실적인 이미지를 합성하는 방법입니다.

기존의 3D 표현 방식(복셀 그리드, 메쉬 등)은 해상도를 높일수록 메모리와 연산량이 폭발적으로 증가하는 문제가 있습니다. NeRF는 이를 MLP(다층 퍼셉트론) 네트워크의 가중치 자체에 장면 정보를 압축적으로 담음으로써 해결합니다.

파이프라인은 세 단계로 구성됩니다:
1. 카메라 레이(ray)를 따라 3D 점들을 샘플링
2. 각 점의 위치 $(x, y, z)$와 시선 방향 $(\theta, \phi)$를 MLP에 입력해 색상과 밀도를 예측
3. 체적 렌더링(volume rendering)으로 2D 이미지를 합성

**핵심 기여**

- **5D Neural Radiance Field**: 장면을 연속적인 5D 함수로 표현
- **미분 가능한 렌더링**: 체적 렌더링이 미분 가능하므로 RGB 이미지만으로 end-to-end 학습 가능
- **Positional Encoding**: 고주파 세부 표현을 위한 입력 좌표 변환
- **계층적 볼륨 샘플링**: 렌더링 효율을 높이는 coarse-to-fine 샘플링 전략

---

## Chapter 2: Related Work

**요약**

기존 접근법들은 크게 두 계열로 나뉩니다.

**Neural 3D Shape Representations**: 암묵적 함수(implicit function)로 3D 형상을 표현하는 방법들 (SDF, Occupancy Field 등)은 3D 정답 데이터가 필요하거나 단순한 형상에만 적용 가능했습니다.

**View Synthesis & Image-based Rendering**: 라이트 필드 보간, 메쉬 기반 미분 렌더러, 복셀 기반 심층 네트워크 등이 있으나 해상도 한계, 복잡한 기하 표현의 어려움, 대용량 저장 공간 필요 등의 문제가 있었습니다.

NeRF는 연속적인 체적 함수를 MLP로 표현함으로써, 고해상도의 복잡한 장면을 작은 네트워크(~5MB)로 표현하는 새로운 패러다임을 제시합니다.

**핵심 개념**

- **Implicit Neural Representation**: 좌표를 입력으로 받아 해당 위치의 물리량을 출력하는 신경망
- **Differentiable Rendering**: 렌더링 과정이 미분 가능하여 이미지 픽셀 값으로 역전파 가능
- **Voxel Grid vs. Continuous**: 복셀은 이산적(discrete)이므로 해상도에 한계가 있고, MLP 기반 연속 표현은 이론적으로 무한 해상도를 지원

---

## Chapter 3: Neural Radiance Field Scene Representation

**요약**

NeRF는 장면을 다음의 5D 연속 함수로 표현합니다:

$$F_\Theta : (\mathbf{x}, \mathbf{d}) \rightarrow (\mathbf{c}, \sigma)$$

**수식 설명**
- **$\mathbf{x} = (x, y, z)$**: 3D 공간에서의 위치
- **$\mathbf{d} = (\theta, \phi)$**: 시선 방향 (구면 좌표)
- **$\mathbf{c} = (r, g, b)$**: 해당 위치에서 방출되는 색상
- **$\sigma$**: 체적 밀도 (그 위치에 물질이 있을 확률, 클수록 불투명)
- **$F_\Theta$**: 가중치 $\Theta$를 가진 MLP 네트워크

**네트워크 구조**

MLP는 두 부분으로 나뉩니다:

1. **위치 처리부**: $\mathbf{x}$를 8개의 완전연결층(ReLU, 256채널)에 통과시켜 $\sigma$와 256차원 특징 벡터 출력
2. **색상 처리부**: 특징 벡터와 시선 방향 $\mathbf{d}$를 합쳐 1개의 완전연결층(ReLU, 128채널)에 통과시켜 RGB 색상 출력

**핵심 설계 원칙**

- **$\sigma$는 위치만으로 결정**: 물체가 어디에 있는지는 시선 방향과 무관
- **색상은 위치+방향으로 결정**: 같은 지점이라도 보는 방향에 따라 색이 다를 수 있음 (정반사, Non-Lambertian 효과)
- **멀티뷰 일관성**: 여러 시점에서 같은 기하를 예측하도록 강제

---

## Chapter 4: Volume Rendering with Radiance Fields

**요약**

카메라 레이 $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$ (원점 $\mathbf{o}$, 방향 $\mathbf{d}$)에 대한 기대 색상은 다음 연속 적분으로 계산됩니다:

$$C(\mathbf{r}) = \int_{t_n}^{t_f} T(t)\,\sigma(\mathbf{r}(t))\,\mathbf{c}(\mathbf{r}(t), \mathbf{d})\, dt$$

$$\text{where} \quad T(t) = \exp\!\left(-\int_{t_n}^{t} \sigma(\mathbf{r}(s))\,ds\right)$$

**수식 설명**
- **$t_n, t_f$**: 카메라 레이의 근거리, 원거리 바운드
- **$T(t)$**: 누적 투과율 — 레이가 $t_n$에서 $t$까지 아무 입자도 만나지 않고 통과할 확률
  - $\sigma$가 크면 $T$가 빠르게 감소 → 뒤의 물체가 가려짐
- **$\sigma(\mathbf{r}(t))$**: 위치 $t$에서의 밀도 — 빛이 얼마나 많이 흡수/산란되는지
- **$\mathbf{c}(\mathbf{r}(t), \mathbf{d})$**: 위치와 방향에 따른 색상

**이산화 (수치 적분)**

연속 적분을 실제로 계산하기 위해 층별 샘플링(stratified sampling)을 사용합니다. 구간 $[t_n, t_f]$를 $N$개의 bin으로 균등 분할하고 각 bin에서 무작위 샘플링:

$$t_i \sim \mathcal{U}\!\left[t_n + \frac{i-1}{N}(t_f - t_n),\; t_n + \frac{i}{N}(t_f - t_n)\right]$$

이산화된 색상 추정값:

$$\hat{C}(\mathbf{r}) = \sum_{i=1}^{N} T_i \bigl(1 - \exp(-\sigma_i \delta_i)\bigr)\, \mathbf{c}_i, \quad T_i = \exp\!\left(-\sum_{j=1}^{i-1} \sigma_j \delta_j\right)$$

**수식 설명**
- **$\delta_i = t_{i+1} - t_i$**: 인접 샘플 간 거리
- **$1 - \exp(-\sigma_i \delta_i)$**: i번째 구간에서의 불투명도 $\alpha_i$ — 밀도가 높거나 구간이 넓을수록 1에 가까워짐
- **$T_i$**: i번째 샘플까지의 누적 투과율 — 앞의 모든 물질을 통과한 비율
- 이 수식은 알파 합성(alpha compositing)과 수학적으로 동일

**핵심 개념**

- **체적 렌더링의 미분 가능성**: $\hat{C}(\mathbf{r})$은 MLP 파라미터에 대해 미분 가능 → 역전파로 학습 가능
- **물리적 직관**: 카메라에서 출발한 레이가 각 지점에서 색을 수집하되, 앞에 있는 불투명한 물체는 뒤를 가림

---

## Chapter 5: Optimizing a Neural Radiance Field

**요약**

기본 MLP만으로는 복잡한 장면의 고주파 세부 정보를 표현하기 어렵습니다. 두 가지 핵심 개선을 도입합니다.

### 5.1 Positional Encoding (위치 인코딩)

MLP는 저주파 함수를 선호하는 경향이 있어(spectral bias), 날카로운 경계나 세밀한 텍스처 표현이 어렵습니다. 이를 해결하기 위해 입력 좌표를 고차원 공간으로 매핑합니다:

$$\gamma(p) = \bigl(\sin(2^0 \pi p),\, \cos(2^0 \pi p),\, \cdots,\, \sin(2^{L-1} \pi p),\, \cos(2^{L-1} \pi p)\bigr)$$

**수식 설명**
- **$p$**: 원래 입력 좌표값 ([-1, 1]로 정규화)
- **$L$**: 주파수 레벨 수 — 위치 $\mathbf{x}$에 $L=10$, 방향 $\mathbf{d}$에 $L=4$ 사용
- **$2^0, 2^1, \ldots, 2^{L-1}$**: 배수로 증가하는 주파수들
- 이 인코딩으로 MLP 입력이 3차원에서 $2 \times 3 \times L = 60$차원으로 확장됨
- Transformer의 positional encoding과 유사하지만, 순서 정보가 아닌 **고주파 함수 근사**가 목적

**직관**: 낮은 주파수 함수는 거친 형상(전체 모양)을, 높은 주파수 함수는 세밀한 디테일(텍스처, 날카로운 경계)을 표현

### 5.2 Hierarchical Volume Sampling (계층적 볼륨 샘플링)

단순히 레이 위에 균일하게 $N$개 샘플을 두면, 빈 공간이나 가려진 영역(렌더링에 기여하지 않는 위치)에도 많은 샘플을 낭비합니다.

**해결 방법**: Coarse + Fine 두 단계 네트워크

1. **Coarse 단계**: $N_c = 64$ 샘플로 레이를 대략 샘플링, 각 위치의 가중치 $w_i$를 계산:

$$\hat{C}_c(\mathbf{r}) = \sum_{i=1}^{N_c} w_i c_i, \qquad w_i = T_i(1 - \exp(-\sigma_i \delta_i))$$

2. **Fine 단계**: Coarse의 가중치를 정규화해 확률분포(PDF)로 만들고, 이 분포에서 $N_f = 128$ 샘플을 추가 샘플링 (중요한 곳에 더 집중)
3. 총 $N_c + N_f = 192$ 샘플로 Fine 네트워크 평가, 최종 색상 $\hat{C}_f(\mathbf{r})$ 계산

**학습 손실 함수**

$$\mathcal{L} = \sum_{\mathbf{r} \in \mathcal{R}} \left[\left\|\hat{C}_c(\mathbf{r}) - C(\mathbf{r})\right\|_2^2 + \left\|\hat{C}_f(\mathbf{r}) - C(\mathbf{r})\right\|_2^2\right]$$

**수식 설명**
- **$\mathcal{R}$**: 배치 내 카메라 레이 집합 (배치 크기 4096 레이)
- **$C(\mathbf{r})$**: 정답 픽셀 색상
- **$\hat{C}_c, \hat{C}_f$**: Coarse/Fine 네트워크의 예측 색상
- Coarse 손실도 포함시켜 Coarse 네트워크의 가중치 분포가 의미있게 학습되도록 함

### 5.3 구현 세부사항

- **옵티마이저**: Adam ($\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-7}$)
- **학습률**: $5 \times 10^{-4}$에서 시작, $5 \times 10^{-5}$까지 지수 감소
- **수렴**: 단일 NVIDIA V100 GPU에서 100k~300k 이터레이션 (약 1~2일)
- **카메라 포즈**: 합성 데이터는 정답 사용, 실제 이미지는 COLMAP SfM 패키지로 추정

---

## Chapter 6: Results

**요약**

NeRF는 세 가지 데이터셋에서 기존 방법들을 정량적·정성적으로 능가합니다.

### 6.1 데이터셋

- **Diffuse Synthetic 360°** (DeepVoxels): 간단한 기하의 4개 Lambertian 물체, 512×512 픽셀
- **Realistic Synthetic 360°**: 복잡한 비-Lambertian 재질의 8개 물체, 800×800 픽셀
- **Real Forward-Facing** (LLFF): 핸드헬드 폰으로 촬영한 실제 8개 장면, 1008×756 픽셀

### 6.2 정량적 비교

| 방법 | Diffuse Synthetic PSNR↑ | Realistic Synthetic PSNR↑ | Real Forward PSNR↑ |
|------|------------------------|--------------------------|-------------------|
| SRN  | 33.20                  | 22.26                    | 22.84             |
| NV   | 29.62                  | 26.05                    | -                 |
| LLFF | 34.38                  | 24.88                    | 24.13             |
| **NeRF (Ours)** | **40.15**     | **31.01**                | **26.50**         |

- PSNR, SSIM (높을수록 좋음), LPIPS (낮을수록 좋음) 모두에서 최고 성능
- 모델 크기: ~5MB (LLFF 대비 3000배 압축)

### 6.3 비교 방법과의 차별점

- **SRN**: 레이당 하나의 깊이·색상 예측 → 과도하게 부드러운 결과
- **Neural Volumes (NV)**: 128³ 복셀 그리드 → 세밀한 기하 표현 한계
- **LLFF**: 빠른 처리(10분)지만 15GB+ 저장 공간, 뷰 간 블렌딩 아티팩트

### 6.4 Ablation Study

핵심 설계 선택의 기여도 (Realistic Synthetic 데이터셋 기준):

| 구성 | PSNR↑ |
|------|-------|
| 기본 MLP (PE/VD/H 없음) | 26.67 |
| + View Dependence | 28.77 |
| + Positional Encoding | 27.66 |
| + Hierarchical Sampling | 30.06 |
| **완전한 모델 (PE+VD+H)** | **31.01** |

→ Positional Encoding의 기여가 가장 크고, View Dependence, Hierarchical Sampling 순으로 중요

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **Neural Radiance Field (NeRF)** | 5D 입력(위치+방향)을 받아 색상과 밀도를 출력하는 MLP로 표현된 연속 장면 표현 |
| **Volume Density $\sigma$** | 공간상의 한 점에서 레이가 종료될 확률 밀도 — 높을수록 불투명 |
| **Transmittance $T(t)$** | 레이가 특정 지점까지 아무것도 만나지 않고 통과할 누적 확률 |
| **Alpha Compositing** | 투명도를 고려해 여러 레이어의 색을 합성하는 방법 — NeRF의 이산화 공식과 동일 |
| **Positional Encoding** | 입력 좌표를 사인/코사인의 다중 주파수로 변환해 고주파 함수 학습을 돕는 기법 |
| **Stratified Sampling** | 구간을 균등 분할 후 각 bin에서 무작위 샘플링 — 연속 표현을 유지하면서 이산 계산 가능 |
| **Hierarchical Sampling** | Coarse 네트워크로 중요 영역을 파악하고 Fine 네트워크에서 그 영역에 집중 샘플링 |
| **View-Dependent Appearance** | 보는 방향에 따라 색이 달라지는 현상 (정반사, 금속 등) — $\mathbf{d}$를 색상 예측에 추가로 반영 |
| **Non-Lambertian** | 보는 방향에 무관하게 색이 일정한 Lambertian 모델과 달리, 방향에 따라 색이 달라지는 재질 특성 |
| **COLMAP (SfM)** | 여러 이미지에서 카메라 포즈를 자동으로 추정하는 Structure-from-Motion 패키지 |

---

## 결론 및 시사점

**논문의 결론**

NeRF는 장면을 5D Neural Radiance Field로 표현하고 체적 렌더링으로 학습함으로써, 이전의 MLP 기반 암묵적 표현 및 복셀 기반 심층 네트워크를 모두 능가하는 새로운 view synthesis 패러다임을 제시했습니다.

**실용적 시사점**

1. **메모리 효율**: 장면 전체를 단 ~5MB의 네트워크 가중치로 압축 — 기존 방법 대비 수천 배 압축
2. **표현력**: 고해상도의 복잡한 기하와 비-Lambertian 재질까지 표현 가능
3. **학습 데이터**: 카메라 포즈가 있는 RGB 이미지만으로 학습 — 3D 스캔이나 depth sensor 불필요

**한계점 및 미래 방향**

- **학습 속도**: 장면 하나에 1~2일 소요 (이후 연구인 Instant-NGP, 3DGS 등이 이를 크게 개선)
- **장면별 학습**: 새로운 장면마다 처음부터 재학습 필요
- **동적 장면**: 정적 장면만 처리 가능
- **해석 가능성**: MLP 가중치에 인코딩된 표현은 직접 분석이 어려움
- **미래 비전**: NeRF 기반 그래픽스 파이프라인 — 실제 촬영된 물체/장면들의 NeRF를 조합해 새로운 콘텐츠 생성
