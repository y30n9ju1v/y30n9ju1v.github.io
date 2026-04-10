---
title: "DrivingGaussian: Composite Gaussian Splatting for Surrounding Dynamic Autonomous Driving Scenes"
date: 2026-04-10T10:00:00+09:00
draft: false
categories: ["Papers"]
tags: ["Gaussian Splatting", "Autonomous Driving", "3D Reconstruction", "Novel View Synthesis", "LiDAR"]
---

## 개요

- **저자**: Xiaoyu Zhou, Zhiwei Lin, Xiaojun Shan, Yongtao Wang, Deqing Sun, Ming-Hsuan Yang
- **소속**: Wangxuan Institute of Computer Technology, Peking University; Google Research; University of California, Merced
- **발행년도**: 2024 (arXiv:2312.07920v3, 20 Mar 2024)
- **주요 내용**: 자율주행 주변 동적 장면을 효율적으로 재구성하기 위해 Composite Gaussian Splatting 프레임워크(DrivingGaussian)를 제안. 정적 배경은 Incremental Static 3D Gaussians으로, 동적 객체는 Composite Dynamic Gaussian Graph로 각각 모델링한 후 결합하여 고품질 서라운드 뷰 합성을 달성

## 목차

1. Introduction
2. Related Work
3. Method
   - 3.1 Composite Gaussian Splatting
   - 3.2 LiDAR Prior with surrounding views
   - 3.3 Global Rendering via Gaussian Splatting
4. Experiments
   - 4.1 Datasets
   - 4.2 Implementation Details
   - 4.3 Results and Comparisons
   - 4.4 Ablation Study
   - 4.5 Corner Case Simulation
5. Conclusion

---

## 1. Introduction

**요약**

자율주행 시스템은 대규모 동적 3D 장면을 정확히 모델링해야 합니다. 기존 NeRF 기반 방법들은 다음의 한계가 있었습니다:
- 멀티카메라 환경에서 다양한 조명 변화 및 뷰 차이에 취약
- LiDAR를 보조 깊이 감독으로만 활용하고 기하학적 프라이어로는 미활용
- 정적 장면 가정으로 빠르게 움직이는 동적 객체 표현 불가

DrivingGaussian은 이 문제를 해결하기 위해 Composite Gaussian Splatting을 도입합니다:
1. **Incremental Static 3D Gaussians**: 정적 배경을 순차적·점진적으로 재구성
2. **Composite Dynamic Gaussian Graph**: 여러 동적 객체를 개별적으로 재구성한 후 씬에 통합
3. **LiDAR Prior**: 보다 정확한 기하 구조와 멀티카메라 일관성 유지

**핵심 개념**

- **서라운드 뷰 합성(Surrounding View Synthesis)**: 차량 주변 6방향 카메라 이미지를 동시에 고품질로 생성하는 기술
- **Composite Gaussian Splatting**: 정적 배경과 동적 객체를 각각 독립적으로 Gaussian으로 모델링한 후 합성하는 방식
- **NeRF 기반 방법의 한계**: Ray sampling에 의존해 멀티카메라 환경에서 품질 저하 발생

---

## 2. Related Work

**요약**

기존 방법들을 크게 세 범주로 정리합니다:

**NeRF for Bounded Scenes**: MipNeRF, Point-NeRF 등은 제한된 공간에서 좋은 성능을 보이지만 자율주행의 대규모 무한 공간에는 적용 어려움.

**NeRF for Unbounded Scenes**: Urban-NeRF, EmerNeRF 등이 대규모 도시 장면 모델링을 시도하지만, 동적 객체를 충분히 처리하지 못하거나 LiDAR 활용이 제한적.

**3D Gaussian Splatting (3D-GS)**: 명시적 표현으로 빠른 렌더링과 미분가능한 최적화가 가능하지만, 원본 3D-GS는 정적 장면용으로 설계됨.

**Dynamic 3D Gaussian Splatting**: HexPlane, D-nerf 등이 동적 단일 객체 씬으로 확장했으나, 멀티카메라 자율주행 씬에는 적합하지 않음.

**핵심 개념**

- **3D Gaussian Splatting (3D-GS)**: 3D 공간을 수백만 개의 Gaussian 타원체로 표현하고, 이를 2D 이미지 평면에 투영(splatting)해 빠른 렌더링을 가능하게 하는 방법
- **미분가능 렌더링(Differentiable Rendering)**: 렌더링 과정 자체를 미분 가능하게 만들어 역전파로 Gaussian 파라미터를 최적화
- **SfM (Structure-from-Motion)**: 여러 이미지에서 카메라 포즈와 희소 3D 포인트 클라우드를 동시에 추정하는 기법

---

## 3. Method

### 3.1 Composite Gaussian Splatting

**요약**

DrivingGaussian의 핵심 구조는 두 컴포넌트로 구성됩니다.

#### Incremental Static 3D Gaussians

자율주행 데이터는 차량이 이동하면서 넓은 범위를 촬영하므로, 전체 장면을 한 번에 모델링하면 먼 과거 프레임과 현재 프레임의 혼동이 발생합니다. 이를 해결하기 위해:

- LiDAR 깊이 범위를 기준으로 전체 씬을 **N개의 bin**으로 균등 분할
- 첫 번째 bin은 LiDAR prior로 Gaussian 모델 초기화 (식 1)
- 이후 bin들은 이전 bin의 Gaussian을 포지션 프라이어로 활용 (식 2)
- 멀티카메라 이미지의 겹치는 영역을 공동 정렬에 활용

$$p_{b_0}(l|\mu, \Sigma) = e^{-\frac{1}{2}(l-\mu)^\top \Sigma^{-1}(l-\mu)}$$

**수식 설명** — LiDAR 포인트 클라우드로 초기 Gaussian을 생성하는 확률 밀도 함수:
- **$l \in \mathbb{R}^3$**: LiDAR prior의 위치 (3D 공간의 x, y, z 좌표)
- **$\mu$**: LiDAR 포인트들의 평균 위치
- **$\Sigma \in \mathbb{R}^{3\times3}$**: 이방성 공분산 행렬 (Gaussian의 형태와 방향을 결정)
- **직관**: 이 수식은 LiDAR 포인트 주변에 Gaussian 구름을 뿌려 초기 3D 구조를 만드는 것

$$\tilde{P}_{b+1}(G_s) = P_b(G_s) \bigcup (x_{b+1}, y_{b+1}, z_{b+1})$$

**수식 설명** — 이전 bin의 Gaussian을 다음 bin의 프라이어로 활용:
- **$\tilde{P}_{b+1}(G_s)$**: b+1번째 bin을 위한 초기 Gaussian 위치 집합
- **$P_b(G_s)$**: b번째 bin에서 학습된 Gaussian 위치
- **$(x_{b+1}, y_{b+1}, z_{b+1})$**: b+1 구역 내의 새로운 LiDAR 좌표
- **직관**: 이미 알고 있는 장면(이전 bin)을 기반으로 새로운 부분을 점진적으로 확장

Incremental Static Gaussian의 렌더링 색상:

$$\tilde{C}(G_s) = \sum_{b=1}^{N} \Gamma_b \, \alpha_b \, C_b, \quad \Gamma_b = \prod_{i=1}^{b-1}(1 - \alpha_i)$$

**수식 설명** — 알파 합성(alpha compositing)으로 여러 bin의 Gaussian을 합쳐 최종 색상 계산:
- **$\tilde{C}(G_s)$**: 해당 카메라 시점의 최종 렌더링 색상
- **$\alpha_b$**: b번째 bin Gaussian의 불투명도 (0=완전 투명, 1=완전 불투명)
- **$C_b$**: b번째 bin의 색상
- **$\Gamma_b$**: b번째 bin에 도달하는 빛의 투과율 (앞쪽 bin들이 얼마나 막는지)
  - 예: 앞 bin이 불투명도 0.3이면 뒤 bin은 70%만 보임
- **직관**: 빛이 여러 레이어를 통과할 때 각 레이어에서 흡수·반사되는 물리 현상 모방

멀티카메라 정렬을 위한 최적 색상:

$$\hat{C} = \varsigma(G_s) \sum \omega(\tilde{C}(G_s) | R, T)$$

**수식 설명**:
- **$\hat{C}$**: 최적화된 픽셀 색상
- **$\varsigma$**: 미분가능 splatting 함수
- **$\omega$**: 서로 다른 카메라 뷰에 대한 가중치
- **$[R, T]$**: 뷰 정렬을 위한 회전·이동 행렬 (카메라 외부 파라미터)

#### Composite Dynamic Gaussian Graph

동적 객체 처리를 위해 그래프 구조를 도입:

$$H = \langle O, G_d, M, P, A, T \rangle$$

**수식 설명** — 동적 Gaussian 그래프의 구성 요소:
- **$O$**: 인스턴스 객체 집합 (각 차량, 보행자 등)
- **$G_d$**: 각 객체에 대응하는 동적 Gaussian
- **$M$**: 각 객체의 변환 행렬 (위치, 방향)
- **$P$**: 바운딩 박스 중심 좌표
- **$A$**: 바운딩 박스 방향
- **$T$**: 각 객체가 등장하는 시간 스텝 집합

각 동적 객체의 좌표계 변환:

$$m_o^{-1} = R_o^{-1} S_o^{-1}$$

**수식 설명**:
- **$m_o^{-1}$**: 객체의 월드 좌표계 → 객체 좌표계 역변환
- **$R_o^{-1}$**: 회전 역행렬
- **$S_o^{-1}$**: 스케일 역행렬
- **직관**: 각 차량을 자신만의 로컬 좌표계에서 독립적으로 모델링하여, 차량이 이동해도 일관된 형태 유지

가려짐(occlusion) 처리를 위한 동적 객체 불투명도:

$$\alpha_{o,t} = \sum \frac{(p_t - b_o)^2 \cdot \cot \alpha_o}{[\rho(b_o R_{o,t} S_{o,t}) - \rho]^2} \pi_{p_0}$$

**수식 설명** — 카메라에서 가까울수록 더 불투명하게 처리:
- **$\alpha_{o,t}$**: 시각 t에서 객체 o의 조정된 불투명도
- **$p_t$**: 시각 t에서의 카메라 중심
- **$b_o$**: 객체 바운딩 박스 중심
- **$\rho$**: 카메라 중심까지의 거리
- **직관**: 빛의 전파 원리에 따라 가까운 객체가 뒤의 객체를 가리는 현상을 구현

최종 복합 Gaussian 필드:

$$G_{comp} = \sum H \langle O, G_d, M, P, A, T \rangle + G_s$$

**수식 설명**:
- **$G_{comp}$**: 정적 배경과 모든 동적 객체를 합친 최종 Gaussian 필드
- **$G_s$**: Incremental Static 3D Gaussians (정적 배경)
- **$\sum H$**: 그래프의 모든 동적 객체 Gaussian

---

### 3.2 LiDAR Prior with surrounding views

**요약**

원래 3D-GS는 SfM으로 Gaussian을 초기화하지만, 자율주행의 대규모 무한 배경에서는 SfM 포인트가 너무 희소합니다. DrivingGaussian은 LiDAR 포인트 클라우드를 프라이어로 활용합니다:

1. 여러 LiDAR 스윕을 합쳐 완전한 포인트 클라우드 $L$ 생성
2. 각 LiDAR 포인트를 카메라 이미지로 투영하여 색상 할당

LiDAR 포인트 → 이미지 투영:

$$x_p^q = K[R_t^q \cdot l_s + T_t^q]$$

**수식 설명**:
- **$x_p^q$**: 이미지 q에서의 2D 픽셀 좌표
- **$K \in \mathbb{R}^{3\times3}$**: 카메라 내부 파라미터 행렬 (초점거리, 주점 등)
- **$R_t^q, T_t^q$**: 시각 t에서 카메라 q의 회전·이동 행렬 (외부 파라미터)
- **$l_s$**: LiDAR 포인트의 3D 위치
- **직관**: 3D 공간의 LiDAR 포인트를 2D 카메라 이미지에 "찍어서" 색상을 얻는 과정

추가로 **Multi-camera Bundle Adjustment (DBA)**를 적용하여 LiDAR 포인트의 정확도를 높이고 Gaussian 기하 구조를 개선합니다.

**핵심 개념**

- **LiDAR Prior**: LiDAR 센서의 정확한 깊이 정보를 Gaussian 초기화에 활용하여 기하 구조의 정확도 향상
- **Bundle Adjustment**: 카메라 포즈와 3D 포인트를 동시에 최적화하는 기법. 멀티카메라로 확장하면 모든 카메라의 일관성 보장 가능
- **멀티카메라 일관성**: 여러 카메라가 겹치는 영역에서 동일한 3D 구조가 일관되게 나타나야 함

---

### 3.3 Global Rendering via Gaussian Splatting

**요약**

복합 Gaussian 필드 $G_{comp}$를 2D 이미지로 렌더링합니다.

2D 공분산 행렬 투영:

$$\tilde{\Sigma} = JE\Sigma E^\top J^\top$$

**수식 설명**:
- **$\tilde{\Sigma}$**: 2D 이미지 평면에서의 Gaussian 공분산 (타원 모양 결정)
- **$\Sigma$**: 3D 공간에서의 Gaussian 공분산
- **$J$**: 원근 투영의 야코비안 행렬 (3D→2D 변환의 국소 선형 근사)
- **$E$**: 월드→카메라 좌표계 변환 행렬
- **직관**: 3D 타원체를 카메라로 바라볼 때 2D 이미지에서 어떤 타원으로 보이는지 계산

**손실 함수**는 세 가지로 구성:

1. **Tile Structural Similarity Loss** (TSSIM):

$$L_{TSSIM}(\delta) = 1 - \frac{1}{Z} \sum SSIM(\Psi(\hat{C}), \Psi(C))$$

**수식 설명**:
- **$\hat{C}$**: 렌더링된 타일
- **$C$**: 실제 이미지 타일 (ground truth)
- **$\Psi$**: 화면을 $M$개 타일로 분할하는 함수
- **$SSIM$**: 구조적 유사도 지수 (밝기, 대비, 구조를 동시에 비교, 1이 완벽한 일치)
- **직관**: 픽셀 단위 차이뿐 아니라 이미지의 구조적 패턴도 보존되도록 학습

2. **Robust Loss** (이상치 제거):

$$L_{Robust}(\delta) = \kappa(\|I - \hat{I}\|_2)$$

**수식 설명**:
- **$\kappa \in [0,1]$**: 이상치 내성을 조절하는 shape 파라미터
- **$I$**: 실제 이미지, **$\hat{I}$**: 합성 이미지
- **직관**: 일반 L2 손실 대신 Barron robust loss를 사용해 Gaussian 이상치에 덜 민감하게 학습

3. **LiDAR Loss** (기하 감독):

$$L_{LiDAR}(\delta) = \frac{1}{S} \sum \|P(G_{comp}) - L_s\|^2$$

**수식 설명**:
- **$P(G_{comp})$**: 복합 Gaussian의 3D 위치
- **$L_s$**: LiDAR 포인트 위치
- **직관**: Gaussian 위치가 실제 LiDAR 측정값과 가까워지도록 강제하여 정확한 기하 구조 유지

---

## 4. Experiments

### 4.1 데이터셋

- **nuScenes**: 1000개 드라이빙 씬, 6카메라 + 1 LiDAR, 23개 객체 클래스. 6개 도전적 씬의 키프레임(총 320K+ 이미지·포인트 클라우드)을 사용
- **KITTI-360**: 멀티센서 데이터셋, 단안 카메라 씬 검증에 활용

### 4.2 주요 결과

**nuScenes 벤치마크 (Table 1)**:

| Methods | Input | PSNR↑ | SSIM↑ | LPIPS↓ |
|---------|-------|-------|-------|--------|
| EmerNeRF | Images + LiDAR | 26.75 | 0.760 | 0.311 |
| 3D-GS | Images + SfM Points | 26.08 | 0.717 | 0.298 |
| **Ours-S** | Images + SfM Points | **28.36** | **0.851** | **0.256** |
| **Ours-L** | Images + LiDAR | **28.74** | **0.865** | **0.237** |

- Ours-S: SfM 초기화, Ours-L: LiDAR prior 사용
- 기존 최고 성능 대비 PSNR 약 2dB 이상 향상

**KITTI-360 벤치마크 (Table 2)**:

| Methods | PSNR↑ | SSIM↑ |
|---------|-------|-------|
| DNMP | 23.41 | 0.846 |
| **Ours-S** | **25.18** | **0.862** |
| **Ours-L** | **25.62** | **0.868** |

### 4.3 Ablation Study

**초기화 방법 비교 (Table 3)**:
- 랜덤 초기화(Random): PSNR 22.18, SSIM 0.653 — 기하 프라이어 없이 최악
- SfM 초기화(NeRF-1M): PSNR 28.51, SSIM 0.858 — 합리적인 기준점
- **LiDAR-2M**: PSNR **28.78**, SSIM **0.867** — 최고 성능, 기하 구조 가장 정확

**모듈별 기여도 (Table 4)**:
- Composite Dynamic Gaussian Graph(CDGG) 제거 시: PSNR 26.97 → 재구성 품질 큰 하락
- $L_{TSSIM}$ 제거 시: PSNR 27.88 → 세부 텍스처 손실
- $L_{Robust}$ 제거 시: PSNR 28.05 → 이상치로 인한 품질 저하
- $L_{LiDAR}$ 제거 시: PSNR 28.45 → 기하 정확도 감소

---

## 4.5 Corner Case Simulation

DrivingGaussian은 재구성된 Gaussian 필드에 임의 객체를 삽입하여 **코너 케이스 시뮬레이션**을 지원합니다:
- 예: 보행자가 갑자기 넘어지거나, 차량이 앞으로 접근하는 상황
- 시간적 일관성과 센서 간 일관성(멀티카메라)을 유지
- 자율주행 안전 검증을 위한 controllable simulation 가능

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **Composite Gaussian Splatting** | 정적 배경(Incremental Static 3D Gaussians)과 동적 객체(Composite Dynamic Gaussian Graph)를 분리 모델링 후 합성 |
| **Incremental Static 3D Gaussians** | LiDAR 깊이 범위로 씬을 N개 bin으로 나눠 순차적으로 Gaussian을 구성. 시간적 이웃 관계 활용 |
| **Composite Dynamic Gaussian Graph** | 각 동적 객체를 그래프 노드로 표현. 객체별 로컬 좌표계에서 독립 모델링 후 월드 좌표로 변환 |
| **LiDAR Prior** | SfM 대신 LiDAR 포인트를 Gaussian 초기화에 사용. 멀티카메라 Bundle Adjustment로 정확도 추가 향상 |
| **Tile SSIM Loss** | 이미지를 타일로 분할하여 구조적 유사도를 국소적으로 측정, Gaussian의 세부 텍스처 학습 강화 |
| **Robust Loss** | Gaussian 이상치 억제를 위한 강인한 손실 함수 |
| **LiDAR Loss** | Gaussian 위치가 LiDAR 포인트에 근접하도록 강제하는 기하 감독 |
| **Grounded SAM** | 바운딩 박스로부터 픽셀 수준의 동적 객체 마스크를 생성하는 데 활용 |

---

## 결론 및 시사점

DrivingGaussian은 자율주행 씬 재구성에서 다음을 달성합니다:

1. **품질**: nuScenes에서 PSNR 28.74 달성, 기존 최고 대비 약 2dB 이상 향상
2. **멀티카메라 일관성**: 6방향 서라운드 뷰를 동시에 고품질로 합성
3. **동적 객체 처리**: 빠르게 움직이는 차량, 보행자를 고스팅·블러링 없이 정확히 재구성
4. **LiDAR 활용**: 깊이 프라이어로만 쓰던 LiDAR를 Gaussian 초기화에 적극 활용
5. **확장성**: 코너 케이스 시뮬레이션으로 안전 검증까지 지원

**실무적 시사점**: 자율주행 시뮬레이터 구축 시 NeRF 기반 방법 대신 Gaussian Splatting 기반 복합 표현을 채택하면 렌더링 속도와 품질 모두 향상 가능. 특히 LiDAR-카메라 융합 방식이 멀티카메라 일관성 문제 해결에 효과적.
