---
title: "DIFIX3D+: Improving 3D Reconstructions with Single-Step Diffusion Models"
date: 2026-04-10T09:00:00+09:00
draft: false
categories: ["Papers", "Novel View Synthesis"]
tags: ["3D Gaussian Splatting", "NeRF", "Novel View Synthesis", "Diffusion", "3D Reconstruction"]
---

## 개요
- **저자**: Jay Zhangjie Wu, Yuxuan Zhang, Haithem Turki, Xuanchi Ren, Jun Gao, Mike Zheng Shou, Sanja Fidler, Zan Gojcic, Huan Ling
- **소속**: NVIDIA, National University of Singapore, University of Toronto, Vector Institute
- **발행년도**: 2025 (arXiv:2503.01774v1, 3 Mar 2025)
- **주요 내용**: 단일 단계(single-step) 확산 모델(DIFIX)을 활용하여 NeRF 및 3DGS 기반 3D 재구성의 아티팩트를 제거하고 품질을 향상시키는 파이프라인 DIFIX3D+를 제안

## 목차
- Chapter 1: Introduction
- Chapter 2: Related Work
- Chapter 3: Background (NeRF & 3DGS)
- Chapter 4: DIFIX - 단일 단계 확산 모델로 3D 재구성 향상
  - 4.1 DIFIX: 사전학습 확산 모델을 3D Artifact Fixer로
  - 4.2 DIFIX3+: 확산 사전(Priors)으로 NVS 향상
- Chapter 5: 실험
- Chapter 6: 결론

---

## Chapter 1: Introduction

**요약**

Neural Radiance Fields(NeRF)와 3D Gaussian Splatting(3DGS)은 새로운 시점 합성(Novel View Synthesis, NVS) 분야를 혁신했지만, 극단적인 시점에서의 사진처럼 사실적인 렌더링은 여전히 어렵습니다. 특히 입력 카메라 포즈와 멀리 떨어진 시점을 렌더링할 때 아티팩트(artifact)가 발생합니다.

이 논문은 **DIFIX3D+**를 제안합니다. 핵심 아이디어는 다음과 같습니다:
- 인터넷 규모 데이터로 학습된 2D 생성 모델의 강력한 사전(prior) 지식을 활용
- 3D 재구성 중 **학습 데이터 정제**에 DIFIX를 사용하여 의사 학습 뷰(pseudo-training views)를 개선
- 추론 시 **실시간 후처리**로 아티팩트 제거

기존 방법들과의 차별점은 매 학습 스텝마다 확산 모델을 쿼리하지 않아 **>10× 빠른** 추론이 가능하며, NeRF와 3DGS 모두에 적용 가능한 **단일 범용 모델**입니다.

**핵심 개념**
- **아티팩트(Artifact)**: 3D 재구성 시 관측이 부족한 영역에서 발생하는 시각적 오류 (흐릿함, 잘못된 구조, 부자연스러운 텍스처 등)
- **Novel View Synthesis (NVS)**: 기존 카메라로 촬영한 이미지로부터 새로운 시점의 이미지를 생성하는 기술
- **단일 단계 확산 모델**: 일반 확산 모델은 수백~수천 스텝이 필요하지만, 단일 단계(single-step)로 추론하여 실시간에 가까운 속도를 달성

---

## Chapter 2: Related Work

**요약**

관련 연구는 크게 세 갈래로 나뉩니다:

**3D 재구성 불일치 개선**: 노이즈 카메라 포즈 최적화, 조명 변화 처리, 일시적 오클루전 처리 등의 방법들이 있지만, 이는 아티팩트를 완전히 제거하지 못합니다.

**NVS를 위한 기하학적 사전(Geometric Priors)**: 깊이 맵, 법선 맵 등을 활용하여 희소 입력 설정에서 렌더링 품질을 높이는 방법들이 있으나, 밀도 높은 캡처에서는 개선이 미미합니다.

**NVS를 위한 생성 사전(Generative Priors)**: GAN, 확산 모델 등 대규모 데이터로 학습된 생성 모델을 활용하는 방법들입니다. 특히 확산 모델은 인터넷 규모 데이터셋에서 강력한 사전 지식을 학습하며, 최소 파인튜닝으로 새 뷰를 생성하거나 3D 표현 최적화를 가이드할 수 있습니다. 그러나 매 학습 스텝마다 확산 모델을 호출하면 학습 속도가 크게 저하됩니다.

**핵심 개념**
- **Deceptive-NeRF**: 확산 모델로 생성한 의사 관측(pseudo-observations)으로 NeRF 재구성을 향상시키는 기법. DIFIX3D+와 유사하지만 두 가지 차이점: (i) 점진적인 3D 업데이트 파이프라인, (ii) 렌더 타임 신경 향상기 역할

---

## Chapter 3: Background

**요약**

**NeRF (Neural Radiance Fields)**: 좌표 기반 MLP를 활용하여 3D 장면을 implicit하게 표현합니다. 임의의 3D 위치에서 색상과 밀도를 쿼리할 수 있으며, 볼륨 렌더링을 통해 이미지를 생성합니다.

**볼륨 렌더링 공식:**

$$\mathcal{C}(\mathbf{p}) = \sum_{i=1}^{N} \alpha_i c_i \prod_{j=1}^{i-1}(1 - \alpha_j) \tag{1}$$

**수식 설명**
이 수식은 카메라 광선을 따라 여러 포인트의 색상을 누적하여 최종 픽셀 색상을 계산합니다:
- **$\mathcal{C}(\mathbf{p})$**: 광선 $\mathbf{p}$의 최종 렌더링 색상 (픽셀에 보이는 색)
- **$c_i$**: $i$번째 샘플 포인트의 색상
- **$\alpha_i = 1 - \exp(-\sigma_i \delta_i)$**: $i$번째 포인트의 불투명도
  - $\sigma_i$: 밀도(density), $\delta_i$: 샘플 간격
- **$\prod_{j=1}^{i-1}(1 - \alpha_j)$**: $i$번째 포인트까지 도달하는 빛의 투과율 (앞의 모든 포인트를 통과한 비율)
- 직관: "각 포인트의 색상 × 그 포인트의 불투명도 × 앞에서 얼마나 빛이 살아남았는지"를 모두 더한 값

**3DGS (3D Gaussian Splatting)**: 장면을 수백만 개의 3D Gaussian 입자로 표현합니다. 각 Gaussian은 위치 $\boldsymbol{\mu}$, 회전 $\mathbf{r}$, 스케일 $\mathbf{s}$, 불투명도 $\eta$, 색상 $\mathbf{c}$를 가집니다.

**3DGS 불투명도 계산:**

$$\alpha_i = \eta_i \exp\left[-\frac{1}{2}(\mathbf{p} - \boldsymbol{\mu}_i)^\top \boldsymbol{\Sigma}_i^{-1}(\mathbf{p} - \boldsymbol{\mu}_i)\right] \tag{2}$$

**수식 설명**
이 수식은 특정 픽셀 위치 $\mathbf{p}$에서 $i$번째 Gaussian이 기여하는 불투명도입니다:
- **$\alpha_i$**: $i$번째 Gaussian의 해당 픽셀에서의 불투명도
- **$\eta_i$**: 전체 불투명도 (학습 가능한 파라미터)
- **$\boldsymbol{\Sigma}_i = \mathbf{R}\mathbf{S}\mathbf{S}^T\mathbf{R}^T$**: 공분산 행렬 (Gaussian의 모양과 방향을 결정)
  - $\mathbf{R}$: 회전 행렬 ($\mathbf{R} \in SO(3)$), $\mathbf{S}$: 스케일 행렬 ($\mathbf{S} \in \mathbb{R}^{3\times3}$)
- **$\exp[-\frac{1}{2}(\mathbf{p} - \boldsymbol{\mu}_i)^\top \boldsymbol{\Sigma}_i^{-1}(\mathbf{p} - \boldsymbol{\mu}_i)]$**: 픽셀이 Gaussian 중심에서 얼마나 떨어져 있는지에 따른 감쇠
  - 중심에 가까울수록 1에 가깝고, 멀어질수록 0에 가까움

**핵심 개념**
- **타일 기반 래스터화(tile-based rasterization)**: 3DGS는 미분 가능한 래스터화로 픽셀당 기여하는 Gaussian 수 $N$을 결정
- NeRF와 3DGS 모두 동일한 볼륨 렌더링 공식(Eq. 1)을 공유

---

## Chapter 4: DIFIX - 단일 단계 확산 모델로 3D 재구성 향상

### 4.1 DIFIX: 사전학습 확산 모델을 3D Artifact Fixer로

**요약**

DIFIX는 SD-Turbo 기반의 U-Net 구조로, 노이즈가 있는 렌더링 이미지를 입력받아 아티팩트가 제거된 깨끗한 이미지를 출력합니다. 핵심은 **크로스-뷰 참조 혼합 레이어(cross-view reference mixing layer)**로, 여러 참조 뷰의 정보를 활용하여 3D 일관성을 유지합니다.

**DIFIX 아키텍처:**
- **입력**: 노이즈가 있는 렌더링 이미지 + 참조 뷰 이미지
- **출력**: 아티팩트가 제거된 향상된 이미지
- **구조**: 동결된 VAE 인코더 + LoRA로 파인튜닝된 디코더
- **참조 혼합 레이어**: U-Net의 attention 연산 후 시점 차원(view dimension)을 따라 rearrange하여 크로스-뷰 의존성을 포착

**손실 함수:**

$$\mathcal{L} = \mathcal{L}_{\text{Recon}} + \mathcal{L}_{\text{LPIPS}} + 0.5\mathcal{L}_{\text{Gram}} \tag{3}$$

$$G_l(I) = \phi_l(I)^\top \phi_l(I) \tag{5}$$

**손실 함수 설명**
총 손실은 세 가지 항으로 구성됩니다:
- **$\mathcal{L}_{\text{Recon}}$**: 재구성 손실 - L2 차이(픽셀 단위 정확도)
- **$\mathcal{L}_{\text{LPIPS}}$**: 지각적 손실(perceptual loss) - 인간의 시각적 인식 기준으로 품질 측정
- **$\mathcal{L}_{\text{Gram}}$**: Gram 행렬 손실 - 이미지의 텍스처/스타일 일관성 유지
  - $\phi_l(I)$: $l$번째 레이어의 특징 맵, $G_l(I)$: 해당 레이어의 Gram 행렬
  - Gram 행렬은 특징 맵의 채널 간 상관관계를 측정하여 전체적인 스타일을 포착

**학습 데이터 큐레이션**: 네 가지 전략으로 쌍(pair) 데이터를 생성합니다:
1. **희소 재구성(Sparse Reconstruction)**: 카메라를 일부만 사용하여 의도적으로 불완전한 재구성 생성
2. **사이클 재구성(Cycle Reconstruction)**: 직선 궤적에서 옆 1-6미터 이동한 카메라로 두 번째 NeRF를 학습하여 저하된 뷰 생성
3. **크로스 참조(Cross Reference)**: 단일 카메라로 재구성 후 나머지 카메라의 뷰를 렌더링
4. **모델 언더피팅(Model Underfitting)**: 전체 학습 에폭의 25-75%만 사용하여 의도적 언더피팅

**핵심 개념**
- **SD-Turbo**: Stable Diffusion의 단일 단계 증류 버전으로, 단 하나의 forward pass로 이미지 생성/편집 가능
- **LoRA (Low-Rank Adaptation)**: 사전학습 모델의 소수 파라미터만 파인튜닝하는 효율적인 방법
- **노이즈 레벨 $\tau$**: DIFIX는 $\tau = 200$을 사용 (범위: 0~1000). 너무 높으면 환각(hallucination), 너무 낮으면 변화 없음. $\tau = 200$이 최적의 아티팩트 제거와 원본 유지 균형을 달성

### 4.2 DIFIX3D+: 확산 사전으로 NVS 향상

**요약**

DIFIX3D+는 두 가지 방식으로 작동합니다:

**방식 1 - 3D 재구성 중 점진적 업데이트:**
직접 DIFIX를 렌더링 뷰에 적용하면 뷰마다 불일치가 발생합니다. 이를 해결하기 위해 향상된 노블 뷰를 3D 표현에 다시 증류(distill)하고 점진적으로 학습합니다:
1. 초기 3D 표현(NeRF/3DGS) 생성
2. 타겟 뷰 렌더링 후 DIFIX로 향상
3. 향상된 이미지를 타겟 뷰로 사용, 그라운드 트루스 카메라를 약간 섭동(perturb)
4. 1.5 이터레이션마다 반복 → 점점 아티팩트 없는 고품질 3D 표현 구축

**방식 2 - 실시간 후처리(DIFIX3D+):**
추론 시 렌더링된 뷰에 DIFIX를 직접 적용하는 신경 향상기(neural enhancer)로 사용. 약 **76ms** (NVIDIA A100 기준, 표준 확산 모델 대비 >10× 빠름)

**핵심 개념**
- **점진적 3D 업데이트(Progressive 3D Update)**: 한 번에 모든 뷰를 추가하지 않고 반복적으로 향상된 뷰를 추가하여 3D 일관성 강화
- **신경 향상기(Neural Enhancer)**: 3D 표현이 완성된 후 렌더링 시점에서 추가적인 품질 향상을 제공하는 모듈
- **카메라 섭동(Camera Perturbation)**: 타겟 카메라를 약간 이동시켜 다양한 훈련 시점을 생성, 3D 일관성 강화

---

## Chapter 5: 실험

**요약**

### 5.1 In-the-Wild 아티팩트 제거 (Nerfbusters & DL3DV 데이터셋)

**학습**: DL3DV 데이터셋 140개 장면 중 112개(80%)에서 80,000개 노이즈-클린 이미지 쌍 생성

**정량적 결과** (Nerfbusters 데이터셋):

| 방법 | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ |
|---|---|---|---|---|
| Nerfacto | 17.29 | 0.6214 | 0.4021 | 134.65 |
| GANeRF | 17.42 | 0.6113 | 0.3539 | 115.60 |
| NeRFLiX | 17.91 | 0.6560 | 0.3458 | 113.59 |
| **DIFIX3D+ (Nerfacto)** | **18.32** | **0.6623** | **0.2789** | **49.44** |
| 3DGS | 17.66 | 0.6780 | 0.3265 | 113.84 |
| **DIFIX3D+ (3DGS)** | **18.51** | **0.6858** | **0.2637** | **41.77** |

DIFIX3D+는 모든 지표에서 최고 성능을 달성하며, 특히 FID가 기준 대비 **2× 이상** 개선됩니다.

### 5.2 자동차 장면 향상 (RDS 데이터셋)

인하우스 실제 주행 장면 데이터셋에서 평가:

| 방법 | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ |
|---|---|---|---|---|
| Nerfacto | 19.95 | 0.4930 | 0.5300 | 91.38 |
| Nerfacto + NeRFLiX | 20.44 | 0.5672 | 0.4686 | 116.28 |
| Nerfacto + DIFIX3D | 21.52 | 0.5700 | 0.4266 | 77.83 |
| **Nerfacto + DIFIX3D+** | **21.75** | **0.5829** | **0.4016** | **73.08** |

### 5.3 어블레이션 연구 (Ablation Study)

점진적 컴포넌트를 추가할수록 성능이 향상됩니다:

| 방법 | PSNR↑ | SSIM↑ | LPIPS↓ | FID↓ |
|---|---|---|---|---|
| Nerfacto | 17.29 | 0.6214 | 0.4021 | 134.65 |
| + (a) DIFIX | 17.40 | 0.6333 | 0.3277 | 63.77 |
| + (b) DIFIX + single-step 3D update | 17.97 | 0.6563 | 0.3424 | 75.94 |
| + (a)+(b)+(c) DIFIX3D | 18.08 | 0.6533 | 0.3277 | 63.77 |
| **+ (a)+(b)+(c)+(d) DIFIX3D+** | **18.32** | **0.6623** | **0.2789** | **49.44** |

**핵심 개념**
- **PSNR (Peak Signal-to-Noise Ratio)**: 픽셀 단위 재구성 정확도 측정 (dB 단위, 높을수록 좋음)
- **SSIM (Structural Similarity Index)**: 구조적 유사도 측정 (0~1, 높을수록 좋음)
- **LPIPS (Learned Perceptual Image Patch Similarity)**: 딥러닝 기반 지각적 유사도 (낮을수록 좋음)
- **FID (Fréchet Inception Distance)**: 실제 이미지 분포와 생성 이미지 분포의 유사도 (낮을수록 좋음)

---

## 핵심 개념 정리

| 개념 | 설명 |
|---|---|
| **DIFIX** | SD-Turbo 기반 단일 단계 확산 모델. 노이즈 있는 NeRF/3DGS 렌더링을 입력으로 받아 아티팩트 제거 |
| **DIFIX3D** | DIFIX를 3D 재구성 중 점진적 업데이트에 사용하는 파이프라인 |
| **DIFIX3D+** | DIFIX3D + 실시간 후처리(신경 향상기)를 결합한 완전한 파이프라인 |
| **단일 단계 확산** | 일반 확산 모델(수백 스텝) 대비 1 스텝으로 추론. >10× 속도 향상 |
| **크로스-뷰 참조 혼합** | 여러 참조 뷰를 attention으로 융합하여 3D 일관성 있는 향상 이미지 생성 |
| **점진적 3D 업데이트** | 향상된 뷰를 반복적으로 3D에 다시 증류하여 3D 일관성 유지 |
| **노이즈 레벨 τ=200** | 아티팩트 제거와 원본 충실도의 최적 균형점 (τ=1000은 환각 발생, τ=10은 변화 미미) |

---

## 결론 및 시사점

DIFIX3D+는 단일 단계 확산 모델(DIFIX)을 핵심으로 한 새로운 3D 재구성 향상 파이프라인입니다.

**주요 기여**:
1. **속도**: 기존 확산 모델 기반 방법 대비 >10× 빠른 추론 (76ms @ A100)
2. **범용성**: NeRF와 3DGS 모두에 적용 가능한 단일 모델
3. **품질**: FID 기준 2× 향상, 3D 일관성 유지
4. **이중 역할**: 3D 최적화 중 데이터 정제 + 추론 시 실시간 아티팩트 제거

**실무적 시사점**:
- 자율주행, AR/VR 등 실시간 3D 렌더링이 필요한 응용에서 활용 가능
- 기존 NeRF/3DGS 파이프라인에 플러그인 형태로 손쉽게 통합 가능
- 인터넷 규모 데이터로 학습된 생성 모델의 사전 지식이 3D 재구성에 효과적으로 전이됨을 입증


---

*관련 논문: [NeRF](/posts/papers/nerf-representing-scenes-as-neural-radiance-fields-for-view-synthesis/), [3D Gaussian Splatting](/posts/papers/3d-gaussian-splatting/), [LDM](/posts/papers/high-resolution-image-synthesis-with-latent-diffusion-models/)*
