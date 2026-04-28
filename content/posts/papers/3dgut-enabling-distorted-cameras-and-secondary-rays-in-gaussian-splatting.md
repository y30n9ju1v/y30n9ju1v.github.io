---
title: "3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting"
date: 2026-04-10T09:00:00+09:00
draft: false
categories: ["Papers", "Novel View Synthesis"]
tags: ["3D Gaussian Splatting", "Novel View Synthesis", "Ray Tracing", "Autonomous Driving"]
---

## 개요

- **저자**: Qi Wu, Janick Martinez Esturo, Ashkan Mirzaei, Nicolas Moenne-Loccoz, Zan Gojcic
- **소속**: NVIDIA, University of Toronto
- **발행년도**: 2025 (arXiv:2412.12507v2, 24 Mar 2025)
- **프로젝트 페이지**: https://research.nvidia.com/labs/toronto-ai/3DGUT
- **주요 내용**: 3D Gaussian Splatting(3DGS)의 EWA splatting 선형화를 Unscented Transform으로 대체하여, 왜곡된 카메라(어안 렌즈, 롤링 셔터 등)와 반사/굴절 같은 secondary ray 조명 효과를 래스터화 효율성을 유지하면서 지원하는 방법을 제안합니다.

## 목차

- Section 1: Introduction — 문제 정의 및 연구 동기
- Section 2: Related Work — 관련 연구 (NeRF, 3DGS, 광선 추적 기반 방법)
- Section 3: Preliminaries — 3DGS, 체적 렌더링, EWA splatting 배경 지식
- Section 4: Method — Unscented Transform 기반 핵심 방법론
- Section 5: Experiments and Ablations — 실험 결과 및 검증
- Section 6: Applications — 복잡한 카메라 모델, Secondary ray 응용
- Section 7: Discussion — 한계점 및 미래 방향
- Supplementary — 일반화 가우시안, 역전파 수식, 가우시안 투영 품질

---

## Section 1: Introduction

**요약**

3D Gaussian Splatting(3DGS)은 복잡한 장면을 실시간으로 고품질 렌더링할 수 있는 강력한 방법입니다. 하지만 기존 3DGS는 두 가지 근본적인 한계를 가집니다: (1) 이상적인 핀홀 카메라만 지원하여 어안 렌즈, 롤링 셔터 같은 왜곡 카메라를 처리하지 못하고, (2) 반사나 굴절 같은 secondary ray 조명 효과를 시뮬레이션하지 못합니다.

기존 해결책으로는 광선 추적(ray tracing) 기반 방법이 있지만, 이는 래스터화 대비 3-4배 느립니다. 이 논문은 래스터화 효율성을 유지하면서 두 한계를 모두 해결하는 **3D Gaussian Unscented Transform(3DGUT)**을 제안합니다.

핵심 아이디어: EWA splatting의 선형화(Jacobian 기반)를 **Unscented Transform(UT)**으로 대체합니다. UT는 비선형 투영 함수를 통해 정확히 변환할 수 있는 Sigma point 집합을 사용하여 가우시안 분포를 근사합니다.

**핵심 개념**

- **EWA Splatting 한계**: 기존 3DGS는 투영 함수의 테일러 1차 선형화(Jacobian)를 사용합니다. 이는 완벽한 핀홀 카메라에서도 근사 오차를 발생시키며, 왜곡이 클수록 오차가 커집니다.
- **Unscented Transform(UT)**: 비선형 변환에서 통계량을 추정하는 기법. 신중하게 선택된 Sigma point들을 실제 비선형 함수에 통과시켜 변환 후 분포를 추정합니다.
- **하이브리드 렌더링**: 1차 광선은 래스터화, 2차 광선은 3DGRT로 추적하는 방식을 통합합니다.

---

## Section 2: Related Work

**요약**

**Neural Radiance Fields(NeRF)**: 볼륨 렌더링 방정식을 신경망으로 학습합니다. 높은 품질을 달성하지만 느린 추론 속도가 한계입니다. MipNeRF360, Zip-NeRF 등이 대표적입니다.

**3D Gaussian Splatting**: 명시적 3D 가우시안 파티클로 장면을 표현합니다. EWA splatting을 통한 미분 가능 래스터화로 실시간 렌더링이 가능합니다.

**왜곡 카메라 지원 방법**: FisheyeGS는 등거리(equidistant) 어안 모델용 Jacobian을 직접 유도합니다. 하지만 카메라 모델마다 별도 Jacobian이 필요하여 범용성이 부족합니다. ZipNeRF는 NeRF 기반으로 왜곡 카메라를 지원하지만 느립니다.

**광선 추적 기반**: 3DGRT는 광선 추적으로 왜곡 카메라와 secondary ray를 지원하지만, 최적화된 래스터화 방법보다 3-4배 느립니다. EVER는 완전 미분 가능 볼륨 광선 추적을 사용합니다.

**핵심 개념**

- **래스터화 vs 광선 추적**: 래스터화는 파티클을 이미지 평면에 투영하는 방식으로 매우 빠르지만 복잡한 카메라 모델 처리가 어렵습니다. 광선 추적은 광선과 파티클의 교차점을 계산하여 정확하지만 느립니다.
- **MLAB(Multi-Layer Alpha Blending)**: 픽셀당 k개의 가장 가까운 히트를 저장하고 알파 블렌딩하는 기법 (StopThePop에서 사용).

---

## Section 3: Preliminaries

**요약**

3DGUT 이해를 위한 세 가지 배경 지식을 정리합니다.

### 3D Gaussian Splatting 표현

장면을 N개의 3D 가우시안 파티클로 표현합니다:

$$\rho(\boldsymbol{x}) = \exp\left(-\frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu})^T \Sigma^{-1} (\boldsymbol{x}-\boldsymbol{\mu})\right)$$

**수식 설명**
- **$\rho(\boldsymbol{x})$**: 3D 공간의 점 $\boldsymbol{x}$에서의 가우시안 응답값 (0~1 사이 밀도)
- **$\boldsymbol{\mu} \in \mathbb{R}^3$**: 가우시안의 3D 위치 (중심점)
- **$\Sigma \in \mathbb{R}^{3\times3}$**: 가우시안의 공분산 행렬 (크기와 방향을 결정)
- **직관**: 중심에서 멀어질수록 밀도가 지수적으로 감소하는 3D 타원체 형태

공분산 행렬은 회전과 스케일로 분해됩니다:

$$\Sigma = RSS^T R^T$$

**수식 설명**
- **$R \in SO(3)$**: 회전 행렬 (사원수 $\boldsymbol{q} \in \mathbb{R}^4$로 저장)
- **$S \in \mathbb{R}^3$**: 스케일 벡터 (각 축의 크기)
- **왜 이렇게 분해?**: $\Sigma$가 항상 양의 정정(positive definite)을 유지하도록 보장

### 체적 파티클 렌더링

카메라 광선 $\boldsymbol{r}(\tau) = \boldsymbol{o} + \tau\boldsymbol{d}$의 색상:

$$\boldsymbol{c}(\boldsymbol{o}, \boldsymbol{d}) = \sum_{i=1}^{N} \boldsymbol{c}_i \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j)$$

**수식 설명**
- **$\boldsymbol{c}(\boldsymbol{o}, \boldsymbol{d})$**: 광선 원점 $\boldsymbol{o}$, 방향 $\boldsymbol{d}$에서의 최종 렌더링 색상
- **$\boldsymbol{c}_i$**: i번째 가우시안의 색상 (구면 조화 함수로 표현)
- **$\alpha_i = \sigma_i \rho_i(\boldsymbol{o} + \tau\boldsymbol{d})$**: i번째 가우시안의 불투명도 × 밀도
- **$\prod_{j=1}^{i-1}(1-\alpha_j)$**: i번째 가우시안까지 도달하는 투과율 (앞의 모든 가우시안을 지나온 빛의 비율)
- **직관**: 가까운 물체부터 순서대로 반투명하게 합성하는 화가 알고리즘

### EWA Splatting

3DGS 래스터화의 핵심. 3D 가우시안을 이미지 평면에 투영할 때, 투영 함수 $\boldsymbol{v} = g(\boldsymbol{x})$의 1차 테일러 근사를 사용:

$$\Sigma' = J_{[2:3]} W \Sigma W^T J_{[2:3]}^T$$

**수식 설명**
- **$\Sigma' \in \mathbb{R}^{2\times2}$**: 이미지 평면에서의 2D 가우시안 공분산 (투영된 결과)
- **$W \in SE(3)$**: 카메라 외부 파라미터 (월드 → 카메라 좌표 변환)
- **$J \in \mathbb{R}^{3\times3}$**: 투영 함수의 아핀 근사를 위한 Jacobian 행렬
- **$[2:3]$**: 행렬의 처음 두 행만 선택 (3D → 2D)
- **한계**: Jacobian $J$는 카메라 모델마다 새로 계산해야 하며, 왜곡이 큰 카메라에서 오차가 커짐

---

## Section 4: Method

**요약**

3DGUT의 세 가지 핵심 구성 요소를 설명합니다.

### 4.1 Unscented Transform

EWA splatting의 선형화 한계를 극복하기 위해 **Unscented Transform(UT)**을 사용합니다. UT는 비선형 투영 함수를 통해 정확히 변환할 수 있는 **Sigma point** 집합 $\mathcal{X} = \{\boldsymbol{x}_i\}_{i=0}^{6}$을 사용합니다:

$$\boldsymbol{x}_i = \begin{cases} \boldsymbol{\mu} & \text{for } i = 0 \\ \boldsymbol{\mu} + (\sqrt{(3+\lambda)\Sigma})_{[i]} & \text{for } i = 1, 2, 3 \\ \boldsymbol{\mu} - (\sqrt{(3+\lambda)\Sigma})_{[i-3]} & \text{for } i = 4, 5, 6 \end{cases}$$

**수식 설명**
- **$\boldsymbol{x}_i$**: 7개의 Sigma point (중심 1개 + 각 축 방향으로 ±3개씩)
- **$\boldsymbol{\mu}$**: 가우시안 중심점
- **$\sqrt{(3+\lambda)\Sigma}$**: 공분산 행렬의 스케일된 제곱근 (가우시안의 "퍼짐" 정도)
- **$\lambda = \alpha^2(3+\kappa) - 3$**: Sigma point들이 중심에서 얼마나 멀리 퍼지는지 제어하는 하이퍼파라미터
- **직관**: 가우시안 분포를 7개의 대표점으로 근사하여, 각 점을 실제 비선형 함수에 통과시켜 변환 후 분포를 추정

가중치:

$$w_i^m = \begin{cases} \frac{\lambda}{3+\lambda} & \text{for } i = 0 \\ \frac{1}{2(3+\lambda)} & \text{for } i = 1, \ldots, 6 \end{cases}$$

**수식 설명**
- **$w_i^m$**: 평균 추정을 위한 가중치 (모든 가중치의 합 = 1)
- **$w_i^\Sigma$**: 공분산 추정을 위한 가중치 (약간 다름, $\beta$ 파라미터 추가)
- **직관**: 중심 Sigma point는 더 큰 가중치를 받고, 주변 점들은 동일하게 가중

이미지 평면에서의 2D 추정:

$$\boldsymbol{v}_\mu = \sum_{i=0}^{6} w_i^m \boldsymbol{v}_{x_i}$$

$$\Sigma' = \sum_{i=0}^{6} w_i^\Sigma (\boldsymbol{v}_{x_i} - \boldsymbol{v}_\mu)(\boldsymbol{v}_{x_i} - \boldsymbol{v}_\mu)^T$$

**수식 설명**
- **$\boldsymbol{v}_\mu \in \mathbb{R}^2$**: 투영된 가우시안의 2D 평균 (이미지 평면에서의 위치)
- **$\boldsymbol{v}_{x_i} = g(\boldsymbol{x}_i)$**: 각 Sigma point를 실제 비선형 투영 함수 $g$에 통과시킨 결과
- **$\Sigma' \in \mathbb{R}^{2\times2}$**: 투영된 가우시안의 2D 공분산
- **핵심 장점**: Jacobian 없이 임의의 비선형 투영 함수 $g$를 직접 적용 → 어떤 카메라 모델도 지원

### 4.2 3D에서의 파티클 응답 평가

3DGS는 2D 이미지 평면에서 가우시안 응답을 평가하지만, 3DGUT는 **3D에서 직접** 평가합니다. 3DGRT를 따라 광선 $\boldsymbol{r}(\tau)$를 따라 최대 응답 지점을 찾습니다:

$$\tau_{\max} = \underset{\tau}{\operatorname{argmax}} \, \rho(\boldsymbol{o} + \tau\boldsymbol{d})$$

이는 분석적으로 계산됩니다:

$$\tau_{\max} = \frac{(\boldsymbol{\mu} - \boldsymbol{o})^T \Sigma^{-1} \boldsymbol{d}}{\boldsymbol{d}^T \Sigma^{-1} \boldsymbol{d}} = \frac{-\boldsymbol{o}_g^T \boldsymbol{d}_g}{\boldsymbol{d}_g^T \boldsymbol{d}_g}$$

**수식 설명**
- **$\tau_{\max}$**: 광선 위에서 가우시안 응답이 최대가 되는 거리
- **$\boldsymbol{o}_g = S^{-1}R^T(\boldsymbol{o} - \boldsymbol{\mu})$**: 정규 가우시안 공간에서의 광선 원점
- **$\boldsymbol{d}_g = S^{-1}R^T\boldsymbol{d}$**: 정규 가우시안 공간에서의 광선 방향
- **직관**: 가우시안 좌표계에서 광선에 수직인 점이 최대 응답점 → 내적이 0인 지점
- **장점**: 투영 함수의 역전파 없이 3D에서 직접 계산 → 수치 안정성 향상

### 4.3 파티클 정렬

UT 기반 렌더링(수식 5)과 파티클 응답 평가(수식 11)의 결합 덕분에 3DGRT와 동일한 $\tau_{\max}$ 순서로 파티클을 정렬할 수 있습니다. **Multi-Layer Alpha Blending(MLAB)**을 사용하여 픽셀당 k=16개의 가장 가까운 히트를 저장하고 순서대로 블렌딩합니다.

**핵심 개념**

- **UT의 범용성**: 어안 렌즈, 롤링 셔터 등 임의의 카메라 모델에 코드 수정 없이 적용
- **3D 파티클 평가의 안정성**: 비선형 투영 역전파 없이 3D에서 그라디언트 계산
- **MLAB vs HT(Hybrid Transparency)**: MLAB는 정확한 k-최근접 히트를 보장, HT는 근사적

---

## Section 5: Experiments and Ablations

**요약**

세 가지 표준 벤치마크(MipNeRF360, Tanks & Temples, Scannet++)와 자율주행 데이터셋(Waymo)에서 평가합니다.

### MipNeRF360

표준 새뷰 합성 벤치마크. 완벽한 핀홀 카메라이므로, Ours와 Ours (sorted)가 3DGS와 비슷한 PSNR을 달성하면서 **265+ FPS**를 유지합니다 (가장 가까운 경쟁자 3DGRT는 52 FPS).

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | FPS↑ |
|--------|-------|-------|--------|------|
| 3DGS | 27.26 | 0.810 | 0.218 | 265 |
| StopThePop | 27.14 | 0.804 | 0.235 | 340 |
| 3DGRT | 27.20 | 0.818 | 0.248 | 52 |
| **Ours (sorted)** | **27.26** | **0.812** | **0.215** | **200** |

### Tanks & Temples

대규모 야외 데이터셋. Ours (sorted)가 PSNR 22.90, SSIM 0.844로 3DGS와 비슷하거나 우수한 성능을 달성합니다.

### Scannet++

**어안 카메라**로 촬영된 실내 데이터셋. FisheyeGS 대비:
- PSNR: 28.15 → **28.46** (+0.31)
- SSIM: 0.901 → **0.910** (+0.009)
- 사용 가우시안 수: 1.07M → **0.38M** (절반 이하)

FisheyeGS는 특정 카메라 모델을 위해 Jacobian을 직접 유도했음에도, 범용적인 UT 방법이 더 우수한 성능을 보입니다.

### Waymo 자율주행

롤링 셔터 카메라로 촬영된 자율주행 데이터셋.

| Method | PSNR↑ | SSIM↑ |
|--------|-------|-------|
| 3DGS | 29.83 | 0.917 |
| 3DGRT | 29.99 | 0.897 |
| **Ours (sorted)** | **30.16** | **0.900** |

**핵심 개념**

- **롤링 셔터 효과**: 카메라가 이동하는 동안 이미지의 각 행이 서로 다른 시간에 촬영됩니다. UT는 이 시간 의존적 투영 함수를 자연스럽게 처리합니다.
- **범용성의 실증**: 단일 구현으로 어안 렌즈, 롤링 셔터, 핀홀 카메라를 모두 처리

---

## Section 6: Applications

**요약**

### 6.1 복잡한 카메라 모델

**왜곡 카메라 모델**: 어안, 롤링 셔터 등 임의의 비선형 카메라를 UT로 직접 지원합니다. Jacobian 유도 없이 투영 함수 $g$만 정의하면 자동으로 처리됩니다.

**롤링 셔터**: 카메라 움직임을 투영 공식에 통합하여 롤링 셔터 효과를 충실히 재현합니다. Fig. 7에서 전통적 splatting은 롤링 셔터 아티팩트를 제대로 처리하지 못하는 반면, UT 기반 방법은 센서 움직임을 충실히 모델링합니다.

### 6.2 Secondary Ray와 조명 효과

렌더링 공식을 3DGRT와 일치시킴으로써 **하이브리드 렌더링**이 가능합니다:
1. **1차 광선**: 래스터화로 빠르게 처리
2. **2차 광선**: 3DGRT로 추적 (반사, 굴절)

Fig. 8에서 3DGS, StopThePop 대비 3DGRT와 가장 일치하는 결과를 달성하며, 반사/굴절 같은 복잡한 조명 효과를 재현합니다.

**핵심 개념**

- **Secondary ray**: 카메라에서 나온 1차 광선이 표면에 부딪힌 후 생성되는 반사/굴절 광선. 이를 추적해야 거울 반사, 유리 굴절 등을 표현 가능
- **하이브리드 렌더링의 장점**: 1차 렌더링은 래스터화 속도를 유지하면서, 조명 효과는 광선 추적의 정확성을 활용

---

## Section 7: Discussion

**요약**

### 장점

- EWA splatting 선형화를 UT로 대체하는 단순한 아이디어로 3DGS를 임의의 비선형 카메라 투영에 일반화
- 래스터화 효율성을 유지하면서 (3DGRT 대비 약 4배 빠름)
- 코드 수정 없이 임의의 카메라 모델 지원 (plug-and-play)
- 3DGRT와 렌더링 공식을 통합하여 secondary ray 지원

### 한계점

- 3DGS보다는 여전히 느림 (UT의 3D 평가 추가 비용)
- 큰 왜곡에서 투영된 결과가 2D 가우시안에서 벗어남 (왜곡이 매우 심할 때)
- 겹치는 가우시안 처리가 부정확 (단일 점으로 평가)
- 현재는 overlapping Gaussian을 정확히 평가하지 못함

### 미래 방향

- EVER 같은 완전 미분 가능 방법이 겹치는 가우시안 처리 개선에 도움이 될 수 있음
- 자율주행, 로봇공학 분야에서 왜곡 카메라 렌더링이 필수적

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **3D Gaussian Splatting (3DGS)** | 장면을 3D 가우시안 파티클들로 표현하고, 이를 이미지 평면에 투영하여 실시간 렌더링하는 방법 |
| **EWA Splatting** | 3D 가우시안을 2D로 투영할 때 투영 함수의 Jacobian을 이용한 선형 근사 방법. 완벽한 핀홀 카메라에서도 오차 발생 |
| **Unscented Transform (UT)** | 비선형 변환에서 확률 분포의 평균과 공분산을 추정하는 기법. 신중하게 선택된 Sigma point들을 실제 비선형 함수에 통과시킴 |
| **Sigma Point** | UT에서 사용하는 대표점들. 3D 가우시안의 경우 7개 사용 (중심 1개 + 각 축 ±방향 6개) |
| **Secondary Ray** | 1차 광선이 표면에 반사/굴절되어 생성되는 2차 광선. 거울 반사, 유리 굴절 등 복잡한 조명 효과 표현에 필요 |
| **롤링 셔터 (Rolling Shutter)** | 이미지 센서가 한 번에 전체를 촬영하지 않고 위에서 아래로 순차적으로 읽는 방식. 카메라 이동 중 이미지 왜곡 발생 |
| **MLAB (Multi-Layer Alpha Blending)** | 픽셀당 k개의 가장 가까운 히트를 저장하고 알파 블렌딩하는 방법. 정확한 파티클 순서 정렬을 위해 사용 |
| **하이브리드 렌더링** | 1차 광선은 빠른 래스터화로, 2차 광선은 광선 추적으로 처리하는 혼합 방식 |
| **KL 발산 (KL Divergence)** | 두 확률 분포 간의 차이를 측정하는 지표. 값이 낮을수록 UT가 Monte Carlo 참조 분포에 가깝게 근사함을 의미 |
| **Generalized Gaussian** | 표준 가우시안(degree=2)을 일반화한 형태. Degree가 클수록 더 조밀하고 좁은 분포를 가짐 |

---

## 결론 및 시사점

**결론**: 3DGUT는 EWA splatting의 Jacobian 기반 선형화를 Unscented Transform으로 대체하는 단순하지만 강력한 아이디어를 제안합니다. 이를 통해:

1. **범용 카메라 지원**: Jacobian 없이 임의의 비선형 카메라 투영 함수를 직접 지원
2. **효율성 유지**: 3DGRT 대비 약 4배 빠른 렌더링 속도 (200+ FPS)
3. **Secondary ray 통합**: 3DGRT와 렌더링 공식을 일치시켜 반사/굴절 표현 가능
4. **우수한 성능**: 어안 카메라에서 FisheyeGS를 능가하고, 롤링 셔터에서 3DGRT를 능가

**실무 시사점**:
- 자율주행, 로봇공학 분야에서 넓은 화각의 왜곡 카메라가 보편적으로 사용됩니다. 3DGUT는 이런 환경에서 고품질 실시간 장면 재구성을 가능하게 합니다.
- 새로운 카메라 모델 지원 시 Jacobian을 새로 유도할 필요 없이 투영 함수만 정의하면 되므로, 개발 및 적용이 크게 단순화됩니다.
- 하이브리드 렌더링을 통해 역 렌더링(relighting) 등 고급 시각 효과 연구에도 응용 가능성이 있습니다.


---

*관련 논문: [3D Gaussian Splatting](/posts/papers/3d-gaussian-splatting/), [3D Gaussian Ray Tracing](/posts/papers/3d-gaussian-ray-tracing/), [HUGSIM](/posts/papers/hugsim-real-time-photorealistic-closed-loop-simulator/)*
