---
title: "Denoising Diffusion Probabilistic Models (DDPM)"
date: 2026-04-19T23:00:00+09:00
draft: false
categories: ["Papers", "Generative Models", "Deep Learning"]
tags: ["diffusion models", "generative models", "image synthesis", "deep learning"]
---

## 개요
- **저자**: Jonathan Ho, Ajay Jain, Pieter Abbeel (UC Berkeley)
- **발행년도**: 2020 (NeurIPS 2020)
- **arXiv**: 2006.11239v2
- **주요 내용**: 확산 확률 모델(Diffusion Probabilistic Models)을 이용한 고품질 이미지 합성. CIFAR10에서 FID 3.17, Inception Score 9.46 달성

## 한계 극복

이 논문은 기존 생성 모델들의 한계를 극복하기 위해 작성되었습니다.

- **기존 GAN의 한계**: 학습 불안정성과 mode collapse 문제, 다양성과 품질 간의 트레이드오프
- **기존 VAE의 한계**: 잠재 공간 표현력 제한으로 인한 흐릿한(blurry) 이미지 생성
- **기존 Flow 모델의 한계**: 정확한 가역 구조 요구로 인한 모델 설계 제약
- **이 논문의 접근 방식**: 비평형 열역학(nonequilibrium thermodynamics)에서 영감을 받아 노이즈를 점진적으로 추가/제거하는 Markov chain 기반 학습. 특히 **denoising score matching**과의 동등성을 발견하여 단순하고 효과적인 학습 목적함수 도출

## 목차
- Section 1: Introduction
- Section 2: Background — Forward & Reverse Process 수식 정의
- Section 3: Diffusion Models and Denoising Autoencoders — 핵심 기여
- Section 4: Experiments — CIFAR10, LSUN 실험 결과
- Section 5: Related Work
- Section 6: Conclusion

---

## Section 1: Introduction

**요약**

딥 생성 모델(GAN, VAE, Flow, Autoregressive)들이 고품질 샘플을 생성하는 데 성공했지만, 확산 모델은 그동안 고품질 이미지 생성 능력이 입증되지 않았습니다. 이 논문은 **확산 모델이 실제로 GAN을 포함한 다른 생성 모델보다 우수한 샘플 품질을 달성할 수 있음**을 최초로 체계적으로 입증합니다.

핵심 기여는 두 가지입니다:
1. 확산 모델의 특정 파라미터화(parameterization)가 **denoising score matching**과 동등하다는 연결을 발견
2. 이 연결을 통해 단순화된 학습 목적함수($L_\text{simple}$)를 도출하여 더 높은 샘플 품질 달성

**핵심 개념**
- **Diffusion Probabilistic Model**: 데이터에 점진적으로 노이즈를 추가하는 forward process와, 그것을 역으로 되돌리는 reverse process로 구성된 잠재 변수 모델
- **Markov Chain**: 각 상태가 오직 직전 상태에만 의존하는 확률 과정. 확산 모델의 forward/reverse process가 모두 Markov chain
- **Denoising Score Matching**: 데이터 분포의 score(로그 밀도의 기울기)를 학습하는 방법

---

## Section 2: Background

**요약**

확산 모델의 수학적 기초를 정립합니다. **Forward process**(데이터 → 노이즈)와 **Reverse process**(노이즈 → 데이터)의 두 Markov chain으로 구성됩니다.

### Forward Process (확산 과정)

원본 이미지 $\mathbf{x}_0$에 T번에 걸쳐 Gaussian 노이즈를 조금씩 추가합니다:

$$q(\mathbf{x}_{1:T} | \mathbf{x}_0) := \prod_{t=1}^{T} q(\mathbf{x}_t | \mathbf{x}_{t-1}), \qquad q(\mathbf{x}_t | \mathbf{x}_{t-1}) := \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\mathbf{x}_{t-1}, \beta_t \mathbf{I})$$

**수식 설명**:
- **$\mathbf{x}_0$**: 원본 이미지 (노이즈 없음)
- **$\mathbf{x}_t$**: t번째 타임스텝의 노이즈가 추가된 이미지
- **$\beta_t$**: 타임스텝 t에서의 노이즈 크기 (분산 스케줄, 매우 작은 값)
- **$\sqrt{1-\beta_t}$**: 원본 신호를 얼마나 유지할지의 비율. $\beta_t$가 작을수록 원본을 많이 보존
- **$\prod$**: T번의 노이즈 추가를 곱으로 표현 (각 단계가 독립적으로 적용)

> 직관: 매 단계마다 이미지에 아주 조금씩 눈 내리듯 노이즈를 뿌립니다. T=1000번 반복하면 완전한 랜덤 노이즈가 됩니다.

### Forward Process Closed-form (임의 타임스텝 직접 계산)

$\alpha_t := 1 - \beta_t$, $\bar{\alpha}_t := \prod_{s=1}^{t} \alpha_s$ 로 정의하면:

$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})$$

**수식 설명**:
- **$\bar{\alpha}_t$**: t 타임스텝까지의 누적 신호 유지 비율. t가 커질수록 0에 가까워짐
- **$\sqrt{\bar{\alpha}_t}\mathbf{x}_0$**: t 타임스텝에서 남은 원본 신호의 양
- **$(1 - \bar{\alpha}_t)\mathbf{I}$**: t 타임스텝에서 추가된 노이즈의 분산

> 핵심: 중간 단계를 건너뛰고 **임의의 타임스텝 t에서의 노이즈 이미지를 한 번에 계산**할 수 있습니다. 이것이 효율적인 학습을 가능하게 합니다.

### Reverse Process (복원 과정)

학습할 파라미터 $\theta$를 가진 신경망이 노이즈를 제거합니다:

$$p_\theta(\mathbf{x}_{0:T}) := p(\mathbf{x}_T) \prod_{t=1}^{T} p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t), \qquad p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) := \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))$$

**수식 설명**:
- **$p(\mathbf{x}_T) = \mathcal{N}(\mathbf{x}_T; \mathbf{0}, \mathbf{I})$**: 완전한 랜덤 노이즈에서 시작
- **$\boldsymbol{\mu}_\theta(\mathbf{x}_t, t)$**: 신경망이 예측하는 한 단계 전 이미지의 평균
- **$\boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t)$**: 신경망이 예측하는 분산 (논문에서는 고정값 사용)

### 학습 목적함수 (ELBO)

$$\mathbb{E}\left[-\log p_\theta(\mathbf{x}_0)\right] \leq \mathbb{E}_q\left[-\log p(\mathbf{x}_T) - \sum_{t \geq 1} \log \frac{p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_t|\mathbf{x}_{t-1})}\right] =: L$$

이를 KL divergence 형태로 분리하면:

$$\mathbb{E}_q\left[\underbrace{D_\text{KL}(q(\mathbf{x}_T|\mathbf{x}_0) \| p(\mathbf{x}_T))}_{L_T} + \sum_{t>1} \underbrace{D_\text{KL}(q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) \| p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t))}_{L_{t-1}} \underbrace{- \log p_\theta(\mathbf{x}_0|\mathbf{x}_1)}_{L_0}\right]$$

**수식 설명**:
- **$L_T$**: Forward process 끝이 순수 가우시안 노이즈가 되도록 하는 항 (학습 파라미터 없음)
- **$L_{t-1}$**: 각 타임스텝에서 reverse process가 forward process posterior를 얼마나 잘 근사하는지
- **$L_0$**: 최종 이미지 복원 품질 (재구성 손실)

**핵심 개념**
- **ELBO (Evidence Lower BOund)**: 로그 가능도의 하한. 이를 최대화하는 것이 생성 모델 학습의 핵심
- **KL Divergence**: 두 확률 분포의 차이를 측정하는 값. 두 분포가 같으면 0

---

## Section 3: Diffusion Models and Denoising Autoencoders

**요약**

이 논문의 핵심 기여가 담긴 섹션입니다. 모델 설계 선택을 정당화하고 **denoising score matching과의 동등성**을 도출합니다.

### 3.1 Forward Process와 $L_T$

Forward process의 분산 $\beta_t$를 학습 가능한 파라미터가 아닌 **고정 상수**로 설정합니다. 이렇게 하면 $L_T$는 학습에 무관한 상수가 됩니다.

### 3.2 Reverse Process와 $L_{t-1}$ — ε-prediction

Forward process의 사후 분포는 closed form으로 계산 가능합니다:

$$q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I})$$

$$\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) := \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{x}_t, \qquad \tilde{\beta}_t := \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$$

**수식 설명**:
- **$\tilde{\boldsymbol{\mu}}_t$**: 원본 이미지 $\mathbf{x}_0$와 현재 노이즈 이미지 $\mathbf{x}_t$의 가중 평균
- **$\tilde{\beta}_t$**: 사후 분포의 분산. Forward process 분산보다 항상 작음

신경망이 이 평균을 예측하도록 설계하면, $L_{t-1}$은:

$$L_{t-1} = \mathbb{E}_q\left[\frac{1}{2\sigma_t^2}\|\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) - \boldsymbol{\mu}_\theta(\mathbf{x}_t, t)\|^2\right] + C$$

더 나아가 **ε-prediction 파라미터화**를 도입합니다. $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$ 관계를 이용하면:

$$\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right)$$

$$L_{t-1} - C = \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}}\left[\frac{\beta_t^2}{2\sigma_t^2 \alpha_t(1-\bar{\alpha}_t)}\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}, t)\|^2\right]$$

**수식 설명**:
- **$\boldsymbol{\epsilon}$**: 원본 이미지에 추가된 실제 노이즈 (정답)
- **$\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$**: 신경망이 현재 노이즈 이미지 $\mathbf{x}_t$를 보고 추정한 노이즈
- **핵심 직관**: 신경망은 "이미지에 어떤 노이즈가 추가되었는가?"를 맞추도록 학습됩니다

> 이 목적함수는 **여러 노이즈 스케일에서의 denoising score matching** (NCSN [55])과 동등합니다. 이것이 이 논문의 핵심 발견입니다.

### 3.4 단순화된 학습 목적함수

가중치를 무시한 단순화된 버전이 실제로 더 좋은 샘플 품질을 냅니다:

$$L_\text{simple}(\theta) := \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}, t)\|^2\right]$$

**수식 설명**:
- **$t \sim \text{Uniform}(\{1, \ldots, T\})$**: 매 학습 반복마다 랜덤 타임스텝 선택
- **$\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$**: 랜덤 Gaussian 노이즈 샘플링
- **$\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$**: t 타임스텝의 노이즈 이미지 ($= \mathbf{x}_t$)
- 전체 손실: 예측 노이즈와 실제 노이즈의 **평균 제곱 오차 (MSE)**

> 직관: 원본 이미지에 노이즈를 섞은 뒤, 신경망에게 "이 혼합 이미지에서 노이즈 성분만 추출하라"고 학습시키는 것. 마치 사진에서 노이즈를 제거하는 denoise 필터를 학습하는 것과 유사합니다.

### 학습 및 샘플링 알고리즘

**학습 (Algorithm 1)**:
1. 원본 이미지 $\mathbf{x}_0$ 샘플링
2. 랜덤 타임스텝 $t$ 선택
3. 랜덤 노이즈 $\boldsymbol{\epsilon}$ 샘플링
4. 신경망이 노이즈를 예측하도록 그래디언트 업데이트

**샘플링 (Algorithm 2)**:
1. 순수 가우시안 노이즈 $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$에서 시작
2. T번의 역방향 단계 반복:
   $$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right) + \sigma_t \mathbf{z}$$
3. 최종 $\mathbf{x}_0$ 반환

**수식 설명**:
- **$\sigma_t \mathbf{z}$**: 샘플링 다양성을 위한 랜덤성 추가 ($\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, 마지막 단계 제외)
- 각 단계에서 신경망이 예측한 노이즈를 빼서 조금씩 이미지를 복원

**핵심 개념**
- **ε-prediction**: 신경망이 평균을 직접 예측하는 대신, 추가된 노이즈를 예측하도록 파라미터화. 더 안정적이고 품질이 높음
- **분산 스케줄**: $\beta_1 = 10^{-4}$에서 $\beta_T = 0.02$로 선형 증가 (T=1000)

---

## Section 4: Experiments

**요약**

CIFAR10, CelebA-HQ 256×256, LSUN 256×256에서 실험합니다.

### 4.1 샘플 품질

| 모델 | IS (↑) | FID (↓) |
|------|--------|---------|
| StyleGAN2+ADA | **10.06** | 2.67 |
| **Ours (L_simple)** | **9.46±0.11** | **3.17** |
| SNGAN | 9.09±0.10 | 15.42 |
| NCSNv2 | 8.87±0.12 | 25.32 |

- CIFAR10 비조건부(unconditional) 생성에서 FID **3.17**, IS **9.46** — 대부분의 조건부 모델을 포함해 최고 수준
- $L_\text{simple}$ 목적함수가 진짜 ELBO보다 샘플 품질이 더 높음 (가중치 제거가 어려운 타임스텝에 집중하게 함)

### 4.2 Reverse Process 파라미터화 Ablation

- $\bar{\boldsymbol{\mu}}$ 예측(베이스라인): 진짜 ELBO로 학습 시에만 잘 동작
- **$\boldsymbol{\epsilon}$ 예측(제안)**: $L_\text{simple}$로 학습 시 FID **3.17** 달성, $\bar{\boldsymbol{\mu}}$ 예측 대비 압도적 우세

### 4.3 Progressive Coding

- Rate: **1.78 bits/dim**, Distortion: **1.97 bits/dim** (RMSE 0.95 on [0,255])
- 코드길이의 절반 이상이 지각 불가능한 세부 정보 묘사에 사용됨
- 이는 확산 모델이 **우수한 손실 압축기(lossy compressor)**임을 보여줌

### 4.4 보간(Interpolation)

잠재 공간에서 두 이미지를 보간하면 포즈, 피부톤, 헤어스타일 등 속성이 부드럽게 전환됩니다. t가 클수록 더 다양하고 coarse한 보간, 작을수록 세밀한 보간.

**핵심 개념**
- **FID (Fréchet Inception Distance)**: 생성 이미지와 실제 이미지의 분포 차이. 낮을수록 좋음
- **IS (Inception Score)**: 생성 이미지의 품질과 다양성. 높을수록 좋음
- **Rate-Distortion**: 정보 이론에서 압축률 vs 품질의 트레이드오프 분석

---

## Section 5: Related Work

**요약**

확산 모델과 관련된 이전 연구들과의 차이를 설명합니다.

- **Flow 모델과의 차이**: 확산 모델은 $q$에 학습 파라미터가 없고 최상위 잠재변수 $\mathbf{x}_T$가 데이터와 거의 상호정보가 없음
- **NCSN (Score Matching) 연결**: $\boldsymbol{\epsilon}$-prediction이 다중 노이즈 스케일의 denoising score matching과 동등. 역으로 denoising score matching의 특정 가중치 형태가 변분 추론과 동일함을 의미
- **에너지 기반 모델과의 연결**: score matching과의 연결을 통해 에너지 기반 모델 연구에도 시사점 제공

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **Forward Process** | 데이터에 T번에 걸쳐 Gaussian 노이즈를 점진적으로 추가. 파라미터 없음 |
| **Reverse Process** | 신경망이 노이즈에서 데이터를 복원. U-Net 구조 사용 |
| **ε-prediction** | 신경망이 추가된 노이즈를 예측하도록 파라미터화. 핵심 기여 |
| **$L_\text{simple}$** | 가중치 없는 단순 MSE 손실. 실제로는 재가중된 ELBO로 해석 가능 |
| **분산 스케줄** | $\beta_t$를 $10^{-4}$에서 $0.02$로 선형 증가, T=1000 |
| **Score Matching 동등성** | ε-prediction 학습이 다중 노이즈 스케일의 denoising score matching과 수학적으로 동일 |
| **Progressive Decoding** | 샘플링 과정이 큰 구조 → 세부 정보 순서로 생성되는 자동 회귀적 디코딩과 유사 |
| **U-Net Backbone** | 시간 정보를 sinusoidal embedding으로 입력받는 수정된 PixelCNN++ 구조 |

---

## 결론 및 시사점

**논문의 결론**

DDPM은 확산 모델이 GAN, VAE, Flow 등 기존 생성 모델과 경쟁하거나 능가하는 고품질 이미지를 생성할 수 있음을 최초로 체계적으로 입증했습니다. 핵심 발견은 **ε-prediction이 denoising score matching과 동등**하다는 것으로, 이는 확산 모델과 score-based 생성 모델을 하나의 프레임워크로 통합하는 이론적 기반을 제공합니다.

**실무적 시사점**

1. **이미지 생성 분야의 패러다임 전환**: 이 논문 이후 Stable Diffusion, DALL-E 2, Imagen 등 현재 주류 텍스트-이미지 생성 모델의 이론적 기반이 됨
2. **학습 안정성**: GAN처럼 두 네트워크의 균형을 맞출 필요 없이 단순한 MSE 손실로 안정적 학습
3. **샘플링 속도 한계**: T=1000번의 신경망 호출이 필요해 속도가 느림 → DDIM, DPM-Solver 등 후속 연구의 동기
4. **확장성**: 텍스트 조건부 생성, 영상, 음성 등 다양한 모달리티로 확장 가능
5. **데이터 압축**: Progressive lossy coding으로 데이터 압축에도 응용 가능
