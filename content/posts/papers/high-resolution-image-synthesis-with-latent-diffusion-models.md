---
title: "High-Resolution Image Synthesis with Latent Diffusion Models"
date: 2026-04-20T08:00:00+09:00
draft: false
categories: ["Papers", "Generative Models", "Diffusion"]
tags: ["LDM", "Stable Diffusion", "Diffusion Models", "Image Synthesis", "Latent Space", "Cross-Attention"]
---

## 개요

- **저자**: Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer (LMU Munich & Runway ML)
- **발표**: CVPR 2022
- **arXiv**: 2112.10752
- **주요 내용**: Diffusion Model을 픽셀 공간이 아닌 **압축된 잠재 공간(latent space)** 에서 학습시켜, 품질 저하 없이 훈련·추론 비용을 대폭 절감한 **Latent Diffusion Models(LDM)** 제안. Stable Diffusion의 직접적인 기반 논문.

---

## 한계 극복

- **기존 한계 1 — 픽셀 공간의 계산 비용**: 기존 Diffusion Model(DDPM, ADM 등)은 고해상도 이미지 픽셀 전체를 대상으로 수백~수천 스텝의 denoising을 반복하므로, 학습에 수백 GPU-day, 추론에 수십 초가 소요됨
- **기존 한계 2 — 지각적으로 무의미한 정보 과다 처리**: 이미지 픽셀의 상당 부분은 고주파 세부 정보(High-frequency imperceptible detail)로, 생성 품질에 거의 기여하지 않음에도 모델이 모든 픽셀을 동일하게 처리해야 함
- **기존 한계 3 — 조건부 생성의 어려움**: 텍스트·레이아웃 등 다양한 조건 입력을 Diffusion Model에 통합하는 범용적인 방법이 부재했음
- **이 논문의 접근 방식**: ① Autoencoder로 이미지를 저차원 latent space로 압축(perceptual compression), ② 그 latent space에서 Diffusion Model을 학습(semantic compression), ③ Cross-attention으로 텍스트·시맨틱맵 등 다양한 조건 입력을 통합

---

## 목차

- Section 1: Introduction
- Section 2: Related Work
- Section 3: Method — Perceptual Compression / Latent Diffusion / Conditioning
- Section 4: Experiments — 이미지 생성·텍스트→이미지·슈퍼해상도·인페인팅
- Section 5: Limitations & Societal Impact
- Section 6: Conclusion

---

## Section 1: Introduction

**요약**

이미지 합성은 딥러닝에서 가장 주목받는 분야 중 하나지만, 고해상도 이미지를 다루는 모델은 엄청난 계산 비용을 요구한다. Diffusion Model은 GAN 대비 학습 안정성과 다양성에서 우수하지만, 픽셀 공간에서 작동하는 탓에 훈련과 추론이 매우 느리다.

이 논문은 **이미지 형성 과정을 두 단계로 분리**하는 아이디어를 제안한다:
1. **Perceptual Compression**: Autoencoder가 지각적으로 무의미한 고주파 세부 정보를 제거하고 효율적인 latent 표현을 학습
2. **Semantic Compression**: Diffusion Model이 latent 공간에서 이미지의 의미론적·개념적 구성을 학습

이 두 단계 분리를 통해 계산 복잡도를 크게 낮추면서도 이미지 품질을 유지한다.

**핵심 개념**

- **Rate-Distortion Tradeoff**: 이미지 압축에서 압축률(Rate)과 화질 손실(Distortion)은 트레이드오프 관계. LDM은 이 곡선에서 "지각적 품질은 유지하되 의미론적 정보만 남기는" 지점을 찾음
- **Perceptual vs Semantic Compression**: Autoencoder는 perceptual compression(눈에 안 보이는 노이즈 제거), Diffusion Model은 semantic compression(이미지의 의미 구조 학습)을 담당
- **두 단계 분리의 장점**: Autoencoder를 한 번만 학습해 여러 Diffusion Model에 재사용 가능

---

## Section 3: Method

### 3.1 Perceptual Image Compression (지각적 이미지 압축)

**요약**

Encoder $\mathcal{E}$와 Decoder $\mathcal{D}$로 구성된 Autoencoder를 학습한다. Encoder는 이미지 $x \in \mathbb{R}^{H \times W \times 3}$를 잠재 표현 $z = \mathcal{E}(x) \in \mathbb{R}^{h \times w \times c}$로 압축하고, Decoder는 이를 복원한다. 다운샘플링 팩터는 $f = H/h = W/w$로 표기하며, $f \in \{1, 2, 4, 8, 16, 32\}$를 실험했다.

**핵심 개념**

- **Perceptual Loss + Patch GAN**: 단순 픽셀 MSE/L1 손실은 흐릿한 복원을 초래하므로, 지각적 유사도(VGG feature 기반)와 patch discriminator를 함께 사용해 선명한 복원을 보장
- **KL-regularization (KL-reg)**: 잠재 공간이 표준 정규분포에 가까워지도록 약한 KL penalty 적용 → VAE와 유사
- **VQ-regularization (VQ-reg)**: Decoder 내부에 벡터 양자화(Vector Quantization) 레이어를 삽입 → VQGAN과 유사
- **적절한 압축률의 중요성**: $f=4$~$8$ 범위가 품질과 효율의 최적 균형. $f=1$(픽셀 공간)은 너무 느리고, $f=32$는 과도한 압축으로 품질 저하

**수식 — 오토인코더 학습 목적함수**

$$\mathcal{L}_{Autoencoder} = \min_{\mathcal{E}, \mathcal{D}} \max_{\psi} \left( \mathcal{L}_{rec}(x, \mathcal{D}(\mathcal{E}(x))) - \mathcal{L}_{adv}(\mathcal{D}(\mathcal{E}(x))) + \log D_\psi(x) + \mathcal{L}_{reg}(x; \mathcal{E}, \mathcal{D}) \right)$$

**수식 설명**
- **$\mathcal{L}_{rec}$**: 재구성 손실 — 원본과 복원 이미지가 얼마나 유사한지 (L1/LPIPS 기반)
- **$\mathcal{L}_{adv}$**: 적대적 손실 — Discriminator $D_\psi$를 속여 생성된 이미지가 진짜처럼 보이도록
- **$\log D_\psi(x)$**: 진짜 이미지에 대한 Discriminator의 판단값 (높을수록 진짜로 분류)
- **$\mathcal{L}_{reg}$**: 잠재 공간 정규화 (KL-reg: 표준 정규분포 유도, VQ-reg: 이산 코드북 사용)
- **직관**: "원본과 비슷하게 복원(재구성)하되, 전문가(Discriminator)가 봐도 진짜 같아야 하고, 잠재 공간이 너무 불규칙해지면 안 된다"

---

### 3.2 Latent Diffusion Models (잠재 확산 모델)

**요약**

압축된 잠재 공간 $z$에서 Diffusion Model을 학습한다. Forward process(노이즈 추가)는 고정된 Markov Chain이고, 모델은 각 스텝에서 노이즈를 예측하는 denoising network $\epsilon_\theta$를 학습한다. 핵심 backbone은 **time-conditional UNet**이다.

**핵심 개념**

- **Latent Space의 이점**: 픽셀 대비 $f^2$배 작은 공간에서 연산하므로 훈련 속도와 메모리 효율 대폭 향상
- **UNet with Attention**: UNet의 중간 레이어에 Self-attention과 Cross-attention을 삽입하여 전역적 구조와 조건 정보를 함께 처리
- **Reweighted ELBO**: 모든 노이즈 수준을 동일하게 가중하는 단순화된 목적함수 사용 (DDPM과 동일)

**수식 — LDM 학습 목적함수**

$$\mathcal{L}_{LDM} := \mathbb{E}_{\mathcal{E}(x), \epsilon \sim \mathcal{N}(0,1), t} \left[ \| \epsilon - \epsilon_\theta(z_t, t) \|_2^2 \right]$$

**수식 설명**

- **$\mathcal{E}(x)$**: Autoencoder Encoder가 이미지를 잠재 표현으로 압축한 것
- **$\epsilon \sim \mathcal{N}(0,1)$**: 가우시안 노이즈를 무작위로 샘플링
- **$z_t$**: 시간 $t$에서 노이즈가 섞인 잠재 벡터. $z_0$(깨끗한 latent)에 점진적으로 노이즈를 더한 것
- **$\epsilon_\theta(z_t, t)$**: 모델이 $z_t$를 보고 예측한 노이즈
- **$\| \cdot \|_2^2$**: 예측된 노이즈와 실제 노이즈의 L2 거리(Mean Squared Error)
- **직관**: "잡음이 섞인 잠재 벡터를 입력받아 '어떤 노이즈가 섞였는지'를 정확히 예측하도록 학습한다. 잘 학습하면 역방향으로 노이즈를 제거(denoising)해서 새로운 이미지를 생성할 수 있다."

---

### 3.3 Conditioning Mechanisms (조건부 생성)

**요약**

텍스트, 시맨틱맵, 이미지 등 다양한 조건 입력 $y$를 처리하기 위해 **Cross-Attention 메커니즘**을 UNet backbone에 추가한다. 조건 $y$는 도메인별 인코더 $\tau_\theta$로 중간 표현 $\tau_\theta(y) \in \mathbb{R}^{M \times d_\tau}$로 변환된 후, UNet의 중간 레이어에서 Cross-Attention을 통해 통합된다.

**핵심 개념**

- **Domain-specific Encoder $\tau_\theta$**: 텍스트의 경우 BERT/Transformer, 시맨틱맵의 경우 CNN 등 입력 모달리티에 맞는 인코더를 유연하게 선택
- **Cross-Attention**: UNet의 Query는 이미지 feature, Key·Value는 조건 표현에서 계산. 이를 통해 "이미지의 어느 부분이 조건의 어떤 부분에 주의해야 하는지"를 학습
- **멀티모달 학습**: 동일한 아키텍처로 텍스트→이미지, 레이아웃→이미지, 시맨틱→이미지 등 다양한 조건부 생성 태스크 수행 가능

**수식 — Cross-Attention**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) \cdot V$$

$$Q = W_Q^{(i)} \cdot \varphi_i(z_t), \quad K = W_K^{(i)} \cdot \tau_\theta(y), \quad V = W_V^{(i)} \cdot \tau_\theta(y)$$

**수식 설명**

- **$Q$ (Query)**: UNet 중간 레이어의 이미지 feature $\varphi_i(z_t)$에서 만들어짐 — "나는 어떤 정보를 찾고 있는가?"
- **$K$ (Key)**: 조건 표현 $\tau_\theta(y)$에서 만들어짐 — "나는 어떤 정보를 가지고 있는가?"
- **$V$ (Value)**: 조건 표현 $\tau_\theta(y)$에서 만들어짐 — "실제로 전달할 정보의 내용"
- **$\frac{QK^T}{\sqrt{d}}$**: Query와 Key의 유사도 계산. $\sqrt{d}$로 나누는 이유는 차원이 커질수록 내적값이 커져 gradient가 소실되는 문제를 방지하기 위함
- **softmax**: 유사도를 확률(합=1)로 변환해 어느 조건 토큰에 얼마나 집중할지 결정
- **직관**: "이미지의 각 위치(Q)가 텍스트 토큰들(K) 중 자신과 가장 관련 있는 것을 찾아, 그 내용(V)을 가져와 합성한다. 예: '파란 하늘'을 생성하는 위치는 'blue'와 'sky' 토큰에 높은 가중치를 부여"

**수식 — 조건부 LDM 학습 목적함수**

$$\mathcal{L}_{LDM} := \mathbb{E}_{\mathcal{E}(x), y, \epsilon \sim \mathcal{N}(0,1), t} \left[ \| \epsilon - \epsilon_\theta(z_t, t, \tau_\theta(y)) \|_2^2 \right]$$

**수식 설명**

- 조건 없는 LDM 목적함수에 조건 정보 $\tau_\theta(y)$를 추가한 형태
- **$y$**: 조건 입력 (텍스트 프롬프트, 시맨틱맵, 클래스 레이블 등)
- **$\tau_\theta(y)$**: 조건 인코더가 변환한 중간 표현
- **$\epsilon_\theta(z_t, t, \tau_\theta(y))$**: 노이즈가 낀 잠재 벡터와 조건 정보를 함께 받아 노이즈를 예측
- $\tau_\theta$와 $\epsilon_\theta$는 동시에(jointly) 최적화됨

---

## Section 4: Experiments

**요약**

LDM은 다양한 이미지 합성 태스크에서 기존 SOTA를 달성하거나 이에 근접하면서, 계산 비용을 크게 절감한다.

**핵심 결과**

- **무조건 이미지 생성**: CelebA-HQ에서 FID 5.11 달성 → GAN 포함 이전 모든 likelihood-based 모델 능가
- **텍스트→이미지 (MS-COCO)**: 1.45B 파라미터 LDM-KL-8이 GLIDE(3.5B) 대비 파라미터 1/2 이하로 유사한 FID 달성
- **클래스 조건부 이미지 생성 (ImageNet 256×256)**: LDM-4가 ADM(600M) 대비 40% 적은 파라미터(400M)로 경쟁력 있는 성능
- **슈퍼해상도 (×4)**: LDM-SR이 SR3 대비 FID 기준 우수, 사용자 선호도 70.6% vs 29.4%
- **인페인팅**: FID 1.50으로 LaMa, CoModGAN 등 전문 아키텍처를 능가
- **훈련 효율**: 픽셀 기반 Diffusion(LDM-1) 대비 LDM-{4~8}은 2.7× 이상 throughput 향상, FID도 38점 개선

**최적 다운샘플링 팩터**

| Factor | 특징 |
|--------|------|
| f=1 | 픽셀 공간과 동일, 학습 느림 |
| f=2 | 개선 미미, 학습 여전히 느림 |
| **f=4~8** | **품질·효율 최적 균형** |
| f=16~32 | 과도한 압축 → 품질 저하 |

---

## Section 5: Limitations & Societal Impact

**요약**

**한계:**
- LDM은 픽셀 기반 DM보다 빠르지만 GAN의 단일 forward pass 대비 여전히 느린 sequential sampling
- 잠재 공간의 압축으로 인해 픽셀 수준의 정밀도가 요구되는 작업(예: 슈퍼해상도 세밀 텍스처)에서는 병목 발생
- $f=4$ autoencoder를 사용할 때 일부 정보 손실이 발생할 수 있음

**사회적 영향:**
- 고품질 이미지 생성의 민주화 → 접근성 향상
- 딥페이크·허위 정보 생성 악용 가능성
- 학습 데이터의 개인정보 노출 위험
- 데이터셋에 내재된 편향이 생성 결과에 반영될 수 있음

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **Latent Diffusion Model (LDM)** | Autoencoder의 잠재 공간에서 Diffusion을 수행하는 2단계 생성 모델 |
| **Perceptual Compression** | Autoencoder가 지각적으로 무의미한 정보를 제거해 효율적 latent 표현 학습 |
| **Semantic Compression** | Diffusion Model이 latent에서 이미지의 의미론적 구조를 학습 |
| **KL-reg** | 잠재 공간을 표준 정규분포로 유도하는 정규화 (VAE 방식) |
| **VQ-reg** | 잠재 공간을 이산 코드북으로 양자화하는 정규화 (VQGAN 방식) |
| **Cross-Attention Conditioning** | 텍스트·시맨틱맵 등 외부 조건을 UNet에 통합하는 범용 메커니즘 |
| **$\tau_\theta$** | 조건 입력을 중간 표현으로 변환하는 도메인별 인코더 |
| **Downsampling Factor f** | 공간 압축률. f=4~8이 품질·효율 최적 균형 |
| **Classifier-Free Guidance** | 조건 없는 생성과 조건부 생성을 결합해 품질↑·다양성↓ 제어 |

---

## 결론 및 시사점

LDM은 Diffusion Model의 고질적 문제였던 **계산 비용**을 두 단계 분리(perceptual + semantic compression)로 해결한 핵심 논문이다. 이미지 품질을 유지하면서 훈련 속도와 추론 효율을 대폭 개선했으며, Cross-Attention 기반 조건부 생성 메커니즘을 통해 텍스트→이미지, 인페인팅, 슈퍼해상도 등 다양한 태스크를 단일 아키텍처로 처리한다.

**자율주행 합성 데이터 관점에서의 시사점:**
- DriveDreamer·MagicDrive·DriveArena 등 자율주행 도메인의 조건부 비디오/이미지 생성 모델이 모두 LDM의 아키텍처(Latent Space + Cross-Attention Conditioning)를 직접 계승
- 특히 **MagicDrive**는 BEV 맵·3D 바운딩박스·카메라 포즈를 $\tau_\theta$로 인코딩해 Cross-Attention으로 통합하는 방식이 LDM Section 3.3을 그대로 적용한 것
- LDM의 "잠재 공간에서의 효율적 생성"은 실시간 또는 대규모 합성 데이터 생성 파이프라인 구축에 핵심적인 설계 원칙
