---
title: "GAIA-1: A Generative World Model for Autonomous Driving"
date: 2026-04-14T14:00:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving"]
tags: ["Autonomous Driving", "World Model", "Generative Model", "Video Generation", "Diffusion"]
---

## 개요

- **저자**: Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, Giancarlo Corrado
- **소속**: Wayve (research@wayve.ai)
- **발행년도**: 2023 (arXiv:2309.17080, 29 Sep 2023)
- **주요 내용**: 비디오, 텍스트, 행동(action)을 입력으로 받아 사실적인 자율주행 시나리오 영상을 생성하는 **생성형 World Model GAIA-1**을 제안. 모든 입력을 이산 토큰 시퀀스로 변환하고, 다음 토큰 예측(next-token prediction) 방식으로 월드 모델을 학습한 뒤, Video Diffusion Decoder로 고품질 영상을 복원함. LLM의 스케일링 법칙(scaling law)이 자율주행 세계 모델에도 적용됨을 최초로 검증.

## 목차

- Section 1: Introduction — 미래 예측의 중요성과 기존 World Model의 한계
- Section 2: Model — 이미지/텍스트/액션 인코딩, Image Tokenizer, World Model, Video Decoder
- Section 3: Data — 4700시간 런던 주행 데이터 및 균형 샘플링 전략
- Section 4: Training Procedure — 각 컴포넌트별 학습 설정 및 하이퍼파라미터
- Section 5: Inference — World Model 샘플링 전략, Classifier-free Guidance
- Section 6: Scaling — LLM 스케일링 법칙의 자율주행 적용
- Section 7: Capabilities and Emerging Properties — 장거리 시나리오 생성, 다중 미래 예측, 세밀한 제어
- Section 8: Related Work — 비디오 생성 모델, World Model, 스케일링 연구
- Section 9: Conclusions — 결론 및 향후 연구 방향

---

## Section 1: Introduction

**요약**

자율주행 시스템이 안전하게 작동하려면 자신의 행동에 따라 세상이 어떻게 변할지를 예측할 수 있어야 합니다. 기존 World Model들은 시뮬레이션 데이터에 의존하거나 저차원 표현을 사용해 실제 세계의 복잡성을 포착하기 어렵다는 한계가 있었습니다.

GAIA-1은 이 문제를 **비지도 시퀀스 모델링**으로 재구성합니다. 모든 입력(영상, 텍스트, 액션)을 이산 토큰으로 변환하고, GPT와 동일한 방식으로 다음 토큰을 예측하도록 학습합니다. 그 결과로 다음 능력들이 창발(emerging property)합니다:

- 고수준 구조와 장면 다이나믹스 학습 (신호등, 차선, 보행자 등)
- 훈련 데이터를 넘어서는 일반화 및 창의적 생성
- 문맥 인식(contextual awareness) — 3D 기하 이해 포함
- 텍스트 및 액션 기반의 세밀한 자아 차량 제어

**핵심 개념**

- **World Model**: 환경에 대한 구조화된 표현을 학습하여 미래 상태를 예측하는 모델. 강화학습의 look-ahead search나 policy learning에 활용될 수 있음
- **Next-token prediction**: GPT와 동일한 자기회귀(autoregressive) 방식으로, 이전 토큰 시퀀스가 주어졌을 때 다음 토큰의 확률 분포를 예측하는 학습 목표
- **창발적 속성(Emerging Properties)**: 명시적으로 가르치지 않았으나 대규모 자기지도 학습을 통해 자연스럽게 나타나는 능력들

---

## Section 2: Model

**요약**

GAIA-1은 세 가지 학습 가능한 컴포넌트로 구성됩니다:

```
[비디오 입력]  →  Image Encoder (VQ-GAN)  →  이미지 토큰 z
[텍스트 입력]  →  T5 인코더 + Linear    →  텍스트 토큰 c
[액션 입력]   →  Linear               →  액션 토큰 a

(c, z, a) 시퀀스  →  World Model (Transformer)  →  다음 이미지 토큰 예측
                                                         ↓
                                            Video Diffusion Decoder  →  고품질 영상
```

### 2.1 인코딩: 비디오, 텍스트, 액션

**이미지 토큰**: 각 프레임(288×512 해상도)을 DINO-distilled VQ-GAN으로 18×32 = **576개의 이산 토큰**으로 압축. 다운샘플링 비율 D=16으로 약 470배 압축.

**텍스트 토큰**: 사전학습된 T5 모델로 인코딩한 뒤 선형 레이어를 통해 d차원으로 투영. 매 타임스텝 **m=32개 토큰** 생성.

**액션 토큰**: 속도(speed)와 곡률(curvature) 두 개의 스칼라 값을 선형 레이어로 투영. 매 타임스텝 **l=2개 토큰** 생성.

각 타임스텝 t에서 입력 순서는 **텍스트 → 이미지 → 액션** 순이며, 팩토화된 시공간 위치 임베딩을 사용합니다.

**핵심 개념**

- **VQ-GAN (Vector Quantized GAN)**: 이미지를 연속 특징이 아닌 유한한 코드북(codebook)의 이산 인덱스로 표현하는 오토인코더. 언어 모델이 단어를 처리하듯 이미지를 처리할 수 있게 함
- **DINO Distillation**: VQ-GAN 학습 시 자기지도 학습 모델 DINO의 특징을 cosine similarity loss로 증류(distill)하여, 압축된 토큰이 의미론적 정보를 더 풍부하게 담도록 유도
- **팩토화된 시공간 위치 임베딩**: T개의 시간 임베딩 × (m+n+l)개의 공간 임베딩으로 분리하여 효율적으로 위치 정보를 표현 (d=4096)

### 2.2 Image Tokenizer

이미지 토크나이저의 두 가지 목표:
1. **시퀀스 압축**: 원본 픽셀의 중복/노이즈를 제거해 Transformer가 처리 가능한 길이로 단축
2. **의미론적 유도**: DINO distillation으로 고주파 노이즈 대신 의미 정보(차량, 도로, 하늘 등) 위주로 압축

학습 손실:
- **이미지 재구성 손실**: $L_1$, $L_2$, Perceptual loss, GAN loss의 가중합
- **양자화 손실**: Commitment loss로 임베딩 벡터를 코드북에 고정
- **Inductive bias 손실**: DINO 특징과의 cosine similarity loss

### 2.3 World Model

World Model은 6.5B 파라미터의 **자기회귀 Transformer**입니다.

학습 목표:

$$L_{\text{world model}} = -\sum_{t=1}^{T} \sum_{i=1}^{n} \log p(z_{t,i} | \mathbf{z}_{t,i<}, z_{t'<t}, \mathbf{c}_{\leq t}, \mathbf{a}_{<t})$$

**수식 설명**:
- **$z_{t,i}$**: 타임스텝 $t$의 $i$번째 이미지 토큰
- **$\mathbf{z}_{t,i<}$**: 같은 타임스텝 내 앞선 이미지 토큰들 (인과적 마스킹)
- **$z_{t'<t}$**: 이전 타임스텝들의 모든 이미지 토큰
- **$\mathbf{c}_{\leq t}$**: 현재 및 이전 타임스텝의 텍스트 토큰
- **$\mathbf{a}_{<t}$**: 이전 타임스텝들의 액션 토큰
- 전체 수식 의미: 주어진 과거 이미지, 텍스트, 액션 토큰들을 조건으로 현재 이미지 토큰을 예측하는 조건부 확률의 로그 합을 최대화 (= 크로스 엔트로피 최소화)

학습 시 **조건부 dropout** 전략으로 3가지 모드를 동시에 학습:
- 무조건부 생성 (20%)
- 액션 조건부 생성 (40%)
- 텍스트 조건부 생성 (40%)

### 2.4 Video Decoder

Video Decoder는 2.6B 파라미터의 **멀티태스크 비디오 Diffusion 모델**입니다. World Model이 생성한 이미지 토큰 시퀀스를 고해상도 픽셀 공간 영상으로 변환합니다.

학습 손실 (v-parameterization):

$$L_{\text{video}} = \mathbb{E}_{x,t',z,m} \left[ \| \epsilon_\theta(\mathbf{x}^{t'}, t', \mathbf{z}, \mathbf{m}) - \epsilon \|_2^2 \right]$$

**수식 설명**:
- **$\epsilon_\theta$**: 학습 대상인 denoising 비디오 모델
- **$\epsilon$**: v-parameterization에 따른 denoising 타겟
- **$t'$**: 이산 확산 타임스텝 (노이즈 수준)
- **$\mathbf{x}^{t'} = \alpha_{t'}\mathbf{x} + \sigma_{t'}\epsilon$**: 노이즈가 추가된 비디오 프레임
- **$\mathbf{z} = (z_1,...,z_{T'}) = E_\theta(\mathbf{x})$**: World Model에서 생성된 이미지 토큰 (conditioning)
- **$\mathbf{m}$**: 어떤 프레임을 예측할지 지정하는 마스크
- 의미: 노이즈가 추가된 프레임에서 노이즈를 제거하되, 이미지 토큰과 컨텍스트 프레임을 조건으로 사용

4가지 학습 태스크를 동시에 수행:
1. **이미지 생성**: 단일 프레임 독립적 생성
2. **비디오 생성**: 시간 레이어 활성화, 공간적 temporal attention
3. **자기회귀 비디오 생성**: 이전 프레임들을 컨텍스트로 사용
4. **비디오 보간(interpolation)**: 두 프레임 사이의 중간 프레임 생성

---

## Section 3: Data

**요약**

- **데이터**: 2019~2023년 런던에서 수집한 **4,700시간**의 독점 주행 데이터 (약 4억 2천만 개의 고유 이미지)
- **검증 세트**: 400시간 (훈련 경로 안/밖의 지오펜스로 분리하여 과적합 및 일반화 모니터링)

**균형 샘플링 전략**:
- Image Tokenizer: 위도, 경도, 날씨 범주로 균형화
- World Model & Video Decoder: 위 3가지 + 조향 행동, 속도 행동 범주 추가
- 샘플링 지수 0.5로 경험적 분포와 균일 분포의 중간점 설정

**핵심 개념**

- **역비례 가중치 샘플링**: 특정 지역이나 날씨 조건에 데이터가 몰린 경우, 드문 조건의 샘플을 더 자주 보여주어 모델이 다양한 시나리오를 골고루 학습하도록 유도
- **지오펜스(Geofence) 검증**: 훈련에 포함되지 않은 도로 구간을 별도 검증셋으로 사용하여 진정한 일반화 성능 측정

---

## Section 4: Training Procedure

**요약**

세 컴포넌트의 주요 학습 설정:

| 컴포넌트 | 파라미터 수 | 하드웨어 | 학습 시간 |
|---------|-----------|---------|---------|
| Image Tokenizer | 0.3B | 32× A100 80GB | 4일 (200k steps) |
| World Model | 6.5B | 64× A100 80GB | 15일 (100k steps) |
| Video Decoder | 2.6B | 32× A100 80GB | 15일 (300k steps) |

**World Model 주요 설정**:
- 시퀀스 길이: T=26 프레임 @ 6.25Hz (4초 분량)
- 전체 시퀀스 길이: T × (m+n+l) = 26 × 610 = **15,860 토큰**
- Top-k=50 샘플링 (argmax 대비 다양성, uniform 대비 안정성 균형)
- FlashAttention v2 + DeepSpeed ZeRO-2 학습 전략

---

## Section 5: Inference

**요약**

### 5.1 World Model 샘플링

World Model은 주어진 컨텍스트로부터 n 스텝 순방향 예측을 수행합니다. 토큰 샘플링 전략의 비교:

- **Argmax**: 다양성 없음, 반복적 미래 생성
- **Uniform sampling**: 비현실적 토큰 포함 가능 (out-of-distribution)
- **Top-k=50 sampling**: 현실과 유사한 perplexity 분포, 균형 잡힌 선택

긴 영상 생성 시 슬라이딩 윈도우 방식으로 컨텍스트 길이 초과 문제 해결.

### 5.2 Classifier-free Guidance (텍스트 조건부 생성)

텍스트-이미지 정렬을 강화하기 위해 Classifier-free Guidance를 적용합니다:

$$l_{\text{final}} = (1 + t) l_{\text{conditioned}} - t \cdot l_{\text{unconditioned}}$$

**수식 설명**:
- **$l_{\text{final}}$**: 최종 샘플링에 사용되는 logit
- **$l_{\text{conditioned}}$**: 텍스트 프롬프트를 조건으로 한 logit
- **$l_{\text{unconditioned}}$**: 조건 없이 생성한 logit
- **$t$ (guidance scale)**: 텍스트 정렬 강도 조절 파라미터. 높을수록 텍스트에 충실하지만 다양성 감소
- 의미: 텍스트 조건부와 무조건부의 차이를 scale factor t로 증폭하여 텍스트 방향으로 강하게 이끌기

**가이던스 스케줄링**: 초반 토큰은 높은 guidance (텍스트 준수), 이후 프레임은 낮은 guidance (다양성 확보)로 점진적 감소.

**Negative prompting**: 양수 프롬프트와 음수 프롬프트의 logit을 활용하여 원하는 특성만 강조하고 원치 않는 특성을 제거.

---

## Section 6: Scaling

**요약**

GAIA-1의 World Model 학습은 LLM과 동일하게 **비지도 시퀀스 모델링**으로 구성되어 있어, LLM에서 검증된 **스케일링 법칙**이 그대로 적용됩니다.

- 모델 크기: 0.65M ~ 6.5B 파라미터까지 5단계 실험
- 검증 교차 엔트로피 vs 컴퓨트(FLOPs) 관계가 **거듭제곱 법칙(power law)**을 따름

$$f(x) = c + (x/a)^b$$

- GAIA-1 (6.5B)의 최종 성능은 **20배 적은 컴퓨트**로 학습된 소형 모델들로 예측 가능
- **시사점**: 더 많은 데이터와 컴퓨트를 투입할수록 예측 가능하게 성능 향상 → 자율주행 World Model의 확장 가능성 입증

**핵심 개념**

- **스케일링 법칙(Scaling Law)**: 모델 크기, 데이터 양, 컴퓨트가 증가함에 따라 성능이 예측 가능한 수식적 규칙에 따라 향상되는 현상. GPT-3, PaLM 등 LLM에서 잘 알려져 있으며, GAIA-1은 이것이 비디오/세계 모델에도 성립함을 최초로 보임

---

## Section 7: Capabilities and Emerging Properties

**요약**

GAIA-1이 자기지도 학습만으로 획득한 세 가지 주요 창발 능력:

### 7.1 장거리 주행 시나리오 생성

**상상만으로** 분 단위의 장거리 주행 영상 생성 가능. 복잡한 도로 배치, 건물, 보행자, 기상 조건을 일관성 있게 유지합니다. 모델이 세계를 지배하는 물리적 규칙을 이해했다는 증거입니다.

### 7.2 다중 가능한 미래 예측

동일한 초기 프롬프트에서 Top-k 반복 샘플링으로 **여러 개의 그럴듯한 미래**를 동시에 생성:
- 양보/비양보 상황에서의 상호작용
- 라운드어바웃에서의 직진/우회전
- 교통 밀도 변화에 따른 시나리오

### 7.3 세밀한 자아 차량 및 장면 제어

**텍스트만으로 장면 생성** (날씨, 조명 조건):
- "It is sunny" / "It's raining" / "It is foggy" / "It is snowing"
- "It is daytime" / "It is twilight" / "It is night"

**분포 외(out-of-distribution) 액션 조건부 생성**:
- 훈련 데이터에 없는 극단적 조향 (strong left / strong right) 생성
- 자아 차량의 행동에 다른 에이전트들이 반응하는 장면까지 생성
- 예: 자아 차량이 역방향 차선으로 이동 시 맞은편 차량이 회피 기동

**핵심 개념**

- **분포 외 일반화(Out-of-distribution generalization)**: 훈련 데이터에 존재하지 않던 상황을 올바르게 처리하는 능력. 전문 운전자 데이터에는 극단적 위험 조향이 없지만, GAIA-1은 그 결과를 물리적으로 타당하게 예측
- **인과성 이해(Causality)**: 자아 차량의 행동 → 다른 에이전트의 반응이라는 인과 관계를 모델이 내재적으로 습득

---

## Section 8: Related Work

**요약**

| 분야 | 기존 방법 | 한계 | GAIA-1의 접근 |
|------|---------|------|-------------|
| 비디오 생성 | VAE, GAN, Diffusion | 미래 예측에 특화되지 않음 | World Model + Diffusion Decoder 결합 |
| World Model | RNN 기반, 저차원 표현 | 실제 도로 복잡성 미반영 | 실제 4700시간 데이터, 고차원 픽셀 공간 |
| 스케일링 | LLM에서만 검증됨 | 비디오/세계 모델 미적용 | AV 세계 모델에서 동일 법칙 성립 확인 |

---

## Section 9: Conclusions

**요약**

GAIA-1은 자율주행을 위한 생성형 World Model의 가능성을 세 가지 측면에서 입증했습니다:

1. **안전성**: 자아 차량의 행동이 환경에 미치는 영향을 이해하여 더 안전한 의사결정 지원
2. **지능성**: 동적 장면의 복잡한 상호작용을 포착하는 풍부한 표현 학습
3. **데이터 생성**: 적대적 예시를 포함한 무한한 합성 데이터 생성으로 AV 훈련/검증 가속

**현재 한계**:
- 자기회귀 생성이 실시간으로 동작하지 않음 (병렬 샘플링으로 완화 가능)

**향후 방향**: 더 많은 데이터와 컴퓨트로 예측 가능하게 성능 향상 가능 (스케일링 법칙 확인)

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **GAIA-1** | Generative AI for Autonomy. 비디오·텍스트·액션을 토큰으로 통합하여 다음 토큰 예측으로 자율주행 세계를 학습하는 생성 모델 |
| **Image Tokenizer** | DINO-distilled VQ-GAN. 이미지 프레임을 576개의 이산 토큰으로 압축. 의미론적 정보 보존이 핵심 |
| **World Model** | 6.5B Transformer. 과거 이미지·텍스트·액션 토큰으로 다음 이미지 토큰을 자기회귀적으로 예측 |
| **Video Diffusion Decoder** | 2.6B 멀티태스크 Diffusion 모델. World Model의 토큰을 고해상도 픽셀 영상으로 변환 |
| **Classifier-free Guidance** | 텍스트 조건부/무조건부 logit 차이를 scale factor로 증폭해 텍스트-영상 정렬 강화 |
| **Scaling Law** | 모델 크기·데이터·컴퓨트 증가에 따른 예측 가능한 성능 향상 법칙. LLM뿐 아니라 AV 세계 모델에도 적용됨 확인 |
| **창발적 속성** | 명시적으로 가르치지 않았으나 자기지도 학습으로 자연 발생한 능력: 3D 기하 이해, 인과성 추론, OOD 일반화 등 |

---

## 결론 및 시사점

GAIA-1은 **UniAD, VAD** 등 인식·계획 스택 논문들과 달리, 자율주행을 **생성 모델 문제**로 바라봅니다. 핵심 기여는 다음과 같습니다:

1. **패러다임 전환**: 명시적 인식→예측→계획 파이프라인 대신, 월드 모델 하나로 모든 것을 학습
2. **스케일링 가능성 입증**: LLM처럼 데이터와 컴퓨트를 늘리면 자율주행 세계 모델도 체계적으로 개선됨
3. **Neural Simulator**: 포토리얼리스틱 영상 생성으로 HUGSIM 같은 시뮬레이터의 **생성형 대안** 제시
4. **NAVSIM과의 연결**: GAIA-1이 생성한 다양한 시나리오를 NAVSIM 스타일 비반응형 벤치마크와 결합하면 더 풍부한 평가 환경 구축 가능

> **로드맵 상의 위치**: `UniAD/VAD (E2E 계획)` → **GAIA-1 (생성형 World Model)** → `HUGSIM (3DGS 기반 Neural Simulator)`  
> UniAD가 "어떻게 주행할지"를 학습한다면, GAIA-1은 "세상이 어떻게 작동하는지"를 학습합니다.


---

*관련 논문: [LDM](/posts/papers/high-resolution-image-synthesis-with-latent-diffusion-models/), [DDPM](/posts/papers/denoising-diffusion-probabilistic-models/), [DriveDreamer](/posts/papers/DriveDreamer/), [DriveArena](/posts/papers/DriveArena/), [MagicDrive](/posts/papers/magicdrive-street-view-generation-3d-geometry-control/), [UniAD](/posts/papers/uniad-planning-oriented-autonomous-driving/), [nuScenes](/posts/papers/nuscenes-multimodal-dataset-autonomous-driving/)*
