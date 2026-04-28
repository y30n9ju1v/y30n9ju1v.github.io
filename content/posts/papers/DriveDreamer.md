---
title: "DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving"
date: 2026-04-19T07:00:00+09:00
draft: false
tags: ["World Model", "Autonomous Driving", "Diffusion Model", "Video Generation", "nuScenes"]
categories: ["Papers", "Autonomous Driving", "Generative Models"]
---

## 개요
- **저자**: Xiaofeng Wang, Zheng Zhu, Guan Huang, Xinze Chen, Jiagang Zhu, Jiwen Lu (GigaAI, Tsinghua University)
- **발행년도**: 2023
- **arXiv**: 2309.09777v2
- **주요 내용**: 실제 주행 영상에서 구축된 최초의 자율주행 World Model. 구조화된 교통 제약(HDMap, 3D Box)과 텍스트 프롬프트, 주행 액션을 조건으로 미래 주행 영상과 주행 정책을 동시에 생성한다.

## 한계 극복

기존 World Model 연구들은 게임 환경이나 시뮬레이션 환경에 집중되어 있었으며, 실제 주행 시나리오의 복잡한 구조적 제약을 충분히 모델링하지 못했다.

- **기존 한계 1 — 시뮬레이션 편향**: ISO-Dream, MILE, SEM2 등 기존 World Model들은 주로 게임 엔진 기반 시뮬레이션 환경에서 학습되어, 실제 도로의 복잡한 동적·정적 요소 표현에 취약했다.
- **기존 한계 2 — 광대한 탐색 공간**: 실제 픽셀 공간에서 주행 장면을 모델링하면 탐색 공간이 지나치게 넓어 샘플링 효율이 떨어진다.
- **기존 한계 3 — 구조적 제약 미반영**: 기존 영상 생성 모델은 차선, 3D 바운딩 박스 같은 교통 구조 정보를 조건으로 활용하지 않아 현실적 제약을 벗어난 영상이 생성됐다.
- **이 논문의 접근 방식**: 실세계 nuScenes 데이터셋 기반, HDMap·3D Box를 구조화 조건으로 활용하는 Auto-DM(Autonomous-driving Diffusion Model)과 ActionFormer를 결합한 2단계 학습 파이프라인으로 위 한계를 동시에 극복한다.

## 목차
- Chapter 1: Introduction
- Chapter 2: Related Work (Diffusion Model / Video Generation / World Models)
- Chapter 3: DriveDreamer (Auto-DM, ActionFormer, 2단계 학습)
- Chapter 4: Experiments
- Chapter 5: Discussion and Conclusion

---

## Chapter 1: Introduction

**요약**

AGI와 구현 AI(embodied AI)에서 얻은 통찰을 바탕으로, 자율주행에서 World Model의 중요성이 부각되고 있다. World Model은 다양하고 현실적인 주행 영상을 생성할 수 있어 긴 꼬리(long-tail) 시나리오 대응, 주행 인지 모델 학습, 엔드-투-엔드 자율주행에 유망하다. 그러나 기존 연구들은 시뮬레이션 환경에 치우쳐 있어 실제 도로 상황을 제대로 표현하지 못한다. DriveDreamer는 이 공백을 메우는 최초의 실세계 기반 자율주행 World Model이다.

**핵심 개념**

- **World Model**: 환경의 동적 모델로, 과거 관측으로부터 미래 상태를 예측하고 다양한 주행 영상을 생성하는 시스템
- **Long-tail scenario**: 실제 주행에서 드물게 발생하지만 안전에 치명적인 엣지 케이스 (예: 역주행 차량, 갑작스러운 보행자 진입)
- **End-to-end driving**: 센서 입력부터 주행 제어 출력까지 하나의 신경망으로 처리하는 자율주행 접근법

---

## Chapter 2: Related Work

**요약**

관련 연구는 세 갈래로 나뉜다: (1) 조건부 생성을 가능케 하는 Diffusion Model (ControlNet, T2I-Adapter 등), (2) VAE·GAN·Flow 기반 Video Generation 모델, (3) VAE·LSTM 기반 World Model (Dreamer, ISO-Dream, MILE, SEM2 등). DriveDreamer는 이 세 영역을 통합하며, 실제 주행 데이터에서 구조화된 조건을 활용하는 점에서 차별화된다.

**핵심 개념**

- **Diffusion Model**: 데이터에 점진적으로 노이즈를 추가한 뒤(forward process) 이를 역방향으로 제거(reverse process)하여 샘플을 생성하는 확률적 생성 모델
- **ControlNet**: 깊이 맵, 세그멘테이션 맵 등 구조 조건을 Diffusion Model에 주입하는 제어 네트워크
- **ISO-Dream**: 시각 다이나믹스를 제어 가능한 상태와 불가능한 상태로 분리하는 World Model
- **BEV (Bird's Eye View)**: 카메라를 하늘에서 내려다보는 시점으로 변환한 표현으로, 차량 간 공간 관계 파악에 유리

---

## Chapter 3: DriveDreamer

**요약**

DriveDreamer는 두 핵심 모듈(Auto-DM, ActionFormer)과 2단계 학습 파이프라인으로 구성된다. 초기 참조 프레임 $I_0$와 도로 구조 정보(HDMap $H_0$, 3D Box $B_0$)를 입력받아, ActionFormer가 잠재 공간에서 미래 도로 구조 특징을 예측하고, Auto-DM이 이를 조건으로 미래 주행 영상을 생성한다. 텍스트 프롬프트로 날씨·시간대를 제어하고, 과거 액션 이력과 Auto-DM의 멀티스케일 특징으로 미래 주행 액션도 예측한다.

### 3.1 Auto-DM (Autonomous-driving Diffusion Model)

**핵심 개념**

- **Spatially Aligned Conditions**: HDMap을 이미지 평면에 투영한 조건. 노이즈 잠재 변수 $\mathcal{Z}_t$와 채널 방향으로 연결(concat)되어 U-Net에 입력된다.
- **Position Embedding (Gated Self-Attention)**: 3D Box의 카테고리·위치 정보를 Fourier 임베딩과 CLIP 임베딩으로 인코딩한 뒤, 원래 UNet 시각 특징에 게이트 방식으로 주입한다.
- **Temporal Attention Layer**: 프레임 간 시간적 일관성을 유지하기 위한 어텐션 레이어로, 2단계 학습에서 추가된다.

**수식 — Position Embedding**

$$H^p = \mathcal{F}_\alpha([C_e, \text{Fourier}(B)])$$

**수식 설명**
3D 박스 위치 임베딩을 생성하는 수식:
- **$H^p$**: 생성된 위치 임베딩 벡터
- **$C_e$**: CLIP으로 인코딩된 박스 카테고리 특징 (예: "Car", "Pedestrian")
- **$\text{Fourier}(B)$**: 3D 박스 좌표를 푸리에 함수로 변환한 위치 인코딩. 절대 좌표를 주기 함수로 표현해 신경망이 위치를 더 잘 학습하게 한다.
- **$[\cdot]$**: 두 벡터를 이어 붙이는 연결 연산 (concatenation)
- **$\mathcal{F}_\alpha$**: MLP 레이어로, 연결된 특징을 최종 임베딩으로 변환

**수식 — Gated Self-Attention**

$$v = v + \tanh(\eta) \cdot \text{TS}(\mathcal{F}_s([v, H^p]))$$

**수식 설명**
시각 특징 $v$에 위치 임베딩을 게이트 방식으로 주입하는 수식:
- **$v$**: 원래 UNet 시각 특징
- **$\eta$**: 학습 가능한 게이트 파라미터. $\tanh(\eta) \in [-1, 1]$로 주입량을 조절
- **$\mathcal{F}_s$**: Self-Attention 연산
- **$\text{TS}(\cdot)$**: 시각 토큰만 선택하는 Token Selection 연산 (위치 토큰 제외)
- **의미**: 처음에는 $\eta \approx 0$이어서 원래 특징이 거의 그대로 유지되다가, 학습을 거치며 점차 위치 정보가 반영된다.

**수식 — Temporal Attention**

$$\mathcal{F}_t(v) = \text{Reshape}(\mathcal{F}_a(\text{Reshape}(v + \mathcal{T}_\text{pos})))$$

**수식 설명**
프레임 간 시간 관계를 모델링하는 수식:
- **$v$**: 공간 차원 $(N \times C \times H \times W)$의 시각 신호
- **$\mathcal{T}_\text{pos}$**: 시간축 위치를 알려주는 temporal position embedding (사인파 인코딩)
- **Reshape**: 공간 차원을 시간 차원으로 재배열 ($\mathcal{R}^{C \times N H W}$). 이렇게 하면 어텐션이 공간 대신 시간 방향으로 작동
- **$\mathcal{F}_a$**: Self-Attention (시간 방향)

**수식 — Auto-DM 학습 목표**

$$\min_\phi \mathcal{L} = \mathbb{E}_{\mathcal{Z}_0, \epsilon \sim \mathcal{N}(0,\mathbf{I}), t, c} \left[ \|\epsilon - \epsilon_\phi(\mathcal{Z}_t, t, c)\|_2^2 \right]$$

**수식 설명**
노이즈 예측 손실로 Diffusion Model을 학습하는 수식:
- **$\epsilon$**: 실제로 추가된 가우시안 노이즈
- **$\epsilon_\phi(\mathcal{Z}_t, t, c)$**: 모델이 예측한 노이즈. $\mathcal{Z}_t$는 노이즈 잠재 변수, $t$는 타임스텝, $c$는 조건(HDMap, 3D Box, 텍스트)
- **$\|\cdot\|_2^2$**: L2 제곱 오차. 예측 노이즈와 실제 노이즈의 차이를 최소화
- **학습 방식**: Step 1에서는 단일 프레임 이미지 감독, Step 2에서는 연속 영상 감독으로 진행

### 3.2 ActionFormer

**핵심 개념**

- **ActionFormer**: 과거 주행 액션 시퀀스 $\{A_t\}_{t=0}^{T-1}$을 받아 미래 도로 구조 조건을 잠재 공간에서 예측하는 모듈. 픽셀 레벨이 아닌 특징 레벨 예측으로 노이즈에 강인하다.
- **GRU (Gated Recurrent Unit)**: 시계열 순서로 은닉 상태를 업데이트하며 미래 상태를 예측하는 순환 신경망

**수식 — 잠재 변수 샘플링**

$$\mathbf{s}_t \sim \mathcal{N}\left(\mu_\theta(\mathcal{F}_{ca}(\mathbf{h}_t, A_t)),\, \sigma_\theta(\mathcal{F}_{ca}(\mathbf{h}_t, A_t))\,\mathbf{I}\right)$$

**수식 설명**
주어진 은닉 상태와 액션에서 잠재 변수를 샘플링하는 수식:
- **$\mathbf{s}_t$**: 시간 $t$의 잠재 변수 (미래 구조 조건을 결정하는 랜덤 요소)
- **$\mathbf{h}_t$**: 시간 $t$의 은닉 상태 (과거 정보 요약)
- **$A_t$**: 시간 $t$의 주행 액션 (조향각, 속도 등)
- **$\mathcal{F}_{ca}$**: Cross-Attention으로 은닉 상태와 액션을 연결
- **$\mu_\theta, \sigma_\theta$**: Gaussian 분포의 평균·표준편차를 출력하는 MLP 레이어

**수식 — GRU 상태 업데이트**

$$\mathbf{h}_{t+1} = \mathcal{F}_\text{GRU}(\mathbf{h}_t, \mathbf{s}_t)$$

**수식 설명**
GRU가 이전 상태와 잠재 변수를 받아 다음 은닉 상태를 생성하는 수식:
- **$\mathbf{h}_{t+1}$**: 다음 타임스텝의 은닉 상태
- **$\mathbf{h}_t$**: 현재 은닉 상태
- **$\mathbf{s}_t$**: 현재 잠재 변수 (액션에 조건화)
- **$\mathcal{F}_\text{GRU}$**: GRU 연산. 게이트 메커니즘으로 과거 정보의 기억/망각을 조절

**수식 — 변분 하한(ELBO)**

$$\log p(I_{1:T},\, A_{T:T+N}) \geq \mathbb{E}_q \underbrace{\left[\log p(I_{1:T} \mid \mathbf{h}_{0:T}, \mathbf{s}_{0:T-1}, A_{0:T-1}, I_0)\right]}_{\text{video prediction}} + \underbrace{\log p(A_{T:T+N} \mid \mathbf{h}_{0:T}, \mathbf{s}_{0:T-1}, A_{0:T-1}, I_0)}_{\text{action prediction}}$$

**수식 설명**
변분 추론으로 World Model을 학습하는 목표 함수:
- **좌변**: 미래 영상 $I_{1:T}$와 미래 액션 $A_{T:T+N}$의 결합 로그 우도
- **$\geq$**: 직접 최적화가 어려우므로 하한(ELBO)을 대신 최대화
- **video prediction 항**: 모델이 미래 영상을 얼마나 잘 예측하는지 측정 (Gaussian 분포 기반, MSE 손실)
- **action prediction 항**: 모델이 미래 주행 액션을 얼마나 잘 예측하는지 측정 (Laplace 분포 기반, L1 손실)
- **의미**: 영상 예측과 액션 예측이 하나의 목표로 동시에 최적화됨

---

## Chapter 4: Experiments

**요약**

nuScenes 데이터셋(700개 학습 영상, 150개 검증 영상, 12Hz, 6-surround view)으로 평가. 평가 지표는 FID(이미지 품질), FVD(영상 품질), mAP/NDS(3D 검출 성능 향상), L2 궤적 오차 및 충돌률(플래닝 성능)을 사용.

**핵심 결과**

- **영상 생성 품질 (Tab. 2)**: 전체 2단계 학습 적용 시 FID 14.9 / FVD 340.8로 DriveGAN 대비 크게 개선. 1단계만 적용해도 FVD 349.6으로 우수.
- **3D 검출 향상 (Tab. 1)**: Auto-DM으로 생성한 4K 합성 데이터를 학습에 추가하면 FCOS3D mAP +0.7, BEVFusion mAP +3.0 향상.
- **플래닝 성능 (Tab. 3)**: L2 궤적 오차 0.29m로 VAD(0.37m) 대비 우수. 충돌률 0.15%로 대부분의 비교 모델보다 낮음.

**핵심 개념**

- **FID (Fréchet Inception Distance)**: 실제 이미지와 생성 이미지의 특징 분포 거리. 낮을수록 품질이 좋음.
- **FVD (Fréchet Video Distance)**: FID의 영상 버전. 시간적 일관성까지 고려한 영상 품질 지표.
- **Open-loop planning**: 실제 환경과 상호작용 없이 미리 정해진 입력으로 궤적을 평가하는 방식. ST-P3 설정을 따름.

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **Auto-DM** | HDMap·3D Box·텍스트를 조건으로 주행 영상을 생성하는 Stable Diffusion 기반 모델 |
| **ActionFormer** | 주행 액션 시퀀스를 받아 GRU로 미래 도로 구조 특징을 예측하는 모듈 |
| **2단계 학습** | 1단계(구조적 교통 제약 이해) → 2단계(미래 예측 및 액션 생성) |
| **Spatially Aligned Condition** | HDMap을 이미지 평면에 직접 투영해 공간적으로 정렬된 조건으로 사용 |
| **Gated Self-Attention** | 3D Box 위치 임베딩을 학습 가능한 게이트로 조절하며 시각 특징에 주입 |
| **Temporal Attention** | 프레임 간 시간적 일관성을 확보하는 어텐션 레이어 (2단계에서 추가) |
| **CLIP Embedding** | 텍스트·카테고리 정보를 시각-언어 공통 공간으로 임베딩하는 사전학습 모델 |
| **Variational Inference** | ActionFormer의 잠재 변수를 추론하기 위한 변분 추론 (ELBO 최적화) |

---

## 결론 및 시사점

DriveDreamer는 실제 주행 데이터에서 구축된 최초의 자율주행 World Model로, 세 가지 핵심 능력을 통합한다: (1) 교통 구조 제약에 정렬된 고품질 주행 영상 생성, (2) 텍스트·액션 기반 다양한 시나리오 제어, (3) 합리적인 미래 주행 정책 예측.

**실무적 시사점**

- **합성 데이터 증강**: Auto-DM이 생성한 데이터로 3D 검출 성능을 3.0 mAP까지 높일 수 있어, 데이터 부족 문제를 해결하는 현실적인 수단이 된다.
- **자율주행 합성 데이터 생성 관점**: HDMap·3D Box를 구조 조건으로 활용하는 접근법은 실제 도로 레이아웃을 반영한 시나리오 생성에 직접 응용 가능하다.
- **엔드-투-엔드 학습**: World Model 기반 플래닝은 Open-loop 평가에서 L2 0.29m를 달성하며, 향후 클로즈드 루프 자율주행으로 확장 가능성을 열어준다.
- **한계**: 현재는 단일 카메라(전방) 위주이며, 클로즈드 루프 평가는 미포함. 생성 품질이 높아도 실제 배포에는 안전성 검증이 추가로 필요하다.


---

*관련 논문: [LDM](/posts/papers/high-resolution-image-synthesis-with-latent-diffusion-models/), [DDPM](/posts/papers/denoising-diffusion-probabilistic-models/), [GAIA-1](/posts/papers/GAIA-1/), [MagicDrive](/posts/papers/magicdrive-street-view-generation-3d-geometry-control/), [DriveArena](/posts/papers/DriveArena/), [nuScenes](/posts/papers/nuscenes-multimodal-dataset-autonomous-driving/)*
