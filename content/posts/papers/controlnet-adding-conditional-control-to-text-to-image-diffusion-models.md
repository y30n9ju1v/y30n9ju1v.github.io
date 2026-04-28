---
title: "ControlNet: Adding Conditional Control to Text-to-Image Diffusion Models"
date: 2026-04-29T02:00:00+09:00
draft: false
categories: ["Papers"]
tags: ["ControlNet", "Diffusion Model", "Stable Diffusion", "Conditional Generation", "Image Synthesis", "Zero Convolution", "LDM", "CLIP"]
---

## 개요

- **저자**: Lvmin Zhang, Anyi Rao, Maneesh Agrawala (Stanford University)
- **발행년도**: 2023
- **arXiv**: 2302.05543
- **주요 내용**: 대규모 사전 훈련된 text-to-image diffusion 모델(Stable Diffusion)에 엣지맵·깊이맵·포즈·세그멘테이션 등 공간적 조건 이미지를 추가 학습 없이 주입하는 신경망 아키텍처. "Zero Convolution"으로 사전 훈련 모델을 보호하면서 소량의 데이터(1k~1M)로도 안정적 학습 가능.

## 한계 극복

- **기존 한계 1 — 텍스트만으로는 공간 제어 불가**: text-to-image 모델은 "사람이 왼팔을 들고 있는 모습"을 정확한 포즈로 생성하기 매우 어려움. 프롬프트 엔지니어링으로는 복잡한 레이아웃·형태·포즈 제어에 한계.
- **기존 한계 2 — 대형 모델 파인튜닝의 위험**: 수십억 개 파라미터 모델을 직접 파인튜닝하면 과적합·catastrophic forgetting 발생. 태스크별 데이터는 ImageNet 대비 5만 배 적음(~100K).
- **기존 한계 3 — 기존 조건부 방법의 성능 부족**: PITI·Sketch-Guided Diffusion 등은 결과 품질과 조건 충실도 모두 ControlNet에 비해 낮음.
- **이 논문의 접근 방식**: 원본 모델을 완전히 잠그고(freeze), 인코더 레이어만 복사한 **trainable copy**를 만들어 **zero convolution**으로 연결. 초기에는 노이즈를 추가하지 않고, 학습이 진행되면서 점진적으로 조건을 반영하는 구조.

## 목차

- Section 1: Introduction
- Section 2: Related Work
- Section 3: Method
  - 3.1 ControlNet 기본 구조
  - 3.2 Stable Diffusion에 ControlNet 적용
  - 3.3 Training
  - 3.4 Inference
- Section 4: Experiments
- Section 5: Conclusion

---

## Section 1: Introduction

**요약**

Stable Diffusion 같은 text-to-image 모델은 텍스트 프롬프트만으로 놀라운 이미지를 생성하지만, 정밀한 공간 레이아웃 제어에는 한계가 있다. 사용자가 원하는 포즈, 엣지 구조, 깊이 구조를 이미지 형태로 직접 제공할 수 있다면 훨씬 정확한 생성이 가능하다.

ControlNet은 다음 세 가지를 동시에 달성한다:
1. 사전 훈련 모델의 품질과 능력을 완전히 보존
2. 소규모 데이터셋(1k~50k)으로도 안정적 학습
3. 엣지, 깊이, 포즈, 세그멘테이션 등 다양한 조건을 단일 프레임워크로 지원

**핵심 개념**

- **Spatial Conditioning**: 픽셀 단위의 공간 정보를 텍스트 임베딩과 함께 diffusion 모델에 주입
- **Large Pretrained Backbone**: Stable Diffusion의 인코더 레이어를 강력한 특징 추출기로 재활용
- **Zero Convolution**: 훈련 초기에 기울기를 0으로 만들어 사전 훈련 모델이 손상되지 않도록 보호

---

## Section 2: Related Work

**요약**

관련 연구들이 해결하려 했던 문제와 한계:

**핵심 개념**

- **HyperNetwork**: 작은 네트워크가 큰 네트워크의 가중치를 생성. 스타일 변환에 활용되지만 공간 조건 제어에는 부적합.
- **Adapter 방법**: 사전 훈련 모델에 경량 모듈 삽입. NLP에서 유래, CLIP과 결합해 vision backbone 전이 학습에 활용.
- **LoRA**: Low-Rank 행렬로 파라미터 오프셋을 학습, catastrophic forgetting 방지. 파라미터 효율적이지만 공간 구조 조건화에는 한계.
- **Zero-Initialized Layers**: 초기 가중치를 0으로 설정해 초기 훈련 시 노이즈 영향을 제거. ControlNet의 핵심 아이디어의 기반.
- **Image Diffusion Models**: DDPM → LDM(Stable Diffusion)으로 발전. LDM은 잠재 공간에서 diffusion을 수행해 계산 비용 절감. 텍스트 인코딩은 CLIP text encoder 사용.

---

## Section 3: Method

### 3.1 ControlNet 기본 구조

**요약**

임의의 신경망 블록 $\mathcal{F}(\cdot;\Theta)$가 입력 특징맵 $\boldsymbol{x}$를 출력 특징맵 $\boldsymbol{y}$로 변환한다고 하자:

$$\boldsymbol{y} = \mathcal{F}(\boldsymbol{x};\Theta) \tag{1}$$

ControlNet을 추가하려면:
1. 원본 블록의 파라미터 $\Theta$를 **완전히 잠금(freeze)**
2. 동일한 구조의 **trainable copy** $\Theta_c$를 생성
3. 조건 벡터 $\boldsymbol{c}$를 trainable copy의 입력으로 주입
4. **Zero convolution** $\mathcal{Z}(\cdot;\cdot)$으로 두 모델을 연결

최종 출력:

$$\boldsymbol{y}_c = \mathcal{F}(\boldsymbol{x};\Theta) + \mathcal{Z}(\mathcal{F}(\boldsymbol{x} + \mathcal{Z}(\boldsymbol{c};\Theta_{z1});\Theta_c);\Theta_{z2}) \tag{2}$$

**수식 설명**

- **$\mathcal{F}(\boldsymbol{x};\Theta)$**: 잠긴 원본 블록의 출력. 원본 Stable Diffusion의 능력을 그대로 보존.
- **$\mathcal{Z}(\boldsymbol{c};\Theta_{z1})$**: 첫 번째 zero convolution. 조건 이미지 $\boldsymbol{c}$를 trainable copy 입력 공간으로 변환. 초기값 = 0.
- **$\mathcal{F}(\boldsymbol{x} + \mathcal{Z}(\boldsymbol{c};\Theta_{z1});\Theta_c)$**: 원본 입력 + 조건을 받은 trainable copy의 출력.
- **$\mathcal{Z}(\cdots;\Theta_{z2})$**: 두 번째 zero convolution. trainable copy 출력을 원본 모델 출력 공간으로 변환. 초기값 = 0.
- **$\boldsymbol{y}_c$**: 원본 출력 + ControlNet 출력의 합산. 초기 훈련 단계에서 $\mathcal{Z}$가 0이므로 $\boldsymbol{y}_c = \boldsymbol{y}$ — 원본 모델과 동일하게 시작.

**Zero Convolution의 의미**

Zero convolution은 가중치와 편향 모두 0으로 초기화된 $1 \times 1$ 컨볼루션이다. 훈련 첫 단계에서:

$$\mathcal{Z}(\boldsymbol{c};\Theta_{z1}) = 0, \quad \mathcal{Z}(\cdots;\Theta_{z2}) = 0$$

따라서 $\boldsymbol{y}_c = \boldsymbol{y}$. trainable copy가 아직 학습되지 않은 상태에서 임의의 노이즈가 원본 모델의 깊은 특징에 영향을 주지 않는다. 학습이 진행되면서 zero convolution의 파라미터가 점진적으로 성장하여 조건 정보를 반영하기 시작한다.

---

### 3.2 Stable Diffusion에 ControlNet 적용

**요약**

Stable Diffusion은 U-Net 구조로 인코더 12블록 + 미들 1블록 + 디코더 12블록, 총 25블록. ControlNet은 인코더 12블록 + 미들 1블록에만 trainable copy를 만든다.

**조건 이미지 인코딩**

조건 이미지(엣지, 깊이, 포즈 등)는 Stable Diffusion의 잠재 공간($64 \times 64$)과 해상도가 다르다($512 \times 512$). 이를 맞추기 위해 소형 인코딩 네트워크 $\mathcal{E}(\cdot)$를 사용:

$$\boldsymbol{c}_f = \mathcal{E}(\boldsymbol{c}_i) \tag{4}$$

**수식 설명**

- **$\boldsymbol{c}_i$**: 원본 조건 이미지 ($512 \times 512$)
- **$\mathcal{E}(\cdot)$**: 4개 컨볼루션 레이어(커널 $4 \times 4$, 스트라이드 $2 \times 2$, 채널 16→32→64→128), ReLU 활성화
- **$\boldsymbol{c}_f$**: 잠재 공간 조건 벡터 — Stable Diffusion의 특징 공간과 동일한 해상도로 변환

ControlNet 출력은 U-Net 디코더의 각 skip connection에 더해진다:

```
SD Encoder Block A (64×64) ─── ControlNet Encoder A (64×64) ──► zero conv ──► SD Decoder C
SD Encoder Block B (32×32) ─── ControlNet Encoder B (32×32) ──► zero conv ──► SD Decoder B
SD Encoder Block C (16×16) ─── ControlNet Encoder C (16×16) ──► zero conv ──► SD Decoder A
SD Encoder Block D ( 8× 8) ─── ControlNet Encoder D ( 8× 8) ──► zero conv ──►
SD Middle Block   ( 8× 8) ─── ControlNet Middle   ( 8× 8) ──► zero conv ──►
```

**핵심 개념**

- **잠긴 파라미터(locked)**: 원본 Stable Diffusion 인코더. 역전파 시 기울기 계산 없음 → GPU 메모리 절약 + 원본 품질 보존.
- **훈련 가능 복사본(trainable copy)**: 인코더 구조만 복제. 디코더는 복제하지 않음 — 계산 효율성.
- **Skip Connection 주입**: ControlNet 출력을 SD 디코더의 skip connection에 직접 더함. U-Net의 멀티스케일 특징 모두에 조건 반영.
- **계산 비용**: ControlNet 없는 SD 대비 GPU 메모리 약 23% 증가, 훈련 시간 약 34% 증가 (A100 40GB 기준).

---

### 3.3 Training

**요약**

훈련 목표는 Stable Diffusion과 동일한 diffusion loss:

$$\mathcal{L} = \mathbb{E}_{z_0, t, \boldsymbol{c}_t, \boldsymbol{c}_f, \epsilon \sim \mathcal{N}(0,1)} \left[ \| \epsilon - \epsilon_\theta(z_t, t, \boldsymbol{c}_t, \boldsymbol{c}_f) \|_2^2 \right] \tag{5}$$

**수식 설명**

- **$z_0$**: 원본 이미지의 잠재 벡터
- **$t$**: diffusion 타임스텝 (노이즈가 얼마나 추가됐는지)
- **$\boldsymbol{c}_t$**: 텍스트 프롬프트 조건 (CLIP text encoder 출력)
- **$\boldsymbol{c}_f$**: 공간 조건 이미지의 잠재 벡터 (엣지맵, 포즈 등)
- **$\epsilon$**: 실제로 추가된 가우시안 노이즈
- **$\epsilon_\theta$**: 노이즈 예측 네트워크 (Stable Diffusion U-Net + ControlNet)
- **$\|\cdot\|_2^2$**: 예측 노이즈와 실제 노이즈의 L2 거리를 최소화 — 올바른 denoising 방향 학습

**훈련 핵심 기법**

- **50% 텍스트 드롭아웃**: 훈련 시 텍스트 프롬프트를 50% 확률로 빈 문자열로 교체. 모델이 조건 이미지만으로 의미를 인식하는 능력을 키움 (텍스트 없이도 동작 가능).
- **Sudden Convergence Phenomenon**: zero convolution 덕분에 초기에는 조건을 무시하다가 특정 시점(~6000 스텝)에 갑자기 조건을 따르기 시작하는 현상. 학습이 붕괴되지 않고 안정적으로 수렴함을 보여줌.
- **소량 데이터 강인성**: 1k 이미지로도 "사자"를 인식하는 모델 학습 가능. 50k로 실용적 품질, 3M으로 최고 품질 달성.

---

### 3.4 Inference

**요약**

추론 시 여러 기법으로 결과를 추가 제어할 수 있다.

**Classifier-Free Guidance (CFG) Resolution Weighting**

CFG는 조건부/무조건부 출력을 혼합하여 품질을 높이는 기법:

$$\epsilon_\text{prd} = \epsilon_\text{uc} + \beta_\text{cfg}(\epsilon_\text{c} - \epsilon_\text{uc})$$

- **$\epsilon_\text{uc}$**: 무조건부(unconditional) 출력
- **$\epsilon_\text{c}$**: 조건부(conditional) 출력
- **$\beta_\text{cfg}$**: 사용자 지정 guidance 강도

ControlNet 조건을 $\epsilon_\text{uc}$와 $\epsilon_\text{c}$ 둘 다에 더하면 텍스트 없이도 조건만으로 제어 가능. 조건을 $\epsilon_\text{c}$에만 더하면 guidance가 매우 강해짐.

**CFG Resolution Weighting**: 해상도별로 ControlNet 연결 강도를 차등 적용:

$$w_i = 64 / h_i$$

- 낮은 해상도 블록($8 \times 8$)에 높은 가중치 → 전체 구조·의미 제어
- 높은 해상도 블록($64 \times 64$)에 낮은 가중치 → 세밀한 텍스처는 텍스트 프롬프트에 맡김

**다중 ControlNet 합성**: 여러 조건(포즈 + 깊이)을 동시에 적용할 때, 각 ControlNet 출력을 단순 합산. 별도 가중치나 보간 불필요.

---

## Section 4: Experiments

**요약**

Stable Diffusion에 8가지 조건 유형으로 ControlNet을 훈련:

| 조건 유형 | 설명 |
|---------|------|
| Canny Edge | 이미지의 엣지맵 |
| Depth Map | 깊이 정보 |
| Normal Map | 표면 법선 벡터 |
| HED Soft Edge | 부드러운 경계 검출 |
| Human Pose (OpenPose) | 인체 관절 키포인트 |
| Segmentation (ADE20K) | 의미론적 분할 맵 |
| M-LSD Lines | 직선 세그먼트 |
| User Scribbles | 사용자 낙서 |

**정량적 결과**

| 방법 | 결과 품질 (AUR↑) | 조건 충실도 (AUR↑) |
|------|----------------|------------------|
| PITI | 1.10 | 1.02 |
| Sketch-Guided (β=1.6) | 3.21 | 2.31 |
| ControlNet-lite | 3.93 | 4.09 |
| **ControlNet** | **4.22** | **4.28** |

(AUR: 1~5, 높을수록 좋음)

**세그멘테이션 조건 FID 점수**:

| 방법 | FID↓ | CLIP-score↑ |
|------|------|------------|
| LDM(seg.) | 25.35 | 0.18 |
| PITI(seg.) | 19.74 | 0.20 |
| ControlNet-lite | 17.92 | 0.26 |
| **ControlNet** | **15.27** | **0.26** |

**주요 발견**:

- **Zero convolution의 중요성**: 일반 초기화로 교체하면 ControlNet-lite 수준으로 성능 하락 (trainable copy backbone이 파괴됨).
- **프롬프트 강인성**: 프롬프트가 없어도, 불충분해도, 충돌해도 조건 이미지를 따름. 완벽한 프롬프트 제공 시 최고 품질.
- **커뮤니티 모델 전이**: ControlNet은 SD의 네트워크 토폴로지를 변경하지 않으므로, Comic Diffusion·Protogen 3.4 등 파생 모델에 재훈련 없이 직접 적용 가능.
- **산업 모델과의 비교**: 단일 RTX 3090Ti + 200k 샘플로 훈련한 ControlNet이 A100 클러스터 + 12M 이미지로 훈련한 SD V2 Depth-to-Image와 거의 구분 불가능한 결과.

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **Zero Convolution** | 가중치·편향 모두 0으로 초기화된 $1\times1$ conv. 훈련 초기 노이즈를 차단하여 원본 모델 보호 |
| **Trainable Copy** | 원본 인코더와 동일 구조의 복사본. 조건 정보를 처리하는 학습 가능한 브랜치 |
| **Locked Parameters** | 원본 Stable Diffusion 파라미터. 완전 동결으로 catastrophic forgetting 방지 |
| **Sudden Convergence** | Zero convolution 덕분에 특정 스텝에서 갑자기 조건을 따르기 시작하는 안정적 수렴 현상 |
| **CFG Resolution Weighting** | 해상도별 ControlNet 출력 강도 차등 적용. 저해상도=구조, 고해상도=텍스처 |
| **50% Text Dropout** | 훈련 시 텍스트 프롬프트 무작위 제거로 조건 이미지만으로도 동작하는 능력 학습 |
| **Multiple ControlNets** | 여러 조건을 동시에 적용 시 출력을 단순 합산. 추가 가중치·보간 불필요 |

## 결론 및 시사점

ControlNet은 "사전 훈련 모델을 건드리지 않고 새로운 제어 능력을 주입"하는 범용 패러다임을 제시했다. Zero convolution이라는 단순한 아이디어가 catastrophic forgetting 문제를 완전히 해결하면서, 소규모 데이터로도 산업 수준의 공간 제어를 가능케 한다.

**자율주행·합성 데이터 관점 시사점**

- **MagicDrive·DriveArena와의 직접 연결**: 두 논문 모두 ControlNet 구조를 핵심으로 사용. BEV 맵·3D 바운딩 박스·카메라 포즈를 "조건 이미지"로 취급하여 Stable Diffusion에 주입. ControlNet 없이는 이 구조를 이해하기 어려움.
- **합성 데이터 정확도**: 3D 바운딩 박스나 HD 맵을 조건으로 주면 생성된 이미지의 객체 위치·크기가 실제 센서 데이터와 일치 → 어노테이션 자동 생성 가능.
- **다중 조건 합성**: 포즈 + 깊이 + 세그멘테이션을 동시에 적용하는 것처럼, 자율주행 장면에서 LiDAR 깊이 + 카메라 포즈 + 객체 레이아웃을 동시에 조건으로 줄 수 있음.
- **적은 데이터로 도메인 적응**: 새로운 지역·날씨·시간대의 데이터가 수천 장만 있어도 ControlNet으로 파인튜닝 가능 → 롱테일 시나리오 합성에 활용.
- **커뮤니티 모델 전이 가능성**: 한 번 훈련한 ControlNet을 다른 Stable Diffusion 파생 모델에 재훈련 없이 사용 가능 → 다양한 시각 스타일의 합성 데이터 생성에 유연하게 적용.
