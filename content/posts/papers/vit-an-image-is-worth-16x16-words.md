---
title: "ViT: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
date: 2026-04-24T12:00:00+09:00
draft: false
categories: ["Papers", "Computer Vision", "Deep Learning"]
tags: ["Vision Transformer", "ViT", "Image Classification", "Self-Attention", "ICLR 2021"]
---

## 개요

- **저자**: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby (Google Brain)
- **발행년도**: 2021 (ICLR 2021)
- **주요 내용**: CNN 없이 순수 Transformer만으로 이미지를 16×16 패치 시퀀스로 처리하여 대규모 사전학습 후 이미지 분류 SOTA 달성

## 한계 극복

- **기존 한계 1 — CNN의 귀납적 편향 의존**: 기존 비전 모델은 지역성(locality), 이동 등변성(translation equivariance) 같은 CNN 고유의 귀납적 편향에 의존. 이는 데이터가 적을 때 유리하지만 대규모 데이터에서는 오히려 표현력을 제한.
- **기존 한계 2 — Transformer의 비전 적용 시 CNN 병용**: 기존 Self-Attention 기반 비전 모델들은 CNN과 결합하거나 특수한 Attention 패턴을 사용해 하드웨어 가속기에서 효율적으로 동작하지 못했음.
- **이 논문의 접근 방식**: 이미지를 고정 크기 패치로 분할하고 각 패치를 NLP의 토큰처럼 취급. CNN 구조 없이 표준 Transformer Encoder를 그대로 적용. 대규모 데이터(JFT-300M)로 사전학습 시 귀납적 편향 없이도 CNN을 능가.

## 목차

- Section 1: Introduction
- Section 2: Related Work
- Section 3: Method — Vision Transformer (ViT)
- Section 4: Experiments
- Section 5: Conclusion
- Appendix A: Multihead Self-Attention
- Appendix B: Experiment Details
- Appendix D: Additional Analyses

---

## Section 1-2: Introduction & Related Work

**요약**

NLP에서 Transformer는 사실상 표준 아키텍처가 되었지만, 비전에서는 CNN이 여전히 지배적이었습니다. Self-Attention을 이미지에 직접 적용하면 픽셀 수의 제곱에 비례하는 계산 비용이 발생하기 때문입니다. 기존 연구들은 이를 해결하기 위해 로컬 영역에만 Attention을 적용하거나 CNN 피처 위에 Attention을 추가하는 방식을 택했습니다.

이 논문은 다른 방향을 택합니다. 이미지를 **16×16 패치로 분할**하고 각 패치를 플래튼(flatten)하여 NLP 토큰처럼 처리합니다. 추가적인 이미지 특화 구조 없이 표준 Transformer를 그대로 사용하며, 대규모 데이터셋으로 사전학습 후 소규모 벤치마크에 파인튜닝합니다.

**핵심 개념**

- **귀납적 편향 (Inductive Bias)**: 모델이 학습 전부터 가지고 있는 가정. CNN은 "가까운 픽셀끼리 관련이 높다(locality)"와 "같은 패턴은 위치가 달라도 같다(translation equivariance)"를 내장. ViT는 이를 제거하여 데이터로부터 직접 학습.
- **스케일링 법칙 (Scaling Law)**: 데이터와 모델 크기가 커질수록 성능이 지속적으로 향상되는 현상. ViT는 NLP의 스케일링 법칙이 비전에도 적용됨을 보임.

---

## Section 3: Method — Vision Transformer (ViT)

**요약**

ViT의 핵심 아이디어는 2D 이미지를 1D 패치 시퀀스로 변환하는 것입니다. 이후 처리는 표준 Transformer Encoder와 동일합니다.

### 3.1 패치 임베딩

$$\mathbf{z}_0 = [\mathbf{x}_{class};\, \mathbf{x}_p^1\mathbf{E};\, \mathbf{x}_p^2\mathbf{E};\, \cdots;\, \mathbf{x}_p^N\mathbf{E}] + \mathbf{E}_{pos}$$

**수식 설명**

입력 이미지 $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$를 $N$개의 패치로 분할하는 과정:
- **$\mathbf{x}_{class}$**: 분류를 위한 학습 가능한 토큰. BERT의 `[CLS]` 토큰과 동일한 역할. 이 토큰의 최종 출력이 이미지 전체 표현으로 사용됨.
- **$\mathbf{x}_p^i$**: $i$번째 패치를 플래튼한 벡터. 크기는 $P^2 \cdot C$ ($P$: 패치 크기, $C$: 채널 수).
- **$\mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}$**: 패치를 Transformer 차원 $D$로 매핑하는 선형 투영 행렬 (학습 파라미터).
- **$\mathbf{E}_{pos} \in \mathbb{R}^{(N+1) \times D}$**: 위치 정보를 담은 학습 가능한 1D 위치 임베딩. 패치의 순서 정보를 Transformer에 전달.
- **$N = HW/P^2$**: 총 패치 수. 예: 224×224 이미지를 16×16 패치로 나누면 $N = 196$.

### 3.2 Transformer Encoder 순전파

$$\mathbf{z}'_\ell = \text{MSA}(\text{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1}, \quad \ell = 1 \ldots L$$

$$\mathbf{z}_\ell = \text{MLP}(\text{LN}(\mathbf{z}'_\ell)) + \mathbf{z}'_\ell, \quad \ell = 1 \ldots L$$

$$\mathbf{y} = \text{LN}(\mathbf{z}_L^0)$$

**수식 설명**

- **$\text{MSA}$**: Multi-Head Self-Attention. 각 패치가 다른 모든 패치와 상호작용하며 전역 문맥을 수집.
- **$\text{LN}$**: Layer Normalization. 각 레이어 입력을 정규화하여 학습 안정화. Pre-norm 방식(Attention 전에 적용)을 사용.
- **$+ \mathbf{z}_{\ell-1}$**: Residual connection. 정보 손실 없이 깊은 네트워크 학습 가능.
- **$\text{MLP}$**: 두 개의 선형 레이어 + GELU 활성화. 각 패치 표현을 독립적으로 변환.
- **$\mathbf{z}_L^0$**: $L$번째 레이어의 `[class]` 토큰 출력. 이것이 최종 이미지 표현 $\mathbf{y}$가 됨.

### 3.3 Multihead Self-Attention (Appendix A)

$$[\mathbf{q}, \mathbf{k}, \mathbf{v}] = \mathbf{z}\mathbf{U}_{qkv}, \quad \mathbf{U}_{qkv} \in \mathbb{R}^{D \times 3D_h}$$

$$A = \text{softmax}\!\left(\mathbf{q}\mathbf{k}^\top / \sqrt{D_h}\right), \quad A \in \mathbb{R}^{N \times N}$$

$$\text{SA}(\mathbf{z}) = A\mathbf{v}$$

**수식 설명**

- **$\mathbf{q}, \mathbf{k}, \mathbf{v}$**: Query, Key, Value. 입력 시퀀스 $\mathbf{z}$에서 선형 투영으로 생성.
  - Query: "나는 어떤 정보를 찾고 있는가?"
  - Key: "나는 어떤 정보를 가지고 있는가?"
  - Value: "실제로 전달할 정보"
- **$A_{ij}$**: $i$번째 패치가 $j$번째 패치에 얼마나 주목하는지의 가중치. 모든 패치 쌍에 대해 계산되므로 전역 문맥 포착 가능.
- **$\sqrt{D_h}$**: 내적값의 스케일을 조정하는 스케일링 팩터. $D_h$가 클수록 내적값이 커져 softmax가 극단적으로 작동하는 것을 방지.
- **MSA**: $k$개의 SA를 병렬 실행 후 연결. 각 head가 서로 다른 유형의 관계를 학습.

**핵심 개념**

- **패치 크기 $P$의 영향**: $P$가 작을수록 패치 수 $N$이 늘어나 계산 비용이 $O(N^2)$으로 증가. ViT-B는 $P=16$을, ViT-H는 $P=14$를 사용.
- **Hybrid Architecture**: 패치 대신 CNN(ResNet)의 중간 피처맵을 입력으로 사용하는 변형. 소규모 데이터에서 순수 ViT보다 약간 유리하지만 대규모에서 차이 소멸.
- **파인튜닝 시 해상도 조정**: 사전학습보다 높은 해상도로 파인튜닝 시, 패치 크기를 유지하면 시퀀스 길이가 늘어남. 기존 위치 임베딩을 2D 보간(interpolation)하여 적응.

---

## Section 4: Experiments

**요약**

ViT를 세 가지 규모(Base, Large, Huge)로 구성하고, 세 가지 데이터셋(ImageNet, ImageNet-21k, JFT-300M)으로 사전학습 후 다양한 벤치마크에서 평가합니다.

### 4.1 모델 변형

| 모델 | 레이어 수 | 히든 크기 $D$ | MLP 크기 | 헤드 수 | 파라미터 수 |
|------|---------|------------|--------|--------|-----------|
| ViT-Base | 12 | 768 | 3072 | 12 | 86M |
| ViT-Large | 24 | 1024 | 4096 | 16 | 307M |
| ViT-Huge | 32 | 1280 | 5120 | 16 | 632M |

표기법: ViT-L/16 = Large 모델, 패치 크기 16×16.

### 4.2 State-of-the-Art 비교 결과

JFT-300M으로 사전학습한 ViT-H/14는 기존 SOTA(BiT-L ResNet152x4, Noisy Student EfficientNet-L2)를 모든 벤치마크에서 능가하면서 **사전학습 비용은 약 4배 절감**:

| 벤치마크 | ViT-H/14 (JFT) | BiT-L | Noisy Student |
|---------|---------------|-------|--------------|
| ImageNet | **88.55%** | 87.54% | 88.5% |
| ImageNet ReaL | **90.72%** | 90.54% | 90.55% |
| CIFAR-100 | **94.55%** | 93.51% | — |
| VTAB (19 tasks) | **77.63%** | 76.29% | — |

TPUv3-core-days: ViT-H/14 = 2.5k, BiT-L = 9.9k, Noisy Student = 12.3k.

### 4.3 사전학습 데이터 요구량

- **소규모 (ImageNet 1.3M)**: ViT가 ResNet(BiT) 대비 몇 퍼센트 포인트 낮음. CNN의 귀납적 편향이 소량 데이터에서 유리하게 작용.
- **중규모 (ImageNet-21k 14M)**: 차이가 좁혀짐.
- **대규모 (JFT-300M 303M)**: ViT가 ResNet을 능가. **대규모 데이터가 귀납적 편향을 대체**.

### 4.4 스케일링 연구

같은 사전학습 연산량 기준으로 ViT는 ResNet 대비 약 **2~4배 적은 연산으로 동일 성능** 달성. Hybrid(CNN+ViT)는 소규모 연산 예산에서 약간 유리하지만 큰 모델에서 차이 소멸.

### 4.5 ViT 내부 분석

- **첫 번째 레이어 필터**: 저차원 주성분 분석 결과가 CNN의 Gabor 필터와 유사한 구조를 자발적으로 학습.
- **위치 임베딩**: 가까운 패치일수록 높은 유사도를 가지며, 행-열 구조가 자연스럽게 학습됨.
- **Attention Distance**: 낮은 레이어에서도 일부 헤드는 이미지 전체에 걸친 전역 Attention을 수행. 깊어질수록 평균 Attention 거리 증가.

### 4.6 자기지도 학습 예비 실험

Masked Patch Prediction(BERT의 MLM과 유사) 방식으로 자기지도 학습 시, ViT-B/16이 ImageNet에서 79.9% 달성. 지도학습 대비 4% 차이로, 자기지도 ViT의 가능성을 시사 (→ MAE, DINO 등으로 이어짐).

---

## 핵심 개념 정리

- **패치 토큰화**: 이미지를 $P \times P$ 패치로 분할 → 플래튼 → 선형 투영. NLP의 단어 임베딩과 동일한 구조.
- **[CLS] 토큰**: 시퀀스 앞에 붙이는 학습 가능한 임베딩. 모든 패치와 Attention 후 최종 표현으로 분류에 사용.
- **1D 위치 임베딩**: 2D 구조를 명시적으로 인코딩하지 않아도 위치 임베딩이 2D 공간 구조를 자발적으로 학습함을 실험으로 확인.
- **대규모 사전학습의 중요성**: ViT의 핵심 전제. 충분한 데이터 없이는 CNN 귀납적 편향을 이길 수 없음. JFT-300M이 결정적 역할.
- **전이 학습 효율**: 사전학습 후 작은 데이터셋(CIFAR, Pets 등)에 파인튜닝 시 CNN 대비 훨씬 적은 연산으로 동등 이상 성능.

---

## 결론 및 시사점

ViT는 "이미지도 패치 시퀀스로 처리할 수 있다"는 단순한 아이디어가 대규모 데이터와 결합될 때 CNN을 능가할 수 있음을 증명했습니다.

**자율주행·AV 인식 계보에서의 의미**:
- BEVFormer, SurroundOcc, TPVFormer 등 대부분의 BEV 인식 논문이 ViT를 backbone으로 채택.
- 멀티카메라 이미지를 패치 단위로 처리하는 방식이 BEV Attention과 자연스럽게 결합.
- 카메라 피처 추출기의 표준이 ResNet에서 ViT 계열(ViT-B/16, InternImage 등)로 전환되는 계기.

**한계 및 후속 연구 방향**:
- 소규모 데이터에서 CNN 대비 열세 → DeiT (데이터 효율 개선), MAE (자기지도 사전학습)로 해결.
- 고해상도 이미지에서 $O(N^2)$ 계산 비용 → Swin Transformer (윈도우 Attention)로 해결.
- 탐지·분할 등 밀집 예측 태스크 미지원 → ViTDet, DINOv2 등으로 확장.


---

*관련 논문: [Attention Is All You Need](/posts/papers/attention-is-all-you-need/), [ResNet](/posts/papers/resnet-deep-residual-learning-for-image-recognition/), [DETR](/posts/papers/detr-end-to-end-object-detection-with-transformers/), [BEVFormer](/posts/papers/BEVFormer/), [EmerNeRF](/posts/papers/EmerNeRF/)*
