---
title: "ResNet: Deep Residual Learning for Image Recognition"
date: 2026-04-24T10:00:00+09:00
draft: false
categories: ["Papers", "Computer Vision", "Deep Learning"]
tags: ["ResNet", "Residual Learning", "Skip Connection", "Image Classification", "CNN", "Microsoft Research", "ILSVRC 2015"]
---

## 개요

- **저자**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (Microsoft Research)
- **발행년도**: 2015 (arXiv: 1512.03385)
- **주요 내용**: 잔차 학습(Residual Learning) 프레임워크를 통해 수백 레이어 이상의 극도로 깊은 신경망을 효과적으로 학습시키는 방법을 제안. ILSVRC 2015에서 1위 달성.

## 한계 극복

이 논문이 기존 연구의 어떤 한계를 극복하기 위해 작성되었는지 설명합니다.

- **기존 한계 1 — 깊이 증가 시 성능 저하 (Degradation Problem)**: 단순히 레이어를 더 쌓으면 학습 오류가 오히려 증가하는 현상이 발생. 과적합이 아닌 최적화 어려움이 원인.
- **기존 한계 2 — Vanishing/Exploding Gradient**: 깊은 네트워크는 역전파 시 그래디언트가 소실되거나 폭발하여 학습이 불안정해짐.
- **기존 한계 3 — 깊이와 성능의 비례 불가**: VGG처럼 레이어를 단순 누적하는 방식은 16~19층 이상으로 깊어지면 오히려 성능이 떨어짐.
- **이 논문의 접근 방식**: 각 레이어가 원하는 함수를 직접 학습하는 대신, **입력 대비 잔차(residual)**를 학습하도록 재정식화. 입력을 출력에 직접 더하는 **Shortcut Connection**을 통해 그래디언트가 깊은 네트워크를 자유롭게 흐를 수 있게 함.

## 목차

- Section 1: Introduction — 깊은 네트워크의 degradation 문제와 해결 동기
- Section 2: Related Work — 잔차 표현 및 Shortcut Connection 관련 선행 연구
- Section 3: Deep Residual Learning — 잔차 학습 이론 및 네트워크 아키텍처
- Section 4: Experiments — ImageNet, CIFAR-10, PASCAL VOC, MS COCO 실험 결과
- Appendix: Object Detection Baselines & Improvements, ImageNet Localization

---

## Section 1: Introduction

**요약**

딥러닝에서 네트워크 깊이는 성능의 핵심 요소입니다. VGG, GoogLeNet 등은 깊이가 깊을수록 좋은 성능을 보였습니다. 하지만 단순히 레이어를 더 많이 쌓으면 학습 오류가 오히려 증가하는 **degradation 문제**가 발생합니다. 이는 과적합 때문이 아닙니다 — 학습 오류 자체가 높아집니다.

이상적으로는 더 깊은 모델이 더 얕은 모델보다 나빠서는 안 됩니다. 깊은 모델의 추가된 레이어가 단순히 identity mapping(입력을 그대로 통과)을 학습하면 되기 때문입니다. 그러나 실제 최적화기는 이를 제대로 찾지 못합니다.

이 논문은 **잔차 학습(Residual Learning)**을 통해 이 문제를 해결합니다.

**핵심 개념**

- **Degradation Problem**: 깊은 네트워크가 얕은 네트워크보다 학습 오류가 높아지는 현상. 과적합이 아닌 최적화 실패가 원인.
- **Identity Mapping**: 추가된 레이어가 입력을 그대로 출력하는 동작. 이상적으로는 이게 가능해야 하지만 실제로는 어려움.
- **Residual Learning**: 레이어가 원하는 함수 $\mathcal{H}(\mathbf{x})$ 전체를 배우는 대신, $\mathcal{F}(\mathbf{x}) := \mathcal{H}(\mathbf{x}) - \mathbf{x}$ (잔차)만 학습하도록 재정식화.

---

## Section 2: Related Work

**요약**

잔차 표현은 이미지 처리와 컴퓨터 비전에서 오랫동안 사용되어 왔습니다. VLAD, Fisher Vector 같은 인코딩 방식이 잔차 벡터를 활용합니다. Shortcut Connection 아이디어도 MLP에서 보조 분류기를 연결하거나, 중간 레이어를 직접 연결하는 형태로 존재했습니다.

이 논문과 가장 관련 깊은 선행 연구는 **Highway Networks**입니다. Highway Networks도 shortcut을 가지지만, 게이팅 함수가 데이터에 의존적이고 파라미터를 가집니다. 반면 이 논문의 identity shortcut은 **파라미터가 전혀 없고**, 정보가 항상 완전히 통과됩니다.

**핵심 개념**

- **Highway Networks**: 게이팅 shortcut을 가진 네트워크. 게이트가 닫히면 non-residual 함수가 됨. 100층 이상에서 성능 향상이 관찰되지 않음.
- **Identity Shortcut의 차별점**: 파라미터 없음, 게이트 없음, 모든 정보 항상 전달. 잔차 함수가 항상 학습됨.

---

## Section 3: Deep Residual Learning

**요약**

### 3.1 잔차 학습 (Residual Learning)

기존 방식은 레이어가 목표 함수 $\mathcal{H}(\mathbf{x})$를 직접 학습합니다. 잔차 학습은 동일한 레이어가 $\mathcal{F}(\mathbf{x}) := \mathcal{H}(\mathbf{x}) - \mathbf{x}$를 학습하도록 바꿉니다. 원래 함수는 $\mathcal{F}(\mathbf{x}) + \mathbf{x}$로 표현됩니다.

만약 identity mapping이 최적이라면, 잔차를 0으로 만드는 것이 목표 함수를 identity로 학습하는 것보다 훨씬 쉽습니다. 실험에서도 학습된 잔차 함수들의 반응이 일반적으로 작게 나타나, identity mapping이 합리적인 기준점임을 시사합니다.

### 3.2 Shortcut Connection에 의한 Identity Mapping

빌딩 블록의 기본 공식:

$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x} \quad (1)$$

**수식 설명**
- **$\mathbf{x}$**: 블록의 입력 벡터
- **$\mathbf{y}$**: 블록의 출력 벡터
- **$\mathcal{F}(\mathbf{x}, \{W_i\})$**: 레이어들이 학습하는 잔차 함수 (예: 두 레이어의 경우 $W_2 \sigma(W_1 \mathbf{x})$, $\sigma$는 ReLU)
- **$+ \mathbf{x}$**: Shortcut Connection. 입력을 출력에 그대로 더함
- 이 덧셈은 element-wise 연산이며 파라미터도, 연산 복잡도도 추가하지 않음

차원이 다를 경우 선형 투영으로 맞춤:

$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + W_s\mathbf{x} \quad (2)$$

**수식 설명**
- **$W_s$**: 차원 맞춤을 위한 투영 행렬 (1×1 컨볼루션으로 구현)
- 차원이 동일하면 identity shortcut ($W_s = I$)을 사용하여 파라미터를 아낌
- $W_s$는 차원이 달라질 때만 사용

### 3.3 네트워크 아키텍처

**Plain Network**: VGG 철학에 기반. 3×3 필터 위주, 출력 크기가 같으면 필터 수 동일, 크기가 절반이 되면 필터 수 두 배.

**Residual Network**: Plain Network의 각 레이어 쌍에 shortcut 추가. 차원이 증가할 때는 (A) zero-padding 또는 (B) 1×1 투영 사용.

**Bottleneck 아키텍처**: 깊은 네트워크(50/101/152층)의 학습 시간을 줄이기 위해 2층 블록 대신 3층 블록 사용.
- 1×1 컨볼루션: 차원 축소
- 3×3 컨볼루션: 실제 특징 추출
- 1×1 컨볼루션: 차원 복원

**핵심 개념**

- **Building Block**: 두 개의 3×3 컨볼루션 레이어 + shortcut. ResNet-18, ResNet-34에 사용.
- **Bottleneck Block**: 1×1, 3×3, 1×1 세 레이어 조합. ResNet-50/101/152에 사용. 시간 복잡도를 유지하면서 더 깊게 쌓을 수 있음.
- **Identity Shortcut의 중요성**: Bottleneck에서 shortcut을 투영으로 바꾸면 시간 복잡도와 모델 크기가 두 배가 됨. 따라서 identity shortcut이 효율적인 모델 설계에 필수적.

---

## Section 4: Experiments

**요약**

### 4.1 ImageNet Classification

**Plain Networks 비교**: 34층 plain net이 18층 plain net보다 높은 검증 오류를 보임 → degradation 문제 확인. Batch Normalization을 사용하므로 gradient vanishing은 원인이 아님.

**Residual Networks 비교**: 34층 ResNet이 18층 ResNet보다 **2.8% 더 낮은** 오류를 기록. Depth가 성능 향상에 기여함을 확인.

**Identity vs Projection Shortcuts**:
- (A) 차원 증가 시 zero-padding: 파라미터 없음
- (B) 차원 증가 시 투영 shortcut, 나머지는 identity
- (C) 모든 shortcut에 투영 사용
- B가 A보다 약간 좋고, C가 B보다 약간 좋지만 차이가 미미. Projection shortcut이 필수적이지 않음을 보여줌. 논문은 B 옵션을 주로 사용.

**Deeper Bottleneck Architectures 결과**:

| 모델 | Top-1 오류 | Top-5 오류 |
|------|-----------|-----------|
| VGG-16 | 28.07% | 9.33% |
| ResNet-34 B | 24.52% | 7.46% |
| ResNet-50 | 22.85% | 6.71% |
| ResNet-101 | 21.75% | 6.05% |
| ResNet-152 | **21.43%** | **5.71%** |

앙상블 결과: **top-5 오류 3.57%** → ILSVRC 2015 분류 1위

### 4.2 CIFAR-10 분석

**Plain vs ResNet**: Plain은 깊어질수록 오류 증가. ResNet은 깊어질수록 성능 향상.

**110층 ResNet**: 6.43% (±0.16) 오류 달성. FitNet, Highway 등 선행 연구 대비 우수.

**1202층 ResNet 탐색**: 학습 오류 <0.1% 달성. 그러나 테스트 오류는 110층보다 나쁨(7.93%). 이 작은 데이터셋에서는 과도하게 큰 모델이 과적합 가능성.

**Layer Response 분석**: ResNet의 레이어 반응(출력의 표준편차)이 plain net보다 전반적으로 작음 → 잔차 함수들이 0에 가까운 값을 학습, identity mapping이 합리적 기준점임을 뒷받침.

### 4.3 Object Detection (PASCAL VOC & MS COCO)

Faster R-CNN의 backbone을 VGG-16에서 ResNet-101로 교체.

**PASCAL VOC 결과**:
- VGG-16: mAP 73.2% (VOC 07 test)
- ResNet-101: mAP **76.4%** (+3.2%p)

**MS COCO 결과**:
- VGG-16: mAP@[.5,.95] = 21.2%
- ResNet-101: mAP@[.5,.95] = **27.2%** (+6.0%p, 28% 상대적 향상)

→ ILSVRC & COCO 2015에서 ImageNet 분류/검출/위치추정, COCO 검출/세그멘테이션 전 부문 1위

**핵심 개념**

- **Top-1 / Top-5 오류율**: ImageNet 평가 지표. Top-5는 모델의 상위 5개 예측 중 정답이 있으면 맞춘 것으로 간주.
- **mAP (mean Average Precision)**: 물체 검출 성능 지표. mAP@[.5,.95]는 IoU 임계값 0.5~0.95 범위에서의 평균.
- **Faster R-CNN**: Region Proposal Network(RPN)와 Fast R-CNN을 결합한 실시간 물체 검출 프레임워크. ResNet을 backbone으로 사용 시 성능 대폭 향상.

---

## 핵심 개념 정리

- **Residual Learning**: 레이어가 목표 함수가 아닌 입력 대비 잔차를 학습. 수식: $\mathcal{F}(\mathbf{x}) = \mathcal{H}(\mathbf{x}) - \mathbf{x}$
- **Shortcut Connection (Skip Connection)**: 레이어를 건너뛰어 입력을 출력에 직접 더하는 연결. 파라미터 없음, 연산 복잡도 없음.
- **Degradation Problem**: 네트워크가 깊어질수록 학습 오류가 오히려 증가하는 현상. 과적합이 아닌 최적화 어려움이 원인.
- **Identity Mapping**: 입력을 변환 없이 그대로 출력. Shortcut이 이를 구현.
- **Bottleneck Design**: 1×1 → 3×3 → 1×1 컨볼루션 조합. 깊은 네트워크에서 연산 효율을 유지하면서 표현력 확보.
- **Batch Normalization**: 각 컨볼루션 후 적용. 안정적인 학습을 가능하게 하며, gradient vanishing 문제 완화에 기여.
- **ResNet 계열**: ResNet-18/34 (Building Block), ResNet-50/101/152 (Bottleneck Block). 깊어질수록 높은 성능.

---

## 결론 및 시사점

ResNet은 **잔차 학습**이라는 단순한 아이디어로 극도로 깊은 신경망 학습 문제를 해결했습니다.

**주요 기여**:
1. Degradation 문제를 명확히 정의하고 잔차 학습으로 해결
2. 파라미터 추가 없이 shortcut connection으로 구현 가능
3. 152층까지 깊은 네트워크에서도 성능이 단조 향상
4. 이미지 분류 외 검출, 위치추정, 세그멘테이션 등 다양한 태스크에 범용 적용

**실무적 시사점**:
- Backbone 교체만으로도 downstream 태스크(검출, 세그멘테이션) 성능이 크게 향상됨
- Identity shortcut은 파라미터·연산 비용 없이 최적화를 돕는 효과적인 설계 원칙
- ResNet은 이후 DenseNet, EfficientNet, Vision Transformer 등 수많은 후속 아키텍처의 기반이 됨
- 자율주행의 특징 추출 backbone, 물체 검출 네트워크(Faster R-CNN + ResNet) 등에 광범위하게 활용


---

*관련 논문: [Attention Is All You Need](/posts/papers/attention-is-all-you-need/), [ViT](/posts/papers/vit-an-image-is-worth-16x16-words/), [DETR](/posts/papers/detr-end-to-end-object-detection-with-transformers/)*
