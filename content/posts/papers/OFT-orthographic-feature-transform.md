---
title: "Orthographic Feature Transform for Monocular 3D Object Detection"
date: 2026-04-19T11:00:00+09:00
draft: false
categories: ["Papers"]
tags: ["3D Object Detection", "Monocular", "BEV", "Autonomous Driving", "KITTI"]
---

## 개요

- **저자**: Thomas Roddick, Alex Kendall, Roberto Cipolla (University of Cambridge)
- **발행년도**: 2018 (arXiv:1811.08188)
- **주요 내용**: 단안 카메라(monocular) 이미지로부터 3D 바운딩 박스를 검출하기 위해, perspective 이미지 피처를 orthographic birds-eye-view(BEV) 공간으로 변환하는 **OFT(Orthographic Feature Transform)**를 제안. KITTI 3D 객체 검출 벤치마크에서 monocular 방식 SOTA 달성.

## 한계 극복

- **기존 한계 1 — 원근 왜곡(Perspective distortion)**: 기존 이미지 기반 방법은 물체가 멀수록 크기가 작아지고 거리 추정이 어려움
- **기존 한계 2 — 독립적 proposal 처리**: Mono3D 등 기존 방법은 각 3D proposal을 독립적으로 처리해 장면 전체의 3D 공간 배치를 함께 추론하지 못함
- **기존 한계 3 — 명시적 깊이 정보 의존**: 스테레오나 LiDAR 없이 단안으로 깊이를 추정하기 어려움
- **이 논문의 접근 방식**: perspective 이미지 피처를 OFT로 BEV 공간에 매핑한 뒤, topdown network로 3D 공간에서 통합적으로 추론. 깊이를 명시적으로 추정하지 않고도 BEV 표현을 학습으로 구성

## 목차

- Section 1: Introduction
- Section 2: Related Work
- Section 3: 3D Object Detection Architecture
  - 3.1 Feature extraction
  - 3.2 Orthographic feature transform
  - 3.3 Topdown network
  - 3.4 Confidence map prediction
  - 3.5 Localization and bounding box estimation
  - 3.6 Non-maximum suppression
- Section 4: Experiments
- Section 5: Discussion
- Section 6: Conclusions

---

## Section 1: Introduction

**요약**

자율주행에서 3D 물체 감지는 예측·회피·경로 계획의 핵심이지만, 단안 이미지 기반 방법은 LiDAR 대비 성능이 10%에도 미치지 못한다. 원인은 perspective 이미지 표현에서 물체의 크기·외관이 거리에 따라 크게 변하고 절대 거리를 알기 어렵기 때문이다.

이 논문은 "3D 추론은 3D 공간에서 해야 한다"는 핵심 통찰 아래, 이미지 피처를 orthographic BEV로 변환하는 OFT를 제안한다. BEV에서는 스케일이 균일하고, 외관이 시점에 덜 의존하며, 물체 간 거리가 의미 있어 3D 추론에 유리하다.

**핵심 개념**

- **Perspective projection**: 멀수록 물체가 작게 보이는 원근 투영. 3D 위치를 역추적하기 매우 어려움
- **Orthographic BEV (Birds-Eye-View)**: 위에서 내려다본 평면 표현. 스케일 일관성이 있어 자율주행 LiDAR 방법에서 주로 사용
- **OFT (Orthographic Feature Transform)**: perspective 이미지 피처 → BEV 피처로 변환하는 미분 가능한 변환

---

## Section 2: Related Work

**요약**

- **2D 검출**: YOLO, SSD(단일 스테이지), Faster-RCNN, FPN(이중 스테이지)로 구분. OFT는 단일 스테이지 구조를 채택
- **LiDAR 기반 3D 검출**: PointNet, BirdNet 등 BEV 표현이 효과적으로 사용됨. 이 논문은 이미지만으로 BEV를 암묵적으로 구성하려 함
- **이미지 기반 3D 검출**: 2D bbox에서 3D 파라미터 회귀(Mono3D, 3DOP 등). 공통 한계: proposal을 독립적으로 처리해 장면 전체 3D 배치 추론 불가
- **Integral images**: Viola-Jones 이후 많은 방법에서 사용되었으나, end-to-end 역전파와 결합한 것은 이 논문이 처음에 가까움

---

## Section 3: 3D Object Detection Architecture

**요약**

전체 파이프라인은 5단계:
1. ResNet-18 프론트엔드로 멀티스케일 이미지 피처 추출
2. OFT로 이미지 피처 → BEV 피처 변환
3. Topdown network(ResNet 잔차 블록)로 BEV 피처 처리
4. 출력 헤드(confidence, position offset, dimension offset, orientation)
5. NMS로 최종 바운딩 박스 디코딩

---

### 3.1 Feature Extraction

ResNet-18(병목 레이어 없음)을 사용해 원본 해상도의 1/8, 1/16, 1/32 크기의 멀티스케일 피처맵 추출. 1×1 conv로 모두 256차원으로 통일. 프론트엔드가 상대적으로 얕은 이유는 3D 추론 부분(topdown network)에 더 많은 역할을 부여하기 위함.

---

### 3.2 Orthographic Feature Transform (OFT)

**요약**

이미지 피처 $\mathbf{f}(u, v) \in \mathbb{R}^n$을 3D 복셀 피처 $\mathbf{g}(x, y, z) \in \mathbb{R}^n$으로 매핑하고, 이를 다시 2D 지면 평면 피처 $\mathbf{h}(x, z)$로 압축한다.

각 복셀 $(x, y, z)$의 이미지 투영 범위(bounding box)는 다음 수식으로 계산:

$$u_1 = f\frac{x - 0.5r}{z + 0.5\frac{x}{|x|}r} + c_u, \quad u_2 = f\frac{x + 0.5r}{z - 0.5\frac{x}{|x|}r} + c_u$$

$$v_1 = f\frac{y - 0.5r}{z + 0.5\frac{y}{|y|}r} + c_v, \quad v_2 = f\frac{y + 0.5r}{z - 0.5\frac{y}{|y|}r} + c_v$$

**수식 설명 (Eq. 1)**

각 3D 복셀(크기 $r$의 정육면체)이 이미지 평면에 투영될 때의 사각형 범위를 계산:
- **$f$**: 카메라 초점 거리 (focal length). 클수록 물체가 크게 찍힘
- **$(c_u, c_v)$**: 이미지 주점 (principal point). 카메라 광학 중심의 이미지 좌표
- **$r$**: 복셀 한 변의 크기 (예: 0.5m)
- **$(u_1, v_1), (u_2, v_2)$**: 복셀이 이미지에 투영된 사각형의 좌상단·우하단 픽셀 좌표
- 멀리 있는 복셀($z$가 클수록) 이미지에서 더 작은 영역으로 투영됨

이미지 영역의 평균 풀링으로 복셀 피처를 계산:

$$\mathbf{g}(x, y, z) = \frac{1}{(u_2 - u_1)(v_2 - v_1)} \sum_{u=u_1}^{u_2} \sum_{v=v_1}^{v_2} \mathbf{f}(u, v)$$

**수식 설명 (Eq. 2)**

해당 복셀의 이미지 투영 영역 내 피처를 단순 평균:
- **$\mathbf{g}(x,y,z)$**: 3D 공간의 복셀 위치 $(x,y,z)$에 할당될 피처 벡터
- **$\mathbf{f}(u,v)$**: 이미지 픽셀 $(u,v)$의 피처 벡터
- **분모 $(u_2-u_1)(v_2-v_1)$**: 투영 영역의 픽셀 수 (평균 정규화)

수직 방향을 압축해 2D BEV 피처 생성:

$$\mathbf{h}(x, z) = \sum_{y=y_0}^{y_0+H} W(y)\,\mathbf{g}(x, y, z)$$

**수식 설명 (Eq. 3)**

3D 복셀 피처를 수직($y$)방향으로 가중 합산해 BEV 피처맵 생성:
- **$\mathbf{h}(x,z)$**: 지면 평면 위 위치 $(x,z)$의 2D BEV 피처 (최종적으로 topdown network의 입력)
- **$W(y) \in \mathbb{R}^{n \times n}$**: 높이 $y$에 따른 학습 가중 행렬. 수직 정보를 보존하면서 압축
- **$y_0$**: 카메라 아래 지면 평면까지의 거리
- 이 중간 복셀 표현을 거쳐야 물체의 높이·수직 위치 정보를 보존할 수 있음

#### 3.2.1 Integral Image를 이용한 빠른 Average Pooling

일반적인 BEV 설정에서 약 15만 개의 복셀 bbox를 처리해야 해, 단순 루프로는 불가능. Integral image(누적합 이미지)를 사용해 복셀 크기에 무관하게 $O(1)$로 pooling:

$$\mathbf{F}(u, v) = \mathbf{f}(u,v) + \mathbf{F}(u-1,v) + \mathbf{F}(u,v-1) - \mathbf{F}(u-1,v-1)$$

**수식 설명 (Eq. 4)**

Integral feature map $\mathbf{F}$의 재귀적 계산:
- **$\mathbf{F}(u,v)$**: $(0,0)$부터 $(u,v)$까지의 피처 누적합
- 이 값을 한 번만 계산해두면, 임의의 사각형 영역의 합을 4번의 참조로 즉시 계산 가능

$$\mathbf{g}(x,y,z) = \frac{\mathbf{F}(u_1,v_1) + \mathbf{F}(u_2,v_2) - \mathbf{F}(u_1,v_2) - \mathbf{F}(u_2,v_1)}{(u_2-u_1)(v_2-v_1)}$$

**수식 설명 (Eq. 5)**

Integral image로 임의 사각형 내 피처 평균을 $O(1)$에 계산:
- 사각형 네 꼭짓점의 누적합 값을 더하고 빼서 영역합 획득 (포함-배제 원리)
- 원본 피처맵 $\mathbf{f}$에 대해 완전히 미분 가능 → end-to-end 학습 가능

**핵심 개념**

- **Integral image**: 픽셀 $(0,0)$~$(u,v)$ 전체의 누적합을 저장하는 이미지. 임의 영역의 합을 $O(1)$로 계산
- **미분 가능성**: 이 연산이 역전파 가능하므로 전체 파이프라인을 end-to-end 학습 가능

---

### 3.3 Topdown Network

**요약**

OFT 출력인 BEV 피처맵 $\mathbf{h}(x,z)$를 입력으로 받아 ResNet 스타일 skip connection 16-layer conv 네트워크로 처리. BEV 공간에서 합성곱 연산을 수행하므로:
- 카메라에서 멀리 있는 영역도 가까운 영역과 동일한 처리를 받음 (scale-invariant)
- 이미지의 perspective 효과 없이 순수하게 3D 구조만 추론

**핵심 개념**

- **Topdown network**: BEV 피처를 처리하는 보조 CNN. "3D에서의 추론"을 담당하는 핵심 모듈
- **위치 불변성(Location invariance)**: conv가 슬라이딩되므로 먼 위치·가까운 위치를 동일한 필터로 처리

---

### 3.4 Confidence Map Prediction

**요약**

분류(cross-entropy) 대신 연속 confidence map을 회귀(L1 loss). 각 지면 위치 $(x,z)$에 대해 객체 존재 확률을 Gaussian으로 표현:

$$S(x, z) = \max_i \exp\!\left(-\frac{(x_i-x)^2 + (z_i-z)^2}{2\sigma^2}\right)$$

**수식 설명 (Eq. 6)**

- **$S(x,z)$**: 위치 $(x,z)$에 객체 중심이 있을 신뢰도 (0~1)
- **$(x_i, z_i)$**: $i$번째 GT 객체 중심의 BEV 좌표
- **$\sigma$**: Gaussian 너비. 중심에서 멀수록 신뢰도가 지수적으로 감소
- **$\max_i$**: 여러 객체가 있을 때 가장 가까운 객체의 신뢰도를 사용

양성 위치보다 음성 위치가 압도적으로 많아 손실이 편향되는 문제를 해결하기 위해, $S < 0.05$인 음성 위치의 손실을 $10^{-2}$ 배로 스케일 다운.

---

### 3.5 Localization and Bounding Box Estimation

각 격자 위치에서 3가지 추가 헤드를 예측:

**위치 오프셋** (격자 셀 중심 → 실제 객체 중심):

$$\Delta_{\text{pos}}(x,z) = \left[\frac{x_i-x}{\sigma},\ \frac{y_i-y_0}{\sigma},\ \frac{z_i-z}{\sigma}\right]^\top$$

**크기 오프셋** (클래스 평균 크기 대비 로그 스케일):

$$\Delta_{\text{dim}}(x,z) = \left[\log\frac{w_i}{\bar{w}},\ \log\frac{h_i}{\bar{h}},\ \log\frac{l_i}{\bar{l}}\right]^\top$$

**방향 벡터** (y축 회전각의 sin/cos):

$$\Delta_{\text{ang}}(x,z) = \left[\sin\theta_i,\ \cos\theta_i\right]^\top$$

**수식 설명**

- BEV 공간에서 직접 y축 회전각 $\theta$를 예측 가능. 이미지 기반 방법은 시점에 따른 보정이 필요한 observation angle $\alpha$를 예측해야 하지만, OFT는 그 필요 없음
- 로그 스케일 크기 오프셋: 크기 비율을 안정적으로 예측하기 위함 (곱셈 관계를 덧셈으로 변환)

---

### 3.6 Non-Maximum Suppression

일반적인 3D NMS는 $O(N^2)$ bbox overlap 계산이 필요하나, BEV confidence map에서 2D local maximum을 찾는 방식으로 대체:

$$\hat{S}(x_i, z_i) \geq \hat{S}(x_i+m, z_i+n) \quad \forall m,n \in \{-1,0,1\}$$

**수식 설명 (Eq. 10)**

- **$\hat{S}$**: Gaussian 스무딩($\sigma_{\text{NMS}}$)이 적용된 confidence map
- 주변 8개 이웃보다 크면 local maximum → 객체로 판정
- BEV 특성상 두 객체가 같은 공간을 점유할 수 없어 피크가 자연스럽게 분리됨

---

## Section 4: Experiments

**실험 설정**

- **데이터셋**: KITTI 3D object detection benchmark (학습 3712장, 검증 3769장)
- **복셀 그리드**: 80m × 4m × 80m, 해상도 0.5m
- **데이터 증강**: random crop, scaling, horizontal flip (카메라 캘리브레이션 파라미터도 함께 조정)
- **학습**: SGD 600 epoch, batch 8, momentum 0.9, lr $10^{-7}$, group normalization 사용

**정량적 결과**

| 방법 | 모달리티 | AP3D Easy | AP3D Mod | AP3D Hard | APBEV Easy | APBEV Mod | APBEV Hard |
|------|----------|-----------|----------|-----------|------------|-----------|------------|
| 3D-SSMFCNN | Mono | 2.28 | 2.39 | 1.52 | 3.66 | 3.19 | 3.45 |
| **OFT-Net (Ours)** | Mono | **2.50** | **3.28** | **2.27** | **9.50** | **7.99** | **7.51** |

검증 세트에서 Mono3D(mono) 대비 전 카테고리 우세, 스테레오 방식 3DOP에 근접한 성능 달성.

**핵심 관찰**

- 멀리 있거나 가려진 물체(Hard category)에서 특히 큰 개선
- Mono3D가 탐지 못하는 먼 거리 객체를 OFT-Net이 일관적으로 탐지

---

## Section 5: Discussion

### 5.1 거리에 따른 성능

최소 거리를 높여가며 평가하면, OFT-Net의 성능 저하가 Mono3D보다 훨씬 완만함. BEV 공간의 스케일 일관성 덕분에 멀리 있는 객체도 동일하게 처리할 수 있기 때문.

### 5.2 학습 중 Confidence Map 진화

- 초기(Epoch 10~30): 객체의 투영선 방향으로 신뢰도가 넓게 퍼짐 → 깊이 불확실성이 큼
- 후반(Epoch 70~90): 피크가 GT 중심 주변으로 점점 수렴 → 깊이 추정 능력 향상
- 이는 단안에서 깊이 추정이 인식보다 훨씬 어렵다는 직관과 일치

### 4.4 Ablation Study

Topdown network 레이어 수를 줄이면 성능이 급격히 하락. ResNet-18 + 16-layer topdown이 ResNet-34 + topdown 없음보다 파라미터 수가 비슷함에도 훨씬 높은 성능 → **3D 추론 모듈의 중요성** 입증.

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **OFT (Orthographic Feature Transform)** | perspective 이미지 피처를 BEV 3D 복셀 → 2D 지면 평면 피처로 변환하는 미분 가능한 모듈 |
| **Integral Image (누적합 이미지)** | 영역 합을 $O(1)$로 계산하는 자료구조. 15만 복셀을 효율적으로 처리하게 함 |
| **BEV (Birds-Eye-View)** | 위에서 아래로 바라보는 평면 표현. 스케일 일관성·의미 있는 거리 표현이 장점 |
| **Topdown Network** | BEV 피처에서 작동하는 CNN. perspective 효과 없이 순수 3D 공간 추론 담당 |
| **Confidence Map** | 각 지면 위치의 객체 존재 확률을 Gaussian으로 표현한 연속 맵 |
| **Group Normalization** | 소 배치 학습 시 Batch Norm 대신 사용. 배치 크기에 덜 민감 |

---

## 결론 및 시사점

OFT-Net은 명시적인 깊이 추정 없이도, 이미지 피처를 BEV 공간으로 변환하고 그 공간에서 3D 추론을 수행함으로써 단안 3D 검출 성능을 크게 향상시켰다.

**실무적 시사점**

- **자율주행 합성 데이터 생성 관점**: BEV 표현은 합성 데이터와 실 데이터의 도메인 갭이 이미지보다 작을 수 있어, 합성 BEV 데이터로의 훈련이 효과적일 수 있음
- **후속 연구의 토대**: 이 논문의 BEV lift 아이디어는 이후 BEVDet, BEVFusion, LSS(Lift-Splat-Shoot) 등 현대 BEV 인식 패러다임의 직접적인 선구 작업
- **단일 스테이지 구조**: OFT 덕분에 anchor-free 단일 스테이지로 설계 가능 → 추론 속도 이점
- **한계**: absolute depth 정보 부재로 인해 인식은 빠르게 학습하지만 깊이 추정은 여전히 어렵고, 먼 거리에서 불확실성이 높음
