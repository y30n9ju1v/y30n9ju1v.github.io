---
title: "BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers"
date: 2026-04-14T00:00:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving"]
tags: ["Autonomous Driving", "BEV", "Transformer", "3D Object Detection", "Multi-Camera"]
---

## 개요

- **저자**: Zhiqi Li, Wenhai Wang, Hongyang Li, Enze Xie, Chonghao Sima, Tong Lu, Yu Qiao, Jifeng Dai
- **소속**: Nanjing University, Shanghai AI Laboratory, The University of Hong Kong
- **발행년도**: 2022 (arXiv:2203.17270)
- **주요 내용**: 다중 카메라 이미지로부터 Bird's-Eye-View(BEV) 표현을 학습하는 Spatiotemporal Transformer 프레임워크. 공간 정보(Spatial Cross-Attention)와 시간 정보(Temporal Self-Attention)를 동시에 활용하여 3D 객체 탐지와 맵 분할을 통합 지원하며, nuScenes test set에서 56.9% NDS를 달성해 LiDAR 기반 방법과 대등한 성능을 보임.

## 목차

- Chapter 1: Introduction — BEV 표현의 필요성과 기존 방법의 한계
- Chapter 2: Related Work — Transformer 기반 2D/3D 인식 방법 정리
- Chapter 3: BEVFormer 구조 — BEV Queries, Spatial Cross-Attention, Temporal Self-Attention
- Chapter 4: Experiments — nuScenes / Waymo 벤치마크 결과 및 소거 실험
- Chapter 5: Discussion and Conclusion

---

## Chapter 1: Introduction

**요약**

자율주행에서 3D 시각 인식(3D bounding box 예측, 의미론적 맵 분할)은 필수적이다. 기존에는 LiDAR 기반 방법이 주류였으나, 카메라 기반 방법은 배포 비용이 낮고 교통 표지판 등 시각 정보를 더 잘 인식한다는 장점이 있다.

단순히 단안(monocular) 프레임워크를 여러 카메라에 적용하는 방법은 카메라 간 정보 공유가 어렵고 성능이 떨어진다. 더 통합적인 접근법으로 **Bird's-Eye-View(BEV)** 표현을 사용하는 방식이 주목받는다. BEV는 객체의 위치와 크기를 직관적으로 나타내며 여러 카메라 뷰를 하나의 공간에서 통합할 수 있다.

그러나 기존 BEV 생성 방식은 깊이 정보(depth estimation)에 의존하기 때문에 오차가 누적되는 문제가 있다. 또한 자율주행에서는 **시간적 정보(temporal information)**가 중요한데 — 예를 들어 움직이는 물체의 속도 추정, 가려진 물체 감지 — 기존 방법은 이를 충분히 활용하지 못한다.

BEVFormer는 이 두 가지 문제를 동시에 해결하는 Transformer 기반 프레임워크다.

**핵심 개념**

- **Bird's-Eye-View (BEV)**: 자동차 위에서 내려다본 2D 평면 표현. 객체의 실제 위치와 방향을 직접 표현할 수 있어 자율주행에 적합
- **Depth estimation 의존 문제**: 카메라 2D 이미지에서 BEV를 만들 때 깊이를 추정해야 하는데, 이 추정 오차가 BEV 품질에 직접 영향을 미침
- **Temporal information**: 이전 시간 단계의 장면 정보. 현재 시점만으로는 추론하기 어려운 속도, 가림 상황 등을 보완

---

## Chapter 2: Related Work

**요약**

**Transformer 기반 2D 인식**: DETR은 object query와 cross-attention으로 end-to-end 탐지를 실현했다. Deformable DETR은 각 query가 소수의 K개 reference point에만 attention을 계산하는 방식으로 효율을 높였다. BEVFormer는 이 Deformable Attention을 3D 공간으로 확장한다.

**카메라 기반 3D 인식**: FCOS3D, PGD 등은 단안 카메라로 3D bounding box를 예측한다. DETR3D, BEVDet 등은 다중 카메라를 활용한다. 맵 분할 분야에서는 Lift-Splat, VPN 등이 BEV 기반 접근을 시도했으나 깊이 의존 문제가 있었다.

**핵심 개념**

- **Deformable Attention**: 전체 feature map 대신 각 query마다 소수의 key point에만 attention을 수행하여 연산량을 줄이는 방식

$$\text{DeformAttn}(q, p, x) = \sum_{i=1}^{N_{\text{head}}} \mathcal{W}_i \sum_{j=1}^{N_{\text{key}}} A_{ij} \cdot \mathcal{W}'_i x(p + \Delta p_{ij})$$

**수식 설명**:
- **$q$**: 현재 처리 중인 query (BEV 그리드의 한 셀)
- **$p$**: reference point (query의 기준 위치)
- **$x$**: 입력 feature map
- **$N_{\text{head}}$**: attention head 수 (병렬로 다양한 패턴을 학습)
- **$N_{\text{key}}$**: 각 head에서 sampling하는 key point 수
- **$A_{ij}$**: j번째 key point에 대한 attention weight (중요도, 합이 1)
- **$\Delta p_{ij}$**: 학습된 offset (reference point로부터 얼마나 떨어진 곳을 볼지)
- **$\mathcal{W}_i, \mathcal{W}'_i$**: 학습 가능한 투영 행렬

즉, 각 query는 정해진 기준점 근처의 K개 위치만 보고 가중 합산하여 attention 결과를 계산한다. 전체를 다 보는 global attention보다 훨씬 효율적이다.

---

## Chapter 3: BEVFormer

**요약**

BEVFormer는 6개의 encoder layer로 구성되며, 각 layer는 **Temporal Self-Attention → Spatial Cross-Attention → Feed-Forward Network** 순서로 동작한다.

### 3.1 전체 구조

타임스텝 $t$에서 다중 카메라 이미지 $\{F_t^i\}_{i=1}^{N_{\text{view}}}$를 backbone(ResNet 등)으로 처리해 multi-camera feature를 얻는다. BEV queries $Q$가 두 가지 attention을 통해 공간·시간 정보를 수집하고, 최종 BEV feature $B_t \in \mathbb{R}^{H \times W \times C}$를 생성한다. 이 $B_t$를 3D 탐지 헤드와 맵 분할 헤드에 입력하여 최종 예측을 수행한다.

### 3.2 BEV Queries

$$Q \in \mathbb{R}^{H \times W \times C}$$

BEV 평면을 $H \times W$ 격자로 나누고, 각 격자 셀마다 학습 가능한 벡터 $Q_p \in \mathbb{R}^{1 \times C}$를 정의한다. 격자 셀의 실제 세계 위치는 다음과 같이 계산된다:

$$x' = (x - \frac{W}{2}) \times s, \quad y' = (y - \frac{H}{2}) \times s$$

**수식 설명**:
- **$(x, y)$**: BEV 격자 내 픽셀 좌표 (0~W, 0~H)
- **$(x', y')$**: 실제 세계 좌표 (미터 단위)
- **$s$**: BEV 격자 해상도 (한 셀 = $s$ 미터, 기본값 0.512m)
- BEV의 중심이 자차(ego car) 위치에 해당

각 쿼리에 위치 임베딩(positional embedding)을 추가하여 공간 정보를 인코딩한다.

### 3.3 Spatial Cross-Attention (SCA)

각 BEV query $Q_p$가 여러 카메라 이미지에서 관련 특징을 추출하는 모듈이다. 핵심 아이디어는 깊이를 직접 추정하지 않고, **여러 높이(anchor heights)에서 3D 포인트를 샘플링**하여 2D 이미지에 투영하는 것이다.

$$\text{SCA}(Q_p, F_t) = \frac{1}{|\mathcal{V}_{\text{hit}}|} \sum_{i \in \mathcal{V}_{\text{hit}}} \sum_{j=1}^{N_{\text{ref}}} \text{DeformAttn}(Q_p, \mathcal{P}(p, i, j), F_t^i)$$

**수식 설명**:
- **$Q_p$**: BEV 위치 $p$의 query
- **$\mathcal{V}_{\text{hit}}$**: query의 3D 포인트가 실제로 투영되는 카메라 뷰의 집합 (모든 카메라가 아닌 관련된 카메라만)
- **$|\mathcal{V}_{\text{hit}}|$**: 히트된 카메라 수 (정규화용)
- **$N_{\text{ref}}$**: 각 query당 reference point 수 (높이 방향으로 $N_z$개 anchor)
- **$\mathcal{P}(p, i, j)$**: j번째 3D 포인트를 i번째 카메라 이미지에 투영한 2D 좌표

3D → 2D 투영 공식:

$$\mathcal{P}(p, i, j) = (x_{ij}, y_{ij})$$
$$z_{ij} \cdot [x_{ij} \quad y_{ij} \quad 1]^T = T_i \cdot [x' \quad y' \quad z'_j \quad 1]^T$$

**수식 설명**:
- **$T_i \in \mathbb{R}^{3 \times 4}$**: i번째 카메라의 투영 행렬 (3D 세계 → 2D 이미지)
- **$z'_j$**: j번째 anchor height (예: -5m ~ 3m 사이 균등 샘플링)
- 한 BEV 셀에서 여러 높이의 포인트를 투영함으로써 깊이 모호성을 처리

### 3.4 Temporal Self-Attention (TSA)

이전 타임스텝의 BEV feature $B_{t-1}$을 활용하는 모듈이다. 먼저 ego-motion을 보정하여 이전 BEV를 현재 좌표계에 정렬($B'_{t-1}$)한 뒤, 현재 query $Q$와 함께 deformable attention을 수행한다.

$$\text{TSA}(Q_p, \{Q, B'_{t-1}\}) = \sum_{V \in \{Q, B'_{t-1}\}} \text{DeformAttn}(Q_p, p, V)$$

**수식 설명**:
- **$Q_p$**: 현재 타임스텝의 BEV query
- **$B'_{t-1}$**: ego-motion 보정된 이전 타임스텝 BEV feature
- **$V$**: attention의 value (현재 query $Q$ 또는 이전 BEV $B'_{t-1}$)
- 두 소스에 대한 attention 결과를 합산하여 시공간 정보를 통합
- 학습 시에는 $t-3, t-2, t-1, t$ 총 4개 타임스텝을 사용해 RNN 방식으로 BEV를 순차 생성

### 3.5 BEV Feature 활용

생성된 $B_t$를 두 가지 태스크 헤드에 연결한다:
- **3D Object Detection**: Deformable DETR 기반 헤드. 3D bounding box와 속도를 NMS 없이 end-to-end로 예측
- **Map Segmentation**: Panoptic SegFormer 기반 헤드. 차량, 도로, 차선 등 의미론적 카테고리별 클래스 고정 query 사용

---

## Chapter 4: Experiments

**요약**

### 4.1 데이터셋

- **nuScenes**: 1000개 장면, 6대 카메라(360° FOV), 2Hz 주석. 평가 지표는 NDS(nuScenes Detection Score)와 mAP
- **Waymo Open Dataset**: 798개 학습 / 202개 검증 시퀀스, 252° FOV. APH(Average Precision with Heading) 사용

### 4.2 3D 탐지 결과

| Method | Modality | NDS↑ | mAP↑ |
|--------|----------|------|------|
| FCOS3D | Camera | 0.428 | 0.358 |
| PGD | Camera | 0.448 | 0.386 |
| DETR3D | Camera | 0.479 | 0.412 |
| **BEVFormer** | **Camera** | **0.569** | **0.481** |
| SSN | LiDAR | 0.569 | 0.463 |

- nuScenes test에서 **56.9% NDS** 달성, 이전 최고 카메라 방법(DETR3D) 대비 9.0 포인트 향상
- LiDAR 기반 SSN(56.9% NDS)과 대등한 성능

### 4.3 소거 실험 (Ablation Study)

**Spatial Cross-Attention 효과**:
- Global attention: NDS 0.404 (GPU 메모리 과다)
- Point-based attention: NDS 0.423
- **Deformable (Local) attention**: NDS 0.448 (최고)

**Temporal Self-Attention 효과**:
- BEVFormer-S (시간 정보 없음): NDS 0.448, mAVE 0.802
- BEVFormer (시간 정보 있음): NDS 0.517, mAVE 0.394
- 시간 정보가 **속도 추정(mAVE) 오차를 절반 이상 감소**시킴

**가시성별 성능**: 0~40% 가시성(심하게 가려진 객체)에서 BEVFormer가 DETR3D 대비 평균 recall 6.0% 이상 향상. 시간 정보가 가려진 물체 감지에 크게 기여함.

**핵심 개념**

- **NDS (nuScenes Detection Score)**: 탐지 정확도(mAP)와 5가지 TP 지표(위치, 크기, 방향, 속도, 속성 오차)를 종합한 점수
$$\text{NDS} = \frac{1}{10}\left[5\text{mAP} + \sum_{\text{mTP} \in \mathbb{TP}} (1 - \min(1, \text{mTP}))\right]$$
- **mAVE (mean Average Velocity Error)**: 예측 속도와 실제 속도의 평균 오차 (m/s). 낮을수록 좋음

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **BEV (Bird's-Eye-View)** | 위에서 내려다본 2D 평면 표현. 자율주행의 표준 좌표계 |
| **BEV Queries** | BEV 평면의 각 격자 셀에 대응하는 학습 가능한 벡터. 공간·시간 정보를 수집하는 주체 |
| **Spatial Cross-Attention** | BEV query가 다중 카메라 이미지에서 관련 특징을 추출. 여러 높이 anchor로 깊이 모호성 해소 |
| **Temporal Self-Attention** | 이전 타임스텝 BEV를 ego-motion 보정 후 현재 BEV와 융합. RNN과 유사한 방식 |
| **Deformable Attention** | 전체 feature map 대신 소수의 reference point만 attention. 효율적이고 지역적 수용 영역 보유 |
| **Ego-motion compensation** | 자차 이동을 고려해 이전 BEV를 현재 좌표계로 정렬하는 전처리 |
| **Anchor Heights** | 각 BEV 셀에서 여러 높이($z$ 값)를 미리 정의하여 3D 포인트를 샘플링. 깊이 추정 없이 다양한 높이 객체를 포착 |

---

## 결론 및 시사점

BEVFormer는 다중 카메라 이미지에서 BEV 표현을 생성하는 새로운 패러다임을 제시했다:

1. **깊이 추정 불필요**: Spatial Cross-Attention이 여러 높이에서 직접 이미지 특징을 샘플링하여 깊이 오차 누적 문제를 회피
2. **시간 정보 효율적 활용**: Temporal Self-Attention이 이전 BEV를 RNN처럼 재귀적으로 활용하여 속도 추정 정확도와 가려진 객체 감지를 크게 향상
3. **통합 멀티태스크 프레임워크**: 하나의 BEV encoder로 3D 탐지와 맵 분할을 동시에 지원
4. **LiDAR급 성능**: 카메라만으로 LiDAR 기반 방법과 대등한 NDS 달성

**한계**: 카메라 기반 방법은 여전히 LiDAR 대비 효율성과 정확도에서 격차가 있으며, 2D 이미지에서 정확한 3D 위치를 추론하는 문제는 근본적 과제로 남아있다.

**실무적 시사점**: BEVFormer의 통합 BEV 표현은 자율주행 스택의 다양한 하위 태스크(계획, 예측 등)에 공통 입력으로 활용될 수 있어, 이후 UniAD 등 end-to-end 자율주행 연구의 기반이 되었다.


---

*관련 논문: [Attention Is All You Need](/posts/papers/attention-is-all-you-need/), [Lift Splat Shoot](/posts/papers/lift-splat-shoot/), [BEVDepth](/posts/papers/bevdepth/), [DETR3D](/posts/papers/detr3d-3d-object-detection-multi-view-images/), [BEVFusion](/posts/papers/bevfusion-multi-task-multi-sensor-fusion/), [UniAD](/posts/papers/uniad-planning-oriented-autonomous-driving/), [nuScenes](/posts/papers/nuscenes-multimodal-dataset-autonomous-driving/)*
