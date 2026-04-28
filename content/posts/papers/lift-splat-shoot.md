---
title: "Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by Implicitly Unprojecting to 3D"
date: 2026-04-17T08:00:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving"]
tags: ["Autonomous Driving", "BEV", "Multi-Camera", "3D Object Detection"]
---

## 개요

- **저자**: Jonah Philion, Sanja Fidler (NVIDIA, University of Toronto, Vector Institute)
- **발행년도**: 2020 (arXiv:2008.05711)
- **주요 내용**: 임의의 카메라 리그로부터 멀티뷰 이미지를 받아, 별도의 깊이 센서(LiDAR) 없이 Bird's-Eye-View(BEV) 표현을 직접 학습하는 end-to-end 아키텍처 "Lift-Splat-Shoot"을 제안. 객체 분할, 지도 분할, 모션 플래닝 태스크에서 기존 방법을 능가함.

## 한계 극복

이 논문이 기존 연구의 어떤 한계를 극복하기 위해 작성되었는지 설명합니다.

- **기존 한계 1 — 단일 이미지 패러다임의 한계**: 기존 컴퓨터 비전 알고리즘은 입력 이미지와 동일한 좌표계 내에서 예측을 출력하도록 설계되어 있어, 자율주행에서 필요한 ego-car 기준의 BEV 좌표계 출력에 바로 적용하기 어려웠습니다.
- **기존 한계 2 — 카메라 간 융합의 비미분성**: 개별 카메라 이미지에 단일 이미지 탐지기를 따로 적용한 뒤 결과를 ego 프레임으로 변환하는 방식은, 카메라 간 최적 융합 전략을 데이터 기반으로 학습하거나 downstream planner의 피드백으로 역전파할 수 없었습니다.
- **기존 한계 3 — 깊이 불확실성 미반영**: OFT(Orthographic Feature Transform) 같은 기존 방법은 한 픽셀의 feature를 깊이와 무관하게 모든 voxel에 동일하게 기여시켜, 실제 깊이 분포를 반영하지 못했습니다.
- **이 논문의 접근 방식**: 각 픽셀마다 깊이에 대한 확률 분포(categorical distribution)를 예측하고, 이를 이용해 frustum 형태의 3D 피처 포인트 클라우드를 생성(Lift)한 뒤, 이를 BEV 평면에 통합(Splat)하여 하나의 통합된 BEV CNN으로 처리합니다. 전체 파이프라인이 end-to-end로 미분 가능하며, 카메라 캘리브레이션에 대한 강건성도 데이터로부터 학습합니다.

## 목차

- Section 1: Introduction
- Section 2: Related Work (Monocular Object Detection / BEV Inference)
- Section 3: Method (Lift / Splat / Shoot)
- Section 4: Implementation (Architecture Details / Frustum Pooling Cumulative Sum Trick)
- Section 5: Experiments and Results (Segmentation / Robustness / Zero-Shot Transfer / Oracle Depth / Motion Planning)
- Section 6: Conclusion

---

## Section 1: Introduction

**요약**

자율주행에서 인식 시스템은 여러 센서로부터 입력을 받아 ego-car 좌표계(BEV)에서 예측을 출력해야 합니다. 기존 컴퓨터 비전 모델들은 입력 이미지 좌표계 내에서 예측하도록 설계되어 있어 이 요구에 맞지 않습니다. 단순하게 각 카메라에 단일 이미지 탐지기를 적용한 뒤 결과를 변환하는 방법은 세 가지 대칭성(Translation equivariance, Permutation invariance, Ego-frame isometry equivariance)을 갖지만, 카메라 간 최적 융합 전략을 데이터 기반으로 학습하거나 역전파할 수 없다는 단점이 있습니다.

저자들은 이 세 가지 대칭성을 설계에 반영하면서도 end-to-end 미분 가능한 "Lift-Splat" 모델을 제안합니다.

**핵심 개념**

- **Translation equivariance**: 이미지 내 픽셀 좌표가 전부 이동하면 출력도 동일하게 이동하는 성질. 완전 합성곱 신경망이 이 성질을 가지며, 멀티뷰 확장도 이를 상속합니다.
- **Permutation invariance**: n개 카메라의 순서가 바뀌어도 최종 출력이 변하지 않는 성질.
- **Ego-frame isometry equivariance**: ego car가 회전/이동해도 동일 객체가 동일하게 탐지되는 성질.
- **End-to-end differentiability**: 센서 입력부터 최종 BEV 예측까지 모든 파라미터를 역전파로 함께 학습할 수 있어, downstream planner의 피드백이 인식 모듈에 전달됩니다.

---

## Section 2: Related Work

**요약**

관련 연구는 크게 두 가지 방향입니다.

1. **단안 객체 탐지(Monocular Object Detection)**: 이미지 평면에서 2D 탐지기를 학습한 뒤 3D bounding box를 회귀하는 방법, 또는 깊이 예측과 BEV 탐지를 분리 학습하는 "pseudolidar" 방법이 있습니다. Orthographic Feature Transform(OFT)은 고정된 voxel 큐브를 이미지에 투영해 feature를 모으지만, 깊이에 무관하게 동일한 feature를 기여하는 한계가 있습니다.
2. **BEV 프레임에서의 추론**: MonoLayout, Pyramid Occupancy Networks(PON), FISHING Net 등이 카메라 extrinsic/intrinsic을 이용해 이미지 표현을 BEV로 변환합니다.

**핵심 개념**

- **Pseudolidar**: 깊이 예측 네트워크와 BEV 탐지 네트워크를 분리 학습하는 방식. BEV 좌표계에서 유클리드 거리가 더 의미 있다는 장점이 있으나, end-to-end 학습이 어렵습니다.
- **OFT (Orthographic Feature Transform)**: 3D 공간의 voxel을 이미지에 투영해 feature를 수집. 그러나 깊이와 무관하게 동일 feature를 기여하는 한계가 있음.

---

## Section 3: Method

### 3.1 Lift: Latent Depth Distribution

**요약**

"Lift" 단계는 각 카메라 이미지를 2D 좌표계에서 3D 공간으로 올리는 과정입니다. 깊이는 근본적으로 단안 이미지에서 모호하므로, 네트워크는 각 픽셀에 대해 깊이에 대한 카테고리 분포 $\alpha$와 컨텍스트 벡터 $\mathbf{c}$를 예측합니다. 각 픽셀에 $|D|$개의 깊이 후보 포인트를 생성하여 frustum 형태의 포인트 클라우드를 만듭니다.

**핵심 개념**

- **카테고리 깊이 분포**: 연속적인 깊이 대신 이산적인 깊이 후보 집합 $D = \{d_0, d_0+\Delta, ..., d_0+|D|\Delta\}$에 대한 확률 분포를 예측합니다. 이를 통해 깊이 불확실성을 모델링합니다.
- **Frustum 형태의 포인트 클라우드**: 하나의 이미지에서 $D \cdot H \cdot W$개의 포인트가 생성됩니다. 멀티뷰 합성 커뮤니티의 "multi-plane image"와 동등하지만, 각 평면의 feature가 $(r,g,b,\alpha)$ 값 대신 추상적인 벡터입니다.

**수식**

$$\mathbf{c}_d = \alpha_d \mathbf{c}$$

**수식 설명**

이 수식은 깊이 $d$에 위치한 포인트 $p_d$의 feature 벡터를 정의합니다:
- **$\mathbf{c}_d \in \mathbb{R}^C$**: 깊이 $d$에 할당된 포인트의 최종 feature 벡터
- **$\alpha_d$**: 해당 픽셀에서 깊이 $d$에 대한 확률값 (깊이 분포의 $d$번째 원소). $\alpha \in \triangle^{|D|-1}$이므로 모든 $\alpha_d$의 합은 1
- **$\mathbf{c} \in \mathbb{R}^C$**: 픽셀 전체를 대표하는 컨텍스트 벡터 (깊이와 무관)
- **직관**: 네트워크가 특정 깊이 $d^*$에 one-hot으로 $\alpha$를 예측하면 pseudolidar처럼 동작하고, 균등 분포를 예측하면 OFT처럼 동작합니다. 실제로는 그 사이 어딘가에서 깊이 불확실성에 따라 유연하게 선택합니다.

### 3.2 Splat: Pillar Pooling

**요약**

"Splat" 단계는 Lift에서 생성된 대형 포인트 클라우드를 BEV 래스터 그리드로 통합합니다. PointPillars 아키텍처를 따라, "Pillar"(높이가 무한한 voxel)를 사용합니다. 각 포인트를 가장 가까운 pillar에 할당하고 sum pooling을 수행하여 $C \times H \times W$ 텐서를 생성합니다.

**핵심 개념**

- **Pillar Pooling**: 포인트들을 BEV 격자의 각 셀(pillar)에 누적하는 과정. Sum pooling을 사용하여 카메라 수에 무관한 고정 크기의 BEV 텐서를 출력합니다.
- **Cumulative Sum Trick**: 메모리 효율적인 sum pooling 구현. 포인트를 bin id로 정렬한 뒤 누적합을 계산하고 구간 경계값을 빼는 방법으로 패딩 없이 구현합니다. 역전파 속도를 2배 향상시킵니다.

### 3.3 Shoot: Motion Planning

**요약**

"Shoot" 단계는 Lift-Splat이 생성한 BEV 표현을 cost map으로 해석하여 end-to-end 모션 플래닝을 수행합니다. 테스트 시에는 K개의 template trajectory를 BEV cost map에 "쏘아서" 비용을 계산하고 최저 비용 경로를 선택합니다.

**핵심 개념**

- **Template trajectory**: K-Means로 클러스터링한 1K개의 자아 차량 궤적 템플릿 $\mathcal{T} = \{\tau_i\}_K = \{\{x_j, y_j, t_j\}_T\}_K$
- **Cost map**: 네트워크가 출력하는 공간적 비용 함수. 특정 위치를 지나는 것의 비용을 나타냅니다.

**수식**

$$p(\tau_i | o) = \frac{\exp\!\left(-\displaystyle\sum_{x_i, y_i \in \tau_i} c_o(x_i, y_i)\right)}{\displaystyle\sum_{\tau \in \mathcal{T}} \exp\!\left(-\displaystyle\sum_{x, y \in \tau} c_o(x, y)\right)}$$

**수식 설명**

이 수식은 관측값 $o$가 주어졌을 때 각 template trajectory $\tau_i$를 선택할 확률을 Boltzmann 분포로 정의합니다:
- **$p(\tau_i | o)$**: 관측 $o$에서 궤적 $\tau_i$를 선택할 확률
- **$c_o(x, y)$**: 네트워크가 예측한 cost map에서 위치 $(x, y)$의 비용값. 값이 클수록 그 위치를 지나는 것이 위험하거나 비쌈
- **$\sum_{x_i, y_i \in \tau_i} c_o(x_i, y_i)$**: 궤적 $\tau_i$를 따라가는 총 비용. 위험한 위치(장애물, 차선 경계 등)를 많이 지날수록 커짐
- **$\exp(-\text{비용})$**: 비용이 낮을수록 확률이 높아지는 softmax 구조
- **분모**: 모든 template에 대한 정규화 상수. 이를 통해 확률의 합이 1이 됨
- **학습**: ground-truth 궤적에 가장 가까운 template을 정답으로 하여 cross-entropy loss로 학습합니다. 이를 통해 cost map $c_o$가 자동으로 학습됨

---

## Section 4: Implementation

**요약**

**아키텍처**: 이미지별 CNN에는 EfficientNet-B0(ImageNet 사전학습)을 사용하고, BEV CNN에는 ResNet-18 기반의 FPN 구조를 사용합니다. 총 파라미터 수는 14.3M입니다.

**주요 하이퍼파라미터**:
- 입력 이미지 크기: $128 \times 352$
- BEV 그리드: $200 \times 200$ (각 셀 0.5m × 0.5m, $-50$m ~ $50$m 범위)
- 깊이 범위: $D \in [4.0\text{m}, 45.0\text{m}]$, 1m 간격 (총 41개 이산 깊이)
- 추론 속도: Titan V GPU에서 35 Hz

**핵심 개념**

- **Frustum Pooling Layer**: Lift 단계의 frustum 포인트 클라우드를 pillar 기반 BEV 텐서로 변환하는 레이어. 카메라 수 $n$에 무관하게 고정 크기 $C \times H \times W$ 출력
- **Cumulative Sum Trick의 수학적 원리**: 패딩 대신 정렬 + 누적합 + 경계 빼기로 sum pooling을 구현하여 메모리를 절약하고, 전체 모듈에 대한 해석적 기울기를 유도하여 autograd 속도를 2배 향상

---

## Section 5: Experiments and Results

**요약**

**데이터셋**: nuScenes(6카메라, 1K 장면, 20초)와 Lyft Level 5(6카메라) 데이터셋을 사용합니다.

**태스크**: 객체 분할(Car, Vehicle), 지도 분할(Drivable Area, Lane Boundary), 모션 플래닝

### 5.1 Segmentation 결과

| Method | nuScenes Car | nuScenes Vehicle | Lyft Car | Lyft Vehicle |
|---|---|---|---|---|
| CNN | 22.78 | 24.25 | 30.71 | 31.91 |
| OFT | 29.72 | 30.05 | 39.48 | 40.43 |
| **Lift-Splat** | **32.06** | **32.07** | **43.09** | **44.64** |

지도 분할에서도 Drivable Area IOU 72.94, Lane Boundary IOU 19.96으로 모든 기준선을 능가합니다.

### 5.2 Robustness (강건성)

- **Extrinsic noise 강건성**: 훈련 시 노이즈가 많은 extrinsic으로 학습한 모델은 테스트 시 noisy extrinsic에 더 강건합니다.
- **Camera dropout 강건성**: 훈련 시 매 샘플마다 카메라 1개를 무작위로 제거하면, 테스트 시 카메라가 누락되어도 성능이 잘 유지됩니다. 특히 6개 카메라가 모두 존재할 때 가장 좋은 모델은 dropout 훈련된 모델입니다.

### 5.3 Zero-Shot Camera Rig Transfer

- nuScenes 4개 카메라로 훈련하고, 테스트 시 추가 2개 카메라를 제공하면 성능이 단조 증가합니다 (IOU 26.53 → 27.94).
- nuScenes로만 훈련한 모델을 Lyft(완전히 다른 카메라 리그)에서 평가해도 기준선 대비 월등한 성능을 보입니다 (Lyft Car IOU: CNN 7.00, OFT 16.25, **Lift-Splat 21.35**).

### 5.4 Benchmarking Against Oracle Depth

LiDAR 깊이를 직접 사용하는 PointPillars와 비교 시, 카메라 전용 Lift-Splat은 여전히 차이가 있습니다. 다만 야간이나 장거리에서의 성능 차이가 두드러지며, 다중 시간 프레임 활용이 향후 과제입니다.

### 5.5 Motion Planning

1K template trajectory 중 ground-truth와 가장 가까운 template을 예측하는 분류 태스크:
- Lidar (1 scan): Top-5 19.27%, Top-10 28.88%, Top-20 41.93%
- **Lift-Splat**: Top-5 15.52%, Top-10 19.94%, Top-20 27.99%

Lidar 기반보다 낮지만, 카메라만으로 교차로 정지, 차선 유지 등 합리적인 궤적을 예측합니다.

---

## 핵심 개념 정리

| 개념 | 설명 |
|---|---|
| **BEV (Bird's-Eye-View)** | 위에서 내려다보는 시점의 표현. 자율주행 플래너가 소비하는 좌표계 |
| **Lift** | 각 픽셀을 깊이 분포를 이용해 3D frustum 포인트 클라우드로 변환 |
| **Splat** | 여러 카메라의 frustum을 BEV 래스터 그리드(Pillar)에 통합 |
| **Shoot** | BEV cost map에 template trajectory를 쏘아 최적 경로 선택 |
| **Latent Depth Distribution** | 깊이를 단일 값이 아닌 카테고리 확률 분포로 예측. 깊이 불확실성을 end-to-end로 학습 |
| **Pillar Pooling** | BEV 격자의 각 셀(Pillar)에 3D 포인트 feature를 누적하는 작업 |
| **Cumulative Sum Trick** | 패딩 없이 sum pooling을 구현하는 효율적인 방법. 2배 속도 향상 |
| **Camera rig agnostic** | 임의의 카메라 수와 배치를 가진 카메라 리그에 적용 가능 |
| **Ego-frame isometry equivariance** | 자차(ego car)의 회전/이동에 무관하게 동일 객체가 동일하게 탐지되는 성질 |

---

## 결론 및 시사점

**결론**: Lift-Splat-Shoot은 임의의 카메라 리그로부터 LiDAR 없이 BEV 표현을 end-to-end로 학습하는 아키텍처입니다. 깊이 불확실성을 카테고리 분포로 모델링하는 핵심 아이디어 덕분에, 단순한 기준선보다 우수한 성능을 보이며 캘리브레이션 오류와 카메라 누락에도 강건합니다.

**실무적 시사점**:

1. **LiDAR 없는 자율주행 인식**: BEVFusion, BEVDet, BEVDepth 등 이후 BEV 인식 연구의 토대가 된 seminal 논문. 카메라만으로 BEV를 추론하는 패러다임을 확립했습니다.
2. **Camera rig 확장성**: 카메라 수와 위치가 바뀌어도 재학습 없이 zero-shot 전이가 가능합니다. 실제 자율주행 차량 개발에서 카메라 배치 변경 시 유용합니다.
3. **End-to-end 모션 플래닝**: 인식과 플래닝을 통합 학습하여, 플래너의 피드백이 인식 모듈 학습에 반영됩니다.
4. **한계 및 향후 과제**: 야간이나 장거리에서 LiDAR 대비 성능 격차 존재. 단일 타임스텝 처리의 한계로, 다중 타임스텝의 temporal 정보 활용이 향후 과제로 남아 있습니다. (이 과제는 이후 BEVDet4D, BEVFormer 등이 해결합니다.)


---

*관련 논문: [BEVDepth](/posts/papers/bevdepth/), [BEVFormer](/posts/papers/BEVFormer/), [BEVFusion](/posts/papers/bevfusion-multi-task-multi-sensor-fusion/), [DETR3D](/posts/papers/detr3d-3d-object-detection-multi-view-images/), [nuScenes](/posts/papers/nuscenes-multimodal-dataset-autonomous-driving/)*
