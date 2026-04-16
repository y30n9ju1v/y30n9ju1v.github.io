---
title: "BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation"
date: 2026-04-17T08:30:00+09:00
draft: false
categories: ["Papers"]
tags: ["Autonomous Driving", "Sensor Fusion", "BEV", "LiDAR", "Camera", "3D Object Detection", "Segmentation"]
---

## 개요
- **저자**: Zhijian Liu, Haotian Tang, Alexander Amini, Xinyu Yang, Huizi Mao, Daniela L. Rus, Song Han
- **발행년도**: 2024 (arXiv 2022, CVPR 2023)
- **주요 내용**: 카메라와 LiDAR 센서를 Bird's-Eye View(BEV) 공간에서 통합하여 3D 객체 탐지와 BEV 맵 분할을 동시에 수행하는 멀티태스크 멀티센서 융합 프레임워크 제안

## 한계 극복

이 논문은 기존 멀티센서 융합 방식의 핵심 한계를 극복하기 위해 작성되었습니다.

- **기존 한계 1 — 포인트 레벨 융합의 의미론적 정보 손실**: LiDAR 포인트 클라우드에 카메라 특징을 투영하는 기존 방식은 LiDAR 포인트가 희소(sparse)하기 때문에 카메라의 풍부한 의미론적 밀도(semantic density)를 대부분 버립니다.
- **기존 한계 2 — 카메라-to-BEV 변환의 속도 병목**: 기존 Camera-to-BEV 변환(LSS 등)은 단일 프레임 처리에 500ms 이상 소요되어 실시간 추론이 불가능했습니다.
- **기존 한계 3 — 단일 태스크 최적화**: 기존 방법들은 탐지(geometric) 또는 분할(semantic) 중 하나에만 특화되어 있어 멀티태스크 구조를 지원하지 않습니다.
- **이 논문의 접근 방식**: 카메라와 LiDAR 특징을 모두 **공유 BEV 공간**으로 변환한 뒤 합산(concatenation)하여 기하학적·의미론적 정보를 동시에 보존하고, 효율적인 BEV pooling으로 속도 병목을 40× 해소합니다.

## 목차
- I. Introduction
- II. Related Work
- III. Method: Unified BEV Representation
  - A. Unified Representation
  - B. Efficient Camera-to-BEV Transformation
  - C. Fully-Convolutional Fusion
  - D. Multi-Task Heads
- IV. Experiments
  - A. 3D Object Detection
  - B. BEV Map Segmentation
  - C. Ablation Studies
- V. Analysis
- VI. Conclusion

---

## I. Introduction

**요약**

자율주행 차량은 다양한 센서를 통해 환경을 인식합니다. 카메라는 풍부한 의미론적 정보(색상, 텍스처, 차선 등)를 제공하고, LiDAR는 정밀한 3D 기하학 정보(거리, 형상)를 제공합니다. 기존 멀티센서 융합의 주류는 **포인트 레벨 융합**(point-level fusion)으로, LiDAR 포인트 위에 카메라 특징을 투영합니다. 그러나 LiDAR는 희소하기 때문에 카메라의 픽셀 대부분이 버려지고, 결국 카메라의 의미론적 밀도가 크게 손실됩니다.

BEVFusion은 이에 대한 해답으로 **공유 BEV 공간(shared BEV space)**을 제안합니다. 카메라와 LiDAR를 각각 BEV 특징 맵으로 변환한 뒤 합산하면, 두 센서의 장점을 동시에 보존할 수 있습니다. 또한 이 구조는 자연스럽게 멀티태스크(탐지 + 분할)로 확장됩니다.

**핵심 개념**
- **Bird's-Eye View (BEV)**: 위에서 아래를 내려다보는 시점의 2D 표현. 자율주행의 탐지·분할에 적합한 표준 공간
- **Sensor Fusion**: 여러 센서의 데이터를 합쳐 단일 센서의 한계를 보완하는 기술
- **Point-level Fusion**: LiDAR 포인트에 카메라 특징을 붙이는 방식. 희소성으로 인해 의미론적 정보가 손실됨

---

## II. Related Work

**요약**

관련 연구는 크게 세 흐름으로 나뉩니다.

1. **LiDAR 기반 3D 탐지**: PointPillars, SECOND, CenterPoint 등. 정밀하지만 카메라의 색상·텍스처 정보를 활용하지 못함
2. **카메라 기반 3D 탐지**: BEVDet, BEVDepth, DETR3D 등. 비용이 저렴하지만 깊이 추정의 불확실성이 큼
3. **멀티센서 융합**: PointPainting, MVP, TransFusion 등. 주로 포인트 레벨에서 융합하여 카메라의 의미론적 밀도를 충분히 활용하지 못함

**핵심 개념**
- **Semantic-oriented tasks**: BEV 맵 분할처럼 픽셀 단위 의미론적 이해가 필요한 작업. LiDAR만으로는 수행이 어려움
- **Geometric-oriented tasks**: 3D 객체 탐지처럼 정확한 위치·형상 추정이 필요한 작업. LiDAR가 핵심

---

## III. Method: Unified BEV Representation

**요약**

BEVFusion의 핵심 아이디어는 카메라와 LiDAR를 동일한 BEV 공간으로 변환한 뒤 합산하는 것입니다. 이 섹션에서는 (A) 통합 표현, (B) 효율적 Camera-to-BEV 변환, (C) 완전 합성곱 융합, (D) 멀티태스크 헤드를 설명합니다.

### A. Unified Representation

카메라 특징은 **원근 뷰(perspective view)**, LiDAR 특징은 **3D 뷰**에서 나옵니다. 두 표현 사이에는 근본적인 뷰 불일치(view discrepancy)가 있습니다. BEVFusion은 두 모달리티를 모두 BEV 공간으로 변환하여 이 불일치를 해소합니다.

- 카메라: 멀티뷰 RGB 이미지 → Camera Encoder → Camera-to-BEV 변환 → Camera BEV Features
- LiDAR: 포인트 클라우드 → LiDAR Encoder → LiDAR Flatten → LiDAR BEV Features
- 두 BEV 특징을 채널 축으로 이어붙여(concatenate) → Fused BEV Features → BEV Encoder → Task Heads

**핵심 개념**
- **View Discrepancy**: 카메라(원근, 픽셀 단위)와 LiDAR(3D, 포인트 단위)의 데이터 공간 차이. 공유 BEV로 해소
- **Shared BEV Space**: 두 센서의 기하학 정보와 의미론 정보를 모두 담는 통합 2D 표현

### B. Efficient Camera-to-BEV Transformation

카메라를 BEV로 변환하는 핵심 단계는 LSS(Lift-Splat-Shoot) 방식을 따릅니다. 각 카메라 픽셀의 깊이 분포를 예측하고, 3D 공간으로 올린 뒤 BEV 그리드에 투영합니다. 기존 구현은 단일 프레임에 500ms 이상 걸려 실시간 불가능했습니다.

BEVFusion은 두 가지 최적화로 **40× 속도 향상**을 달성합니다:

**1) Precomputation (사전 계산)**

카메라 내부 행렬과 외부 행렬(intrinsic/extrinsic)로부터 각 픽셀이 BEV 그리드의 어느 셀에 해당하는지를 미리 계산하여 DRAM에 저장합니다. 추론 시에는 이 인덱스 테이블만 조회합니다.

**2) Interval Reduction (구간 합산 커널)**

BEV pooling은 동일 BEV 셀에 속하는 모든 포인트의 특징을 집계(aggregate)합니다. 기존 prefix sum은 불필요한 중간 계산이 많습니다. BEVFusion은 GPU 스레드를 BEV 셀마다 할당하는 전용 커널을 구현하여, 트리 구조 없이 경계 값만 직접 합산합니다. 이로써 DRAM 접근을 최소화하고 집계 지연을 500ms → 2ms로 단축합니다.

$$
F_{\text{BEV}}(u, v) = \text{Aggregate}\left(\{f_i \mid (x_i, y_i) \in \text{cell}(u,v)\}\right)
$$

**수식 설명**
- **$F_{\text{BEV}}(u, v)$**: BEV 그리드의 $(u, v)$ 셀에 대응하는 BEV 특징 벡터
- **$f_i$**: 3D 공간으로 올려진 $i$번째 카메라 특징 포인트
- **$(x_i, y_i)$**: 해당 포인트의 BEV 평면 좌표
- **$\text{cell}(u,v)$**: BEV 그리드에서 $(u,v)$ 셀이 커버하는 공간 영역
- **$\text{Aggregate}$**: 해당 셀 안의 모든 특징을 하나로 합치는 연산 (합산, 평균 등)
- **직관**: 카메라에서 본 여러 픽셀이 3D 공간에서 같은 위치에 모이면, 그것들을 하나의 BEV 셀 특징으로 압축하는 과정

**핵심 개념**
- **LSS (Lift-Splat-Shoot)**: 카메라 이미지의 각 픽셀에 깊이 분포를 부여하고 3D로 올린 뒤 BEV에 투영하는 방법
- **BEV Pooling**: BEV 셀별로 포인트 특징을 집계하는 연산. 기존에는 느렸으나 BEVFusion이 GPU 최적화로 40× 가속
- **Precomputation**: 카메라 파라미터가 고정이므로 픽셀-셀 매핑을 미리 계산해 캐시하는 전략

### C. Fully-Convolutional Fusion

카메라 BEV 특징과 LiDAR BEV 특징을 채널 축으로 이어붙인 뒤, **완전 합성곱 BEV 인코더(fully-convolutional BEV encoder)**로 처리합니다. 이 인코더는 잔여 오정렬(spatial misalignment)을 보정합니다. Camera-to-BEV 변환에서 깊이 추정이 부정확할 수 있어 카메라 BEV 특징이 LiDAR BEV 특징과 살짝 어긋날 수 있는데, 합성곱이 수용 영역(receptive field)으로 이를 흡수합니다.

**핵심 개념**
- **Spatial Misalignment**: 깊이 추정 오차로 카메라 BEV 특징이 LiDAR BEV 특징과 공간적으로 어긋나는 현상
- **Receptive Field**: 합성곱 레이어가 한 번에 볼 수 있는 입력 영역의 크기. 넓을수록 오정렬 보정 가능

### D. Multi-Task Heads

융합된 BEV 특징 맵 위에 태스크별 헤드를 붙입니다.

- **3D Object Detection**: CenterPoint 스타일. 클래스별 히트맵으로 중심점 예측 + 회귀 헤드로 크기·회전·속도 예측
- **BEV Map Segmentation**: CVT 스타일. 클래스별 이진 분할(drivable area, 차선, 횡단보도 등). 겹치는 클래스 허용

**핵심 개념**
- **Heatmap Head**: 각 BEV 셀이 객체 중심일 확률을 예측하는 맵. 3D 탐지의 핵심 출력
- **Binary Segmentation**: 각 BEV 셀이 특정 클래스(예: 주행 가능 영역)에 속하는지 0/1로 분류

---

## IV. Experiments

**요약**

nuScenes와 Waymo 데이터셋에서 3D 탐지와 BEV 분할 성능을 검증합니다.

### A. 3D Object Detection (nuScenes)

| 방법 | 모달리티 | mAP (test) | NDS (test) | MACs (G) | Latency (ms) |
|---|---|---|---|---|---|
| PointPainting | C+L | 65.8 | 69.6 | 370.0 | 185.8 |
| TransFusion | C+L | 67.5 | 71.3 | 485.8 | 156.6 |
| **BEVFusion (Ours)** | **C+L** | **70.2** | **72.9** | **253.2** | **119.2** |

BEVFusion은 기존 최고 방법 대비 **mAP +1.3%, NDS +1.6%** 향상을 달성하면서 연산량은 **1.9×**, 지연은 **1.3×** 감소시킵니다.

### B. BEV Map Segmentation (nuScenes)

| 방법 | 모달리티 | Mean IoU |
|---|---|---|
| BEVFusion (C only) | C | **56.6** |
| PointPillars | L | 43.8 |
| BEVFusion (Ours) | C+L | **62.7** |

카메라 전용 모델에서도 BEVFusion은 기존 최고 대비 **13.6% 향상**. C+L 융합 시 추가 향상.

### C. Ablation Studies

- **카메라 BEV 풀링 최적화**: 지연 500ms → 12ms (40×)
- **BEV 인코더**: 공간 오정렬 보정에 핵심. 제거 시 성능 급락
- **모달리티별 기여**: 카메라는 의미론, LiDAR는 기하학 기여. 둘 다 필요

---

## V. Analysis

**요약**

추가 분석에서 BEVFusion의 강건성과 우월성을 확인합니다.

### 날씨·조명 강건성

- **악천후(비)**: 단일 모달리티 LiDAR 대비 mAP +7.1%, IoU +13.3 향상
- **야간**: BEVFusion C+L 융합 시 LiDAR 전용 대비 mAP +8.1%, IoU +6.6 향상
- 카메라가 야간·악천후에서 취약한 것을 LiDAR가 보완하고, LiDAR의 의미론 부족을 카메라가 보완하는 상호 보완 효과

### 객체 크기·거리별 성능

- **작은 객체**: BEVFusion이 LiDAR 전용 및 MVP 대비 크게 우세 (카메라의 의미론 덕분)
- **큰 객체**: 모든 방법이 비슷하나 BEVFusion이 여전히 최고
- **먼 거리**: BEVFusion이 기존 방법보다 일관되게 우수

### Sparse LiDAR 강건성

LiDAR 빔 수를 32빔 → 1빔으로 줄였을 때, BEVFusion은 LiDAR 전용 대비 훨씬 덜 떨어집니다. 카메라 정보가 희소 LiDAR의 한계를 보완하기 때문입니다.

**핵심 개념**
- **LiDAR Beam**: LiDAR 스캐너의 레이저 빔 수. 많을수록 밀도 높은 포인트 클라우드 획득. 비용이 비쌈
- **Modality Complementarity**: 두 센서가 서로의 약점을 보완하는 성질. BEVFusion의 핵심 강점

---

## 핵심 개념 정리

| 개념 | 설명 |
|---|---|
| **BEV (Bird's-Eye View)** | 위에서 내려다본 2D 평면. 자율주행 인식의 표준 출력 공간 |
| **Camera-to-BEV (LSS)** | 카메라 픽셀에 깊이 분포를 부여해 3D→BEV로 변환하는 방법 |
| **BEV Pooling** | BEV 셀마다 해당 3D 포인트 특징을 집계하는 연산 |
| **Precomputation** | 카메라 파라미터 고정 시 픽셀-셀 매핑을 미리 계산해 추론 가속 |
| **Interval Reduction** | 구간 경계 값만 GPU에서 직접 합산하여 DRAM 접근을 최소화하는 커널 |
| **Fully-Convolutional Fusion** | 합성곱으로 두 모달리티의 공간 오정렬을 흡수하는 융합 방법 |
| **Multi-Task Learning** | 하나의 모델로 탐지·분할 등 여러 태스크를 동시에 수행 |
| **Modality Complementarity** | 카메라(의미론)와 LiDAR(기하학)가 서로의 약점을 보완 |

---

## 결론 및 시사점

BEVFusion은 카메라와 LiDAR를 **공유 BEV 공간**에서 통합하는 간결하면서도 강력한 프레임워크입니다.

**주요 기여**
1. **통합 BEV 표현**: 기하학적(LiDAR)·의미론적(카메라) 정보를 동시에 보존
2. **40× Camera-to-BEV 가속**: Precomputation + Interval Reduction으로 실시간 추론 가능
3. **멀티태스크 지원**: 단일 모델로 3D 탐지 + BEV 분할 동시 수행
4. **SOTA 달성**: nuScenes 3D 탐지 mAP 70.2, Waymo 1위, BEV 분할 62.7 mIoU

**실무적 시사점**
- 카메라 전용 모델도 BEV 공간으로 올리면 LiDAR와 자연스럽게 융합 가능
- Camera-to-BEV의 속도 병목은 전처리 캐싱과 GPU 커널 최적화로 해결 가능
- 악천후·야간 등 단일 센서가 취약한 환경에서 멀티센서 융합의 강건성이 두드러짐
- BEV 공간은 탐지, 분할, 추적, 모션 예측 등 다양한 태스크로 쉽게 확장 가능
