---
title: "CenterPoint: Center-based 3D Object Detection and Tracking"
date: 2026-04-19T22:00:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving", "3D Object Detection"]
tags: ["LiDAR", "3D Detection", "Object Tracking", "Point Cloud", "BEV"]
---

## 개요

- **저자**: Tianwei Yin, Xingyi Zhou, Philipp Krähenbühl (UT Austin)
- **발행년도**: 2021 (arXiv 2020, CVPR 2021)
- **논문**: [arXiv:2006.11275](https://arxiv.org/abs/2006.11275)
- **코드**: [github.com/tianweiy/CenterPoint](https://github.com/tianweiy/CenterPoint)
- **주요 내용**: LiDAR 포인트 클라우드에서 3D 객체를 **중심점(center point)**으로 표현·탐지·추적하는 프레임워크. Waymo와 nuScenes 두 벤치마크에서 당시 SOTA 달성

---

## 한계 극복

- **기존 한계 1 — Anchor 기반 표현의 방향 문제**: 기존 3D 탐지기는 axis-aligned 바운딩 박스(anchor)를 사용. 차량이 회전하거나 자전거·보행자처럼 세로로 긴 객체를 탐지할 때 anchor가 맞지 않아 성능 저하
- **기존 한계 2 — IoU 기반 anchor 할당의 복잡성**: 방향별로 anchor 수가 늘어나고 IoU threshold 튜닝이 필요해 복잡도 증가, false positive 증가
- **기존 한계 3 — 추적의 복잡성**: 기존 3D 추적은 Kalman filter + Mahalanobis 거리 등 복잡한 연산 필요
- **이 논문의 접근 방식**: 객체를 3D 바운딩 박스 대신 **중심점(heatmap peak)**으로 표현 → 방향에 무관한 탐지, 중심점 velocity 예측으로 추적을 greedy closest-point matching으로 단순화

---

## 목차

1. Introduction
2. Related Work
3. Preliminaries (2D CenterNet, 3D Detection 정의)
4. CenterPoint
   - 4.1 Two-Stage CenterPoint
   - 4.2 Architecture
5. Experiments
   - 5.1 Main Results
   - 5.2 Ablation Studies
6. Conclusion

---

## 3. Preliminaries

### 2D CenterNet

CenterPoint의 직접적 전신인 2D CenterNet은 이미지 기반 객체 탐지를 **키포인트 추정(keypoint estimation)** 문제로 재정의합니다.

- 입력 이미지에서 $K$개 클래스별 **heatmap** $\hat{Y} \in [0,1]^{w \times h \times K}$ 예측
- heatmap의 각 **local maximum(peak)**이 탐지된 객체의 중심
- 각 탐지 객체에 대해 크기, 오프셋 등 속성을 중심 위치에서 회귀

### 3D Detection 문제 정의

포인트 클라우드 $\mathcal{P} = \{(x, y, z, r)_i\}$에서 3D 바운딩 박스 집합 $\mathcal{B} = \{b_k\}$를 예측합니다.

각 박스 $b = (u, v, d, w, l, h, \alpha)$:
- **$(u, v)$**: bird's-eye view(BEV) 상의 중심 위치
- **$d$**: 지면으로부터의 높이
- **$(w, l, h)$**: 가로·세로·높이 크기
- **$\alpha$**: yaw 회전각

현대 3D 탐지기는 VoxelNet 또는 PointPillars 같은 **3D 인코더**로 포인트 클라우드를 voxel/pillar로 양자화하여 map-view 피처맵 $\mathbf{M} \in \mathbb{R}^{W \times L \times F}$을 생성합니다.

---

## 4. CenterPoint

### 전체 파이프라인

```
입력: LiDAR 포인트 클라우드
         │
         ▼
[3D Backbone] — VoxelNet 또는 PointPillars
         │  map-view 피처맵 M ∈ R^{W×L×F}
         ▼
[Detection Head (2D CNN)]
  ├─ Center Heatmap Head → K채널 heatmap (클래스별 중심 확률)
  ├─ Sub-voxel Offset Head → 양자화 오차 보정
  ├─ Height-above-ground Head → 3D 높이
  ├─ 3D Size Head → (w, l, h)
  ├─ Rotation Head → (sin α, cos α)
  └─ Velocity Head → (vx, vy) [추적용]
         │
         ▼ (Two-Stage: 선택적)
[Point Feature Extraction at Box Faces]
         │
         ▼
[MLP] → IoU-guided confidence score + box refinement
         │
         ▼
출력: 3D 바운딩 박스 + 신뢰도 점수 + 속도 벡터
```

### Center Heatmap Head

목표: 탐지된 객체의 중심 위치에 **heatmap peak** 생성.

훈련 시, 3D 바운딩 박스 중심점을 BEV에 투영하고 **2D Gaussian 커널**로 렌더링하여 GT heatmap을 만듭니다.

Gaussian 반경:

$$\sigma = \max(f(w \cdot l),\ \tau), \quad \tau = 2$$

- **$w, l$**: 객체의 가로·세로 크기 (BEV)
- **$f(\cdot)$**: CornerNet의 반경 함수 — 박스 크기에 비례해 Gaussian을 넓게 설정
- **$\tau = 2$**: 최소 Gaussian 반경 (작은 객체 보호)
- **직관**: 큰 차량일수록 넓은 supervision 영역 → 학습 안정성 향상

Focal loss로 최적화합니다 (heatmap이 매우 희소하므로 배경 억제 중요).

### Regression Heads

각 탐지 객체에 대해 중심점 위치에서 다음 속성을 회귀합니다:

| 출력 | 차원 | 설명 |
|---|---|---|
| sub-voxel offset $o$ | $\mathbb{R}^2$ | voxel 양자화 오차 보정 |
| height-above-ground $h_g$ | $\mathbb{R}$ | 지면 기준 높이 (depth 복원) |
| 3D size $s$ | $\mathbb{R}^3$ | $(w, l, h)$, log-scale 회귀 |
| rotation | $\mathbb{R}^2$ | $(\sin\alpha, \cos\alpha)$ |
| velocity $\mathbf{v}$ | $\mathbb{R}^2$ | $(v_x, v_y)$, 추적에 사용 |

**rotation을 $\sin/\cos$로 표현하는 이유**: 각도는 $0°$와 $360°$가 같지만 수치상 멀어서 회귀가 불안정. $(\sin\alpha, \cos\alpha)$ 쌍은 연속적이고 순환 문제가 없음.

모든 회귀 출력은 **L1 loss**로 학습합니다.

### Velocity Head와 추적

$$\mathbf{v} \in \mathbb{R}^2$$: 현재 프레임과 직전 프레임 사이의 중심점 이동량.

추적 알고리즘 (greedy closest-point matching):
1. 현재 프레임 탐지 결과에서 velocity $\mathbf{v}$로 중심을 이전 프레임으로 역투영
2. 이전 프레임 tracklet과 **최근접점 거리**로 매칭
3. 매칭 실패 tracklet은 최대 $T=3$ 프레임 유지 (이후 삭제)
4. 추적 시간 $T_{track} = 1\text{ms}$ — Kalman filter(73ms) 대비 73배 빠름

**직관**: 객체가 점이면 추적도 점 간 거리 매칭으로 충분. 박스 IoU 매칭보다 회전·크기에 무관하게 robust함.

### 4.1 Two-Stage CenterPoint

1단계(one-stage)에서 중심점 위치만으로 속성을 추론하면, 센서가 객체 측면만 보는 경우(자율주행 흔한 상황) 중심점 피처가 불충분할 수 있습니다.

**2단계 보정**:
- 예측된 3D 바운딩 박스의 **4개 outward-facing 면 중심점** + **객체 중심점** = 총 5점
- 각 점에서 backbone map-view 피처맵 $\mathbf{M}$을 bilinear interpolation으로 피처 추출
- 5개 포인트 피처를 concat → MLP → IoU-guided confidence score + box refinement

**IoU-guided confidence score**:

$$I = \min(1,\ \max(0,\ 2 \times IoU_t - 0.5))$$

$$L_{score} = -I_t \log(\hat{I}_t) - (1-I_t)\log(1-\hat{I}_t)$$

**수식 설명**:
- **$IoU_t$**: t번째 제안 박스와 GT 간의 3D IoU
- **$I_t$**: IoU를 $[0,1]$ 범위로 정규화한 confidence target
  - IoU = 0.5 → $I$ = 0 (완전 잘못된 박스)
  - IoU = 1.0 → $I$ = 1 (완벽한 박스)
- **$\hat{I}_t$**: 예측 confidence
- **직관**: NMS 없이도 confidence score가 박스 품질을 직접 반영하게 학습

최종 confidence:

$$\hat{Q}_t = \sqrt{\hat{Y}_t \cdot \hat{I}_t}$$

- 1단계 heatmap 점수 $\hat{Y}_t$와 2단계 IoU 점수 $\hat{I}_t$의 기하평균

---

## 5. Experiments

### 5.1 주요 결과

**Waymo Open Dataset — 3D Detection (test set, Level 2)**

| 방법 | Vehicle mAP | Vehicle mAPH | Ped. mAP | Ped. mAPH |
|---|---|---|---|---|
| PointPillars | 55.6 | 55.1 | 45.1 | — |
| PV-RCNN | 65.1 | 64.7 | — | — |
| **CenterPoint-Voxel (ours)** | **72.2** | **71.8** | **72.2** | **66.4** |

**nuScenes — 3D Detection (test set)**

| 방법 | mAP | NDS | PKL |
|---|---|---|---|
| PointPillars | 40.1 | 55.0 | 1.00 |
| CBGS (이전 1위) | 52.8 | 63.3 | 0.77 |
| **CenterPoint (ours)** | **58.0** | **65.5** | **0.69** |

**nuScenes — 3D Tracking (test set)**

| 방법 | AMOTA |
|---|---|
| AB3D | 15.1 |
| Chiu et al. | 55.0 |
| **CenterPoint (ours)** | **63.8** |

### 5.2 Ablation 결과

**Center-based vs Anchor-based (Waymo validation, Level 2 mAPH)**

| Encoder | 방법 | Vehicle | Pedestrian | 평균 |
|---|---|---|---|---|
| VoxelNet | Anchor-based | 66.1 | 54.4 | 60.3 |
| VoxelNet | **Center-based** | **66.5** | **62.7** | **64.6** |
| PointPillars | Anchor-based | 64.1 | 50.8 | 57.5 |
| PointPillars | **Center-based** | **66.5** | **57.4** | **62.0** |

→ 단순히 anchor → center로 표현만 바꿔도 **3-4 mAPH 향상**

**회전각별 성능 (Waymo validation, Level 2 mAPH)**

| 방법 | Vehicle 0°-15° | 15°-30° | 30°-45° | Ped 0°-15° | 15°-30° | 30°-45° |
|---|---|---|---|---|---|---|
| Anchor-based | 67.1 | **47.7** | 45.4 | 55.9 | 32.0 | 26.5 |
| Center-based | **67.8** | 46.4 | **51.6** | **64.0** | **42.1** | **35.7** |

→ 특히 **회전된 객체(30°-45°)**에서 center-based가 크게 우세 — anchor의 방향 의존성 한계 입증

**2단계 정제 효과 (Waymo validation)**

| Stage | Vehicle mAPH | Ped. mAPH | $T_{proposal}$ | $T_{refine}$ |
|---|---|---|---|---|
| First Stage | 66.5 | 62.7 | 71ms | — |
| + Box Center | 68.0 | 64.9 | 71ms | 5ms |
| + Surface Center | **68.3** | **65.4** | 71ms | 6ms |

→ 2단계 추가 비용 5-6ms로 **1.8 mAPH 향상**

---

## 6. 핵심 개념 정리

| 개념 | 설명 |
|---|---|
| **Center-based 표현** | 3D 객체를 바운딩 박스 대신 중심점으로 표현. 방향 불변성 확보, anchor 수·IoU threshold 튜닝 불필요 |
| **Heatmap** | K채널 BEV 확률맵. 각 픽셀 값이 해당 위치에 객체 중심이 있을 확률. Local maximum이 탐지 결과 |
| **Gaussian 렌더링** | GT 중심점 주변에 박스 크기 비례 Gaussian 분포로 supervision 영역 확장. 작은 객체도 안정적 학습 |
| **Velocity Head** | 두 연속 프레임의 중심 이동량을 직접 회귀. Kalman filter 없이 추적 가능하게 함 |
| **Greedy Closest-Point Matching** | velocity로 역투영한 중심점과 이전 tracklet 중심점 간 최소 거리로 추적. 1ms의 극단적 속도 |
| **Two-Stage Refinement** | 박스 면 중심점 피처로 confidence 재추정 + box 보정. 추가 비용 5-8ms로 성능 향상 |
| **IoU-guided Score** | 2단계 confidence를 박스-GT IoU로 직접 지도. NMS 없이도 품질 반영 가능 |
| **VoxelNet / PointPillars** | CenterPoint가 지원하는 3D 인코더. CenterPoint는 출력 표현(head)만 바꾸므로 어떤 인코더와도 호환 |

---

## 7. 결론 및 시사점

**CenterPoint의 의의**:
- 3D 탐지를 **keypoint estimation으로 재정의** — 방향 불변성, 단순한 파이프라인
- anchor → center 전환만으로 3-4 mAPH 향상, 회전 객체에서 특히 강함
- 추적을 1ms greedy matching으로 단순화하면서 SOTA 추적 성능 달성
- VoxelNet·PointPillars 어느 backbone과도 호환 — plug-in head로 설계

**자율주행 로드맵 내 위치**:
- **BEVFusion의 LiDAR 브랜치 선행 연구**: BEVFusion은 카메라 BEV + LiDAR BEV를 융합하는데, LiDAR BEV 처리에 CenterPoint 구조를 직접 차용
- **UniAD·VAD의 인식 upstream**: E2E 계획 논문들이 탐지 결과를 downstream으로 받을 때 CenterPoint를 perception 기준으로 사용
- LiDAR 기반 3D 탐지의 사실상 표준 baseline으로, 이후 대부분의 비교 논문에서 참조됨

**한계**:
- LiDAR 전용 — 카메라 전용 또는 카메라+LiDAR 융합 탐지에는 직접 적용 불가
- PointPillars 백본 사용 시 보행자처럼 1픽셀 크기의 객체에서 2단계 정제 효과 제한적 (양자화 한계)
- velocity 예측이 비선형 기동(급회전 등)에서는 부정확할 수 있음


---

*관련 논문: [PointNet](/posts/papers/pointnet-deep-learning-on-point-sets-for-3d-classification-and-segmentation/), [PointPillars](/posts/papers/pointpillars-fast-encoders-object-detection-point-clouds/), [BEVFusion](/posts/papers/bevfusion-multi-task-multi-sensor-fusion/), [nuScenes](/posts/papers/nuscenes-multimodal-dataset-autonomous-driving/), [Waymo Open Dataset](/posts/papers/waymo-open-dataset/)*
