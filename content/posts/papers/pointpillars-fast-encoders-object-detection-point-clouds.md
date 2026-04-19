---
title: "PointPillars: Fast Encoders for Object Detection from Point Clouds"
date: 2026-04-19T22:30:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving", "3D Object Detection"]
tags: ["LiDAR", "Point Cloud", "3D Detection", "BEV", "Real-time"]
---

## 개요

- **저자**: Alex H. Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, Oscar Beijbom (nuTonomy / APTIV)
- **발행년도**: 2019 (arXiv 2018, CVPR 2019)
- **논문**: [arXiv:1812.05784](https://arxiv.org/abs/1812.05784)
- **코드**: [github.com/nutonomy/second.pytorch](https://github.com/nutonomy/second.pytorch)
- **주요 내용**: LiDAR 포인트 클라우드를 **수직 기둥(pillar)**으로 조직화하고 PointNet으로 피처를 학습하여 **2D CNN만으로** 3D 객체 탐지를 수행하는 인코더. 62Hz의 실시간 처리 속도로 KITTI 벤치마크 SOTA 달성

---

## 한계 극복

- **기존 한계 1 — 3D 컨볼루션의 속도 문제**: VoxelNet은 3D voxel별 PointNet + 3D Conv를 사용하여 정확하지만 4.4Hz로 실시간 불가. SECOND가 희소 3D Conv로 20Hz까지 개선했지만 여전히 느림
- **기존 한계 2 — 고정 인코더의 표현력 한계**: PIXOR, MV3D, Complex-YOLO 등은 수작업(hand-crafted) 피처 인코더를 사용. 새로운 포인트 클라우드 설정에 일반화 어려움
- **기존 한계 3 — Z축 binning 파라미터 튜닝**: voxel 방식은 수직 방향 bin 크기를 수동 설정해야 함
- **이 논문의 접근 방식**: 포인트 클라우드를 **pillar(수직 기둥)**로 분할 → PointNet으로 학습된 피처 인코딩 → 2D pseudo-image로 scatter → 표준 2D CNN + SSD head. 3D Conv 완전 제거로 속도 62Hz 달성

---

## 목차

1. Introduction
2. PointPillars Network
   - 2.1 Pointcloud to Pseudo-Image (Pillar Feature Net)
   - 2.2 Backbone (2D CNN)
   - 2.3 Detection Head (SSD)
3. Implementation Details
4. Experimental Setup
5. Results
6. Realtime Inference
7. Ablation Studies

---

## 2. PointPillars Network

### 전체 파이프라인

```
입력: LiDAR 포인트 클라우드
         │
         ▼
[Pillar Feature Net]
  포인트 클라우드 → Stacked Pillars 텐서 (D, P, N)
  → 간소화된 PointNet (Linear + BN + ReLU + MaxPool)
  → Learned Features (C, P)
  → 2D pseudo-image (C, H, W)로 scatter
         │
         ▼
[Backbone (2D CNN)]
  Top-down 다중 스케일 피처 추출
  → Upsampling + Concatenation
         │
         ▼
[Detection Head (SSD)]
  Anchor 기반 3D 바운딩 박스 회귀 + 분류
         │
         ▼
출력: 3D 회전 바운딩 박스 (차량, 보행자, 자전거)
```

### 2.1 Pointcloud to Pseudo-Image

**Step 1 — Pillar 생성**

포인트 클라우드를 x-y 평면에 고정 간격(기본 0.16m × 0.16m) 그리드로 분할합니다. 각 그리드 셀이 하나의 **pillar**입니다.

- z축 binning **불필요** — pillar는 바닥부터 하늘까지 수직으로 무한 확장
- 빈 pillar는 대부분(~97%)으로, 비어 있지 않은 pillar만 처리

**Step 2 — 포인트 장식(decoration)**

각 pillar 안의 포인트 $l$에 9차원 피처 벡터를 부여합니다:

$$l = (x,\ y,\ z,\ r,\ x_c,\ y_c,\ z_c,\ x_p,\ y_p)$$

**변수 설명**:
- **$(x, y, z)$**: 포인트의 3D 좌표
- **$r$**: 반사율(reflectance)
- **$(x_c, y_c, z_c)$**: 포인트와 pillar 내 모든 포인트의 **산술 평균** 간의 거리 — 포인트가 pillar 중심에서 얼마나 떨어져 있는지
- **$(x_p, y_p)$**: 포인트와 pillar의 **x, y 중심** 간의 오프셋

**직관**: 원본 좌표만 쓰면 pillar 내에서 포인트의 상대 위치를 알 수 없습니다. $x_c, y_c, z_c$는 포인트가 객체 표면의 어느 부분에 있는지(중심부 vs 가장자리), $x_p, y_p$는 pillar 격자 내에서 정확한 위치를 제공합니다.

**텐서 구성**: 샘플당 최대 $P$개 비어 있지 않은 pillar, pillar당 최대 $N$개 포인트 → $(D, P, N) = (9, 12000, 100)$ 크기 텐서. 넘치면 랜덤 샘플링, 모자라면 zero padding.

**Step 3 — PointNet으로 피처 학습**

간소화된 PointNet (단일 Linear layer):

$$\text{Linear}(D \to C) \to \text{BatchNorm} \to \text{ReLU} \to \text{MaxPool over } N$$

출력: $(C, P)$ 크기 피처 텐서 (pillar당 하나의 $C$차원 벡터)

**Step 4 — Pseudo-image로 scatter**

각 pillar의 피처 벡터를 원래 x-y 위치에 배치 → $(C, H, W)$ 크기의 **2D pseudo-image** 생성

이후 모든 연산은 표준 2D CNN — GPU 효율 극대화.

### 2.2 Backbone (2D CNN)

FPN(Feature Pyramid Network) 스타일의 top-down + upsampling 구조:

```
Pseudo-image (C, H, W)
    │
    ├─ Block1(S=2, L, 4C) — stride 2로 다운샘플 + L개 3×3 Conv
    ├─ Block2(S=4, L, 2C) — stride 4
    └─ Block3(S=8, L, 2C) — stride 8
         │
    Up1(S→1, 2C), Up2(S→1, 2C), Up3(S→1, 2C)  — transposed conv로 업샘플
         │
    Concat → 6C 채널 피처맵
```

- **Block(S, L, F)**: 스트라이드 $S$, $L$개 레이어, $F$ 출력 채널
- 서로 다른 스케일의 피처를 업샘플하여 합침 → 다중 스케일 맥락 통합

### 2.3 Detection Head (SSD)

Single Shot Detector(SSD) 방식으로 anchor 기반 3D 박스를 예측합니다.

**Anchor 설계**:
- 차량: 1.6×3.9×1.5m, z=-1m, 0°/90° 두 방향
- 보행자: 0.6×0.8×1.73m, z=-0.6m
- 자전거: 0.6×1.76×1.73m, z=-0.6m

**Localization 회귀 타겟**:

$$\Delta x = \frac{x^{gt} - x^a}{d^a}, \quad \Delta y = \frac{y^{gt} - y^a}{d^a}, \quad \Delta z = \frac{z^{gt} - z^a}{h^a}$$

$$\Delta w = \log\frac{w^{gt}}{w^a}, \quad \Delta l = \log\frac{l^{gt}}{l^a}, \quad \Delta h = \log\frac{h^{gt}}{h^a}$$

$$\Delta\theta = \sin(\theta^{gt} - \theta^a)$$

**수식 설명**:
- **위치 $(x, y, z)$**: GT와 anchor 간 차이를 대각선 크기 $d^a = \sqrt{(w^a)^2 + (l^a)^2}$로 정규화 → 스케일 불변
- **크기 $(w, l, h)$**: log 스케일로 회귀 → 다양한 크기 객체를 균일하게 학습
- **각도 $\theta$**: $\sin$ 변환으로 $[-\pi, \pi]$ 범위를 연속적으로 표현
- **$\Delta\theta$ 한계**: $\sin$은 $0°$와 $180°$를 구분 못함 → 별도 방향 분류 손실 $\mathcal{L}_{dir}$ 추가

**총 손실**:

$$\mathcal{L} = \frac{1}{N_{pos}} \left(\beta_{loc}\mathcal{L}_{loc} + \beta_{cls}\mathcal{L}_{cls} + \beta_{dir}\mathcal{L}_{dir}\right)$$

$$\mathcal{L}_{cls} = -\alpha_a (1-p^a)^\gamma \log p^a \quad \text{(Focal Loss)}$$

**수식 설명**:
- **$N_{pos}$**: positive anchor 수 — 배치당 positive 비율 차이를 정규화
- **$\mathcal{L}_{loc}$**: SmoothL1 로컬라이제이션 손실
- **$\mathcal{L}_{cls}$**: Focal Loss — 쉬운 배경 anchor의 기여를 $(1-p^a)^\gamma$로 억제 ($\alpha=0.25, \gamma=2$)
- **$\mathcal{L}_{dir}$**: softmax 기반 방향 분류 손실 ($\beta_{loc}=2, \beta_{cls}=1, \beta_{dir}=0.2$)

---

## 5. 주요 결과

### KITTI test BEV Detection benchmark

| 방법 | 입력 | Speed(Hz) | mAP | Car Mod. | Ped Mod. | Cyclist Mod. |
|---|---|---|---|---|---|---|
| VoxelNet | Lidar | 4.4 | 58.25 | 79.26 | 40.74 | 54.76 |
| SECOND | Lidar | 20 | 60.56 | 79.37 | 46.27 | 56.04 |
| **PointPillars** | **Lidar** | **62** | **66.19** | **86.10** | **50.23** | **62.25** |

### KITTI test 3D Detection benchmark

| 방법 | 입력 | Speed(Hz) | mAP | Car Mod. | Ped Mod. | Cyclist Mod. |
|---|---|---|---|---|---|---|
| VoxelNet | Lidar | 4.4 | 49.05 | 65.11 | 33.69 | 48.36 |
| SECOND | Lidar | 20 | 56.69 | 76.48 | 46.27 | 56.04 |
| **PointPillars** | **Lidar** | **62** | **59.20** | **74.99** | **52.08** | **75.78** |

→ **LiDAR only** 방법 중 속도와 정확도 모두 1위. 일부 fusion(LiDAR+카메라) 방법도 능가

### 인코더 유형별 비교 (KITTI val BEV mAP)

| 인코더 | 유형 | $0.16^2$ | $0.20^2$ | $0.24^2$ | $0.28^2$ |
|---|---|---|---|---|---|
| MV3D | Fixed | 72.8 | 71.0 | 70.8 | 67.6 |
| PIXOR | Fixed | 71.8 | 71.3 | 69.9 | 65.6 |
| VoxelNet | Learned | **74.4** | **74.0** | 72.9 | 71.9 |
| **PointPillars** | **Learned** | 73.7 | 72.6 | **72.9** | **72.0** |

→ 동일 속도 조건에서 PointPillars가 VoxelNet보다 더 나은 operating point 제공

---

## 6. Realtime Inference 분석

전체 파이프라인 단계별 소요 시간 (Intel i7 CPU + NVIDIA 1080ti):

| 단계 | 시간 |
|---|---|
| 포인트 클라우드 로드 & 필터링 | 1.4ms |
| Pillar 생성 & 장식 | 2.7ms |
| GPU 업로드 + 인코딩 | 2.9 + 1.3ms |
| Pseudo-image scatter | 0.1ms |
| Backbone + Detection Head | 7.7ms |
| NMS (CPU) | 0.1ms |
| **총합** | **~16.2ms (62Hz)** |

**속도의 핵심 — PointPillar 인코딩이 1.3ms**: VoxelNet 인코더(190ms) 대비 **146배 빠름**. 3D Conv 완전 제거가 핵심.

TensorRT 적용 시: 45.5% 추가 가속 → **105Hz** 달성 가능

---

## 7. Ablation Studies 요약

**7.1 공간 해상도**: 작은 pillar(0.12m) → 정밀하지만 느림. 큰 pillar(0.28m) → 빠르지만 작은 객체 성능 저하. 0.16m이 최적 균형점

**7.2 박스별 데이터 증강**: VoxelNet·SECOND 권고와 달리 PointPillars에서는 최소한의 증강이 더 효과적 — GT 샘플링이 증강 필요성을 대체

**7.3 포인트 장식**: $x_p, y_p$ 오프셋 추가로 mAP **+0.5** 향상 — pillar 내 정확한 위치 정보의 효과

**7.4 학습된 인코더 vs 고정 인코더**: 해상도가 클수록 학습 인코더의 우위가 뚜렷 — 표현력 차이가 희소성 증가시 더 중요

---

## 8. 핵심 개념 정리

| 개념 | 설명 |
|---|---|
| **Pillar** | x-y 평면에서 고정 크기 격자로 분할된 수직 기둥. z축 binning 불필요, 비어 있지 않은 pillar만 처리 |
| **Point Decoration** | 좌표 외 pillar 평균 대비 거리($x_c,y_c,z_c$)와 pillar 중심 오프셋($x_p,y_p$)을 추가해 상대 위치 정보 제공 |
| **Pseudo-image** | Pillar 피처를 2D 격자에 scatter한 텐서. 이후 표준 2D CNN 적용 가능 |
| **Simplified PointNet** | 단일 Linear+BN+ReLU+MaxPool. VoxelNet의 2개 순차 PointNet 대비 2.5ms 절감 |
| **Anchor 매칭** | 2D BEV IoU로 positive/negative 결정. 높이·고도는 회귀 타겟으로만 사용 |
| **방향 모호성 해결** | $\sin(\theta^{gt}-\theta^a)$ 회귀 + softmax 방향 분류 $\mathcal{L}_{dir}$ 조합 |
| **GT 샘플링** | SECOND에서 제안한 기법. 다른 샘플의 GT 박스를 현재 포인트 클라우드에 붙여넣어 학습 다양성 증가 |

---

## 9. 결론 및 시사점

**PointPillars의 의의**:
- 3D Conv를 완전히 제거하고 **2D CNN만으로** LiDAR 3D 탐지 SOTA 달성
- 학습된 인코더(PointNet) + 2D CNN의 조합이 speed-accuracy 최적 균형점 제공
- Z축 binning 하이퍼파라미터 제거로 다양한 포인트 클라우드 설정(멀티 스캔, 레이더)에 즉시 적용 가능

**자율주행 로드맵 내 위치**:
- **CenterPoint의 backbone 선택지**: CenterPoint는 VoxelNet 또는 PointPillars를 3D 인코더로 지원하며, PointPillars 사용 시 실시간 추론 가능
- **BEVFusion의 LiDAR 브랜치**: BEVFusion의 LiDAR → BEV 피처 추출에 PointPillars 구조가 직접 활용됨
- LiDAR 기반 3D 탐지의 **속도 표준**을 정립 — 이후 논문들의 실시간성 기준이 됨

**한계**:
- Pillar 단위 처리로 z축 정보(높이)가 MaxPool로 집약 → 세밀한 수직 구조 표현 불가
- 보행자·자전거 탐지 성능이 차량보다 낮음 (작은 객체, 희소 포인트)
- 카메라와의 융합을 고려하지 않은 LiDAR 전용 설계
