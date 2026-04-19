---
title: "VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection"
date: 2026-04-19T22:40:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving", "3D Object Detection"]
tags: ["LiDAR", "Point Cloud", "3D Detection", "Voxel", "End-to-End"]
---

## 개요

- **저자**: Yin Zhou, Oncel Tuzel (Apple Inc)
- **발행년도**: 2018 (arXiv 2017, CVPR 2018)
- **논문**: [arXiv:1711.06396](https://arxiv.org/abs/1711.06396)
- **주요 내용**: LiDAR 포인트 클라우드를 **3D voxel**로 분할하고, 각 voxel 내 포인트들을 **VFE(Voxel Feature Encoding) 레이어**로 학습하여 수작업 피처 없이 end-to-end로 3D 객체를 탐지하는 최초의 통합 프레임워크

---

## 한계 극복

- **기존 한계 1 — 수작업 피처(hand-crafted features)**: bird's-eye view 높이 맵, 점령 밀도 등을 수동으로 설계. 복잡한 형상과 다양한 장면에 일반화 어려움
- **기존 한계 2 — 정보 병목**: 수작업 인코더는 3D 형상 정보를 충분히 활용하지 못해 검출 성능에 한계
- **기존 한계 3 — PointNet의 스케일 문제**: PointNet은 ~1k 포인트에서 동작하지만, LiDAR는 ~100k 포인트 → 직접 적용 시 메모리·연산량 폭발
- **이 논문의 접근 방식**: 포인트 클라우드를 3D voxel로 분할 → 각 voxel에 **VFE 레이어**(PointNet 간소화 버전) 적용 → **희소 4D 텐서**로 표현 → 3D Conv 중간 레이어 → RPN으로 박스 예측. 완전 end-to-end 학습

---

## 목차

1. Introduction
2. VoxelNet
   - 2.1 Architecture (Feature Learning Network, Convolutional Middle Layers, RPN)
   - 2.2 Loss Function
   - 2.3 Efficient Implementation
3. Training Details
4. Experiments
5. Conclusion

---

## 2. VoxelNet Architecture

### 전체 파이프라인

```
입력: LiDAR 포인트 클라우드 (~100k points)
         │
         ▼
[Feature Learning Network]
  1. Voxel Partition — 3D 공간을 균일한 voxel 격자로 분할
  2. Grouping — 각 voxel에 포인트 할당
  3. Random Sampling — voxel당 최대 T개 포인트로 샘플링
  4. Stacked VFE Layers — voxel별 포인트 피처 학습
  → 희소 4D 텐서 (C × D' × H' × W')
         │
         ▼
[Convolutional Middle Layers]
  3D Conv 레이어들 — voxel 피처를 공간적으로 집약
  → 4D 텐서를 2D BEV 피처맵으로 압축
         │
         ▼
[Region Proposal Network (RPN)]
  2D Conv로 anchor 기반 3D 박스 회귀 + 분류
         │
         ▼
출력: 3D 회전 바운딩 박스
```

### 2.1.1 Feature Learning Network — VFE Layer

**핵심 아이디어**: 각 voxel 안의 포인트들은 서로 **상호작용**하며 voxel의 형상을 함께 표현합니다. VFE 레이어는 이 상호작용을 학습합니다.

**입력 피처 구성**

비어 있지 않은 voxel $\mathbf{V}$에 $t \leq T$개 포인트가 있을 때, 각 포인트에 7차원 입력 피처를 부여합니다:

$$\hat{\mathbf{p}}_i = [x_i, y_i, z_i, r_i, x_i - v_x, y_i - v_y, z_i - v_z]^T \in \mathbb{R}^7$$

**수식 설명**:
- **$(x_i, y_i, z_i)$**: i번째 포인트의 3D 좌표
- **$r_i$**: 반사율(reflectance)
- **$(x_i - v_x, y_i - v_y, z_i - v_z)$**: 포인트와 voxel 내 모든 포인트의 **무게중심** $(v_x, v_y, v_z)$ 간의 상대 거리
- **직관**: 절대 좌표만으로는 포인트가 voxel 내 어느 부분(앞면/뒷면/중심)에 있는지 알 수 없음. 무게중심 대비 상대 위치를 추가하면 voxel 내 포인트 분포(=형상 정보)를 학습 가능

**VFE-1 레이어 동작**:

```
입력: {p̂_i} ∈ R^7 (각 포인트)
  │
  ▼ FCN (Linear → BatchNorm → ReLU)
point-wise feature f_i ∈ R^m
  │
  ├─ 각 포인트 피처 유지
  └─ Element-wise MaxPool → locally aggregated feature f̄ ∈ R^m (voxel 전체 요약)
  │
  ▼ Concatenate: f_i^out = [f_i^T, f̄^T] ∈ R^{2m}
```

**핵심**: 포인트별 피처와 voxel 전체 집약 피처를 **concat** → 각 포인트가 "자신의 지역 정보 + voxel 전체 맥락"을 동시에 보유

**VFE 스택킹**: VFE-1(7→32), VFE-2(32→128) 두 레이어를 순차 적용. VFE-2 출력에 추가 FCN 적용 → voxel-wise feature $\mathbf{f} \in \mathbb{R}^C$ (C=128)

**희소 4D 텐서**: 비어 있지 않은 voxel만 처리 → 메모리·연산량 대폭 절감 (전체 voxel의 90% 이상이 빈 공간)

### 2.1.2 Convolutional Middle Layers

3D Conv로 voxel 피처를 공간적으로 집약합니다.

차량 탐지 예시 (voxel 크기 $v_D=0.4, v_H=0.2, v_W=0.2$m):

```
희소 4D 텐서: 128 × 10 × 400 × 352
  │
  Conv3D(128, 64, 3, (2,1,1), (1,1,1))  — D축 압축
  Conv3D(64, 64, 3, (1,1,1), (0,1,1))
  Conv3D(64, 64, 3, (2,1,1), (1,1,1))
  │
  → 4D 텐서: 64 × 2 × 400 × 352
  reshape
  → 2D BEV 피처맵: 128 × 400 × 352
```

D축을 3D Conv로 점진적 압축 → 최종적으로 BEV 2D 피처맵으로 변환. 이후 RPN은 표준 2D CNN 적용.

### 2.1.3 Region Proposal Network (RPN)

FPN 스타일 3-block 구조:

```
BEV 피처맵
  │
  ├─ Block1: stride 2 다운샘플 + Conv×q → feature at scale 1
  ├─ Block2: stride 2 다운샘플 + Conv×q → feature at scale 2  
  └─ Block3: stride 2 다운샘플 + Conv×q → feature at scale 4
       │
  각 블록 출력을 고정 크기로 업샘플 후 Concat
       │
  Probability score map (2채널: foreground/background)
  Regression map (14채널: 2 anchors × 7 속성)
```

### 2.2 Loss Function

$$L = \alpha \frac{1}{N_{pos}} \sum_i L_{cls}(p_i^{pos}, 1) + \beta \frac{1}{N_{neg}} \sum_j L_{cls}(p_j^{neg}, 0) + \frac{1}{N_{pos}} \sum_i L_{reg}(\mathbf{u}_i, \mathbf{u}_i^*)$$

**수식 설명**:
- **$p_i^{pos}$, $p_j^{neg}$**: positive/negative anchor의 예측 확률
- **$L_{cls}$**: binary cross-entropy 분류 손실
- **$L_{reg}$**: SmoothL1 로컬라이제이션 손실
- **$\alpha=1.5, \beta=1$**: positive/negative 손실 균형 가중치
- **$N_{pos}, N_{neg}$**: 각 배치의 positive/negative anchor 수로 정규화

**회귀 타겟** (7차원 잔차 벡터 $\mathbf{u}^* \in \mathbb{R}^7$):

$$\Delta x = \frac{x_c^g - x_c^a}{d^a}, \quad \Delta y = \frac{y_c^g - y_c^a}{d^a}, \quad \Delta z = \frac{z_c^g - z_c^a}{h^a}$$

$$\Delta l = \log\frac{l^g}{l^a}, \quad \Delta w = \log\frac{w^g}{w^a}, \quad \Delta h = \log\frac{h^g}{h^a}, \quad \Delta\theta = \theta^g - \theta^a$$

**수식 설명**:
- **$d^a = \sqrt{(l^a)^2 + (w^a)^2}$**: anchor 기저 대각선 — 위치 잔차를 anchor 크기에 비례해 정규화
- **크기 회귀에 log 사용**: 다양한 크기(소형차·대형트럭)를 균일하게 학습
- **각도 $\Delta\theta = \theta^g - \theta^a$**: VoxelNet은 단순 차이로 회귀 (PointPillars의 sin 변환과 차이)

### 2.3 Efficient Implementation (희소 텐서 처리)

LiDAR의 ~100k 포인트를 voxel별로 처리하려면 효율적인 구현이 필수입니다.

**해시 테이블 기반 voxel 초기화**:
1. 각 포인트의 voxel 좌표를 해시 키로 사용 → O(1) 조회
2. voxel이 초기화되지 않았으면 새로 생성, voxel 좌표를 버퍼에 저장
3. voxel당 포인트 수 < T: 포인트 삽입 / > T: 무시
4. 전체 포인트 클라우드를 **single pass**로 처리 → O(n) 복잡도

**Voxel Input Feature Buffer**: $K \times T \times 7$ 텐서 (K=최대 비어 있지 않은 voxel 수)
- Stacked VFE: 포인트 레벨·voxel 레벨 dense 연산 → GPU 병렬 처리
- 최종적으로 좌표 버퍼로 희소 텐서 재구성 → 3D Conv 레이어 입력

---

## 4. Experiments

### KITTI Validation Set — BEV Detection (AP %)

| 방법 | 입력 | Car Easy | Car Mod. | Car Hard | Ped Mod. | Cyclist Mod. |
|---|---|---|---|---|---|---|
| HC-baseline | LiDAR | 88.26 | 78.42 | 77.66 | 53.79 | 42.75 |
| **VoxelNet** | **LiDAR** | **89.60** | **84.81** | **78.57** | **61.05** | **52.18** |

### KITTI Validation Set — 3D Detection (AP %)

| 방법 | 입력 | Car Easy | Car Mod. | Car Hard | Ped Mod. | Cyclist Mod. |
|---|---|---|---|---|---|---|
| HC-baseline | LiDAR | 71.73 | 59.75 | 55.69 | 40.18 | 36.07 |
| **VoxelNet** | **LiDAR** | **81.97** | **65.46** | **62.85** | **53.42** | **47.65** |

### KITTI Test Set

| 카테고리 | Easy | Moderate | Hard |
|---|---|---|---|
| Car (3D) | 77.47 | 65.46 | 57.73 |
| Car (BEV) | 89.35 | 79.26 | 77.39 |
| Pedestrian (3D) | 39.48 | 33.69 | 31.51 |
| Cyclist (3D) | 61.22 | 48.36 | 44.37 |

**HC-baseline 대비**: end-to-end VFE 학습이 BEV에서 ~8%, 3D 탐지에서 ~12% 향상 → 학습된 피처의 우월성 입증

**추론 시간 (TitanX GPU, 1.7GHz CPU)**:
- Voxel 입력 피처 계산: 5ms
- Feature Learning Network: 20ms
- Convolutional Middle Layers: 170ms ← 병목
- RPN: 30ms
- **총합: ~225ms (4.4Hz)**

---

## 5. Data Augmentation

VoxelNet은 세 가지 데이터 증강을 **on-the-fly**로 적용합니다:

**1. 박스별 perturbation**: 각 GT 박스를 Z축 회전 $\Delta\theta \sim U[-\pi/10, +\pi/10]$, 이동 $(\Delta x, \Delta y, \Delta z) \sim \mathcal{N}(0, 1)$으로 독립적으로 변형. 박스와 내부 포인트를 함께 변형하므로 일관성 유지. 충돌 시 원복

**2. 전체 스케일링**: 포인트 클라우드와 모든 박스 XYZ 좌표를 균일 스케일 $s \sim U[0.95, 1.05]$로 곱함 → 다양한 거리·크기 객체 학습

**3. 전체 회전**: 전체 포인트 클라우드와 박스를 Z축 기준 $\phi \sim U[-\pi/4, +\pi/4]$로 회전 → 차량 방향 변화 시뮬레이션

---

## 6. 핵심 개념 정리

| 개념 | 설명 |
|---|---|
| **VFE (Voxel Feature Encoding)** | voxel 내 포인트들을 FCN으로 인코딩 후 MaxPool로 집약. point-wise 피처와 voxel-wise 피처를 concat하여 지역·전역 정보 동시 포착 |
| **무게중심 상대 좌표** | $(x_i - v_x, y_i - v_y, z_i - v_z)$ 추가로 voxel 내 포인트 분포(형상) 표현 |
| **희소 4D 텐서** | 비어 있지 않은 voxel만 처리하는 희소 표현. 90%+ 빈 voxel 무시로 메모리·연산 절감 |
| **3D Convolutional Middle Layers** | voxel 피처를 Z축 방향으로 압축하여 BEV 2D 피처맵으로 변환. 이후 표준 2D RPN 적용 가능 |
| **해시 테이블 voxel 초기화** | 포인트 클라우드를 single pass(O(n))로 voxel에 할당. GPU dense 연산과 희소 텐서의 다리 역할 |
| **HC-baseline** | VoxelNet 아키텍처에 수작업 피처를 사용한 비교 모델. end-to-end 학습의 효과를 측정하는 내부 대조군 |

---

## 7. 결론 및 시사점

**VoxelNet의 의의**:
- LiDAR 3D 탐지에서 **수작업 피처를 완전히 제거**한 최초의 end-to-end 프레임워크
- VFE 레이어로 voxel 내 포인트 상호작용을 학습 → 복잡한 3D 형상 표현 가능
- 희소 텐서 표현으로 메모리 효율화, 병렬 GPU 연산 가능

**자율주행 로드맵 내 위치**:
- **PointPillars의 직접 선행 연구**: PointPillars는 VoxelNet의 3D Conv를 제거하고 z축 압축을 pillar 구조로 대체 → 속도 146배 향상 (225ms → 1.3ms 인코딩)
- **CenterPoint의 backbone 옵션**: CenterPoint는 VoxelNet을 3D 인코더로 지원 (CenterPoint-Voxel)
- **BEVFusion의 LiDAR 브랜치 기반**: BEVFusion의 LiDAR → BEV 변환에 VoxelNet/PointPillars 구조 활용
- LiDAR 3D 탐지 **end-to-end 학습 패러다임**을 정립한 기반 논문

**한계**:
- 3D Conv가 전체 추론 시간의 75% 차지 (170ms/225ms) → 실시간 불가 (4.4Hz)
- 보행자·자전거는 LiDAR 포인트가 매우 희박하여 성능 제한
- KITTI 단일 데이터셋 평가 — 다른 LiDAR 설정(nuScenes 32-beam 등)으로의 일반화 미검증
