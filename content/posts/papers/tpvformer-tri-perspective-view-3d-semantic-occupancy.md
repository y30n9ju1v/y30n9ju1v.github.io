---
title: "TPVFormer: Tri-Perspective View for Vision-Based 3D Semantic Occupancy Prediction"
date: 2026-04-19T14:00:00+09:00
draft: false
categories: ["Papers"]
tags: ["3D Occupancy", "BEV", "Autonomous Driving", "Transformer", "Semantic Scene Completion", "nuScenes"]
---

## 개요

- **저자**: Yuanhui Huang, Wenzhao Zheng, Yunpeng Zhang, Jie Zhou, Jiwen Lu (Tsinghua University, PhiGent Robotics)
- **발행년도**: 2023 (CVPR 2023)
- **arXiv**: 2302.07817
- **주요 내용**: BEV(Bird's-Eye-View)의 단일 평면 표현을 세 개의 직교 평면(Tri-Perspective View, TPV)으로 확장하여, RGB 카메라 이미지만으로 3D 공간의 모든 복셀에 대해 시맨틱 점유 예측을 수행하는 프레임워크. LiDAR 없이도 LiDAR 기반 방법과 유사한 성능을 달성한다.

## 한계 극복

- **기존 한계 1 — BEV의 높이 정보 손실**: BEV는 3D 공간을 2D 평면으로 압축하면서 z축(높이) 정보를 완전히 버린다. 자동차나 보행자처럼 높이가 중요한 객체의 정밀한 3D 구조를 표현하기 어렵다.
- **기존 한계 2 — Voxel의 계산 비용**: Voxel은 3D 구조를 정밀하게 표현하지만 저장·연산 복잡도가 O(HWD)로 실시간 온보드 적용이 어렵다.
- **기존 한계 3 — 비전 기반 LiDAR 분할의 한계**: 기존 비전 방법들은 LiDAR 분할 성능에서 LiDAR 기반 방법과 큰 격차가 있었다.
- **이 논문의 접근 방식**: 서로 수직인 세 평면(Top HW, Side DH, Front WD)으로 3D 공간을 표현하는 TPV를 제안한다. 저장·연산 복잡도를 O(HW + DH + WD)로 줄이면서도 임의 해상도의 3D 특징을 생성할 수 있다.

## 목차

- Section 1: Introduction
- Section 2: Related Work
- Section 3: Proposed Approach
  - 3.1 Generalizing BEV to TPV
  - 3.2 TPVFormer
  - 3.3 Applications of TPV
- Section 4: Experiments
  - 4.1 Task Descriptions
  - 4.2 Implementation Details
  - 4.3 3D Semantic Occupancy Prediction Results
  - 4.4 LiDAR Segmentation Results
  - 4.5 Semantic Scene Completion Results
  - 4.6 Ablation Study
- Section 5: Conclusion

---

## Section 1: Introduction

**요약**

자율주행 인식의 핵심 과제는 3D 공간을 얼마나 효율적이고 표현력 있게 나타내느냐이다. 기존의 두 가지 주요 표현 방식은 각각 한계가 있었다.

- **Voxel 표현**: 3D 구조를 정밀하게 담지만, 공간 크기 H×W×D에 비례하는 O(HWD) 연산량이 필요해 실시간 온보드 적용이 어렵다.
- **BEV 표현**: 높이 차원을 압축하여 O(HW)로 효율적이지만, 그 과정에서 z축 정보가 사라져 세밀한 3D 구조 표현이 불가능하다.

이 논문은 BEV를 세 개의 직교 평면으로 일반화한 **Tri-Perspective View(TPV)** 표현을 제안한다. TPV는 Top(HW), Side(DH), Front(WD) 세 평면으로 구성되며, 임의의 3D 점은 각 평면에 투영된 세 특징의 합산으로 표현된다. 이로써 BEV 수준의 연산 효율성을 유지하면서 Voxel에 가까운 3D 표현력을 확보한다.

**핵심 개념**

- **TPV (Tri-Perspective View)**: Top(H×W), Side(D×H), Front(W×D) 세 직교 평면의 집합. 각 평면은 서로 다른 시점에서 장면의 구조 정보를 담는다.
- **Point Querying**: 3D 공간의 임의 점 (x,y,z)을 세 평면에 투영하여 특징을 조합, 그 점의 시맨틱 레이블을 예측하는 방식.
- **Voxel Feature**: 각 TPV 평면을 직교 방향으로 broadcast·summation하여 완전한 HWD 규모의 voxel 특징 텐서를 복원하는 방식.

---

## Section 3: Proposed Approach

### 3.1 Generalizing BEV to TPV

**요약**

BEV는 3D 점 (x,y,z)을 (h,w) 좌표로 투영하며, 같은 (x,y)에 있는 모든 z 값의 점들이 동일한 특징을 공유한다.

$$\mathbf{f}_{x,y,\mathbf{Z}} = \mathbf{b}_{h,w} = \mathcal{S}(\mathbf{B}, \mathcal{P}_{bev}(x,y)) \tag{2}$$

**수식 설명**
- **$\mathbf{f}_{x,y,\mathbf{Z}}$**: 같은 (x,y) 위치에 있는 모든 z값 점들의 특징 (높이 방향 전체가 동일한 값을 가짐)
- **$\mathbf{B}$**: BEV 특징 맵 $\mathbb{R}^{H \times W \times C}$
- **$\mathcal{P}_{bev}(x,y)$**: 3D 좌표를 BEV 좌표 (h,w)로 투영하는 함수
- **$\mathcal{S}$**: 해당 위치의 특징을 샘플링하는 함수 (bilinear interpolation)
- **핵심**: z 좌표 정보가 완전히 버려지는 것이 BEV의 근본적 한계

TPV는 이 문제를 세 평면으로 해결한다.

$$\mathbf{T} = [\mathbf{T}^{HW}, \mathbf{T}^{DH}, \mathbf{T}^{WD}] \tag{3}$$

$$\mathbf{T}^{HW} \in \mathbb{R}^{H \times W \times C},\quad \mathbf{T}^{DH} \in \mathbb{R}^{D \times H \times C},\quad \mathbf{T}^{WD} \in \mathbb{R}^{W \times D \times C}$$

**수식 설명**
- **$\mathbf{T}^{HW}$**: 위에서 내려다본 Top 평면 (기존 BEV와 동일한 시점)
- **$\mathbf{T}^{DH}$**: 옆에서 바라본 Side 평면 (깊이×높이, x축 방향 시점)
- **$\mathbf{T}^{WD}$**: 앞에서 바라본 Front 평면 (너비×깊이, y축 방향 시점)
- **H, W, D**: 각각 높이(Height), 너비(Width), 깊이(Depth) 방향 해상도
- **C**: 특징 차원

임의 3D 점 (x,y,z)의 특징은 세 평면에서 샘플링한 특징의 합산으로 계산된다.

$$\mathbf{t}_{h,w} = \mathcal{S}(\mathbf{T}^{HW}, \mathcal{P}_{hw}(x,y)), \quad \mathbf{t}_{d,h} = \mathcal{S}(\mathbf{T}^{DH}, \mathcal{P}_{dh}(z,x)), \quad \mathbf{t}_{w,d} = \mathcal{S}(\mathbf{T}^{WD}, \mathcal{P}_{wd}(y,z)) \tag{4}$$

$$\mathbf{f}_{x,y,z} = \mathcal{A}(\mathbf{t}_{h,w}, \mathbf{t}_{d,h}, \mathbf{t}_{w,d}) \tag{5}$$

**수식 설명**
- **$\mathbf{t}_{h,w}$**: Top 평면에서 (x,y) 투영 좌표로 샘플링한 특징
- **$\mathbf{t}_{d,h}$**: Side 평면에서 (z,x) 투영 좌표로 샘플링한 특징
- **$\mathbf{t}_{w,d}$**: Front 평면에서 (y,z) 투영 좌표로 샘플링한 특징
- **$\mathcal{A}$**: 세 특징을 합산(summation)하는 집약 함수
- **직관**: 세 시점이 서로 다른 정보를 보완하기 때문에, 합산만으로도 3D 점을 풍부하게 표현할 수 있다

**저장·연산 복잡도 비교**

| 표현 방식 | 복잡도 |
|----------|--------|
| Voxel | O(H×W×D) |
| BEV | O(H×W) |
| **TPV** | **O(H×W + D×H + W×D)** |

Voxel 대비 한 차원 낮은 복잡도로 모든 축의 정보를 보존한다.

---

### 3.2 TPVFormer

**요약**

TPV 특징을 2D 이미지에서 효과적으로 생성하기 위해 **TPVFormer** 인코더를 제안한다. 핵심 구성 요소는 두 종류의 어텐션 블록이다.

**전체 구조**

```
이미지 백본 (ResNet)
    │
    ▼ 멀티스케일 이미지 특징
TPVFormer
  ├─ HCAB (Hybrid Cross-Attention Block) × N₁
  │    ├─ ICA: Image Cross-Attention (이미지 → TPV)
  │    └─ CVHA: Cross-View Hybrid-Attention (TPV 평면 간 상호작용)
  └─ HAB (Hybrid-Attention Block) × N₂
       └─ CVHA만 사용 (문맥 정보 정제)
    │
    ▼ TPV 특징 T = [T^HW, T^DH, T^WD]
예측 헤드 (경량 MLP)
    │
    ▼ 시맨틱 점유 예측
```

**Image Cross-Attention (ICA)**

TPV 쿼리가 2D 이미지 특징에서 시각 정보를 수집하는 단계. TPV 쿼리의 3D 좌표를 역투영하여 실세계 좌표를 구하고, 이를 카메라 이미지에 투영하여 레퍼런스 포인트를 만든다.

상단 평면(Top) 쿼리 $\mathbf{t}_{h,w}$의 실세계 좌표:

$$(x, y) = \mathcal{P}_{hw}^{-1}(h, w) = \left((h - \frac{H}{2}) \times s,\ (w - \frac{W}{2}) \times s\right) \tag{6}$$

레퍼런스 포인트 집합 (z축 방향으로 균일 샘플링):

$$\mathbf{Ref}_{h,w}^{world} = (\mathcal{P}_{hw}^{-1}(h,w), \mathbf{Z}) = \{(x,y,z_i)\}_{i=1}^{N_{ref}^{HW}} \tag{7}$$

픽셀 좌표 변환:

$$\mathbf{Ref}_{h,w}^{pix} = \mathcal{P}_{pix}(\mathbf{Ref}_{h,w}^{world}) = \mathcal{P}_{pix}(\{(x,y,z_i)\}) \tag{8}$$

최종 ICA 출력 (유효 카메라 수 $N_{h,w}^{val}$에 걸쳐 deformable attention 평균):

$$\text{ICA}(\mathbf{t}_{h,w}, \mathbf{I}) = \frac{1}{|N_{h,w}^{val}|} \sum_{j \in N_{h,w}^{val}} \text{DA}(\mathbf{t}_{h,w}, \mathbf{Ref}_{h,w}^{pix,j}, \mathbf{I}_j) \tag{9}$$

**수식 설명**
- **$s$**: TPV 그리드 셀 하나가 실세계에서 차지하는 크기(미터)
- **$\mathbf{Z}$**: z축 방향으로 균일하게 샘플링한 고도값 집합
- **$N_{ref}^{HW}$**: Top 평면 쿼리당 레퍼런스 포인트 수
- **$\mathcal{P}_{pix}$**: 실세계 좌표를 카메라 픽셀 좌표로 변환하는 원근 투영
- **$\text{DA}$**: Deformable Attention — 레퍼런스 포인트 주변에서 오프셋을 예측해 필요한 위치만 샘플링하는 효율적 어텐션
- **핵심**: 쿼리가 어떤 픽셀을 봐야 하는지를 3D 기하 정보로 초기화하므로, 학습 없이도 올바른 영역을 참조하는 귀납적 편향이 생긴다

**Cross-View Hybrid-Attention (CVHA)**

세 TPV 평면이 서로 정보를 교환하는 단계. Top 평면 쿼리 $\mathbf{t}_{h,w}$의 레퍼런스 포인트를 세 평면에 분배:

$$\mathbf{R}_{h,w} = \mathbf{R}_{h,w}^{top} \cup \mathbf{R}_{h,w}^{side} \cup \mathbf{R}_{h,w}^{front} \tag{10}$$

$$\mathbf{R}_{h,w}^{side} = \{(d_i, h)\}_i, \quad \mathbf{R}_{h,w}^{front} = \{(w, d_i)\}_i \tag{11}$$

$$\text{CVHA}(\mathbf{t}_{h,w}) = \text{DA}(\mathbf{t}_{h,w}, \mathbf{R}_{h,w}, \mathbf{T}) \tag{12}$$

**수식 설명**
- **$\mathbf{R}_{h,w}^{top}$**: Top 평면 내 이웃 포인트 (로컬 문맥)
- **$\mathbf{R}_{h,w}^{side}$**: Top 쿼리에 수직인 Side 평면의 포인트들 (높이 정보 획득)
- **$\mathbf{R}_{h,w}^{front}$**: Top 쿼리에 수직인 Front 평면의 포인트들 (깊이-높이 정보 획득)
- **직관**: Top 평면 쿼리가 자기 평면 이웃 + Side + Front의 보완 정보를 동시에 수집하여 3D 문맥을 이해한다

---

### 3.3 Applications of TPV

**Point Feature**: 3D 점 (x,y,z)을 세 평면에 투영 후 특징 합산 → 경량 MLP로 시맨틱 레이블 예측  
**Voxel Feature**: 각 TPV 평면을 직교 방향으로 broadcast 후 합산 → 완전한 HWD 복셀 특징 복원

---

## Section 4: Experiments

### 태스크 및 데이터셋

| 태스크 | 데이터셋 | 설명 |
|--------|---------|------|
| 3D Semantic Occupancy Prediction | Panoptic nuScenes | 희소 LiDAR 감독으로 학습, 모든 복셀 예측 |
| LiDAR Segmentation | nuScenes test | 카메라만 입력, LiDAR 포인트로 평가 |
| Semantic Scene Completion (SSC) | SemanticKITTI | 카메라 입력, 전체 복셀 시맨틱 완성 |

### 구현 세부사항

- **Backbone**: TPVFormer-Base = ResNet101-DCN (FCOS3D 사전학습), TPVFormer-Small = ResNet-50 (ImageNet)
- **TPV 해상도**: Base = 200×200×16, Small = 100×100×8
- **학습**: AdamW, lr=2e-4, 24 epoch, 8× A100

### 주요 결과

**nuScenes LiDAR 분할 (test set)**

| 방법 | 입력 | mIoU |
|------|------|------|
| Cylinder3D++ (SOTA LiDAR) | LiDAR | 77.9 |
| BEVFormer-Base | Camera | 56.2 |
| **TPVFormer-Small (ours)** | **Camera** | **59.2** |
| **TPVFormer-Base (ours)** | **Camera** | **69.4** |

카메라만으로 LiDAR 기반 방법의 약 70% mIoU 수준을 달성. BEVFormer-Base 대비 +12.7% 향상.

**SemanticKITTI SSC (test set)**

| 방법 | SC IoU | SSC mIoU |
|------|--------|----------|
| MonoScene | 34.16 | 11.08 |
| **TPVFormer (ours)** | **34.25** | **11.26** |

파라미터 6.0M vs MonoScene 15.7M, 연산량 128G vs 500G FLOP으로 압도적 효율성.

### Ablation Study 요점

- **손실 함수**: Voxel 예측 + Point 예측 모두 손실에 사용할 때 최고 성능 (mIoU 64.80)
  - Voxel만: 63.17, Point만: 49.94 — 두 예측이 서로를 정규화하는 효과
- **해상도 vs 특징 차원**: 해상도를 키우는 것이 특징 차원을 키우는 것보다 일관되게 효과적
  - BEVFormer 200×200 (256dim): 56.21 → TPVFormer 200×200×16 (128dim): **68.86**
- **HCAB vs HAB 비율**: HCAB 많을수록 IoU 향상 (시각 정보 중요), HAB 적당히 섞을 때 mIoU 최고

---

## 핵심 개념 정리

- **TPV (Tri-Perspective View)**: 3D 공간을 Top·Side·Front 세 직교 평면으로 분해한 표현. BEV의 높이 정보 손실 문제를 해결하면서 Voxel보다 한 차원 낮은 복잡도 O(HW+DH+WD)를 유지한다.
- **Point Querying**: 임의의 3D 좌표를 세 평면에 투영한 특징의 합으로 표현하는 방식. 테스트 시 해상도를 재학습 없이 조정 가능하게 한다.
- **ICA (Image Cross-Attention)**: TPV 쿼리의 3D 기하 정보를 이용해 참조할 이미지 픽셀 위치를 초기화하는 deformable attention. 3D 기하 귀납적 편향이 학습 효율을 높인다.
- **CVHA (Cross-View Hybrid-Attention)**: 세 TPV 평면이 서로 정보를 교환하는 메커니즘. Top 평면 쿼리가 Side·Front 평면의 수직 방향 특징을 직접 참조한다.
- **Sparse Supervision**: 학습 시 희소 LiDAR 포인트로만 감독 신호를 받지만, 추론 시 모든 복셀의 점유와 시맨틱 레이블을 예측한다.

---

## 결론 및 시사점

TPVFormer는 BEV와 Voxel 표현의 장점을 결합한 TPV 표현을 통해, **카메라만으로 LiDAR 수준에 근접한 3D 시맨틱 장면 이해**를 달성했다.

**로드맵 내 위치**: BEVFormer의 직접적 후속으로, BEV 단일 평면의 높이 정보 손실을 해결하는 핵심 방향을 제시한다. Occ3D와 함께 3D 점유 예측 분야의 기반 논문으로 자리잡았으며, 이후 SurroundOcc, OccNet 등의 출발점이 된다.

**실무 시사점**:
- 합성 데이터로 3D 점유 레이블을 생성할 때, TPV 해상도를 테스트 시 자유롭게 조정할 수 있는 특성이 유용하다
- 카메라 전용 파이프라인으로 LiDAR 없는 회귀 테스트 시나리오에서 3D 장면 이해가 가능해진다
- BEVFormer 인프라를 그대로 재사용할 수 있어 기존 BEV 기반 시스템에 통합하기 쉽다
