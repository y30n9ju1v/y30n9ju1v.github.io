---
title: "BEVDepth: Acquisition of Reliable Depth for Multi-view 3D Object Detection"
date: 2026-04-17T00:00:00+09:00
draft: false
categories: ["Papers"]
tags: ["3D Object Detection", "BEV", "Depth Estimation", "Autonomous Driving", "Computer Vision"]
---

## 개요

- **저자**: Yinhao Li, Zheng Ge, Guanyi Yu, Jinrong Yang, Zengran Wang, Yukang Shi, Jianjian Sun, Zeming Li
- **소속**: Chinese Academy of Sciences, MEGVII Technology, Huazhong University of Science and Technology, Xi'an Jiaotong University
- **발행년도**: 2022 (arXiv:2206.10092v2, 30 Nov 2022)
- **주요 내용**: 카메라 기반 Bird's-Eye-View(BEV) 3D 객체 탐지에서 신뢰할 수 있는 깊이(depth)를 획득하기 위한 새로운 방법론 BEVDepth 제안. nuScenes 테스트셋에서 60.9% NDS 달성으로 당시 SOTA.

## 한계 극복

이 논문은 기존 Lift-Splat-Shoot(LSS) 기반 BEV 탐지기들이 깊이 추정에서 겪는 근본적 문제를 해결하기 위해 작성되었습니다.

- **기존 한계 1 - Inaccurate Depth**: 기존 방법에서 깊이 예측 모듈은 최종 탐지 손실을 통해 간접적으로만 지도(supervise)됩니다. 따라서 절대적 깊이 품질이 낮으며, 탐지 성능은 그럭저럭 나오더라도 깊이 자체는 부정확합니다.
- **기존 한계 2 - Depth Module Over-fitting**: 대부분의 픽셀이 합리적인 깊이를 예측하도록 훈련되지 않아, 깊이 모듈이 이미지 크기·카메라 파라미터 등 하이퍼파라미터에 과적합됩니다. 일반화 능력이 크게 저하됩니다.
- **기존 한계 3 - Imprecise BEV Semantics**: Lift-splat에서 학습된 깊이로 이미지 특징을 BEV 공간에 unproject할 때, 깊이가 부정확하면 특징이 잘못된 위치에 투영되어 BEV 의미론(semantics)이 부정확해집니다.
- **이 논문의 접근 방식**: (1) 포인트 클라우드 기반 명시적 깊이 지도학습(Explicit Depth Supervision), (2) 카메라 내재 파라미터를 깊이 추정에 통합하는 Camera-aware Depth Prediction, (3) 잘못 투영된 특징을 정제하는 Depth Refinement Module을 도입하여 세 가지 문제를 동시에 해결합니다.

## 목차

- Section 1: Introduction
- Section 2: Related Work
- Section 3: Delving into Depth Prediction in Lift-splat
- Section 4: BEVDepth
- Section 5: Experiment
- Section 6: Conclusion

---

## Section 1: Introduction

**요약**

자율주행 인식 시스템에서 LiDAR와 카메라는 두 가지 주요 센서입니다. LiDAR 기반 방법이 신뢰할 수 있는 3D 탐지를 보여주는 반면, 멀티뷰 카메라 기반 방법은 낮은 비용 덕분에 주목받고 있습니다.

카메라로 3D를 인식하는 핵심 접근은 **LSS(Lift-Splat-Shoot)**: 이미지 특징을 추정된 깊이로 3D frustum(절두체)에 "lift"한 뒤, BEV 평면에 "splat"하는 방식입니다. 그런데 nuScenes 벤치마크에서 Lift-splat 기반 탐지기가 30 mAP에 달해도 깊이 품질은 매우 낮다는 사실을 발견합니다. 이 관찰에서 출발하여, 저자들은 **깊이의 품질이 정확한 3D 탐지의 핵심**임을 주장하고 BEVDepth를 제안합니다.

BEVDepth는 카메라 인식 깊이 추정 모듈(DepthNet)과 Depth Refinement Module을 통해 신뢰할 수 있는 깊이를 생성하며, nuScenes 테스트셋에서 60.9% NDS를 달성합니다.

**핵심 개념**

- **BEV (Bird's-Eye-View)**: 하늘에서 내려다보는 시점의 특징 맵. 자율주행에서 여러 카메라의 정보를 통합하고, 3D 위치 추정에 자연스러운 좌표계를 제공합니다.
- **LSS (Lift-Splat-Shoot)**: 이미지 픽셀마다 깊이 분포를 예측하고, 해당 깊이에 이미지 특징을 배치(lift)한 뒤 BEV 격자에 집계(splat)하는 파이프라인.
- **NDS (nuScenes Detection Score)**: 탐지 품질을 종합 평가하는 지표로, mAP + 5가지 TP 메트릭(mATE, mASE, mAOE, mAVE, mAAE)의 가중합.

---

## Section 2: Related Work

**요약**

### 2.1 Vision-based 3D Object Detection

단일 카메라로 3D 바운딩 박스를 추정하는 문제는 본질적으로 ill-posed입니다. 깊이 추정이 핵심 구성 요소이며, 다양한 방법들이 이를 해결하려 했습니다:

- **CenterNet, MUD-RPN**: 2D 이미지에서 직접 3D 예측
- **D4LCN**: 깊이 맵을 활용해 3D 타겟을 인식
- **DD3D**: 대규모 깊이 사전 학습으로 성능 향상
- **DETR3D, PETR**: 3D 쿼리 기반 Transformer 구조로 멀티뷰 통합

### 2.2 LiDAR-based 3D Object Detection

LiDAR 기반 방법들은 포인트 클라우드에서 직접 깊이 정보를 얻을 수 있어 카메라 방법보다 정확도가 높습니다. VoxelNet, PointPillars, CenterPoint 등이 대표적입니다.

### 2.3 Depth Estimation

깊이 추정의 정확도가 3D 탐지 전반의 품질에 영향을 미칩니다. 이 논문은 특히 깊이 감독 방식의 개선에 초점을 맞춥니다.

**핵심 개념**

- **ill-posed problem**: 단안 카메라로 절대 깊이를 추정하는 것은 수학적으로 해가 유일하지 않은 문제. 동일한 2D 이미지가 여러 3D 장면에서 생성될 수 있습니다.
- **Frustum**: 카메라 시야각이 이루는 3D 피라미드 형태의 공간. 픽셀 위치 + 깊이 분포로 3D 공간을 표현합니다.

---

## Section 3: Delving into Depth Prediction in Lift-splat

**요약**

이 섹션에서는 기존 Lift-splat 기반 탐지기의 깊이 예측 문제를 세 가지로 구체적으로 분석합니다.

### 3.1 Base Detector 구조

Base Detector는 LSS 파이프라인을 따릅니다:
- **Image Encoder**: ResNet 등 백본으로 이미지 특징 추출
- **Depth Module**: 각 픽셀의 깊이 분포 $D^{pred}$ 예측
- **View Transformer**: 깊이 분포와 이미지 특징을 곱하여 BEV 특징 생성
- **Detection Head**: BEV 특징에서 3D 바운딩 박스 출력

### 3.2 깊이 품질 평가 실험

학습된 깊이 $D^{pred}$를 LiDAR 포인트 클라우드 기반 ground-truth와 비교 평가:

| 방법 | mAP↑ | AbsRel↓ | SqRel↓ | RMSE↓ |
|------|-------|---------|--------|-------|
| 학습된 깊이 | 27.62 | 0.23 | 2.09 | 5.78 |
| 랜덤 깊이 | 27.87 | 0.38 | 4.06 | 8.29 |
| 정답 깊이 | 34.12 | 0 | 0 | 0 |

AbsRel이 0.23으로 기존 단안 깊이 추정 알고리즘보다 훨씬 나쁜 수치입니다. 이는 깊이가 암묵적으로만 학습되어 부정확함을 증명합니다.

### 3.3 Depth Module Over-fitting 분석

Enhanced Detector(명시적 깊이 지도학습 적용)와 Base Detector를 서로 다른 이미지 크기(192×640, 256×704, 320×864)로 테스트한 결과, Base Detector는 학습 이미지 크기 외에서 성능이 크게 떨어지는 반면 Enhanced Detector는 견고합니다. 이는 Base Detector가 카메라 파라미터에 과적합됨을 보여줍니다.

### 3.4 Imprecise BEV Semantics 분석

BEV 위의 각 특징을 분류 heatmap으로 평가하면, Enhanced Detector(깊이 지도학습 적용)가 더 많은 구조 정보를 보존합니다(그림 3 참조). 부정확한 깊이는 특징이 잘못된 BEV 위치에 투영되어 분류 성능을 저하시킵니다.

**핵심 개념**

- **AbsRel (Absolute Relative Error)**: 깊이 추정 오차 지표. $\frac{1}{N}\sum |d - d^*| / d^*$. 값이 낮을수록 좋음.
- **View Transformer**: 각 카메라의 2D 특징을 3D → BEV로 변환하는 모듈. 깊이의 정확도에 결과가 크게 의존합니다.

---

## Section 4: BEVDepth

**요약**

BEVDepth는 세 가지 핵심 구성 요소를 통해 신뢰할 수 있는 깊이를 획득합니다.

### 4.1 Explicit Depth Supervision

포인트 클라우드 P에서 직접 획득한 ground-truth 깊이 $D^{gt}$로 깊이 모듈을 명시적으로 지도학습합니다.

**LiDAR → 카메라 투영 공식:**

$$P_i^{img}(u, v, d) = K_i(R_i P + t_i)$$

**수식 설명**
- $P_i^{img}(u, v, d)$: i번째 카메라 이미지에 투영된 포인트의 좌표 (u, v는 픽셀 위치, d는 깊이)
- $K_i \in \mathbb{R}^{3 \times 3}$: i번째 카메라의 내재 파라미터 행렬 (초점 거리, 주점 등)
- $R_i \in \mathbb{R}^{3 \times 3}$: LiDAR 좌표계 → 카메라 좌표계 회전 행렬
- $t_i \in \mathbb{R}^3$: LiDAR 좌표계 → 카메라 좌표계 변환 벡터
- $P$: 포인트 클라우드의 3D 포인트

중간 깊이 예측 $D^{pred}$을 구하는 과정:

$$D_i^{pt} = \phi(P_i^{img})$$

$$D_i^{pred} = \psi(SE(F_i^{2d}|MLP(\xi(R_i) \oplus \xi(t_i) \oplus \xi(K_i))))$$

**수식 설명**
- $\phi$: 투영된 포인트 클라우드를 깊이 이미지로 변환하는 함수
- $F_i^{2d}$: i번째 카메라의 2D 이미지 특징
- $\xi$: Flatten 연산 (행렬을 1D 벡터로 펼침)
- $R_i, t_i, K_i$: 카메라 외재·내재 파라미터
- $\oplus$: 벡터 연결(concatenation)
- $MLP(\cdot)$: 카메라 파라미터를 임베딩으로 변환하는 다층 퍼셉트론
- $SE(\cdot)$: Squeeze-and-Excitation 모듈 — 카메라 파라미터 임베딩으로 이미지 특징을 재가중
- $\psi$: 최종 깊이 분포를 출력하는 함수

깊이 손실은 Binary Cross Entropy를 사용합니다:

$$L_{depth} = BCE(D^{pred}, D^{gt})$$

### 4.2 Camera-aware Depth Prediction

카메라마다 FOV, 해상도, 위치가 다른 nuScenes 데이터셋에서 카메라 파라미터를 깊이 추정에 명시적으로 통합합니다.

- 카메라 내재 파라미터를 MLP로 임베딩하여 이미지 특징을 조정(Squeeze-and-Excitation)
- 이를 통해 DepthNet이 자율주행 환경에서 다양한 카메라 설정에 적응 가능

**기존 방법과의 차이**: Park et al.(2021b)도 카메라 인식을 활용하지만, 회귀 타겟을 카메라 내재 파라미터로 스케일링하여 복잡한 카메라 설정에 적응하기 어렵습니다. BEVDepth는 카메라 파라미터를 DepthNet 내부에 직접 모델링하여 더 일반적입니다.

### 4.3 Depth Refinement Module

부정확하게 투영된 frustum 특징을 정제하기 위한 모듈입니다.

- View Transformer 이전 단계에서, $F^{2d}$를 $[C_F, H, W]$에서 $[C_D \times H, W]$로 reshape
- $C_D \times W$ 평면에 3×3 합성곱 스택 적용
- 출력은 다시 reshape되어 Voxel/Pillar Pooling에 입력

**역할**: 
- 깊이 예측 신뢰도가 낮을 때 깊이 축을 따라 특징을 집계
- 잘못 투영된 특징을 올바른 위치로 정제
- Depth Refinement Module이 분리된 구조이므로, 탐지 헤드 변경 없이 적용 가능

### 4.4 전체 아키텍처

```
멀티뷰 이미지
    ↓
[Image Backbone] → 이미지 특징 F^2d
    ↓
[DepthNet (카메라 파라미터 입력)] → 깊이 분포 + Context 특징
    ↓                    ↓
[Depth Supervision]  [Depth Refinement Module]
(LiDAR GT로 학습)         ↓
                   [Efficient Voxel Pooling]
                         ↓
                    BEV Feature Map
                         ↓
                   [Detection Head]
                         ↓
                    3D 바운딩 박스
```

**핵심 개념**

- **Squeeze-and-Excitation (SE)**: 채널별 중요도를 동적으로 재조정하는 어텐션 메커니즘. 카메라 파라미터 임베딩으로 이미지 특징의 각 채널 가중치를 조정합니다.
- **Efficient Voxel Pooling**: GPU 병렬성을 활용해 각 frustum 특징을 해당 BEV 격자에 CUDA 스레드로 할당. Lift-splat 대비 풀링 속도 80배 향상.
- **Multi-frame Fusion**: 여러 프레임의 BEV 특징을 ego 좌표계로 정렬 후 연결하여, 속도 추정(velocity estimation) 성능 향상.

---

## Section 5: Experiment

**요약**

### 5.1 실험 설정

- **데이터셋**: nuScenes — 6개 카메라, 1개 LiDAR, 5개 레이더. 700/150/150 scenes (train/val/test).
- **평가 지표**: NDS, mAP, mATE, mASE, mAOE, mAVE, mAAE
- **구현**: ResNet-50/101 백본, 이미지 크기 256×704, AdamW optimizer, 24 epochs

### 5.2 Ablation Study

각 구성 요소의 기여도 분석 (nuScenes val):

| DL | CA | DR | MF | mAP↑ | mATE↓ | mAOE↓ | NDS↑ |
|----|----|----|-----|------|-------|-------|------|
|    |    |    |    | 0.304 | 0.768 | 0.698 | 0.327 |
| ✓  |    |    |    | 0.306 | 0.747 | 0.612 | 0.344 |
| ✓  | ✓  |    |    | 0.314 | 0.706 | 0.647 | 0.357 |
| ✓  | ✓  | ✓  |    | 0.322 | 0.707 | 0.636 | 0.367 |
| ✓  | ✓  | ✓  | ✓  | 0.330 | 0.699 | 0.545 | 0.442 |

*(DL: Depth Loss, CA: Camera-awareness, DR: Depth Refinement, MF: Multi-frame)*

- Depth Loss: mAP +0.2%, NDS +1.7%
- Camera-awareness: mATE -0.41 (가장 큰 효과)
- Depth Refinement: mAP +0.8%, NDS +1.0%
- Multi-frame: NDS +7.5% (속도 추정 대폭 향상)

### 5.3 nuScenes 벤치마크 비교 (test set)

| Method | Modality | mAP↑ | NDS↑ |
|--------|----------|-------|------|
| BEVDet | C | 0.422 | 0.463 |
| BEVFormer | C | — | 0.448 |
| PETRv2 | C | 0.490 | 0.582 |
| **BEVDepth** | C | 0.503 | 0.600 |
| **BEVDepth†** | C | 0.520 | **0.609** |

BEVDepth†는 ConvNeXt 백본 사용. 카메라 전용 방법 중 1위 달성.

### 5.4 Efficient Voxel Pooling & Multi-frame Fusion

- **Efficient Voxel Pooling**: 기존 Lift-splat의 "sort+cumsum" 트릭 대신 CUDA 기반 직접 할당. 풀링 속도 **80배** 향상, 전체 학습 시간 5일 → 1.5일로 단축.
- **Multi-frame Fusion**: 과거 프레임의 BEV 특징을 현재 ego 좌표로 변환 후 연결. 속도 추정(mAVE)을 크게 개선.

**핵심 개념**

- **mATE (mean Average Translation Error)**: 탐지된 객체 위치의 평균 오차 (미터 단위). 낮을수록 좋음.
- **mAVE (mean Average Velocity Error)**: 속도 추정 오차. Multi-frame Fusion이 이를 크게 개선합니다.

---

## Section 6: Conclusion

**요약**

BEVDepth는 멀티뷰 카메라 기반 3D 탐지에서 신뢰할 수 있는 깊이를 획득하기 위한 새로운 네트워크 구조입니다. 기존 3D 탐지기의 깊이 학습 메커니즘을 분석하여 세 가지 결함(부정확한 깊이, 과적합, 부정확한 BEV 의미론)을 발견하고, 이를 Camera-aware Depth Prediction과 Depth Refinement Module로 해결합니다.

BEVDepth는 nuScenes 리더보드에서 카메라 전용 방법 중 1위(60.9% NDS)를 달성하며, 미래 멀티뷰 3D 탐지 연구의 강력한 베이스라인이 될 것으로 기대됩니다.

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **BEV (Bird's-Eye-View)** | 위에서 내려다보는 시점의 2D 특징 맵. 자율주행 인식에 자연스러운 좌표계 |
| **LSS (Lift-Splat-Shoot)** | 이미지 → 깊이 분포 → 3D frustum → BEV 변환 파이프라인 |
| **Explicit Depth Supervision** | LiDAR 포인트 클라우드를 GT로 활용한 명시적 깊이 지도학습 |
| **Camera-aware Depth Prediction** | 카메라 내재/외재 파라미터를 SE 모듈로 깊이 추정에 통합 |
| **Depth Refinement Module** | 잘못 투영된 frustum 특징을 Voxel Pooling 전에 정제하는 모듈 |
| **Efficient Voxel Pooling** | GPU CUDA 병렬화로 BEV 풀링 속도 80배 향상 |
| **Multi-frame Fusion** | 과거 프레임 BEV를 현재 좌표로 정렬 후 연결해 속도 추정 향상 |
| **NDS (nuScenes Detection Score)** | mAP + 5개 TP 메트릭의 가중 합산 종합 평가 지표 |

## 결론 및 시사점

**핵심 인사이트**: 카메라 기반 3D 탐지에서 **깊이의 품질이 탐지 성능의 병목**입니다. 탐지 손실로만 간접적으로 깊이를 학습하면, 탐지 지표는 어느 정도 나오더라도 깊이 자체는 매우 부정확합니다.

**실무적 시사점**:
1. **LiDAR-Camera 융합**: LiDAR를 직접 탐지에 사용하지 않아도, GT 깊이 생성에 활용함으로써 카메라 전용 시스템의 성능을 크게 끌어올릴 수 있습니다.
2. **카메라 파라미터 모델링**: 다양한 FOV를 가진 멀티카메라 시스템에서는 카메라 파라미터를 깊이 추정 네트워크에 명시적으로 통합하는 것이 필수적입니다.
3. **효율성**: Efficient Voxel Pooling으로 실용적인 훈련 속도를 달성하며, Multi-frame Fusion으로 추가 센서 없이 속도 추정이 가능합니다.
4. **BEVDepth는 베이스라인**: 이후 BEVStereo, BEVFusion 등 많은 후속 연구의 기반이 되는 강력한 베이스라인을 제공합니다.
