---
title: "BEVDet: High-Performance Multi-Camera 3D Object Detection in Bird-Eye-View"
date: 2026-04-18T10:00:00+09:00
draft: false
categories: ["Papers"]
tags: ["3D Object Detection", "Autonomous Driving", "BEV", "Multi-Camera", "Computer Vision"]
---

## 개요

- **저자**: Junjie Huang, Guan Huang, Zheng Zhu, Yun Ye, Dalong Du (PhiGent Robotics)
- **발행년도**: 2022 (arXiv:2112.11790v3, 16 Jun 2022)
- **주요 내용**: Bird-Eye-View(BEV) 공간에서 멀티카메라 이미지를 이용해 3D 물체를 탐지하는 BEVDet 패러다임 제안. 기존 이미지 뷰 기반 방식 대비 뛰어난 속도-정확도 트레이드오프 달성.

## 한계 극복

이 논문은 기존 비전 기반 3D 물체 탐지의 다음 한계를 극복하기 위해 작성되었습니다.

- **기존 한계 1 — 이미지 뷰와 BEV 분리**: FCOS3D, PGD 같은 기존 방법들은 이미지 뷰에서만 탐지를 수행하여 물체의 translation(위치), velocity(속도), orientation(방향) 예측에 약점이 있었습니다.
- **기존 한계 2 — 속도와 정확도의 트레이드오프**: FCOS3D(1.7 FPS, 2,008 GFLOPs)와 같이 높은 연산 비용을 요구하거나, DETR3D처럼 느린 추론 속도를 가졌습니다.
- **기존 한계 3 — 과적합 문제**: BEV 공간에서의 학습은 데이터 수가 적어(카메라 수만큼 배치 크기가 분할) 과적합이 발생하기 쉬웠습니다.
- **이 논문의 접근 방식**: BEV 시맨틱 분할 패러다임을 3D 물체 탐지로 확장하고, BEV 공간 전용 데이터 증강(BDA)과 Scale-NMS를 도입하여 과적합을 방지하고 탐지 성능을 크게 향상시켰습니다.

## 목차

- Chapter 1: Introduction — BEV 공간에서의 3D 탐지 필요성과 BEVDet 개요
- Chapter 2: Related Works — 2D 비전 인식, BEV 시맨틱 분할, 3D 물체 탐지 관련 연구
- Chapter 3: Methodology — 네트워크 구조, 데이터 증강 전략, Scale-NMS
- Chapter 4: Experiment — 벤치마크 결과 및 ablation study
- Chapter 5: Conclusion — 요약 및 향후 연구

---

## Chapter 1: Introduction

**요약**

자율주행은 주변 환경을 인식하여 의사결정을 내려야 하는 고도로 복잡한 시나리오입니다. 2D 물체 탐지에서는 Mask R-CNN 같은 뛰어난 패러다임이 등장했지만, 자율주행의 핵심 과제인 3D 물체 탐지와 BEV 맵 복원은 여전히 서로 다른 패러다임으로 처리되고 있었습니다.

BEVDet는 BEV 시맨틱 분할에 성공적으로 사용된 프레임워크를 3D 물체 탐지로 확장한 패러다임입니다. Image-view Encoder → View Transformer → BEV Encoder → Detection Head라는 4개의 모듈로 구성됩니다. 기존 모듈을 재사용하되, BEV 공간 전용 데이터 증강 전략과 Scale-NMS를 통해 성능을 크게 높였습니다.

**핵심 성과**
- **BEVDet-Tiny**: 31.2% mAP, 39.2% NDS (nuScenes val), 215.3 GFLOPs, **15.6 FPS** — FCOS3D 대비 9.2배 빠름
- **BEVDet-Base**: 39.3% mAP, 47.2% NDS — 모든 발표된 결과를 크게 상회

---

## Chapter 2: Related Works

**요약**

관련 연구는 크게 세 가지로 나뉩니다.

**핵심 개념**

- **비전 기반 2D 인식**: AlexNet에서 시작된 딥러닝 이미지 분류는 ResNet, SwinTransformer로 발전했고, Faster R-CNN, RetinaNet 등 강력한 탐지기로 이어졌습니다. 멀티태스크 학습(Mask R-CNN)은 효율성 면에서 주목받고 있습니다.
- **BEV 시맨틱 분할**: 자율주행에서 도로, 차선, 주차 공간 등을 BEV에서 분할하는 작업. PON, Lift-Splat-Shoot, VPN 등이 대표 방법입니다. 이 성공이 BEVDet의 설계에 직접적인 영감을 주었습니다.
- **비전 기반 3D 물체 탐지**: FCOS3D는 3D 탐지를 2D 문제로 접근, PGD는 깊이 예측을 개선했습니다. DETR3D는 어텐션 패턴으로 탐지하지만 추론 속도가 느립니다. BEV 기반 선행 연구로 Lift-Splat-Shoot 응용 연구들이 있었으나 LiDAR 의존성 등의 한계가 있었습니다.

---

## Chapter 3: Methodology

**요약**

BEVDet는 4개의 모듈형 컴포넌트로 설계되었습니다. 각 모듈은 독립적으로 교체 가능하며, 기존 검증된 구조를 재사용합니다.

### 3.1 네트워크 구조

**핵심 개념**

- **Image-view Encoder**: 입력 이미지에서 고수준 특징 추출. Backbone(ResNet, SwinTransformer)과 Neck(FPN-LSS)으로 구성. FPN-LSS는 1/32 해상도 특징을 1/16으로 업샘플링하여 연결합니다.

- **View Transformer**: 이미지 뷰 특징을 BEV 특징으로 변환하는 핵심 모듈. Lift-Splat-Shoot 방식을 사용하여, 각 픽셀의 깊이를 분류 방식으로 예측하고 3D 포인트 클라우드를 생성한 뒤 수직 방향(Z축)으로 풀링하여 BEV 특징을 만듭니다. 깊이 범위는 [1, 60]미터, 간격은 $1.25 \times r$ (r은 출력 해상도).

- **BEV Encoder**: BEV 공간에서 특징을 추가 인코딩. 구조는 Image-view Encoder와 유사하며(Backbone + FPN-LSS Neck), scale, orientation, velocity 같이 BEV 공간에서 정의되는 속성들을 정밀하게 인식합니다.

- **Detection Head**: BEV 특징 위에 CenterPoint의 1단계 헤드를 그대로 사용. 위치, 크기, 방향, 속도, 속성을 예측합니다.

**BEVDet 주요 구성 비교**

| 모듈 | BEVDet-Base | BEVDet-Tiny |
|------|------------|------------|
| 입력 해상도 | 1600×640 | 704×256 |
| Image-view Encoder | SwinTransformer-Base + FPN-LSS-512 | SwinTransformer-Tiny + FPN-LSS-512 |
| View Transformer | Lift-Splat-Shoot-64, -0.4×0.4 | Lift-Splat-Shoot-64, -0.4×0.4 |
| BEV Encoder | 2×Basic-{128,256,512} + FPN-LSS-512 | 2×Basic-{128,256,512} + FPN-LSS-256 |

### 3.2 맞춤형 데이터 증강 전략

**요약**

BEVDet에서 데이터 증강은 두 뷰 공간의 분리(decoupling)를 이해해야 제대로 설계할 수 있습니다.

**핵심 개념**

- **뷰 공간의 분리 (Isolated View Spaces)**: View Transformer는 이미지 뷰와 BEV를 픽셀 단위로 연결합니다. 이미지 뷰 픽셀 $\mathbf{p}_{image} = [x_i, y_i, 1]^T$는 깊이 $d$와 함께 3D 카메라 좌표로 변환됩니다:

$$\mathbf{p}_{camera} = \mathbf{I}^{-1}(\mathbf{p}_{image} * d)$$

**수식 설명**:
- **$\mathbf{p}_{camera}$**: 3D 카메라 좌표계에서의 점
- **$\mathbf{I}$**: 3×3 카메라 내부 파라미터 행렬 (초점 거리, 주점 등을 포함)
- **$\mathbf{p}_{image}$**: 이미지 평면에서의 2D 픽셀 좌표 (동차 좌표)
- **$d$**: 해당 픽셀의 깊이(거리)
- **$\mathbf{I}^{-1}$**: 역행렬을 이용해 2D 픽셀을 3D 공간으로 역투영

이미지에 증강 변환 $\mathbf{A}$를 적용하면 View Transformer에도 역변환 $\mathbf{A}^{-1}$을 적용해야 BEV 특징의 공간 일관성이 유지됩니다:

$$\mathbf{p}'_{camera} = \mathbf{I}^{-1}(\mathbf{A}^{-1}\mathbf{p}'_{image} * d) = \mathbf{p}_{camera}$$

**수식 설명**:
- **$\mathbf{A}$**: 이미지 뷰에 적용되는 증강 변환 행렬 (예: 뒤집기, 회전, 크롭 등을 3×3 행렬로 표현)
- **$\mathbf{A}^{-1}$**: 역변환 — 이미지 뷰 증강이 BEV 공간에 영향을 주지 않도록 상쇄
- 결과: 이미지 뷰 증강은 BEV 공간의 특징 분포를 바꾸지 않음 → 이미지 뷰에서는 자유롭게 복잡한 증강 가능

- **이미지 뷰 증강 (IDA)**: 플리핑, 스케일링, 회전, 크롭 등을 이미지 뷰에 적용. BEV 인코더가 없으면 정규화 효과가 있으나, BEV 인코더가 있으면 오히려 성능을 저하시킵니다.

- **BEV 공간 증강 (BDA)**: BEV 공간에서 플리핑, 스케일링(범위 [0.95, 1.05]), 회전(범위 [-22.5°, 22.5°])을 적용. View Transformer 출력 특징과 3D 탐지 타겟 모두에 동시에 적용하여 공간 일관성을 유지합니다. 과적합 방지에 가장 핵심적인 역할을 합니다.

- **Ablation 결과**:
  - 증강 없음(기준): 최고 23.0% mAP (epoch 4), 최종 17.4% (과적합 심각)
  - BDA만 적용: 최고 26.2% mAP (+3.2%)
  - IDA+BDA+BEV Encoder: **최고 31.6% mAP, 최종 31.2%** (과적합 -0.4%로 거의 해결)

### 3.3 Scale-NMS

**요약**

BEV 공간에서는 카테고리마다 물체 크기가 크게 다릅니다. 보행자나 교통 콘처럼 작은 물체는 BEV 출력 해상도(0.8m/voxel)보다 작아서, 중복 예측과 정답 예측이 겹치지 않아 기존 NMS가 중복 박스를 제거하지 못하는 문제가 발생합니다.

**핵심 개념**

- **기존 NMS의 문제**: 이미지 뷰에서는 모든 카테고리가 카메라 투영으로 인해 비슷한 공간 분포를 가져 고정 IOU 임계값이 잘 작동합니다. 그러나 BEV에서는 카테고리별 실제 크기 차이가 크고, 작은 물체는 IOU가 항상 0에 가까워 중복 제거가 안 됩니다.

- **Scale-NMS**: NMS 전에 각 카테고리의 물체 크기를 카테고리별 스케일링 팩터로 확대한 뒤 기존 NMS를 적용하고, 완료 후 다시 원래 크기로 복원합니다. 이를 통해 작은 물체들 사이의 IOU가 증가하여 중복 예측이 제거됩니다.

- **효과**: 보행자 +4.8% AP, 교통 콘 +7.5% AP, 전체 mAP 29.5% → 31.2% (+1.7%)

---

## Chapter 4: Experiment

**요약**

nuScenes 벤치마크(1,000 scenes, 6 cameras, 10 class, 1.4M 3D boxes)에서 종합 평가를 진행했습니다.

### 4.1 실험 설정

**핵심 개념**

- **평가 지표**: mAP (평균 정밀도), NDS (NuScenes Detection Score — 복합 지표), ATE (위치 오차), ASE (크기 오차), AOE (방향 오차), AVE (속도 오차), AAE (속성 오차)
- **학습 환경**: AdamW optimizer, 학습률 2e-4, 배치 크기 64, 8×NVIDIA RTX 3090, 20 에폭
- **데이터 증강**: 이미지 뷰에서 랜덤 플리핑/스케일링(s ∈ [Win/1600-0.06, Win/1600+0.11])/회전(r ∈ [-5.4°, 5.4°])/크롭, BEV 공간에서 랜덤 플리핑/회전([-22.5°, 22.5°])/스케일링([0.95, 1.05])

### 4.2 벤치마크 결과

**nuScenes val set 비교**

| 방법 | 입력 해상도 | GFLOPs | mAP | NDS | FPS |
|------|-----------|--------|-----|-----|-----|
| FCOS3D | 1600×900 | 2,008.2 | 29.5% | 37.2% | 1.7 |
| DETR3D | 1600×900 | 1,016.8 | 30.3% | 37.4% | 2.0 |
| PGD | 1600×900 | 2,223.0 | 33.5% | 40.9% | 1.4 |
| **BEVDet-Tiny** | **704×256** | **215.3** | **31.2%** | **39.2%** | **15.6** |
| **BEVDet-Base** | **1600×640** | **2,962.6** | **39.3%** | **47.2%** | **1.9** |

**핵심 개념**

- **BEVDet-Tiny**: FCOS3D 대비 1/8 입력 크기로 유사한 정확도를 달성하면서 **9.2배** 빠른 추론 속도. 연산량은 89% 절감.
- **BEVDet-Base**: 동일 추론 속도에서 PGD 대비 +5.8% mAP, +6.3% NDS로 모든 기존 결과를 상회.
- **BEV 공간의 이점**: 물체의 translation, scale, orientation, velocity 예측에서 특히 강점. 이미지 외형에 의존하는 attribute 예측에서는 이미지 뷰 방법 대비 약점.

**nuScenes test set**: BEVDet 42.2% mAP, 48.2% NDS로 vision-based 3D 탐지 1위 달성 (PGD 대비 +3.6% mAP, +3.4% NDS). 심지어 LiDAR 사전학습을 사용한 DD3D, DETR3D와 비교 가능한 수준.

### 4.3 Ablation Studies

**핵심 개념**

- **데이터 증강 ablation**: BDA가 IDA보다 훨씬 중요. IDA는 BDA 없이 단독 사용하면 오히려 성능 저하. BEV Encoder는 BDA와 함께일 때만 진가 발휘 (+1.7% mAP).

- **Resolution ablation**: 입력 해상도가 높을수록 정확도 향상 (704×256: 31.2% mAP → 1408×512: 36.0% mAP). BEV 인코더 해상도(0.8m vs 0.4m)도 정확도와 연산량 트레이드오프에 영향.

- **Backbone ablation**: SwinTransformer-Tiny가 ResNet-50 대비 +1.4% mAP, +1.3% NDS (704×256 해상도). ResNet-101은 1056×384 해상도에서 큰 수용 영역 덕분에 성능 향상 두드러짐.

- **추론 가속화**: View Transformer의 누적합 연산을 보조 인덱스(auxiliary index)로 대체하여 BEVDet-Tiny 추론 지연을 137ms → 64ms로 **53.3% 감소**. 카메라 파라미터가 고정된 추론 시간에 보조 인덱스를 사전 계산하여 가능.

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **Bird-Eye-View (BEV)** | 차량 위에서 내려다보는 시점. 자율주행에서 물체의 실제 위치·크기·속도·방향이 자연스럽게 정의되는 공간 |
| **View Transformer** | 이미지 픽셀 특징을 3D 공간으로 올린 뒤 BEV로 투영하는 모듈. Lift-Splat-Shoot 방식 사용 |
| **Lift-Splat-Shoot** | 깊이를 확률분포로 예측(Lift)하여 3D 포인트 클라우드 생성, 수직 방향으로 Pooling(Splat)하여 BEV 특징 획득 |
| **BEV Space Data Augmentation (BDA)** | BEV 공간에서 직접 플리핑·회전·스케일링을 적용하여 BEV 인코더의 과적합 방지 |
| **Scale-NMS** | BEV 공간의 카테고리별 크기 차이를 보정하여 NMS를 수행. 작은 물체(보행자, 콘)의 중복 제거 성능 향상 |
| **CenterPoint Head** | BEV 특징 맵에서 중심점 기반으로 3D 박스를 예측하는 헤드. LiDAR 기반 방법에서 검증된 구조를 카메라 기반에 재사용 |
| **mAP (mean Average Precision)** | 모든 카테고리의 평균 정밀도. 2D 탐지와 달리 BEV 평면에서 2D 중심 거리로 매칭 |
| **NDS (NuScenes Detection Score)** | mAP와 ATE, ASE, AOE, AVE, AAE를 종합한 nuScenes 공식 지표 |
| **FPN-LSS** | Lift-Splat-Shoot에서 제안된 Neck 구조. 1/32 해상도를 1/16으로 업샘플링하여 이어 붙임 |
| **Auxiliary Index** | View Transformer 내 누적합 연산을 병렬화하기 위해 사전 계산하는 인덱스. 추론 속도 53% 향상 |

---

## 결론 및 시사점

BEVDet는 BEV 시맨틱 분할 패러다임을 3D 물체 탐지로 성공적으로 확장한 강력하고 확장 가능한 프레임워크입니다. 기존 모듈을 재사용하면서도 두 가지 핵심 기여를 통해 성능 경계를 크게 넓혔습니다:

1. **BEV 전용 데이터 증강 (BDA)**: View Transformer의 뷰 공간 분리 특성을 활용하여 BEV 공간에서 독립적인 증강을 설계. 과적합을 효과적으로 방지하고 +8.6% mAP 성능 향상.

2. **Scale-NMS**: BEV 공간의 카테고리별 크기 분포 차이를 보정하여 기존 NMS의 한계를 극복. 소형 물체 탐지 성능을 크게 향상.

**실무적 시사점**:
- **자율주행 시스템 설계**: BEVDet-Tiny는 215 GFLOPs, 15.6 FPS로 실시간 처리 가능. 온보드 추론에 적합.
- **멀티태스크 확장성**: BEV 공간을 공유하므로 3D 탐지와 BEV 시맨틱 분할을 단일 프레임워크에서 처리하는 멀티태스크 학습으로 자연스럽게 확장 가능.
- **향후 과제**: 물체 attribute(예: 자전거 탑승 여부) 예측은 이미지 외형 정보가 필요하여 BEV 방식이 약점. 이미지 뷰와 BEV 뷰의 결합이 유망한 방향.
