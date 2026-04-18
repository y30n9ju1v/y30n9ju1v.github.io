---
title: "EmerNeRF: Emergent Spatial-Temporal Scene Decomposition via Self-Supervision"
date: 2026-04-18T10:30:00+09:00
draft: false
categories: ["Papers"]
tags: ["NeRF", "Autonomous Driving", "Dynamic Scene Reconstruction", "Self-Supervised Learning", "Scene Flow"]
---

## 개요

- **저자**: Jiawei Yang, Boris Ivanovic, Or Litany, Xinshuo Weng, Seung Wook Kim, Boyi Li, Tong Che, Danfei Xu, Sanja Fidler, Marco Pavone, Yue Wang
- **소속**: USC, NVIDIA Research, Georgia Tech, University of Toronto, Stanford, Technion
- **발행년도**: 2023 (arXiv:2311.02077)
- **주요 내용**: 자율주행 동적 장면을 위한 4D(공간+시간) NeRF 표현 학습. 정적/동적 분해와 씬 플로우 추정을 **완전 자기지도(self-supervision)** 방식으로 수행

## 한계 극복

- **기존 한계 1 — GT 어노테이션 의존**: 기존 동적 장면 분리 방법들(UniSim, NSG 등)은 동적 객체 분할 및 추적에 GT 바운딩 박스 어노테이션이 필요
- **기존 한계 2 — 사전 학습 Optical Flow 모델 의존**: SUDS, NSFP 등은 씬 플로우 추정에 별도의 사전 학습 플로우 모델을 요구하여 계산 비용이 크고 오류가 누적됨
- **기존 한계 3 — 다시점 동기화 가정**: 많은 방법들이 여러 시점에서 동기화된 영상을 요구하여 실제 AV 환경에 적용이 어려움
- **이 논문의 접근 방식**: 정적 필드 $\mathcal{S}$와 동적 필드 $\mathcal{D}$, 그리고 플로우 필드 $\mathcal{V}$를 결합한 하이브리드 NeRF를 구성하고, RGB 재구성 손실만으로 씬 플로우가 자연스럽게 **emergent(창발)**되도록 설계

## 목차

- Chapter 1: Introduction
- Chapter 2: Related Work
- Chapter 3: Self-Supervised Spatial-Temporal Neural Fields
  - 3.1 Scene Representations
  - 3.2 Emergent Scene Flow
  - 3.3 Vision Transformer Feature Lifting
  - 3.4 Optimization
- Chapter 4: Experiments
  - 4.1 Rendering
  - 4.2 Flow Estimation
  - 4.3 Leveraging Foundation Model Features
- Chapter 5: Conclusion

---

## Chapter 1: Introduction

**요약**

자율주행 차량은 카메라, LiDAR 등 다양한 센서로 주변 환경을 관측합니다. 기존 방법들은 정적 장면은 NeRF로 잘 표현하지만, 빠르게 움직이는 차량·보행자 같은 동적 객체가 많은 환경에서는 어려움을 겪습니다. 그 이유는 **단일 traversal(한 번의 주행)**에서 각 공간 포인트가 단 한 번만 관측되어 멀티-뷰 일관성 가정이 성립하지 않기 때문입니다.

EmerNeRF는 이 문제를 해결하기 위해:
1. 장면을 정적 필드와 동적 필드로 자동 분리
2. 씬 플로우를 별도 감독 없이 창발적으로 학습
3. 2D 비전 파운데이션 모델(DINOv2 등) 특징을 4D 공간으로 리프팅

**핵심 개념**

- **NeRF (Neural Radiance Field)**: 3D 공간의 각 좌표에서 색상과 밀도를 예측하는 신경망. 여러 이미지로부터 3D 장면을 암묵적으로 표현
- **4D 표현**: 3D 공간(x, y, z) + 시간(t). 동적 객체를 시간 축으로 추적 가능
- **Self-supervision**: 레이블 없이 입력 데이터 자체(예: RGB 재구성)로부터 지도 신호를 얻는 학습 방식
- **Emergent flow**: 플로우를 직접 감독하지 않아도 씬 재구성 과정에서 자연스럽게 생성되는 씬 플로우

---

## Chapter 2: Related Work

**요약**

동적 장면 NeRF 연구들은 크게 두 흐름으로 나뉩니다. Nerfies, HyperNeRF처럼 모든 관측을 정준 공간(canonical space)으로 변형(deformation)하는 방법과, D²NeRF, SUDS처럼 정적/동적을 명시적으로 분리하는 방법입니다.

**핵심 개념**

- **D²NeRF**: 모노큘러 영상에서 정적/동적 분리를 수행하나, 실외 장면의 복잡한 하이퍼파라미터 튜닝에 취약
- **SUDS**: 다중 traversal 로그에 최적화되어 있고, 사전 학습된 Optical Flow에 크게 의존
- **NeuralGroundplan**: 단일 카메라 기반이라 멀티-센서 AV 환경에 부적합
- **NSFP (Neural Scene Flow Prior)**: 씬 플로우를 Chamfer Loss로 최적화하지만 타임스텝마다 별도 최적화가 필요해 매우 느림

---

## Chapter 3: Self-Supervised Spatial-Temporal Neural Fields

### 3.1 Scene Representations

**요약**

EmerNeRF는 4D 장면을 **정적 필드** $\mathcal{S}$와 **동적 필드** $\mathcal{D}$로 분리합니다. 두 필드 모두 해시 그리드(Müller et al., 2022)로 파라미터화됩니다.

$$\mathbf{g}_s, \sigma_s = g_s(\mathcal{H}_s(\mathbf{x})) \qquad \mathbf{g}_d, \sigma_d = g_d(\mathcal{H}_d(\mathbf{x}, t)) \tag{1}$$

**수식 설명**
- **$\mathbf{x} = (x, y, z)$**: 3D 공간 좌표
- **$t$**: 타임스텝
- **$\mathcal{H}_s(\mathbf{x})$**: 정적 필드용 해시 그리드 인코딩 (시간 독립)
- **$\mathcal{H}_d(\mathbf{x}, t)$**: 동적 필드용 해시 그리드 인코딩 (시간 의존)
- **$g_s, g_d$**: 경량 MLP — 해시 인코딩을 feature $\mathbf{g}$와 밀도 $\sigma$로 변환
- **$\sigma_s, \sigma_d$**: 각각 정적·동적 포인트의 밀도 (클수록 해당 위치에 물체가 있을 가능성 높음)

멀티헤드 예측으로 색상($c_s, c_d$), 하늘($c_\text{sky}$), 그림자($\rho$)를 별도 MLP로 예측:

$$\mathbf{c}_s = \text{MLP}_\text{color}(\mathbf{g}_s, \gamma(\mathbf{d})) \qquad \mathbf{c}_d = \text{MLP}_\text{color}(\mathbf{g}_d, \gamma(\mathbf{d})) \tag{2}$$

$$\mathbf{c}_\text{sky} = \text{MLP}_\text{color\_sky}(\gamma(\mathbf{d})) \qquad \rho = \text{MLP}_\text{shadow}(\mathbf{g}_d) \tag{3}$$

렌더링 시 정적·동적 필드를 밀도 기반 가중치로 합산:

$$\mathbf{c} = \frac{\sigma_s}{\sigma_s + \sigma_d} \cdot (1 - \rho) \cdot \mathbf{c}_s + \frac{\sigma_d}{\sigma_s + \sigma_d} \cdot \mathbf{c}_d \tag{4}$$

**수식 설명 (4)**
- 분자 $\sigma_s$, $\sigma_d$는 각 필드의 기여 비중을 결정
- $(1-\rho)$: 동적 객체가 드리운 그림자를 정적 배경에 반영하는 그림자 계수

최종 픽셀 색상 $\hat{C}$는 K개 샘플의 누적 투과율로 계산:

$$\hat{C} = \sum_{i=1}^{K} T_i \alpha_i \mathbf{c}_i + \left(1 - \sum_{i=1}^{K} T_i \alpha_i\right) \mathbf{c}_\text{sky} \tag{5}$$

**수식 설명 (5)**
- **$T_i = \prod_{j=1}^{i-1}(1 - \alpha_j)$**: 누적 투과율. i번째 샘플에 도달하기까지 앞의 모든 샘플을 통과한 빛의 비율
- **$\alpha_i = 1 - \exp(-\sigma_i(\mathbf{x}_{i+1} - \mathbf{x}_i))$**: i번째 샘플의 불투명도
- **$\mathbf{c}_\text{sky}$**: 아무 물체도 없는 배경(하늘)의 색상

동적 필드가 필요 이상으로 밀도를 생성하지 않도록 정규화:

$$\mathcal{L}_{\sigma_d} = \mathbb{E}(\sigma_d) \tag{6}$$

**핵심 개념**

- **해시 그리드 인코딩**: 3D/4D 좌표를 고차원 특징 벡터로 빠르게 변환하는 데이터 구조 (Instant-NGP 방식)
- **밀도(density) $\sigma$**: 해당 위치에 물체가 존재할 확률을 나타내는 스칼라. 렌더링 시 가중치로 사용
- **Shadow head**: 동적 객체(예: 차량)가 지면에 드리우는 그림자를 별도로 모델링하여 정적 배경 재구성 품질 향상

### 3.2 Emergent Scene Flow

**요약**

씬 플로우 필드 $\mathcal{V} := \mathcal{H}_v(\mathbf{x}, t)$는 각 3D 포인트의 다음 타임스텝 위치를 예측합니다:

$$\mathbf{v} = \text{MLP}_v(\mathcal{H}_v(\mathbf{x}, t)) \qquad \mathbf{x}' = \mathbf{x} + \mathbf{v} \tag{7}$$

이 플로우를 이용해 인접 타임스텝의 특징을 집계:

$$\mathbf{g}'_d = 0.25 \cdot g_d(\mathcal{H}_d(\mathbf{x} + \mathbf{v}_b, t-1)) + 0.5 \cdot g_d(\mathcal{H}_d(\mathbf{x}, t)) + 0.25 \cdot g_d(\mathcal{H}_d(\mathbf{x} + \mathbf{v}_f, t+1)) \tag{8}$$

**수식 설명 (8)**
- **$\mathbf{v}_f$**: 순방향 플로우 — 현재 포인트가 다음 타임스텝에서 어디 있는지
- **$\mathbf{v}_b$**: 역방향 플로우 — 현재 포인트가 이전 타임스텝에서 어디 있었는지
- 인접 프레임의 feature를 가중 평균하여 동적 객체의 시간적 일관성을 강화
- 이 집계를 통해 플로우 필드가 **별도 감독 없이** RGB 재구성 손실만으로 학습됨

**핵심 개념**

- **Temporal feature aggregation**: 이전·현재·다음 타임스텝의 특징을 플로우로 정렬하여 합산. 동적 객체의 외관 예측을 더 안정적으로 만듦
- **Cycle consistency loss $\mathcal{L}_\text{cycle}$**: 순방향 + 역방향 플로우를 적용하면 원래 위치로 돌아와야 한다는 제약. 플로우 품질을 간접적으로 향상

### 3.3 Vision Transformer Feature Lifting

**요약**

DINOv2 같은 Vision Transformer 기반 모델의 2D 특징을 4D 공간으로 리프팅합니다. 이 과정에서 Transformer의 **Positional Embedding(PE) 패턴** 문제를 발견하고 해결합니다.

PE 패턴은 2D 이미지 좌표에 고정되어 있어 3D 뷰포인트가 바뀌어도 같은 위치에 나타나는 아티팩트입니다. 이를 제거하기 위해 학습 가능한 2D PE 맵 $\mathcal{U}$를 도입:

$$\hat{F} = \underbrace{\sum_{i=1}^{K} T_i \alpha_i \mathbf{f}_i + \left(1 - \sum_{i=1}^{K} T_i \alpha_i\right) \mathbf{f}_\text{sky}}_{\text{Volume-rendered PE-free feature}} + \underbrace{\text{MLP}_\text{PE}(\text{interp}((u,v), \mathcal{U}))}_{\text{PE feature}} \tag{11}$$

**수식 설명 (11)**
- 첫 번째 항: 볼륨 렌더링으로 얻은 **PE-free** 3D 특징 (뷰포인트에 독립적)
- 두 번째 항: 학습 가능한 2D 맵 $\mathcal{U}$에서 픽셀 좌표 $(u,v)$로 보간하여 얻은 PE 패턴
- 두 항의 합이 원래 DINOv2 특징을 복원하도록 학습 → PE-free 3D 특징은 PE 오염이 제거됨

**핵심 개념**

- **Positional Embedding (PE) 패턴**: Transformer가 학습한 위치 정보가 이미지에 격자 모양 아티팩트로 나타나는 현상. 멀티뷰 일관성을 깨뜨려 3D 인식 성능을 저하시킴
- **Feature lifting**: 2D 이미지에서 추출한 특징(semantic, appearance)을 3D/4D 공간으로 투영하는 과정
- **Few-shot perception**: 10%의 레이블 데이터만으로 3D semantic occupancy 예측 수행 가능

### 3.4 Optimization

**요약**

전체 손실 함수는 픽셀 레이와 LiDAR 레이 손실의 합:

$$\mathcal{L} = \underbrace{\mathcal{L}_\text{rgb} + \mathcal{L}_\text{sky} + \mathcal{L}_\text{shadow} + \mathcal{L}_{\sigma_d(\text{pixel})} + \mathcal{L}_\text{cycle} + \mathcal{L}_\text{feat}}_{\text{for pixel rays}} + \underbrace{\mathcal{L}_\text{depth} + \mathcal{L}_{\sigma_d(\text{LiDAR})}}_{\text{for LiDAR rays}} \tag{12}$$

**핵심 개념**

- **$\mathcal{L}_\text{rgb}$**: 렌더링된 색상과 GT 픽셀 색상 간의 L2 손실
- **$\mathcal{L}_\text{depth}$**: LiDAR 포인트를 이용한 Line-of-Sight 깊이 손실 — 단봉형(unimodal) 밀도 분포 유도
- **$\mathcal{L}_\text{cycle}$**: 플로우 사이클 일관성 손실 (forward → backward 적용 시 원점 복귀)
- **$\mathcal{L}_{\sigma_d}$**: 동적 밀도 정규화 — 필요할 때만 동적 필드가 활성화되도록 유도

---

## Chapter 4: Experiments

### 4.1 Rendering 결과

**NOTR (NeRF On-The-Road) 데이터셋**을 새로 구축하여 평가. Waymo Open Dataset에서 120개 시퀀스 선별:
- Static-32, Dynamic-32, Diverse-56 세 가지 split
- 다양한 조명·날씨·속도 조건 포함

| 방법 | Dynamic-32 (Scene Recon PSNR) | Dynamic-32 (NVS PSNR) |
|------|-------------------------------|------------------------|
| D²NeRF | 24.35 | 24.17 |
| HyperNeRF | 25.17 | 24.71 |
| **EmerNeRF (Ours)** | **28.87** | **27.62** |

### 4.2 Flow Estimation 결과

| 방법 | EPE3D (m) ↓ | Acc₅ (%) ↑ | Acc₁₀ (%) ↑ |
|------|-------------|------------|-------------|
| NSFP | 0.365 | 51.76 | 67.36 |
| **EmerNeRF (Ours)** | **0.014** | **93.92** | **96.27** |

NSFP 대비 EPE3D에서 **26배** 이상의 오차 감소. 명시적 플로우 감독 없이도 압도적 성능.

### 4.3 Foundation Model Features 결과

PE 제거 효과 (3D Semantic Occupancy 평균 micro-accuracy):

| 설정 | DINOv1 | DINOv2 |
|------|--------|--------|
| PE 있음 | 43.12% | 38.73% |
| **PE 제거** | **55.02%** | **63.21%** |
| 향상 | +27.60% | +63.22% |

DINOv2에서 PE 제거 시 micro-accuracy가 평균 **37.50%** 상대적 향상.

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **Static Field $\mathcal{S}$** | 시간 불변 배경(건물, 도로 등)을 표현하는 3D 해시 그리드 NeRF |
| **Dynamic Field $\mathcal{D}$** | 시간 의존적 동적 객체(차량, 보행자)를 표현하는 4D 해시 그리드 NeRF |
| **Flow Field $\mathcal{V}$** | 각 동적 포인트의 3D 변위 벡터를 예측하는 해시 그리드 + MLP |
| **Dynamic Density Regularization** | 동적 필드가 정적 배경을 잘못 모델링하지 않도록 밀도를 최소화하는 정규화 |
| **Emergent Flow** | 플로우 GT 없이 RGB 재구성만으로 플로우가 창발적으로 학습되는 현상 |
| **PE 패턴 제거** | ViT의 위치 임베딩 아티팩트를 학습 가능한 2D 맵으로 분리·제거 |
| **NOTR Benchmark** | 120개의 다양한 자율주행 시퀀스를 담은 새로운 동적 NeRF 벤치마크 |

---

## 결론 및 시사점

EmerNeRF는 **그라운드 트루스 어노테이션 없이** 자율주행 동적 장면의 정적/동적 분해, 씬 플로우 추정, 의미론적 특징 이해를 동시에 달성하는 최초의 통합 프레임워크입니다.

**실무적 시사점:**

1. **센서 시뮬레이션 고품질화**: 정적·동적 분리로 배경 변경이나 동적 객체 추가/제거가 용이해져 합성 데이터 생성 파이프라인에 바로 활용 가능
2. **어노테이션 비용 절감**: GT 박스나 플로우 레이블 없이도 동적 장면을 표현하므로 레이블링 비용을 크게 절감
3. **Few-shot 3D 인식**: PE-free DINOv2 특징으로 10% 레이블만으로도 3D occupancy 예측 가능 — 자율주행 인식 파이프라인의 반자동 레이블링에 응용 가능
4. **PE 문제의 보편성**: ViT 기반 모델을 3D 공간에 활용하는 모든 작업에서 PE 패턴 제거가 필요함을 시사
