---
title: "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation"
date: 2026-04-19T22:56:00+09:00
draft: false
categories: ["Papers", "Computer Vision"]
tags: ["Point Cloud", "3D Classification", "3D Segmentation", "Deep Learning"]
---

## 개요
- **저자**: Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas (Stanford University)
- **발행년도**: 2017 (CVPR 2017)
- **주요 내용**: 포인트 클라우드(point cloud)를 복셀이나 이미지로 변환하지 않고 원시 형태(raw point set)로 직접 처리하는 최초의 딥러닝 아키텍처. 3D 물체 분류, 파트 분할, 장면 의미 분할을 단일 통합 아키텍처로 처리한다.

## 한계 극복
이 논문이 기존 연구의 어떤 한계를 극복하기 위해 작성되었는지 설명합니다.
- **기존 한계 1 — 복셀 변환의 비효율성**: VoxNet, 3DShapeNets 등 기존 방법은 포인트 클라우드를 3D 복셀 격자로 변환한 뒤 3D CNN을 적용. 이 과정에서 데이터가 불필요하게 방대해지고 해상도 제한(quantization artifacts)이 발생.
- **기존 한계 2 — 멀티뷰 렌더링 의존성**: 멀티뷰 CNN 방법은 3D 데이터를 2D 이미지 여러 장으로 렌더링한 뒤 2D CNN 적용. 장면 이해·완성 등 확장에 한계가 있음.
- **기존 한계 3 — 포인트 집합의 비순서성 무시**: 포인트 클라우드는 순서 없는 집합(unordered set)이라 N개 점이면 N! 가지 순열에 불변해야 하는데, 기존 방법은 이 성질을 제대로 다루지 못함.
- **이 논문의 접근 방식**: MaxPooling이라는 대칭 함수(symmetric function)를 집계 연산으로 사용해 입력 순열 불변성을 수학적으로 보장. MLP를 각 점에 독립적으로 적용해 O(N) 복잡도 달성.

## 목차
- Chapter 1: Introduction
- Chapter 2: Related Work
- Chapter 3: Problem Statement
- Chapter 4: Deep Learning on Point Sets
- Chapter 5: Experiments
- Chapter 6: Conclusion

## Chapter 1: Introduction

**요약**

포인트 클라우드는 LiDAR 센서나 RGB-D 카메라가 생성하는 3D 기하 데이터의 핵심 형식이다. 기존 딥러닝 방법들은 이를 처리하기 위해 복셀 격자나 멀티뷰 이미지로 변환했는데, 이 변환 과정이 데이터를 불필요하게 크게 만들고 자연스러운 불변성(invariance)을 망가뜨린다.

PointNet은 포인트 클라우드를 순서 없는 점의 집합으로 직접 처리한다. 핵심 설계 아이디어는 세 가지다: (1) MaxPooling 대칭 함수로 순열 불변성 보장, (2) T-Net(Joint Alignment Network)으로 강체 변환에 대한 불변성 달성, (3) 분류와 분할을 동일한 백본으로 처리하는 통합 아키텍처.

**핵심 개념**
- **순열 불변성(Permutation Invariance)**: N개 점의 집합 {p₁, …, pₙ}은 어떤 순서로 입력해도 같은 결과가 나와야 한다. N!개의 순열 모두에 불변해야 하는 조건.
- **대칭 함수(Symmetric Function)**: 입력 순서에 무관한 함수. +, max, ×가 대표적인 예. PointNet은 MAX를 선택함.
- **Critical Point Set (임계 점 집합, $\mathcal{C}_S$)**: MaxPooling에 기여한 점들의 집합. 이 집합만 있으면 $f(S)$가 결정되므로 형상의 "뼈대(skeleton)"를 표현함.

## Chapter 2: Related Work

**요약**

기존 3D 딥러닝 방법을 네 갈래로 정리한다.

- **Volumetric CNN**: 3D 복셀에 3D Conv 적용 (VoxNet, 3DShapeNets). 해상도와 메모리 한계.
- **Multiview CNN**: 여러 시점의 2D 이미지로 렌더링 후 2D CNN 적용. 장면 이해로 확장 어려움.
- **Spectral CNN**: 메시(mesh) 위의 그래프 스펙트럼 CNN. 비등방성(non-isometric) 형상에 제한.
- **Feature-based DNN**: 포인트 클라우드에서 수작업 특징을 추출한 뒤 FC 분류기 적용.

PointNet은 어떤 전처리도 없이 원시 포인트 클라우드를 직접 처리하는 첫 번째 방법이다.

## Chapter 3: Problem Statement

**요약**

입력은 3D 포인트 집합 $\{P_i \mid i=1,\ldots,n\}$이며, 각 점 $P_i$는 $(x, y, z)$ 좌표 (필요 시 색상, 법선 등 추가 채널 포함)를 갖는다.

- **분류(Classification)**: 전체 포인트 클라우드에 대해 k개 클래스 점수 출력
- **분할(Segmentation)**: 각 점마다 m개 의미 카테고리의 점수 $n \times m$ 출력

## Chapter 4: Deep Learning on Point Sets

**요약**

PointNet 아키텍처의 수학적 기반과 구조를 설명한다.

### 4.1 포인트 집합의 세 가지 성질

1. **비순서성(Unordered)**: 포인트 집합은 배열이 아닌 집합. 순열에 불변해야 함.
2. **점 간 상호작용(Interaction among points)**: 인접 점들이 의미있는 부분 구조를 형성. 지역 구조 포착 필요.
3. **변환 불변성(Invariance under transformations)**: 회전·이동해도 분류/분할 결과가 같아야 함.

### 4.2 PointNet 아키텍처

**핵심 개념**
- **공유 MLP (Shared MLP)**: 모든 점에 동일한 가중치의 MLP를 독립적으로 적용. Pointwise feature extraction.
- **MaxPooling (대칭 집계)**: n개 점의 피처를 원소별 최대값으로 압축해 고정 길이 전역 특징 벡터(1024-dim) 생성. 순열 불변 보장.
- **T-Net (Joint Alignment Network / Input Transform)**: 3×3 변환 행렬을 예측해 입력 점에 곱함. 데이터를 정규 자세(canonical pose)로 정렬.
- **Feature Transform**: 64×64 특징 변환 행렬도 T-Net으로 예측. 고차원 변환이라 정규화 손실 추가.

**수식 예제 — 대칭 함수 근사**

$$f(\{x_1, \ldots, x_n\}) \approx g(h(x_1), \ldots, h(x_n))$$

**수식 설명**
- **$f$**: 포인트 집합 전체에 대한 목표 함수 (분류 점수, 분할 레이블 등)
- **$h$**: 각 점에 독립적으로 적용되는 MLP (공유 가중치). $h: \mathbb{R}^3 \to \mathbb{R}^K$
- **$g$**: 대칭 집계 함수. PointNet에서는 원소별 MAX (MaxPooling)
- 전체 의미: "각 점을 MLP로 피처 추출한 뒤 MaxPooling으로 집계하면 어떤 순열 불변 집합 함수도 근사할 수 있다"

**수식 예제 — Feature Transform 정규화 손실**

$$L_{reg} = \|I - AA^T\|_F^2$$

**수식 설명**
- **$A$**: T-Net이 예측하는 64×64 피처 변환 행렬
- **$I$**: 단위 행렬 (identity matrix)
- **$\|\cdot\|_F$**: Frobenius 노름 (행렬 원소 제곱합의 제곱근)
- **직관**: A가 직교 행렬(orthogonal matrix)에 가까워지도록 강제. 직교 변환은 정보를 손실하지 않으므로 학습 안정성이 향상됨.
- softmax 분류 손실에 0.001 가중치로 더해진다.

### 4.3 이론적 분석

**Theorem 1 (Universal Approximation)**: $f: 2^{\mathbb{R}^N} \to \mathbb{R}$이 Hausdorff 거리에 대한 연속 집합 함수이면, $\forall \epsilon > 0$에 대해 $|f(S) - \gamma \circ \text{MAX}\{h(x_i)\}| < \epsilon$을 만족하는 연속함수 $h$와 $\gamma$가 존재한다. (MaxPooling 레이어 크기 K가 충분히 크면)

**Theorem 2 (Bottleneck & Stability)**: $\mathbf{u} = \text{MAX}_{x_i \in S}\{h(x_i)\}$, $f = \gamma \circ \mathbf{u}$이면:
- (a) $\forall S, \exists \mathcal{C}_S, \mathcal{N}_S \subseteq \mathcal{X}$: $\mathcal{C}_S \subseteq T \subseteq \mathcal{N}_S$인 임의의 T에 대해 $f(T) = f(S)$
- (b) $|\mathcal{C}_S| \leq K$

이는 PointNet이 핵심 점들의 집합으로 형상을 요약함을 증명한다. Critical point set $\mathcal{C}_S$의 점들이 형상의 뼈대(skeleton)를 구성하며, 여기에 속하지 않는 점들을 제거해도 $f(S)$가 변하지 않는다.

## Chapter 5: Experiments

**요약**

세 가지 과제에서 PointNet의 성능을 검증한다.

### 5.1 3D 물체 분류 (ModelNet40)

- **데이터셋**: ModelNet40 — 40개 카테고리, 12,311개 CAD 모델 (9,843 train / 2,468 test)
- **입력**: 메시 면 면적 비례로 1,024개 점 균일 샘플링
- **결과**:

| 방법 | 입력 | 평균 클래스 정확도 | 전체 정확도 |
|------|------|------|------|
| 3DShapeNets | 볼륨 | 77.3% | 84.7% |
| VoxNet | 볼륨 | 83.0% | 85.9% |
| Subvolume | 볼륨 | 86.0% | **89.2%** |
| MVCNN | 이미지 | **90.1%** | - |
| **PointNet (Ours)** | **포인트** | **86.2%** | **89.2%** |

3D 입력 방법 중 최고 성능 달성. 멀티뷰 대비 작은 차이는 세밀한 기하 디테일 손실 때문.

### 5.2 3D 파트 분할 (ShapeNet Part)

- **데이터셋**: ShapeNet Part — 16개 카테고리, 16,881개 형상, 50개 파트
- **메트릭**: mIoU (포인트별)
- **결과**: mIoU **83.7%** — 기존 최고 대비 2.3% 향상

### 5.3 장면 의미 분할 (Stanford 3D Semantic Parsing)

- **데이터셋**: Matterport 스캐너로 취득한 6개 공간, 271개 방, 13개 의미 카테고리
- **결과**: mIoU **47.71%**, 전체 정확도 78.62%
- 기존 baseline 대비 mIoU 2배 이상 향상 (20.12 → 47.71)

### 5.4 시간·공간 복잡도

| 방법 | 파라미터 수 | FLOPs/샘플 |
|------|------|------|
| PointNet (vanilla) | 0.8M | 148M |
| PointNet | 3.5M | 440M |
| Subvolume | 16.6M | 3,633M |
| MVCNN | 60.0M | 62,057M |

PointNet은 복잡도가 입력 점 수 N에 선형(O(N))이며, GPU(1080X) 기준 초당 100만 점 이상 처리 가능.

**핵심 개념**
- **순열 불변 방법 비교**: MLP(정렬 없음) < MLP(정렬) < LSTM < Attention sum < Average pooling < **Max pooling (87.1% 최고)**
- **T-Net 효과**: 입력 변환만 추가해도 0.8% 향상, 피처 변환까지 추가하면 최고 성능
- **강건성**: 포인트 50% 결손 시 정확도 3.7%만 하락 (VoxNet은 40.3% 하락)

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **MaxPooling 집계** | 순열 불변성을 보장하는 대칭 함수. 각 차원에서 최대값만 취해 N개 점 → 1개 전역 벡터로 압축 |
| **공유 MLP** | 모든 점에 같은 가중치 적용. 점마다 독립 처리라 연산량 O(N) |
| **T-Net** | 3×3 또는 64×64 변환 행렬을 예측하는 미니 네트워크. STN(Spatial Transformer Network) 아이디어를 3D로 확장 |
| **Critical Point Set $\mathcal{C}_S$** | MaxPooling 활성화를 지배하는 희소한 점들의 집합. 객체의 기하학적 뼈대 표현 |
| **Upper-bound Shape $\mathcal{N}_S$** | $\mathcal{C}_S \subseteq T \subseteq \mathcal{N}_S$를 만족하는 최대 집합. 같은 전역 피처를 만드는 가장 큰 점 집합 |
| **Local+Global 결합** | 분할 네트워크는 각 점의 지역 피처(1088-dim)에 전역 피처(1024-dim)를 concat해 per-point 예측 |

## 결론 및 시사점

PointNet은 포인트 클라우드를 직접 처리하는 딥러닝의 시초 논문으로, 이후 PointNet++ (지역 구조 계층화), VoxelNet (PointNet + RPN), PointPillars (Pillar 단위 PointNet 인코딩), CenterPoint 등 LiDAR 탐지의 모든 후속 연구에 피처 인코딩 모듈로 활용된다.

**핵심 한계**:
- 지역 이웃 구조를 명시적으로 모델링하지 않음 → PointNet++에서 FPS + Ball Query로 해결
- 대규모 장면 처리 시 점 수 증가에 따른 연산 부담

**자율주행 관련 시사점**:
- VoxelNet은 각 복셀 내 점들에 PointNet 인코더를 적용
- PointPillars는 수직 기둥(pillar)을 PointNet으로 인코딩
- BEVFusion, CenterPoint 등 현대 LiDAR 탐지기의 포인트 인코딩 설계 철학이 이 논문에서 출발
- 합성 데이터(Blensor 시뮬레이터)로 생성된 부분 스캔에서도 성능 저하 5.3%로 견고하여 도메인 갭 연구에도 시사점 제공
