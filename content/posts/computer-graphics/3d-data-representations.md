---
title: "3D 데이터 표현 방식 총정리: Mesh부터 3DGS까지"
date: 2026-04-30T10:05:00+09:00
draft: false
tags: ["컴퓨터 그래픽스", "3D", "폴리곤", "메시", "포인트 클라우드", "복셀", "SDF", "NeRF", "3DGS", "Gaussian Splatting"]
categories: ["컴퓨터 그래픽스"]
description: "Polygon Mesh, Point Cloud, Voxel, SDF, NeRF, 3D Gaussian Splatting까지 — 각 3D 표현 방식의 수학적 정의, 파이프라인 내에서의 역할, 장단점을 비교합니다."
---

## 대상 독자

선형대수와 기초 컴퓨터 그래픽스 파이프라인을 알고 있으며, 자율주행·3D 비전·뉴럴 렌더링 분야에 입문하려는 개발자/연구자를 위한 글입니다.

---

## 1. Polygon Mesh

**표면을 삼각형의 집합으로 근사**하는 방식입니다. 3D 물체를 정점(Vertex), 엣지(Edge), 페이스(Face)의 구조체로 표현하며, 실시간 렌더링 파이프라인의 사실상 표준입니다.

### 왜 삼각형인가?

공간상의 세 점은 항상 유일한 평면을 정의합니다. 사각형(Quad)은 네 꼭짓점이 같은 평면에 없을 수 있어 렌더링 시 분할(Tessellation)이 필요합니다. 삼각형은 Rasterization 파이프라인에서 Barycentric Coordinate를 이용한 보간이 간단하고, 현대 GPU의 고정 함수 유닛이 삼각형 처리에 최적화되어 있습니다.

### 핵심 개념

- **Vertex Buffer / Index Buffer**: 정점 데이터를 중복 없이 저장하고 인덱스로 참조합니다.
- **Normal Vector**: 표면의 법선 벡터. 라이팅 계산(Phong, PBR 등)의 기반입니다.
- **UV Mapping**: 3D 표면을 2D 텍스처 공간에 대응시키는 파라미터화입니다.
- **LOD (Level of Detail)**: 카메라 거리에 따라 폴리곤 수를 동적으로 조절해 렌더링 비용을 줄입니다.

**장점:** GPU 파이프라인과 완전히 통합되어 있어 실시간 렌더링(60fps+)이 가능합니다. 기존 DCC 툴(Blender, Maya)과의 호환성도 뛰어납니다.

**단점:** 표면(Shell)만 표현합니다. 연기·불꽃처럼 볼류메트릭한 현상, 또는 위상 변화(topology change)를 다루기 어렵습니다.

---

## 2. Point Cloud

**위치 좌표 (x, y, z)와 선택적 속성(RGB, 법선, 반사율 등)을 가진 비정형 점 집합**입니다. 위상(topology) 정보 없이 공간 샘플만 존재합니다.

### 주요 취득 방법

| 방법 | 원리 | 특징 |
| :--- | :--- | :--- |
| **LiDAR** | ToF(Time-of-Flight) 레이저 | 희소(sparse), 노이즈 적음, 실시간 |
| **RGB-D 카메라** | 구조광 또는 ToF | 밀도 높음, 실내 근거리에 적합 |
| **SfM (Structure-from-Motion)** | 다중 뷰 기하학 | 카메라만으로 복원, COLMAP 등 오픈소스 활용 |
| **MVS (Multi-View Stereo)** | SfM 이후 치밀화 단계 | Dense Point Cloud 생성 |

### 활용 파이프라인

자율주행에서는 LiDAR Point Cloud를 직접 신경망의 입력으로 사용합니다. PointNet(Qi et al., 2017)은 점 집합의 순서 불변성(permutation invariance)을 MLP와 Symmetric Function으로 처리한 대표적인 아키텍처입니다.

**장점:** 센서 원시 데이터를 그대로 사용할 수 있고, 표면 재구성 없이도 3D 인식 태스크(분류, 검출, 분할)에 직접 활용 가능합니다.

**단점:** 위상 정보가 없어 렌더링에는 적합하지 않습니다. 표면 재구성이 필요한 경우 Poisson Surface Reconstruction, Ball-Pivoting 등의 알고리즘을 적용해야 합니다.

---

## 3. Voxel (Volumetric Pixel)

**3D 공간을 균일한 격자(grid)로 분할하고, 각 셀에 밀도·색상·점유율(occupancy) 등을 저장**하는 방식입니다. 2D 이미지의 픽셀을 3D로 확장한 개념입니다.

### 표현 방식

- **Binary Voxel Grid**: 각 셀을 0/1로 나타냅니다. 3D shape 학습(ShapeNet 등)에 많이 사용됩니다.
- **TSDF (Truncated Signed Distance Field)**: 각 복셀에 표면까지의 부호 있는 거리를 저장합니다. KinectFusion(Newcombe et al., 2011)에서 실시간 3D 재구성에 사용했습니다.
- **Sparse Voxel Octree**: 빈 공간을 생략하는 트리 구조로, 메모리 효율을 크게 향상시킵니다.

### 해상도 문제

복셀 해상도를 `N`이라 하면 메모리 복잡도는 O(N³)입니다. 256³만 해도 16M 셀이며, float32 기준 64MB가 됩니다. 이 때문에 실제 딥러닝 모델에서는 Sparse Convolution(MinkowskiEngine, spconv)을 사용해 비어있는 복셀 연산을 생략합니다.

**장점:** 볼류메트릭 데이터(의료 CT/MRI, 유체 시뮬레이션)를 자연스럽게 표현합니다. 3D Convolution 연산과 직접 호환됩니다.

**단점:** 균일 격자 구조상 해상도 증가에 따른 메모리 폭발 문제가 있습니다. 표면 세부 표현을 위해 해상도를 높이면 대부분이 빈 공간인 비효율이 발생합니다.

---

## 4. Implicit Surface와 SDF

표면을 명시적인 점이나 면으로 저장하는 대신, **공간의 임의 좌표 p에 대해 실수 값을 반환하는 함수 f(p)로 표면을 정의**합니다.

```
f(p) < 0  →  물체 내부
f(p) = 0  →  표면 (Level Set)
f(p) > 0  →  물체 외부
```

**SDF (Signed Distance Field)**는 f(p)가 표면까지의 부호 있는 유클리드 거리를 반환하는 특수한 암시적 표면입니다. Eikonal Equation `|∇f(p)| = 1`을 만족합니다.

### 주요 활용

- **폰트 렌더링**: GPU에서 SDF를 샘플링해 해상도 독립적인 벡터 폰트를 렌더링합니다 (Valve, 2007).
- **충돌 판정 및 물리**: SDF 값이 0에 가까운 지점을 충돌 경계로 사용합니다.
- **레이 트레이싱**: Sphere Marching으로 ray-surface intersection을 효율적으로 계산합니다.
- **3D 재구성**: TSDF Fusion(KinectFusion), DeepSDF(Park et al., 2019)가 이 방식을 사용합니다.

**표면 추출:** SDF로부터 Mesh를 얻으려면 Marching Cubes 알고리즘으로 f(p) = 0인 등가면(isosurface)을 삼각형으로 근사합니다.

**장점:** 해상도 독립적이며, 위상 변화(두 표면의 합집합·교집합)를 CSG 연산으로 자연스럽게 처리할 수 있습니다.

**단점:** 임의의 복잡한 형상을 하나의 닫힌 수식으로 표현하기 어렵습니다. 표면 추출 시 Marching Cubes의 계산 비용이 발생합니다.

---

## 5. NeRF (Neural Radiance Field)

**Mildenhall et al., ECCV 2020** — 다시점 이미지로부터 새로운 시점의 이미지를 합성(Novel View Synthesis)하기 위해 장면 전체를 연속적인 신경망 함수로 표현합니다.

### 핵심 수식

NeRF는 5D 함수를 MLP로 근사합니다:

```
F_θ : (x, y, z, θ, φ) → (RGB, σ)
```

- **(x, y, z)**: 3D 위치
- **(θ, φ)**: 시선 방향 (구면 좌표계)
- **σ (density)**: 볼류메트릭 밀도. 불투명도에 해당합니다.

### Volume Rendering

카메라 레이 **r(t) = o + td** 를 따라 샘플링한 지점의 색상과 밀도를 적분해 픽셀 색상을 계산합니다:

```
C(r) = ∫ T(t) · σ(r(t)) · c(r(t), d) dt

T(t) = exp(-∫₀ᵗ σ(r(s)) ds)  // Transmittance (누적 투과율)
```

실제 구현에서는 계층적 샘플링(Hierarchical Sampling: Coarse + Fine network)으로 빈 공간 샘플링을 줄입니다.

### Positional Encoding

MLP는 고주파 신호를 학습하기 어려우므로, 좌표를 삼각함수 기저로 인코딩합니다:

```
γ(p) = (sin(2⁰πp), cos(2⁰πp), ..., sin(2^(L-1)πp), cos(2^(L-1)πp))
```

**장점:** 기하 구조와 외형(appearance)을 단일 모델로 표현합니다. 반투명, 반사, 굴절 등 복잡한 광학 현상을 암시적으로 학습합니다.

**단점:** 장면 하나를 학습하는 데 수 시간, 렌더링에 수 초가 필요합니다. 장면 일반화(generalization)가 어렵고, 편집이 불가능에 가깝습니다.

> 이후 Instant-NGP(Müller et al., 2022)의 Hash Encoding, TensoRF의 분해 기법 등 다양한 가속 방법이 등장했습니다.

---

## 6. 3D Gaussian Splatting (3DGS)

**Kerbl et al., SIGGRAPH 2023** — 장면을 수백만 개의 3D 가우시안 분포의 집합으로 명시적으로 표현하여, NeRF 수준의 렌더링 품질을 실시간으로 달성합니다.

### 표현 방식

각 가우시안 `G_i`는 다음 파라미터로 정의됩니다:

| 파라미터 | 의미 | 비고 |
| :--- | :--- | :--- |
| **μ ∈ ℝ³** | 중심 위치 | |
| **Σ ∈ ℝ³ˣ³** | 공분산 행렬 (크기·방향) | 수치 안정성을 위해 회전 쿼터니언 q + 스케일 s로 분해 |
| **α ∈ [0,1]** | 불투명도 (Opacity) | |
| **SH 계수** | 구면 조화 함수(Spherical Harmonics) 기반 색상 | 시점 의존적 외형 표현 |

3D 가우시안의 확률 밀도는 다음과 같습니다:

```
G(x) = exp(-½ (x - μ)ᵀ Σ⁻¹ (x - μ))
```

### Splatting 렌더링 파이프라인

1. **3D → 2D 투영**: 각 가우시안을 카메라 시점에서 2D 가우시안으로 투영합니다 (EWA Splatting 기법).
2. **깊이 정렬**: 가우시안을 카메라 거리 순으로 정렬합니다.
3. **Alpha Blending**: 앞에서 뒤 순서로 타일 기반 레스터라이저(Tile-based Rasterizer)로 합성합니다.

GPU 상에서 병렬 처리가 가능해 **실시간 30fps+ 렌더링**이 가능합니다.

### 학습 과정

1. SfM(COLMAP)으로 초기 Point Cloud와 카메라 파라미터를 추정합니다.
2. 각 점을 초기 가우시안으로 설정하고, 렌더링된 이미지와 Ground Truth의 차이(L1 + SSIM Loss)로 최적화합니다.
3. **Adaptive Density Control**: 학습 중 가우시안을 분열(split)·복제(clone)·제거(prune)하여 밀도를 조절합니다.

### NeRF와의 비교

| | NeRF | 3DGS |
| :--- | :--- | :--- |
| **표현** | 암시적 (신경망 함수) | 명시적 (가우시안 집합) |
| **렌더링 속도** | ~수 초/장 | ~33ms/장 (실시간) |
| **학습 시간** | 수 시간 | 20~40분 |
| **메모리** | 수십 MB (모델 가중치) | 수백 MB~수 GB (가우시안 파라미터) |
| **편집 가능성** | 매우 어려움 | 개별 가우시안 조작 가능 |
| **동적 장면** | 확장 연구 존재 (D-NeRF 등) | 4DGS, Dynamic 3DGS 등 연구 진행 중 |

**장점:** 실시간 Novel View Synthesis와 씬 편집 가능성. NeRF 기반 방법보다 빠른 학습 속도.

**단점:** 가우시안 수에 비례하는 스토리지 용량. SfM 초기화에 의존하므로 텍스처리스(textureless) 표면이나 투명 물체에서 품질이 떨어집니다.

---

## 7. Parametric Surface (NURBS / Spline)

**제어점(Control Point)의 가중 선형 결합으로 곡선/곡면을 정의**하는 방식입니다. NURBS(Non-Uniform Rational B-Spline)는 CAD/CAM 산업의 표준 포맷(IGES, STEP)입니다.

기저 함수의 매개변수(knot vector)를 조작해 로컬 편집이 가능하며, 원뿔 곡선(원, 타원)을 정확하게 표현할 수 있습니다. 게임 엔진 파이프라인에서는 렌더링 전 GPU Tessellation으로 삼각형 메시로 변환됩니다.

---

## 요약 비교

| 표현 방식 | 표현 유형 | 렌더링 적합성 | 편집·조작 | 대표 활용 |
| :--- | :--- | :--- | :--- | :--- |
| **Polygon Mesh** | 명시적, 표면 | ★★★★★ | 쉬움 | 게임, 영화 CG, CAD |
| **Point Cloud** | 명시적, 비정형 | ★☆☆☆☆ | 중간 | LiDAR, SfM, 3D 인식 |
| **Voxel** | 명시적, 볼류메트릭 | ★★★☆☆ | 쉬움 | CT/MRI, 3D CNN, 게임 VFX |
| **SDF (Implicit)** | 암시적, 연속 | ★★☆☆☆ | CSG 연산 | 물리 시뮬레이션, 레이 트레이싱 |
| **NeRF** | 암시적, 신경망 | ★★☆☆☆ | 매우 어려움 | NVS, 자율주행 씬 재구성 |
| **3DGS** | 명시적, 확률 분포 | ★★★★☆ | 가우시안 단위 | 실시간 NVS, XR, 자율주행 시뮬레이션 |
| **NURBS** | 파라메트릭, 표면 | ★★☆☆☆ | 정밀 제어 | 산업용 CAD |

---

## 자율주행·3D 비전에서의 현실

실제 시스템은 단일 표현에 의존하지 않습니다.

- **인식 단계**: LiDAR Point Cloud → Sparse Voxel Grid → 3D Object Detection (VoxelNet, CenterPoint 등)
- **재구성·시뮬레이션**: 다시점 카메라 + LiDAR → NeRF/3DGS 기반 포토리얼리스틱 씬 생성 (UniSim, MARS, EmerNeRF 등)
- **HD Map**: 도로 구조를 Parametric Curve로 표현하고 Mesh로 시각화

특히 3DGS는 실시간 렌더링 가능성 덕분에 센서 시뮬레이터(closed-loop)와의 통합이 활발히 연구되고 있으며, 자율주행 데이터 증강(data augmentation)의 핵심 기술로 부상하고 있습니다.
