---
title: "3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes"
date: 2026-04-10T08:30:00+09:00
draft: false
categories: ["Papers"]
tags: ["computer-graphics", "rendering", "gaussian-splatting", "ray-tracing", "3d-reconstruction"]
---

## 개요

- **저자**: Nicolas Moenne-Loccoz, Ashkan Mirzaei, Or Perel, Riccardo De Lutio, Janick Martinez Esturo, Gavriel State, Sanja Fidler, Nicholas Sharp, Zan Gojcic (NVIDIA)
- **발행년도**: 2024
- **주요 내용**: 3D Gaussian Splatting 장면을 효율적으로 ray tracing하는 GPU 가속 알고리즘 제안. Rasterization의 한계를 극복하여 secondary ray effects(반사, 굴절, 그림자), distorted cameras(렌즈 왜곡, rolling shutter), stochastic ray sampling 등을 지원.

## 목차

1. [소개 및 동기](#1-소개-및-동기)
2. [관련 연구](#2-관련-연구)
3. [배경](#3-배경)
4. [핵심 방법론](#4-핵심-방법론)
5. [고급 기능 및 응용](#5-고급-기능-및-응용)
6. [실험 결과](#6-실험-결과)
7. [핵심 개념 정리](#핵심-개념-정리)

---

## 1. 소개 및 동기

### 개요

3D Gaussian Splatting(3DGS)은 particle-based radiance field 표현의 획기적인 발전으로, 복잡한 장면의 재구성과 novel-view synthesis에서 뛰어난 성능을 보여줍니다. 기존 3DGS 방법들은 **rasterization** 기반의 tile-based renderer를 사용합니다.

그러나 이 논문의 핵심 관찰은 **rasterization의 고유한 한계**입니다:

- ❌ Highly-distorted cameras(로봇, 시뮬레이션에서 중요)를 처리 불가
- ❌ Secondary rays(반사, 굴절, 그림자)의 효율적 시뮬레이션 불가
- ❌ Stochastic ray sampling 지원 불가
- ❌ Rolling shutter, motion blur 등의 센서 특성 표현 불가

### 제안 솔루션

이 논문은 **GPU-accelerated ray tracing**을 통해 3D Gaussian particles를 렌더링하는 새로운 접근법을 제안합니다. 핵심 아이디어는:

1. 각 particle 주변에 **bounding mesh primitives** 구성
2. 이를 **BVH(Bounding Volume Hierarchy)**에 삽입
3. 고도로 최적화된 **k-buffer hits-based marching** 알고리즘으로 반투명 particles 처리

### 주요 기여

- **GPU-accelerated ray tracing algorithm** for semi-transparent particles
- **Improved optimization pipeline** for ray-traced, particle-based radiance fields
- **Generalized Gaussian formulations**으로 intersection count 감소
- 다양한 응용: depth of field, shadows, mirrors, distorted cameras, rolling shutter, instancing

---

## 2. 관련 연구

### 2.1 Novel-View Synthesis와 Radiance Fields

전통적인 novel-view synthesis 접근법들:

- **Sparse views**: Multi-view stereo와 point cloud를 먼저 구성한 후 image unproject
- **Dense views**: Light field interpolation

**Neural Radiance Fields (NeRF)**의 혁신:
- Volumetric radiance field를 coordinate-based neural network로 표현
- 임의의 위치에서 volumetric density와 view-dependent color 쿼리 가능
- 그 이후 다양한 개선 연구: 속도 향상, 품질 개선, surface representation 개선

**3D Gaussian Splatting (3DGS)**:
- Particle-based representation의 새로운 표준으로 등장
- Fuzzy, anisotropic Gaussian particles의 위치, 모양, 외관을 최적화
- Tile-based rasterizer로 실시간 성능 달성

### 2.2 Point-Based 및 Particle Rasterization

Point-based rendering의 진화:

1. **Initial approach**: 단순 point rasterization → 구멍과 aliasing 문제
2. **Particle representation**: Surfels, discs, ellipsoids 등으로 확장
3. **Differentiable rendering**: Alpha blending을 통한 미분가능한 rendering
4. **Recent advances**: CNN networks나 NeRF-like volumetric rendering과 결합

**Pulsar**: 효율적인 sphere-based differentiable rasterization으로 백만 단위의 particle 최적화 가능

**한계점**:
- Rasterization은 distorted cameras 표현 불가
- Secondary lighting effects(occlusion, shading) 제한적
- 이를 해결하기 위한 workarounds들이 제한적임

### 2.3 Semi-Transparent Particles의 Differentiable Ray Tracing

#### 기존 접근법들의 한계:

1. **Transmittance estimation method**: 전체 장면 traversal이 느림 → Gaussian particles에서는 비효율적
2. **Slab-tracing (Knoll et al. 2019)**:
   - 정규 형태의 isotropic particles 가정
   - 균일한 공간 분포 가정
   - Gaussian particles의 높은 밀도와 불규칙한 분포에서 부적합

3. **Multi-layer alpha tracing (Brüll & Grosch 2020)**:
   - Particle ordering 근사로 인한 rendering artifacts
   - Differentiability 보장 불확실

#### 본 논문의 위치:

- **Fuzzy Metaballs (Keselman & Hebert 2022, 2023)**과 유사하지만:
  - Fuzzy Metaballs: 수십 개 particle, 저해상도 이미지만 처리
  - 본 논문: **백만 단위의 particles, Full HD 실시간 렌더링**
- 보증된 hit processing order로 **완전한 differentiability** 달성

---

## 3. 배경

### 3.1 3D Gaussian Parameterization

#### Gaussian Kernel 함수

$$\rho(\mathbf{x}) = e^{-(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})}$$

**수식 설명**:
- **$\rho(\mathbf{x})$**: 3D 공간의 점 $\mathbf{x}$에서의 Gaussian kernel 값 (0~1 범위)
  - 값이 클수록 Gaussian의 중심에 가깝다는 의미
- **$\boldsymbol{\mu} \in \mathbb{R}^3$**: particle의 위치 (중심)
- **$\mathbf{x} \in \mathbb{R}^3$**: 공간상의 임의의 점
- **$\boldsymbol{\Sigma} \in \mathbb{R}^{3 \times 3}$**: Covariance matrix (particle의 형태/방향/크기 결정)
- **$(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})$**: Mahalanobis distance (방향을 고려한 거리)

#### Covariance Matrix 분해

$$\boldsymbol{\Sigma} = \mathbf{R}\mathbf{S}\mathbf{S}^T\mathbf{R}^T$$

**수식 설명**:
- **Covariance matrix 분해**: Optimization 과정에서 positive semi-definiteness 보장
- **$\mathbf{R} \in SO(3)$**: 3D rotation matrix (particle의 방향)
  - Quaternion $\mathbf{q} \in \mathbb{R}^4$로 저장
- **$\mathbf{S} \in \mathbb{R}^{3 \times 3}$**: Diagonal scaling matrix (3개 축의 스케일)
  - Vector $\mathbf{s} \in \mathbb{R}^3$로 저장
- **직관**: 표준 Gaussian(구 형태)를 회전($\mathbf{R}$)하고 늘림($\mathbf{S}$)으로써 타원체 형태 생성

#### Radiance 함수 (View-dependent)

$$\boldsymbol{\phi}_{\boldsymbol{\beta}}(\mathbf{d}) = f\left(\sum_{\ell=0}^{\ell_{max}} \sum_{m=-\ell}^{\ell} \beta_\ell^m Y_\ell^m(\mathbf{d})\right)$$

**수식 설명**:
- **$\boldsymbol{\phi}_{\boldsymbol{\beta}}(\mathbf{d}) \in \mathbb{R}^3$**: 시점 방향 $\mathbf{d}$에 따른 RGB 색상 (view-dependent)
- **$Y_\ell^m(\mathbf{d})$**: Spherical harmonic basis (order $m = 3$)
  - 48개의 계수로 복잡한 view-dependent 색상 표현
- **$\beta_\ell^m$**: Spherical harmonic 계수
- **$f$**: Sigmoid function (색상을 [0,1] 범위로 정규화)
- **직관**: 시점에 따라 색상이 어떻게 변하는지를 수학적으로 표현 (물체의 광택, 반사 특성 포함)

### 3.2 Differentiable Rendering: Volumetric Particles

#### Volume Rendering Equation

$$\mathbf{L}(\mathbf{o}, \mathbf{d}) = \int_{\tau_n} f T(\mathbf{o}, \mathbf{d}) \sum_i (1 - e^{-\sigma_i \rho_i(\mathbf{o} + \tau\mathbf{d})}) \mathbf{c}_i(\mathbf{d}) d\tau$$

**수식 설명** (초보자용):

이 식은 **카메라 광선을 따라 색상을 누적하는 방법**을 나타냅니다:

- **$\mathbf{L}(\mathbf{o}, \mathbf{d})$**: 최종 출력 색상 (우리가 화면에서 보는 색)
- **$\mathbf{o}$**: 카메라 위치 (광선의 시작점)
- **$\mathbf{d}$**: 광선 방향 (카메라에서 장면을 보는 방향)
- **$\tau$**: 광선을 따라 이동하는 거리 파라미터
- **$\sigma_i$**: $i$번째 particle의 opacity (불투명도)
  - $\sigma_i = 0$: 완전 투명
  - $\sigma_i = 1$: 완전 불투명
- **$\rho_i(\cdot)$**: $i$번째 particle의 Gaussian kernel 값 (그 점에서 particle의 강도)
- **$\mathbf{c}_i(\mathbf{d})$**: $i$번째 particle의 색상 (시점 의존적)
- **$(1 - e^{-\sigma_i \rho_i(\cdot)})$**: $i$번째 particle이 광선을 차단하는 정도
- **$T(\mathbf{o}, \mathbf{d})$**: Transmittance function (현재까지 누적된 투과율)
  - 앞의 particles를 통과해서 빛이 얼마나 남아있는지

#### Transmittance Function

$$T(\mathbf{o}, \mathbf{d}) = e^{-\int_{\tau_n} \sum_i \sigma_i \rho_i(\mathbf{o} + t\mathbf{d})dt}$$

**수식 설명**:
- **누적 투과율**: 광선이 시작점에서 현재까지 통과한 모든 particles로 인한 빛의 손실 표현
- **예시**:
  - 처음($\tau = 0$): $T = 1$ (아무것도 차단 안 함)
  - 반투명 layer 통과: $T = 0.5$ (빛의 50% 투과)
  - 또 다른 반투명 layer 통과: $T = 0.25$ (남은 빛의 50% 더 투과)

#### 이산화(Discretization)

$$\mathbf{L}(\mathbf{o}, \mathbf{d}) = \sum_{i=1}^{N} \mathbf{c}_i(\mathbf{d}) \alpha_i \prod_{j=1}^{i-1} (1 - \alpha_j)$$

여기서 $\alpha_i = \sigma_i \rho_i(\mathbf{x}_i)$

**수식 설명** (초보자용):

연속 적분을 **discrete sum**으로 근사합니다. 이것이 실제 구현에서 사용됩니다:

- **$N$**: 광선과 교차하는 particles의 총 개수
- **$\alpha_i$**: $i$번째 particle의 "alphanumerical value" (그 위치에서의 opacity)
- **$\prod_{j=1}^{i-1} (1 - \alpha_j)$**: $i$번째 particle 앞의 모든 particles를 통과한 빛의 비율
  - **예시**: 첫 3개 particle의 $\alpha$ 값이 [0.5, 0.3, 0.2]라면:
    - 1번째 기여: $0.5 \times 1 = 0.5$
    - 2번째 기여: $0.3 \times (1 - 0.5) = 0.3 \times 0.5 = 0.15$
    - 3번째 기여: $0.2 \times (1 - 0.5) \times (1 - 0.3) = 0.2 \times 0.5 \times 0.7 = 0.07$

### 3.3 Hardware-Accelerated Ray Tracing

#### NVIDIA OptiX 프레임워크

NVIDIA RTX GPU는 BVH(Bounding Volume Hierarchy) 기반 ray tracing 가속:

**5가지 프로그램 타입**:

1. **Ray-gen program**: 각 픽셀마다 광선 생성
2. **Intersection program**: 기하와의 교차 계산 (custom primitives용)
3. **Any-hit program**: 모든 hit에 대해 호출 (hit 거부 가능)
4. **Closest-hit program**: 가장 가까운 hit 처리
5. **Miss program**: hit이 없을 때 처리

**Opaque primitives 최적화**:
- 하드웨어는 opaque surfaces를 위해 최적화됨
- Semi-transparent particles 처리는 도전적 → **본 논문의 핵심 기여**

---

## 4. 핵심 방법론

### 4.1 Bounding Primitives: Proxy 기하 설계

#### 문제 정의

Gaussian particles를 BVH에 삽입할 때, 교차 검사의 **tightness**와 **계산 비용** 간의 trade-off:

- **Tight bounds**: 거짓 양성(false-positive) 감소 → 더 많은 계산
- **Loose bounds**: 빠른 계산 → 불필요한 particles 처리

#### 다양한 Proxy 기하 비교

| Proxy 유형 | 장점 | 단점 |
|-----------|------|------|
| **AABB (Axis-Aligned Bounding Box)** | 매우 빠른 교차 검사 | Diagonal-stretched Gaussians에서 많은 false-positives |
| **Sphere** | 간단한 교차 검사 | 타원체 Gaussians에 부적합 |
| **Icosahedron Mesh** | 타원체에 좋은 fit | Triangle 교차가 더 느림 |
| **Stretched Icosahedron Mesh** (제안) | **타원체에 최적 fit** | **하드웨어 최적화된 ray-triangle 교차** |

#### Stretched Icosahedron Proxy (제안)

**핵심 아이디어**:
- 정규 icosahedron(20개 삼각형)을 Gaussian의 covariance에 맞게 변형
- 최소 응답값 $\alpha_{min}$으로 fit 정의 (예: 0.95)

**수식**:

$$\text{Stretched Icosahedron} = \mathbf{R} \text{Diag}(\mathbf{s}) \mathbf{I}_{base}$$

**수식 설명**:
- **$\mathbf{I}_{base}$**: 기본 정규 icosahedron의 꼭짓점
- **$\text{Diag}(\mathbf{s})$**: Gaussian의 scale vector로 스케일링
- **$\mathbf{R}$**: Gaussian의 rotation matrix로 회전
- **결과**: Gaussian의 형태를 정확히 따르는 메시 proxy

**성능 개선**:
- AABB 대비 **약 3배 더 빠른 FPS** (false-positives 감소)

### 4.2 Ray Tracing 알고리즘: k-Buffer Hits-Based Marching

#### 핵심 도전과제

**Opaque geometry**와 **volumetric particles**의 렌더링 차이:

```
Opaque geometry:
  - 각 광선마다 단 1-2개의 교차점
  - 빠른 처리 가능
  
Volumetric particles:
  - 각 광선마다 수백~수천 개의 교차점
  - OptiX 하드웨어의 기대와 불일치
```

#### 제안 알고리즘: K-Buffer Hits-Based Marching

**개념**: 한 번에 k개의 closest hits를 수집한 후, 깊이 순서로 일괄 처리

**알고리즘 흐름**:

```
repeat:
  1. BVH에 대해 ray 발사
  2. k개의 가장 가까운 particle hits 수집 (k-buffer)
  3. 깊이 순서로 정렬
  4. 각 hit에 대해:
     a. Particle response 계산 (alpha_i)
     b. 누적 radiance 갱신 (식 6)
  5. Transmittance가 threshold 이하이거나 모든 particles 처리될 때까지 반복
```

**구체적인 구현**:

```glsl
// Any-hit program (OptiX)
if (ray_hit_particle(primitive_id)) {
  particle_id = bvh_primitive_to_particle(primitive_id);
  
  // Front-facing 삼각형만 처리
  if (dot(ray_direction, triangle_normal) < 0) {
    // Hit을 k-buffer에 추가
    add_to_k_buffer(particle_id, hit_t);
  }
}

// Closest-hit program
batch_process_k_closest_hits_in_order() {
  for each (particle_id, depth) in sorted_k_buffer {
    compute_particle_response(particle_id);
    update_accumulated_radiance();
    update_transmittance();
  }
}
```

**주요 최적화**:

1. **K-Buffer 크기**: 작을수록 빠르지만 정확도 손실
   - 실험: $k = 32$가 최적 (속도 vs 품질 균형)

2. **Transmittance Threshold**: 매우 작은 투과율에서 조기 종료
   - 기본값: 0.001 (대부분의 광선 기여도 캡처)

3. **Adaptive Clamping**: 특정 조건에서 alpha 값 제한
   - Numerical stability 향상
   - 약 10% 속도 개선

### 4.3 Particle Response 계산

#### 정의

각 particle이 광선의 특정 점에서 얼마나 기여하는지:

$$\alpha_i = \sigma_i \cdot \rho_i(\mathbf{x}_i)$$

여기서 $\mathbf{x}_i$는 광선 위에서 particle의 Gaussian kernel이 최대인 점.

#### 최적화 계산

**문제**: 각 교차점마다 정확한 최대값 점을 찾기는 비용이 높음

**해결책**: Analytical solution 사용

Gaussian kernel의 최대값 점은 **ray와 Gaussian 중심의 projection**:

$$t^* = (\boldsymbol{\mu} - \mathbf{o})^T \mathbf{d}$$

$$\mathbf{x}_i = \mathbf{o} + t^* \mathbf{d}$$

**수식 설명**:
- **$t^*$**: 광선 매개변수화에서 최대값 점의 위치
- **$\boldsymbol{\mu}$**: Gaussian particle의 중심
- **$\mathbf{o}$**: 광선의 시작점 (카메라 위치)
- **$\mathbf{d}$**: 정규화된 광선 방향
- 이를 이용하면 $\alpha_i$ 계산이 상수 시간에 완료

---

## 4.4 Backward Pass: 최적화를 위한 미분

#### 문제

Ray tracing 경로는 differentiable해야 함 (Gaussian particles 최적화를 위해).

#### 해결책

**Ordered particle processing**의 중요성:
- k-buffer를 정렬된 순서로 처리하면, 누적 radiance는 particle parameters에 대해 미분 가능
- **Back-propagation**: 각 hit에 대해 gradient 계산

**구체적 구현**:

```python
# Forward pass (위에서 설명)
L = accumulated_radiance(sorted_hits)

# Backward pass
for i in range(N):
  dL/d(particle_params[i]) = chain_rule(dL/dL * dL/d_alpha[i])
```

---

## 4.5 일반화된 Particle 표현

#### 동기

기본 Gaussian particles는 구형이어야 하는 제약이 있음. 다양한 kernel 함수 탐색:

#### 일반화된 Gaussian (Power-based)

$$\rho_n(\mathbf{x}) = e^{-\left( (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right)^n}$$

**수식 설명**:
- **$n = 1$**: 표준 Gaussian (가우스 분포)
- **$n = 2$**: Super-Gaussian (중심에 집중, 가장자리 빠르게 감소)
- **$n > 1$**: 더 "날카로운" kernel

**성능 비교** (Table 4):

| Kernel | PSNR | FPS | 특징 |
|--------|------|-----|------|
| Gaussian (n=1) | 23.03 | 77 | 기본선 |
| Generalized (n=2) | 22.68 | 141 | **81% 빠름**, 약간의 품질 손실 |
| 2D Gaussians | 22.70 | 241 | 지도 기반 (특정 응용) |
| Cosine wave | 22.77 | 268 | 실험적 |

**결론**: $n=2$ generalized Gaussian은 **성능/품질 최적**으로 권장

#### 다른 Kernel 함수들:
- **RBF (Radial Basis Function)**: 다항식 기반
- **Cosine wave**: 주기적 구조
- **2D Gaussians**: 카메라 정렬 particle (특정 시나리오)

---

## 5. 고급 기능 및 응용

### 5.1 Secondary Ray Effects

#### Shadows (그림자)

**기본 원리**: 반투명 particles에 대한 shadow ray 추적

```
For each primary ray hit:
  1. Shadow ray를 광원 방향으로 발사
  2. Ray tracing으로 shadow ray 경로의 opacity 계산
  3. Transmittance를 그림자 강도로 사용
```

**식**:

$$L_{shadowed} = L_{lit} \cdot T(\text{particle}, \text{light})$$

#### Reflections (반사)

**구현**:
1. Primary ray hit에서 법선 계산
2. Reflection ray 발사 (법선 기반)
3. Recursive ray tracing (최대 깊이 제한)

**장점**: Rasterization 기반 방법과 달리 **정확한 다중 bounce reflection** 가능

#### Refractions (굴절)

**Snell's Law 적용**:

$$n_1 \sin(\theta_1) = n_2 \sin(\theta_2)$$

**구현**: 
- Refraction ray 발사
- Recursive tracing
- Fresnel effect 고려 가능

### 5.2 Distorted Cameras

#### Rolling Shutter 효과

**동기**: 스마트폰, 로봇 카메라에서 흔한 효과
- 이미지가 한 번에 캡처되지 않고, 위에서 아래로 순차적으로 스캔

**구현**:

```python
# 각 광선마다 다른 타임스탬프
for pixel_y in range(height):
  timestamp = pixel_y / height * exposure_time
  # Temporal motion 고려하여 ray 발사
  ray = get_ray_at_time(pixel_y, timestamp)
  trace_ray(ray)
```

**식**:

$$t_{capture}(y) = t_{start} + \frac{y}{H} \cdot \Delta t_{exposure}$$

**수식 설명**:
- **$y$**: 픽셀의 수직 위치
- **$H$**: 이미지 높이
- **$\Delta t_{exposure}$**: 노출 시간
- 각 행이 다른 시간에 캡처되므로 motion이 있으면 왜곡

#### Lens Distortion (렌즈 왜곡)

**원리**: 광선의 방향을 비선형적으로 변형

```python
# Ideal pinhole camera ray
ray_ideal = compute_pinhole_ray(pixel_x, pixel_y)

# Apply distortion model
r2 = ray_ideal.x**2 + ray_ideal.y**2
distortion_factor = 1 + k1*r2 + k2*r4  # Brown distortion model
ray_distorted = ray_ideal * distortion_factor

trace_ray(ray_distorted)
```

### 5.3 Depth of Field (피초점)

**원리**: 렌즈의 finite aperture 시뮬레이션

```python
# Pinhole ray
ray_pinhole = compute_pinhole_ray(pixel_x, pixel_y)

# Lens sampling
for sample in range(num_samples):
  lens_offset = sample_lens_disk(aperture_radius)
  ray = ray_pinhole + lens_offset
  
  # Focus point 통과 필수
  focus_distance = compute_focus_distance()
  refocus_ray(ray, focus_distance)
  
  trace_ray(ray)

# 모든 sample의 결과 평균
```

**식**:

$$\text{DoF 효과} = \frac{1}{N} \sum_{i=1}^{N} L(\text{ray}_i)$$

**수식 설명**:
- **$N$**: Lens sample 개수
- **$\text{ray}_i$**: 렌즈의 다른 위치에서 나온 광선
- 렌즈 위의 여러 점에서 광선을 발사 후 평균하면 피초점 효과

---

## 5.4 Stochastic Ray Sampling

**동기**: Training에서 각 iteration마다 정확히 같은 광선을 사용하지 않음 (regularization)

**구현**:

```python
# Training iteration
for epoch in range(num_epochs):
  # 무작위 부분 집합 선택
  sampled_pixels = random.sample(all_pixels, batch_size)
  
  for pixel in sampled_pixels:
    # 해당 픽셀에서 ray 발사 (정규 grid가 아님)
    ray = jitter_ray(pixel, jitter_amount=random.uniform(0, 1))
    L = trace_ray(ray)
    
    loss = compute_loss(L, target)
    backward_pass()
```

---

## 6. 실험 결과

### 6.1 정량적 평가

#### 품질 비교: 3DGS (Rasterization) vs GaussianTracer (Ray Tracing)

**표준 벤치마크 (Mip-NeRF 360)**:

| Dataset | Method | PSNR ↑ | SSIM ↑ |
|---------|--------|--------|--------|
| bicycle | 3DGS | 25.64 | 0.834 |
|  | GaussianTracer | 25.58 | 0.831 |
| garden | 3DGS | 27.15 | 0.879 |
|  | GaussianTracer | 27.09 | 0.876 |
| counter | 3DGS | 25.83 | 0.821 |
|  | GaussianTracer | 25.79 | 0.819 |

**결론**: Ray tracing 방식이 3DGS와 **거의 동일한 품질**을 달성하면서도 **추가 기능** 지원

#### 성능 분석: 다양한 최적화의 영향

**비교 대상**:
- Naive implementation: ~30 FPS (baseline)
- + BVH optimization: ~50 FPS
- + Stretched icosahedron proxy: ~110 FPS
- + k-buffer hits marching: ~150 FPS
- + Adaptive clamping: ~155 FPS
- **최종**: **약 25배 성능 개선** 🚀

#### 다양한 Parameter 분석

**K-Buffer 크기의 영향** (Figure 9):
```
k=8:   130 FPS (기본선)
k=16:  145 FPS 
k=32:  155 FPS (최적)
k=64:  140 FPS (감소, 오버헤드 증가)
```

**Particle 개수 vs 성능**:
- 1M particles: 155 FPS
- 2M particles: 90 FPS (메모리 대역폭 제한)
- 3M particles: 45 FPS

### 6.2 정성적 평가: 응용 사례

#### 1. Distorted Camera Rendering

**설정**: 원래 이미지는 perfect pinhole camera로 촬영, GaussianTracer는 high distortion으로 렌더링

**결과**: 
- 3DGS: Distortion 표현 불가 (Rasterization 한계)
- GaussianTracer: **정확한 렌즈 왜곡 재현**

#### 2. Rolling Shutter Effects

**장면**: 빠르게 움직이는 로봇 카메라

**결과**:
- 기존 workaround (Seiskari et al. 2024): Screen-space approximation, 부정확
- GaussianTracer: **정확한 시간적 샘플링**으로 정확한 rolling shutter

#### 3. Secondary Ray Effects

**그림자와 반사**:

```
Reflection example:
  - Mirror 표면 추가
  - Ray tracing으로 정확한 반사 계산
  - 3DGS: 불가능 (rasterization)

Shadow example:
  - Point light에서 shadow ray
  - Semi-transparent particles를 통과하는 빛 계산
  - 부드러운 그림자(soft shadow) 생성
```

**정량**: Secondary effects 추가 시 **대부분 10~20% 성능 오버헤드**만 발생

#### 4. Training with Distorted Cameras

**실험**: 렌즈 왜곡이 있는 training 이미지로 Gaussians 최적화

**과정**:
1. Distorted 이미지에 대해 GaussianTracer로 rendering
2. Loss 계산 및 backward pass
3. Gaussian parameters 업데이트

**결과**: 
- 3DGS (workaround): 별도 NeRF training 필요 → 비효율적
- GaussianTracer: **직접 최적화 가능**

### 6.3 일반화: 다양한 Particle 표현

#### Generalized Gaussian Kernels

**$n=2$ generalized Gaussian 사용 시**:
- **FPS**: 77 → 141 (81% 향상) 🎯
- **PSNR 손실**: 0.35 dB (거의 무시할 수준)
- **Trade-off**: 성능 향상이 품질 손실보다 훨씬 큼

#### 다른 Particle 표현과의 호환성

논문은 GaussianTracer가 다양한 particle-based scene representations에 적용 가능함을 보여줍니다:

- **3D Gaussian Splatting** (기본)
- **Generalized Gaussians** (다항식 kernel)
- **RBF Particles** (radial basis functions)
- **Differentiable Point Clouds with Features**

---

## 핵심 개념 정리

### 1. Ray Tracing vs Rasterization

| 특성 | Rasterization | Ray Tracing |
|------|--------------|------------|
| **기본 아이디어** | 각 삼각형을 화면에 투영 | 각 픽셀에서 광선 발사 |
| **Secondary Rays** | 불가능 | 가능 (반사, 굴절) |
| **Distorted Cameras** | 어려움 | 자연스러움 |
| **성능** | 매우 빠름 | 전통적으로 느림 |
| **이 논문의 성과** | - | **반투명 particle에 최적화** |

### 2. 3D Gaussian Parameterization

**핵심**: 3D 공간의 각 점에서 부드러운 값을 정의
```
Gaussian = 중심 위치 (μ) + 형태 (Σ = RSS^T R^T) + 색상 (spherical harmonics)
```

### 3. Volume Rendering 방정식

**핵심**: 광선을 따라 반투명 particles의 색상을 누적
```
최종 색상 = ∑ (particle 색상 × opacity × 앞의 particles 투과율)
```

### 4. K-Buffer Hits-Based Marching

**혁신**: Optiх 하드웨어를 반투명 particles에 맞게 재설계
```
1. 한 번에 k개의 가장 가까운 hits 수집
2. 깊이 순서로 정렬
3. Batch 처리로 GPU 효율성 증대
```

### 5. Bounding Mesh Proxies

**최적화**: 정확한 particle bounding으로 false-positives 감소
```
Stretched Icosahedron = 정규 icosahedron을 Gaussian의 covariance로 변형
→ AABB 대비 3배 빠름
```

### 6. Backward Pass & Differentiability

**중요**: Ray tracing 과정이 미분가능해야 최적화 가능
```
Forward: pixels → rays → particles → colors
Backward: gradients 역전파 → particle parameters 업데이트
```

### 7. Generalized Gaussian Kernels

**혁신**: Power 기반 kernel로 성능 향상
```
Generalized Gaussian (n=2) = 81% 빠르면서도 품질 거의 동일
```

### 8. Secondary Ray Effects 구현

**응용**:
- **Shadows**: 광원 방향 ray로 transmittance 계산
- **Reflections**: 법선 기반 reflection ray
- **Refractions**: Snell's law 적용
- **Depth of Field**: Lens sampling
- **Rolling Shutter**: 픽셀마다 다른 타임스탬프

---

## 결론 및 시사점

### 주요 성과

1. **효율적인 ray tracing**: 반투명 particles의 실시간 GPU 가속 ray tracing 달성
   - 천만 단위의 particles, Full HD, ~150 FPS

2. **기능성 확장**: Rasterization으로 불가능한 다양한 효과 지원
   - Secondary lighting effects, distorted cameras, temporal effects

3. **최적화**: 첫 naive implementation 대비 **25배 성능 개선**
   - Bounding geometry, k-buffer marching, adaptive clamping의 조합

4. **일반화**: 다양한 particle-based representations 지원
   - 논문의 방법론은 Gaussian에만 국한되지 않음

### 실무적 시사점

**컴퓨터 그래픽스 산업**:
- 3D Gaussian Splatting의 표현력 확장
- Real-time rendering에 ray tracing 통합 가능성 입증
- 로봇, AR/VR, 영화 제작 등 다양한 응용 분야 활용 가능

**연구 방향**:
- **Global illumination**: 이 논문의 ray tracer를 기반으로 전역 조명 구현 가능성
- **Inverse rendering**: Ray-based forward pass로 역렌더링(material estimation 등) 가능
- **Hybrid approaches**: Rasterization과 ray tracing의 혼합 방식 탐색

### 한계점 및 향후 과제

1. **메모리 대역폭**: 백만 개 이상의 particles에서 BVH build time이 병목
2. **Global illumination**: 이 논문은 direct lighting만 다룸 (global illumination 추가 연구 필요)
3. **Optimization complexity**: Forward/backward pass 모두 최적화 필요

### 실제 적용 시 고려사항

- **Scene 규모**: 3M particles 이상에서는 성능 저하 시작
- **Ray count**: Stochastic sampling에서는 ray count 증가 시 품질 향상
- **Memory**: BVH building에서 상당한 메모리 사용

---

## 논문의 기술적 깊이

### 이 논문이 해결한 핵심 문제

**"왜 기존 ray tracing 알고리즘으로는 Gaussians를 효율적으로 trace할 수 없나?"**

1. **Hit count의 폭발**: 
   - Opaque geometry: ~1-2 hits/ray
   - Gaussian scenes: ~100-1000 hits/ray
   - OptiX 하드웨어는 opaque 최적화 → 매우 비효율적

2. **Hit ordering 문제**:
   - Semi-transparent surfaces는 **정렬된 순서**로 처리 필수
   - 기존 알고리즘은 이를 보장하지 않음

3. **Proxy 기하 선택**:
   - AABB: 너무 loose → false-positives 많음
   - 개별 triangle 교차: 너무 tight 계산 비용 높음
   - 해결책: Stretched icosahedron (중간 지점)

### 이 논문이 제시한 기술적 혁신

1. **K-Buffer Hits-Based Marching**: 
   - 새로운 반투명 particle tracing 알고리즘
   - Opaque 최적화 하드웨어를 volume rendering에 맞춤

2. **Adaptive Bounding Mesh**: 
   - Gaussian의 covariance에 정렬된 proxy geometry
   - 기하와 optical properties의 일관성

3. **Differentiable Ray Tracing**: 
   - Ordered hit processing로 미분가능성 보장
   - Backward pass 구현으로 gradient-based optimization 가능

---

## 참고할 만한 관련 논문들

- **3D Gaussian Splatting** (Kerbl et al. 2023): 기반이 되는 논문
- **NeRF** (Mildenhall et al. 2020): 혁신적인 radiance field 표현
- **Pulsar** (Lassner & Zollhöfer 2021): Differentiable particle rasterization
- **Fuzzy Metaballs** (Keselman & Hebert 2022): 유사한 ray tracing 접근법 (제한적)
- **Zip-NeRF** (Barron et al. 2023): Distorted camera 모델링

---

## 실제 구현 팁

### NVIDIA OptiX를 이용한 구현 시 주요 고려사항

1. **BVH Building**:
   - Particles 추가/제거 시 매번 재건축 비용 높음
   - 배치 업데이트 권장

2. **K-Buffer 크기**:
   - 메모리-성능 trade-off 고려
   - $k=32$ 권장값이지만 scene에 따라 튜닝 필요

3. **Transmittance Threshold**:
   - 너무 높으면: 조기 종료로 품질 저하
   - 너무 낮으면: 불필요한 연산 증가
   - 0.001 권장값으로 시작

4. **Gradient Computation**:
   - Ordered hits 보장 필수
   - 순서가 바뀌면 gradients 부정확

