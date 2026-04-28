---
title: "3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes"
date: 2026-04-10T09:00:00+09:00
draft: false
categories: ["Papers", "Novel View Synthesis"]
tags: ["3D Gaussian Splatting", "Ray Tracing", "Novel View Synthesis", "Neural Rendering"]
---

## 개요

- **저자**: Nicolas Moenne-Loccoz, Ashkan Mirzaei, Or Perel, Riccardo de Lutio, Janick Martinez Esturo, Gavriel State, Sanja Fidler, Nicholas Sharp, Zan Gojic (NVIDIA)
- **발행년도**: 2024 (arXiv:2407.07090v3, 10 Oct 2024)
- **발표**: ACM SIGGRAPH Asia 2024
- **프로젝트 페이지**: GaussianTracer.github.io
- **주요 내용**: 3D Gaussian Splatting(3DGS)의 래스터화 한계를 극복하기 위해, 각 Gaussian 파티클에 캡슐화 프리미티브(encapsulating primitive)를 구성하고 BVH(Bounding Volume Hierarchy)에 삽입하여 GPU 가속 레이 트레이싱을 수행하는 방법을 제안. 반사, 굴절, 그림자, 피사계 심도(depth of field), 왜곡 카메라(fisheye), 롤링 셔터 등 다양한 이차 조명 효과를 실시간에 가깝게 지원.

## 목차

- Section 1: Introduction — 래스터화의 한계와 레이 트레이싱의 필요성
- Section 2: Related Work — 관련 연구 (Novel View Synthesis, NeRF, 파티클 래스터화, 볼류메트릭 레이 트레이싱)
- Section 3: Background — 3D Gaussian 표현, 볼류메트릭 렌더링, RTX 하드웨어 가속
- Section 4: Method — 바운딩 프리미티브, 레이 트레이싱 렌더러, 파티클 응답 평가, 미분 가능 최적화, 파티클 커널 함수
- Section 5: Experiments & Ablations — 벤치마크 비교, 추적 알고리즘 분석, 커널 함수 비교
- Section 6: Applications — 레이 기반 시각 효과, 인스턴싱, 노이즈 제거, 복합 카메라 및 자율주행
- Section 7: Discussion — 래스터화 대비 차이점, 한계 및 미래 연구

---

## Section 1: Introduction

**요약**

3D Gaussian Splatting(3DGS)은 대규모 장면 재구성과 실시간 Novel View Synthesis의 표준으로 자리잡았습니다. 그러나 3DGS는 타일 기반 래스터라이저를 사용하기 때문에 구조적 한계가 있습니다:

1. **왜곡 카메라 지원 불가**: 로봇공학에서 흔히 쓰이는 fisheye, 롤링 셔터 카메라를 제대로 렌더링하지 못함
2. **이차 조명 효과 없음**: 반사(reflection), 굴절(refraction), 그림자(shadow) 등을 시뮬레이션할 수 없음
3. **확률적 샘플링 불가**: 훈련 중 랜덤 픽셀 샘플링이 불가하여 스토캐스틱 최적화를 지원하지 못함

이 논문은 각 파티클 주위에 캡슐화 프리미티브를 구성하고 BVH에 삽입한 뒤, GPU 레이 트레이서(NVIDIA OptiX)로 고속으로 교차를 계산하는 방법을 제안합니다. 제안 방법은 래스터화 대비 약 2배 느리지만, 위의 모든 한계를 해결합니다.

**핵심 기여**

- **GPU 가속 레이 트레이싱 알고리즘**: 반투명 파티클 씬을 위한 GPU 가속 레이 트레이서
- **개선된 최적화 파이프라인**: 레이 트레이싱 기반 파티클 씬 학습을 위한 최적화 파이프라인
- **일반화된 Gaussian 파티클 공식**: 교차 수를 줄이고 렌더링 효율을 높이는 새로운 커널 함수들
- **다양한 응용**: 피사계 심도, 거울, 그림자, 왜곡 카메라, 롤링 셔터, 확률적 광선 샘플링

---

## Section 2: Related Work

**요약**

관련 연구를 세 흐름으로 정리합니다.

**Novel View Synthesis 및 NeRF**: NeRF(Neural Radiance Field)는 좌표 기반 신경망으로 장면을 표현하며 고품질 렌더링을 가능하게 했습니다. MipNeRF360, Zip-NeRF 등의 후속 연구들이 품질과 속도를 개선했지만, 모두 1 FPS 미만으로 실시간 렌더링은 불가합니다.

**포인트/파티클 래스터화**: 3DGS는 비등방성(anisotropic) Gaussian 파티클을 타일 기반 래스터라이저로 렌더링하여 높은 품질과 실시간 속도를 달성했습니다. 하지만 핀홀 카메라 가정에 종속되어 있어 일반 카메라 모델을 지원하기 어렵습니다.

**볼류메트릭 파티클의 미분 가능 레이 트레이싱**: Fuzzy Metaballs(Keselman & Hebert 2022)가 가장 유사한 접근이지만 저해상도 소수 파티클에만 적용 가능합니다. Knoll et al. 2019는 slab tracing 기반 semi-transparent RBF 파티클을 다루나, 효율이 공간 균일도에 의존합니다.

**핵심 개념**

- **NeRF (Neural Radiance Field)**: 3D 공간의 임의 위치와 시선 방향에 대해 색상과 밀도를 출력하는 신경망으로 장면을 표현하는 방법
- **3DGS (3D Gaussian Splatting)**: 3D 공간에 수백만 개의 Gaussian 타원체를 배치하고 카메라 방향으로 투영(splatting)하여 래스터화하는 방법
- **래스터화(Rasterization)**: 3D 장면을 카메라 공간에 투영하여 픽셀 단위로 처리하는 렌더링 방식
- **레이 트레이싱(Ray Tracing)**: 각 픽셀에서 광선을 쏘아 장면 내 교차를 계산하는 렌더링 방식

---

## Section 3: Background

**요약**

논문의 기반이 되는 세 가지 핵심 기술을 설명합니다.

### 3.1 3D Gaussian 파라미터화

각 파티클 $\tilde{\rho}$는 커널 함수로 표현됩니다. 표준 3D Gaussian의 경우:

$$\rho(\mathbf{x}) = e^{-(\mathbf{x}-\boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x}-\boldsymbol{\mu})}$$

**수식 설명**
- **$\rho(\mathbf{x})$**: 공간의 점 $\mathbf{x}$에서 파티클의 존재 강도 (0에 가까울수록 약함)
- **$\boldsymbol{\mu} \in \mathbb{R}^3$**: 파티클의 3D 위치(중심점)
- **$\Sigma \in \mathbb{R}^{3\times3}$**: 공분산 행렬 — 파티클의 크기와 방향을 결정하는 타원체 형태
- **$(\mathbf{x}-\boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x}-\boldsymbol{\mu})$**: 마할라노비스 거리 — 중심에서 얼마나 멀리 있는지를 타원체 기준으로 측정

공분산 행렬 $\Sigma$는 수치적 안정성을 위해 다음과 같이 분해됩니다:

$$\Sigma = RSS^T R^T$$

**수식 설명**
- **$R \in SO(3)$**: 회전 행렬 (파티클의 방향)
- **$S \in \mathbb{R}^{3\times3}$**: 스케일 행렬 (파티클의 크기)
- 이 분해를 통해 항상 양의 준정부호(positive semi-definite) 행렬을 보장하며, 그래디언트 기반 최적화가 안정적으로 수행됨

각 파티클은 불투명도 $\sigma \in \mathbb{R}$와 뷰 의존적 방사 함수를 가집니다:

$$\phi_{\boldsymbol{\beta}}(\mathbf{d}) = f\left(\sum_{l=0}^{L} \sum_{m=-l}^{l} \beta_l^m Y_l^m(\mathbf{d})\right)$$

**수식 설명**
- **$\phi_{\boldsymbol{\beta}}(\mathbf{d})$**: 방향 $\mathbf{d}$에서 보이는 파티클의 색상
- **$Y_l^m(\mathbf{d})$**: 구면 조화 함수(Spherical Harmonics) — 방향에 따른 색상 변화를 효율적으로 표현
- **$\beta_l^m \in \mathbb{R}^{48}$**: 구면 조화 계수 (최적화 대상 파라미터)
- **$f$**: 시그모이드 함수 (색상을 0~1 범위로 정규화)
- **직관**: 유리창처럼 보는 각도에 따라 색이 달라지는 효과를 수식으로 표현

### 3.2 파티클 표현의 미분 가능 렌더링

볼류메트릭 렌더링 방정식:

$$L(\mathbf{o}, \mathbf{d}) = \int_{t_{fn}}^{t_f} T(\mathbf{o}, \mathbf{d}) \left(\sum_i (1 - e^{-\sigma_i \rho_i(\mathbf{o}+\tau\mathbf{d})}) c_i(\mathbf{d})\right) d\tau$$

**수식 설명**
- **$L(\mathbf{o}, \mathbf{d})$**: 원점 $\mathbf{o}$에서 방향 $\mathbf{d}$로 쏜 광선의 최종 색상
- **$T(\mathbf{o}, \mathbf{d})$**: 투과율 함수 — 광선이 얼마나 통과할 수 있는지 (앞 물체에 의한 차폐)
- **$c_i(\mathbf{d})$**: i번째 파티클의 색상 (방향 의존)
- **$\sigma_i \rho_i$**: i번째 파티클의 유효 불투명도

투과율 $T$는 다음과 같습니다:

$$T(\mathbf{o}, \mathbf{d}) = e^{-\int_{t_{fn}}^{\tau} \sum_i \sigma_i \rho_i(\mathbf{o}+t\mathbf{d}) dt}$$

이를 수치 적분으로 근사하면:

$$L(\mathbf{o}, \mathbf{d}) \approx \sum_{i=1}^{N} c_i(\mathbf{d}) \alpha_i \prod_{j=1}^{i-1} (1 - \alpha_j)$$

**수식 설명**
- **$\alpha_i = \sigma_i \rho_i(\mathbf{x}_i)$**: i번째 파티클의 기여 불투명도
- **$\prod_{j=1}^{i-1}(1-\alpha_j)$**: i번째 파티클까지 도달하는 빛의 투과율 — 앞의 파티클들을 통과한 빛의 비율
- **직관**: 반투명 유리 여러 장을 겹쳐 놓았을 때, 뒤의 유리는 앞 유리들을 통과한 빛만큼만 기여함

### 3.3 하드웨어 가속 레이 트레이싱

NVIDIA RTX 하드웨어와 OptiX 프로그래밍 인터페이스를 사용합니다. OptiX는 삼각형 메시, 구, AABB(Axis-Aligned Bounding Box) 등을 BVH에 삽입하여 광선-기하 교차를 하드웨어로 가속합니다.

프로그래밍 모델의 진입점:
- **ray-gen**: 픽셀마다 광선을 시작하는 프로그램
- **intersection**: 광선과 프리미티브의 정확한 교차를 계산
- **any-hit**: 교차 발견 시 호출 (채택/거부 결정)
- **closest-hit**: 가장 가까운 교차점 처리
- **miss**: 교차 없을 때 처리

---

## Section 4: Method

**요약**

제안 방법의 두 핵심 요소: (1) 파티클을 BVH에 효율적으로 삽입하기 위한 바운딩 프리미티브 전략, (2) NVIDIA OptiX 위에서 효율적으로 스케줄링되는 렌더링 알고리즘.

### 4.1 바운딩 프리미티브 (Bounding Primitives)

**Stretched Polyhedron Proxy Geometry**: 정규 정이십면체(regular icosahedron)를 각 파티클에 맞게 늘려(stretch) 사용합니다. 정이십면체는 레이-삼각형 교차 계산에 하드웨어 최적화되어 있으며 타이트하게 파티클을 감쌉니다.

각 꼭짓점 $v$는 다음과 같이 변환됩니다:

$$v \leftarrow \sqrt{2\log(\sigma/a_{min})} S R^T v + \boldsymbol{\mu}$$

**수식 설명**
- **$\sigma$**: 파티클의 불투명도 계수
- **$a_{min}$**: 캡처해야 할 최소 반응값 (보통 $a_{min} = 0.01$)
- **$\sqrt{2\log(\sigma/a_{min})}$**: 파티클이 $a_{min}$ 이상의 기여를 하는 반경
- **$S, R$**: 파티클의 스케일 및 회전 행렬
- **직관**: 불투명한 파티클은 더 큰 바운딩 박스를 가져야 하고, 거의 투명한 파티클은 작아도 됨

**Adaptive Clamping**: 불투명도에 기반하여 이십면체 크기를 조정(adaptive clamping)합니다. 이를 통해 크고 거의 투명한 파티클의 불필요한 교차 계산을 줄입니다.

### 4.2 레이 트레이싱 렌더러

**알고리즘 개요** (Procedure 1 & 2):

`ray-gen` 프로그램이 BVH에 광선을 쏘아 다음 $k$개의 hits를 수집합니다. `any-hit` 프로그램이 각 hit에 대해 파티클 응답을 평가하고 소팅된 버퍼에 삽입합니다. 렌더링 후 전송 임계값이 충족되거나 모든 파티클이 평가될 때까지 반복합니다.

```
Ray-Gen(o, d, T_min, α_min, k, SceneMin, SceneMax):
  L = (0,0,0)  // 누적 방사량
  T = 1        // 현재 투과율
  τ_curr = 최소 거리 // 씬 시작점
  
  while τ_curr < SceneMax AND T > T_min:
    H = TraceRay(o + τ_curr*d, d, k)  // 다음 k hits 추적
    for each hit in H:
      if α_particle > α_min:
        L_hit, α_hit = ComputeRadiance(o, d, particle)
        L = L + T * α_hit * L_hit
        T = T * (1 - α_hit)
    τ_curr = 마지막 처리된 hit에서 재시작
  
  return L, T
```

이 방법은 파티클을 일관된 순서로 처리하며 투과율을 정확히 추정합니다.

### 4.3 파티클 응답 평가

광선-파티클 교차 후, 광선을 따라 어느 위치 $\tau$에서 파티클 응답을 평가할지 선택해야 합니다. 최대 응답 위치를 해석적으로 계산합니다:

$$\tau_{max} = \underset{\tau}{\text{argmax}} \ \rho(\mathbf{o} + \tau\mathbf{d}) = \frac{(\boldsymbol{\mu} - \mathbf{o})^T \Sigma^{-1} \mathbf{d}}{\mathbf{d}^T \Sigma^{-1} \mathbf{d}} = \frac{-\mathbf{o}_\sigma^T \mathbf{d}_\sigma}{\mathbf{d}_\sigma^T \mathbf{d}_\sigma}$$

**수식 설명**
- **$\tau_{max}$**: 광선 위에서 파티클 응답이 최대가 되는 위치 (= 광선이 Gaussian 중심에 가장 가까운 점)
- **$\mathbf{o}_\sigma = S^{-1}R^T(\mathbf{o} - \boldsymbol{\mu})$**: 스케일/회전 공간으로 변환된 광선 원점
- **$\mathbf{d}_\sigma = S^{-1}R^T\mathbf{d}$**: 변환된 광선 방향
- **직관**: 타원체 공간으로 변환하면, 광선과 가장 가까운 점을 내적으로 쉽게 계산할 수 있음

### 4.4 미분 가능 레이 트레이싱 및 최적화

Forward pass에서 일반 전방 렌더링을 수행한 후, Backward pass에서 동일한 광선들을 재캐스팅하여 공유 버퍼에 원자적 연산(atomic scatter-add)으로 그래디언트를 누적합니다. 3DGS의 최적화 스킴(가지치기, 복제, 분열)을 채택하며, 주요 차이점은 BVH를 매 반복마다 재구성해야 한다는 점입니다.

### 4.5 파티클 커널 함수

표준 3D Gaussian 외에도 세 가지 커널 변형을 지원합니다:

**일반화 Gaussian (degree n=2)**:

$$\tilde{\rho}_n(\mathbf{x}) = \sigma e^{-(\mathbf{x}-\boldsymbol{\mu})^T \Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})^n}$$

**코사인 파동 변조 (Cosine wave modulation)**:

$$\tilde{\rho}_c(\mathbf{x}) = \tilde{\rho}(\mathbf{x})(0.5 + 0.5\cos(\psi \Sigma^{-1}R^T(\mathbf{x}-\boldsymbol{\mu})_i))$$

**수식 설명**
- **$n=2$ 일반화 Gaussian**: 표준 Gaussian보다 중심에 더 집중된 분포 — 더 뾰족하고 경계가 명확한 파티클 표현
- **코사인 변조**: 공간적으로 변화하는 방사량을 가진 파티클 표현 (줄무늬/패턴)
- **커널화 표면(Kernelized surface)**: z 스케일이 0에 가까운 납작한 파티클 — 삼각형 메시처럼 표면을 표현하며 트레이서의 삼각형 BVH와 잘 맞음

---

## Section 5: Experiments & Ablations

**요약**

### 5.1 Novel View Synthesis 벤치마크

**평가 데이터셋**:
- **MipNeRF360**: 4개 실내 + 3개 실외 장면
- **Tanks & Temples**: 대형 야외 장면 (트럭, 기차)
- **Deep Blending**: 실내 장면 (Playroom, Dr Johnson)

**정량적 결과 (Table 1)**:

| 방법 | MipNeRF360 PSNR | MipNeRF360 SSIM | 속도(FPS) |
|------|-----------------|-----------------|-----------|
| 3DGS (checkpoint) | 28.69 | 0.871 | 387 |
| Ours (reference) | 28.71 | 0.871 | 77 |
| Ours | 28.71 | 0.854 | 363 |
| Zip-NeRF | 30.38 | 0.883 | <1 |

- 레이 트레이싱 구현이 3DGS와 동등한 품질을 달성
- 3DGS 래스터화 대비 약 2배 느림 (기본 핀홀 카메라 기준)
- 래스터화 불가능한 왜곡 카메라에서는 품질 우위

**렌더링 성능 비교 (Table 2)**:

| 방법 | MipNeRF360 FPS | Tanks & Temples FPS | Deep Blending FPS |
|------|---------------|---------------------|-------------------|
| 3DGS (checkpoint) | 238 | 319 | 267 |
| Ours (reference) | 55 | 143 | 77 |
| Ours | 78 | 190 | 119 |

### 5.2 레이 트레이싱 분석 및 Ablation

**비교된 추적 알고리즘들**:
- **Naive closest-hit tracing**: 깊이 순서대로 모든 파티클을 재방문 — 가장 느림
- **Slab tracing (Knoll 2019)**: 슬랩 단위로 추적하여 any-hit 통합 — 근사로 품질 저하
- **Multi-layer alpha tracing (Brüll & Grosch 2020)**: k-buffer 활용 — 근사로 미분 불가
- **제안 방법**: 동적 k-hits 추적으로 정확한 순서 보장, 최고 품질과 경쟁력 있는 속도

**파티클 커널 비교 (Table 4)**:

| 커널 | Tanks&Temples PSNR | FPS |
|------|-------------------|-----|
| Gaussian (reference) | 23.03 | 143 |
| Generalized Gaussian (n=2) | 22.68 | 277 |
| 3D Gaussians | 22.70 | 241 |
| Cosine wave modulation | 22.77 | 268 |

일반화 Gaussian(n=2)은 더 조밀한 파티클로 교차 수를 줄여 품질 대비 속도를 약 2배 향상시킵니다.

---

## Section 6: Applications

**요약**

레이 트레이싱의 유연성이 가능하게 하는 다양한 응용을 보여줍니다.

**핵심 개념**

- **반사/굴절/삽입 메시 (Reflections, Refractions, Inserted Meshes)**: 광선을 광학 법칙에 따라 새 방향으로 리다이렉트하여 계속 추적. 추가적인 가속 구조를 메시 면만으로 구성하여 3D Gaussian 파티클과 함께 렌더링 가능.

- **피사계 심도 (Depth of Field)**: 픽셀당 독립적인 레이 샘플을 추적하고 이동 평균으로 디노이즈. 64-256 spp로 실감나는 보케(bokeh) 효과 생성.

- **인공 그림자 (Artificial Shadows)**: 광원을 향해 그림자 레이를 캐스팅하여 빛이 차단되면 해당 파티클의 기여를 어둡게 처리.

- **인스턴싱 (Instancing)**: BVH의 서브트리에 객체의 연결된 복사본을 저장하여 1024개의 트럭 인스턴스를 25 FPS로 렌더링. 래스터화에서는 불가능한 방식.

- **확률적 샘플링 및 노이즈 제거 (Stochastic Sampling & Denoising)**: any-hit 프로그램에서 중요도 샘플링 $q = \tilde{\rho}(\mathbf{x})$를 적용, NVIDIA OptiX 디노이저로 결과 정제.

- **복합 카메라 모델 (Complex Cameras)**: 픽셀마다 독립적인 광선을 생성하므로 fisheye, 롤링 셔터 등 비선형 카메라를 자연스럽게 지원.

- **자율주행 차량 씬 (Autonomous Vehicle Scenes)**: Waymo Open Perception 데이터셋의 9개 씬에서 평가. 롤링 셔터와 카메라 왜곡을 동시에 처리. 3DGS 대비 PSNR +0.14 개선 (29.99 vs 29.83).

---

## Section 7: Discussion

**요약**

### 래스터화 대비 핵심 차이점

| 특성 | 3DGS 래스터화 | 제안 (레이 트레이싱) |
|------|--------------|---------------------|
| 광선 종류 | 단일 뷰포인트에서 발산하는 기본 광선만 | BVH로 임의 방향 광선 가능 |
| 이차 조명 | 지원 불가 | 반사, 굴절, 그림자 등 완전 지원 |
| 카메라 모델 | 핀홀 카메라 전용 | 임의 카메라 모델 (fisheye, rolling shutter 등) |
| Forward 렌더링 속도 | ~2배 빠름 | 기준선 |
| 미분 가능 최적화 | ~2-5배 빠름 | 기준선 |
| 픽셀 외관 | 픽셀 단위 컨볼루션 (anti-aliasing) | 포인트 샘플링 (denoiser 필요) |

### 한계 및 미래 연구

1. **BVH 재구성 비용**: 매 최적화 반복마다 BVH를 재구성해야 하는 추가 비용 발생
2. **최적화 속도**: 래스터화 대비 2-5배 느린 최적화
3. **서브픽셀 외관**: 래스터화와 달리 anti-aliasing을 위해 디노이저 필요
4. **전역 조명 미해결**: 역렌더링, 재조명, 재질 분해는 미래 연구 과제

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **BVH (Bounding Volume Hierarchy)** | 3D 객체들을 계층적 바운딩 박스 트리로 구성하여 광선-객체 교차를 $O(\log N)$으로 가속하는 자료구조 |
| **NVIDIA OptiX** | NVIDIA GPU의 RT Core를 활용한 레이 트레이싱 API. 삼각형, 구, AABB를 BVH에 삽입 가능 |
| **Encapsulating Primitive** | 각 Gaussian 파티클을 감싸는 대리 기하체. 이 논문에서는 늘린 정이십면체(stretched icosahedron)를 사용 |
| **Adaptive Clamping** | 파티클의 불투명도에 기반하여 프리미티브 크기를 동적으로 조정, 불필요한 교차 계산 감소 |
| **k-buffer** | GPU에서 광선 당 $k$개의 hit를 정렬하여 저장하는 버퍼. 반투명 파티클의 in-order 처리에 필수 |
| **Rolling Shutter** | CMOS 센서가 행 단위로 순차 노출하여 움직임이 있을 때 이미지가 왜곡되는 현상. 레이 트레이싱에서는 행마다 다른 타임스탬프의 광선을 생성하여 자연스럽게 시뮬레이션 |
| **Visibility Pruning** | 훈련 뷰에 대한 누적 기여도가 낮은 파티클을 제거하여 파티클 수를 최대 $3\times10^6$ 이하로 유지 |
| **Generalized Gaussian (n=2)** | 표준 Gaussian의 지수부를 제곱하여 더 조밀하고 경계가 뚜렷한 파티클 표현. 교차 수를 줄여 2배 빠른 렌더링 |
| **Importance Sampling** | 확률 분포 $q = \tilde{\rho}(\mathbf{x})$에 따라 파티클 hit를 채택/거부하여 훈련 중 확률적 샘플링 가능 |

---

## 결론 및 시사점

3D Gaussian Ray Tracing은 3DGS의 래스터화 한계를 극복하는 중요한 진전입니다.

**실용적 시사점**:
1. **로봇공학/자율주행**: fisheye 및 롤링 셔터 카메라로 촬영된 데이터를 직접 학습 가능
2. **시각 효과 파이프라인**: 반사, 굴절, 그림자 등 전통적 레이 트레이싱 효과를 NeRF 스타일 씬에 통합
3. **확장 가능성**: 인스턴싱, 글로벌 조명, 역렌더링 등 차세대 응용을 위한 기반

**향후 연구 방향**:
- 역렌더링(Inverse Rendering) 및 재질 추정
- BVH 재구성 비용 최소화
- 글로벌 조명 및 포토리얼리스틱 렌더링
- 더 효율적인 파티클 커널 설계


---

*관련 논문: [3D Gaussian Splatting](/posts/papers/3d-gaussian-splatting/), [3DGUT](/posts/papers/3dgut-enabling-distorted-cameras-and-secondary-rays-in-gaussian-splatting/), [Street Gaussians](/posts/papers/street-gaussians-modeling-dynamic-urban-scenes/)*
