---
title: "3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting"
date: 2026-04-10T08:30:00+09:00
draft: false
categories: ["Papers"]
---

## 개요

- **저자**: Qi Wu, Janick Martinez Esturo, Ashkan Mirzaei, Nicolas Moenne-Loccoz, Zan Gojcic (NVIDIA, University of Toronto)
- **발행년도**: 2025
- **arXiv**: 2412.12507v2
- **주요 내용**: 3D Gaussian Splatting(3DGS)을 확장하여 왜곡된 카메라와 이차 광선(secondary rays)을 지원하는 방법을 제안합니다. Unscented Transform(UT)을 사용하여 임의의 비선형 카메라 프로젝션을 정확하게 처리하면서도 실시간 렌더링 효율성을 유지합니다.

## 목차

1. Introduction & Motivation
2. Related Work
3. Preliminaries: 3D Gaussian Splatting 기초
4. Method: Unscented Transform 기반 접근
5. Experiments and Results
6. Applications: Complex Cameras & Secondary Rays
7. Limitations and Future Work

---

## 1. Introduction & Motivation

### 배경

3D Gaussian Splatting(3DGS)은 소비자 수준의 하드웨어에서 고충실도의 실시간 렌더링을 가능하게 하는 획기적인 방법입니다. 그러나 래스터화 기반 공식화로 인해 두 가지 심각한 제한이 있습니다:

1. **이상적인 핀홀 카메라만 지원**: 광학 왜곡이나 롤링 셔터를 처리할 수 없음
2. **이차 광선 불가능**: 반사, 굴절, 그림자 같은 현상을 표현할 수 없음

### 문제 정의

**EWA Splatting의 문제점**

3DGS는 가우시안 입자를 카메라 이미지 평면에 투영하기 위해 EWA Splatting을 사용합니다:

$$\mathbf{\Sigma'} = \mathbf{J}_{[:2,:3]}\mathbf{W}\mathbf{\Sigma}\mathbf{W}^T\mathbf{J}_{[:2,:3]}^T$$

**수식 설명**
- **$\mathbf{\Sigma'}$**: 이미지 좌표계의 2D 가우시안 공분산 행렬 (우리가 화면에서 보는 타원)
- **$\mathbf{J}$**: 비선형 프로젝션 함수의 야코비안 (1차 테일러 근사)
- **$\mathbf{W}$**: 월드에서 카메라 좌표계로의 변환 행렬
- **$\mathbf{\Sigma}$**: 원본 3D 가우시안의 공분산 행렬
- **문제**: 야코비안 근사는 고차 항을 무시하므로 왜곡이 클수록 오차가 증가합니다.

### 3DGUT의 핵심 아이디어

**문제를 역으로 생각하기**:
- 기존: 프로젝션 함수를 근사 (선형화)
- **제안**: 입자 자체를 근사 (Sigma points로 표현)

Unscented Transform을 사용하면:
- ✅ 어떤 비선형 프로젝션 함수에도 정확히 적용 가능
- ✅ 야코비안 도출 불필요 (일반화 가능)
- ✅ 롤링 셔터 같은 시간 의존적 효과를 쉽게 처리 가능
- ✅ 래스터화 효율성 유지 (레이 트레이싱보다 3-4배 빠름)

---

## 2. Related Work

### Neural Radiance Fields (NeRFs)

NeRF는 좌표 기반 신경망으로 장면을 암호화하고 볼륨 렌더링으로 신규 뷰를 합성합니다. 하지만 계산 비용이 많아 실시간 성능을 달성하기 어렵습니다.

### 3D Gaussian Splatting

3DGS는 NeRF의 이후 작업으로, 레이 마칭 대신 효율적인 래스터화를 사용합니다:
- 구(sphere) 대신 비등방 3D 가우시안 사용
- 실시간 렌더링 가능 (265+ FPS)
- 높은 시각적 충실도

**단점**: 이상적인 핀홀 카메러로 제한, 이차 광선 불가능

### Ray Tracing 기반 접근 (3DGRT, EVER)

최근 연구들이 래스터화를 벗어나 레이 트레이싱으로 3D 가우시안을 렌더링하는 방법을 제안했습니다:
- ✅ 복잡한 카메라 모델 자연스럽게 지원
- ✅ 반사, 굴절, 그림자 가능
- ❌ 렌더링 속도 3-4배 느림 (50-200 FPS 수준)

### Unscented Transform (UT)

Kalman Filter 문헌에서 비롯된 기법으로, 비선형 변환을 거친 확률변수의 통계를 계산합니다:

**핵심 원리**:
1. 분포를 Sigma points로 근사
2. 각 Sigma point를 정확히 변환
3. 변환된 점들로부터 목표 영역의 통계 재계산

---

## 3. Preliminaries: 3D Gaussian Splatting 기초

### 3D 가우시안 표현

각 입자의 응답 함수:

$$\rho(\mathbf{x}) = \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\mathbf{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

**수식 설명**
- **$\rho(\mathbf{x})$**: 3D 공간의 점 $\mathbf{x}$에서 이 가우시안의 강도 (0 = 투명, 1 = 완전 불투명)
- **$\boldsymbol{\mu}$**: 가우시안의 중심 위치
- **$\mathbf{\Sigma}$**: 공분산 행렬 (입자의 크기와 방향 결정)
- **$(\mathbf{x}-\boldsymbol{\mu})^T\mathbf{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})$**: Mahalanobis 거리 (방향을 고려한 거리)

**수치적 안정성**:
$$\mathbf{\Sigma} = \mathbf{R}\mathbf{S}\mathbf{S}^T\mathbf{R}^T$$

- **$\mathbf{R}$**: 회전 행렬 (쿼터니언 $q$로 저장)
- **$\mathbf{S}$**: 스케일 행렬 (벡터 $s$로 저장)

각 입자는 또한 다음을 포함합니다:
- **불투명도**: $\sigma$ (0 = 완전 투명, 1 = 완전 불투명)
- **시점 의존 방사값**: 구면 조화 함수로 표현

### 볼륨 입자 렌더링 방정식

카메라 광선 $r(\tau) = o + \tau d$ 방향으로의 색상:

$$c(o, d) = \sum_{i=1}^N c_i(d) \alpha_i \prod_{j=1}^{i-1}(1-\alpha_j)$$

**수식 설명**
- **$c(o, d)$**: 최종 렌더링 색상 (우리가 화면에서 보는 색)
- **$c_i(d)$**: i번째 입자의 색상 (시점에 따라 다름)
- **$\alpha_i$**: i번째 입자의 불투명도 (0 = 투명, 1 = 불투명)
- **$\sum$**: 모든 입자를 앞에서 뒤로 누적
- **$\prod_{j=1}^{i-1}(1-\alpha_j)$**: 앞의 모든 입자를 지나온 빛의 투과율
  - 예: 앞의 입자들이 50% 불투명이면, 뒤 입자는 50%만 보임
- **$i$**: 깊이 순서대로 정렬된 입자 인덱스

이 공식은 "forward alpha blending"으로도 알려진 표준 합성 공식입니다.

---

## 4. Method: Unscented Transform 기반 접근

### 4.1 Unscented Transform을 사용한 프로젝션

**핵심 아이디어**: 3D 가우시안을 **Sigma points**라는 신중하게 선택된 점들로 근사합니다.

#### Sigma Points 정의

7개의 Sigma point (3D 가우시안: 2N+1 = 7개):

$$\mathbf{x}_i = \begin{cases}
\boldsymbol{\mu} & \text{for } i = 0 \\
\boldsymbol{\mu} + \sqrt{(3 + \lambda)\mathbf{\Sigma}}_{[i]} & \text{for } i = 1, 2, 3 \\
\boldsymbol{\mu} - \sqrt{(3 + \lambda)\mathbf{\Sigma}}_{[i-3]} & \text{for } i = 4, 5, 6
\end{cases}$$

**수식 설명**
- **$\boldsymbol{\mu}$**: 가우시안의 중심 (평균)
- **$\mathbf{\Sigma}$**: 공분산 행렬 (분포의 넓이와 방향)
- **$\sqrt{(3+\lambda)\mathbf{\Sigma}}_{[i]}$**: 공분산 행렬의 i번째 열의 제곱근
  - 이것들은 평균 주변에 "원" 형태로 배치된 점들입니다
- **$\lambda = \alpha^2(3+\kappa) - 3$**: 점들의 분산을 제어하는 하이퍼파라미터
  - $\alpha = 1.0$: 표준 선택
  - $\kappa = 0.0$: 일반적인 설정

**Sigma Points의 의미**:
- 중심 1개 + 주축 방향 각 6개 = 총 7개의 대표점
- 이 점들로 원본 가우시안의 1차, 2차 모멘트를 정확히 표현

#### 가중치 (Weights)

각 Sigma point에 할당되는 가중치:

$$w_i^\mu = \begin{cases}
\frac{\lambda}{3 + \lambda} & \text{for } i = 0 \\
\frac{1}{2(3 + \lambda)} & \text{for } i = 1, \ldots, 6
\end{cases}$$

$$w_i^\Sigma = \begin{cases}
\frac{\lambda}{3 + \lambda} + (1 - \alpha^2 + \beta) & \text{for } i = 0 \\
\frac{1}{2(3 + \lambda)} & \text{for } i = 1, \ldots, 6
\end{cases}$$

**수식 설명**
- **$w_i^\mu$**: 평균 계산에 사용되는 가중치
- **$w_i^\Sigma$**: 공분산 계산에 사용되는 가중치
- **$\beta = 2.0$**: 초과 쿠르토시스 보정 항 (정규분포면 2.0)
- 이 가중치들은 Sigma point들이 원본 분포의 통계를 정확히 재현하도록 설계됨

#### 프로젝션 및 2D 원뿔 계산

각 Sigma point를 독립적으로 프로젝션:

$$\mathbf{v}_{x_i} = g(\mathbf{x}_i)$$

여기서 $g$는 어떤 임의의 복잡한 카메라 모델이든 가능합니다.

2D 가우시안의 평균과 공분산:

$$\mathbf{v}_\mu = \sum_{i=0}^6 w_i^\mu \mathbf{v}_{x_i}$$

$$\mathbf{\Sigma'} = \sum_{i=0}^6 w_i^\Sigma (\mathbf{v}_{x_i} - \mathbf{v}_\mu)(\mathbf{v}_{x_i} - \mathbf{v}_\mu)^T$$

**수식 설명**
- **$\mathbf{v}_{x_i}$**: i번째 Sigma point의 이미지 좌표 프로젝션
- **$\mathbf{v}_\mu$**: 프로젝션된 가우시안의 평균 (화면에서의 중심)
- **$\mathbf{\Sigma'}$**: 프로젝션된 2D 가우시안의 공분산 (화면에서의 타원 형태)
- 이 계산은 어떤 비선형 함수 $g$에도 정확하게 적용됨

**EWA vs UT 비교**:

| 측면 | EWA (3DGS) | UT (3DGUT) |
|------|-----------|-----------|
| 접근법 | 프로젝션 함수 선형화 | 입자 자체 근사 |
| 야코비안 필요 | 예 (매번 도출) | 아니오 |
| 왜곡 카메라 | 부정확 | 정확 |
| 롤링 셔터 | 불가능 | 가능 |
| 속도 | 빠름 | 약간 느림 |
| KL 발산 | 증가함 (왜곡 커질수록) | 일정함 |

### 4.2 3D에서의 입자 응답 평가

기존 3DGS와 다르게, 우리는 입자 응답을 **3D 공간에서 직접** 평가합니다.

광선을 따라 최대 응답점:

$$\tau_{\max} = \frac{(\boldsymbol{\mu} - \mathbf{o})^T\mathbf{\Sigma}^{-1}\mathbf{d}}{\mathbf{d}^T\mathbf{\Sigma}^{-1}\mathbf{d}}$$

또는 표준 가우시안 공간에서:

$$\tau_{\max} = -\frac{\mathbf{o}_g^T\mathbf{d}_g}{\mathbf{d}_g^T\mathbf{d}_g}$$

**수식 설명**
- **$\tau_{\max}$**: 광선 원점 $\mathbf{o}$에서 출발하여 이 거리를 가면 입자 응답이 최대
- **$\boldsymbol{\mu}$**: 입자의 중심
- **$\mathbf{d}$**: 광선의 방향 (단위 벡터)
- **$\mathbf{\Sigma}^{-1}$**: 공분산의 역행렬 (입자의 역함수 모양)
- **$\mathbf{o}_g = \mathbf{S}^{-1}\mathbf{R}^T(\mathbf{o} - \boldsymbol{\mu})$**: 표준 가우시안 좌표계의 광선 원점
- **직관**: 광선이 입자의 "가장 두꺼운 부분"을 지날 때의 거리

**2D vs 3D 평가의 차이**:
- **3DGS (2D)**: 투영된 2D 가우시안에서 응답을 계산 → 프로젝션 함수를 통한 역전파 필요
- **3DGUT (3D)**: 3D 입자에서 응답을 계산 → 프로젝션 함수를 거치지 않음

**이점**:
- ✅ 프로젝션 근사 오류 제거
- ✅ 수치적 안정성 개선
- ✅ 임의의 카메라 모델에 자연스럽게 적용 가능

### 4.3 입자 정렬

볼륨 렌더링의 정확성을 위해 입자들을 깊이 순서대로 정렬해야 합니다.

3DGRT와의 일치를 위해, **다중 계층 알파 블렌딩 (MLAB, Multi-Layer Alpha Blending)**을 사용:

1. 각 광선에서 $k$개의 가장 먼 입자들을 저장 (보통 $k=16$)
2. 가장 가까운 입자들은 점진적으로 알파 블렌딩
3. 블렌딩 부분의 투과율이 소실되면 중단

이렇게 함으로써 3DGUT를 3DGRT 방식으로도 렌더링할 수 있어서 **하이브리드 렌더링**이 가능합니다.

---

## 5. Experiments and Results

### 5.1 표준 벤치마크

#### MipNeRF360 Dataset

일반적인 핀홀 카메라로 촬영된 장면들:

**정량적 결과**:

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FPS ↑ |
|--------|--------|--------|---------|-------|
| ZipNeRF | 28.54 | 0.828 | 0.219 | 0.2 |
| 3DGS | 27.26 | 0.803 | 0.240 | 347 |
| **Ours** | **27.26** | **0.810** | **0.218** | **265** |
| StopThePop | 27.14 | 0.804 | 0.235 | 340 |
| 3DGRT | 23.64 | 0.837 | 0.196 | 476 |
| EVER | 23.21 | 0.841 | 0.178 | 277 |
| Ours (sorted) | 27.20 | 0.812 | 0.215 | 200 |

**결과 해석**:
- ✅ 우리의 방법은 3DGS와 비슷한 품질 달성
- ✅ 래스터화 기반 방법 중 가장 빠름 (265 FPS)
- ✅ 레이 트레이싱 방법들(3DGRT, EVER)보다 훨씬 빠름

#### Tanks & Temples Dataset

두 개의 대규모 실외 장면 (Truck, Train):
- 우리 방법과 3DGS의 성능이 유사하며, 레이 트레이싱 방법들을 능가

### 5.2 왜곡된 카메라 벤치마크

#### Scannet++ (Fisheye Camera)

어안 카메라로 촬영된 실내 데이터:

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | N. Gaussians ↓ |
|--------|--------|--------|---------|-----------------|
| 3DGS† | 22.76 | 0.798 | / | 1.31M |
| FisheyeGS† | 27.86 | 0.897 | / | 1.25M |
| FisheyeGS | 28.15 | 0.901 | 0.261 | 1.07M |
| **Ours (sorted)** | **29.11** | **0.910** | **0.252** | **0.38M** |

**놀라운 결과**:
- 특정 어안 카메라를 위해 야코비안을 유도한 FisheyeGS를 능가!
- 필요한 가우시안 개수도 절반 이하 (1.07M → 0.38M)
- 일반적인 UT 공식이 특수화된 선형화보다 나음

**이유**: 우리의 일반화된 접근이 카메라 왜곡을 더 정확하게 처리

#### Waymo (Autonomous Driving)

왜곡된 카메라 + 롤링 셔터 효과:

| Method | PSNR ↑ | SSIM ↑ |
|--------|--------|--------|
| 3DGS | 29.83 | 0.917 |
| 3DGRT | 29.99 | 0.897 |
| **Ours (sorted)** | **30.16** | **0.900** |

**특징**:
- 롤링 셔터 효과를 정확하게 처리
- 3DGRT(전문 레이 트레이싱)와 경쟁력 있는 품질

### 5.3 KL 발산 분석

Monte Carlo 샘플링을 기준으로 EWA vs UT의 투영 정확도 비교:

**핀홀 카메라**: 둘 다 유사하지만 UT가 약간 나음

**어안 카메라 (FoV 60° → 260°)**:
- EWA (3DGS): FoV 증가에 따라 KL 발산 증가 ↑
- UT (3DGUT): KL 발산 일정 → 더 정확하고 안정적 ✅

**방사 왜곡 + 롤링 셔터**:
- EWA: 왜곡 계수 증가에 따라 급격히 악화
- UT: 거의 일정한 오차 유지

→ **UT는 근본적으로 더 정확한 투영 방법**

---

## 6. Applications: Complex Cameras & Secondary Rays

### 6.1 복잡한 카메라

#### 1. 왜곡된 카메라 모델

핀홀로 학습한 장면을 다양한 왜곡 카메라 모델로 렌더링 가능:

```
핀홀 학습 → 어안 렌더링 (자동)
         → 방사 왜곡 렌더링
         → 접선 왜곡 렌더링
         → 임의의 커스텀 모델 렌더링
```

**코드 수정 없음** - UT는 모든 프로젝션 함수를 지원!

#### 2. 롤링 셔터

카메라가 이동하면서 각 행이 다른 시간에 노출되는 현상:

$$\mathbf{x}_i^{(t)} = \mathbf{x}_i - \mathbf{v} \cdot (t_{\text{top}} - t_i)$$

**수식 설명**
- **$\mathbf{x}_i^{(t)}$**: i번째 행의 카메라 위치 (시간 $t$)
- **$\mathbf{v}$**: 카메라의 이동 속도
- **$t_i$**: i번째 행의 노출 시간
- **$t_{\text{top}}$**: 맨 위 행의 노출 시간

기존 3DGS:
- 불가능 (선형화 공식에 시간 변수 추가 불가)

3DGUT:
- 각 Sigma point마다 다른 외부 행렬 사용
- 자연스럽게 시간 의존성 처리 가능 ✅

**결과 (Figure 7)**:
- 3DGS: 심한 찢김 현상 (tearing artifacts)
- 3DGRT: 정확하지만 느림
- **3DGUT: 정확하면서도 빠름** ✅

### 6.2 이차 광선 및 조명 효과

#### 3DGRT와의 렌더링 공식 일치

3DGUT를 3DGRT와 동일하게 렌더링할 수 있도록 설계했습니다:

1. **입자 기여도 결정**: Sigma points의 2D 원뿔로 판정
2. **입자 순서**: MLAB로 깊이 정렬
3. **응답 평가**: 3D 공간에서 직접 계산

→ 동일한 3D 표현을 **래스터화와 레이 트레이싱으로 모두 렌더링 가능**

#### 하이브리드 렌더링

**전략**:
```
1차 광선 → 래스터화 (빠름, 3DGUT)
         ↓
각 픽셀의 가장 가까운 교점 결정
         ↓
2차 광선 → 레이 트레이싱 (3DGRT)
         ↓
반사, 굴절 계산
         ↓
최종 이미지 합성
```

**결과**:
- 반사 (reflections) 정확하게 표현
- 굴절 (refractions) 정확하게 표현
- 레이 트레이싱만큼 정확하지만 훨씬 빠름

---

## 7. Limitations and Future Work

### 현재 제한사항

1. **약간의 속도 저하**
   - 3DGS: 347 FPS
   - 3DGUT: 265 FPS
   - 이유: UT 평가 + 3D 입자 평가의 추가 복잡도
   - 그러나 여전히 실시간 (≥200 FPS)

2. **큰 왜곡에서의 형태 왜곡**
   - 극단적인 왜곡 하에서 프로젝션 타원이 2D 가우시안을 벗어날 수 있음
   - 입자 기여도 판정 정확도 감소

3. **겹친 가우시안 렌더링**
   - 단일 샘플로 각 입자를 평가하므로 겹친 입자들을 정확하게 처리 어려움
   - EVER 같은 방법들이 가능성 제시

### 향후 연구 방향

- 자율주행 및 로봇공학: 왜곡 카메라가 필수적
- 역 렌더링 및 재조명: 3DGRT와의 정렬이 기반 제공

---

## 핵심 개념 정리

### 1. Unscented Transform (UT)

**정의**: 비선형 변환을 거친 확률분포의 통계를 정확히 추정하는 방법

**작동 원리**:
1. 원본 분포를 Sigma points로 근사 (7개의 대표점)
2. 각 점을 비선형 함수에 통과
3. 변환된 점들로부터 새로운 분포의 평균과 공분산 계산

**장점**:
- 선형화 오차 없음
- 일반화 가능 (모든 비선형 함수에 적용)
- 야코비안 계산 불필요

### 2. Sigma Points

**정의**: 확률분포의 1차, 2차 모멘트를 정확히 표현하는 선택된 점들

**3D 가우시안의 경우**:
- 중심 1개
- 주축 방향 3개 쌍 (+ 방향, - 방향)
- 총 7개

**의미**: 원본 분포와 동일한 평균과 공분산을 가지는 점 집합

### 3. EWA Splatting vs UT Projection

| 특성 | EWA | UT |
|------|-----|-----|
| 원리 | 함수 선형화 | 분포 근사 |
| 오차 | 고차 항 무시 | 없음 (정확) |
| 왜곡 증가 | 오차 증가 ↑ | 오차 일정 |
| 롤링 셔터 | 불가능 | 가능 |
| 복잡도 | 낮음 | 중간 |

### 4. 2D 원뿔 (2D Conic)

**정의**: 3D 가우시안이 이미지 평면에 투영된 2D 타원

**역할**:
- 어떤 입자들이 어떤 픽셀에 영향을 주는지 빠르게 판정
- 모든 입자를 검사할 필요 없음 (가속 구조)

### 5. 알파 합성 (Alpha Compositing)

$$c = \sum_{i=1}^N c_i \alpha_i \prod_{j=1}^{i-1}(1-\alpha_j)$$

**과정**:
1. 가장 먼 입자부터 시작
2. 색상 $c_i$와 투명도 $\alpha_i$로 누적
3. 앞 입자들의 투명도 고려 ($\prod(1-\alpha_j)$)

**직관**: "유리 여러 장을 겹쳐서 보는 것"

### 6. 하이브리드 렌더링

**전략**:
- **1차 광선**: 래스터화 (빠름)
- **2차 광선**: 레이 트레이싱 (정확함)
- **결합**: 두 방법의 장점 활용

---

## 결론 및 시사점

### 주요 기여

1. **일반화된 카메라 모델 지원**
   - 어안, 방사 왜곡, 접선 왜곡 등 모두 지원
   - 새로운 카메라 모델 추가 시 코드 수정 불필요

2. **롤링 셔터 자연스러운 처리**
   - 자율주행, 드론 영상에 필수적
   - 기존 방법 불가능, 새로운 가능성 열음

3. **효율성과 정확성의 균형**
   - 레이 트레이싱 수준의 정확성
   - 래스터화 수준의 속도 (200+ FPS)

4. **통일된 렌더링 공식**
   - 같은 표현을 래스터화와 트레이싱으로 렌더링
   - 하이브리드 렌더링 가능

### 실무적 시사점

**자율주행/로봇공학**:
- 실제 카메라의 왜곡을 정확하게 처리 가능
- 시뮬레이션 → 실제 환경 전환 개선

**컴퓨터 그래픽스**:
- 복잡한 광학 효과를 래스터화로 처리 가능
- 실시간 반사/굴절 렌더링 가능

**신경 장면 표현**:
- NeRF 다음 세대인 3DGS의 한계 극복
- 더 일반적인 장면 재구성 가능

### 기술적 통찰

**왜 UT가 효과적인가?**

선형화는 "한 점에서의 기울기"만 봅니다:
```
실제 곡선: ╱╱╱╱
선형화:   //// (한 점에서의 기울기)
오차:     ╱와 /의 차이
```

UT는 여러 점에서의 실제 값을 봅니다:
```
Sigma points: •  •  •  •  •  •  •
실제 곡선:    ╱╱╱╱
변환 후:      적응적으로 추정
오차:         훨씬 작음
```

→ **선형화 대신 샘플링으로 더 정확하게 근사**

---

## 참고 자료

- **저자 홈페이지**: https://research.nvidia.com/labs/toronto-ai/3DGUT
- **GitHub**: https://github.com/nv-tlabs/3dgrut
- **arXiv**: 2412.12507v2

## 관련 논문

- 3D Gaussian Splatting (Kerbl et al., 2023)
- 3DGRT (Moenne-Loccoz et al., 2025) - Ray Tracing 기반
- FisheyeGS (Li et al., 2024) - 어안 카메라 특화
- Unscented Kalman Filter (Julier & Uhlmann, 1997)
