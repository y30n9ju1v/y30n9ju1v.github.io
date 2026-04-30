---
title: "3DGS의 야코비안 vs 3DGUT의 Unscented Transform: 가우시안 투영의 두 가지 방법"
date: 2026-04-30T14:00:00+09:00
draft: false
tags: ["3DGS", "3DGUT", "Jacobian", "Unscented Transform", "Gaussian Splatting", "렌더링", "수학"]
categories: ["컴퓨터 그래픽스"]
description: "3D Gaussian Splatting에서 3D 가우시안을 2D 화면에 투영할 때 야코비안 선형 근사를 쓰는 이유, 그리고 3DGUT가 Unscented Transform으로 이를 대체한 이유를 직관적으로 설명합니다."
---

## 들어가며

3D Gaussian Splatting(3DGS) 코드를 읽다 보면 다음과 같은 주석이 나옵니다:

```python
# Compute 2D covariance via Jacobian of the projection
J = compute_jacobian(...)
cov2D = J @ cov3D @ J.T
```

그리고 3DGUT 논문을 읽으면 이런 문장이 나옵니다:

> "We replace the Jacobian-based linear approximation with the Unscented Transform to better handle the nonlinearity of the fisheye projection."

**야코비안**과 **Unscented Transform**, 둘 다 같은 목적(3D 가우시안 → 2D 가우시안 변환)을 위한 도구입니다. 왜 두 가지 방법이 존재하고, 언제 어떤 걸 써야 할까요?

---

## 1. 핵심 문제: 3D 가우시안을 2D로 투영하기

3DGS에서 각 가우시안은 3D 공간의 타원체입니다. 이것을 화면에 렌더링하려면 **2D 타원**으로 납작하게 눌러야 합니다.

```
3D 가우시안                     2D 가우시안 (화면)
  (x, y, z)     →  투영 함수 f  →    (u, v)
  공분산 Σ₃D                         공분산 Σ₂D
```

3D 가우시안은 두 가지 정보로 정의됩니다:
- **평균** μ₃D = (x, y, z)
- **공분산** Σ₃D (타원체의 크기, 방향, 모양)

투영 후에도 결과가 가우시안이 되려면 **Σ₂D**를 어떻게 계산할지가 핵심입니다.

문제는 투영 함수 `f`가 **비선형**이라는 점입니다.

---

## 2. 비선형 함수에 가우시안을 통과시키면?

### 선형 함수의 경우 (쉬운 경우)

함수 `f`가 선형이면(행렬 곱) 가우시안 변환은 정확하게 계산됩니다:

```
μ_out = A · μ_in
Σ_out = A · Σ_in · Aᵀ
```

예: 카메라 뷰 행렬(회전 + 이동)은 선형이므로 이 공식이 정확합니다.

### 비선형 함수의 경우 (어려운 경우)

**원근 투영** `f(x, y, z) = (x/z, y/z)`은 비선형입니다. 가우시안을 이 함수에 통과시키면 결과는 **더 이상 완벽한 가우시안이 아닙니다.**

```
입력: 구형 가우시안 (3D 공간)
출력: 찌그러진 형태 (이론적으로는 가우시안이 아님)
```

실제 렌더링에서는 이 결과를 "가우시안으로 근사"해야 합니다. 이 근사 방법이 **야코비안**과 **Unscented Transform**의 차이입니다.

---

## 3. 방법 1: 야코비안 선형 근사 (3DGS)

### 아이디어

비선형 함수를 **평균점 근처에서 선형으로 근사**합니다. 마치 곡선을 접선으로 근사하는 것과 같습니다.

```
실제 함수 f(x)   ≈   f(μ) + J · (x - μ)
      ↑                    ↑
  비선형 곡선           평균점에서의 접선 (선형)
```

**야코비안(Jacobian) J**는 이 접선의 기울기 행렬입니다:

```
J = ∂f/∂x |_{x=μ}    (평균점 μ에서 편미분)
```

### 3DGS에서의 야코비안 계산

원근 투영 `f(x, y, z) = (x/z, y/z)`의 야코비안:

```
        [∂(x/z)/∂x  ∂(x/z)/∂y  ∂(x/z)/∂z]   [1/z   0   -x/z²]
J = ∂f/∂(x,y,z) = [∂(y/z)/∂x  ∂(y/z)/∂y  ∂(y/z)/∂z] = [ 0   1/z  -y/z²]
```

이를 이용해 2D 공분산을 계산합니다:

```python
def project_covariance_jacobian(cov3D, mean3D, K, W):
    """
    cov3D: 3x3 공분산 행렬
    mean3D: 3D 평균 위치 (카메라 좌표계)
    K: 카메라 내부 행렬
    W: 뷰 행렬
    """
    t = W @ mean3D  # 카메라 좌표계로 변환
    
    # 원근 투영의 야코비안
    tx, ty, tz = t
    J = np.array([
        [1/tz,    0, -tx/tz**2],
        [   0, 1/tz, -ty/tz**2],
    ])
    
    # 카메라 좌표계의 공분산
    cov_cam = W @ cov3D @ W.T
    
    # 야코비안으로 2D 공분산 계산
    cov2D = J @ cov_cam @ J.T  # 2x2 행렬
    
    return cov2D
```

### 야코비안의 장점과 한계

**장점:**
- 계산이 빠름 (행렬 곱 몇 번)
- 일반 핀홀 카메라에서 충분히 정확

**한계:**
- 평균점에서의 **1차 근사**이므로 함수가 많이 휘어진 곳에서는 오차 발생
- **어안 렌즈(fisheye)** 같은 심하게 비선형인 투영에서 크게 부정확해짐
- 가우시안이 **카메라 가까이** 있을 때(z가 작을 때) 투영이 강하게 휘어 오차 증가

```
정확도 저하 조건:
  - 넓은 화각 (fisheye, 180°)
  - 가우시안이 z축 기준으로 비대칭
  - 가우시안이 카메라에 매우 가까울 때
```

---

## 4. 방법 2: Unscented Transform (3DGUT)

### 아이디어

"함수를 근사하지 말고, **가우시안에서 대표 샘플 몇 개를 뽑아서** 실제 함수에 통과시키고, 그 결과로 새로운 가우시안을 피팅하자."

이 샘플 포인트들을 **Sigma Points(시그마 포인트)**라고 부릅니다.

```
① 가우시안에서 시그마 포인트 선택 (결정론적, 랜덤 아님)
② 각 시그마 포인트를 실제 비선형 함수 f에 통과
③ 변환된 시그마 포인트들로 새 가우시안의 평균·공분산 추정
```

### 시그마 포인트 선택 방법

n차원 가우시안 (μ, Σ)에서 **2n + 1개**의 시그마 포인트를 뽑습니다:

```python
import numpy as np

def compute_sigma_points(mu, Sigma, alpha=1e-3, beta=2.0, kappa=0.0):
    """
    mu: 평균 벡터 (n,)
    Sigma: 공분산 행렬 (n, n)
    반환: sigma_points (2n+1, n), 가중치 Wm, Wc
    """
    n = len(mu)
    lam = alpha**2 * (n + kappa) - n  # 스케일링 파라미터
    
    # Cholesky 분해: Sigma = L @ L.T
    L = np.linalg.cholesky((n + lam) * Sigma)
    
    sigma_points = np.zeros((2*n + 1, n))
    sigma_points[0] = mu                          # 중심점
    for i in range(n):
        sigma_points[i + 1]     = mu + L[:, i]   # +방향
        sigma_points[i + 1 + n] = mu - L[:, i]   # -방향
    
    # 평균 복원 가중치
    Wm = np.full(2*n + 1, 1.0 / (2 * (n + lam)))
    Wm[0] = lam / (n + lam)
    
    # 공분산 복원 가중치
    Wc = Wm.copy()
    Wc[0] += (1 - alpha**2 + beta)
    
    return sigma_points, Wm, Wc


def unscented_transform(mu, Sigma, f, **ukf_params):
    """
    비선형 함수 f를 통과한 가우시안의 평균과 공분산 추정
    """
    sigma_pts, Wm, Wc = compute_sigma_points(mu, Sigma, **ukf_params)
    
    # ② 각 시그마 포인트를 실제 함수에 통과
    transformed = np.array([f(pt) for pt in sigma_pts])
    
    # ③ 가중 평균으로 새 평균 계산
    mu_out = np.sum(Wm[:, None] * transformed, axis=0)
    
    # ③ 가중 외적으로 새 공분산 계산
    diff = transformed - mu_out
    Sigma_out = sum(Wc[i] * np.outer(diff[i], diff[i])
                    for i in range(len(sigma_pts)))
    
    return mu_out, Sigma_out
```

### 3DGUT에서의 적용

```python
def project_covariance_unscented(cov3D, mean3D, projection_fn):
    """
    projection_fn: 임의의 투영 함수 (fisheye, equirectangular 등)
    """
    mu_2d, cov_2d = unscented_transform(
        mu=mean3D,
        Sigma=cov3D,
        f=projection_fn,   # 비선형 함수를 그대로 넘김
    )
    return mu_2d, cov_2d


# 예: fisheye 투영 (야코비안으로는 부정확)
def fisheye_projection(point_3d):
    x, y, z = point_3d
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(r, z)          # 입사각
    phi = np.arctan2(y, x)            # 방위각
    rho = theta                        # equidistant fisheye 모델
    return np.array([rho * np.cos(phi), rho * np.sin(phi)])
```

### Unscented Transform의 장점과 한계

**장점:**
- **2차 정확도**: 야코비안(1차)보다 더 정확한 근사
- **임의 투영 모델에 적용 가능**: fisheye, equirectangular, 어떤 비선형 함수든 함수만 바꾸면 됨
- **야코비안을 손으로 구할 필요 없음**: 해석적 미분 불필요

**한계:**
- 시그마 포인트 수가 `2n + 1`개 → 함수를 여러 번 평가해야 함
- 야코비안보다 계산 비용이 더 높음

---

## 5. 두 방법의 비교

```
                  야코비안 근사              Unscented Transform
─────────────────────────────────────────────────────────────────
근사 방식       함수를 선형화              함수에 샘플 통과
정확도          1차 (테일러 1항)           2차 (테일러 2항까지)
계산 비용       낮음 (행렬 곱)             높음 (2n+1번 함수 평가)
구현 난이도     해석적 미분 필요           함수 블랙박스로 사용 가능
핀홀 카메라     충분히 정확               오버스펙
어안 렌즈       부정확 (오차 큼)          정확
광각 (>120°)    주의 필요                 안정적
카메라 근접     오차 발생                 상대적으로 안정
```

### 직관적 비유

| | 야코비안 | Unscented Transform |
|---|---|---|
| 비유 | 곡선 도로를 직선으로 근사해서 달림 | 곡선 도로 위에 점들을 찍어 실제 경로를 파악 |
| 일반 도로 | 충분히 정확 | 과하게 정밀 |
| 급커브 (fisheye) | 차선 이탈 가능 | 안전하게 통과 |

---

## 6. 왜 3DGS는 야코비안을 쓰는가?

3DGS가 대상으로 하는 시나리오는 **핀홀 카메라 + 실내/야외 장면**입니다. 이 경우:

1. 투영 함수의 **비선형성이 약함** (z가 충분히 크고 화각이 보통)
2. **수백만 개의 가우시안**을 실시간으로 렌더링해야 함
3. 야코비안 근사 오차가 **시각적으로 무시 가능한 수준**

실시간 성능을 위해 정확도보다 속도를 선택한 것입니다.

---

## 7. 왜 3DGUT는 Unscented Transform을 쓰는가?

3DGUT는 **자율주행, 로보틱스, 드론** 같은 응용을 겨냥합니다. 이 카메라들의 특징:

- **어안 렌즈 (fisheye)**: 화각 180°~220°, 투영 함수가 극도로 비선형
- **넓은 시야각**: 가우시안이 화면 가장자리에서 심하게 찌그러짐
- **정확도가 중요**: 자율주행에서 기하학적 오차는 치명적

야코비안으로는 fisheye 투영의 비선형성을 감당할 수 없기 때문에 Unscented Transform을 도입했습니다.

```
핀홀 (3DGS 대상)          어안 (3DGUT 대상)
     화각 ~60°                화각 ~180°
    투영 곡선 완만            투영 곡선 급격
    야코비안 ✅               야코비안 ❌ → UT 필요
```

---

## 8. 핵심 수식 정리

### 야코비안 방법

```
Σ₂D = J · W · Σ₃D · Wᵀ · Jᵀ

J = ∂f/∂x |_{x=μ}    (투영 함수의 야코비안, 평균점에서 평가)
W = 뷰 행렬 (회전)
```

### Unscented Transform

```
시그마 포인트: χᵢ = μ + (√((n+λ)Σ))ᵢ   (i = 1,...,n)
              χ₋ᵢ = μ - (√((n+λ)Σ))ᵢ  (i = 1,...,n)
              χ₀ = μ

변환: yᵢ = f(χᵢ)

평균: μ_y = Σᵢ Wᵢᵐ · yᵢ

공분산: Σ_y = Σᵢ Wᵢᶜ · (yᵢ - μ_y)(yᵢ - μ_y)ᵀ
```

---

## 마치며

| | 야코비안 (3DGS) | Unscented Transform (3DGUT) |
|---|---|---|
| **핵심 아이디어** | 함수를 직선으로 근사 | 대표 샘플로 함수 통과 |
| **적합한 상황** | 핀홀, 약한 비선형성 | 어안, 강한 비선형성 |
| **속도** | 빠름 | 상대적으로 느림 |
| **정확도** | 1차 근사 | 2차 근사 |

3DGS의 야코비안은 "빠르고 충분히 좋은" 선택이고, 3DGUT의 Unscented Transform은 "더 느리지만 더 정확한" 선택입니다. 두 방법 모두 같은 문제(비선형 함수를 통과한 가우시안의 통계 추정)를 다루며, 트레이드오프가 다를 뿐입니다.

어안 렌즈나 비표준 카메라 모델로 3DGS를 확장하려 한다면, Unscented Transform이 자연스러운 선택입니다.

---

## 참고자료

- **3DGS 논문**: Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (SIGGRAPH 2023)
- **3DGUT 논문**: Huang et al., "3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting" (2024)
- **Unscented Kalman Filter**: Julier & Uhlmann, "A New Extension of the Kalman Filter to Nonlinear Systems" (1997)
- **3D 데이터 표현**: [3D 데이터 표현 방식 총정리]({{< ref "3d-data-representations.md" >}})
