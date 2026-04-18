---
title: "Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields"
date: 2026-04-17T08:40:00+09:00
draft: false
categories: ["Papers", "Novel View Synthesis"]
tags: ["NeRF", "Novel View Synthesis", "Neural Rendering", "3D Reconstruction"]
---

## 개요

- **저자**: Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, Peter Hedman (Google Research, Harvard University)
- **발행년도**: 2022 (CVPR 2022)
- **arXiv**: 2111.12077
- **주요 내용**: 카메라가 360도 어느 방향으로든 향할 수 있고 콘텐츠가 임의의 거리에 존재하는 **무한 장면(unbounded scene)**을 사실적으로 합성하는 mip-NeRF의 확장판. 비선형 장면 파라미터화, 온라인 디스틸레이션, 새로운 정규화 손실 함수를 통해 mip-NeRF 대비 평균 제곱 오차 57% 감소.

---

## 한계 극복

이 논문은 기존 NeRF 계열 방법들이 **무한(unbounded) 장면**을 다루지 못하는 한계를 극복합니다.

- **기존 한계 1 — 파라미터화 문제**: mip-NeRF는 3D 좌표가 유한 범위 안에 있어야 하므로 카메라가 360도 자유롭게 움직이는 야외 장면에 적용하기 어렵습니다.
- **기존 한계 2 — 학습 비효율**: 크고 상세한 장면은 각 광선마다 대형 MLP를 반복 조회해야 하므로 학습이 매우 느립니다.
- **기존 한계 3 — 모호성(Ambiguity)**: 무한 장면에서는 표면이 임의의 거리에 존재할 수 있어 소수의 광선만이 한 지점을 관측하므로, 2D 이미지에서 3D를 복원할 때 본질적인 모호성이 커집니다.
- **이 논문의 접근 방식**: (1) contract 함수를 이용한 비선형 장면 파라미터화로 무한 공간을 유한 구 안에 매핑, (2) 소형 Proposal MLP + 대형 NeRF MLP의 온라인 디스틸레이션 구조로 효율 향상, (3) 새로운 distortion 정규화 손실로 floater·background collapse 아티팩트 억제.

---

## 목차

1. Preliminaries: mip-NeRF
2. Scene and Ray Parameterization
3. Coarse-to-Fine Online Distillation
4. Regularization for Interval-Based Models
5. Optimization
6. Results
7. Conclusion

---

## Chapter 1: Preliminaries — mip-NeRF 복습

**요약**

mip-NeRF는 NeRF의 앨리어싱(계단 현상) 문제를 해결하기 위해 단일 3D 점 대신 **원뿔형 절두체(conical frustum)**를 따라 Gaussian을 사용합니다. 각 구간 $T_i = [t_i, t_{i+1}]$에 대해 평균 $\mu$와 공분산 $\Sigma$를 계산하고, 통합된 위치 인코딩(IPE)으로 특징 벡터를 만들어 MLP에 입력합니다.

**핵심 개념**

- **Conical Frustum (원뿔 절두체)**: 카메라 광선을 따라 원뿔 모양으로 잘라낸 구간. 픽셀 크기와 초점 거리에 따라 반경이 결정됩니다.
- **Integrated Positional Encoding (IPE)**: 원뿔 절두체 내 Gaussian 분포를 위치 인코딩의 기댓값으로 표현하여 앤티앨리어싱을 달성합니다.
- **Coarse-to-Fine 샘플링**: 먼저 균일하게 "coarse" 샘플링 후 해당 가중치 히스토그램에 따라 "fine" 샘플을 추가로 뽑는 방식으로 샘플 효율을 높입니다.

**핵심 수식**

$$\gamma(\mu, \Sigma) = \left\{ \begin{bmatrix} \sin(2^l \mu) \exp(-2^{2l-1} \text{diag}(\Sigma)) \\ \cos(2^l \mu) \exp(-2^{2l-1} \text{diag}(\Sigma)) \end{bmatrix} \right\}_{l=0}^{L-1}$$

**수식 설명**

이 수식은 **통합 위치 인코딩(IPE)**으로, Gaussian 분포를 가진 원뿔 절두체를 주파수 공간으로 변환합니다:
- **$\mu$**: Gaussian의 평균 (공간 위치의 중심)
- **$\Sigma$**: Gaussian의 공분산 (퍼짐 정도 — 공간적 불확실성)
- **$2^l$**: 주파수 레벨. $l$이 클수록 고주파(세밀한 디테일)를 포착합니다.
- **$\exp(-2^{2l-1} \text{diag}(\Sigma))$**: 공분산이 클수록(물체가 흐릿하게 보일수록) 고주파 성분을 줄여 자동으로 블러 처리 → 앨리어싱 방지.

$$C(\mathbf{r}, \mathbf{t}) = \sum_i w_i c_i, \quad w_i = \left(1 - e^{-\tau_i(t_{i+1}-t_i)}\right) e^{-\sum_{i' < i} \tau_{i'}(t_{i'+1}-t_{i'})}$$

**수식 설명**

볼류메트릭 렌더링 적분의 이산화:
- **$C(\mathbf{r}, \mathbf{t})$**: 광선 $\mathbf{r}$을 따라 렌더링된 픽셀 색상
- **$w_i$**: $i$번째 구간의 투과 가중치 (해당 구간이 색상에 기여하는 비율)
- **$\tau_i$**: $i$번째 구간의 밀도 (얼마나 불투명한가)
- **$(1 - e^{-\tau_i \Delta t})$**: 해당 구간에서 빛이 흡수되는 비율
- **$e^{-\sum_{i' < i} \tau_{i'} \Delta t'}$**: 앞 구간들을 모두 통과한 빛의 투과율 (앞에 뭔가 있으면 뒤가 덜 보임)

---

## Chapter 2: Scene and Ray Parameterization

**요약**

무한 장면을 유한 공간에 표현하기 위해 핵심 기법인 **contract 함수**를 도입합니다. 이 함수는 반경 1 이내의 점은 그대로 두고, 반경 1 밖의 무한 공간을 반경 2인 구 안으로 압축합니다. 또한 광선 거리를 역수(disparity) 공간에서 선형으로 샘플링해, 카메라 근처는 촘촘하게 먼 곳은 성기게 샘플링합니다.

**핵심 개념**

- **contract 함수**: 무한 3D 공간을 유한 구($\|\mathbf{x}\| \le 2$) 안으로 수축하는 비선형 좌표 변환.
- **Disparity Sampling**: 거리 $t$의 역수(시차)에서 선형 샘플링. 카메라에 가까운 물체에 더 많은 샘플을 배치해 해상도를 높임.
- **Kalman Filter 유사성**: contract 함수를 Gaussian에 적용할 때 선형 근사(야코비안)를 사용하며, 이는 확장 칼만 필터(EKF)와 동일한 원리.

**핵심 수식**

$$\text{contract}(\mathbf{x}) = \begin{cases} \mathbf{x} & \|\mathbf{x}\| \le 1 \\ \left(2 - \frac{1}{\|\mathbf{x}\|}\right) \frac{\mathbf{x}}{\|\mathbf{x}\|} & \|\mathbf{x}\| > 1 \end{cases}$$

**수식 설명**

**contract 함수**는 무한 공간을 반경 2인 구 안으로 접어 넣는 핵심 변환입니다:
- **$\|\mathbf{x}\| \le 1$ (구 안쪽)**: 좌표를 그대로 유지 — 가까운 물체는 왜곡 없음.
- **$\|\mathbf{x}\| > 1$ (구 바깥)**: $(2 - 1/\|\mathbf{x}\|)$ 로 스케일 조정. $\|\mathbf{x}\| \to \infty$이면 반경 → 2로 수렴.
- **직관**: 지도에서 세계를 원 안에 그리듯, 무한히 먼 하늘/배경도 구의 표면 근처에 압축해서 표현.

$$s \triangleq \frac{g(t) - g(t_n)}{g(t_f) - g(t_n)}, \quad \hat{t} \triangleq g^{-1}(s \cdot g(t_f) + (1-s) \cdot g(t_n))$$

**수식 설명**

**정규화된 disparity 거리 $s$**:
- **$g(\cdot)$**: 단조 스칼라 함수 (예: $g(x) = 1/x$로 선택)
- **$s \in [0, 1]$**: 시차 공간에서의 정규화된 위치 (0=가까운 평면, 1=먼 평면)
- **$t_n, t_f$**: 카메라 near/far 평면 거리
- **효과**: $s$를 균일하게 샘플링 → 실제 거리 $t$는 역수 간격으로 샘플링 → 가까운 곳 촘촘, 먼 곳 성김.

---

## Chapter 3: Coarse-to-Fine Online Distillation

**요약**

mip-NeRF의 coarse-to-fine 구조를 완전히 재설계합니다. 기존 방식(하나의 MLP를 두 번 평가)이 비효율적이고 최적이 아닌 것과 달리, **소형 Proposal MLP**와 **대형 NeRF MLP** 두 개를 분리합니다. Proposal MLP는 색상 없이 밀도만 예측해 효율적으로 좋은 샘플 위치를 찾고, NeRF MLP는 그 샘플들을 이용해 고품질 렌더링을 수행합니다. 이를 "온라인 디스틸레이션"이라 부릅니다.

**핵심 개념**

- **Proposal MLP ($\Theta_\text{prop}$)**: 작고 빠른 MLP. 밀도만 예측해 샘플링 가중치 $\mathbf{w}$를 생성. 색상을 예측하지 않아 학습이 쉽습니다.
- **NeRF MLP ($\Theta_\text{NeRF}$)**: 크고 표현력 높은 MLP. 밀도와 색상을 모두 예측해 최종 렌더링에 사용.
- **온라인 디스틸레이션**: NeRF MLP의 출력을 지도 삼아 Proposal MLP를 지속적으로 업데이트. Proposal MLP는 NeRF의 가중치 분포를 "상한으로 포함"하도록 학습.
- **Stop Gradient**: Proposal 손실이 NeRF MLP를 역방향 전파로 수정하지 않도록 Proposal MLP의 출력에만 기울기를 적용.

**핵심 수식 — Proposal 손실**

$$\mathcal{L}_\text{prop}(\mathbf{t}, \mathbf{w}, \hat{\mathbf{t}}, \hat{\mathbf{w}}) = \sum_i \frac{1}{\hat{w}_i} \max(0, \hat{w}_i - \text{bound}(\mathbf{t}, \mathbf{w}, \hat{T}_i))^2$$

$$\text{bound}(\mathbf{t}, \mathbf{w}, T) = \sum_{j;\, T_j \cap T \neq \emptyset} \tilde{w}_j$$

**수식 설명**

**Proposal 손실**은 Proposal MLP의 히스토그램이 NeRF MLP의 히스토그램을 항상 상한으로 포함하도록 강제합니다:
- **$\hat{\mathbf{t}}, \hat{\mathbf{w}}$**: Proposal MLP가 예측한 구간과 가중치 (상한 히스토그램)
- **$\mathbf{t}, \mathbf{w}$**: NeRF MLP가 예측한 구간과 가중치 (정답 히스토그램)
- **$\text{bound}(\mathbf{t}, \mathbf{w}, T)$**: Proposal 구간 $T$와 겹치는 NeRF 가중치의 합
- **$\max(0, \ldots)^2$**: Proposal 가중치가 NeRF 가중치보다 작을 때만(= 놓쳤을 때만) 페널티 부과. 과대 추정은 괜찮지만 과소 추정은 안 됩니다.
- **$1/\hat{w}_i$**: 불확실한 구간(가중치 작음)에 더 큰 패널티를 줘 안정적 최적화.

---

## Chapter 4: Regularization for Interval-Based Models

**요약**

학습된 NeRF에서 자주 나타나는 두 가지 아티팩트를 억제하는 새로운 정규화 손실을 제안합니다:
1. **Floaters**: 공중에 떠 있는 반투명 물질 덩어리. 특정 시점에서는 씬을 가리지만 다른 각도에서는 흐릿한 구름처럼 보입니다.
2. **Background Collapse**: 먼 배경이 카메라 바로 앞에 반투명 면으로 모델링되는 현상. 깊이 맵이 비현실적으로 납작해집니다.

**Distortion Loss**는 광선 가중치 분포를 가능한 한 좁고 집중되게 유지하도록 강제해 두 아티팩트를 동시에 억제합니다.

**핵심 개념**

- **Floaters**: 여러 시점에서 장면 일부를 설명하기 위해 MLP가 공중에 만들어내는 가짜 덩어리.
- **Background Collapse**: 먼 표면이 가까운 반투명 레이어로 압축되는 현상. 노이즈 주입(기존 방법)보다 이 논문의 distortion loss가 더 효과적.
- **Distortion Loss**: 광선을 따라 가중치들의 "퍼짐 정도"를 최소화하는 손실. 가중치가 단일 표면에 집중되도록 유도.

**핵심 수식**

$$\mathcal{L}_\text{dist}(\mathbf{s}, \mathbf{w}) = \iint_{-\infty}^{\infty} w_\mathbf{s}(u) w_\mathbf{s}(v) |u - v| \, du \, dv$$

$$= \sum_{i,j} w_i w_j \left| \frac{s_i + s_{i+1}}{2} - \frac{s_j + s_{j+1}}{2} \right| + \frac{1}{3} \sum_i w_i^2 (s_{i+1} - s_i)$$

**수식 설명**

**Distortion Loss**는 광선 가중치 분포의 "총 퍼짐"을 측정합니다:
- **$w_\mathbf{s}(u)$**: 정규화 거리 $s$에서의 가중치 함수 (밀도 분포를 나타냄)
- **$|u - v|$**: 두 점 $u, v$ 사이의 거리 — 분포가 넓을수록 이 값이 커짐
- **첫 번째 합 항**: 모든 구간 쌍의 중심 거리의 가중 합 → 분포의 퍼짐을 패널티
- **두 번째 합 항**: 각 구간의 폭에 비례하는 페널티 → 개별 구간이 넓어지지 못하게 억제
- **직관**: 가중치가 뾰족하게 집중될수록(= 표면이 명확할수록) 이 손실이 0에 가까워짐.

---

## Chapter 5: Optimization

**요약**

두 MLP의 학습에 사용된 구체적인 설정을 설명합니다. Proposal MLP는 4 레이어·256 hidden units, NeRF MLP는 8 레이어·1024 hidden units으로 구성됩니다. Proposal MLP를 32샘플로 2번 평가해 샘플링 구간을 생성하고, NeRF MLP를 32샘플로 최종 색상을 렌더링합니다.

**최종 손실 함수**

$$\mathcal{L}_\text{recon}(C(\mathbf{r}), C^*) + \lambda \mathcal{L}_\text{dist}(\mathbf{s}, \mathbf{w}) + \sum_{k=0}^{K} \mathcal{L}_\text{prop}(\mathbf{s}, \mathbf{w}, \hat{\mathbf{s}}^k, \hat{\mathbf{w}}^k)$$

**수식 설명**

세 항의 균형을 통해 학습:
- **$\mathcal{L}_\text{recon}$**: 렌더링된 색상 $C(\mathbf{r})$와 실제 색상 $C^*$ 사이의 재구성 손실 (Charb. 손실)
- **$\lambda \mathcal{L}_\text{dist}$**: 가중치 분포 집중 정규화 ($\lambda = 0.01$)
- **$\sum \mathcal{L}_\text{prop}^k$**: 각 Proposal MLP 단계의 온라인 디스틸레이션 손실
- **학습**: 256K 스텝, Adam 옵티마이저, 학습률 $2 \times 10^{-3}$ → $2 \times 10^{-5}$ 코사인 감쇠

---

## Chapter 6: Results

**요약**

9개 씬(야외 5개, 실내 4개)의 새로 수집한 데이터셋에서 평가했습니다. 기존 최신 방법들(NeRF++, mip-NeRF, SVS, DONeRF 등)과 비교한 결과:

- **PSNR** 기준 모든 기존 방법 대비 최고 성능
- mip-NeRF 대비 **MSE 57% 감소**
- 사실적인 색상 렌더링과 함께 **상세한 깊이 맵** 동시 생성
- 학습 시간: 7시간 (A100 기준), 파라미터 수: 9M

**정량적 비교 (주요 지표)**

| 방법 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| NeRF++ | 22.41 | 0.641 | 0.359 |
| mip-NeRF | 22.22 | 0.628 | 0.424 |
| SVS | 22.41 | 0.619 | 0.355 |
| **Our Model** | **24.37** | **0.687** | **0.300** |

**Ablation 연구 주요 결과**

- $\mathcal{L}_\text{dist}$ 제거 시 PSNR 20.49로 급락 (floater 발생)
- Proposal MLP 제거 시 PSNR 23.45로 하락 (학습 시간 75h로 폭증)
- IPE 제거 시 PSNR 23.77로 하락 → mip-NeRF 기반의 중요성 확인
- DONeRF의 수축 방식 사용 시 PSNR 24.26으로 하락

---

## Chapter 7: Conclusion

**요약**

mip-NeRF 360은 실제 세계의 360도 무한 장면을 사실적으로 합성하는 최초의 NeRF 계열 모델입니다. 세 가지 핵심 기여를 통해 mip-NeRF 대비 57% MSE 감소를 달성했습니다.

**한계**

- 미세 구조(자전거 바퀴살, 나뭇잎 결) 일부 누락
- 카메라가 씬 중심에서 멀리 이동하면 품질 저하
- 학습에 수 시간이 필요 → 온디바이스 학습 불가

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **Conical Frustum** | NeRF 광선을 원뿔 모양 구간으로 취급해 앤티앨리어싱 구현 |
| **IPE (Integrated Positional Encoding)** | Gaussian 분포를 주파수 공간에서 기댓값으로 표현, 해상도 의존성 제거 |
| **contract(x)** | 무한 3D 공간을 반경 2 구 안으로 수축하는 비선형 변환 |
| **Disparity Sampling** | 역수 거리 공간에서 균일 샘플링 → 가까운 곳 고해상도, 먼 곳 저해상도 |
| **Proposal MLP** | 색상 없이 밀도만 예측하는 소형 MLP. 빠른 샘플 위치 추정용 |
| **NeRF MLP** | 밀도와 색상을 모두 예측하는 대형 MLP. 최종 렌더링 담당 |
| **Online Distillation** | NeRF MLP의 가중치 분포를 정답 삼아 Proposal MLP를 실시간 학습 |
| **Distortion Loss** | 광선 가중치를 좁은 구간에 집중시켜 floater·background collapse 억제 |
| **Floaters** | 공중에 떠 있는 반투명 가짜 물질 — 나쁜 NeRF에서 자주 등장 |
| **Background Collapse** | 먼 배경이 카메라 앞 얇은 레이어로 잘못 모델링되는 현상 |
| **NDC (Normalized Device Coordinates)** | 기존 NeRF에서 전방 카메라용으로 쓰던 좌표계. 360도 장면에는 부적합 |

---

## 결론 및 시사점

mip-NeRF 360은 NeRF의 실용적 한계를 크게 허무는 연구입니다.

**기술적 시사점**
- **비선형 좌표 변환**은 무한 공간을 다루는 신경망 렌더링의 핵심 설계 원칙이 됩니다.
- **모델 분리(Proposal + NeRF)** 전략은 이후 Instant-NGP, 3D Gaussian Splatting 등 여러 후속 연구에 영향을 주었습니다.
- **분포 정합 손실(Distortion Loss)**은 볼류메트릭 밀도 학습의 일반적인 정규화 기법으로 채택될 수 있습니다.

**실무적 시사점**
- 드론 촬영, 자율주행 씬 재구성, 문화재 디지털화 등 360도 자유 카메라가 필요한 애플리케이션에 직접 적용 가능합니다.
- 학습 시간(수 시간)과 하드웨어(A100) 요구사항은 여전히 실시간 응용의 장벽이며, 이를 해결하는 방향(Instant-NGP, 3DGS)이 후속 연구의 주요 흐름이 되었습니다.
