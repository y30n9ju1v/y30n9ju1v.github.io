---
title: "HUGS: Holistic Urban 3D Scene Understanding via Gaussian Splatting"
date: 2026-04-10T09:00:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving", "Novel View Synthesis"]
tags: ["3D Gaussian Splatting", "Autonomous Driving", "Novel View Synthesis", "Semantic Segmentation"]
---

## 개요

- **저자**: Hongyu Zhou, Jiahao Shao, Lu Xu, Dongfeng Bai, Weichao Qiu, Bingbing Liu, Yue Wang, Andreas Geiger, Yiyi Liao
- **소속**: Zhejiang University, Huawei Noah's Ark Lab, University of Tübingen, Tübingen AI Center
- **발행년도**: 2024 (arXiv:2403.12722, March 2024)
- **주요 내용**: RGB 이미지만으로 도시 3D 장면의 외관(appearance), 의미(semantics), 움직임(motion)을 통합적으로 이해하는 3D Gaussian Splatting 기반 프레임워크. LiDAR나 수동 주석 3D 바운딩 박스 없이도 실시간 novel view synthesis, 3D 시맨틱 재구성, 동적 객체 추적을 동시에 수행함.

## 목차

1. Introduction
2. Related Work
3. Method
   - 3.1 Decomposed Scene Representation
   - 3.2 Holistic Urban Gaussian Splatting
   - 3.3 Loss Functions
   - 3.4 Implementation Details
4. Experiments
   - 4.1 Novel View Synthesis
   - 4.2 Semantic and Geometric Scene Understanding
   - 4.3 Scene Editing
   - 4.4 Ablation Study
5. Conclusion

---

## Chapter 1: Introduction

**요약**

도시 장면을 RGB 이미지만으로 전체론적(holistic)으로 이해하는 것은 자율주행 시뮬레이터 구축에 핵심적인 과제입니다. 기존 접근법들은 새로운 시점 합성, 시맨틱 라벨 파싱, 동적 객체 추적 등 특정 측면에만 집중했으며, 대부분 LiDAR 스캔이나 수동 주석 3D 바운딩 박스에 의존합니다.

HUGS는 다음을 동시에 달성합니다:
- 실시간 novel view synthesis (약 93 fps)
- 2D/3D 시맨틱 정보의 정확한 재구성
- 동적 3D 객체 추적 (3D bounding box 예측 포함)

핵심 아이디어는 노이즈 있는 2D 예측값(시맨틱 라벨, 광학 흐름, 3D 추적 결과)을 3D Gaussian으로 올려(lift) 물리적 제약 조건을 통해 정제하는 것입니다.

**핵심 개념**

- **Holistic Understanding**: 외관, 기하, 의미, 움직임을 하나의 통합 모델로 다루는 접근법
- **2D-to-3D Lifting**: 2D 이미지에서 얻은 예측을 3D 공간으로 변환하는 기술
- **Physical Constraints**: 유니사이클 모델 등 물리적 법칙을 활용해 노이즈 있는 예측을 정제

---

## Chapter 2: Related Work

**요약**

관련 연구는 크게 세 가지 범주로 나뉩니다:

1. **도시 3D 장면 이해**: LiDAR 기반 포인트 클라우드를 활용하는 방법들이 주를 이루며, RGB 전용 방법들은 기하 정확도가 낮음
2. **동적 장면의 NeRF/Gaussian**: NSG, MARS 등은 ground-truth 3D 바운딩 박스가 필요하여 실용성이 제한됨
3. **시맨틱 NeRF**: 2D 시맨틱을 3D로 올리는 작업들은 정적 장면에만 집중하거나 동적 객체를 개별적으로 분해하는 능력이 부족

HUGS의 차별점은 노이즈 있는 단안(monocular) 예측만으로 동적 객체를 개별 분해하면서도 전체 장면을 통합적으로 표현한다는 점입니다.

---

## Chapter 3: Method

**요약**

HUGS는 도시 장면을 정적 Gaussian과 N개의 동적 차량 Gaussian으로 분해하여 표현합니다. 각 동적 객체는 유니사이클 모델(unicycle model)로 움직임이 모델링됩니다.

### 3.1 Decomposed Scene Representation (분해된 장면 표현)

**Static and Dynamic 3D Gaussians**

3D Gaussian Splatting을 기반으로 정적/동적 영역 모두를 표현합니다. 각 Gaussian은 다음으로 정의됩니다:

$$G(\mathbf{x}) = \alpha \exp\left(-\frac{1}{2}(\mathbf{x} - \mu)^T \Sigma^{-1}(\mathbf{x} - \mu)\right)$$

**수식 설명**
- $G(\mathbf{x})$: 위치 $\mathbf{x}$에서 이 Gaussian이 기여하는 값 (투명도 가중치)
- $\mu \in \mathbb{R}^3$: Gaussian의 중심 위치 (3D 공간에서의 좌표)
- $\Sigma \in \mathbb{R}^{3 \times 3}$: 공분산 행렬 (Gaussian의 모양과 방향을 결정)
- $\alpha$: 기본 불투명도 (opacity)
- $(\mathbf{x} - \mu)^T \Sigma^{-1}(\mathbf{x} - \mu)$: 마할라노비스 거리 — 중심에서 멀수록 값이 커지고 Gaussian 기여가 줄어듦

각 Gaussian은 추가로:
- **색상 벡터** $\mathbf{c} \in \mathbb{R}^3$: 구면 조화 함수(SH) 계수로 파라미터화
- **시맨틱 로짓** $\mathbf{s} \in \mathbb{R}^S$: 각 클래스에 대한 확률 점수
- **광학 흐름** $\mathbf{f}_{t_1 \to t_2} \in \mathbb{R}^2$: 두 시간 $t_1, t_2$ 사이의 픽셀 이동 벡터

**Unicycle Model (유니사이클 모델)**

N개의 동적 차량 각각은 유니사이클 모델로 움직임이 파라미터화됩니다. 상태 $(x_t, y_t, \theta_t)$와 속도 $(v_t, \omega_t)$를 학습 가능한 파라미터로 두며, 시간 $t$에서 $t+1$로의 전환은:

$$x_{t+1} = x_t + \frac{v_t}{\omega_t}(\sin\theta_{t+1} - \sin\theta_t)$$

$$y_{t+1} = y_t - \frac{v_t}{\omega_t}(\cos\theta_{t+1} - \cos\theta_t)$$

$$\theta_{t+1} = \theta_t + \omega_t$$

**수식 설명**
- $(x_t, y_t)$: 시간 $t$에서 차량의 2D 위치 (지면 좌표계)
- $\theta_t$: 시간 $t$에서 차량의 진행 방향 (요우각, yaw angle)
- $v_t$: 전진 속도 (앞으로 가는 빠르기)
- $\omega_t$: 각속도 (방향이 바뀌는 빠르기)
- $\frac{v_t}{\omega_t}$: 회전 반경 — 빠르게 달리고 천천히 회전할수록 큰 원을 그림
- $\sin\theta_{t+1} - \sin\theta_t$: 방향 변화에 따른 x 방향 이동량

이 모델의 장점: 프레임별로 독립적으로 위치를 최적화하는 것보다 물리적으로 자연스러운 궤적을 생성하여 로컬 미니마에 빠질 가능성이 낮음.

### 3.2 Holistic Urban Gaussian Splatting (전체론적 렌더링)

**Novel View Synthesis (신규 시점 합성)**

정적/동적 Gaussian을 이미지 평면에 투영하고 $\alpha$-블렌딩으로 합성합니다:

$$\pi: \quad \mathbf{C} = \sum_{i \in \mathcal{N}} \mathbf{c}_i \alpha_i' \prod_{j=1}^{i-1}(1 - \alpha_j')$$

**수식 설명**
- $\mathbf{C}$: 최종 렌더링된 픽셀 색상
- $\mathbf{c}_i$: i번째 Gaussian의 색상 (카메라 방향에 따라 달라지는 구면 조화 함수 기반)
- $\alpha_i'$: i번째 Gaussian의 투영된 불투명도 (3D 모양이 2D로 투영된 값)
- $\prod_{j=1}^{i-1}(1 - \alpha_j')$: 앞에 있는 Gaussian들을 투과한 빛의 비율
  - 앞의 객체가 완전 불투명(α=1)이면 뒤의 객체는 보이지 않음 (곱이 0)
  - 앞의 객체가 반투명이면 뒤의 객체도 부분적으로 보임

**Exposure Modeling (노출 모델링)**

도시 장면은 자동 노출로 촬영되어 프레임마다 밝기가 다릅니다. 카메라의 외부/내부 파라미터를 이용해 아핀 변환으로 노출을 보정합니다:

$$\tilde{\mathbf{C}} = \mathbf{A} \times \mathbf{C} + \mathbf{b}$$

**수식 설명**
- $\mathbf{A} \in \mathbb{R}^{3 \times 3}$: 색상 변환 행렬 (화이트 밸런스, 채도 등 전역적 색상 조정)
- $\mathbf{b} \in \mathbb{R}^3$: 밝기 오프셋 벡터
- 이 행렬은 카메라 외부 파라미터로부터 MLP를 통해 소규모로 학습

**Semantic Reconstruction (시맨틱 재구성)**

3D 시맨틱 로짓 $\mathbf{s}_i$에 소프트맥스를 적용 후 $\alpha$-블렌딩:

$$\pi: \quad \mathbf{S} = \sum_{i \in \mathcal{N}} \text{softmax}(\mathbf{s}_i) \alpha_i' \prod_{j=1}^{i-1}(1 - \alpha_j')$$

**수식 설명**
- $\mathbf{S}$: 렌더링된 픽셀의 시맨틱 클래스 확률 분포
- $\text{softmax}(\mathbf{s}_i)$: 3D 공간에서 정규화된 클래스 확률 — 2D에서 소프트맥스 적용하는 것보다 더 안정적인 3D 시맨틱을 생성
- 핵심 차이: 2D 소프트맥스는 큰 로짓 값을 가진 단일 Gaussian이 전체 렌더링을 지배할 수 있으나, 3D 소프트맥스는 이런 floater 현상을 방지

**Optical Flow (광학 흐름)**

두 타임스탬프 $t_1, t_2$ 사이의 각 Gaussian 중심 $\mu$의 투영 차이로 모션 벡터를 계산합니다:

$$\mu_1' = \mathbf{K}[\mathbf{R}_{t_1}^{\text{cam}} | \mathbf{t}_{t_1}^{\text{cam}}]\mu, \quad \mu_2' = \mathbf{K}[\mathbf{R}_{t_2}^{\text{cam}} | \mathbf{t}_{t_2}^{\text{cam}}]\mu$$

$$\pi: \quad \mathbf{F} = \sum_{i \in \mathcal{N}} \mathbf{f}_i \alpha_i' \prod_{j=1}^{i-1}(1 - \alpha_j')$$

**수식 설명**
- $\mathbf{K}$: 카메라 내부 파라미터 행렬 (3D→2D 투영)
- $\mathbf{R}^{\text{cam}}, \mathbf{t}^{\text{cam}}$: 각 타임스탬프에서 카메라의 회전/이동
- $\mathbf{f}_i = \mu_2' - \mu_1'$: 3D Gaussian 중심의 2D 이미지상 이동 벡터
- $\mathbf{F}$: 최종 렌더링된 광학 흐름 맵

### 3.3 Loss Functions (손실 함수)

전체 손실 함수:

$$\mathcal{L} = \mathcal{L}_I + \lambda_S \mathcal{L}_S + \lambda_F \mathcal{L}_F + \lambda_t \mathcal{L}_t + \lambda_{uni} \mathcal{L}_{uni} + \lambda_{reg} \mathcal{L}_{reg}$$

**수식 설명**
- $\mathcal{L}_I$: 이미지 재구성 손실 (렌더링 품질)
- $\mathcal{L}_S$: 시맨틱 분할 손실 (클래스 정확도)
- $\mathcal{L}_F$: 광학 흐름 손실 (움직임 정확도)
- $\mathcal{L}_t$: 3D 바운딩 박스 위치 손실 (추적 정확도)
- $\mathcal{L}_{uni}$: 유니사이클 모델 정규화 손실 (물리적 움직임 제약)
- $\mathcal{L}_{reg}$: 속도/각속도 가속도 정규화 (부드러운 궤적)
- $\lambda$ 값들: 각 손실의 가중치 (하이퍼파라미터)

**이미지 재구성 손실:**

$$\mathcal{L}_I = (1 - \lambda_{SSIM})\|\tilde{\mathbf{I}} - \mathbf{I}\| + \lambda_{SSIM} \text{SSIM}(\tilde{\mathbf{I}}, \mathbf{I})$$

**시맨틱 손실 (크로스 엔트로피):**

$$\mathcal{L}_S = -\sum_{k=0}^{S-1} \hat{\mathbf{S}}_k \log(\mathbf{S}_k)$$

**유니사이클 정규화 손실:**

$$\mathcal{L}_{uni} = \sum_t \left\|x_{t+1} - x_t - \frac{v_t}{\omega_t}(\sin\theta_{t+1} - \sin\theta_t)\right\|_2 + \sum_t \left\|y_{t+1} - y_t + \frac{v_t}{\omega_t}(\cos\theta_{t+1} - \cos\theta_t)\right\|_2 + \sum_t \|\theta_{t+1} - \theta_t - \omega_t\|$$

**수식 설명**
- 이 손실은 학습된 차량 상태 $(x_t, y_t, \theta_t)$가 물리적으로 타당한 유니사이클 운동 방정식을 따르도록 강제
- 노이즈 있는 3D 바운딩 박스 예측값을 물리 법칙으로 정제하는 핵심 메커니즘

**가속도 정규화:**

$$\mathcal{L}_{reg} = \sum_t \|v_{t+1} + v_{t-1} - 2v_t\|_2 + \sum_t \|\theta_{t+1} + \theta_{t-1} - 2\theta_t\|_2$$

**수식 설명**
- 2차 차분(second-order difference)으로 가속도를 계산하여 급격한 속도 변화 억제
- 부드러운 궤적 생성을 통해 현실적인 차량 동작 모델링

---

## Chapter 4: Experiments

**요약**

### 4.1 Novel View Synthesis

KITTI, vKITTI2, KITTI-360 데이터셋에서 NSG, MARS와 비교 평가합니다.

| 방법 | KITTI Scene02 PSNR↑ | KITTI Scene02 SSIM↑ | KITTI Scene02 LPIPS↓ |
|------|---------------------|---------------------|----------------------|
| NSG  | 23.00 | 0.664 | 0.373 |
| MARS | 23.70 | 0.731 | 0.310 |
| **Ours** | **25.42** | **0.821** | **0.092** |

노이즈 있는 3D 바운딩 박스(QD-3DT 예측)를 사용한 동적 장면에서도 PSNR 기준 약 2dB 향상을 달성합니다.

### 4.2 Semantic and Geometric Scene Understanding

**3D Semantic Reconstruction** (KITTI-360 기준):

| 방법 | acc.↓ | comp.↓ | mIoU↑ |
|------|-------|--------|-------|
| Semantic Nerfacto | 1.508 | 24.28 | 0.055 |
| **Ours** | **0.233** | **0.214** | **0.505** |

기하 정확도(acc, comp)와 시맨틱 정확도(mIoU) 모두에서 큰 폭으로 앞섭니다. 기존 2D→3D 리프팅 방법들은 2D 공간 내 소프트맥스 정규화의 한계로 인해 floater가 발생하나, HUGS는 3D 소프트맥스로 이를 해결합니다.

### 4.3 Scene Editing

분해된 장면 표현 덕분에 다양한 편집이 가능합니다:
- **전경/배경 분리**: 차량 제거 및 배경만 렌더링
- **동적 객체 교체**: 다른 차량으로 대체
- **위치/방향 조작**: 차량의 회전 및 이동

### 4.4 Ablation Study

**동적 장면 (KITTI, 다양한 노이즈 수준):**

| 설정 | PSNR↑ | SSIM↑ | LPIPS↓ | $\epsilon_R$↓ | $\epsilon_t$↓ |
|------|-------|-------|--------|---------------|---------------|
| w/o opt., w/o uni. | 22.56 | 0.879 | 0.062 | 0.031 | 0.027 |
| w/ opt., w/o uni. | 24.80 | 0.897 | 0.038 | 0.022 | 0.051 |
| **w/ opt., w/ uni. (Ours)** | **28.78** | **0.928** | **0.023** | **0.017** | **0.022** |

유니사이클 모델이 노이즈 있는 바운딩 박스 상황에서 추적 정확도와 렌더링 품질 모두를 크게 향상시킵니다.

**정적 장면 (KITTI-360):**

| 설정 | PSNR↑ | SSIM↑ | LPIPS↓ | Depth↓ |
|------|-------|-------|--------|--------|
| w/o Affine transform | 24.18 | 0.827 | 0.083 | — |
| w/o $\mathcal{L}_S$ | 24.47 | 0.831 | 0.081 | 0.892 |
| w/o $\mathcal{L}_F$ | 24.41 | 0.831 | 0.081 | 1.031 |
| **Ours** | **24.52** | **0.833** | **0.081** | **0.872** |

광학 흐름 손실이 기하 품질(Depth)에 명확한 기여를 하며, 아핀 변환이 외관 품질에 중요합니다.

---

## 핵심 개념 정리

- **3D Gaussian Splatting**: 3D 장면을 수백만 개의 소형 타원형 Gaussian으로 표현하는 방법. 각 Gaussian은 위치, 모양, 색상, 불투명도를 가지며 실시간 렌더링이 가능함.

- **Decomposed Scene Representation**: 장면을 정적 배경 Gaussian과 N개의 동적 차량 Gaussian으로 분리. 이 분해가 편집, 추적, 시맨틱 이해를 가능케 하는 핵심.

- **Unicycle Model**: 차량을 단순화된 두 바퀴 자전거 역학으로 모델링. 전진 속도 $v$와 각속도 $\omega$만으로 물리적으로 타당한 궤적을 생성하여 노이즈 있는 예측을 정제.

- **3D Softmax vs 2D Softmax**: 기존 방법들은 렌더링 후 2D에서 소프트맥스 적용 → 단일 Gaussian이 시맨틱을 지배하는 floater 문제 발생. HUGS는 3D 로짓에 먼저 소프트맥스 적용 → 더 일관된 3D 시맨틱 표현.

- **Exposure Modeling**: 자율주행 데이터의 자동 노출 변화를 카메라 파라미터 기반 아핀 변환으로 보정. 고품질 렌더링의 필수 요소.

- **Pseudo Ground Truth**: LiDAR나 수동 주석 없이 InverseForm(시맨틱), QD-3DT(3D 추적), Unimatch(광학 흐름)에서 자동으로 생성한 의사 정답 사용.

- **$\alpha$-blending**: Gaussian들을 앞에서 뒤로 정렬하여 누적 투명도를 계산하며 색상/시맨틱/흐름을 합성하는 볼륨 렌더링 방식.

---

## 결론 및 시사점

HUGS는 RGB 이미지만으로 도시 3D 장면의 **외관, 의미, 움직임을 통합적으로 표현**하는 최초의 실시간 프레임워크입니다. 주요 기여:

1. **통합 표현**: 단일 3D Gaussian 기반으로 novel view synthesis, 시맨틱 재구성, 동적 추적을 동시 수행
2. **노이즈 강건성**: 유니사이클 모델을 통해 단안 모노큘러 3D 추적의 노이즈를 효과적으로 제거
3. **실시간 성능**: RTX 4090에서 약 93 fps (NSG, MARS 대비 압도적 속도)
4. **라벨 효율성**: LiDAR, 수동 3D 바운딩 박스 없이도 SOTA 성능 달성

**한계점**:
- 동적 객체는 현재 단순 회전만 가능 (차량 문 열림 등 비강체 변형 불가)
- 카테고리 수준(category-level) 사전 지식 미활용
- 조명 편집 등 더 많은 자유도는 미지원

**실무적 시사점**: 자율주행 시뮬레이터에서 포토리얼리스틱한 장면 재구성, 센서 시뮬레이션, 데이터 증강에 직접 활용 가능. 특히 고가의 LiDAR 없이도 고품질 3D 이해가 가능하다는 점이 산업적으로 의미 있음.
