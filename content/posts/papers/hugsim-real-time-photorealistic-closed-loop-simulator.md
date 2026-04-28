---
title: "HUGSIM: A Real-Time, Photo-Realistic and Closed-Loop Simulator for Autonomous Driving"
date: 2026-04-10T09:00:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving", "Novel View Synthesis"]
tags: ["3D Gaussian Splatting", "Autonomous Driving", "Simulation", "Closed-Loop"]
---

## 개요

- **저자**: Hongyu Zhou, Longzhong Lin, Jiabao Wang, Yichong Lu, Dongfeng Bai, Bingbing Liu, Yue Wang, Andreas Geiger, Yiyi Liao
- **발행년도**: 2024 (arXiv:2412.01718)
- **학회**: CVPR 2024 Extension
- **주요 내용**: 3D Gaussian Splatting을 기반으로 실시간, 포토리얼리스틱, 클로즈드-루프 자율주행 시뮬레이터 HUGSIM을 제안. 보간 뷰와 외삽 뷰 모두에서 고품질 렌더링을 제공하며, 70개 이상의 시퀀스에 걸친 종합적인 벤치마크를 통해 기존 자율주행 알고리즘 평가 플랫폼으로 활용 가능.

## 목차

1. Introduction
2. Related Work
3. Urban Scene Reconstruction
4. Simulation
5. Experiments
6. Closed-Loop Benchmark
7. Conclusion

---

## 1. Introduction

**요약**

자율주행 알고리즘은 최근 몇 년간 인식(Perception), 계획(Planning), 제어(Control) 분야에서 비약적인 발전을 이루었다. 그러나 기존의 AD 평가 방식은 개별 컴포넌트 성능을 독립적으로 측정하는 오픈루프(Open-loop) 방식에 머물러 있어, 알고리즘이 실제 주행 환경에서 어떻게 동작하는지 전체적(holistic)으로 평가하기 어렵다.

이 문제를 해결하기 위해 본 논문은 **HUGSIM**을 제안한다. HUGSIM은 3D Gaussian Splatting(3DGS)를 기반으로 한 클로즈드-루프(closed-loop) 시뮬레이터로, 다음 세 가지 주요 특성을 갖는다:

1. **실시간성(Real-Time)**: 빠른 속도로 렌더링 가능
2. **포토리얼리즘(Photo-Realistic)**: 고품질 이미지 합성
3. **클로즈드-루프(Closed-Loop)**: Ego 차량의 행동에 따라 환경이 동적으로 반응

**핵심 개념**

- **오픈루프 vs 클로즈드루프**: 오픈루프 평가는 사전 수집된 데이터에서 알고리즘을 테스트하지만, 알고리즘의 행동이 미래 관측에 영향을 주지 않음. 클로즈드루프는 시뮬레이터가 알고리즘의 제어 명령에 실시간으로 반응하여 실제 주행과 유사한 피드백 루프를 형성.
- **기존 방법의 한계**: NeRF 기반 방법들은 고품질 렌더링을 제공하지만 실시간성이 부족. 게임 엔진 기반 방법들은 실시간성은 있지만 현실감이 떨어짐.

---

## 2. Related Work

**요약**

관련 연구를 세 카테고리로 분류하여 정리한다.

**2.1 Open-Loop 벤치마크**

대부분의 기존 데이터셋(KITTI, nuScenes, Waymo)과 벤치마크는 오픈루프 방식으로 AD 알고리즘을 평가한다. NAVSIM은 클로즈드루프와 오픈루프의 중간 형태를 제공하지만, 반응형 시뮬레이터가 없어 새로운 뷰 합성 기능을 갖추지 못했다.

**2.2 도시 장면 재구성 방법**

- **정적 장면**: NeRF, mesh 기반, 3DGS 기반 방법들이 발전했지만 동적 객체 처리에 한계
- **동적 장면**: EmerNeRF, StreetGaussians 등이 동적 물체를 별도로 모델링하는 접근법 제시

**2.3 클로즈드-루프 시뮬레이터**

DriveArena, UniSim 등이 클로즈드루프 시뮬레이션을 시도했으나, HD 맵 없이도 동작하거나 실시간 렌더링을 갖추는 동시에 포토리얼리스틱 품질을 달성하지 못하는 한계가 있었다.

**핵심 개념**

- **HD-Score**: NAVSIM과 DriveArena에서 영감을 받아 제안한 HUGSIM만의 평가 지표. NC(No Collision), DAC(Drive Area Compliance), TTC(Time to Collision), COM(Comfort)의 복합 점수.

---

## 3. Urban Scene Reconstruction

**요약**

HUGSIM의 장면 재구성 파이프라인은 정적(static) 배경과 동적(dynamic) 전경을 분리하여 모델링한다.

### 3.1 기초 개념: 3D Gaussian Splatting

3DGS는 장면을 이방성(anisotropic) 3D Gaussian들의 집합으로 표현한다. 각 Gaussian $\mathbf{g}$는 다음 속성을 갖는다:

- 위치 $\mu \in \mathbb{R}^3$ (공간 내 중심점)
- 회전 $\mathbf{R}$ (쿼터니언으로 표현)
- 스케일 $\mathbf{S}$ (각 축의 크기)
- 투명도 $\alpha \in [0, 1]$
- 구면 조화 함수(Spherical Harmonics, $SH$)로 표현된 색상

색상 렌더링은 볼륨 렌더링 방정식으로 계산된다:

$$\pi = \mathbf{C} = \sum_{i \in N} c_i \alpha'_i \prod_{j=1}^{i-1}(1 - \alpha'_j)$$

**수식 설명**
- **$\mathbf{C}$**: 최종 렌더링된 픽셀 색상
- **$c_i$**: i번째 Gaussian의 색상
- **$\alpha'_i$**: i번째 Gaussian의 projected 투명도 (카메라에서 본 실제 불투명도)
- **$\prod_{j=1}^{i-1}(1-\alpha'_j)$**: 앞의 j-1개 Gaussian을 통과한 빛의 투과율 (앞 객체들이 얼마나 빛을 가리는지)
- 직관적으로: 뒤쪽 Gaussian일수록 앞에 있는 Gaussian들에 의해 가려지는 효과가 누적됨

### 3.2 분해된 장면 표현

**비-지면 정적 Gaussian (Non-Ground Static Gaussians)**

기존 3DGS 정의에 추가로 **3D 의미론적 레이블(semantic label)**을 각 Gaussian에 부여한다. 이를 통해:
- 전경/배경 분리를 위한 의미 정보 활용
- 충돌 감지에 의미 정보 활용 가능

**지면 Gaussian (Ground Gaussians)**

자율주행 장면에서 지면은 평평해야 한다는 물리적 제약을 활용한다. 단순한 단일 평면으로 가정하면 경사면 등 복잡한 지형을 처리하기 어려우므로, **다중 평면 지면 모델(Multi-Plane Ground Model)**을 제안한다.

최적화 문제:

$$\min_{\{x_i, y_i, \sigma_i\}} (1 - \lambda_{SSM})\|\mathbf{I} - \hat{\mathbf{I}}\|_1 + \lambda_{SSM}\text{SSIM}(\mathbf{I}, \hat{\mathbf{I}})$$

$$\text{subject to} \quad \min_{\{x_i, y_i, \sigma_i\}} \sqrt{\lambda} \frac{1}{N} \sum_{z_i > z_{th}} (\mu_{y,i}^{cam} - \mu_{y,i}^{gm})^2 = 0$$

**수식 설명**
- **$\mathbf{I}$, $\hat{\mathbf{I}}$**: 렌더링 이미지와 실제(ground truth) 이미지
- **$\lambda_{SSM}$**: SSIM 손실 가중치
- **$\mu_{y,i}^{cam}$, $\mu_{y,i}^{gm}$**: 카메라 좌표와 지면 모델 좌표에서의 Gaussian y축 위치
- 직관적으로: 렌더링 품질을 유지하면서 Gaussian들이 실제 지면에 붙어 있도록 제약을 부과

**비-네이티브 동적 차량 Gaussian (Non-Native Dynamic Vehicles)**

동적 차량은 RGB 이미지와 노이즈가 있는 3D 예측값(3D bounding box)에서 재구성된다. 각 차량의 Gaussian은 유니사이클 모델(unicycle model)에 의해 파라미터화된 변환으로 최적화된다:

$$\begin{pmatrix} \dot{x} \\ \dot{y} \\ \dot{\theta} \\ \dot{v} \end{pmatrix} = \frac{dS}{dt} = \begin{pmatrix} v\cos\theta \\ v\sin\theta \\ \frac{v\tan\delta}{L} \\ a \end{pmatrix}$$

**수식 설명**
- **$x, y$**: 차량의 BEV(Bird's Eye View) 위치
- **$\theta$**: 차량 방향각 (yaw)
- **$v$**: 전진 속도
- **$\delta$**: 조향각 (steering angle)
- **$L$**: 차량 축간거리 (wheelbase)
- **$a$**: 가속도
- 직관적으로: 자동차의 물리적 움직임 특성(앞바퀴 조향으로 인한 회전)을 모델링하여 자연스러운 차량 궤적을 예측

### 3.3 전체 Gaussian Splatting (Holistic Gaussian Splatting)

**새로운 뷰 합성 (Novel View Synthesis)**

정적 Gaussian과 동적 Gaussian을 결합하여 최종 이미지를 렌더링한다. 의미 레이블(s), 광학 흐름(F), 깊이(D)도 동일한 볼륨 렌더링 방식으로 계산된다:

$$\pi_S: \mathbf{S} = \sum_{i \in N} \text{softmax}(s_i) \alpha'_i \prod_{j=1}^{i-1}(1 - \alpha'_j) \quad (7)$$

$$\pi_F: \mathbf{F} = \sum_{i \in N} F_i \alpha'_i \prod_{j=1}^{i-1}(1 - \alpha'_j) \quad (9)$$

$$\pi_D: \mathbf{D} = \sum_{i \in N} d_i \alpha'_i \prod_{j=1}^{i-1}(1 - \alpha'_j) \quad (10)$$

**수식 설명 (의미 레이블)**
- **$s_i$**: i번째 Gaussian의 3D 의미 로짓(logit) 벡터 (차, 나무, 도로 등 클래스별 점수)
- **$\text{softmax}(s_i)$**: 3D 공간에서 소프트맥스를 적용하여 정규화 (2D 소프트맥스보다 floating Gaussian 문제를 줄임)
- **$\pi_S$**: 최종 의미 레이블 맵

**광학 흐름 계산**

두 타임스탬프 $t_1$, $t_2$ 사이의 광학 흐름을 계산하기 위해 각 Gaussian 중심 $\mu$를 두 이미지 평면에 투영한다:

$$\mu'_1 = \mathbf{K}[\mathbf{R}_{t_1}^{cam}; \mathbf{t}_{t_1}^{cam}]\mu, \quad \mu'_2 = \mathbf{K}[\mathbf{R}_{t_2}^{cam}; \mathbf{t}_{t_2}^{cam}]\mu \quad (8)$$

**수식 설명**
- **$\mathbf{K}$**: 카메라 내부 파라미터 행렬 (focal length, principal point)
- **$\mathbf{R}_t^{cam}$, $\mathbf{t}_t^{cam}$**: 시간 $t$에서의 카메라 외부 파라미터 (회전과 이동)
- **$\mu'_1$, $\mu'_2$**: 각 타임스탬프에서의 2D 이미지 투영 좌표
- 광학 흐름 벡터: $\mathbf{f}_{t_1 \to t_2} = \mu'_2 - \mu'_1$

### 3.4 손실 함수 (Loss Functions)

**이미지 기반 손실**

$$\mathcal{L}_I = (1 - \lambda_{SSIM})\|\mathbf{I} - \hat{\mathbf{I}}\|_1 + \lambda_{SSIM}\text{SSIM}(\mathbf{I}, \hat{\mathbf{I}}) \quad (11)$$

**의미 손실**

$$\mathcal{L}_S = -\sum_{k=0}^{\hat{S}-1} \hat{S}_k \log(S_k) \quad (12)$$

**알파 마스크 손실**

$$\mathcal{L}_A = \|\mathcal{A} - \hat{\mathbf{I}}_M\|_2 \quad (13)$$

**수식 설명**
- **$\mathcal{L}_I$**: 렌더링 이미지와 실제 이미지의 픽셀 차이 (L1)와 구조적 유사도(SSIM)의 결합
- **$\mathcal{L}_S$**: 2D 의미 분할 pseudo-label과의 크로스 엔트로피 손실
- **$\mathcal{L}_A$**: 렌더링된 알파 맵과 마스크 ground truth의 L2 거리
- **$\hat{\mathbf{I}}_M$**: 차량 마스크 ground truth (비-네이티브 차량이 투명하게 보이지 않도록 제약)

**물리 기반 정규화 (지면 모델)**

$$\mathcal{L}_{ground} = \frac{1}{N} \sum_{z_i > z_{th}} (\mu_{y,i}^{cam} - \mu_{y,i}^{gm})^2 \quad (14)$$

**유니사이클 모델 손실**

$$\mathcal{L}_t = \sum_t \|x_t - \hat{x}_t\|_2 + \|z_t - \hat{z}_t\|_2 \quad (15)$$

$$\mathcal{L}_{uni} = \sum_t \|x_{t+1} - x_t - \frac{v_t}{\omega_t}(\sin\theta_{t+1} - \sin\theta_t)\| + \sum_t \|z_{t+1} - z_t + \frac{v_t}{\omega_t}(\cos\theta_{t+1} - \cos\theta_t)\| + \sum_t \|\theta_{t+1} - \theta_t - \omega_t\| \quad (16)$$

**수식 설명**
- **$\mathcal{L}_t$**: 예측 bounding box 위치와 실제 위치의 차이 (x, z는 수평 좌표)
- **$\mathcal{L}_{uni}$**: 유니사이클 모델의 운동 방정식을 따르도록 강제하는 정규화
  - 첫 번째 항: x축 방향 이동이 속도와 각도 변화에 부합하는지
  - 두 번째 항: z축 방향 이동이 물리 법칙에 부합하는지
  - 세 번째 항: 각도 변화가 각속도와 일치하는지

---

## 4. Simulation

**요약**

HUGSIM의 시뮬레이션 모듈은 세 가지 핵심 요소로 구성된다: GUI 인터페이스, 클로즈드루프 시뮬레이션 엔진, 액터 행동 생성 시스템.

### 4.1 GUI 설정 인터페이스

시뮬레이터 설정을 위한 그래픽 사용자 인터페이스(GUI)를 제공한다. 주요 설정 항목:
1. **카메라 설정**: 카메라 수, 내부/외부 파라미터
2. **Ego 차량 파라미터**: 운동학 모델, 제어 빈도, 초기 상태
3. **액터 설정**: 네이티브/비-네이티브 차량, 공격적 행동 여부

3DRealCar 데이터셋에서 100가지 이상의 후보 3D 차량 모델을 제공한다.

### 4.2 클로즈드루프 시뮬레이션

**시뮬레이터-사용자 통신**

3D Gaussian 재구성 결과를 Gymnasium 환경으로 캡슐화하여, AD 알고리즘과 병렬로 실행한다. 두 시스템이 같은 기기에서 실행될 때는 **named pipes**, 다른 경우에는 **web sockets**을 사용한다.

**컨트롤러**

AD 알고리즘이 계획된 waypoints 또는 다음 몇 초간의 제어 명령 시퀀스를 반환하면, 시뮬레이터는 이를 바탕으로 LQR(Linear Quadratic Regulator) 제어를 적용한다.

**Ego 차량 운동학 모델**

제어 명령(조향각 $\delta$, 가속도 $a$)이 주어질 때, 이산화된 자전거 모델(kinematic bicycle model)을 사용한다:

$$S = \begin{pmatrix} x \\ y \\ \theta \\ v \end{pmatrix}, \quad \frac{dS}{dt} = \begin{pmatrix} v\cos\theta \\ v\sin\theta \\ \frac{v\tan\delta}{L} \\ a \end{pmatrix} \quad (18)$$

**충돌 감지 (Collision Detection)**

두 종류의 충돌을 정의한다:
1. **전경 충돌**: BEV(Bird's Eye View) bounding box 겹침으로 감지
2. **배경 충돌**: Ego 차량 3D bounding box 안의 배경 Gaussian 수가 임계값 초과 시 감지 (의미 레이블로 지면, 식생 등 제외)

### 4.3 액터 행동 모델

**재생(Replayed) 행동**: 원본 데이터셋의 관측 궤적을 유니사이클 모델로 재구성하여 재생

**일반(Normal) 행동**: IDM(Intelligent Driver Model)을 따르는 차선 추종 행동. HD 맵이 없는 경우 일정 속도로 사전 정의된 방향 주행

**공격적(Aggressive) 행동**: 

$$\min C_{total}(s_{1:T}^{(a)}) = C_{attack}(s_{1:T}^{(a)}) + \lambda C_{collision}(s_{1:T}^{(a)}) \quad (19)$$

$$C_{attack}(s_{1:T}^{(a)}) = \min_{t=1:T} \|s_t^{(j)} - s_t^{(a)}\|$$

$$C_{collision}(s_{1:T}^{(a)}) = \sum_{j=1}^{M} \mathbb{1}(\min_{t=1:T} \|s_t^{(j)} - s_t^{(a)}\| < \text{tolerance})$$

**수식 설명**
- **$s_{1:T}^{(a)}$**: 공격적 액터의 T 타임스탬프 궤적
- **$C_{attack}$**: 공격 비용 - Ego 차량과 공격 액터 간 최소 거리 (작을수록 위험)
- **$C_{collision}$**: 충돌 비용 - 다른 액터와의 충돌 횟수 (회피하면서 공격해야 함)
- **$\lambda$**: 충돌 페널티 가중치
- 직관적으로: 다른 차량들과 충돌하지 않으면서 Ego 차량에 최대한 가까이 접근하는 최적 궤적을 탐색

### 4.4 평가 지표: HD-Score

$$\text{HD-Score}_t = \left(\prod_{m \in \{NC, DAC\}} score_m\right) \times \frac{\sum_{w \in \{TTC, COM\}} weight_w \times score_w}{\sum_{w \in \{TTC, COM\}} weight_w} \quad (20)$$

**수식 설명**
- **$NC$ (No Collision)**: 충돌 없음 여부 (0 or 1, 충돌 시 전체 점수 0)
- **$DAC$ (Drive Area Compliance)**: 주행 구역 준수 여부 (도로 이탈 시 전체 점수 0)
- **$TTC$ (Time to Collision)**: 충돌까지의 시간 - 낮을수록 위험 상황
- **$COM$ (Comfort)**: 승차감 - 급격한 가속/감속/조향 페널티
- 직관적으로: NC와 DAC는 안전의 최소 조건(AND 조건)이며, TTC와 COM은 주행 품질을 나타내는 보완적 지표

---

## 5. Experiments

**요약**

HUGSIM은 다섯 가지 측면에서 평가된다: 보간 뷰 합성, 외삽 뷰 합성, 3D 의미 재구성, 3D 시맨틱 재구성, 기하학적 재구성.

### 5.1 새로운 뷰 합성 (보간 뷰)

KITTI-360과 Waymo 데이터셋에서 평가. 주요 지표: PSNR, SSIM, LPIPS.

| 방법 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| NSG | 23.00 | 0.664 | 0.373 |
| StreetGaussian | 25.59 | 0.873 | 0.174 |
| **Ours (HUGSIM)** | **25.42** | **0.821** | **0.092** |

HUGSIM은 NeRF 기반 방법 대비 렌더링 속도 면에서 압도적 우위를 보이며, 품질 면에서도 경쟁력 있는 성능을 달성한다.

### 5.2 새로운 뷰 합성 (외삽 뷰)

외삽 뷰는 훈련 데이터에 없는 새로운 시점에서의 렌더링으로, 클로즈드루프 시뮬레이션에서 필수적이다.

- HUGSIM은 NeuRAD, StreetGaussian 대비 LiDAR 입력 없이도 경쟁력 있는 외삽 성능을 달성
- 물리적 제약(지면 모델, 유니사이클 모델)이 외삽 뷰 품질을 크게 향상

### 5.3 3D 의미 재구성

KITTI-360에서 3D 의미 포인트 클라우드를 추출하여 평가:
- **정확도(acc)**: 예측 포인트에서 가장 가까운 LiDAR 포인트까지의 평균 거리
- **완전성(comp)**: LiDAR 포인트에서 가장 가까운 예측 포인트까지의 평균 거리
- **mIoU**: 3D 의미 분류 정확도

### 5.4 절제 연구 (Ablation Study)

**동적 장면 절제**: 노이즈 있는 3D bounding box에서 유니사이클 모델 최적화의 효과를 검증. 최적화 없이는 렌더링 품질과 3D 추적 정확도 모두 저하됨.

**정적 장면 절제**: 노출 모델링과 의미 손실의 효과 분석.

**지면 모델 절제**: $\mathcal{L}_{ground}$가 없으면 외삽 뷰에서 부유하는 Gaussian이 나타나는 아티팩트 발생.

---

## 6. Closed-Loop Benchmark

**요약**

HUGSIM 벤치마크는 KITTI-360, Waymo, nuScenes, PandaSet에서 70개 이상의 시퀀스로 구성된다. 난이도는 Easy, Medium, Hard, Extreme의 4단계.

### 6.1 벤치마크 구성

**지원 데이터셋**: KITTI-360 (6 카메라), Waymo (5 카메라), nuScenes (6 카메라)

**시나리오 난이도**:
- **Easy**: 주로 정적 장면, 간단한 직진 주행
- **Medium**: 차선 변경, IDM 행동 차량 포함
- **Hard**: 교차로, 회전, 공격적 액터 포함
- **Extreme**: 다수의 공격적 액터, 복잡한 교통 상황

### 6.2 평가 결과

평가된 AD 알고리즘:
- **UniAD**: 가장 강력한 성능, 대부분의 복잡한 시나리오에서 우수
- **VAD**: 중간 수준 성능
- **LatentTransformer (LTF)**: 쉬운 시나리오에서 양호하나 복잡한 상황에서 한계

**핵심 발견**:
1. nuScenes에서 훈련된 모델들은 KITTI-360과 Waymo에서 일반화 어려움 → 다양한 도메인 일반화 연구 필요
2. HD 맵 없이는 도로 이탈(DAC) 문제가 현저히 증가
3. 공격적 액터가 있는 Extreme 시나리오에서는 모든 기존 방법이 어려움을 겪음

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **3D Gaussian Splatting (3DGS)** | 장면을 3D Gaussian들의 집합으로 표현, 빠른 렌더링과 고품질 이미지 합성 가능 |
| **클로즈드루프 시뮬레이션** | 자율주행 알고리즘의 제어 명령이 시뮬레이터에 실시간으로 반영되는 피드백 루프 |
| **다중 평면 지면 모델** | 경사로 등 복잡한 지형을 처리하기 위해 지역적으로 평평한 여러 평면으로 지면을 모델링 |
| **유니사이클 모델** | 자동차의 단순화된 물리 모델로, 전진 속도와 조향각으로 움직임을 표현 |
| **HD-Score** | NC, DAC, TTC, COM을 결합한 자율주행 종합 평가 지표 |
| **공격적 액터** | 최적화 기반 생성으로 Ego 차량과의 충돌을 시도하는 시나리오 액터 |
| **구면 조화 함수 (SH)** | 방향에 따른 색상 변화(반사, 하이라이트 등)를 표현하는 수학적 도구 |
| **볼륨 렌더링** | 3D 공간의 Gaussian들을 누적하여 2D 이미지로 합성하는 방법 |
| **LQR 제어** | 유한한 비용으로 최적 제어 명령을 계산하는 선형 이차 조정기 |
| **IDM (Intelligent Driver Model)** | 앞 차량과의 거리와 속도를 기반으로 안전한 차간 거리를 유지하는 차량 추종 모델 |

---

## 결론 및 시사점

**주요 기여**

1. **HUGSIM 시스템**: 3DGS 기반 실시간, 포토리얼리스틱, 클로즈드루프 자율주행 시뮬레이터 최초 제안
2. **물리 기반 재구성**: 다중 평면 지면 모델과 유니사이클 기반 차량 재구성으로 외삽 뷰 품질 향상
3. **종합 벤치마크**: 70개 이상 시퀀스, 4개 데이터셋, 400개 이상 시나리오로 구성된 AD 평가 플랫폼
4. **HD-Score**: 클로즈드루프 평가에 특화된 새로운 복합 평가 지표

**실무적 시사점**

- 기존 AD 알고리즘들은 클로즈드루프 환경에서 오픈루프 평가 대비 현저히 낮은 성능을 보임 → 클로즈드루프 평가의 필요성 강조
- 도메인 일반화(domain generalization) 문제가 여전히 AD 알고리즘의 주요 과제
- 공격적 시나리오 생성을 통해 AD 알고리즘의 안전성을 체계적으로 평가 가능
- 포토리얼리스틱 시뮬레이션이 HD 맵 없이도 가능함을 보여줌으로써, 실용적인 AD 시스템 개발 경로 제시

**한계점 및 향후 연구 방향**

- 보행자 등 비-차량 동적 객체(pedestrian) 처리 필요 (현재 버전은 차량에 집중)
- 클로즈드루프에서 장거리 주행 시 누적 오차 문제 존재
- 극단적으로 빠른 속도나 급격한 방향 전환 시 렌더링 품질 저하


---

*관련 논문: [3D Gaussian Splatting](/posts/papers/3d-gaussian-splatting/), [4D Gaussian Splatting](/posts/papers/4d-gaussian-splatting/), [Street Gaussians](/posts/papers/street-gaussians-modeling-dynamic-urban-scenes/), [DrivingGaussian](/posts/papers/driving-gaussian-composite-gaussian-splatting/), [OmniRe](/posts/papers/omnire-omni-urban-scene-reconstruction/), [UniSim](/posts/papers/unisim-neural-closed-loop-sensor-simulator/), [CARLA](/posts/papers/CARLA-An-Open-Urban-Driving-Simulator/), [nuScenes](/posts/papers/nuscenes-multimodal-dataset-autonomous-driving/)*
