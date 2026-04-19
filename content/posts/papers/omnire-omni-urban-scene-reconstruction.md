---
title: "OmniRe: Omni Urban Scene Reconstruction"
date: 2026-04-19T11:00:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving", "Novel View Synthesis"]
tags: ["3D Gaussian Splatting", "Autonomous Driving", "Novel View Synthesis", "Dynamic Scene", "SMPL", "Human Modeling", "Digital Twin"]
---

## 개요

- **저자**: Ziyu Chen, Jiawei Yang, Jiahui Huang, Riccardo de Lutio, Janick Martinez Esturo, Boris Ivanovic, Or Litany, Zan Gojcic, Sanja Fidler, Marco Pavone, Li Song, Yue Wang
- **소속**: Shanghai Jiao Tong University, NVIDIA Research, University of Southern California, Stanford University, University of Toronto
- **발행년도**: 2025 (arXiv:2408.16760v2, 19 Apr 2025)
- **주요 내용**: 실제 주행 로그로부터 동적 도심 장면의 고품질 디지털 트윈을 효율적으로 생성하는 종합 시스템. 차량뿐만 아니라 보행자, 자전거, 기타 동적 객체를 모두 포함하는 holistic 재구성을 3D Gaussian Splatting 기반의 동적 Gaussian 씬 그래프로 구현하여 ~60 Hz 시뮬레이션 달성

## 한계 극복

- **기존 한계 1 — 차량 중심의 재구성**: StreetGaussians, HUGS 등 기존 방법들은 차량(강체)에 집중하며 보행자·자전거 등 Non-Rigid 객체를 대부분 무시
- **기존 한계 2 — 편집 불가능한 표현**: EmerNeRF, SUDS 등 단일 동적 필드 방식은 씬 전체를 하나로 모델링하여 객체별 제어 불가
- **기존 한계 3 — 실제 환경에서의 인간 모델링 어려움**: 드라이빙 로그의 심각한 가림(occlusion), 희소한 관측, 복잡한 자세 등으로 인해 기존 인간 자세 추정기가 실패하는 경우 다수
- **이 논문의 접근 방식**: Gaussian 씬 그래프에 SMPL 기반 SMPL 노드(보행자), Deformable 노드(자전거·기타), Rigid 노드(차량), Background 노드를 통합하여 모든 동적 요소를 holistic하게 재구성하고 joint-level 제어 지원

## 목차

1. Introduction
2. Related Work
3. Preliminaries (3DGS, SMPL)
4. Method
   - 4.1 Dynamic Gaussian Scene Graph Modeling
   - 4.2 Reconstructing In-The-Wild Humans
   - 4.3 Optimization
5. Experiments
6. Conclusion

## Chapter 3: Preliminaries

**요약**
OmniRe의 두 가지 핵심 기반 기술인 3D Gaussian Splatting(3DGS)과 SMPL 인체 모델을 소개합니다. 3DGS는 씬을 수백만 개의 작은 타원체(Gaussian)로 표현하고, SMPL은 인체를 수학적 파라미터로 제어 가능하게 표현합니다.

**핵심 개념**
- **3D Gaussian Splatting (3DGS)**: 씬을 colored blob들의 집합 $\mathcal{G} = \{g\}$으로 표현. 각 Gaussian $g = (o, \mu, \mathbf{q}, s, c)$는 불투명도, 위치, 회전(쿼터니언), 크기, 구형 조화 색상 계수를 가짐
- **SMPL**: 인체를 형태 파라미터 $\beta \in \mathbb{R}^{10}$와 자세 파라미터 $\theta \in \mathbb{R}^{23 \times 3 \times 3}$으로 표현하는 선형 블렌드 스키닝(LBS) 기반 파라메트릭 모델

**렌더링 수식**

$$C = \sum_{i \in \mathcal{N}} c_i \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j)$$

**수식 설명**
이 수식은 카메라 방향으로 여러 Gaussian들을 겹쳐 최종 픽셀 색상을 계산합니다:
- **$C$**: 최종 픽셀 색상 (화면에 보이는 색)
- **$c_i$**: $i$번째 Gaussian의 색상
- **$\alpha_i$**: $i$번째 Gaussian의 투명도. $\alpha_i = o_i \exp(-\frac{1}{2}(\mathbf{p} - \mu_i)^T \Sigma_i^{-1} (\mathbf{p} - \mu_i))$로 계산
- **$\prod_{j=1}^{i-1}(1 - \alpha_j)$**: $i$번째 Gaussian에 도달하기까지 앞의 모든 Gaussian을 통과한 빛의 투과율 (앞이 불투명할수록 뒤가 덜 보임)

**Affine 변환**

$$\mathbf{T} \otimes \mathcal{G} = (o, \mathbf{R}\mu + \mathbf{t}, \text{Rot}(\mathbf{R}, \mathbf{q}), s, c)$$

**수식 설명**
강체 변환 $\mathbf{T} = (\mathbf{R}, \mathbf{t})$를 Gaussian 집합에 적용하는 연산:
- **$\mathbf{R}$**: 회전 행렬
- **$\mathbf{t}$**: 이동 벡터
- **$\text{Rot}(\mathbf{R}, \mathbf{q})$**: 회전 행렬로 쿼터니언을 업데이트하는 연산

## Chapter 4: Method

**요약**
OmniRe는 씬의 모든 구성 요소를 Gaussian 씬 그래프로 모델링합니다. 배경, 차량(강체), 보행자(SMPL 기반), 기타 비강체(Deformable), 하늘의 5가지 노드 타입으로 씬을 분해하고, 각각에 최적화된 Gaussian 표현을 사용합니다.

### 4.1 Dynamic Gaussian Scene Graph

**핵심 개념**

- **Sky Node**: 하늘처럼 멀리 있는 영역을 최적화 가능한 환경 텍스처 맵으로 표현
- **Background Node**: 건물, 도로, 식생 등 정적 배경을 LiDAR 포인트로 초기화한 정적 Gaussian으로 표현
- **Rigid Nodes** (차량·버스): 로컬 정준 공간에 정의된 Gaussian을 시간 $t$의 포즈 $\mathbf{T}_v(t) \in \mathbb{SE}(3)$로 월드 공간에 변환
- **SMPL Nodes** (보행자): SMPL 파라미터 $(\theta(t), \beta)$로 구동되는 Gaussian으로 joint-level 제어 지원
- **Deformable Nodes** (자전거·기타): 정준 공간 Gaussian을 변형 네트워크 $\mathcal{F}_\varphi$로 변형하여 템플릿 없는 비강체 표현

**Rigid Node 변환 수식**

$$\mathcal{G}_v^{\text{rigid}}(t) = \mathbf{T}_v(t) \otimes \bar{\mathcal{G}}_v^{\text{rigid}}$$

**수식 설명**
- **$\bar{\mathcal{G}}_v^{\text{rigid}}$**: 로컬 공간에서의 차량 Gaussian (시간에 따라 변하지 않음)
- **$\mathbf{T}_v(t)$**: 시간 $t$에서의 차량 포즈 변환 행렬
- 차량은 자체 모양은 변하지 않고 위치·방향만 바뀌므로 강체 변환만 적용

**SMPL Node 수식**

$$\mathcal{G}_h^{\text{SMPL}}(t) = \mathbf{T}_h(t) \otimes \text{LBS}(\theta(t), \mathcal{G}_h^{\text{SMPL}})$$

**수식 설명**
- **$\mathbf{T}_h(t)$**: 시간 $t$에서의 보행자 전역 포즈
- **$\text{LBS}(\cdot)$**: 선형 블렌드 스키닝. SMPL 자세 파라미터 $\theta(t)$에 따라 Gaussian들을 변형
- **$\mathcal{G}_h^{\text{SMPL}}$**: 'Da' 포즈(기본 T자 포즈)의 정준 Gaussian
- 각 Gaussian은 SMPL 메쉬의 특정 꼭짓점에 묶여 자세 변화에 따라 자연스럽게 움직임

**Deformable Node 수식**

$$\mathcal{G}_h^{\text{deform}}(t) = \mathbf{T}_h(t) \otimes \left(\bar{\mathcal{G}}_h^{\text{deform}} \oplus \mathcal{F}_\varphi(\mathcal{G}_h^{\text{deform}}, e_h, t)\right)$$

**수식 설명**
- **$\mathcal{F}_\varphi$**: 변형 네트워크. 정준 Gaussian의 위치·회전·크기 변화량 $(\delta\mu, \delta q, \delta s)$을 예측
- **$e_h$**: 인스턴스 임베딩. 각 노드의 정체성을 구분하는 벡터 (하나의 네트워크로 여러 객체 처리 가능)
- **$\oplus$**: 변화량을 Gaussian에 적용하는 연산자

**최종 렌더링 수식**

$$C = C_{\mathcal{G}} + (1 - O_{\mathcal{G}}) C_{\text{sky}}$$

**수식 설명**
- **$C_{\mathcal{G}}$**: 모든 Gaussian(배경+전경)을 렌더링한 결과
- **$O_{\mathcal{G}} = \sum_{i=1}^{N} \alpha_i \prod_{j=1}^{i-1}(1-\alpha_j)$**: Gaussian의 전체 불투명도 마스크
- **$C_{\text{sky}}$**: 하늘 환경 맵 색상. Gaussian이 불투명하지 않은 부분(하늘)만 합성

### 4.2 실제 환경에서의 인간 재구성

**요약**
드라이빙 환경에서 보행자의 정확한 SMPL 자세를 추출하는 것은 심각한 가림과 다중 카메라 환경 때문에 매우 어렵습니다. OmniRe는 두 단계로 이 문제를 해결합니다.

**핵심 개념**

- **Tracklet Matching**: 각 카메라별로 독립적으로 추출된 인체 트랙렛을 ground truth 바운딩 박스와 매칭
  - $\tilde{\theta}_h = \mathcal{M}(h, \hat{\theta}, \mathbf{T}, \hat{\mathbf{T}})$ — 2D 투영의 최대 평균 IoU로 최적 매칭 탐색
- **Pose Completion**: 가림으로 누락된 자세를 보간하여 완성
  - $\theta_h = \mathcal{H}(\tilde{\theta}_h, \mathbf{T}, \hat{\mathbf{T}})$ — 누락 탐지 후 인접 프레임에서 보간

### 4.3 최적화

**목적 함수**

$$\mathcal{L} = (1 - \lambda_r)\mathcal{L}_1 + \lambda_r \mathcal{L}_{\text{SSIM}} + \lambda_{\text{depth}}\mathcal{L}_{\text{depth}} + \lambda_{\text{opacity}}\mathcal{L}_{\text{opacity}} + \mathcal{L}_{\text{reg}}$$

**수식 설명**
5가지 손실의 가중합:
- **$\mathcal{L}_1$, $\mathcal{L}_{\text{SSIM}}$**: 렌더링 이미지와 실제 이미지의 픽셀 및 구조적 유사도 손실 ($\lambda_r = 0.2$)
- **$\mathcal{L}_{\text{depth}}$**: LiDAR 희소 깊이와 렌더링 깊이의 L1 손실 ($\lambda_{\text{depth}} = 0.1$)
- **$\mathcal{L}_{\text{opacity}}$**: 하늘 마스크를 활용한 불투명도 손실 ($\lambda_{\text{opacity}} = 0.05$)
- **$\mathcal{L}_{\text{reg}}$**: 각 Gaussian 표현별 정규화 손실 (자세 부드러움 등 포함)

모든 파라미터를 **단일 단계(single stage)**에서 동시 최적화. 30,000회 반복, NVIDIA RTX 4090 GPU 1장, 씬당 약 1시간 소요.

## Chapter 5: Experiments

**요약**
Waymo Open Dataset의 32개 동적 씬(그 중 8개는 보행자·자전거 포함 복잡 씬)에서 평가. NuScenes, Argoverse2, PandaSet, KITTI, NuPlan 등 5개 추가 데이터셋으로 일반화 검증.

**주요 결과 (Waymo, 씬 재구성 Full Image PSNR)**

| 방법 | Full PSNR | Human PSNR | Vehicle PSNR |
|---|---|---|---|
| EmerNeRF | 31.92 | 22.88 | 24.05 |
| 3DGS | 26.00 | 16.88 | 16.18 |
| DeformableGS | 28.40 | 17.80 | 19.53 |
| StreetGS | 29.08 | 16.83 | 27.73 |
| **OmniRe (Ours)** | **34.25** | **28.15** | **28.91** |

- 전체 이미지 PSNR: +1.88 dB 향상 (vs. 최고 기존 방법)
- 인간 영역 PSNR: **+4.09 dB** 향상 (가장 큰 차이)
- LiDAR 깊이 재구성 CD: 0.242 (3DGS 0.415, StreetGS 0.274 대비 현저히 낮음)

**Ablation 주요 결과**

| 설정 | Full PSNR | Human PSNR |
|---|---|---|
| (a) Ours default | 34.25 | 28.15 |
| (b) w/o SMPL actors | 32.80 | 24.71 |
| (c) w/o Body pose refine | 33.84 | 26.97 |
| (d) w/o Deformed actors | 33.64 | 25.26 |

SMPL 모델링이 인간 영역 PSNR에 +3.44 dB 기여.

## 핵심 개념 정리

- **Gaussian 씬 그래프**: 씬을 여러 노드(Sky, Background, Rigid, SMPL, Deformable)로 분해하는 계층적 표현. 각 노드가 독립적으로 편집·이동 가능
- **SMPL 기반 Gaussian**: SMPL 인체 모델의 꼭짓점에 Gaussian을 묶어 자세를 정확히 추적하며 joint-level 제어 지원
- **Per-Node Deformation Field**: 전체 씬에 하나의 변형 네트워크를 쓰는 대신, 각 비강체 노드마다 독립적인 deformation field를 인스턴스 임베딩으로 구분하여 표현력 대폭 향상
- **Tracklet Matching + Pose Completion**: 다중 카메라 환경에서 카메라 간 ID 불일치를 해소하고 가림으로 누락된 자세를 보간하는 in-the-wild 인간 자세 파이프라인
- **Holistic Reconstruction**: 차량·보행자·자전거·기타 등 모든 동적 객체를 단일 프레임워크에서 통합 재구성. 이전 방법들은 차량 또는 인간에만 집중

## 결론 및 시사점

OmniRe는 기존 방법들이 차량에만 집중하던 한계를 깨고, 도심 씬의 모든 동적 요소(차량, 보행자, 자전거, 기타)를 통합적으로 재구성하는 최초의 종합 시스템입니다.

**실무적 시사점**
- **자율주행 시뮬레이션**: 보행자 행동 시뮬레이션, 차량-보행자 상호작용 등 ~60 Hz 실시간 시뮬레이션 가능
- **합성 데이터 생성**: 재구성된 씬에서 동적 객체를 자유롭게 삽입·편집하여 다양한 시나리오 생성 (다른 씬의 자산 이식 포함)
- **회귀 테스트**: 실제 로그 기반 디지털 트윈으로 알고리즘 평가 환경 구성 가능

**한계점**
- 조명 효과를 명시적으로 모델링하지 않아 다른 조건의 요소를 합성 시 시각적 불일치 가능
- 훈련 궤적에서 카메라가 크게 벗어난 Novel View에서의 품질 저하
- 향후 데이터 기반 생성 모델(비디오 생성 등)과의 결합으로 개선 가능
