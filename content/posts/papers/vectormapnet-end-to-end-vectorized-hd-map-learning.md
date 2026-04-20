---
title: "VectorMapNet: End-to-end Vectorized HD Map Learning"
date: 2026-04-20T20:00:00+09:00
draft: false
categories: ["Papers"]
tags: ["Autonomous Driving", "HD Map", "BEV", "Transformer", "DETR", "Polyline"]
---

## 개요

- **저자**: Yicheng Liu, Tianyuan Yuan, Yue Wang, Yilun Wang, Hang Zhao
- **소속**: Shanghai Qi Zhi Institute, Tsinghua University, MIT, Li Auto
- **학회**: ICML 2023 (Proceedings of the 40th International Conference on Machine Learning)
- **arXiv**: 2206.08920
- **주요 내용**: 온보드 센서(카메라, LiDAR)로부터 벡터화된 HD 맵을 end-to-end로 직접 예측하는 최초의 학습 기반 파이프라인

## 한계 극복

이 논문은 기존 HD 맵 생성 방법들의 근본적인 한계를 극복하기 위해 작성되었습니다.

- **기존 한계 1 — 오프라인 수동 어노테이션**: 기존 HD 맵은 사람이 직접 주석을 달아야 하므로 확장성이 매우 낮고 비용이 큼
- **기존 한계 2 — 래스터 기반 출력**: HDMapNet 등의 학습 기반 방법은 픽셀 단위의 세그멘테이션 맵을 출력하므로 인스턴스 정보가 없고, 벡터 맵 변환을 위해 별도의 휴리스틱 후처리가 필요함
- **기존 한계 3 — 공간적 일관성 부재**: 래스터 예측은 인접 픽셀 간에 모순된 기하 구조가 생길 수 있으며, 2D/3D 다운스트림 작업과 호환되지 않음
- **이 논문의 접근 방식**: 맵 요소를 폴리라인(polyline) 집합으로 표현하고, 탐지(detection) + 생성(generation) 두 단계로 구성된 end-to-end 파이프라인으로 벡터 맵을 직접 예측

## 목차

- Chapter 1: Introduction
- Chapter 2: Related Works
- Chapter 3: VectorMapNet (방법론)
  - 3.1 Method Overview
  - 3.2 BEV Feature Extractor
  - 3.3 Map Element Detector
  - 3.4 Polyline Generator
  - 3.5 Learning (손실 함수)
- Chapter 4: Experiments
  - 4.1 Comparison with Baselines
  - 4.2 Qualitative Analysis
  - 4.3 Ablation Studies
  - 4.4 Motion Forecasting with Vectorized HD Maps
- Chapter 5: Discussions
- Chapter 6: Conclusions

---

## Chapter 1: Introduction

**요약**

자율주행 시스템은 차선, 보행자 횡단보도, 도로 경계 등 다양한 맵 요소를 담은 HD(High-Definition) 시맨틱 맵이 필요합니다. 기존에는 이러한 맵을 오프라인으로 사람이 직접 어노테이션했는데, 이는 확장성 문제가 심각합니다. 최근 학습 기반 방법들은 BEV(Bird's Eye View) 래스터 세그멘테이션으로 맵을 구성하지만, 이는 인스턴스 정보가 없고 후처리가 필요합니다.

VectorMapNet은 이러한 한계를 극복하기 위해 온보드 센서 데이터(카메라 이미지, LiDAR 포인트)를 직접 입력받아 BEV 피처를 추출하고, 맵 요소의 위치를 탐지한 후, 각 요소를 폴리라인으로 생성하는 end-to-end 파이프라인입니다. nuScenes와 Argoverse2 데이터셋에서 기존 SOTA를 각각 14.2 mAP, 14.6 mAP 이상 뛰어넘었습니다.

**핵심 개념**

- **HD Map (고정밀 지도)**: 차선 경계, 차선 구분선, 보행자 횡단보도 등 cm 단위 정밀도의 맵. 자율주행의 측위 및 경로 계획에 필수
- **BEV (Bird's Eye View)**: 차량 위에서 내려다본 시점으로 변환한 특징 맵. 카메라나 LiDAR의 여러 뷰를 통합하기 좋은 공간
- **Polyline (폴리라인)**: 순서가 있는 정점(vertex)들의 집합으로 구성된 선분. 도로 경계처럼 임의 형태의 곡선을 잘 표현함
- **End-to-end Learning**: 전처리·후처리 없이 입력에서 최종 출력까지 하나의 네트워크로 학습하는 방식

---

## Chapter 2: Related Works

**요약**

관련 연구는 크게 세 가지로 나뉩니다: (1) BEV 시맨틱 맵 학습, (2) 차선 탐지, (3) 기하 데이터 모델링.

BEV 시맨틱 맵 학습은 주로 항공 이미지, LiDAR, 파노라마 등을 활용한 세그멘테이션 방식이며, HDMapNet이 가장 가까운 선행 연구입니다. 차선 탐지는 픽셀 수준 세그멘테이션 또는 베지어 곡선 등 핸드크래프트된 파라미터 기반 방법이 주를 이룹니다. 기하 데이터 모델링 분야에서는 폴리곤, 스트로크, SVG 프리미티브 등을 시퀀스로 생성하는 자기회귀 모델들이 연구되었으며, VectorMapNet은 이 접근법을 HD 맵으로 가져옵니다.

**핵심 개념**

- **HDMapNet**: 래스터 세그멘테이션 + 후처리로 벡터 맵을 생성하는 선행 연구. VectorMapNet의 주요 비교 대상
- **STSU (Can et al., 2021)**: 단안 카메라로 BEV HD 맵을 직접 구성하는 방법. 고정 크기 세그먼트를 사용해 세밀한 기하 구조 포착이 어려움
- **PolyGen / DETR**: 폴리곤 생성을 자기회귀 방식으로 학습하거나 집합 예측(set prediction) 문제로 다루는 접근법

---

## Chapter 3: VectorMapNet

### 3.1 문제 정의 및 방법 개요

**요약**

VectorMapNet은 맵 구성 문제를 **희소 폴리라인 집합 예측 문제(sparse set detection problem)**로 정의합니다. 맵 $\mathcal{M}$은 $N$개의 폴리라인 $\mathcal{V}^{\text{poly}} = \{V_1^{\text{poly}}, \ldots, V_N^{\text{poly}}\}$로 표현되며, 각 폴리라인은 순서가 있는 정점들의 집합입니다.

이 파이프라인은 세 가지 핵심 모듈로 구성됩니다:
1. **BEV Feature Extractor**: 카메라/LiDAR 입력을 공통 BEV 피처로 변환
2. **Map Element Detector**: BEV 피처에서 맵 요소의 위치(키포인트)와 클래스를 탐지
3. **Polyline Generator**: 탐지된 위치 정보를 바탕으로 실제 폴리라인 정점 시퀀스를 생성

**핵심 개념**

- **폴리라인 표현의 장점 3가지**:
  1. 점, 선, 곡선, 폴리곤 등 다양한 기하 구조를 하나의 형식으로 통합
  2. 정점 순서가 자연스럽게 맵 요소의 방향 정보를 인코딩
  3. 모션 예측 등 다운스트림 작업에서 폴리라인 형식을 직접 소비할 수 있음
- **Ramer-Douglas-Peucker 알고리즘**: 폴리곤을 폴리라인으로 근사 변환할 때 사용하는 곡선 단순화 알고리즘

### 3.2 BEV Feature Extractor

**요약**

BEV Feature Extractor는 서로 다른 센서 모달리티(카메라, LiDAR)의 특징을 공통 BEV 공간으로 정렬·통합합니다. 최종 출력은 $\mathcal{F}_{\text{BEV}} \in \mathbb{R}^{W \times H \times (C_1 + C_2)}$ 형태의 BEV 피처 맵입니다.

- **카메라 브랜치**: ResNet으로 이미지 피처 추출 후, IPM(Inverse Perspective Mapping)을 활용해 BEV 피처 $\mathcal{F}_{\text{BEV}}^{\mathcal{I}} \in \mathbb{R}^{W \times H \times C_1}$ 생성
- **LiDAR 브랜치**: PointPillars(dynamic voxelization 변형)로 포인트 클라우드를 BEV 피처 $\mathcal{F}_{\text{BEV}}^{\mathcal{P}} \in \mathbb{R}^{W \times H \times C_2}$ 로 변환
- **Fusion**: 두 피처를 채널 방향으로 concat하고 2층 컨볼루션 레이어로 처리

**수식**

$$\mathcal{F}_{\text{BEV}} = \text{Conv}([\mathcal{F}_{\text{BEV}}^{\mathcal{I}}; \mathcal{F}_{\text{BEV}}^{\mathcal{P}}])$$

**수식 설명**

- **$\mathcal{F}_{\text{BEV}}^{\mathcal{I}}$**: 카메라로부터 얻은 BEV 피처 맵 (이미지 브랜치)
- **$\mathcal{F}_{\text{BEV}}^{\mathcal{P}}$**: LiDAR로부터 얻은 BEV 피처 맵 (포인트 클라우드 브랜치)
- **$[;]$**: 채널 방향 연결(concatenation)
- **$\text{Conv}(\cdot)$**: 두 피처를 합쳐 하나로 정제하는 2층 컨볼루션 네트워크

**핵심 개념**

- **IPM (Inverse Perspective Mapping)**: 카메라의 원근 뷰를 기하학적 변환으로 조감도(BEV)로 바꾸는 방법. 바닥이 평평하다는 가정 하에 동작
- **PointPillars**: 3D 공간을 기둥(pillar) 단위로 분할하고 포인트 클라우드를 2D BEV 피처로 빠르게 변환하는 LiDAR 인코더

### 3.3 Map Element Detector

**요약**

Map Element Detector는 BEV 피처에서 각 맵 요소의 개략적인 위치(키포인트)와 클래스를 예측합니다. DETR(Detection Transformer) 구조를 맵 탐지에 맞게 적용했으며, 학습 가능한 **Element Query** $q_i^{\text{elem}} \in \mathbb{R}^{k \times d}$를 사용해 각 맵 요소를 표현합니다.

각 요소 쿼리는 $k$개의 **Keypoint Embedding** $q_{i,j}^{\text{kp}}$로 구성되며, 이 키포인트들이 맵 요소의 전체적인 위치와 형태를 추상화합니다. Transformer Decoder가 self/cross-attention을 통해 BEV 피처와 상호작용하며 키포인트 위치 $\mathcal{A}_i \in \mathbb{R}^{k \times 2}$와 클래스 $l_i$를 예측합니다.

**키포인트 표현 종류 (ablation에서 비교)**

| 종류 | k | 설명 |
|------|---|------|
| Bounding Box (Bbox) | 2 | 폴리라인을 감싸는 최소 박스의 우상단·좌하단 점 |
| Start-Middle-End (SME) | 3 | 폴리라인의 시작, 중간, 끝점 |
| Extreme Points | 4 | 가장 왼쪽, 오른쪽, 위쪽, 아래쪽 점 |

**수식**

$$a_{i,j} = \text{MLP}_{kp}(q_{i,j}^{\text{kp}}), \quad l_i = \text{MLP}_{cls}([q_{i,1}^{\text{kp}}, \ldots, q_{i,k}^{\text{kp}}])$$

**수식 설명**

- **$a_{i,j}$**: $i$번째 맵 요소의 $j$번째 키포인트 위치 예측값 (2D 좌표)
- **$q_{i,j}^{\text{kp}}$**: $i$번째 요소의 $j$번째 키포인트 임베딩 벡터
- **$\text{MLP}_{kp}$**: 키포인트 위치를 회귀하는 MLP 헤드
- **$\text{MLP}_{cls}$**: 모든 키포인트 임베딩을 concat해 클래스를 분류하는 MLP 헤드
- **$[\cdot]$**: 벡터 연결(concatenation)

**핵심 개념**

- **Deformable Attention**: 표준 cross-attention 대신 예측된 2D 위치 주변의 특정 포인트들에만 집중하는 어텐션. 수렴 속도가 빠르고 효율적
- **Bipartite Matching Loss**: 예측 집합과 정답 집합 사이의 최적 매칭을 헝가리안 알고리즘으로 찾아 NMS 없이 훈련하는 DETR 방식의 손실

### 3.4 Polyline Generator

**요약**

Polyline Generator는 Map Element Detector가 예측한 키포인트 위치 $\mathcal{A}_i$와 클래스 $l_i$를 조건으로, 각 폴리라인의 실제 정점 시퀀스를 자기회귀(autoregressive) 방식으로 생성합니다.

정점 좌표를 **이산(discrete) 변수**로 처리합니다: 연속 좌표값을 양자화(quantize)해 이산 토큰으로 변환하고, 각 정점을 범주형 분포(categorical distribution)로 모델링합니다. 이렇게 하면 멀티모달, 비대칭적, 두꺼운 꼬리를 가진 복잡한 분포도 쉽게 표현할 수 있습니다.

아키텍처는 Transformer이며, 각 정점 토큰은 세 가지 임베딩의 합으로 표현됩니다:
- **Coordinate Embedding**: 현재 토큰이 $x$ 좌표인지 $y$ 좌표인지 구분
- **Position Embedding**: 시퀀스에서 몇 번째 정점인지 표시
- **Value Embedding**: 양자화된 좌표값 자체

**수식**

$$p(\mathcal{V}_i^{\text{poly}} | \mathcal{A}_i, l_i, \mathcal{F}_{\text{BEV}}; \theta) = \prod_{n=1}^{2N_v} p(v_{i,n}^f | v_{i,<n}^f, \mathcal{A}_i, l_i, \mathcal{F}_{\text{BEV}})$$

**수식 설명**

폴리라인의 모든 정점을 순서대로 예측하는 자기회귀 분해입니다:

- **$\mathcal{V}_i^{\text{poly}}$**: $i$번째 맵 요소의 폴리라인 (예측 대상)
- **$N_v$**: 폴리라인의 정점 수. 각 정점은 $(x, y)$ 2개의 좌표값을 가지므로 총 $2N_v$개의 토큰
- **$v_{i,n}^f$**: $n$번째 좌표값 토큰 (평탄화된 시퀀스)
- **$v_{i,<n}^f$**: $n$번째 이전까지 이미 예측된 좌표값들 (자기회귀 조건)
- **$\mathcal{A}_i, l_i$**: 탐지기에서 넘겨받은 키포인트 위치와 클래스 레이블 (조건)
- **$\mathcal{F}_{\text{BEV}}$**: BEV 피처 맵 (세부 기하 구조 파악에 활용)
- **$\prod$**: 각 토큰을 독립적으로 조건부 예측하여 전체 시퀀스 확률을 곱으로 구성

### 3.5 Learning (학습 방법)

**요약**

총 손실은 탐지기 손실과 생성기 손실의 합입니다:

$$\mathcal{L} = \mathcal{L}_{det} + \mathcal{L}_{gen}$$

**탐지기 손실** $\mathcal{L}_{det}$: DETR 방식의 bipartite matching loss. NMS 없이 집합 수준의 예측을 직접 학습

**생성기 손실** $\mathcal{L}_{gen}$: 폴리라인 정점의 negative log-likelihood

$$\mathcal{L}_{gen} = -\frac{1}{2N_v} \sum_{n=1}^{2N_v} \log \hat{p}(v_{i,n}^f | v_{i,<n}^f, \mathcal{A}_i, l_i, \mathcal{F}_{\text{BEV}})$$

**수식 설명**

- **$\hat{p}(v_{i,n}^f | \ldots)$**: 모델이 예측한 $n$번째 좌표 토큰의 조건부 확률
- **$\log$**: 확률의 로그를 취해 곱셈을 덧셈으로 변환
- **$-\frac{1}{N_v}$**: 정규화 계수. 정점 수가 다른 폴리라인들 간의 손실 크기를 균일하게 맞춤
- 이 손실을 최소화하면 모델이 정답 좌표 시퀀스에 높은 확률을 부여하도록 학습됨

**두 단계 학습 전략**: Teacher forcing으로 먼저 학습한 후, 예측된 키포인트를 입력으로 fine-tuning. 이를 통해 exposure bias(훈련-추론 간 분포 차이)를 완화하고 성능을 향상시킴

---

## Chapter 4: Experiments

**요약**

nuScenes와 Argoverse2 두 데이터셋에서 실험. 평가 지표는 Chamfer AP(Chamfer distance 기반 평균 정밀도)와 Fréchet AP를 사용. 임계값 $\{0.5, 1.0, 1.5\}$m에서 AP를 계산해 mAP로 평균.

**nuScenes 결과 (Table 1)**

| 방법 | AP_ped | AP_divider | AP_boundary | mAP |
|------|--------|-----------|-------------|-----|
| STSU (Camera) | 7.0 | 11.6 | 16.5 | 11.7 |
| HDMapNet (Fusion) | 16.3 | 29.6 | 46.7 | 31.0 |
| **VectorMapNet (Fusion)** | 37.6 | 50.5 | 47.5 | 45.2 |
| **VectorMapNet (Fusion) + fine-tune** | **48.2** | **60.1** | **53.0** | **53.7** |

VectorMapNet은 HDMapNet 대비 +22.7 mAP를 달성. Sensor fusion이 단일 모달보다 각각 +4.3 mAP(Camera 대비), +11.2 mAP(LiDAR 대비) 향상.

**정성적 분석**

- **폴리라인의 장점 1**: 날카로운 코너(모서리)를 정확하게 표현. 래스터 방법은 코너가 흐릿해짐
- **폴리라인의 장점 2**: 자기루프(self-looping) 오류 없음. 래스터 기반 방법은 반복되는 패턴에서 루프를 생성하는 오류 발생
- **탐지 방식의 장점**: 모델이 어노테이션되지 않은 보행자 횡단보도도 발견(Figure 6) — 씬 이해 능력 보유

**Ablation: 키포인트 표현 (Table 3)**

Bounding Box(k=2)가 Fréchet mAP 52.4, Chamfer mAP 40.9로 가장 우수. SME, Extreme Points 대비 각각 +2.0, +7.3 Chamfer mAP 우세.

**모션 예측 응용 (Table 4)**

VectorMapNet으로 예측한 HD 맵을 모션 예측 모델(mmTransformer)에 입력하면, 궤적만 사용할 때보다 minADE −0.083, minFDE −0.100 개선. 정답 맵 사용 시와 성능 차이가 작음(MR −0.2%), 예측 맵의 실용성 검증.

**핵심 개념**

- **Chamfer Distance**: 두 점 집합 간 거리의 평균. 순서 무관하게 가장 가까운 점끼리 매칭
- **Fréchet Distance**: 두 곡선 사이의 최대 거리를 최소화하는 방향(순서 고려). 폴리라인의 형태와 방향을 동시에 평가
- **minADE / minFDE**: 여러 예측 궤적 중 가장 정답에 가까운 것의 평균/최종 변위 오차

---

## Chapter 5: Discussions

**요약**

VectorMapNet의 세 가지 한계:

1. **시간 정보 부재**: 단일 프레임 입력으로 시간적 일관성을 보장하지 못함. 프레임 간 예측 결과가 일관되지 않을 수 있음
2. **Two-stage 불일치**: 탐지기와 생성기 간 피처 공간 불일치. Teacher forcing으로 인해 fine-tuning 스케줄이 복잡함
3. **환각(Hallucination) 능력**: 폐색 영역에서도 맵 요소를 예측할 수 있어 씬 이해력을 보여주지만, 해석 가능성을 낮춤

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **HD Map** | 자율주행을 위한 cm 단위 정밀 시맨틱 지도 |
| **BEV (Bird's Eye View)** | 다양한 센서 뷰를 통합하는 조감도 특징 공간 |
| **Polyline** | 순서 있는 정점 집합으로 임의 형태의 맵 요소를 표현하는 기본 프리미티브 |
| **DETR** | 이분 매칭으로 NMS 없이 집합 예측을 학습하는 탐지 트랜스포머 |
| **Deformable Attention** | 예측 위치 주변 지점에만 집중해 효율적이고 빠르게 수렴하는 어텐션 메커니즘 |
| **Autoregressive Generation** | 이전 출력을 조건으로 다음 토큰을 순차 예측하는 생성 방식 |
| **Teacher Forcing** | 훈련 중 정답 시퀀스를 다음 스텝의 입력으로 사용하는 학습 기법 |
| **Chamfer / Fréchet AP** | 폴리라인 간 유사도를 측정하는 두 가지 거리 기반 평균 정밀도 지표 |
| **IPM** | 카메라 원근 뷰를 BEV로 변환하는 역 원근 매핑 |
| **PointPillars** | LiDAR 포인트 클라우드를 2D BEV 피처로 변환하는 빠른 인코더 |

---

## 결론 및 시사점

VectorMapNet은 HD 맵 학습을 **래스터 세그멘테이션 문제**에서 **희소 폴리라인 집합 예측 문제**로 재정의한 첫 번째 end-to-end 접근법입니다.

**핵심 성과**:
- nuScenes에서 HDMapNet 대비 +22.7 mAP (Fusion + fine-tune 기준)
- 후처리 없이 인스턴스 수준의 벡터 맵 직접 생성
- 모션 예측 등 다운스트림 작업에 바로 적용 가능한 출력 형식

**실무적 시사점**:
- 자율주행 합성 데이터 생성 시, 현실적인 맵 요소 형태를 폴리라인으로 표현하면 다운스트림 호환성이 높아짐
- 탐지 + 생성 두 단계 분리는 각 모듈의 역할을 명확히 하고 디버깅을 용이하게 함
- 향후 시계열 입력(multi-frame fusion)과 단일 스테이지 통합을 통한 개선 여지 존재
- MapTR, BeMapNet 등 후속 연구의 발판이 되는 핵심 기초 논문
