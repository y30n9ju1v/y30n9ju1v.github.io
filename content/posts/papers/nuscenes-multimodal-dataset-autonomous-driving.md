---
title: "nuScenes: A Multimodal Dataset for Autonomous Driving"
date: 2026-04-10T09:00:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving", "Benchmark & Dataset"]
tags: ["Autonomous Driving", "Dataset", "3D Object Detection", "Multi-Modal", "LiDAR"]
---

## 개요

- **저자**: Holger Caesar, Varun Bankiti, Alex H. Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, Oscar Beijbom (nuTonomy, an APTIV company)
- **발행년도**: 2020
- **학회**: CVPR 2020
- **arXiv**: 1903.11027
- **주요 내용**: 자율주행을 위한 최초의 완전한 멀티모달 공개 데이터셋. 6개의 카메라, 5개의 레이더, 1개의 라이다로 구성된 전방위 센서 슈트를 갖추고 있으며, 1000개의 씬, 23개 클래스, 140만 개 이상의 3D 바운딩 박스 어노테이션을 포함한다.

## 목차

- Section 1: Introduction
- Section 2: The nuScenes Dataset
- Section 3: Tasks & Metrics
- Section 4: Experiments
- Section 5: Conclusion
- Supplementary: Dataset Details & Implementation

---

## Section 1: Introduction

**요약**

자율주행 시스템에서 환경 내 객체의 탐지(detection)와 추적(tracking)은 핵심 과제다. 기존에는 이미지 기반 벤치마크가 주를 이뤘지만, 라이다·레이더 같은 거리 센서를 포함한 멀티모달 데이터셋은 부족했다. 특히 KITTI는 이미지와 라이다만 제공하며 레이더 데이터가 없고 어노테이션 수도 적었다.

nuScenes는 이러한 간극을 메우기 위해 제작된 첫 자율주행 차량용 완전한 센서 슈트 데이터셋이다. KITTI 대비 7배 이상의 어노테이션, 100배 이상의 데이터를 제공하며, 레이더 데이터를 포함한 최초의 자율주행 공개 데이터셋이다.

**핵심 개념**

- **멀티모달 센서 융합의 필요성**: 카메라는 색상·질감·분류에 강하지만 3D 위치 추정이 어렵다. 라이다는 3D 정확도가 높지만 의미론적 정보가 부족하다. 레이더는 200~300m까지 감지하고 도플러 효과로 속도를 직접 측정한다. 세 가지를 결합하면 각각의 단점을 보완할 수 있다.
- **KITTI의 한계**: 레이더 없음, 어노테이션 수 적음(KITTI 대비 nuScenes는 7배 어노테이션, 100배 데이터), 카메라가 전방만 촬영
- **nuScenes의 차별점**: 360° 커버리지, 레이더 포함, 대규모 어노테이션, 새로운 3D 탐지·추적 메트릭 정의

---

## Section 2: The nuScenes Dataset

**요약**

nuScenes 데이터셋의 수집 방식, 센서 구성, 지도, 어노테이션 방법론을 설명한다. 미국 보스턴과 싱가포르에서 두 대의 Renault Zoe 차량으로 수집되었으며, 주·야간, 맑음·비·건설현장 등 다양한 환경 조건을 포함한다.

**핵심 개념**

- **센서 구성**
  - **6개 카메라**: 1/1.8" CMOS 센서, 1600×900 해상도, 12Hz 촬영. 전방위 360° 커버리지
  - **1개 라이다**: 360° 수평 시야, 초당 20회전, 32채널, 최대 70m 범위
  - **5개 레이더**: 전방 1개 + 전방 좌우 + 후방 좌우 배치, 200~300m 범위, 도플러 속도 측정
  - **GPS & IMU**: 100Hz, 위치·자세 정보 제공

- **데이터 규모**
  - 1000개 씬, 각 20초 길이
  - 40,000개 키프레임 (2Hz 샘플링)
  - 23개 카테고리, 8개 속성
  - 140만 개 이상의 3D 바운딩 박스 어노테이션

- **지도(Map)**: 11개 시맨틱 레이어(도로, 인도, 교차로 등)를 포함하는 정밀 HD 맵 제공. 벡터화된 방식으로 저장하여 효율적 접근 가능

- **센서 동기화**: 라이다 1회전 완료 시점에 카메라 셔터가 트리거됨 → 타임스탬프 정렬이 자연스럽게 달성됨

- **로컬라이제이션**: Monte Carlo Localization 기반, IMU와 오도메트리를 결합하여 30cm 이내 정확도 달성

- **어노테이션 통계**:
  - 클래스 불균형이 심함: car가 가장 많고 ambulance가 가장 적음 (비율 1:10K)
  - 프레임당 평균 보행자 7명, 차량 20명
  - 야간 비율 19.4%, 비 내리는 상황 11.6%

---

## Section 3: Tasks & Metrics

**요약**

nuScenes는 3D 탐지와 3D 추적 두 가지 주요 태스크를 정의한다. 기존의 KITTI IoU 기반 메트릭의 한계를 극복하는 새로운 메트릭 체계를 제안한다.

### 3.1 Detection (탐지)

**Average Precision (mAP)**

기존 IoU 대신 **2D 중심점 거리(center distance on ground plane)**를 매칭 기준으로 사용한다.

$$\text{mAP} = \frac{1}{|\mathbb{C}||\mathbb{D}|} \sum_{c \in \mathbb{C}} \sum_{d \in \mathbb{D}} \text{AP}_{c,d}$$

**수식 설명**
- **$\mathbb{C}$**: 모든 클래스의 집합 (예: car, pedestrian, bicycle 등 10개)
- **$\mathbb{D}$**: 매칭 거리 임계값 집합 $\{0.5, 1, 2, 4\}$ 미터
- **$\text{AP}_{c,d}$**: 클래스 $c$, 거리 임계값 $d$에서의 Average Precision
- 네 가지 거리 임계값에서 AP를 구한 뒤 평균 → 거리에 따른 성능을 종합적으로 평가

> **왜 IoU가 아닌 거리인가?** 보행자나 자전거처럼 바닥 면적이 작은 객체는 IoU가 작은 위치 오차에도 0이 돼버려 이미지 기반 방법이 불리하게 평가된다. 중심점 거리를 사용하면 이런 편향을 줄일 수 있다.

**True Positive Metrics (TP Metrics)**

탐지의 질을 다각도로 평가하는 5가지 TP 메트릭:

| 메트릭 | 약자 | 의미 |
|--------|------|------|
| Average Translation Error | ATE | 중심점 유클리드 거리 (미터) |
| Average Scale Error | ASE | 크기 IoU 오차 ($1 - \text{IOU}$) |
| Average Orientation Error | AOE | 최소 yaw 각도 차이 (라디안) |
| Average Velocity Error | AVE | 2D 속도 L2 오차 (m/s) |
| Average Attribute Error | AAE | 속성 분류 오차 ($1 - \text{acc}$) |

$$\text{mTP} = \frac{1}{|\mathbb{C}|} \sum_{c \in \mathbb{C}} \text{TP}_c$$

**수식 설명**
- 각 클래스별 TP 오차를 평균하여 전체 TP 성능 산출
- 매칭 거리는 $d = 2$m 고정 사용

**nuScenes Detection Score (NDS)**

mAP와 5개의 TP 메트릭을 하나의 스칼라로 통합:

$$\text{NDS} = \frac{1}{10} \left[ 5 \cdot \text{mAP} + \sum_{\text{mTP} \in \mathbb{TP}} (1 - \min(1, \text{mTP})) \right]$$

**수식 설명**
- **mAP**에 가중치 5를 주어 탐지 정확도를 중시
- 각 **mTP** 오차를 $1 - \text{오차}$ 형태로 변환 → 오차가 낮을수록 점수 높음
- $\min(1, \text{mTP})$로 최대 페널티를 1로 제한
- 최종 NDS 범위: [0, 1]

### 3.2 Tracking (추적)

**sAMOTA (scaled Average Multi Object Tracking Accuracy)**

$$\text{sMOTA}_r = \max\left(0, 1 - \frac{IDS_r + FP_r + FN_r - (1-r)P}{rP}\right)$$

$$\text{sAMOTA} = \frac{1}{|\mathcal{R}|} \sum_{r \in \mathcal{R}} \text{sMOTA}_r$$

**수식 설명**
- **$IDS_r$**: recall 임계값 $r$에서의 identity switch 수 (객체 ID가 바뀐 횟수)
- **$FP_r$**: False Positive (잘못 탐지한 수)
- **$FN_r$**: False Negative (놓친 객체 수)
- **$P$**: Ground truth 객체 총 수
- **$(1-r)P$**: recall 수준에 따른 보정 항 → 낮은 recall에서 FN 페널티를 완화
- $r \in \{0.1, 0.2, ..., 1.0\}$ 전 구간에 걸쳐 평균 → 특정 recall 설정에 종속되지 않음

**TID & LGD 메트릭**
- **TID (Track Initialization Duration)**: 객체가 처음 등장한 시점부터 추적이 시작될 때까지 걸린 시간 (초)
- **LGD (Longest Gap Duration)**: 추적 중 가장 긴 공백 구간 (초). 짧은 추적 단절이 긴 단절보다 용인 가능하다는 AV의 실용적 요구를 반영

---

## Section 4: Experiments

**요약**

탐지와 추적 베이스라인 결과를 제시하고 데이터셋 특성을 분석한다.

### 4.1 Baselines

**라이다 탐지 베이스라인 (PointPillars)**

- 시간적 정보 활용: 여러 라이다 스윕을 누적하여 포인트클라우드를 풍부하게 만듦
- 각 포인트에 키프레임으로부터의 시간 델타를 추가 특징으로 사용
- 10개 스윕 사용 시 가장 좋은 성능, 하지만 수확 체감(diminishing returns) 발생

**이미지 탐지 베이스라인 (OFT)**

- Orthographic Feature Transform을 6개 카메라 전체에 적용
- 6개 카메라 이미지를 NMS로 통합하여 360° 예측

**주요 발견**

1. **데이터 규모의 중요성**: nuScenes에서 더 많은 데이터를 사용할수록 PointPillars 성능이 지속적으로 향상 (Figure 6)

2. **매칭 함수의 중요성**: IoU 매칭 사용 시 보행자·자전거 AP가 0에 가까워짐 → 중심점 거리 매칭이 이미지 기반 방법에도 공정

3. **다중 라이다 스윕의 효과**

| 스윕 수 | mAP (%) | NDS |
|---------|---------|-----|
| 1 (KITTI 방식) | 30.5 | - |
| 5 | 45.3 | - |
| 10 | 47.8 | 0.59 |

4. **사전학습(Pre-training)**: 라이다 베이스라인은 ImageNet 사전학습이 KITTI와 동일한 패턴을 보임 (성능 향상)

5. **클래스별 어려움**: 자전거와 건설 차량이 가장 어려운 클래스 (형태 변이 크고 라이다 반사율 낮음)

**탐지 챌린지 결과 (Table 4)**

| 방법 | mAP | NDS | 비고 |
|------|-----|-----|------|
| Megvii | 52.8 | 63.3 | 라이다 기반, 1위 |
| PointPillars | 30.5 | 45.6 | 라이다 기반 베이스라인 |
| MonoDIS | 30.4 | 38.4 | 이미지 전용, 최고 성능 |
| OFT | 6% | - | 이미지 기반 베이스라인 |

**추적 챌린지 결과 (Table 8)**

| 방법 | sAMOTA (%) | AMOTP (m) |
|------|-----------|-----------|
| Stan (1위) | 55.0 | 0.80 |
| PointPillars + AB3DMOT | 2.9 | 1.70 |
| MonoDIS + AB3DMOT | 1.8 | 1.79 |

### 4.2 Analysis

**클래스 불균형 문제**: 자전거는 PointPillars에서 45.7% AP이지만 MonoDIS에서 1.1% AP로 극단적 차이 → 이미지 기반 방법이 작은 객체 탐지에 매우 불리

**시맨틱 맵 활용**: 탐지 박스가 시맨틱 맵 도로에 가까울수록 AP가 높아짐 → 맵 사전 정보가 탐지 성능을 향상시킬 수 있음

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **멀티모달 데이터셋** | 카메라, 라이다, 레이더를 동시에 포함하는 데이터셋. 단일 센서의 한계를 상호 보완 |
| **360° 커버리지** | 자율주행차 주변 전 방향을 빠짐없이 감지. 기존 KITTI는 전방만 커버 |
| **중심점 거리 매칭** | IoU 대신 3D 바운딩 박스의 2D 지면 투영 중심점 거리로 매칭. 소형 객체에 공정 |
| **NDS (nuScenes Detection Score)** | mAP + 5개 TP 메트릭을 하나의 점수로 통합. AV 실용 요구를 반영한 종합 지표 |
| **sAMOTA** | recall 임계값 전 구간에 걸친 MOTA 평균. 특정 신뢰도 임계값에 종속되지 않는 추적 지표 |
| **TID / LGD** | 추적 초기화 지연(TID)과 최대 추적 공백(LGD). AV에서 단기 추적 단절이 장기보다 허용 가능하다는 도메인 지식 반영 |
| **다중 라이다 스윕 누적** | 여러 시간 프레임의 포인트클라우드를 하나의 프레임에 합산. 포인트 밀도를 높여 성능 향상 |
| **시맨틱 HD 맵** | 도로, 인도, 교차로 등 11개 레이어의 정밀 지도. 탐지 사전 정보로 활용 가능 |
| **클래스 불균형** | car:ambulance = 10000:1 수준의 극심한 불균형. nuScenes 커뮤니티가 해결해야 할 도전 과제 |
| **도플러 속도** | 레이더가 전파의 주파수 변화로 직접 측정한 객체 속도. 카메라나 라이다로는 얻기 어려운 직접 속도 정보 |

---

## 결론 및 시사점

**논문의 결론**

nuScenes는 자율주행 연구를 위한 대규모 멀티모달 데이터셋을 제공한다. 주요 기여는:

1. **최초의 완전한 센서 슈트** 공개 데이터셋 (카메라 + 라이다 + 레이더)
2. **새로운 3D 탐지·추적 메트릭** 정의 (NDS, sAMOTA, TID, LGD)
3. **대규모 어노테이션**: KITTI 대비 7배 어노테이션, 100배 데이터
4. **레이더 기반 연구**의 길을 열어줌 (레이더 포함 공개 데이터셋 최초)

**실무적 시사점**

- **IoU 메트릭의 한계**: 소형 객체(보행자, 자전거)에서 IoU 기반 평가는 이미지 방법에 불공정. 중심점 거리가 더 적합
- **클래스 불균형 처리**: 중요도 샘플링(importance sampling)이 nuScenes에서 핵심임이 실험으로 확인됨
- **센서 융합의 실질적 가치**: 단일 센서 대비 멀티모달 접근이 어려운 조건(야간, 비)에서 강인성 제공
- **대규모 데이터의 중요성**: KITTI 수준의 데이터에서 최적처럼 보이는 알고리즘이 더 큰 nuScenes에서 순위가 바뀜 → 데이터 규모가 알고리즘 평가의 신뢰성에 직접 영향
- **레이더 활용 가능성**: 현재 베이스라인들은 레이더를 충분히 활용하지 못하고 있어, 레이더 기반 fusion 연구가 유망한 미래 방향
