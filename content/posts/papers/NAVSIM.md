---
title: "NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking"
date: 2026-04-14T00:00:00+09:00
draft: false
categories: ["Papers"]
tags: ["Autonomous Driving", "Benchmark", "Simulation", "End-to-End Planning", "BEV", "nuPlan"]
---

## 개요

- **저자**: Daniel Dauner, Marcel Hallgarten, Tianyu Li, Xinshuo Weng, Zhiyu Huang, Zetong Yang, Hongyang Li, Igor Gilitschenski, Boris Ivanovic, Marco Pavone, Andreas Geiger, Kashyap Chitta
- **소속**: University of Tübingen, Tubingen AI Center, OpenDriveLab at Shanghai AI Lab, NVIDIA Research, Robert Bosch GmbH, Nanyang Technological University, University of Toronto, Vector Institute, Stanford University
- **발행년도**: 2024 (arXiv:2406.15349, NeurIPS 2024 Datasets and Benchmarks Track)
- **주요 내용**: 오픈루프(open-loop)와 클로즈드루프(closed-loop) 평가의 장점을 결합한 **비반응형(non-reactive) 시뮬레이션 벤치마크** NAVSIM을 제안. 대규모 실제 데이터를 활용하면서 충돌·도로 준수 등 시뮬레이션 기반 지표를 계산할 수 있어, 기존 변위 오차(ADE) 지표의 한계를 극복. CVPR 2024 챌린지에서 143팀 463건 제출이 이루어졌으며, 간단한 TransFuser가 UniAD 등 대형 모델과 대등한 성능을 보이는 놀라운 결과를 도출.

## 목차

- Chapter 1: Introduction — 기존 평가 방식의 한계와 NAVSIM의 등장 배경
- Chapter 2: Related Work — 오픈루프/클로즈드루프 벤치마크 및 시뮬레이터 동향
- Chapter 3: NAVSIM — 비반응형 시뮬레이션 설계, PDMS 지표, 데이터 필터링
- Chapter 4: Experiments — 오픈/클로즈드루프 정렬 분석, 최신 방법 벤치마크
- Chapter 5: Discussion — 한계와 향후 과제

---

## Chapter 1: Introduction

**요약**

자율주행 알고리즘 평가에는 크게 두 가지 방식이 있다:

1. **오픈루프(Open-loop)**: 실제 데이터로 쉽게 평가할 수 있지만, 인간 주행 궤적과의 변위 오차(ADE)를 측정하기 때문에 **안전하지만 인간과 다른 궤적도 나쁘게 평가**한다.
2. **클로즈드루프(Closed-loop)**: 시뮬레이터 안에서 실제 주행처럼 평가할 수 있지만, **계산 비용이 크고 시뮬레이터와 실제 세계 사이의 도메인 갭**이 크다.

Figure 1이 보여주듯, ADE가 낮아도 위험한 주행일 수 있고(충돌), ADE가 높아도 안전한 주행일 수 있다(도로 주행, 무충돌). 이는 ADE 지표가 실제 주행 품질을 반영하지 못함을 의미한다.

또한 nuScenes 같은 기존 데이터셋은 계획(planning) 연구보다 인식(perception) 연구를 위해 만들어졌기 때문에, 약 75%의 장면이 자차 이동 경로를 단순 외삽하는 "blind" 정책으로도 쉽게 풀린다는 문제가 있다.

NAVSIM은 이 두 가지 문제를 해결하기 위해 **비반응형 시뮬레이션**이라는 중간 지점을 제안한다: 대규모 실제 데이터를 사용하면서, 에이전트와 환경이 서로 영향을 주지 않는 짧은 시간 구간(4초)에서 충돌·도로 준수 등의 시뮬레이션 지표를 계산한다.

**핵심 개념**

- **Open-loop evaluation**: 고정된 실제 데이터로 예측 궤적을 평가. 빠르고 확장 가능하지만 ADE 같은 지표가 주행 품질을 잘못 반영
- **Closed-loop evaluation**: 시뮬레이터에서 에이전트가 환경과 상호작용하며 평가. 현실적이지만 도메인 갭과 계산 비용 문제
- **Non-reactive simulation**: 에이전트 행동이 다른 에이전트나 환경에 영향을 미치지 않는다고 가정. 단기(4초) 지평선에서 타당한 근사이며 대규모 실제 데이터 활용 가능
- **Label leakage**: nuScenes에서 인간 궤적을 이산 방향 명령으로 변환해 입력으로 사용함으로써 정답 정보가 새어 들어가는 문제

---

## Chapter 2: Related Work

**요약**

**End-to-End Driving**: CARLA 기반 클로즈드루프 방법(TransFuser, NEAT 등)과 nuScenes 기반 오픈루프 방법(UniAD, VAD 등)이 별도로 발전해 왔으며, 두 접근법의 공정한 비교가 어려웠다. NAVSIM은 처음으로 두 계열을 동일한 평가 환경에서 비교한다.

**클로즈드루프 시뮬레이터**: CARLA, nuPlan, Waymax 등이 있으나, 그래픽 렌더링 기반 센서 시뮬레이션은 실제 데이터와 도메인 갭이 크다. NAVSIM은 실제 센서 데이터를 그대로 사용하여 이 문제를 우회한다.

**오픈루프 지표 문제**: ADE/FDE는 클로즈드루프 성능과 상관관계가 낮다는 연구들이 축적되었다. NAVSIM의 PDMS 지표는 이보다 높은 클로즈드루프 상관관계를 달성한다.

---

## Chapter 3: NAVSIM — 비반응형 자율주행 시뮬레이션

**요약**

### 3.1 태스크 정의

에이전트는 타임스텝 $t$에서 **$h$초 동안의 미래 궤적**(미래 포즈의 시퀀스)을 계획해야 한다. 입력은 카메라, LiDAR 등 과거 센서 스트림과 자차 속도·가속도·방향 목표(ego status)이다. NAVSIM에서는 $h = 4$초를 사용한다.

**비반응형 가정**: 에이전트가 계획한 궤적은 다른 에이전트나 환경의 미래 행동에 영향을 주지 않는다. 이를 통해 다른 에이전트의 미래 상태를 실제 데이터에서 직접 가져올 수 있다. 단기(4초) 지평선에서는 이 가정이 합리적이다.

실제 궤적 실행은 LQR 컨트롤러로 시뮬레이션하여 4초 지평선에 걸쳐 조향과 가속을 계산한다.

### 3.2 PDM Score (PDMS)

NAVSIM의 핵심 평가 지표로, Predictive Driver Model(PDM) 기반 규칙 기반 플래너의 클로즈드루프 점수 체계를 차용한다. 두 단계로 계산된다:

$$\text{PDMS} = \underbrace{\left(\prod_{m \in \{\text{NC, DAC}\}} \text{score}_m\right)}_{\text{penalties}} \times \underbrace{\left(\frac{\sum_{w \in \{\text{EP, TTC, C}\}} \text{weight}_w \times \text{score}_w}{\sum_{w \in \{\text{EP, TTC, C}\}} \text{weight}_w}\right)}_{\text{weighted average}}$$

**수식 설명**:
- 전체 점수는 **패널티 곱** × **가중 평균**의 구조
- **패널티 곱**: 충돌이나 도로 이탈 같은 허용 불가 행동이 발생하면 전체 점수를 0으로 만드는 역할
- **가중 평균**: 안전 범위 내에서의 진행도, 편안함 등을 종합

**서브스코어 상세**:

| 서브스코어 | 의미 | 비중 |
|-----------|------|------|
| **NC (No Collision)** | 무충돌 여부. 충돌 시 패널티로 점수 < 1 | 패널티 |
| **DAC (Drivable Area Compliance)** | 주행 가능 영역 준수. 이탈 시 패널티 | 패널티 |
| **EP (Ego Progress)** | 자차가 PDM-Closed 플래너 상한 대비 경로를 얼마나 진행했는지 | weight=5 |
| **TTC (Time-to-Collision)** | 다른 차량과의 충돌 예상 시간. 안전 마진 유지 여부 | weight=5 |
| **C (Comfort)** | 가속도와 저크(jerk)가 사전 정의된 임계값 이내인지 | weight=2 |

- 정적 장애물과의 충돌에는 소프트 패널티($\text{score}_\text{NC} = 0.5$) 적용
- 점수는 프레임별로 계산 후 평균

### 3.3 도전적 장면 필터링

대부분의 실제 주행 데이터는 직진이나 일정 속도 유지 같은 단순 상황으로 구성되어 있어, 이를 그대로 쓰면 단순한 "constant velocity" 정책이 약 91%의 PDMS를 달성한다.

NAVSIM은 다음 기준으로 단순 장면을 제거한다:
1. **constant velocity 에이전트**가 PDMS 0.8 이상 달성하는 장면 제거
2. 심각한 어노테이션 오류가 있는 장면 제거

필터링 후 constant velocity의 PDMS는 **22%로 하락**하고, 인간 전문가는 **95%**를 달성하여 벤치마크의 난이도가 크게 높아진다. 최종적으로 103k 학습(navtrain)과 12k 테스트(navtest) 샘플을 구성한다.

**데이터셋**: OpenScene(nuPlan 재배포, 2TB → 450GB로 압축)을 기반으로 하며, 8대 카메라(1920×1080), 5개 LiDAR 병합 포인트 클라우드, HD 맵을 포함한다.

---

## Chapter 4: Experiments

**요약**

### 4.1 오픈루프-클로즈드루프 정렬 분석

37개 규칙 기반 플래너와 15개 학습 기반 플래너로 오픈루프 지표(OLS, PDMS)와 클로즈드루프 지표(CLS)의 상관관계를 측정했다.

**핵심 발견**:
- **PDMS가 OLS(변위 오차 기반)보다 CLS와 일관되게 더 높은 상관관계**를 보임 (Spearman rank & Pearson linear 모두)
- 시뮬레이션 지평선을 $d = 15$초에서 $d = 4$초로 줄이면 PDMS-CLS 상관관계가 더 높아짐 (단기 비반응형 가정이 더 정확)
- 계획 주파수를 10Hz에서 2Hz로 낮추면 상관관계 상승 (낮은 주파수에서 누적 오차 감소)

### 4.2 최신 방법 벤치마크 결과

| Method | 입력 | NC↑ | DAC↑ | TTC↑ | Comfort↑ | EP↑ | PDMS↑ |
|--------|------|-----|------|------|----------|-----|-------|
| Constant Velocity | - | 68.0 | 57.8 | 50.0 | 100 | 19.4 | 20.6 |
| Ego Status MLP | Ego | 93.0 | 77.3 | 83.6 | 100 | 62.8 | 65.6 |
| LTF | Ego+Img | 97.4 | **92.8** | 92.1 | 100 | 79.0 | 83.8 |
| TransFuser | Ego+Img+LiDAR | 97.7 | **92.8** | 92.4 | 100 | 79.2 | **84.0** |
| UniAD | Ego+Img | 97.8 | 91.9 | 92.0 | 100 | 78.8 | 83.1 |
| PARA-Drive | Ego+Img | **97.9** | 92.4 | **93.0** | 100 | **79.3** | **84.0** |
| Human | - | 100 | 100 | 99.9 | 100 | 87.5 | 94.8 |

**핵심 발견**:
- **Ego Status만 보는 MLP가 65.6 PDMS** → ego status(속도, 가속도, 방향 목표)가 충돌 회피에 매우 중요
- **TransFuser(카메라+LiDAR, 1 GPU 1일 학습)가 UniAD(카메라, 80 GPU 3일 학습)와 대등** → nuScenes에서의 UniAD 우위가 NAVSIM에서는 재현되지 않음
- **인간과 최고 모델 사이 10 PDMS 격차** → DAC(주행 가능 영역 준수)와 EP(진행도)에서 주로 차이 발생

### 4.3 TransFuser 소거 실험

| 설정 | PDMS↑ |
|------|-------|
| Default (Ego+Img+LiDAR, 140° FOV, 3cam) | 84.0 |
| Goal only (방향 목표만) | 77.3 |
| 1 카메라 (60° FOV) | 80.3 |
| 3 카메라 (160° FOV) | 82.8 |
| LiDAR 제거 (LTF) | 83.8 |
| BEV segmentation 없음 | 81.6 |

- LiDAR를 제거해도 성능이 크게 떨어지지 않음 (카메라만으로도 충분)
- FOV 확장(카메라 수 증가)이 단순히 성능을 올리지 않음 → LiDAR 좌표계 기반 보조 태스크와의 불일치 가능성

### 4.4 CVPR 2024 NAVSIM 챌린지

- 143팀 463건 제출
- 우승팀: 궤적 샘플링 및 스코어링 방식(VADv2 영감), VLM과 인간 모방을 가중 합산
- TransFuser 베이스라인을 챌린지 마감 전까지 넘지 못한 팀이 다수 → 벤치마크의 난이도와 엔지니어링 요구사항 확인

**NAVSIM 1.1 Leaderboard** (navtest):

| Method | PDMS↑ |
|--------|-------|
| TransFuser | 83.9 ± 0.4 |
| LTF | 83.5 ± 0.6 |
| Ego Status MLP | 66.4 ± 0.9 |
| Hydra-MDP | 91.3 |
| Constant Velocity | 20.6 |

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **Non-reactive simulation** | 에이전트의 행동이 환경에 영향을 주지 않는다고 가정. 실제 데이터를 대규모로 활용하면서 시뮬레이션 지표 계산 가능 |
| **PDMS (PDM Score)** | 충돌·도로 이탈을 패널티로, 진행도·TTC·편안함을 가중 평균으로 종합한 시뮬레이션 기반 점수 |
| **OLS (Open-Loop Score)** | 변위 오차(ADE)와 방향 오차를 결합한 오픈루프 지표. CLS와의 상관관계가 낮음 |
| **CLS (Closed-Loop Score)** | nuPlan 클로즈드루프 시뮬레이터의 공식 점수. 계산 비용이 크지만 현실적 평가 가능 |
| **Label leakage** | nuScenes에서 인간 궤적에서 추출한 방향 명령을 입력으로 주면 정답이 새어 들어가 성능이 과대평가됨 |
| **Blind driving policy** | 센서 입력 없이 ego status(속도, 가속도 등)만 보거나 단순 외삽으로 주행하는 정책. 기존 벤치마크에서 의외로 높은 성능 |
| **Challenging scenario filtering** | constant velocity로 쉽게 풀리는 단순 장면을 제거하여 벤치마크 난이도를 높이는 전처리 |
| **Ego progress (EP)** | PDM-Closed 플래너의 이론적 상한 대비 실제 진행한 경로 비율. 도로를 잘 따라 진행했는지 측정 |
| **TTC (Time-to-Collision)** | 현재 속도와 방향을 유지할 때 다른 차량과 충돌까지 걸리는 예상 시간 |

---

## 결론 및 시사점

NAVSIM은 자율주행 평가의 두 가지 핵심 문제를 동시에 해결하는 실용적인 프레임워크다:

1. **ADE의 한계 극복**: 시뮬레이션 기반 PDMS는 클로즈드루프 성능과 훨씬 높은 상관관계를 보이며, 안전하지만 인간과 다른 궤적을 공정하게 평가
2. **확장성 확보**: 비반응형 가정으로 대규모 실제 데이터(100k+ 장면) 활용 가능, 그래픽 렌더링 없이 실제 센서 데이터 직접 사용

**놀라운 발견**:
- nuScenes에서 강력했던 UniAD가 NAVSIM에서 TransFuser와 대등 → 기존 벤치마크의 label leakage 문제가 성능을 과대평가했을 가능성
- Ego status(속도, 가속도, 방향 목표)만으로도 65.6 PDMS → 단순한 운동학적 정보가 충돌 회피에 매우 중요
- 인간과 최고 모델 사이의 10 PDMS 격차 → 특히 복잡한 교차로·합류 상황에서 DAC와 EP 개선 필요

**한계**:
- 비반응형 가정으로 인해 장기(15초 이상) 계획이나 복잡한 상호작용 평가 불가
- 후방 충돌이 "at-fault"로 분류되지 않아 뒤에서 오는 차량에 대한 평가 미흡
- 교통 법규(신호등, 정지선 등)를 PDMS가 직접 다루지 않음

**실무적 시사점**: NAVSIM은 HuggingFace를 통한 공개 평가 서버를 제공하여 재현성과 공정한 비교를 보장한다. 자율주행 연구자들이 CARLA와 nuScenes 사이의 갭을 메우는 새로운 표준 벤치마크로 자리잡을 잠재력이 있다.
