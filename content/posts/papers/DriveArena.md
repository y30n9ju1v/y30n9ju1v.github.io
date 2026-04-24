---
title: "DriveArena: A Closed-loop Generative Simulation Platform for Autonomous Driving"
date: 2026-04-19T13:00:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving", "Generative Models"]
tags: ["Autonomous Driving", "Simulation", "Generative Model", "Closed-loop", "Diffusion"]
---

## 개요

- **저자**: Xuemeng Yang, Licheng Wen, Yukai Ma, Jianbiao Mei, Xin Li, Tiantian Wei, Wenjie Lei, Daocheng Fu, Pinlong Cai, Min Dou, Botian Shi, Liang He, Yong Liu, Yu Qiao
- **소속**: Shanghai Artificial Intelligence Laboratory, Zhejiang University, Shanghai Jiao Tong University, Technical University of Munich, East China Normal University
- **발행년도**: 2024 (arXiv:2408.00415)
- **주요 내용**: 자율주행 에이전트를 위한 최초의 고충실도 폐루프(closed-loop) 생성 시뮬레이션 플랫폼인 DriveArena를 제안. Traffic Manager와 World Dreamer 두 핵심 컴포넌트를 통해 실제 도로 이미지 기반의 반복적 시뮬레이션 루프를 구현한다.

## 한계 극복

이 논문은 기존 자율주행 평가 방식과 시뮬레이터의 핵심 한계를 극복하기 위해 작성되었습니다.

- **개방형 루프(Open-loop) 평가의 한계**: 기존 공개 데이터셋 기반 평가는 에이전트의 행동이 미래 데이터 분포에 영향을 주지 않아 실제 누적 오류를 반영하지 못함. 에이전트가 단순히 현재 상태를 유지하는 것만으로도 높은 성능처럼 보임
- **다른 차량의 반응 부재**: 기록된 데이터셋에서 주변 차량은 ego 차량의 행동에 반응하지 않아, 복잡한 상호작용 평가가 불가능
- **기존 시뮬레이터의 도메인 갭**: CARLA 같은 게임 엔진 기반 시뮬레이터는 실제 영상과의 외관 차이(domain gap)가 크고, NeRF/3DGS 기반 재구성 시뮬레이터는 사전 구축된 에셋에 의존하여 다양한 시나리오 생성이 어려움
- **이 논문의 접근 방식**: Traffic Manager(OpenStreetMap 기반 실시간 교통 시뮬레이션) + World Dreamer(조건부 확산 모델 기반 고충실도 이미지 생성)를 결합하여 폐루프 상호작용이 가능한 photo-realistic 시뮬레이터를 구축

## 목차

- Section 1: Introduction
- Section 2: DriveArena Framework
- Section 3: Methodology (Traffic Manager, World Dreamer, Driving Agent, Evaluation Metrics)
- Section 4: Experiments
- Section 5: Related Works
- Section 6: Conclusions and Future Works

---

## Section 1: Introduction

**요약**

자율주행 알고리즘은 모듈식 파이프라인부터 엔드-투-엔드 모델, 지식 기반 방법으로 빠르게 발전하고 있습니다. 하지만 공개 데이터셋의 개방형 루프 평가에서는 에이전트가 현재 상태를 유지하기만 해도 좋은 성능을 보이는 문제가 있습니다. 기존 시뮬레이터들은 sensor input을 생성하지 못하거나 실제 세계와 큰 도메인 차이를 보입니다. DriveArena는 이를 해결하기 위해 생성형 모델 기반의 고충실도 폐루프 시뮬레이터를 제안합니다.

**핵심 개념**

- **Open-loop 평가의 문제**: 에이전트의 결정이 다음 프레임에 영향을 주지 않으므로 실제 운전 상황의 연쇄 오류(compounding error)를 측정할 수 없음
- **Closed-loop 평가의 필요성**: 에이전트의 행동이 환경에 반영되고, 환경이 다시 에이전트에게 피드백을 주는 양방향 상호작용이 필요
- **DriveArena의 핵심 기여**:
  1. 최초의 고충실도 폐루프 시뮬레이터
  2. OpenStreetMap 기반으로 전 세계 도로 지원
  3. 모듈형 설계로 각 컴포넌트 교체 가능

---

## Section 2: DriveArena Framework

**요약**

DriveArena는 두 가지 핵심 컴포넌트로 구성됩니다: **Traffic Manager**(교통 흐름 시뮬레이터)와 **World Dreamer**(이미지 생성 모델). 이 두 컴포넌트와 Driving Agent가 네트워크 인터페이스를 통해 통신하며 폐루프 시뮬레이션을 실현합니다.

**폐루프 동작 순서**

1. World Dreamer가 현재 맵/차량 배치를 기반으로 surround-view 이미지를 생성
2. Driving Agent가 이미지를 보고 ego 궤적(trajectory)을 출력
3. Traffic Manager가 궤적을 받아 ego 차량 및 주변 차량 위치를 업데이트
4. 업데이트된 레이아웃이 World Dreamer에게 다시 전달 → 반복

**핵심 개념**

- **Traffic Manager**: OpenStreetMap에서 다운로드한 HD 맵을 처리하고, LimSim 기반으로 다중 차량 교통 흐름을 시뮬레이션. 충돌 감지 및 폐루프 평가를 담당
- **World Dreamer**: Stable Diffusion 기반의 조건부 생성 모델. 맵, 차량 배치, 텍스트 프롬프트, 참조 이미지를 조건으로 받아 photo-realistic한 서라운드뷰 이미지를 생성
- **Driving Agent**: 비전 기반 자율주행 알고리즘(예: UniAD). 생성된 이미지를 입력받아 궤적 계획을 출력. HTTP 프로토콜로 통신하여 실제 구현 방식에 무관하게 교체 가능
- **모듈형 설계**: 각 컴포넌트가 표준 네트워크 인터페이스로 통신하므로, World Dreamer나 Driving Agent를 다른 방법으로 대체 가능

---

## Section 3: Methodology

### 3.1 Traffic Manager

**요약**

LimSim을 기반으로 구현된 Traffic Manager는 계층적 다중 차량 의사결정 및 계획 프레임워크를 사용합니다. OpenStreetMap에서 임의 도시의 HD 맵을 다운로드하여 사용하며, 개방형 루프와 폐루프 두 가지 시뮬레이션 모드를 지원합니다.

**핵심 개념**

- **계층적 계획 프레임워크**: 교통 흐름 내 모든 차량의 의사결정을 공동으로 수행하고 고주파 계획 모듈로 동적 환경에 빠르게 반응
- **협력 인자(cooperation factor)**: 자율주행 차량의 교통 내 다양성을 사회적·개인적 수준에서 도입
- **OpenStreetMap 연동**: 인터넷에서 직접 다운로드한 도로 네트워크를 사용하므로 사전 구축된 에셋 불필요
- **두 가지 모드**:
  - **Open-loop 모드**: Traffic Manager가 ego 차량을 직접 제어하며 에이전트 궤적은 기록만 됨
  - **Closed-loop 모드**: 에이전트 궤적 출력으로 ego 차량을 직접 제어

### 3.2 World Dreamer

**요약**

World Dreamer는 Stable Diffusion v1.5를 기반으로 구축된 조건부 생성 모델입니다. 맵 레이아웃, 차량 배치, 텍스트 설명, 카메라 파라미터, 참조 이미지를 조건으로 받아 기하학적·문맥적으로 정확한 surround-view 이미지를 생성합니다. 특히 cross-view 일관성과 시간적 일관성을 위한 자기회귀(auto-regressive) 생성 방식을 채택합니다.

**핵심 개념**

- **조건 인코딩(Condition Encoding)**: 다양한 조건 정보를 UNet에 통합
  - 텍스트: CLIP 텍스트 인코더로 임베딩 $e_{text}$
  - 카메라 파라미터 및 3D 바운딩 박스: Fourier 임베딩으로 인코딩 → $e_{cam}$, $e_{box}$
  - BEV 맵 그리드: 맵과 박스 레이아웃을 카메라 뷰에 투영하여 레이아웃 캔버스 생성 → $e_{layout}$
  - 참조 이미지: 과거 L개 프레임 중 랜덤하게 선택, CLIP으로 인코딩 → $e_{ref}$ (외관·날씨 일관성 제공)

**레이아웃 특징 수식**

$$e_{layout} = \text{Encoder}(\text{LayoutCanvas}(e_{map}, e_{box}))$$

**수식 설명**:
- **$e_{layout}$**: 맵과 차량 박스 정보를 통합한 레이아웃 임베딩
- **$e_{map}$**: HD 맵의 차선 경계, 횡단보도 등 정보를 카메라 뷰에 투영한 맵 캔버스
- **$e_{box}$**: 3D 바운딩 박스를 카메라 뷰에 투영한 박스 캔버스
- **LayoutCanvas**: 맵 캔버스와 박스 캔버스를 합쳐 최종 레이아웃 캔버스를 생성
- **Encoder**: ControlNet 조건 인코더로 레이아웃 캔버스를 UNet에 주입할 임베딩으로 변환

- **Cross-view 일관성**: MagicDrive의 cross-view attention 모듈을 통합하여 여러 카메라 뷰 간 일관된 이미지 생성
- **자기회귀 생성(Auto-regressive Generation)**: 추론 시 이전 프레임의 생성 결과와 상대적 ego 포즈를 참조 조건으로 사용하여 시간적으로 연속적인 영상 생성. 이론적으로 무한 길이의 영상 생성 가능

### 3.3 Driving Agent

**요약**

DriveArena는 특정 Driving Agent 구현을 강제하지 않습니다. 표준 네트워크 API로 통신하므로 어떤 카메라 입력 기반 자율주행 알고리즘도 통합 가능합니다. 논문에서는 대표적인 엔드-투-엔드 자율주행 모델인 UniAD를 사용하여 실험을 진행하였습니다.

### 3.4 평가 지표

**요약**

DriveArena는 개방형 루프와 폐루프 평가를 모두 지원하며, NAVSIM에서 차용한 PDM Score와 자체 개발한 Arena Driving Score(ADS)를 사용합니다.

**PDM Score 수식**

$$\text{PDMS}_t = \left(\prod_{m \in \{NC, DAC\}} \text{score}_m\right) \times \frac{\sum_{w \in \{EP, TTC, C\}} \text{weight}_w \times \text{score}_w}{\sum_{w \in \{EP, TTC, C\}} \text{weight}_w}$$

**수식 설명**:
- **$\text{PDMS}_t$**: 타임스텝 $t$에서의 PDM Score (0~1, 높을수록 좋음)
- **NC (No Collision)**: 충돌 없음 페널티 — 다른 차량과의 충돌 여부
- **DAC (Drivable Area Compliance)**: 주행 가능 영역 준수 — 도로 이탈 여부
- **EP (Ego Progress)**: Ego 진행률 — 목표 경로 대비 실제 진행 거리
- **TTC (Time-to-Collision)**: 충돌까지의 시간 — 안전 마진 측정
- **C (Comfort)**: 승차감 — 급가속/급감속 측정
- **$\prod$**: NC와 DAC 중 하나라도 위반하면 전체 점수가 0이 됨 (페널티 구조)

$$\text{PDMS} = \frac{\sum_{t=0}^{T} \text{PDMS}_t}{T} \in [0, 1]$$

**Arena Driving Score 수식**

$$\text{ADS} = R_c \times \text{PDMS}$$

**수식 설명**:
- **$\text{ADS}$**: 폐루프 시뮬레이션의 최종 종합 점수
- **$R_c \in [0, 1]$**: 경로 완주율 — 에이전트가 완주한 경로의 비율. 폐루프에서는 충돌이나 도로 이탈 시 시뮬레이션이 종료되므로 경로 완주율이 중요한 지표가 됨
- **PDMS**: 궤적 품질 점수 (안전성, 진행률, 승차감 종합)
- **의미**: 아무리 PDMS가 높아도 경로를 완주하지 못하면 ADS는 낮아지므로, 폐루프 환경에서 실제 주행 능력을 종합적으로 평가

---

## Section 4: Experiments

**요약**

World Dreamer는 nuScenes 데이터셋(700개 학습, 150개 검증 씬)으로 학습하였으며, Stable Diffusion v1.5를 사전학습 가중치로 초기화합니다. 8개의 NVIDIA A100(80GB) GPU로 200K 이터레이션 학습을 수행합니다. Traffic Manager는 10Hz로 동작하며, World Dreamer는 2Hz(0.5초마다)로 surround-view 이미지를 생성합니다.

**핵심 실험 결과**

- **생성 충실도 비교 (Table 1)**: MagicDrive 대비 드라이브 에이전트(UniAD) 기반 3D 탐지, BEV 세분화, L2 거리, 충돌률 등 다수 지표에서 우수한 성능을 달성. 일부 지표에서는 실제 nuScenes 데이터를 능가
- **제로샷 일반화**: nuScenes만으로 학습했음에도 Pittsburgh, Las Vegas(nuPlan 데이터셋) 등 새로운 도시와 카메라 설정에서 일관된 이미지 생성
- **개방형 루프 평가 (Table 2)**: UniAD의 nuScenes 원본 PDMS는 0.910, World Dreamer 생성 이미지에서는 0.902(1% 미만 하락), DriveArena 오픈루프 시뮬레이션에서는 0.636으로 큰 폭 하락 → 미지의 도로·교통 흐름에서의 분포 외 시나리오가 얼마나 어려운지 보여줌
- **폐루프 평가 (Table 3)**: 4개 경로 평균 PDMS 0.667, Route Completion 13.7%, ADS 0.086. UniAD가 직선 구간에서는 양호하지만 교차로 회전에서 실패하는 경향을 발견

**지원 맵**: singapore-onenorth, boston-seaport, boston-thomaspark, carla-town05 및 OpenStreetMap의 모든 도시

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **Closed-loop Simulation** | 에이전트의 행동이 환경에 반영되고, 변화된 환경이 다시 에이전트에게 피드백되는 반복적 상호작용 시뮬레이션 |
| **World Dreamer** | Stable Diffusion 기반 조건부 생성 모델. 맵·차량 배치·텍스트·참조이미지를 조건으로 photo-realistic surround-view 이미지 생성 |
| **Traffic Manager** | LimSim 기반 교통 시뮬레이터. OpenStreetMap HD 맵으로 전 세계 도로 지원, 다중 차량 계층적 계획 수행 |
| **Auto-regressive Generation** | 이전 생성 프레임을 참조 조건으로 삼아 현재 프레임을 생성하는 방식. 이론적으로 무한 길이의 시간적 일관성 있는 영상 생성 가능 |
| **Cross-view Attention** | 여러 카메라 뷰 간 일관성을 유지하기 위한 어텐션 메커니즘 |
| **PDM Score (PDMS)** | 충돌 없음, 주행영역 준수, Ego 진행률, 충돌시간, 승차감을 종합한 궤적 품질 점수 |
| **ADS (Arena Driving Score)** | PDMS × 경로완주율. 폐루프 시뮬레이션에서 실제 주행 능력을 종합 평가하는 지표 |
| **ControlNet** | 조건 신호를 확산 모델에 주입하기 위한 훈련 가능한 어댑터 구조 |
| **Zero-shot 일반화** | 학습에 사용하지 않은 도시·카메라 설정에서도 일관된 이미지를 생성하는 능력 |

---

## 결론 및 시사점

DriveArena는 생성형 AI와 교통 시뮬레이션을 결합하여 자율주행 평가의 패러다임을 오픈루프에서 폐루프로 전환하는 중요한 이정표를 제시합니다.

**주요 기여**:
1. **최초의 고충실도 폐루프 생성 시뮬레이터**: 카메라 기반 자율주행 에이전트가 실제 세계와 유사한 환경에서 상호작용적으로 평가받을 수 있는 플랫폼 제공
2. **전 세계 도로 지원**: OpenStreetMap 연동으로 사전 구축된 HD 맵 에셋 없이도 임의 도시 시뮬레이션 가능
3. **모듈형 아키텍처**: 각 컴포넌트를 독립적으로 교체·개선 가능한 유연한 설계

**실무적 시사점**:
- 합성 데이터 기반 시뮬레이션이 실제 주행과 유사한 분포를 제공할 수 있음을 검증
- 오픈루프에서 좋은 성능(PDMS 0.91)을 보이는 UniAD가 폐루프에서는 경로 완주율 13.7%에 불과 → **오픈루프 평가가 실제 성능을 과대평가**하고 있음을 시사
- 자율주행 회귀 테스트 및 코너 케이스 평가 플랫폼으로 활용 가능

**한계 및 향후 과제**:
1. 데이터 다양성: 현재 nuScenes만으로 학습, 더 다양한 데이터셋 포함 필요
2. 시간적 일관성: 장시간 일관성 유지가 여전히 과제
3. 런타임 효율: 실시간 시뮬레이션을 위한 빠른 샘플링 방법 연구 필요
4. 더 많은 에이전트 테스트 및 생성 모델 평가 플랫폼("A Real Arena")으로 확장 계획
