---
title: "GaussianWorld: Gaussian World Model for Streaming 3D Occupancy Prediction"
date: 2026-04-24T13:00:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving", "3D Occupancy"]
tags: ["3D Gaussian Splatting", "Occupancy Prediction", "World Model", "Temporal Modeling", "Streaming", "nuScenes"]
---

## 개요

- **저자**: Sicheng Zuo, Wenzhao Zheng, Yuanhui Huang, Jie Zhou, Jiwen Lu (Tsinghua University)
- **발행년도**: 2024
- **arXiv**: 2412.10373
- **주요 내용**: 3D Gaussian을 장면 표현으로 채택한 World Model 기반 프레임워크. 3D occupancy 예측을 현재 센서 입력에 조건화된 4D occupancy 예측 문제로 재정의하고, 추가 연산 없이 단일 프레임 대비 mIoU 2% 이상 향상.

## 한계 극복

- **기존 한계 1 — 단순 멀티 프레임 융합의 물리적 비일관성**: BEVFormer 등 기존 시간적 퍼셉션 방법들은 각 프레임의 BEV/볼륨 피처를 독립적으로 추출한 뒤 정합·융합하는 방식을 사용. 이는 인접 프레임의 장면 표현이 자차 이동과 동적 객체 움직임으로부터 직접 진화한다는 자연스러운 연속성(continuity)과 단순성(simplicity)을 무시함.
- **기존 한계 2 — 추가 연산 비용**: 시간적 융합을 위해 추가 인코딩·정합·융합 모듈이 필요하여 단일 프레임 대비 레이턴시·메모리가 크게 증가.
- **기존 한계 3 — 암묵적 동적 객체 모델링**: StreamPETR 같은 객체 쿼리 기반 방법은 동적 객체의 움직임을 암묵적으로만 표현하여 밀집 occupancy 예측에 부적합.
- **이 논문의 접근 방식**: 3D Gaussian으로 장면을 명시적으로 표현하고, 장면 진화를 세 요소(자차 정합·동적 객체 이동·신규 영역 완성)로 분해하여 World Model이 이를 학습. 단일 프레임 입력만으로 스트리밍 방식의 4D occupancy 예측 수행.

## 목차

- Section 1: Introduction
- Section 2: Related Work
- Section 3: Proposed Approach
  - 3.1 World Models for Perception
  - 3.2 Explicit Scene Evolution Modeling
  - 3.3 3D Gaussian World Model
- Section 4: Experiments
- Section 5: Conclusion

---

## Section 1 & 2: Introduction & Related Work

**요약**

3D semantic occupancy prediction은 장면 내 모든 복셀의 점유 여부와 시맨틱 레이블을 예측하는 과제로, 자율주행의 안전한 경로 계획에 필수적입니다. 기존의 시간적 퍼셉션 방법들은 과거 프레임의 표현을 현재로 정합·융합하는 방식을 쓰지만, 이는 드라이빙 씬이 연속적으로 진화한다는 강한 사전 지식(prior)을 활용하지 못합니다.

저자들은 **World Model 기반 패러다임**을 도입하여 장면 표현이 어떻게 진화하는지를 명시적으로 학습합니다. 3D Gaussian을 장면 표현으로 채택함으로써, BEV/Voxel 같은 암묵적 표현과 달리 객체의 위치·움직임을 연속적이고 명시적으로 모델링할 수 있습니다.

**핵심 개념**

- **World Model**: 미래 상태를 예측하는 생성 모델. 자율주행에서는 "세상이 어떻게 작동하는지"를 학습하여 센서 입력 없이 다음 상태를 예측하는 데 활용.
- **Streaming Prediction**: 매 프레임마다 이전 프레임의 예측 결과를 조건으로 현재 상태를 갱신하는 온라인 방식. 고정된 시간 윈도우를 재처리하지 않아 효율적.
- **4D Occupancy Forecasting**: 3D 공간 + 시간 축을 포함한 점유 예측. 현재 프레임의 센서 입력에 조건화되어 시공간적 장면을 예측하는 문제로 재정의.

---

## Section 3: Proposed Approach

### 3.1 World Models for Perception

**기존 시간적 퍼셉션 파이프라인 수식**

$$\mathbf{z}^n = P_{er}(\mathbf{x}^n),\quad \mathbf{a}^n = T_{trans}(\mathbf{z}^n, \mathbf{p}^n),\quad \mathbf{y}^T = F_{use}(\mathbf{a}^T, \ldots, \mathbf{a}^{T-t})$$

**수식 설명**
- **$\mathbf{z}^n$**: n번째 프레임의 장면 표현 (BEV 피처 또는 볼륨 피처)
- **$P_{er}$**: 각 프레임을 독립적으로 인코딩하는 퍼셉션 모듈
- **$\mathbf{a}^n$**: ego 궤적 $\mathbf{p}^n$을 이용해 현재 프레임 좌표계로 정합된 표현
- **$T_{trans}$**: ego 자세 기반 좌표 변환 모듈
- **$F_{use}$**: 과거 $t$개 프레임의 정합된 표현을 융합하는 모듈
- **직관**: 각 프레임을 따로 처리한 뒤 뭉치는 방식 → 인접 프레임 간 연속성 사전 지식을 무시.

**GaussianWorld의 World Model 수식**

$$\mathbf{z}^T = \mathbf{w}(\mathbf{z}^{T-1}, \mathbf{x}^T)$$

$$\mathbf{y}^T = \mathbf{h}(\mathbf{w}(\mathbf{z}^{T-1}, \mathbf{x}^T))$$

**수식 설명**
- **$\mathbf{z}^{T-1}$**: 이전 프레임에서 예측된 3D Gaussian 집합
- **$\mathbf{x}^T$**: 현재 프레임의 RGB 카메라 입력
- **$\mathbf{w}$**: World Model — 이전 장면 표현과 현재 관측을 받아 현재 장면 표현을 예측
- **$\mathbf{h}$**: 퍼셉션 헤드 — 3D Gaussian에서 occupancy로 변환
- **직관**: 이전 Gaussian이 어떻게 이동·변형되는지를 학습하여 현재 장면을 예측. 추가 인코딩 없이 단 하나의 과거 프레임만 사용.

### 3.2 Explicit Scene Evolution Modeling

**요약**

드라이빙 씬의 진화는 대부분 단순하고 연속적입니다. GaussianWorld는 이를 세 가지 독립적 요소로 분해하여 각각을 명시적으로 모델링합니다.

**3D Gaussian 표현**

$$\mathbf{g} = \{\mathbf{p}, \mathbf{s}, \mathbf{r}, \mathbf{c}, \mathbf{f}\}$$

**수식 설명**
- **$\mathbf{p}$**: 3D 위치 (position)
- **$\mathbf{s}$**: 크기 (scale)
- **$\mathbf{r}$**: 회전 (rotation)
- **$\mathbf{c}$**: 시맨틱 확률 (semantic probability) — 각 클래스에 속할 확률
- **$\mathbf{f}$**: 시간적 피처 (temporal feature) — Gaussian의 역사적 정보를 담는 추가 속성

**세 가지 장면 진화 요소**

**① 자차 이동 정합 (Ego Motion Alignment)**

$$\mathbf{g}_A^T = A_{lign}(\mathbf{g}^{T-1}, \mathbf{M}_{ego}) = R_{ef}(\mathbf{g}^{T-1}; \mathbf{M}_{ego} \cdot A_{ttr}(\mathbf{g}^{T-1}; \mathbf{p}); \mathbf{p})$$

- 이전 프레임의 3D Gaussian 전체에 ego 궤적 기반 전역 아핀 변환을 적용
- **$\mathbf{M}_{ego}$**: 이전 프레임에서 현재 프레임으로의 ego 이동 변환 행렬
- **직관**: 자동차가 앞으로 이동하면 정적 건물들이 뒤로 밀리는 것처럼, 모든 Gaussian을 새 자차 좌표계로 평행이동·회전

**② 동적 객체 이동 (Local Movements of Dynamic Objects)**

$$\mathbf{g}_M^T = M_{ove}(\mathbf{g}_A^T, \mathbf{x}_T) = R_{ef}(\mathbf{g}_A^T; E_{nc}(\mathbf{g}_A^T, \mathbf{x}_T) \cdot I(\mathbf{g}_A^T \in \{\mathbf{g}_D\}); \mathbf{p})$$

- 정합된 Gaussian을 동적($\{g_D\}$)과 정적($\{g_S\}$) 집합으로 분리
- 동적 Gaussian의 시맨틱 확률을 소프트 가중치로 사용하여 위치만 업데이트
- **$I(\cdot)$**: 동적 Gaussian 여부를 나타내는 지시 함수
- **직관**: 주변 차량·보행자의 실제 이동을 RGB 관측으로부터 학습하여 반영

**③ 신규 영역 완성 (Completion of Newly-Observed Areas)**

$$\mathbf{g}_C^T = P_{er}(\mathbf{g}_I^T, \mathbf{x}_T) = R_{ef}(\mathbf{g}_I^T; E_{nc}(\mathbf{g}_I^T, \mathbf{x}_T); \{\mathbf{p}, \mathbf{s}, \mathbf{r}, \mathbf{c}, \mathbf{f}\})$$

- ego가 전진하면 일부 Gaussian은 인식 범위 밖으로 나가고, 새 영역이 보이게 됨
- 새 영역에 랜덤 초기화 Gaussian $\mathbf{g}_I^T$를 배치하고 퍼셉션 레이어로 모든 속성 예측
- **직관**: "처음 보는 교차로"를 현재 카메라 이미지로부터 채워 넣는 과정

### 3.3 3D Gaussian World Model

**요약**

세 요소를 모델링하는 통합 프레임워크입니다. Motion Layer와 Perception Layer가 같은 아키텍처를 공유하여 계산 효율을 높입니다.

**통합 Evolution Layer**

$$\mathbf{g}_{l+1}^T = E_{vol}(\mathbf{g}_l^T, \mathbf{x}_T) = \begin{cases} P_{er}(\mathbf{g}_l^T, \mathbf{x}_T) & \text{if new} \\ M_{ove}(\mathbf{g}_l^T, \mathbf{x}_T) & \text{otherwise} \end{cases}$$

- 새 Gaussian이면 Perception 모드(모든 속성 예측), 역사 Gaussian이면 Motion 모드(위치만 업데이트)
- $n_e$개 evolution layer를 스택하여 반복 정제

**Unified Refinement Block**

Evolution Layer 이후 추가 $n_r$개 정제 레이어를 사용:

$$\mathbf{g}_{n+1}^T = R_{efine}(\mathbf{g}_n^T, \mathbf{x}_T) = R_{ef}(\mathbf{g}_n^T; E_{nc}(\mathbf{g}_n^T, \mathbf{x}_T); \{\mathbf{p}, \mathbf{s}, \mathbf{r}, \mathbf{c}, \mathbf{f}\})$$

- Evolution Layer와 달리 모든 Gaussian의 모든 속성을 업데이트
- 3D Gaussian 표현과 실제 세계 간 정렬 오차를 보정

**핵심 모듈 구성**

| 모듈 | 역할 |
|------|------|
| Self-Encoding | 3D Gaussian을 복셀화 후 3D sparse convolution으로 Gaussian 간 상호작용 학습 |
| Cross-Attention | Deformable Attention으로 3D Gaussian과 멀티스케일 이미지 피처 간 상호작용 |
| Unified Refinement Block | Motion / Perception 모드를 통합한 Gaussian 속성 예측 블록 |
| GS-to-Occ | 정제된 3D Gaussian에서 occupancy 복셀 그리드로 변환 |

**스트리밍 학습 전략**

초기에 짧은 시퀀스로 시작하여 점진적으로 길이를 늘리고, 일정 확률 $p$로 이전 프레임의 3D Gaussian 표현을 랜덤 폐기하여 긴 시퀀스에 적응:

- 시퀀스 길이: [5, 10, 20, 30, 38] 단계적 증가
- 확률적 모델링(Probabilistic Modeling): 다양한 길이의 시퀀스를 처리할 수 있도록 훈련 안정성 향상
- **직관**: 처음에는 "짧은 기억"으로 학습하다가 점점 "긴 기억"으로 확장

---

## Section 4: Experiments

**요약**

nuScenes 데이터셋에서 기존 SOTA 방법들과 비교합니다. GaussianWorld는 추가 연산 없이 단일 프레임 베이스라인을 크게 상회합니다.

### 4.1 주요 결과 (nuScenes validation)

| Method | mIoU | IoU |
|--------|------|-----|
| MonoScene | 7.31 | 23.96 |
| BEVFormer | 26.88 | 30.50 |
| TPVFormer | 11.66 | 11.51 |
| SurroundOcc | 20.30 | 31.49 |
| OccFormer | 19.03 | 31.39 |
| GaussianFormer-B (단일 프레임) | 19.10 | 29.83 |
| GaussianFormer-T (시간 융합) | 20.42 | 31.34 |
| **GaussianWorld (ours)** | **22.13** | **33.40** |

GaussianWorld는 단일 프레임 GaussianFormer-B 대비 **mIoU +2.4%, IoU +2.7%** 향상, 시간 융합 GaussianFormer-T 대비 **mIoU +1.7%, IoU +2.0%** 향상.

### 4.2 시간적 모델링 방법 비교 (효율성)

| 방법 | 역사 프레임 수 | 레이턴시 | 메모리 | mIoU |
|------|-------------|---------|-------|------|
| Single-Frame | 0 | 225 ms | 6958 M | 19.73 |
| 3D Gaussian Fusion | 3 | 379 ms | 9993 M | 20.24 |
| Perspective View Fusion | 3 | 382 ms | 10019 M | 20.42 |
| **GaussianWorld** | **1** | **228 ms** | **7030 M** | **21.87** |

GaussianWorld는 역사 프레임 1개만 사용하면서 단일 프레임과 거의 같은 레이턴시·메모리로 시간 융합 방법들을 모두 상회.

### 4.3 Ablation: 장면 진화 세 요소의 기여

| Ego 정합 | 동적 이동 | 신규 완성 | mIoU | IoU |
|---------|---------|---------|------|-----|
| ✗ | ✓ | ✓ | 18.47 | 28.88 |
| ✓ | ✗ | ✓ | 21.17 | 32.49 |
| ✓ | ✓ | ✗ | 학습 붕괴 | 학습 붕괴 |
| ✓ | ✓ | ✓ | **21.50** | **32.72** |

- **신규 영역 완성이 없으면 학습 자체가 붕괴**: ego가 계속 전진하면 결국 모든 Gaussian이 인식 범위 밖으로 나가 장면 표현이 소실됨
- **ego 정합**이 가장 큰 기여 (3.0% mIoU), **동적 이동** 모델링도 유의미한 향상

### 4.4 스트리밍 시퀀스 길이와 성능

스트리밍 프레임 수가 늘어날수록 성능이 향상되나, 약 20 프레임 이후 소폭 하락:
- **향상 이유**: 더 많은 역사 프레임으로 장면 진화를 더 잘 모델링
- **하락 이유**: 기존 3D occupancy GT가 멀티 프레임 LiDAR 누적으로 생성되어 가장자리가 희소하므로 긴 시퀀스에서 주석 품질 한계에 봉착

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **GaussianWorld** | 3D Gaussian World Model 기반 스트리밍 3D occupancy 예측 프레임워크 |
| **장면 진화 3요소** | ①자차 이동 정합(정적 배경), ②동적 객체 이동, ③신규 영역 완성 |
| **Streaming Prediction** | 이전 프레임의 3D Gaussian을 조건으로 현재 장면을 예측하는 온라인 방식 |
| **Evolution Layer** | Motion / Perception 모드를 통합하여 역사·신규 Gaussian을 동시에 처리 |
| **Unified Refinement Block** | Gaussian 표현과 실제 세계 간 정렬 오차를 보정하는 정제 블록 |
| **확률적 시퀀스 모델링** | 훈련 중 다양한 길이의 시퀀스를 처리하여 스트리밍 안정성 확보 |
| **GS-to-Occ** | 3D Gaussian → occupancy 복셀 변환 모듈 |
| **mIoU / IoU** | 시맨틱 클래스별 교집합/합집합 평균 / 전체 점유 기하 교집합/합집합 |

---

## 결론 및 시사점

GaussianWorld는 3D Gaussian의 명시적·연속적 장면 표현 특성을 활용하여 "장면이 어떻게 진화하는가"를 World Model로 학습합니다. 핵심 기여는 다음 세 가지입니다:

1. **4D 점유 예측으로의 재정의**: 3D occupancy를 이전 Gaussian + 현재 관측에 조건화된 예측 문제로 재구성하여 시간적 연속성을 자연스럽게 활용
2. **추가 연산 없는 성능 향상**: 단일 과거 프레임만 사용하면서 단일 프레임 대비 mIoU +2%, 기존 시간 융합 방법 대비 낮은 레이턴시·메모리로 우월한 성능
3. **명시적 장면 진화 모델링**: 자차 이동·동적 객체·신규 영역을 분리하여 각각을 물리적으로 타당한 방식으로 처리

**로드맵 위치**: 이 논문은 **GaussianFormer**(3DGS 기반 occupancy 표현)의 직접적 확장이며, **3DGS → occupancy prediction** 계보와 **World Model(GAIA-1 등)** 계보가 만나는 교차점입니다. 자율주행 합성 데이터 생성 관점에서는, 3D Gaussian으로 장면을 명시적으로 진화시키는 이 패러다임이 향후 클로즈드루프 센서 시뮬레이터의 장면 상태 관리에 직접 응용될 가능성이 있습니다.

**한계**: 동적 요소와 정적 요소의 분리가 완전하지 않아 정적 장면의 크로스 프레임 일관성을 완벽히 보장하지 못함.


---

*관련 논문: [3D Gaussian Splatting](/posts/papers/3d-gaussian-splatting/), [MonoScene](/posts/papers/monoscene-monocular-3d-semantic-scene-completion/), [TPVFormer](/posts/papers/tpvformer-tri-perspective-view-3d-semantic-occupancy/), [SurroundOcc](/posts/papers/SurroundOcc/), [GAIA-1](/posts/papers/GAIA-1/), [nuScenes](/posts/papers/nuscenes-multimodal-dataset-autonomous-driving/)*
