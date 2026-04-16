---
title: "UniSim: A Neural Closed-Loop Sensor Simulator"
date: 2026-04-17T08:45:00+09:00
draft: false
categories: ["Papers"]
tags: ["autonomous-driving", "neural-rendering", "sensor-simulation", "NeRF", "LiDAR", "closed-loop"]
---

## 개요

- **저자**: Ze Yang, Yun Chen, Jingkang Wang, Sivabalan Manivasagam, Wei-Chiu Ma, Anqi Joyce Yang, Raquel Urtasun
- **소속**: Waabi, University of Toronto, Massachusetts Institute of Technology
- **발행년도**: 2023 (arXiv:2308.01898, 3 Aug 2023)
- **주요 내용**: 단일 주행 로그(recorded log)로부터 현실적인 멀티센서 시뮬레이션을 생성하는 신경망 기반 폐루프(closed-loop) 센서 시뮬레이터. 카메라 이미지와 LiDAR 포인트 클라우드를 동시에 시뮬레이션하며, 새로운 시점·새로운 액터 추가·기존 액터 제거 등 다양한 장면 편집이 가능하다.

---

## 한계 극복

자율주행 시스템을 안전하게 검증하려면 실제 도로에서 좀처럼 발생하지 않는 위험 상황(safety-critical scenario)을 대규모로 생성해야 한다. 기존 방법들은 다음과 같은 한계를 가지고 있었다.

- **기존 한계 1 — 개방 루프(Open-loop) 평가**: 기존 시뮬레이터는 SDV(자율주행차)가 실제로 행동을 바꿨을 때 주변 센서 데이터가 어떻게 달라지는지 반영하지 못한다. 로그를 그대로 재생할 뿐이므로 자율주행 시스템이 다른 결정을 내렸을 때의 결과를 테스트할 수 없다.
- **기존 한계 2 — 편집 불가능한 정적 재생**: 기존 신경 렌더링 방법(NeRF 계열)은 장면을 사실적으로 렌더링하지만, 새로운 뷰포인트에서만 렌더링할 뿐 액터를 추가·제거하거나 SDV의 궤적을 바꾸는 편집은 지원하지 않는다.
- **기존 한계 3 — 단일 센서 모달리티**: 대부분의 시뮬레이터는 카메라 또는 LiDAR 중 하나만 지원하는 반면, 실제 SDV는 두 가지 센서를 동시에 사용하므로 단일 모달리티 시뮬레이션은 현실과 괴리가 있다.
- **이 논문의 접근 방식**: 장면을 정적 배경과 동적 액터로 분리하여 각각 신경 특징 그리드(neural feature grid)로 표현한 뒤, 볼륨 렌더링으로 카메라 이미지와 LiDAR를 동시에 합성한다. 이를 통해 SDV가 새로운 궤적을 취할 때마다 실시간으로 새로운 센서 데이터를 생성하는 **폐루프 멀티센서 시뮬레이션**을 실현한다.

---

## 목차

- Section 1: Introduction
- Section 2: Related Work
- Section 3: Neural Scene Simulation
  - 3.1 Preliminaries
  - 3.2 Compositional Scene Representation
  - 3.3 Multi-modal Sensor Simulation
  - 3.4 Learning
- Section 4: Experiments
  - 4.1 Experimental Details
  - 4.2 UniSim Controllability
  - 4.3 Realism Evaluation
  - 4.4 Perception Evaluation and Training
  - 4.5 Full Autonomy Evaluation
- Section 5: Conclusion

---

## Section 1: Introduction

**요약**

고속도로를 주행하다 갑자기 옆 차선에서 차량이 끼어드는 상황을 생각해보자. 만약 이 장면의 디지털 트윈(digital twin)이 있다면, SDV가 다른 결정을 내렸을 때 어떤 일이 일어났을지 시뮬레이션해볼 수 있다. 그러나 현실에서는 이런 위험한 장면을 안전하게 수집하기 어렵고, 기존 로그를 그대로 재생하는 방식으로는 SDV가 능동적으로 반응할 때의 새로운 센서 데이터를 얻을 수 없다.

UniSim은 이 문제를 해결하기 위해 **단일 주행 로그**에서 조작 가능한 디지털 트윈을 구성한다. 자율주행 시스템이 시뮬레이터와 상호작용하면서 차선을 바꾸거나 다른 경로를 선택하면, UniSim은 그에 맞는 새로운 카메라 이미지와 LiDAR 포인트 클라우드를 실시간으로 생성한다. 이를 통해 데이터로는 커버하기 어려운 위험 상황(vehicle cut-in, new actor insertion 등)을 폐루프로 평가할 수 있다.

**핵심 개념**

- **폐루프 시뮬레이션(Closed-loop simulation)**: SDV의 행동 → 새로운 센서 데이터 생성 → SDV의 다음 행동 결정이 반복되는 구조. 개방 루프(open-loop)와 달리 SDV의 실제 반응을 테스트할 수 있다.
- **디지털 트윈(Digital twin)**: 현실 세계의 물리적 객체나 시스템을 가상 공간에 복제한 것. 여기서는 도로 장면 전체를 신경망으로 표현한 가상 환경을 의미한다.
- **Safety-critical scenario**: 긴급 제동, 차선 끼어들기, 보행자 돌출 등 실제 도로에서 발생하기 어렵지만 자율주행 시스템의 안전성 검증에 반드시 필요한 극한 상황.

---

## Section 2: Related Work

**요약**

관련 연구는 크게 세 흐름으로 나뉜다.

1. **자율주행용 시뮬레이션 환경**: CARLA, LGSVL, GeoSim 등은 물리 엔진이나 3D 에셋 기반으로 시뮬레이션 환경을 구축한다. 현실감과 편집 자유도 사이의 균형이 어렵다.
2. **신규 뷰 합성(Novel View Synthesis)**: NeRF, Instant-NGP, Mip-NeRF 360 등은 실제 이미지로부터 새로운 시점의 이미지를 생성한다. 그러나 대규모 야외 장면과 동적 객체 처리, 그리고 LiDAR 시뮬레이션에는 한계가 있다.
3. **데이터 기반 자율주행 시뮬레이션**: Neural Scene Graphs, FVS 등은 장면을 분해하여 편집 가능성을 높이지만 폐루프 평가나 멀티센서 지원이 부족하다.

UniSim은 이 세 흐름을 통합하여 **편집 가능한 멀티센서 폐루프 시뮬레이션**을 최초로 구현한다.

**핵심 개념**

- **Neural Scene Graphs**: 장면을 노드(배경·액터)로 구성된 그래프로 표현하여 각 객체를 독립적으로 편집·조합하는 방법.
- **Volume rendering**: 3D 공간의 밀도(density)와 색상을 적분하여 2D 이미지를 생성하는 렌더링 기법. NeRF의 핵심 원리이다.

---

## Section 3: Neural Scene Simulation

### 3.1 Preliminaries — 신경 특징 필드(Neural Feature Field)

**요약**

UniSim의 기본 표현 단위는 **신경 특징 필드(Neural Feature Field, NFF)**이다. 3D 공간의 임의의 점 $\mathbf{x} \in \mathbb{R}^3$에 $N_f$차원의 특징 벡터 $f(\mathbf{x}) \in \mathbb{R}^{N_f}$를 매핑하는 연속 함수로, 실제로는 신경망(MLP)으로 구현된다.

**핵심 개념**

- **SDF(Signed Distance Function)**: 임의의 3D 점에서 가장 가까운 표면까지의 부호 있는 거리. 양수면 표면 밖, 음수면 내부를 의미한다. 표면 형태를 부드럽게 표현하는 데 유용하다.
- **암시적 기하학(Implicit geometry)**: 메쉬(mesh)처럼 명시적으로 꼭짓점을 저장하지 않고 수식으로 표면을 정의하는 방법. SDF가 대표적 예시이다.
- **하이퍼네트워크(Hypernetwork)**: 다른 신경망의 가중치를 출력하는 신경망. UniSim에서는 각 액터의 잠재 코드(latent code)를 받아 해당 액터의 특징 그리드 가중치를 생성한다.

**다중 해상도 특징 그리드(Multi-resolution features grid)**

단일 NFF는 대규모 야외 장면에서 세밀한 디테일과 넓은 범위를 동시에 표현하기 어렵다. UniSim은 이를 해결하기 위해 **다중 해상도 특징 그리드**를 도입한다. 구체적으로, 주어진 3D 점 $\mathbf{x}$에 대해 해상도가 다른 여러 그리드 $\{g^l\}_{l=1}^{L}$에서 특징을 삼선형 보간(trilinear interpolation)으로 추출한 뒤 연결(concatenate)하여 최종 특징 벡터를 만든다.

$$\mathbf{f} = f\left(\left\{\text{interp}(g^l, \mathbf{x})\right\}_{l=1}^{L}, \mathbf{x}\right)$$

**수식 설명**
- **$\mathbf{f}$**: 최종 특징 벡터 — 이 점의 색상·재질·투명도 등을 인코딩한다.
- **$g^l$**: $l$번째 해상도 특징 그리드 (coarse에서 fine까지)
- **$\text{interp}(g^l, \mathbf{x})$**: 그리드 $g^l$에서 $\mathbf{x}$ 위치의 값을 삼선형 보간으로 추출
- **$f(\cdot)$**: 추출된 다중 해상도 특징을 조합하여 최종 특징을 계산하는 신경망

---

### 3.2 Compositional Scene Representation — 합성적 장면 표현

**요약**

UniSim은 장면을 두 가지 요소로 분리하여 표현한다.

1. **정적 배경(Static background)**: 도로, 건물, 나무 등 움직이지 않는 요소. 전체 주행 범위를 아우르는 대규모 희소 특징 그리드(sparse feature grid)로 표현한다.
2. **동적 액터(Dynamic actors)**: 차량, 보행자 등 움직이는 객체. 각 액터는 바운딩 박스로 정의되며, 하이퍼네트워크가 액터의 잠재 코드를 받아 개별 특징 그리드를 생성한다.

이 분리 덕분에 특정 액터만 제거하거나, 새 액터를 삽입하거나, SDV의 궤적을 바꿀 때 배경은 그대로 두고 액터 부분만 재합성하면 된다. 이것이 UniSim이 폐루프 편집을 가능하게 하는 핵심 아이디어이다.

**핵심 개념**

- **희소 복셀 그리드(Sparse voxel grid)**: 3D 공간 전체를 균일한 격자로 나누는 대신, 실제로 데이터가 있는 영역(도로 주변)의 복셀만 저장하여 메모리를 절약한다.
- **점유 그리드(Occupancy grid $V_\text{occ}$)**: 각 복셀이 비어있는지(free space) 아니면 물체가 있는지를 나타내는 3D 격자. 렌더링 시 물체가 없는 공간의 불필요한 연산을 건너뛰는 데 사용된다.
- **잠재 코드(Latent code $\mathbf{z}_A$)**: 각 액터의 외관(색상·형태)을 압축적으로 표현하는 벡터. 하이퍼네트워크가 이 코드를 입력받아 해당 액터의 특징 그리드를 생성한다.

**일반화 액터 모델(Generalized actor model)**

개별 액터마다 별도의 신경망을 학습하면 파라미터 수가 폭발적으로 늘어난다. UniSim은 이를 해결하기 위해 **하이퍼네트워크** $f_\theta$를 사용한다. 모든 액터의 잠재 코드 $\mathbf{z}_A$와 샘플링된 3D 점 $\mathbf{x}_{A,i}$를 받아 특징 그리드를 생성하고, 다음과 같이 특징을 회귀(regress)한다.

$$\mathcal{F}_A = f_\theta(\mathbf{z}_A), \quad \hat{f}_{A,i} = f_k(\mathcal{F}_A, \mathbf{x}_{A,i})$$

**수식 설명**
- **$\mathcal{F}_A$**: 액터 $A$의 특징 그리드 — 하이퍼네트워크가 잠재 코드로부터 생성한다.
- **$f_\theta$**: 하이퍼네트워크 — 잠재 코드를 입력받아 특징 그리드의 가중치를 출력한다.
- **$\mathbf{z}_A$**: 액터 $A$의 잠재 코드 — 해당 액터의 외관 정보를 담고 있다.
- **$\hat{f}_{A,i}$**: $i$번째 샘플 점에서의 특징 벡터 — 이 값이 색상과 투명도로 디코딩된다.

---

### 3.3 Multi-modal Sensor Simulation — 멀티모달 센서 시뮬레이션

**요약**

배경과 액터의 특징이 합성되면, 이를 카메라 이미지와 LiDAR 포인트 클라우드로 렌더링한다.

#### 카메라 시뮬레이션

볼륨 렌더링 방정식을 사용하여 카메라 중심 $\mathbf{o}$에서 픽셀 방향 $\mathbf{d}$로 레이(ray)를 쏘아 $N_r$개의 샘플 점에서 특징을 집계한다.

$$\mathbf{f}(\mathbf{r}) = \sum_{i=1}^{N_r} w_i \mathbf{f}_i, \quad w_i = \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j)$$

**수식 설명**
- **$\mathbf{f}(\mathbf{r})$**: 레이 $\mathbf{r}$에 대한 최종 특징 벡터 — 이 값이 RGB 이미지로 디코딩된다.
- **$w_i$**: $i$번째 샘플의 가중치 — 해당 점이 레이 색상에 기여하는 비율.
- **$\alpha_i$**: $i$번째 샘플의 불투명도(opacity) — SDF $s_i$로부터 $\alpha = 1/(1+\exp(\beta \cdot s))$로 계산한다.
- **$\prod_{j=1}^{i-1}(1-\alpha_j)$**: 앞의 모든 샘플을 통과해 온 빛의 투과율 — 가려진(occluded) 객체는 자동으로 낮은 가중치를 받는다.

집계된 특징 벡터는 2D CNN $g_\text{rgb}$를 통해 최종 RGB 이미지 $\mathbf{I}_\text{rgb}$로 변환된다.

$$g_\text{rgb}: \mathbf{F} \in \mathbb{R}^{H_f \times W_f \times N_f} \rightarrow \mathbf{I}_\text{rgb} \in \mathbb{R}^{H \times W \times 3}$$

**수식 설명**
- **$\mathbf{F}$**: 모든 픽셀의 특징 벡터를 쌓은 2D 특징 맵 (해상도 $H_f \times W_f$)
- **$g_\text{rgb}$**: 특징 맵을 RGB 이미지로 변환하는 CNN 디코더 — upsampling 역할도 겸한다.
- **$H_f \times W_f$**: 렌더링 시 사용하는 낮은 해상도 (메모리·속도 효율을 위해 실제 이미지보다 작게 설정)

#### LiDAR 시뮬레이션

LiDAR는 레이저 빔을 쏘아 반사 거리(depth)와 세기(intensity)를 측정한다. UniSim은 LiDAR 센서 모델을 카메라와 동일한 볼륨 렌더링 프레임워크 안에서 처리한다.

$$D(\mathbf{r}) = \sum_{i=1}^{N_r} w_i t_i$$

**수식 설명**
- **$D(\mathbf{r})$**: 레이 $\mathbf{r}$에 대해 예측된 LiDAR 거리값(depth)
- **$t_i$**: 레이 원점으로부터 $i$번째 샘플까지의 거리
- **$w_i$**: 카메라 렌더링과 동일한 가중치 — SDF 기반 불투명도로 계산된다.

반사 세기는 LiDAR 특징을 MLP 디코더 $g_\text{int}$로 디코딩하여 예측한다: $\hat{l}^\text{int}(\mathbf{r}) = g_\text{int}(\mathbf{f}(\mathbf{r}))$.

**핵심 개념**

- **Time-of-flight 센서**: LiDAR처럼 빛(레이저)을 쏘고 반사되어 돌아오는 시간으로 거리를 측정하는 센서. UniSim은 이 원리를 신경망 볼륨 렌더링으로 모사한다.
- **Intensity (반사 세기)**: LiDAR가 측정하는 두 번째 값. 재질(예: 도로 vs. 금속)에 따라 반사율이 다르며, UniSim은 신경망으로 이를 예측한다.

---

### 3.4 Learning — 학습

**요약**

UniSim은 배경 특징 그리드, 액터 잠재 코드, 하이퍼네트워크, MLP 헤드, CNN 디코더를 공동으로 최적화한다. 전체 목적 함수는 다음과 같다.

$$\mathcal{L} = \mathcal{L}_\text{rgb} + \lambda_\text{lidar}\mathcal{L}_\text{lidar} + \lambda_\text{reg}\mathcal{L}_\text{reg} + \lambda_\text{adv}\mathcal{L}_\text{adv}$$

**수식 설명**
- **$\mathcal{L}_\text{rgb}$**: 이미지 재현 손실 — 픽셀 수준 $\ell_2$ 손실과 VGG 퍼셉추얼 손실의 합.
- **$\mathcal{L}_\text{lidar}$**: LiDAR 재현 손실 — 예측 depth/intensity와 실제 측정값의 $\ell_2$ 차이.
- **$\mathcal{L}_\text{reg}$**: 정규화 손실 — SDF가 매끄러운 표면을 형성하도록 강제하는 Eikonal 손실 포함.
- **$\mathcal{L}_\text{adv}$**: 적대적 손실 — 미관측 뷰포인트에서 현실감을 높이기 위한 GAN 판별자 손실.
- **$\lambda_\text{lidar}, \lambda_\text{reg}, \lambda_\text{adv}$**: 각 손실의 균형을 조절하는 하이퍼파라미터.

**이미지 손실 상세**

$$\mathcal{L}_\text{rgb} = \frac{1}{N_\text{ob}} \sum_{i=1}^{N_\text{ob}} \left(\|\mathbf{I}_i^\text{gb} - \hat{\mathbf{I}}_i^\text{gb}\|_2 + \lambda \sum_{j=1}^{M} \|V^j(\mathbf{I}_i^\text{gb}) - V^j(\hat{\mathbf{I}}_i^\text{gb})\|_1\right)$$

**수식 설명**
- **$\mathbf{I}_i^\text{gb}$**: 실제 관측 이미지 패치
- **$\hat{\mathbf{I}}_i^\text{gb}$**: 시뮬레이션된 이미지 패치
- **$V^j(\cdot)$**: 사전 학습된 VGG 네트워크의 $j$번째 레이어 출력 — 픽셀 차이 대신 고수준 특징 차이를 측정하여 더 자연스러운 이미지를 생성한다.
- **$N_\text{ob}$**: 관측된 뷰포인트 수

**LiDAR 손실 상세**

$$\mathcal{L}_\text{lidar} = \frac{1}{N} \sum_{i=1}^{N} \left(\|D(\mathbf{r}_i) - D_i^\text{gt}\|_2 + \|\hat{l}^\text{int}(\mathbf{r}_i) - \hat{l}_i^\text{int,gt}\|_2\right)$$

**수식 설명**
- **$D(\mathbf{r}_i)$**: 레이 $i$에 대한 예측 depth
- **$D_i^\text{gt}$**: 실제 LiDAR 측정 depth (ground truth)
- **$\hat{l}^\text{int}$**: 예측 반사 세기, **$\hat{l}_i^\text{int,gt}$**: 실제 반사 세기
- LiDAR 노이즈를 고려하여 각 배치에서 depth 오차가 가장 작은 95%의 레이만 사용한다.

**적대적 손실(Adversarial loss)**

관측된 뷰포인트에서만 지도 학습하면 미관측 뷰포인트(예: 차선 변경 후의 새 시점)에서 아티팩트가 발생한다. GAN 판별자 $D_\text{adv}$를 사용하여 미관측 뷰포인트에서도 현실적인 이미지를 생성하도록 강제한다.

$$\mathcal{L}_\text{adv} = \frac{1}{N_\text{adv}} \sum_{i=1}^{N_\text{adv}} \log(1 - D_\text{adv}(\mathbf{I}_i^\text{gb,R}))$$

**수식 설명**
- **$N_\text{adv}$**: 미관측(무작위 이동) 뷰포인트 수
- **$D_\text{adv}$**: 실제 이미지와 시뮬레이션 이미지를 구별하는 CNN 판별자
- 관측 뷰포인트에서는 픽셀 단위 지도 손실로, 미관측 뷰포인트에서는 GAN 손실로 현실감을 높인다.

---

## Section 4: Experiments

### 4.1 Experimental Details

**요약**

실험은 샌프란시스코 도심 지역의 공개 데이터셋 **PandaSet**에서 수행되었다. 총 103개의 주행 장면, 각 장면은 8초(80프레임, 10Hz), 전방 와이드 앵글 카메라(1920×1080)와 360° 스피닝 LiDAR를 사용한다.

---

### 4.2 UniSim Controllability

**요약**

UniSim은 다음과 같은 장면 편집을 지원한다.

- **액터 제거(Actor Removal)**: 특정 차량이나 보행자를 장면에서 삭제하고 배경으로 채운다.
- **액터 수정(Actor Modification)**: 기존 액터를 다른 차종으로 교체하거나 위치를 변경한다.
- **SDV 센서 위치 변경(SDV Sensor Lift)**: 카메라/LiDAR 높이를 바꾸어 다른 센서 설정에서의 뷰를 시뮬레이션한다.
- **SDV 차선 변경(SDV Lane Change)**: SDV의 궤적을 옆 차선으로 이동시켜 완전히 새로운 시점의 센서 데이터를 생성한다.
- **새 액터 삽입(New Actor Insertion)**: 기존에 없던 차량이나 트럭을 원하는 위치에 추가한다.

---

### 4.3 Realism Evaluation

**요약**

UniSim의 센서 시뮬레이션 현실감을 기존 방법과 비교한다.

**카메라 시뮬레이션 비교 (Table 1)**

| 방법 | PSNR↑ | SSIM↑ | LPIPS↓ | FID@2m↓ | FID@3m↓ |
|------|-------|-------|--------|---------|---------|
| FVS | 21.09 | 0.700 | 0.299 | 112.6 | 135.8 |
| NSG | 20.74 | 0.600 | 0.556 | 319.2 | 343.0 |
| Instant-NGP | 24.03 | 0.708 | 0.451 | 192.8 | 220.1 |
| **UniSim (Ours)** | **25.63** | **0.745** | **0.288** | **74.7** | **97.5** |

- **보간(Interpolation)**: 실제 관측 뷰포인트 사이의 중간 시점 렌더링
- **차선 이동(Lane shift)**: SDV를 2~3m 측면으로 이동한 새로운 시점 — 이 설정에서 UniSim의 우위가 더욱 두드러진다.

**LiDAR 시뮬레이션 비교 (Table 3)**

| 방법 | Median $r_z$ Error↓ | Hit Rate↑ | Intensity RMSE↓ |
|------|---------------------|-----------|-----------------|
| LiDARsim | 0.11 | 92.2% | 0.091 |
| **UniSim (Ours)** | **0.10** | **99.6%** | **0.065** |

UniSim은 Hit Rate를 92.2% → 99.6%로 크게 향상시키며, Intensity RMSE도 28% 감소시킨다.

**핵심 개념**

- **PSNR (Peak Signal-to-Noise Ratio)**: 이미지 품질 지표. 높을수록 실제 이미지에 더 가깝다.
- **SSIM (Structural Similarity Index)**: 밝기·대비·구조 세 가지 측면에서 이미지 유사도를 측정. 1에 가까울수록 좋다.
- **LPIPS (Learned Perceptual Image Patch Similarity)**: 신경망이 인식하는 지각적 유사도. 낮을수록 좋다.
- **FID (Fréchet Inception Distance)**: 생성된 이미지와 실제 이미지의 분포 차이를 측정. 낮을수록 실제에 가깝다.
- **Hit Rate**: LiDAR 레이가 실제 포인트 클라우드와 일치하는 비율. UniSim은 99.6%로 거의 모든 레이가 정확히 맞는다.

---

### 4.4 Perception Evaluation and Training

**요약**

단순히 예쁜 이미지를 생성하는 것 이상으로, 시뮬레이션 데이터가 **다운스트림 인식 모델 학습**에도 유용한지 검증한다.

- **Domain gap 측정**: 실제 데이터로 학습하고 UniSim 시뮬레이션 데이터로 테스트했을 때 (또는 그 반대), 성능 저하(domain gap)를 측정한다.
- **데이터 증강(Data Augmentation)**: 실제 데이터에 UniSim으로 생성한 데이터를 추가하면 3D 객체 탐지 모델의 mAP가 향상된다 (Table 5).
- UniSim 시뮬레이션 데이터만으로 학습해도 실제 데이터 대비 경쟁력 있는 탐지 성능을 달성한다.

---

### 4.5 Full Autonomy Evaluation with UniSim

**요약**

UniSim의 핵심 응용: 자율주행 시스템 전체를 폐루프로 평가한다. BEVFormer 기반 자율주행 시스템을 UniSim 시뮬레이터에 연결하여 평가한다.

- **평가 지표**: IoU 0.3 기준 예측 일치율(detection agreement), 평균 변위 오차(ADE), 충돌률(collision rate), 경로 계획 일관성 등.
- **폐루프 시뮬레이션**: 자율주행 시스템이 차선 변경 결정을 내리면 UniSim이 즉시 새로운 센서 데이터를 생성하고, 시스템은 업데이트된 데이터로 다음 행동을 결정한다.
- UniSim이 기존 방법(Instant-NGP, FVS)보다 실제 주행 상황에 더 가까운 시뮬레이션 환경을 제공하여 도메인 갭이 가장 작다.
- 시뮬레이션에서 학습한 데이터로 추가 파인튜닝 시 실제 도로 성능도 향상된다.

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **Neural Feature Field (NFF)** | 3D 공간의 임의 점에 특징 벡터를 매핑하는 신경망 기반 표현. SDF로 기하학을 암시적으로 표현한다. |
| **다중 해상도 특징 그리드** | 해상도가 다른 여러 3D 격자에서 특징을 추출·연결하여 세밀한 디테일과 넓은 범위를 동시에 표현한다. |
| **합성적 장면 표현** | 배경(정적)과 액터(동적)를 독립적으로 표현하여 개별 편집이 가능하게 한다. |
| **하이퍼네트워크** | 액터의 잠재 코드를 받아 그 액터의 특징 그리드 가중치를 생성하는 신경망. 모든 액터를 단일 모델로 처리한다. |
| **볼륨 렌더링** | SDF 기반 불투명도 가중치로 3D 특징을 레이 방향으로 적분하여 2D 이미지 또는 LiDAR depth를 생성한다. |
| **폐루프 시뮬레이션** | SDV의 행동 → 새로운 센서 데이터 생성 → SDV의 다음 행동이 반복되는 구조. 실제 반응을 테스트할 수 있다. |
| **GAN 기반 적대적 학습** | 미관측 뷰포인트에서도 현실적인 이미지를 생성하도록 판별자 손실을 추가한다. |
| **Eikonal 정규화** | SDF의 그래디언트 크기가 1이 되도록 강제하여 매끄러운 표면 표현을 유도하는 정규화 손실. |

---

## 결론 및 시사점

UniSim은 단일 주행 로그에서 카메라와 LiDAR를 동시에 사실적으로 시뮬레이션하는 신경망 기반 폐루프 센서 시뮬레이터이다. 장면을 정적 배경과 동적 액터로 분리하고, 하이퍼네트워크와 다중 해상도 특징 그리드를 결합하여 자유로운 장면 편집과 새로운 시점 합성을 실현한다.

**주요 기여**

1. **최초의 신경망 기반 멀티센서 폐루프 시뮬레이터**: 카메라와 LiDAR를 단일 프레임워크에서 동시 시뮬레이션한다.
2. **자유로운 장면 편집**: 액터 추가·제거·수정, SDV 궤적 변경, 센서 설정 변경이 가능하다.
3. **인식 모델 학습 지원**: 시뮬레이션 데이터로 데이터 증강 시 탐지 성능이 향상된다.
4. **폐루프 자율주행 평가**: 실제 도로와 가장 유사한 시뮬레이션 환경에서 자율주행 시스템 전체를 검증한다.

**실무적 시사점**

- 데이터 수집 비용이 높은 안전 임계 시나리오(급제동, 끼어들기 등)를 저비용으로 대규모 생성할 수 있다.
- 신규 센서 설정(카메라 위치 변경, LiDAR 업그레이드)에 대한 영향 분석을 실제 차량 없이 수행할 수 있다.
- 향후 연구로 더 긴 시간 범위의 시뮬레이션, 날씨·조명 변화 등 외부 조건 변화 지원이 기대된다.
