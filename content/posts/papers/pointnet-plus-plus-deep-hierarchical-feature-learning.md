---
title: "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space"
date: 2026-04-29T00:00:00+09:00
draft: false
categories: ["Papers"]
tags: ["Point Cloud", "3D Classification", "3D Segmentation", "Deep Learning", "LiDAR", "Hierarchical Learning", "PointNet"]
---

## 개요

- **저자**: Charles R. Qi, Li Yi, Hao Su, Leonidas J. Guibas (Stanford University)
- **발행년도**: 2017
- **arXiv**: 1706.02413
- **주요 내용**: PointNet의 핵심 한계인 "로컬 구조 무시"를 계층적 특징 학습으로 해결한 후속 논문. FPS + Ball Query + Mini-PointNet 조합으로 CNN과 유사한 다중 스케일 특징 추출을 비정형 포인트 클라우드에서 실현.

## 한계 극복

- **기존 한계 1 — PointNet의 글로벌 집계**: PointNet은 모든 점의 특징을 단일 max pooling으로 합쳐 전역 특징만 추출. 세밀한 로컬 기하 구조를 학습하지 못함.
- **기존 한계 2 — 밀도 불균일 취약성**: 실제 LiDAR 스캔은 원근 효과·가려짐 등으로 밀도가 위치마다 다름. 균일 밀도로 훈련한 모델은 희소 영역에서 성능이 급락.
- **이 논문의 접근 방식**: PointNet을 재귀적으로 중첩 파티션(nested partition)에 적용하는 계층 구조를 도입. 밀도 적응형 레이어(MSG, MRG)로 다중 스케일 특징을 학습하여 불균일 밀도에도 강인하게 동작.

## 목차

- Section 1: Introduction
- Section 2: Problem Statement
- Section 3: Method
  - 3.1 PointNet 복습
  - 3.2 계층적 포인트셋 특징 학습
  - 3.3 비균일 샘플링 밀도에서의 강인한 특징 학습
  - 3.4 세그멘테이션을 위한 포인트 특징 전파
- Section 4: Experiments
- Section 5: Related Work
- Section 6: Conclusion

## Section 1: Introduction

**요약**

3D 스캐너로 얻은 포인트 클라우드는 순서 없는(unordered) 점들의 집합으로, 순열 불변성(permutation invariance)을 요구한다. PointNet이 이 문제를 처음으로 직접 해결했지만, 단일 max pooling 집계 방식은 CNN처럼 점진적으로 로컬→글로벌 특징을 쌓아 올리는 구조가 아니었다.

PointNet++는 CNN의 핵심 성공 요인인 **계층적 수용 영역(hierarchical receptive field)** 개념을 포인트 클라우드에 이식한다. 작은 이웃에서 세밀한 기하 패턴을 잡고, 이를 점점 더 큰 단위로 통합하여 최종 전역 표현을 만든다.

**핵심 개념**

- **포인트 클라우드**: 3D 공간의 점 $(x, y, z)$ (+ 추가 특징) 집합. 격자가 없어 CNN 직접 적용 불가.
- **순열 불변성**: 입력 순서가 바뀌어도 결과가 동일해야 함. max pooling이 이를 보장.
- **계층적 특징 학습**: 저수준(엣지·코너) → 중간(부품) → 고수준(전체 물체) 표현을 단계적으로 구성.

---

## Section 2: Problem Statement

**요약**

$\mathcal{X} = (M, d)$를 유클리드 공간 $\mathbb{R}^n$에서 상속된 거리 메트릭 $d$를 가진 이산 메트릭 공간이라 할 때, 목표는 집합 함수 $f: \mathcal{X} \to \mathbb{R}$을 학습하는 것이다. $f$는 분류 함수(전체 레이블)이거나 세그멘테이션 함수(점별 레이블)일 수 있다.

**핵심 개념**

- **메트릭 공간**: 거리 함수가 정의된 공간. 유클리드 거리 외에 측지선 거리(geodesic distance)도 사용 가능 → 비강체(non-rigid) 형상 분류에 활용.
- **밀도 비균일성**: 실제 포인트 클라우드는 위치에 따라 밀도가 다름 (원근 효과, 가려짐, 모션 블러 등).

---

## Section 3: Method

### 3.1 PointNet 복습

**요약**

PointNet은 unordered point set $\{x_1, x_2, ..., x_n\}$에 대해 아래 집합 함수를 학습한다:

$$f(x_1, x_2, ..., x_n) = \gamma\left(\underset{i=1,...,n}{\text{MAX}}\{h(x_i)\}\right)$$

**수식 설명**

- **$h$**: 각 점에 독립적으로 적용하는 MLP (점의 공간 인코딩)
- **$\text{MAX}$**: 모든 점에서 채널별 최댓값을 취하는 대칭 함수 → 순열 불변성 보장
- **$\gamma$**: 집계된 전역 특징에서 최종 출력을 만드는 MLP
- **한계**: 단 한 번의 MAX pooling이므로 로컬 구조 정보가 소실됨

---

### 3.2 계층적 포인트셋 특징 학습 (Set Abstraction)

**요약**

PointNet++의 핵심 빌딩 블록은 **Set Abstraction (SA) 레이어**이다. 각 SA 레이어는 세 단계로 구성된다:

1. **Sampling Layer** — 입력 $N$개 점에서 $N'$개 centroid를 선택
2. **Grouping Layer** — 각 centroid 주변의 이웃 점들을 묶어 로컬 영역 구성
3. **PointNet Layer** — 각 로컬 영역을 mini-PointNet으로 인코딩

이 과정이 반복되면서 점의 수는 줄어들고($N \to N' \to ...$), 각 점이 표현하는 공간 범위는 넓어진다 (CNN의 downsampling + receptive field 확장과 동일한 원리).

**핵심 개념**

- **FPS (Farthest Point Sampling)**: centroid 선택 알고리즘. 이미 선택된 점들로부터 가장 먼 점을 반복적으로 고름.
  - 랜덤 샘플링 대비 전체 공간을 균등하게 커버
  - 입력 분포에 의존적인 수용 영역 생성

- **Ball Query**: 반경 $r$ 이내의 점을 이웃으로 정의 (최대 $K$개).
  - kNN 대비 고정된 실제 공간 스케일 → 특징이 공간적으로 일관됨
  - 세그멘테이션처럼 로컬 패턴 인식이 필요한 태스크에 유리

- **로컬 좌표계 변환**: 각 이웃 점의 좌표를 centroid 기준 상대 좌표로 변환
  $$x_i^{(j)} = x_i^{(j)} - \hat{x}^{(j)}$$
  점 간 공간 관계(point-to-point relations)를 포착하기 위함.

---

### 3.3 비균일 샘플링 밀도에서의 강인한 특징 학습

**요약**

실제 데이터의 밀도 불균일 문제를 해결하기 위해 두 가지 **밀도 적응형 레이어**를 제안한다.

**Multi-scale Grouping (MSG)**

각 abstraction 레벨에서 여러 반경($r_1 < r_2 < r_3$)으로 동시에 grouping하고, 각 스케일의 PointNet 출력을 concat하여 멀티스케일 특징 벡터를 만든다.

훈련 시 **Random Input Dropout**: 각 훈련 샘플에 대해 dropout ratio $\theta \sim \text{Uniform}[0, p]$를 샘플링하여 점을 무작위 제거. 모델이 다양한 밀도 환경을 경험하게 함.

**Multi-resolution Grouping (MRG)**

MSG보다 계산 효율적인 대안. 각 레벨 $L_i$의 특징은 두 벡터의 concat:
- **벡터 1**: 하위 레벨 $L_{i-1}$의 SA 출력을 요약 (세밀한 정보)
- **벡터 2**: 해당 로컬 영역의 raw 점들을 단일 PointNet으로 직접 처리 (큰 스케일 정보)

밀도가 낮은 영역에서는 벡터 1의 신뢰도가 낮아지고 벡터 2가 더 중요해지는 효과가 자동으로 발생.

**핵심 개념**

- **MSG**: 정확도 우선. 여러 스케일을 명시적으로 모두 계산.
- **MRG**: 효율 우선. 하위 레벨 결과를 재활용하여 대형 이웃 재계산 회피.

---

### 3.4 포인트 특징 전파 (Feature Propagation for Segmentation)

**요약**

세그멘테이션은 모든 원본 점에 레이블이 필요하다. 그러나 SA 레이어를 거치면서 점이 $N \to N_1' \to N_2' \to ...$로 줄어든다. 이를 복원하기 위해 **계층적 보간(interpolation) + skip connection** 전략을 사용한다.

레벨 $l$에서 $l-1$로의 특징 전파:

$$f^{(j)}(x) = \frac{\sum_{i=1}^{k} w_i(x) f_i^{(j)}}{\sum_{i=1}^{k} w_i(x)}, \quad w_i(x) = \frac{1}{d(x, x_i)^p}$$

**수식 설명**

- **$f^{(j)}(x)$**: 보간으로 복원된 점 $x$의 $j$번째 채널 특징값
- **$k$**: 보간에 사용할 nearest neighbor 수 (기본값 $k=3$)
- **$w_i(x)$**: 역거리 가중치 — 가까운 점의 특징을 더 많이 반영
- **$d(x, x_i)$**: 점 $x$와 $i$번째 이웃 사이의 거리
- **$p$**: 거리 감쇠 지수 (기본값 $p=2$)
- 보간 후 skip connection으로 이어진 SA 레벨의 특징과 concat → unit PointNet (1×1 conv 역할)로 정제

이 과정은 U-Net의 encoder-decoder 구조와 동일한 원리.

---

## Section 4: Experiments

**요약**

4개 데이터셋에서 평가: MNIST (2D), ModelNet40 (3D 강체), SHREC15 (3D 비강체), ScanNet (실내 장면 세그멘테이션).

**주요 결과**

| 태스크 | 데이터셋 | PointNet | PointNet++ | 개선 |
|--------|---------|---------|-----------|------|
| 분류 | MNIST | 0.78% error | **0.51% error** | 34.6% 오류 감소 |
| 분류 | ModelNet40 | 89.2% acc | **91.9% acc** | +2.7%p |
| 세그멘테이션 | ScanNet | 73.0% | **84.5% (MSG+DP)** | +11.5%p |
| 비강체 분류 | SHREC15 | - | **96.09%** | SOTA |

**밀도 강인성 실험**: 1024개 → 256개로 점을 줄였을 때 MSG+DP의 정확도 하락이 1% 미만. PointNet vanilla는 훨씬 큰 폭으로 하락.

**비유클리드 메트릭 공간 실험**: SHREC15 비강체 형상 분류에서 측지선 거리 기반 메트릭 공간을 사용하면 XYZ 좌표 기반 대비 크게 향상 (60.18% → 96.09%). 포즈 변화에 불변한 내재적 구조를 잡기 때문.

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **Set Abstraction (SA)** | FPS + Ball Query + mini-PointNet의 조합으로 로컬 영역 특징 추출 |
| **FPS** | Farthest Point Sampling — 공간 전체를 균등하게 커버하는 centroid 선택 |
| **Ball Query** | 반경 $r$ 이내 점을 이웃으로 정의. 고정 공간 스케일로 특징 일관성 보장 |
| **MSG** | Multi-Scale Grouping — 여러 반경 동시 사용, 정확도 우선 |
| **MRG** | Multi-Resolution Grouping — 하위 레벨 재활용, 효율 우선 |
| **Random Input Dropout** | 훈련 시 점을 무작위 제거해 밀도 불균일에 강인한 모델 학습 |
| **Feature Propagation** | 역거리 가중 보간 + skip connection으로 세그멘테이션용 점 특징 복원 |

## 결론 및 시사점

PointNet++는 "포인트 클라우드에서 어떻게 CNN처럼 계층적 특징을 뽑을 것인가"에 대한 명확한 해답을 제시한다.

**자율주행·합성 데이터 관점 시사점**

- **LiDAR 센서 모델링**: 실제 LiDAR는 거리에 따라 포인트 밀도가 달라짐. PointNet++의 MSG/MRG는 이 불균일성을 명시적으로 처리 → 시뮬레이터에서 생성한 균일 밀도 포인트 클라우드와 실제 데이터 사이의 도메인 갭을 줄이는 아이디어로 활용 가능.
- **CenterPoint, PointPillars와의 관계**: 두 논문 모두 LiDAR 포인트를 BEV 격자로 변환하여 CNN 적용. PointNet++는 "격자 변환 없이" 직접 포인트 처리하는 대안으로, 세밀한 기하 정보 보존에 유리.
- **세그멘테이션 응용**: Feature Propagation의 보간 + skip connection 아이디어는 이후 3D 의미론적 세그멘테이션(ScanNet 등) 모델의 표준 구조로 정착.
- **한계**: MSG는 계산 비용이 높음. 저자들도 inference 속도 향상을 향후 과제로 명시 → 이후 PointPillars, CenterPoint가 속도 우선 설계로 실용화.
