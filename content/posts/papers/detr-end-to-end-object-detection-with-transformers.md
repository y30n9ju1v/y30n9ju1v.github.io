---
title: "DETR: End-to-End Object Detection with Transformers"
date: 2026-04-20T14:00:00+09:00
draft: false
categories: ["Papers"]
tags: ["Object Detection", "Transformer", "Bipartite Matching", "DETR", "Facebook AI"]
---

## 개요

- **저자**: Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko
- **소속**: Facebook AI
- **발표**: ECCV 2020
- **주요 내용**: 객체 탐지를 직접 집합 예측 문제로 보는 end-to-end Transformer 기반 탐지기 — NMS, anchor 등 수작업 설계 컴포넌트를 완전히 제거

## 한계 극복

- **기존 한계 1 — 수작업 설계 컴포넌트 의존**: Faster R-CNN 등 기존 탐지기는 anchor 생성, NMS(Non-Maximum Suppression), proposal 매칭 휴리스틱 등 도메인 지식을 하드코딩한 컴포넌트에 의존합니다.
- **기존 한계 2 — 중복 예측 문제**: 고정 anchor나 proposal 기반 방법은 같은 객체에 여러 box를 예측하고, NMS로 후처리해야 합니다.
- **기존 한계 3 — 자기회귀 디코더의 느린 추론**: 이전 end-to-end 시도(RNN 기반)는 순차적으로 박스를 예측해 추론이 느리고 병렬화가 어렵습니다.
- **이 논문의 접근 방식**: 이분 매칭(bipartite matching)으로 예측-GT를 1:1 매칭하는 집합 손실 + Transformer 인코더-디코더로 모든 객체를 병렬로 예측 → NMS·anchor 완전 제거.

## 목차

- Section 1: Introduction
- Section 2: Related Work (집합 예측, Transformer, 객체 탐지)
- Section 3: DETR 모델 (집합 예측 손실 + 아키텍처)
- Section 4: Experiments (COCO 비교, Ablation, Panoptic Segmentation)
- Section 5: Conclusion

---

## Section 1: Introduction

**요약**

현대 객체 탐지기들은 실제로 원하는 것("이미지에서 객체 위치와 클래스를 예측")을 대신하는 surrogate task(anchor 분류, proposal 회귀 등)를 풀도록 설계되어 있습니다. 이 과정에서 NMS, anchor 설계 같은 수작업 컴포넌트가 필수가 됩니다.

DETR은 이를 근본적으로 다르게 접근합니다. 객체 탐지를 **직접 집합 예측(direct set prediction)** 문제로 정의하고, 이분 매칭으로 중복 없는 1:1 예측을 보장하며, Transformer의 self-attention으로 전체 이미지 맥락에서 객체 간 관계를 모델링합니다.

**핵심 개념**

- **Direct Set Prediction**: 고정 크기 N개의 예측을 한 번에 출력 — NMS 없이 중복 제거
- **Bipartite Matching**: 헝가리안 알고리즘으로 예측 집합과 GT 집합을 최적 1:1 매칭 → 순열 불변 손실
- **Object Query**: 디코더에 입력되는 N개의 학습 가능한 위치 임베딩 — 각 쿼리가 이미지의 서로 다른 영역/크기를 담당

---

## Section 2: Related Work

**요약**

관련 연구는 세 갈래입니다: (1) 집합 예측을 위한 이분 매칭 손실, (2) Transformer와 병렬 디코딩, (3) 현대 객체 탐지 방법론.

핵심 선행 연구로 헝가리안 알고리즘 기반 매칭 손실을 쓴 초기 탐지기들이 있지만, CNN 기반이라 객체 간 관계 모델링이 약하고 NMS가 필요했습니다. DETR은 Transformer의 전역 self-attention으로 이 한계를 극복합니다.

**핵심 개념**

- **Hungarian Algorithm (헝가리안 알고리즘)**: 비용 행렬에서 최소 비용 1:1 매칭을 찾는 고전 알고리즘 — DETR 매칭의 핵심
- **Permutation-invariant Loss**: 어떤 순서로 예측해도 같은 손실값 → 병렬 예측 가능
- **Non-autoregressive Decoding**: RNN처럼 순차적으로 출력하지 않고, 모든 출력을 동시에(병렬로) 생성

---

## Section 3.1: Object Detection Set Prediction Loss (집합 예측 손실)

**요약**

DETR은 고정 크기 N개의 예측을 출력합니다(N은 이미지 내 객체 수보다 충분히 크게 설정). 학습 시 예측 집합과 GT 집합을 헝가리안 알고리즘으로 최적 매칭한 뒤, 매칭된 쌍에 대해서만 손실을 계산합니다.

**수식 — 최적 매칭 탐색**

$$\hat{\sigma} = \arg\min_{\sigma \in \mathfrak{S}_N} \sum_{i}^{N} \mathcal{L}_{\text{match}}(y_i, \hat{y}_{\sigma(i)})$$

**수식 설명**

- **$\mathfrak{S}_N$**: N개 원소의 모든 순열 집합
- **$y_i$**: i번째 GT 객체 (클래스 레이블 $c_i$와 박스 $b_i$로 구성)
- **$\hat{y}_{\sigma(i)}$**: 순열 $\sigma$에서 i번째 GT에 매칭된 예측
- **$\mathcal{L}_{\text{match}}$**: 매칭 비용 — 클래스 확률 + 박스 위치 유사도
- 이 식의 핵심: N개 예측을 N개 GT에 **1:1로** 할당하는 최적 순열 $\hat{\sigma}$를 찾는다. 나머지 슬롯은 "no object(∅)"에 할당됨

**수식 — 헝가리안 손실 (Hungarian Loss)**

$$\mathcal{L}_{\text{Hungarian}}(y, \hat{y}) = \sum_{i=1}^{N} \left[ -\log \hat{p}_{\hat{\sigma}(i)}(c_i) + \mathbb{1}_{\{c_i \neq \varnothing\}} \mathcal{L}_{\text{box}}(b_i, \hat{b}_{\hat{\sigma}(i)}) \right]$$

**수식 설명**

매칭이 완료된 후 실제 학습에 사용되는 손실입니다:
- **$-\log \hat{p}_{\hat{\sigma}(i)}(c_i)$**: 분류 손실 — 예측 클래스 확률의 음의 로그 우도 (cross-entropy)
- **$\mathbb{1}_{\{c_i \neq \varnothing\}}$**: GT가 실제 객체일 때만 박스 손실 계산 (빈 슬롯 제외)
- **$\mathcal{L}_{\text{box}}$**: 박스 손실 — $\ell_1$ 손실 + GIoU 손실의 선형 결합
- **왜 GIoU도 쓰나?**: $\ell_1$ 손실은 크고 작은 박스에서 스케일이 달라 불균형 발생 → GIoU(스케일 불변)를 함께 사용

**수식 — 박스 손실**

$$\mathcal{L}_{\text{box}}(b_i, \hat{b}_{\hat{\sigma}(i)}) = \lambda_{\text{iou}} \mathcal{L}_{\text{iou}}(b_i, \hat{b}_{\hat{\sigma}(i)}) + \lambda_{\text{L1}} \| b_i - \hat{b}_{\hat{\sigma}(i)} \|_1$$

**수식 설명**

- **$\lambda_{\text{iou}}, \lambda_{\text{L1}}$**: GIoU 손실과 L1 손실의 가중치 하이퍼파라미터
- **$\mathcal{L}_{\text{iou}}$**: GIoU(Generalized IoU) — 박스가 겹치지 않아도 기울기 제공, 스케일 불변
- **$\| b_i - \hat{b} \|_1$**: 박스 좌표의 L1 거리 (중심점 x, y, 너비, 높이 — 이미지 크기로 정규화된 값)

---

## Section 3.2: DETR Architecture (아키텍처)

**요약**

DETR은 세 가지 주요 컴포넌트로 구성됩니다: CNN 백본, Transformer 인코더-디코더, FFN 예측 헤드.

```
이미지 → CNN Backbone → feature map (C×H×W)
       → 1×1 conv → (d×H×W) → flatten → (HW×d) sequence
       → Positional Encoding 추가
       → Transformer Encoder (전역 self-attention)
       → Transformer Decoder (N개 object query)
       → FFN Head → N개 (클래스, 박스) 예측
```

**핵심 개념**

- **CNN Backbone**: ResNet으로 이미지에서 피처 맵 추출. 기본값 $C=2048$, $H=W=H_0/32$
- **1×1 Convolution**: 채널 차원을 $C$에서 $d$로 축소 (보통 $d=256$)
- **Positional Encoding (위치 인코딩)**: Transformer는 순서를 모르므로, 각 피처 위치에 고정 사인파 인코딩을 추가 — 인코더의 모든 attention 레이어에 반복 추가됨
- **Object Query**: 디코더에 입력되는 N개의 학습 가능한 임베딩 (=출력 위치 인코딩) — 각 쿼리가 서로 다른 객체를 탐지하도록 학습됨
- **FFN (Feed-Forward Network)**: 3층 MLP + ReLU → 박스 4좌표 예측 (정규화된 중심점 x, y, 너비, 높이)
- **Auxiliary Decoding Loss**: 디코더의 각 레이어마다 FFN 헤드와 헝가리안 손실을 추가 → 학습 안정성 향상, +8.2 AP

**Transformer Encoder의 역할**

인코더의 self-attention은 이미지 전체에서 전역 관계를 학습합니다. 실험에서 인코더가 개별 인스턴스를 이미 분리하는 것이 관찰됐습니다 (Figure 3): 특정 점에 대한 self-attention이 같은 객체 내부는 높게, 다른 객체는 낮게 활성화됩니다.

**Transformer Decoder의 역할**

디코더는 N개의 object query를 인코더 출력과 cross-attention으로 상호작용시킵니다. 디코더의 attention은 객체 경계(머리, 다리 등 extremities)에 집중하는 경향이 있습니다 — 인코더가 이미 전역 분리를 수행했으므로 디코더는 세부 위치에 집중.

---

## Section 4: Experiments (실험)

**요약**

COCO 2017 데이터셋(118k 학습, 5k 검증)에서 Faster R-CNN과 비교합니다. 평가 지표는 박스 AP(다양한 IoU 임계값의 평균).

**Faster R-CNN과의 비교 (ResNet-50 기준)**

| 모델 | AP | AP$_{50}$ | AP$_S$ | AP$_M$ | AP$_L$ | FPS |
|------|-----|-----------|--------|--------|--------|-----|
| Faster RCNN-FPN+ | 42.0 | 62.1 | 26.6 | 45.4 | 53.4 | 26 |
| **DETR** | 42.0 | 62.4 | 20.5 | 45.8 | **61.1** | 28 |
| **DETR-DC5** | 43.3 | 63.1 | 22.5 | 47.3 | **61.1** | 12 |

**핵심 관찰**:
- **대형 객체(AP$_L$)**: DETR이 Faster R-CNN보다 **+7.8 AP** — Transformer의 전역 reasoning 덕분
- **소형 객체(AP$_S$)**: DETR이 -5.5 AP 낮음 — 소형 객체는 지역 특징이 중요한데, 전역 attention이 불리
- **추론 속도**: 비슷한 FPS (28 vs 26 FPS, ResNet-50 기준)
- **구현 단순성**: PyTorch로 50줄 이내 구현 가능

**Ablation 주요 결과**

| 컴포넌트 제거 | AP 변화 |
|-------------|---------|
| Encoder 없음 (0층) | -3.9 AP (특히 대형 객체 -6.0) |
| Decoder 1층만 | -8.2 AP (NMS 추가하면 회복) |
| FFN 제거 | -2.3 AP |
| Positional Encoding 없음 | -7.8 AP |
| GIoU Loss 없음 | AP$_M$, AP$_L$ 하락 |

**인코더의 중요성**: 인코더 레이어 수를 늘릴수록 AP가 단조 증가 (0층: 36.7 → 12층: 41.6). 전역 self-attention이 객체 분리에 핵심.

**Panoptic Segmentation으로 확장**

DETR 디코더 출력에 마스크 헤드(Multi-head attention → FPN → pixel-wise argmax)를 추가하면 별도의 stuff/thing 분기 없이 통합 panoptic segmentation이 가능합니다. COCO val에서 DETR-R101이 PQ 45.1로 UPSNet(43.0)과 PanopticFPN(44.1)을 능가합니다.

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **Direct Set Prediction** | 후처리 없이 고정 크기 N개의 객체 집합을 한 번에 예측 |
| **Hungarian Algorithm** | 예측-GT 간 최소 비용 1:1 최적 매칭 탐색 — 순열 불변 손실의 핵심 |
| **Object Query** | 디코더의 N개 학습 가능한 위치 임베딩 — 서로 다른 영역/크기 객체 담당으로 특화됨 |
| **Bipartite Matching Loss** | GT와 예측을 1:1 매칭하여 중복 없이 손실 계산 → NMS 불필요 |
| **GIoU Loss** | 박스가 겹치지 않아도 기울기를 제공하는 스케일 불변 IoU 손실 |
| **Auxiliary Decoding Loss** | 디코더 각 레이어에 손실을 추가 → 중간 레이어도 의미있는 예측 학습 |
| **Encoder Self-attention** | 전역 맥락에서 객체 인스턴스를 분리하는 역할 — 대형 객체 AP 향상의 주원인 |
| **Decoder Cross-attention** | Object query가 인코더 피처에서 객체 경계(extremities) 위치 추출 |
| **∅ (No Object Class)** | N개 슬롯 중 GT와 매칭되지 않은 슬롯에 할당되는 특수 클래스 — background 역할 |
| **Panoptic Segmentation** | DETR 위에 마스크 헤드만 추가하면 stuff/thing 통합 세그멘테이션 가능 |

---

## 결론 및 시사점

DETR은 객체 탐지 패러다임에서 두 가지 핵심 기여를 했습니다:

1. **패러다임 전환**: anchor·NMS·proposal 없는 순수 end-to-end 탐지기 — 도메인 지식 없이 표준 CNN + Transformer만으로 구현 가능
2. **쿼리 기반 탐지의 원형**: Object query 개념이 이후 DETR3D, BEVFormer, MapTR 등 자율주행 인식 연구 전반에 확산

**MapTR와의 연결**:
- MapTR의 계층적 쿼리(인스턴스 쿼리 + 포인트 쿼리)는 DETR의 object query를 HD 맵 요소에 맞게 확장한 것
- MapTR의 계층적 이분 매칭(인스턴스 레벨 → 포인트 레벨)은 DETR의 헝가리안 매칭을 2단계로 발전시킨 것
- DETR이 없었다면 MapTR, DETR3D, UniAD의 query-based 설계 모두 나오기 어려웠음

**한계 및 후속 연구 방향**:
- 소형 객체 성능 Faster R-CNN 대비 열세 → Deformable DETR(2021)이 deformable attention으로 해결
- 매우 긴 학습 스케줄 필요 (500 epoch) → Conditional DETR, DAB-DETR 등으로 수렴 속도 개선
- 인코더의 높은 계산 비용 (HW 크기 self-attention) → Deformable DETR, Sparse DETR로 효율화
