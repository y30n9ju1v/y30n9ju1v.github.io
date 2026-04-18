---
title: "DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries"
date: 2026-04-18T09:00:00+09:00
draft: false
tags: ["3D Object Detection", "Multi-view Camera", "Transformer", "Autonomous Driving", "nuScenes"]
categories: ["Papers"]
description: "카메라 이미지만으로 3D 바운딩 박스를 예측하는 top-down 방식의 멀티뷰 3D 객체 검출 프레임워크"
math: true
---

## 개요

- **저자**: Yue Wang, Vitor Guizilini, Tianyuan Zhang, Yilun Wang, Hang Zhao, Justin Solomon
- **소속**: MIT, Toyota Research Institute, CMU, Li Auto, Tsinghua University
- **발행년도**: 2021
- **arXiv**: 2110.06922
- **주요 내용**: depth prediction 없이 3D object query를 2D 이미지에 back-projection하여 멀티뷰 카메라 기반 3D 객체 검출을 수행하는 DETR3D 프레임워크 제안

---

## 한계 극복

- **기존 한계 1 — bottom-up 방식의 depth 오류 누적**: CenterNet, FCOS3D 같은 기존 방법은 2D 검출 파이프라인에서 depth를 예측한 뒤 3D로 변환합니다. depth 추정 오류가 3D 검출 성능에 직접적으로 누적됩니다.
- **기존 한계 2 — NMS 의존**: 멀티뷰 카메라에서 각 이미지를 독립적으로 처리한 뒤, 중복 박스를 제거하기 위해 NMS(Non-Maximum Suppression) 같은 후처리가 필요합니다. NMS는 병렬화가 어렵고 추론 속도를 저하시킵니다.
- **기존 한계 3 — 카메라 오버랩 영역 처리 취약**: 카메라 뷰가 겹치는 영역에서 객체를 검출할 때, 각 뷰에서 독립 예측 후 합치는 방식은 정보를 충분히 활용하지 못합니다.
- **이 논문의 접근 방식**: 3D 공간에서 object query를 정의하고, 그 query를 카메라 변환 행렬로 2D 이미지에 back-projection하여 특징을 샘플링합니다. depth를 명시적으로 예측하지 않으며, set-to-set loss로 NMS 없이 end-to-end 학습합니다.

---

## 목차

- Section 1: Introduction
- Section 2: Related Work
- Section 3: Multi-view 3D Object Detection (핵심 방법론)
  - 3.1 Overview
  - 3.2 Feature Learning
  - 3.3 Detection Head
  - 3.4 Loss
- Section 4: Experiments
  - 4.1 Implementation Details
  - 4.2 Comparison to Existing Works
  - 4.3 Comparison in Overlap Regions
  - 4.4 Comparison to pseudo-LiDAR Methods
  - 4.5 Ablation & Analysis
- Section 5: Conclusion

---

## Section 1: Introduction

**요약**

카메라 기반 3D 객체 검출은 저비용 자율주행 시스템에서 핵심 과제입니다. 기존 방법들은 크게 두 가지 방향으로 나뉩니다. 첫째, 2D 검출 파이프라인(CenterNet, FCOS)을 그대로 써서 depth를 예측하는 방법. 둘째, 이미지에서 pseudo-LiDAR point cloud를 복원한 뒤 3D 검출기를 적용하는 방법. 두 방식 모두 depth 추정 오류에 취약하고 NMS 같은 후처리를 필요로 합니다.

DETR3D는 이 문제를 top-down 방식으로 접근합니다. 3D 공간에 sparse한 object query 집합을 두고, 각 query를 카메라 변환 행렬로 2D 이미지 특징 맵에 투영하여 필요한 정보만 선택적으로 가져옵니다. DETR에서 영감을 받은 set-to-set loss를 사용해 NMS 없이 end-to-end 학습이 가능합니다.

**핵심 기여**
- **최초의 멀티카메라 3D set-to-set 예측**: 각 카메라 뷰의 정보를 계산의 매 레이어에서 융합합니다.
- **기하학적 back-projection 모듈**: depth 예측 없이 3D → 2D 투영으로 이미지 특징을 직접 수집합니다.
- **NMS-free 추론**: 후처리 없이도 NMS 기반 방법과 동등하거나 우수한 성능을 달성합니다.

---

## Section 2: Related Work

**요약**

2D 객체 검출의 계보(RCNN → Fast RCNN → Faster RCNN → Mask RCNN → SSD → YOLO → CenterNet)와 set 기반 검출(DETR, Deformable DETR)을 정리합니다. 3D 검출에서는 Mono3D, OFT 등의 단안 카메라 방법과, FCOS3D 같은 멀티뷰 확장 방법을 기존 연구로 소개합니다.

**핵심 개념**
- **DETR**: Transformer로 객체 검출을 set prediction 문제로 정식화. Hungarian algorithm으로 예측과 GT를 매칭해 NMS 불필요
- **Deformable DETR**: DETR의 느린 수렴 문제를 deformable self-attention으로 개선
- **pseudo-LiDAR**: 이미지에서 depth map을 예측해 point cloud로 변환 후 LiDAR 기반 검출기 적용. depth 오류가 누적되는 치명적 약점 존재

---

## Section 3: Multi-view 3D Object Detection

### 3.1 Overview

**요약**

DETR3D의 입력은 카메라 intrinsic/extrinsic 행렬이 알려진 멀티뷰 RGB 이미지입니다. 출력은 BEV(Bird's Eye View) 공간에서의 3D 바운딩 박스와 클래스 레이블입니다. 모델은 세 구성요소로 이뤄집니다.

1. **Image Feature Extraction**: ResNet + FPN으로 멀티스케일 2D 특징 추출
2. **Detection Head**: 3D object query를 2D 이미지에 투영해 특징을 수집하고 refinement
3. **Set-to-set Loss**: Hungarian matching으로 예측과 GT를 매칭

설계 원칙:
- 2D 연산이 아닌 3D 공간에서 중간 계산 수행
- dense 3D 씬 복원 없이 작동
- NMS 같은 후처리 배제

### 3.2 Feature Learning

**요약**

입력 이미지 집합 $\mathcal{I} = \{\mathbf{im}_1, \ldots, \mathbf{im}_K\}$를 ResNet과 FPN으로 인코딩해 4개의 멀티스케일 특징 집합 $\mathcal{F}_1, \mathcal{F}_2, \mathcal{F}_3, \mathcal{F}_4$를 추출합니다. 각 특징 $\mathcal{F}_k = \{f_{k1}, \ldots, f_{k6}\} \subset \mathbb{R}^{H \times W \times C}$는 6개 카메라 이미지의 특징입니다.

**핵심 개념**
- **FPN(Feature Pyramid Network)**: 여러 해상도의 특징 맵을 동시에 추출해 다양한 크기의 객체를 포착
- **멀티스케일 특징**: $\frac{1}{8}$, $\frac{1}{16}$, $\frac{1}{32}$, $\frac{1}{64}$ 크기의 특징 맵 4종류 생성

### 3.3 Detection Head

**요약**

DETR3D의 핵심입니다. $L$개의 레이어가 반복적으로 object query를 refinement합니다. 각 레이어 $\ell$은 다음 4단계로 구성됩니다.

1. object query에서 3D reference point 예측
2. reference point를 카메라 변환 행렬로 2D에 투영
3. bilinear interpolation으로 이미지 특징 샘플링
4. multi-head attention으로 query 간 상호작용 모델링

**수식 — Reference Point 예측**

$$\mathbf{c}_{\ell i} = \Phi^{\text{ref}}(\mathbf{q}_{\ell i}) \tag{1}$$

**수식 설명**
- $\mathbf{q}_{\ell i}$: $\ell$번째 레이어의 $i$번째 object query 벡터
- $\mathbf{c}_{\ell i} \in \mathbb{R}^3$: query로부터 예측된 3D reference point (객체 중심의 가설)
- $\Phi^{\text{ref}}$: reference point를 예측하는 신경망

**수식 — 카메라 투영**

$$\mathbf{c}^*_{\ell i} = \mathbf{c}_{\ell i} \oplus 1 \qquad \mathbf{c}_{\ell m i} = T_m \mathbf{c}^*_{\ell i} \tag{2}$$

**수식 설명**
- $\oplus 1$: 3D 좌표를 동차 좌표(homogeneous coordinates)로 변환 — 4D 벡터로 만들어 행렬 곱이 가능하게 함
- $T_m$: $m$번째 카메라의 변환 행렬 (intrinsic × extrinsic)
- $\mathbf{c}_{\ell m i}$: 3D reference point를 $m$번째 카메라 이미지 평면에 투영한 2D 좌표
- 이 단계에서 depth 추정 없이 기하학적으로 3D → 2D 매핑이 이뤄집니다

**수식 — Bilinear Feature Sampling**

$$\mathbf{f}_{\ell k m i} = f^{\text{bilinear}}(\mathcal{F}_{km}, \mathbf{c}_{\ell m i}) \tag{3}$$

**수식 설명**
- $\mathcal{F}_{km}$: $k$번째 스케일, $m$번째 카메라의 특징 맵
- $\mathbf{c}_{\ell m i}$: 투영된 2D 좌표
- $f^{\text{bilinear}}$: 2D 좌표 위치의 특징을 bilinear interpolation으로 샘플링
- 특징 맵은 이산 격자이므로, 실수 좌표에 해당하는 값을 주변 4픽셀의 가중 평균으로 보간합니다

**수식 — 유효 특징 집계 및 Query 업데이트**

$$\mathbf{f}_{\ell i} = \frac{1}{\sum_k \sum_m \sigma_{\ell k m i} + \epsilon} \sum_k \sum_m \mathbf{f}_{\ell k m i} \sigma_{\ell k m i} \qquad \mathbf{q}_{(\ell+1)i} = \mathbf{f}_{\ell i} + \mathbf{q}_{\ell i} \tag{4}$$

**수식 설명**
- $\sigma_{\ell k m i}$: 3D reference point가 $m$번째 카메라 이미지 안에 투영되는지 나타내는 binary mask (이미지 밖이면 0)
- 이미지 밖에 투영된 point의 특징은 무시됩니다
- $\epsilon$: 분모가 0이 되는 것을 방지하는 작은 수
- 유효한 카메라 뷰의 특징을 평균내어 query를 업데이트합니다

**수식 — 최종 예측**

$$\hat{\mathbf{b}}_{\ell i} = \Phi^{\text{reg}}_\ell(\mathbf{q}_{\ell i}) \qquad \hat{c}_{\ell i} = \Phi^{\text{cls}}_\ell(\mathbf{q}_{\ell i}) \tag{5}$$

**수식 설명**
- $\hat{\mathbf{b}}_{\ell i} \in \mathbb{R}^9$: 예측된 3D 바운딩 박스 파라미터 (위치, 크기, heading, velocity)
- $\hat{c}_{\ell i}$: 예측된 클래스 레이블
- $\Phi^{\text{reg}}_\ell$, $\Phi^{\text{cls}}_\ell$: 각 레이어마다 독립된 예측 헤드 (학습 시 모든 레이어의 예측을 감독, 추론 시 마지막 레이어만 사용)

### 3.4 Loss

**요약**

DETR과 동일하게 set-to-set loss를 사용합니다. 예측 집합과 GT 집합을 Hungarian algorithm으로 최적 매칭한 뒤 loss를 계산합니다.

$$\mathcal{L}_{\text{sup}} = \sum_{j=1}^{N} \left[ -\log \hat{p}_{\sigma^*(j)}(c_j) + \mathbf{1}_{\{c_j \neq \varnothing\}} \mathcal{L}_{\text{box}}(\mathbf{b}_j, \hat{\mathbf{b}}_{\sigma^*(j)}) \right]$$

**수식 설명**
- $\sigma^* = \arg\min_{\sigma \in \mathcal{P}} \sum_{j=1}^{M}$ ...: Hungarian algorithm으로 구한 최적 예측-GT 매칭
- $-\log \hat{p}_{\sigma^*(j)}(c_j)$: 매칭된 예측의 클래스 확률에 대한 focal loss
- $\mathcal{L}_{\text{box}}$: 바운딩 박스 파라미터에 대한 $L^1$ loss
- $\mathbf{1}_{\{c_j \neq \varnothing\}}$: GT가 실제 객체일 때만 box loss 계산 (패딩된 "no object"는 분류 loss만)
- GT 박스 수 $M$이 예측 수 $M^*$보다 적으므로, GT를 $\varnothing$으로 패딩해 매칭

---

## Section 4: Experiments

### 4.1 Implementation Details

**데이터셋**: nuScenes — 1,000 시퀀스, 6개 카메라(front, front_left, front_right, back_left, back, back_right), 28k 훈련 / 6k 검증 / 6k 테스트 샘플

**평가 지표**:
- **NDS(nuScenes Detection Score)**: $\frac{1}{10}[5\text{mAP} + \sum_{\text{mTP} \in \text{TP}}(1 - \min(1, \text{mTP}))]$ — 종합 지표
- **mAP**: mean Average Precision
- **mATE/mASE/mAOE/mAVE/mAAE**: 위치/크기/방향/속도/속성 오류 (낮을수록 좋음)

**모델 구성**:
- Backbone: ResNet101 (deformable convolution, 3·4번째 stage)
- Neck: FPN → 4개 스케일 특징 맵
- Detection Head: 6 레이어, hidden dim 256, object query 900개

### 4.2~4.4 실험 결과 요약

| 비교 대상 | 결과 |
|---|---|
| CenterNet, FCOS3D (NMS 사용) | DETR3D가 NMS 없이도 동등하거나 우수 |
| 카메라 오버랩 영역 | DETR3D가 FCOS3D 대비 NDS에서 현저히 우수 (멀티뷰 정보를 동시에 활용하기 때문) |
| pseudo-LiDAR | DETR3D가 NDS 0.374 vs. 0.160으로 압도 |
| 테스트셋 SOTA (2021.10 기준) | NDS 0.479로 1위 |

### 4.5 Ablation & Analysis

- **레이어 수**: 레이어가 깊어질수록 바운딩 박스 예측이 GT에 수렴 (Figure 2)
- **Query 수**: 30 → 900개까지 성능 향상, 이후 포화
- **Backbone**: ResNet101 > ResNet50 > DLA34

---

## 핵심 개념 정리

- **3D-to-2D Query**: 3D 공간의 object query를 카메라 행렬로 2D에 투영해 이미지 특징을 수집하는 핵심 아이디어. depth 추정 없이 3D 정보를 2D 특징과 연결합니다.
- **Iterative Refinement**: L개 레이어가 반복적으로 object query를 정제. 각 레이어마다 예측을 감독(auxiliary loss)해 학습 안정성을 높입니다.
- **Set-to-set Loss**: Hungarian matching으로 예측과 GT를 1:1 매칭 후 loss 계산. NMS 없이 end-to-end 학습이 가능한 이유입니다.
- **Visibility Mask ($\sigma$)**: 3D point가 카메라 이미지 밖에 투영될 경우 해당 특징을 무시하는 binary mask. 유효한 뷰의 정보만 집계합니다.
- **BEV(Bird's Eye View)**: 자율주행에서 객체의 위치를 위에서 내려다본 2D 평면으로 표현하는 방식. 바운딩 박스 위치·크기·heading이 BEV 기준으로 정의됩니다.

---

## 결론 및 시사점

DETR3D는 depth 추정과 NMS라는 두 가지 병목을 동시에 제거한 멀티뷰 3D 객체 검출 프레임워크입니다. 3D object query를 카메라 행렬로 직접 2D에 투영하는 아이디어는 이후 BEVFormer, PETR 등 BEV 기반 검출 방법론의 핵심 설계 원칙으로 이어집니다.

**실무적 시사점**:
- 카메라 오버랩 영역에서 특히 효과적 — 여러 뷰의 정보를 동시에 활용하기 때문
- NMS-free 설계는 실시간 추론에 유리
- 여전히 translation error(mATE)가 높은 편 — depth 정보 없이 위치를 정확히 추정하는 것은 근본적인 한계

**후속 연구로의 연결**:
- **BEVFormer**: 명시적 BEV 특징 맵을 만들어 DETR3D의 sparse query 방식을 dense로 확장
- **PETR**: position embedding으로 3D 위치 정보를 특징에 인코딩해 query 표현력 강화
