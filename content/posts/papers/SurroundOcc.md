---
title: "SurroundOcc: Multi-Camera 3D Occupancy Prediction for Autonomous Driving"
date: 2026-04-19T10:00:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving", "3D Occupancy"]
tags: ["3D Occupancy", "Autonomous Driving", "Multi-Camera", "Scene Understanding"]
---

## 개요
- **저자**: Yi Wei, Linqing Zhao, Wenzhao Zheng, Zheng Zhu, Jie Zhou, Jiwen Lu
- **소속**: Tsinghua University, Tianjin University, PhiGent Robotics
- **arXiv**: 2303.09551v2 (2023)
- **주요 내용**: 멀티카메라 RGB 이미지만으로 주변 3D 공간의 밀집 시맨틱 점유(occupancy)를 예측하는 SurroundOcc 프레임워크 제안. 2D-3D 공간 어텐션, 멀티스케일 3D 볼륨 특징, Poisson Reconstruction 기반 밀집 GT 생성 파이프라인이 핵심.

## 한계 극복

이 논문이 기존 연구의 어떤 한계를 극복하기 위해 작성되었는지 설명합니다.

- **기존 한계 1 — 3D 객체 검출의 불완전성**: 기존 3D 검출 방법은 bounding box 기반으로, 임의의 형태나 무한한 클래스의 객체를 표현하기 어렵습니다.
- **기존 한계 2 — Depth map의 한계**: depth-based 방법은 각 광선에서 가장 가까운 점만 예측하므로, 가려진(occluded) 영역을 복원할 수 없습니다.
- **기존 한계 3 — BEV 표현의 정보 손실**: BEV(Bird's Eye View) 특징은 높이 방향 정보를 압축하여 3D 공간 정보를 손실합니다.
- **기존 한계 4 — 희소한 GT 레이블**: nuScenes 같은 데이터셋은 단일 프레임 희소 LiDAR 포인트만 제공하여, 밀집 점유 예측 학습에 부적합합니다.
- **이 논문의 접근 방식**: 2D-3D 공간 어텐션으로 BEV 대신 3D 볼륨 쿼리를 구성하고, 멀티프레임 LiDAR + Poisson Reconstruction으로 밀집 GT를 자동 생성하여 위 한계들을 동시에 해결합니다.

## 목차

- Section 1: Introduction
- Section 2: Related Work
- Section 3: Approach (2D-3D Spatial Attention, Multi-scale Occupancy Prediction)
- Section 4: Dense Occupancy Ground Truth Generation
- Section 5: Experiments
- Section 6: Limitations and Future Work
- Section 7: Conclusion

---

## Section 1: Introduction

**요약**

자율주행 시스템에서 주변 3D 공간을 이해하는 것은 필수적입니다. LiDAR는 정확한 기하 정보를 제공하지만 고가이고 희소하여 실용성이 제한됩니다. 최근 비전 기반 자율주행이 주목받으며, 멀티카메라 이미지로 3D 장면을 재구성하는 연구가 활발해졌습니다.

기존 depth map 방법은 각 픽셀에서 가장 가까운 표면만 예측하므로 가려진 영역을 복원하지 못합니다. 반면 3D 점유(occupancy)는 공간의 모든 복셀(voxel)에 점유 확률을 부여하여 가려진 영역까지 표현 가능합니다. SurroundOcc는 멀티카메라 이미지를 입력받아 밀집하고 정확한 3D 시맨틱 점유를 예측합니다.

**핵심 개념**

- **3D Occupancy (3D 점유)**: 3D 공간을 격자(복셀)로 나누고 각 격자가 차지되어 있는지(occupied) 여부와 그 의미(semantic class)를 예측하는 표현 방식. depth map보다 포괄적이고 다운스트림 태스크(분할, 흐름 추정)에 유연하게 활용 가능
- **Multi-camera Vision**: LiDAR 없이 여러 대의 RGB 카메라만으로 3D 공간을 추론하는 비전 중심(vision-centric) 접근법
- **BEV vs. Volumetric**: BEV(새의 눈 시점)는 2D 평면에 3D 정보를 투영하여 높이 정보를 잃는 반면, 볼류메트릭(volumetric) 표현은 전체 3D 구조를 보존

---

## Section 2: Related Work

**요약**

관련 연구는 세 가지 흐름으로 나뉩니다.

1. **Voxel 기반 장면 표현**: LiDAR 분할 등에서 성공한 복셀 표현을 실외 장면 재구성에 적용. MonoScene(단안 카메라), TPVFormer(멀티카메라) 등이 선행 연구.
2. **3D 장면 재구성**: depth estimation → 3D 공간 투영 방식(BEVFormer 등) 또는 직접 3D 특징 학습 방식(SurfaceNet, Atlas, TransformerFusion 등). 대부분 실내 환경 대상.
3. **비전 기반 3D 인식**: BEVFormer처럼 2D 이미지 특징을 BEV 그리드로 변환하는 방법이 주류. SurroundOcc는 BEV 대신 완전한 3D 볼륨을 사용하여 정보 손실을 방지.

**핵심 개념**

- **MonoScene**: 단안 카메라 이미지로 3D 시맨틱 장면을 재구성하는 최초 시도 중 하나
- **TPVFormer**: 희소 LiDAR를 supervision으로 사용하는 멀티카메라 3D 시맨틱 점유 예측. 희소 GT로 인해 예측도 희소함
- **BEVFormer**: cross-attention으로 2D 특징을 BEV 그리드로 점진적으로 정제하는 대표적 BEV 방법

---

## Section 3: Approach

**요약**

SurroundOcc의 전체 파이프라인은 세 단계로 구성됩니다.

1. **2D 백본(backbone)**: ResNet-101-DCN으로 N개 카메라의 멀티스케일 2D 특징 추출
2. **2D-3D 공간 어텐션**: 각 3D 복셀 쿼리를 2D 뷰에 투영하여 deformable attention으로 특징 집계 → 3D 볼륨 특징 구성
3. **멀티스케일 3D U-Net**: 3D convolution으로 저해상도 볼륨을 점진적으로 업샘플하고, 각 레벨에서 점유 예측에 decayed loss 적용

**핵심 개념**

- **2D-3D Spatial Attention**: BEV 쿼리 대신 3D 볼륨 쿼리 $Q \in \mathbb{R}^{C \times H \times W \times Z}$ 를 정의하고, 각 쿼리의 3D 기준점을 카메라 내/외부 파라미터로 2D에 투영하여 샘플링

- **Cross-view Attention**: 동일 3D 위치에 대해 여러 카메라에서 다르게 기여하는 정보를 deformable attention으로 통합. BEV 어텐션과 달리 가려짐(occlusion)이나 흐림이 있는 뷰의 가중치를 자동 조절

- **Multi-scale 3D U-Net**: 여러 해상도에서 3D 볼륨 특징 $\{F_j\}_{j=1}^M$을 추출하고, 낮은 해상도 특징을 3D deconv로 업샘플하여 상위 레벨 특징과 합산:

**수식 — Deformable Attention**

$$\text{DeformAttn}(q, p, x) = \sum_{i=1}^{N_\text{head}} \mathcal{W}_i \sum_{j=1}^{N_\text{key}} \mathcal{A}_{ij} \cdot \mathcal{W}'_i x(p + \Delta p_{ij})$$

**수식 설명**

이 수식은 3D 볼륨 쿼리가 여러 카메라 이미지에서 관련 특징을 선택적으로 집계하는 방법을 나타냅니다:
- **$q$**: 3D 볼륨 쿼리 벡터 (어디에 있는 복셀인지 나타내는 질문)
- **$p$**: 3D 기준점을 2D 이미지에 투영한 2D 위치
- **$\Delta p_{ij}$**: $i$번째 헤드, $j$번째 샘플링 포인트의 오프셋 (기준점에서 얼마나 이동하여 샘플링할지)
- **$\mathcal{A}_{ij}$**: 어텐션 가중치 (쿼리와 키의 내적으로 계산, 중요한 위치에 더 큰 가중치 부여)
- **$\mathcal{W}_i, \mathcal{W}'_i$**: 학습 가능한 projection 행렬
- **$N_\text{head}$**: 멀티헤드 어텐션의 헤드 수 (여러 관점에서 병렬로 어텐션 계산)
- **직관**: "이 복셀을 설명하는 2D 특징이 이미지 어디에 있는지" 를 네트워크가 스스로 학습하여 찾아냄

**수식 — Multi-scale Upsampling**

$$Y_j = F_j + \text{Deconv}(Y_{j-1})$$

**수식 설명**

3D U-Net에서 저해상도 예측을 고해상도로 점진적으로 정제하는 수식입니다:
- **$Y_j$**: $j$ 레벨의 최종 볼륨 특징 (고해상도에 가까울수록 $j$가 큼)
- **$F_j$**: 2D-3D 어텐션으로 추출한 $j$ 레벨의 원본 볼륨 특징
- **$\text{Deconv}(Y_{j-1})$**: 아래 레벨(더 저해상도)의 출력을 3D deconvolution으로 2배 업샘플한 것
- **직관**: skip connection처럼 원본 특징을 보존하면서 전역 맥락(저해상도)과 지역 세부 정보(고해상도)를 모두 학습

**수식 — Decayed Loss Weight**

$$\alpha_j = \frac{1}{2^j}$$

**수식 설명**

멀티스케일 supervision에서 각 레벨의 loss 가중치를 정하는 수식입니다:
- **$\alpha_j$**: $j$ 레벨(낮을수록 저해상도)의 loss 가중치
- **직관**: 최종 고해상도 예측($j$가 작음)이 가장 중요하므로 가중치를 크게, 중간 레벨은 보조 supervision 역할

---

## Section 4: Dense Occupancy Ground Truth Generation

**요약**

nuScenes 데이터셋은 희소 LiDAR 포인트만 제공합니다. 희소 GT로 학습하면 모델도 희소한 점유만 예측합니다. SurroundOcc는 다음 4단계로 밀집 GT를 자동 생성합니다:

1. **멀티프레임 포인트 클라우드 스티칭**: 동적 객체(차량 등)와 정적 장면을 분리하여 각각 누적. 객체는 bounding box로 분리하고 전역 좌표계로 변환 후 재합성
2. **Poisson Surface Reconstruction**: 밀집화된 포인트 클라우드의 법선 벡터를 추정하고 Poisson 재구성으로 삼각형 메쉬 $\mathcal{M} = \{\mathcal{V}, \mathcal{E}\}$ 생성 → 구멍(hole) 채움
3. **복셀화(Voxelization)**: 메쉬를 밀집 복셀 $V_d$로 변환
4. **NN 시맨틱 레이블링**: 희소 포인트 $V_s$의 시맨틱 레이블을 밀집 복셀 $V_d$에 최근접 이웃(NN)으로 전파

**핵심 개념**

- **두 스트림 스티칭**: 움직이는 객체(동적)와 고정 환경(정적)을 별도로 누적하지 않으면, 동적 객체의 잔상(ghost)이 장면에 남아 잘못된 GT가 생성됨
- **Poisson Surface Reconstruction**: 포인트 클라우드의 법선 정보를 이용해 연속적인 표면을 재구성하는 알고리즘. 구멍 채움과 균일한 정점 분포가 특징
- **NN 시맨틱 전파**: 기하(geometry)와 의미(semantics) 복원을 분리. 기하는 Poisson으로 밀집화하고, 의미는 가장 가까운 기존 포인트의 레이블을 상속

---

## Section 5: Experiments

**요약**

nuScenes(멀티카메라)와 SemanticKITTI(단안 카메라 제로샷) 두 벤치마크에서 평가했습니다.

**nuScenes 3D 시맨틱 점유 예측 (Table 1)**

| Method | SC IoU | SSC mIoU |
|--------|--------|----------|
| MonoScene | 23.96 | 7.31 |
| BEVFormer | 30.50 | 16.75 |
| TPVFormer* | 30.86 | 17.10 |
| **SurroundOcc** | **31.49** | **20.30** |

- SC(Scene Completion) IoU: 점유 여부만 평가
- SSC(Semantic Scene Completion) mIoU: 시맨틱 클래스별 평균 IoU

**nuScenes 3D 장면 재구성 (Table 4)**

| Method | CD ↓ | F-score ↑ |
|--------|------|----------|
| Atlas | 2.163 | 0.257 |
| TransformerFusion | 0.771 | 0.453 |
| **SurroundOcc** | **0.724** | **0.483** |

**모델 효율성 (Table 8, RTX 3090 기준)**

| Method | Latency(s) | Memory(G) |
|--------|-----------|----------|
| BEVFormer | **0.31** | **4.5** |
| TPVFormer | 0.32 | 5.1 |
| **SurroundOcc** | 0.34 | 5.9 |

SurroundOcc는 BEVFormer 대비 latency 10% 증가로 성능을 크게 향상시킵니다.

**Ablation Study 핵심 결과**

- 2D-3D 공간 어텐션 제거 시 SSC mIoU 17.34로 하락 (vs. 20.30)
- BEV 기반 어텐션 사용 시 SSC mIoU 18.94로 하락
- 희소 LiDAR GT로만 학습 시 SSC mIoU 12.17 (vs. 밀집 GT 사용 시 20.30)

**핵심 개념**

- **IoU (Intersection over Union)**: 예측과 정답이 겹치는 비율. $\text{IoU} = \frac{TP}{TP + FP + FN}$
- **mIoU**: 모든 클래스의 IoU 평균. 클래스 불균형에 공정한 평가 지표
- **Chamfer Distance (CD)**: 예측 포인트 클라우드와 GT 포인트 클라우드 간 최근접 거리의 평균. 낮을수록 좋음
- **F-score**: Precision과 Recall의 조화 평균. 재구성 정확도 평가

**수식 — IoU & mIoU**

$$\text{IoU} = \frac{TP}{TP + FP + FN}$$

$$\text{mIoU} = \frac{1}{C} \sum_{i=1}^{C} \frac{TP_i}{TP_i + FP_i + FN_i}$$

**수식 설명**
- **$TP$**: True Positive — 점유했다고 예측했고 실제로도 점유된 복셀 수
- **$FP$**: False Positive — 점유했다고 예측했지만 실제로는 비어있는 복셀 수
- **$FN$**: False Negative — 비어있다고 예측했지만 실제로는 점유된 복셀 수
- **$C$**: 클래스 수 (nuScenes: 17 클래스)
- **직관**: IoU = "내가 맞혔다고 주장하는 것" 중 실제로 맞은 비율. 분모에 FP와 FN이 모두 포함되어 엄격한 지표

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **3D Occupancy** | 3D 공간을 복셀 격자로 표현하고 각 복셀의 점유 여부와 시맨틱 클래스를 예측 |
| **2D-3D Spatial Attention** | 3D 볼륨 쿼리를 2D 카메라 뷰에 투영하여 deformable attention으로 멀티뷰 특징 집계 |
| **Cross-view Attention** | 여러 카메라에서 동일 3D 위치의 특징을 가중 통합. 가려진 뷰의 가중치를 자동 감소 |
| **Multi-scale 3D U-Net** | 저해상도에서 고해상도로 3D 볼륨을 점진적 업샘플. 각 레벨에서 decayed loss로 supervision |
| **Poisson Reconstruction** | 포인트 클라우드에서 연속 표면 메쉬 생성. 구멍 채움에 효과적 |
| **두 스트림 GT 생성** | 동적 객체와 정적 장면을 분리 누적하여 잔상 없는 밀집 GT 생성 |
| **Dense Occupancy Supervision** | 희소 LiDAR 대신 밀집 GT로 학습하면 SSC mIoU가 12.17 → 20.30으로 대폭 향상 |

---

## 결론 및 시사점

**결론**

SurroundOcc는 멀티카메라 이미지만으로 밀집하고 정확한 3D 시맨틱 점유를 예측하는 방법을 제시합니다. 핵심 기여는 두 가지입니다:

1. **방법론**: 2D-3D 공간 어텐션으로 BEV 대신 3D 볼륨 특징을 구성하고, 멀티스케일 3D U-Net으로 고해상도 예측
2. **데이터**: 멀티프레임 LiDAR 스티칭 + Poisson Reconstruction으로 비용 없이 밀집 GT 자동 생성

nuScenes와 SemanticKITTI 벤치마크에서 기존 방법 대비 큰 폭의 성능 향상을 달성했습니다.

**실무적 시사점**

- **자율주행 인식 스택**: occupancy 기반 표현은 detection box보다 임의 형태 객체(공사 구간, 이상 물체)에 강인하여 안전 마진 계산에 유리
- **합성 데이터 생성**: 밀집 GT 생성 파이프라인(멀티프레임 스티칭 + Poisson Reconstruction)은 annotation 비용 없이 고품질 3D 레이블을 생성하는 실용적 방법으로, 회귀 테스트용 합성 데이터 생성에도 응용 가능
- **한계**: 단일 프레임 입력만 처리하므로 시간적 연속성(occupancy flow)이 없어 motion prediction에 직접 활용 어려움. 향후 멀티프레임 입력으로 확장 예정


---

*관련 논문: [MonoScene](/posts/papers/monoscene-monocular-3d-semantic-scene-completion/), [TPVFormer](/posts/papers/tpvformer-tri-perspective-view-3d-semantic-occupancy/), [Occ3D](/posts/papers/occ3d-large-scale-3d-occupancy-prediction-benchmark/), [BEVFormer](/posts/papers/BEVFormer/), [GaussianWorld](/posts/papers/gaussianworld-gaussian-world-model-for-streaming-3d-occupancy-prediction/)*
