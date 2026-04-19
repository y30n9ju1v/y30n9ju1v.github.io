---
title: "OccFormer: Dual-path Transformer for Vision-based 3D Semantic Occupancy Prediction"
date: 2026-04-19T22:58:00+09:00
draft: false
categories: ["Papers"]
---

## 개요
- **저자**: Yunpeng Zhang, Zheng Zhu, Dalong Du (PhiGent Robotics)
- **발행년도**: 2023 (arXiv 2304.05316)
- **주요 내용**: 멀티뷰 카메라 이미지만으로 3D 의미론적 점유(semantic occupancy)를 예측하는 듀얼패스 Transformer 네트워크. 3D 특징 볼륨을 수평 슬라이스(local)와 BEV 붕괴(global) 두 경로로 병렬 처리해 장거리·동적·효율적 인코딩을 달성하고, Mask2Former를 3D 점유 디코더로 최초 적용한다.

## 한계 극복

- **기존 한계 1 — 3D Conv의 파라미터 과다**: 3D Conv는 3D 특징 볼륨을 직접 처리하므로 파라미터와 연산량이 크게 증가하며, 수용 영역(receptive field)이 고정되어 다양한 크기의 객체에 취약.
- **기존 한계 2 — MonoScene의 공간 불연속성**: MonoScene은 sight projection으로 2D→3D 변환 후 3D UNet 적용. 카메라로 생성된 3D 특징은 희소하고 불연속적이어서 3D Conv가 효과적으로 처리하지 못함.
- **기존 한계 3 — TPVFormer의 세밀한 의미 손실**: 세 직교 평면으로 BEV를 표현하면 높이 방향의 세밀한 의미 정보가 손실되어 성능 한계 존재.
- **기존 한계 4 — 희소성·클래스 불균형**: 3D 점유 공간은 2D 이미지보다 훨씬 희소하고 클래스 분포가 극단적으로 불균형하여 표준 Mask2Former 학습 방식 부적합.
- **이 논문의 접근 방식**: Local(슬라이스별 윈도우 어텐션) + Global(BEV 붕괴 후 ASPP) 듀얼패스로 3D를 효율적으로 처리. Preserve-pooling(MaxPool 다운샘플)과 Class-guided Sampling으로 희소성·불균형 해소.

## 목차
- Chapter 1: Introduction
- Chapter 2: Related Work
- Chapter 3: Approach (3.1 Overview / 3.2 Dual-path Transformer Encoder / 3.3 Transformer Occupancy Decoder)
- Chapter 4: Experiments
- Chapter 5: Conclusion

## Chapter 1: Introduction

**요약**

자율주행 인식은 BEV(Bird's-Eye-View) 표현에서 3D 시맨틱 점유로 진화하고 있다. BEV는 높이 방향 정보를 소실하는 반면, 3D 점유는 복셀 단위로 의미 + 점유를 동시에 예측해 도로 위 장애물을 완전히 표현한다.

기존 카메라 기반 3D 점유 방법(MonoScene, TPVFormer)은 3D Conv의 고정 수용 영역이나 tri-plane의 의미 손실 문제를 갖는다. OccFormer는 Transformer의 Self-attention을 3D 볼륨에 효율적으로 적용하는 듀얼패스 구조로 이를 해결한다.

**핵심 개념**
- **3D Semantic Occupancy**: 공간을 X×Y×Z 복셀 격자로 분할하고 각 복셀에 의미 클래스 레이블 부여. 사전 정의된 객체 카테고리(out-of-vocabulary 포함)를 모두 표현 가능.
- **듀얼패스(Dual-path)**: 3D 특징 볼륨을 Local(슬라이스) + Global(BEV) 두 경로로 분리 처리해 세밀한 구조와 장면 수준 레이아웃을 동시에 학습.

## Chapter 2: Related Work

**요약**

- **카메라 기반 BEV 인식**: LSS, BEVDet, BEVDepth, BEVFormer 등이 2D→BEV 변환을 개척. OccFormer는 BEV를 넘어 3D 볼륨으로 확장.
- **3D SSC(Semantic Scene Completion)**: SSCNet이 기하+의미 동시 추론을 제안. MonoScene이 단안 카메라 기반 최초 방법. TPVFormer가 tri-plane으로 카메라 전용 3D 점유 예측 시도.
- **효율적 3D 네트워크**: EsscNet(희소 그룹 Conv), DDRNet(1D Conv 분해), LMSCNet(2D UNet + 높이 복원). OccFormer는 Transformer 기반으로 이들을 대체.

## Chapter 3: Approach

### 3.1 전체 파이프라인 개요

**요약**

OccFormer의 파이프라인은 세 단계로 구성된다:

1. **Image Encoder**: 멀티뷰 카메라 이미지 → 멀티스케일 2D 피처 추출 (EfficientNetB7 / ResNet-101)
2. **Image-to-3D Transformation**: LSS 패러다임 적용. 깊이 분포(Depth Distribution) × 컨텍스트 피처 = 포인트 클라우드 → Voxel Pooling → 3D 피처 볼륨 $\mathbf{F}^{3d} \in \mathbb{R}^{C_{con} \times X \times Y \times Z}$
3. **Dual-path Transformer Encoder + Transformer Occupancy Decoder**: 3D 볼륨 인코딩 → 클래스별 마스크 예측

**핵심 개념**
- **LSS (Lift-Splat-Shoot)**: 픽셀별 깊이 분포를 예측해 2D 피처를 3D 포인트로 "Lift"하고, 복셀 풀링으로 BEV/3D 특징 생성
- **3D Feature Volume**: $128 \times 128 \times 16$ 해상도, 128채널. 이후 듀얼패스 인코더로 처리

### 3.2 Dual-path Transformer Encoder

**요약**

3D 특징 볼륨을 효율적으로 처리하기 위한 핵심 모듈. Local path와 Global path가 병렬 동작 후 가중합으로 결합된다.

**Local Path — 슬라이스별 윈도우 어텐션**:
- 높이 차원을 배치 차원으로 병합(merge)해 Z개의 독립적인 BEV 슬라이스로 처리
- 각 슬라이스에 **공유 가중치** 윈도우 셀프 어텐션(Swin Transformer의 Shifted Window Attention) 적용
- 수평 방향의 세밀한 의미 구조(fine-grained semantic structures) 포착
- 파라미터를 모든 슬라이스가 공유해 파라미터 효율 극대화

**Global Path — BEV 붕괴 + ASPP**:
- 높이 차원을 평균 풀링으로 붕괴(collapse)해 BEV 피처 생성
- Local path와 같은 공유 윈도우 어텐션으로 이웃 의미 처리
- **ASPP(Atrous Spatial Pyramid Pooling)**: 병목(bottleneck) 구조로 채널 4× 축소 후 다중 팽창률(dilated conv)로 장거리 전역 맥락 수집
- 장면 수준 의미 레이아웃(scene-level semantic layout) 포착

**핵심 수식 — 듀얼패스 융합**

$$\mathbf{F}_{out} = \mathbf{F}_{local} + \sigma(\mathbf{W}\mathbf{F}_{local}) \cdot \text{unsqueeze}(\mathbf{F}_{global}, -1)$$

**수식 설명**
- **$\mathbf{F}_{local} \in \mathbb{R}^{C \times X \times Y \times Z}$**: Local path 출력 (3D 전체 볼륨)
- **$\mathbf{F}_{global} \in \mathbb{R}^{C \times X \times Y}$**: Global path 출력 (높이 차원 붕괴된 BEV 피처)
- **$\mathbf{W}$**: FFN(Feed-Forward Network) — Local 피처로부터 높이 방향 집계 가중치 생성
- **$\sigma(\cdot)$**: Sigmoid 함수 — 가중치를 [0, 1]로 정규화
- **$\text{unsqueeze}(\mathbf{F}_{global}, -1)$**: BEV 피처를 높이 차원으로 브로드캐스트
- **직관**: Global path가 "어느 높이에 무엇이 있는지" 결정하는 가중치를 Local path에 덧붙여 3D 복원

### 3.3 Transformer Occupancy Decoder

**요약**

Mask2Former를 3D 점유 예측에 최초로 적용. 픽셀(복셀) 디코더로 per-voxel 임베딩을 생성하고, Transformer 디코더로 클래스별 쿼리를 이터레이티브하게 업데이트해 최종 3D 시맨틱 마스크를 출력한다.

#### 3.3.1 Pixel Decoder (픽셀/복셀 디코더)

멀티스케일 3D 피처를 입력으로 받아 per-voxel 임베딩 $\mathcal{E}_{voxel} \in \mathbb{R}^{C_\mathcal{E} \times X \times Y \times Z}$ 생성.

**수식 — 멀티스케일 3D 변형 가능 어텐션(Deformable Attention) 업데이트**

$$\mathbf{F}_i^{3d} = \mathbf{F}_i^{3d} + \sum_{j=1}^{N_l} \left[\mathbf{W}_j^{3d} \mathbf{F}_j^{3d}\left(\mathbf{P}_i^{3d} + \mathbf{\Delta}_j^{3d}\right)\right]$$

**수식 설명**
- **$\mathbf{F}_i^{3d}$**: i번째 레벨의 3D 피처 볼륨
- **$\mathbf{P}_i^{3d} \in \mathbb{R}^{X_i \times Y_i \times Z_i \times 3}$**: 각 복셀의 실세계 3D 좌표
- **$\mathbf{\Delta}_j^{3d}$**: j번째 레벨에서 예측한 오프셋 (어디를 볼지 동적으로 조정)
- **$\mathbf{W}_j^{3d}$**: j번째 레벨 어텐션 가중치
- **직관**: 고정된 격자 대신 각 복셀이 "중요한 위치"를 동적으로 선택해 참조 → 다중 스케일 의미 집계

#### 3.3.2 Transformer Decoder (쿼리 기반 디코더)

클래스별 쿼리 피처 $\mathbf{Q}$를 이터레이티브하게 업데이트.

**수식 — 마스크 어텐션**

$$\mathbf{Q}_{l+1} = \text{softmax}\left[\mathcal{M}_{l-1} + \mathbf{W}_q \mathbf{Q}_l \left(\mathbf{W}_k \mathbf{F}_l^{3d}\right)^T\right] \mathbf{W}_v \mathbf{F}_l^{3d} + \mathbf{Q}_l$$

**수식 설명**
- **$\mathbf{Q}_l$**: l번째 이터레이션의 쿼리 피처 (클래스별 표현)
- **$\mathbf{F}_l^{3d}$**: l번째 레이어의 3D 복셀 피처
- **$\mathcal{M}_{l-1}$**: 이전 이터레이션에서 예측된 어텐션 마스크 (foreground 영역만 참조)
- **$\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v$**: Query/Key/Value 선형 투영 행렬
- **직관**: 쿼리가 이전에 찾은 마스크 영역 내에서만 어텐션 → 전역 어텐션보다 효율적이고 안정적

**최종 예측**

$$\mathbf{Y} = \sum_{i=1}^{N_q} \mathbf{p}_i \cdot \mathbf{M}_i$$

- **$\mathbf{p}_i$**: i번째 쿼리의 의미 로짓 (클래스 확률)
- **$\mathbf{M}_i$**: i번째 쿼리의 이진 3D 마스크 (per-voxel 임베딩과의 dot product + sigmoid)
- **$N_q$**: 전체 쿼리 수
- **직관**: 각 쿼리가 하나의 "클래스 세그먼트"를 담당하고, 이를 합산해 최종 3D 시맨틱 점유 생성

#### 3.3.3 Preserve-Pooling

Mask2Former의 기본 다운샘플링은 **bilinear interpolation**이지만, 3D 점유는 희소하고 불연속적이라 bilinear가 소수(minority) 클래스 구조를 제거할 수 있다.

→ **MaxPooling으로 대체**: 인접 복셀 중 "가장 강한 활성화"를 보존해 희소 전경 객체(차량, 보행자) 구조 유지.

ablation에서 bilinear → MaxPool 전환만으로 mIoU +0.5 향상.

#### 3.3.4 Class-Guided Sampling

**배경**: Mask2Former는 균일 샘플링으로 K개 지점을 선택해 매칭 비용과 손실을 계산. 3D 점유에서는 배경 복셀이 압도적이어서 희소 클래스(트럭, 보행자, 교통 표지판 등)의 supervision이 부족.

**방법**: 클래스별 빈도 $\mathbf{n}_c \in \mathbb{R}^{N_c}$의 역수로 샘플링 가중치 계산:

$$\mathbf{w}_c = \frac{1}{\mathbf{n}_c}, \quad \mathbf{w}_c \leftarrow \frac{\mathbf{w}_c}{\min(\mathbf{w}_c)}, \quad \text{sampling weight} = (\mathbf{w}_c)^\beta$$

- $\beta$: 하이퍼파라미터 (OccFormer에서 0.25 사용)
- 역수 가중치이므로 희소 클래스일수록 샘플링 확률 증가
- nuScenes의 경우 LiDAR 포인트와 랜덤 좌표를 1:1 비율로 혼합

ablation에서 uniform → class-guided 전환으로 SemanticKITTI mIoU 12.13 → 13.46 (+1.33 향상).

### 3.4 Loss Functions

$$\mathcal{L} = \mathcal{L}_{\text{mask-cls}} + \mathcal{L}_{\text{depth}}$$

- **$\mathcal{L}_{\text{mask-cls}}$**: Hungarian matching 기반 이중 파티셔닝 매칭 후 클래스 손실 + 이진 마스크 손실 (Mask2Former 방식)
- **$\mathcal{L}_{\text{depth}}$**: BEVDepth를 따른 깊이 분포 감독 (이진 cross-entropy)

## Chapter 4: Experiments

**요약**

두 데이터셋에서 검증한다: SemanticKITTI(SSC 태스크), nuScenes(LiDAR 시맨틱 분할 태스크).

### 4.1 Semantic Scene Completion — SemanticKITTI

- **데이터**: KITTI Odometry 기반, 256×256×32 복셀 (각 복셀 0.2m×0.2m×0.2m), 21개 의미 클래스
- **입력**: 단안 카메라(left camera), 백본: EfficientNetB7

| 방법 | 입력 | SC IoU | SSC mIoU |
|------|------|------|------|
| MonoScene | Camera | 34.16 | 11.08 |
| TPVFormer | Camera | 34.25 | 11.26 |
| **OccFormer (ours)** | **Camera** | **34.53** | **12.32** |

- MonoScene 대비 mIoU **+1.24** (11% 상대 향상)
- 모노큘러 방법 중 **test leaderboard 1위**

### 4.2 LiDAR Semantic Segmentation — nuScenes

- **데이터**: 1000개 주행 시퀀스, 700/150/150 train/val/test 분할
- 3D 점유를 LiDAR 포인트에 쿼리해 시맨틱 레이블 획득 (LiDAR 직접 사용 ×)
- **입력**: 멀티뷰 카메라, 백본 R50(704×256) / R101(1600×900)

| 방법 | 입력 | Backbone | mIoU |
|------|------|------|------|
| TPVFormer | Camera | R50 | 59.3 |
| **OccFormer** | **Camera** | **R50** | **68.1** |
| TPVFormer | Camera | R101 | 68.9 |
| **OccFormer** | **Camera** | **R101** | **70.4** |

- TPVFormer(R50) 대비 **+8.8 mIoU** 향상
- 카메라 전용으로 **70%+ mIoU 최초 달성**
- LiDAR 기반 방법(Cylinder3D++: 76.1 mIoU)에 근접

### 4.3 Ablation Studies

**듀얼패스 인코더 ablation (SemanticKITTI val)**

| Local | Global | Params | GFLOPs | IoU | mIoU |
|-------|--------|--------|--------|-----|------|
| ✓ | | 74.1M | 494.2 | 36.42 | 12.95 |
| | ✓ | 81.4M | 407.4 | 36.37 | 12.93 |
| ✓ | ✓ | 81.4M | 515.3 | **36.50** | **13.46** |

두 경로 모두 필요하며, 조합 시 시너지 효과 발생.

**Transformer Decoder ablation**

| Resize 방법 | Sampling 방법 | IoU | mIoU |
|-------------|---------------|-----|------|
| Tri-linear | Uniform | 35.04 | 11.61 |
| Max-pool | Uniform | 35.41 | 12.13 |
| Tri-linear | Class-guided | 36.21 | 13.01 |
| **Max-pool** | **Class-guided** | **36.50** | **13.46** |

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **Dual-path Transformer Block** | 3D 볼륨을 Local(슬라이스 윈도우 어텐션) + Global(BEV 붕괴+ASPP) 두 경로로 병렬 처리 후 sigmoid 가중합으로 융합 |
| **공유 가중치 윈도우 어텐션** | 모든 높이 슬라이스가 동일한 어텐션 파라미터 공유 → 파라미터 효율화, 높이 방향 일반화 |
| **ASPP (Atrous Spatial Pyramid Pooling)** | 다양한 팽창률의 합성곱으로 다중 스케일 수용 영역 커버, Global path의 장거리 맥락 수집에 활용 |
| **Preserve-Pooling** | bilinear 대신 MaxPooling으로 어텐션 마스크 다운샘플링 → 희소 전경 객체 구조 보존 |
| **Class-Guided Sampling** | 클래스 빈도 역수 가중치로 샘플링 → 희소 클래스(트럭, 보행자 등) supervision 강화 |
| **3D Mask Classification** | 점유를 "클래스별 이진 마스크의 합"으로 정식화. 쿼리 기반 Mask2Former 디코더를 3D로 확장 |

## 결론 및 시사점

OccFormer는 MonoScene→TPVFormer로 이어지는 카메라 기반 3D 점유 예측 계보의 다음 단계이다. 3D Conv를 Transformer 기반 듀얼패스로 대체해 장거리 맥락 학습과 효율적 3D 처리를 동시에 달성했다.

**점유 예측 계보 내 위치**:
- MonoScene (2022): 단안 카메라 SSC 최초 → 3D Conv + FLoSP
- TPVFormer (2023): tri-plane BEV 일반화 → 카메라 전용 3D 점유
- **OccFormer (2023)**: 듀얼패스 Transformer + Mask2Former 디코더 → SemanticKITTI SOTA

**자율주행 실무 시사점**:
- 단일 모델로 3D 시맨틱 점유 + LiDAR 시맨틱 분할을 동시에 처리 (TPVFormer는 두 모델 필요)
- 카메라만으로 70%+ LiDAR seg mIoU 달성 → LiDAR 없는 자율주행 가능성 시사
- Class-Guided Sampling은 합성 데이터 생성·회귀 테스트에서 희소 시나리오(소형 객체, 희귀 교통 상황) 커버리지 향상에도 응용 가능
