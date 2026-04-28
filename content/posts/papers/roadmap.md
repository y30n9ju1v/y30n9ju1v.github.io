---
title: "논문 로드맵: 자율주행 & 3D 장면 표현"
date: 2027-04-20T18:30:00+09:00
draft: false
---

이 포스트는 블로그에 정리된 논문들이 서로 어떻게 연결되는지를 보여주는 로드맵입니다.  
크게 두 줄기—**자율주행 스택**과 **3D 장면 표현**—가 최근 **Neural Simulation**이라는 교차점에서 만납니다.  
두 줄기 모두 **Transformer**와 **Latent Diffusion**, 그리고 **ViT**라는 공통 기반 기술 위에 서 있습니다. **ResNet**은 ViT 이전 CNN backbone의 표준으로, 이 계보의 출발점입니다.  
여기에 **온라인 벡터화 HD 맵** 계보가 BEV 인식과 계획 사이를 잇는 새로운 흐름으로 추가됩니다.  
**CLIP**은 자연어-이미지 멀티모달 기반으로 생성형 시뮬레이션(MagicDrive, DriveArena 등)의 텍스트 조건화 핵심 기반이 됩니다.  
**ControlNet**은 LDM 위에서 공간적 조건(엣지·깊이·포즈)으로 정밀한 이미지 제어를 가능케 하는 조건부 생성 확장입니다.

---

## 전체 흐름 요약

```
[기반 기술]
 ResNet (2016) ───────────────────────────── Skip Connection으로 깊은 CNN 학습, ViT 이전 backbone 표준
 Transformer (2017) ─────────────────────── Self-Attention, 모든 Transformer 계열의 원류
 ViT (2021) ──────────────────────────────── 이미지를 패치 시퀀스로 처리, ResNet을 대체한 BEV 인식 backbone 표준
 LDM (2022) ──────────────────────────────── Latent Space Diffusion + Cross-Attention, 생성 모델 계열의 원류
 ControlNet (2023) ──────────────────────── Zero Convolution으로 LDM에 공간 조건 주입, 정밀 이미지 제어
         │
         ▼
[데이터/시뮬레이터]
 nuScenes / Waymo / CARLA / nuPlan
         │
         ▼
[인식: BEV 카메라 계보]                    [인식: LiDAR 포인트 클라우드 계보]
 Lift-Splat-Shoot                          PointNet (2017)
  → BEVDepth → BEVFormer ◄─────────────── Transformer 기반 BEV Attention
 DETR → DETR3D (top-down query, 별개)      → PointNet++ (2017) — 계층적 로컬 특징
         │                                  → PointPillars (2019)
         │                                  → CenterPoint (2021)
         └──────────────────┬───────────────┘
                            ▼
               [센서 융합 & E2E]
                TransFuser / BEVFusion
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
     [점유 예측]        [HD 맵 구성]    [예측·계획]
  MonoScene → Occ3D    VectorMapNet   UniAD → VAD
  TPVFormer → SurroundOcc → MapTR      ▲
               → StreamMapNet  └─── Transformer Query 기반
  GaussianWorld (3DGS)
              │                           │
              └─────────────┬─────────────┘
                            ▼
               [평가 & 강화학습]
                DQN / PPO / NAVSIM / nuPlan / Bench2Drive
                            │
                            ▼
          [World Model & 생성형 시뮬레이션]
           DDPM → LDM → DriveDreamer / MagicDrive / DriveArena
                   └──► ControlNet (조건부 생성 제어 확장)
                   └──► GAIA-1 (Transformer 기반 World Model, 별개 계보)
           CLIP (2021) ─ 텍스트 조건화 기반 (MagicDrive·DriveArena text embedding)
                            │
          ┌─────────────────┴──────────────────────┐
          ▼                                        ▼
[3D 장면 표현]                              [Neural Simulation]
 NeRF                                        LiDARsim (2020) — 실제 데이터 기반 LiDAR 시뮬 (UniSim 선행)
  ├─► EmerNeRF — 자기지도 동적 장면 분해      UniSim (2023) — NeRF 기반 멀티센서
  └─► Instant-NGP                            NeuRAD (2024) — 범용 AD NeRF
       └─► 3DGS                              HUGSIM (2024) — 3DGS 기반 실시간
            ├─► [렌더링 품질]
            │    3DGS-RT / 3DGUT / DIFIX3D+
            ├─► [AV 장면 재구성]
            │    4D-GS → Street Gaussians
            │             DrivingGaussian
            │             OmniRe (2025)
            └─► [점유 예측]
                 GaussianWorld (2024)
```

---

## 0. 기반 기술: Transformer & Latent Diffusion

자율주행 스택과 생성형 시뮬레이션 양쪽 모두의 **공통 기반**이 되는 두 논문입니다.

```
ResNet — Deep Residual Learning (CVPR 2016)
 ├─► Skip Connection으로 Degradation Problem 해결, 수백 레이어 학습 가능
 ├─► ILSVRC 2015 1위, VGG·AlexNet 계보를 넘어 CNN backbone의 표준 확립
 └─► ViT 이전까지 BEVDepth·BEVFormer 등 인식 계열 backbone으로 광범위하게 사용

Transformer (NeurIPS 2017)
 ├─► BEVFormer, DETR3D, TPVFormer — BEV 인식 계열
 ├─► UniAD, VAD — 통합 계획 계열
 ├─► TransFuser, BEVFusion — 센서 융합 계열
 └─► GAIA-1 — Decoder-only Transformer 기반 World Model

ViT — An Image is Worth 16x16 Words (ICLR 2021)
 ├─► 이미지를 16×16 패치 시퀀스로 처리, CNN 없이 순수 Transformer로 SOTA 달성
 ├─► 대규모 사전학습(JFT-300M)으로 CNN(ResNet)의 귀납적 편향 없이도 능가
 ├─► BEVFormer, SurroundOcc, TPVFormer 등 BEV 인식 계열의 표준 backbone
 └─► 카메라 피처 추출기가 ResNet에서 ViT 계열로 전환되는 계기

DETR — End-to-End Object Detection with Transformers (ECCV 2020)
 ├─► 이분 매칭(Hungarian) 기반 집합 예측으로 NMS·anchor 완전 제거
 ├─► Encoder-Decoder Transformer + learned object query 패러다임 확립
 ├─► DETR3D — 3D object query를 2D에 back-projection (3D 탐지 계열)
 └─► VectorMapNet, MapTR — polyline/point query 기반 HD 맵 계열

LDM — Latent Diffusion Models (CVPR 2022)
 ├─► DDPM의 픽셀 공간 한계 → Latent Space Diffusion으로 해결
 ├─► Cross-Attention 조건부 생성 메커니즘 확립
 ├─► DriveDreamer, MagicDrive, DriveArena의 직접 기반
 └─► ControlNet — 공간 조건(엣지·깊이·포즈)으로 사전 훈련 모델 확장 (ICCV 2023)

ControlNet — Adding Conditional Control to Text-to-Image Diffusion Models (ICCV 2023)
 ├─► 원본 LDM을 freeze하고 인코더 복사본에 Zero Convolution으로 조건 주입
 ├─► 엣지맵·깊이맵·포즈·세그멘테이션 등 다양한 공간 조건을 단일 프레임워크로 지원
 └─► 소규모 데이터(1k~50k)로도 안정적 학습, 텍스트 조건화와 공간 조건화 동시 적용

CLIP — Learning Transferable Visual Models From Natural Language Supervision (ICML 2021)
 ├─► 4억 쌍 (이미지, 텍스트)을 대조 학습으로 이미지·텍스트 임베딩 공간 정렬
 ├─► 클래스 이름을 텍스트로 표현해 추가 학습 없이 30개+ 태스크 zero-shot 전이
 ├─► MagicDrive, DriveArena 등 생성 모델의 텍스트 조건화(text embedding) 기반
 └─► 분포 이동(distribution shift)에 강인한 표현 — 시뮬→실세계 도메인 갭 완화
```

| 논문 | 역할 |
|------|------|
| [ResNet](https://y30n9ju1v.github.io/posts/papers/resnet-deep-residual-learning-for-image-recognition) | Skip Connection으로 Degradation Problem 해결, 수백 레이어 학습 가능 — ViT 이전 CNN backbone 표준 (CVPR 2016) |
| [Transformer](https://y30n9ju1v.github.io/posts/papers/attention-is-all-you-need) | Self-Attention만으로 RNN·CNN을 대체, 현대 딥러닝의 기반 아키텍처 (NeurIPS 2017) |
| [ViT](https://y30n9ju1v.github.io/posts/papers/vit-an-image-is-worth-16x16-words) | 이미지를 16×16 패치 시퀀스로 처리하는 순수 Transformer, 대규모 사전학습으로 ResNet 능가 — BEV 인식 계열의 표준 backbone (ICLR 2021) |
| [DETR](https://y30n9ju1v.github.io/posts/papers/detr-end-to-end-object-detection-with-transformers) | 이분 매칭 기반 집합 예측으로 NMS·anchor 제거, object query 패러다임 확립 — DETR3D·MapTR의 직접 기반 (ECCV 2020) |
| [LDM](https://y30n9ju1v.github.io/posts/papers/high-resolution-image-synthesis-with-latent-diffusion-models) | Autoencoder 잠재 공간에서 Diffusion 수행, Cross-Attention 조건부 생성 확립 — Stable Diffusion의 기반 (CVPR 2022) |
| [ControlNet](https://y30n9ju1v.github.io/posts/papers/controlnet-adding-conditional-control-to-text-to-image-diffusion-models) | Zero Convolution으로 사전 훈련 LDM을 보호하면서 공간 조건(엣지·깊이·포즈) 주입, 소규모 데이터로도 안정적 학습 (ICCV 2023) |
| [CLIP](https://y30n9ju1v.github.io/posts/papers/clip-learning-transferable-visual-models-from-natural-language-supervision) | 4억 쌍 대조 학습으로 이미지·텍스트 공간 정렬, zero-shot 전이 및 생성 모델 텍스트 조건화 기반 (ICML 2021) |

> **흐름**: ResNet은 **CNN backbone의 표준**으로 ViT 이전까지 인식 계열 전반에 사용되고,  
> Transformer는 **인식·계획·World Model** 전반에 적용되며,  
> ViT는 **카메라 피처 추출기의 표준**으로 ResNet을 대체하고,  
> LDM은 **DDPM → 조건부 이미지/비디오 생성** 계열의 핵심 연결 고리이며,  
> ControlNet은 **LDM에 공간 조건을 추가**하여 정밀한 이미지 제어를 가능케 하고,  
> CLIP은 **자연어-이미지 정렬**로 생성형 시뮬레이션의 텍스트 조건화와 zero-shot 인식을 가능케 합니다.

---

## 1. 기반: 데이터셋 & 시뮬레이터

자율주행 연구는 **대규모 실제 데이터**와 **시뮬레이터** 위에서 시작됩니다.

| 논문 | 역할 |
|------|------|
| [nuScenes](https://y30n9ju1v.github.io/posts/papers/nuscenes-multimodal-dataset-autonomous-driving) | 카메라·LiDAR·레이더 통합 멀티모달 데이터셋 (CVPR 2020) |
| [Waymo Open Dataset](https://y30n9ju1v.github.io/posts/papers/waymo-open-dataset) | 대규모 카메라+LiDAR 데이터, 15배 지리 다양성 (CVPR 2020) |
| [CARLA](https://y30n9ju1v.github.io/posts/papers/carla-an-open-urban-driving-simulator) | 오픈소스 도시 주행 시뮬레이터, 모방/강화학습 벤치마크 (CoRL 2017) |
| [nuPlan](https://y30n9ju1v.github.io/posts/papers/nuplan) | 클로즈드루프 ML 계획 벤치마크, 1500시간 실제 데이터 (NeurIPS 2021) |

> **흐름**: nuScenes·Waymo가 **인식 연구의 표준 평가셋**이 되고,  
> CARLA는 **폐쇄 루프 시뮬레이션 플랫폼**으로, nuPlan은 **계획 벤치마크**로 자리잡습니다.

---

## 2. 자율주행 스택: 인식 → 예측 → 계획

### 2-1. 인식: BEV 표현 (카메라)

카메라 이미지를 **Bird's-Eye-View(BEV)** 공간으로 변환하는 것이 핵심 과제입니다.

```
공통 backbone
ResNet (2016) ─── Skip Connection 기반 deep CNN, ViT 이전의 표준 피처 추출기
    │ → ViT 등장 이후 대부분의 BEV 인식 계열에서 ViT로 교체됨
    ▼
ViT (2021) ─── 이미지 패치 → Transformer Encoder, BEV 인식 계열의 현재 표준 피처 추출기
    │
    ▼
다중 카메라 (Lift 계보)
Lift-Splat-Shoot (2020) ──► BEVDepth (2022) ──► BEVFormer (2022)
(Latent Depth Distribution)  (명시적 깊이 감독)   (Spatial + Temporal Attention)

다중 카메라 (Query 계보)
DETR3D (2021) ─── 3D object query → 2D back-projection (depth 예측 없음, NMS-free)
```

| 논문 | 핵심 아이디어 |
|------|-------------|
| [Lift-Splat-Shoot](https://y30n9ju1v.github.io/posts/papers/lift-splat-shoot) | 깊이 분포로 frustum 포인트 클라우드 생성 후 BEV Pillar Pooling, end-to-end 모션 플래닝 (ECCV 2020) |
| [DETR3D](https://y30n9ju1v.github.io/posts/papers/detr3d-3d-object-detection-multi-view-images) | 3D object query를 2D에 back-projection, depth 예측 없이 NMS-free end-to-end 검출 (CoRL 2021) |
| [BEVDepth](https://y30n9ju1v.github.io/posts/papers/bevdepth) | LiDAR 깊이 감독으로 LSS의 깊이 추정 신뢰도 개선 (AAAI 2023) |
| [BEVFormer](https://y30n9ju1v.github.io/posts/papers/bevformer) | 다중 카메라 + 시간 정보를 Transformer로 BEV 통합 (ECCV 2022) |

### 2-2. 인식: LiDAR 기반 3D 탐지

**LiDAR 포인트 클라우드에서 직접 3D 객체를 탐지**하는 계보입니다. BEVFusion의 LiDAR 브랜치 기반이 됩니다.

```
PointNet (2017) ─── 포인트 클라우드 직접 처리, 글로벌 MaxPooling
    │ 로컬 구조를 학습하지 못하는 한계
    ▼
PointNet++ (2017) ── FPS + Ball Query + mini-PointNet 계층 구조
    │ CNN의 receptive field 확장을 비정형 포인트 클라우드에 이식
    ▼
PointPillars (2019) ─ Pillar + PointNet 인코딩 → 2D CNN, 62Hz 실시간
    ▼
CenterPoint (2021) ── 중심점 heatmap + velocity 회귀, 1ms 추적
```

| 논문 | 핵심 아이디어 |
|------|-------------|
| [PointNet](https://y30n9ju1v.github.io/posts/papers/pointnet-deep-learning-on-point-sets-for-3d-classification-and-segmentation) | 포인트 클라우드를 raw 형태로 직접 처리하는 최초의 딥러닝, MaxPooling으로 순열 불변성 보장 (CVPR 2017) |
| [PointNet++](https://y30n9ju1v.github.io/posts/papers/pointnet-plus-plus-deep-hierarchical-feature-learning) | FPS + Ball Query + mini-PointNet 계층 구조로 로컬 기하 특징 학습, 밀도 불균일 강인성(MSG/MRG) (NeurIPS 2017) |
| [PointPillars](https://y30n9ju1v.github.io/posts/papers/pointpillars-fast-encoders-object-detection-point-clouds) | 포인트 클라우드를 수직 pillar로 조직화 + PointNet 인코딩 → 2D CNN만으로 62Hz 실시간 3D 탐지 (CVPR 2019) |
| [CenterPoint](https://y30n9ju1v.github.io/posts/papers/centerpoint-center-based-3d-object-detection-and-tracking) | 3D 객체를 중심점(heatmap)으로 표현, anchor 불필요·방향 불변, velocity 회귀로 1ms 추적 (CVPR 2021) |

### 2-3. 센서 융합 & E2E 주행

단일 모달의 한계를 넘어 **카메라 + LiDAR 융합**을 통한 엔드-투-엔드 주행으로 발전합니다.

```
TransFuser (2022)
 └─ Camera + LiDAR → Transformer Fusion → Waypoint 예측
     └─ CARLA 리더보드 1위, NAVSIM baseline으로 재활용

BEVFusion (2023)
 └─ Camera-to-BEV + LiDAR-to-BEV → 공유 BEV 공간 Concat → 멀티태스크 Head
     └─ 포인트 레벨 융합의 의미론적 손실 해소, Efficient BEV Pooling으로 40× 속도 향상
```

| 논문 | 핵심 아이디어 |
|------|-------------|
| [TransFuser](https://y30n9ju1v.github.io/posts/papers/transfuser) | Transformer self-attention으로 카메라+LiDAR 전역 융합 (TPAMI 2022) |
| [BEVFusion](https://y30n9ju1v.github.io/posts/papers/bevfusion-multi-task-multi-sensor-fusion) | BEV 공간 멀티센서 융합으로 탐지+분할 멀티태스크 동시 수행, Efficient BEV Pooling (CVPR 2023) |

### 2-4. 점유 예측: 3D Occupancy

BEV 탐지를 넘어 **빈/점유 복셀 단위의 밀도 높은 3D 장면 이해**로 확장합니다. 일반 객체(out-of-vocabulary)까지 표현 가능한 점에서 자율주행 안전성과 직결됩니다.

| 논문 | 핵심 아이디어 |
|------|-------------|
| [MonoScene](https://y30n9ju1v.github.io/posts/papers/monoscene-monocular-3d-semantic-scene-completion) | 단일 RGB 이미지만으로 실내·실외 3D SSC 최초 수행, FLoSP(시선 투영) + 3D CRP(맥락 관계 사전) 제안 (CVPR 2022) |
| [Occ3D](https://y30n9ju1v.github.io/posts/papers/occ3d-large-scale-3d-occupancy-prediction-benchmark) | 반자동 visibility-aware 레이블 파이프라인으로 Waymo·nuScenes 기반 대규모 3D 점유 벤치마크 구축, CTF-Occ 제안 (NeurIPS 2023) |
| [TPVFormer](https://y30n9ju1v.github.io/posts/papers/tpvformer-tri-perspective-view-3d-semantic-occupancy) | BEV를 Top·Side·Front 세 직교 평면으로 일반화, 카메라만으로 LiDAR 수준 3D 시맨틱 점유 예측 (CVPR 2023) |
| [SurroundOcc](https://y30n9ju1v.github.io/posts/papers/surroundocc) | 2D-3D 공간 어텐션 + 멀티스케일 볼류메트릭 쿼리로 멀티카메라 밀집 3D 시맨틱 점유 예측, Poisson Reconstruction 기반 밀집 GT 자동 생성 (2023) |
| [GaussianWorld](https://y30n9ju1v.github.io/posts/papers/gaussianworld-gaussian-world-model-for-streaming-3d-occupancy-prediction) | 3D Gaussian World Model로 장면 진화(자차 정합·동적 이동·신규 완성)를 명시적 모델링, 추가 연산 없이 단일 프레임 대비 mIoU +2% (Tsinghua 2024) |

### 2-5. 온라인 벡터화 HD 맵

BEV 인식 위에서 **정적 맵 요소(차선, 도로 경계, 횡단보도)를 polyline 벡터로 실시간 예측**합니다. 오프라인 HD 맵 없이도 경로 계획에 필요한 구조화된 맵을 온보드에서 직접 생성합니다.

```
DETR (object query 패러다임)
    │
    ▼
VectorMapNet (ICML 2023) ─── map element detection + autoregressive polyline 생성 최초 end-to-end
    │
    ▼
MapTR (ICLR 2023) ──────── unified point set permutation 동등 손실, 실시간(5.1FPS) 벡터화 맵
    │
    ▼
StreamMapNet (2023) ─────── streaming 시간 정보 통합 + cross-attention 기반 넓은 범위(100×50m) 지원
```

| 논문 | 핵심 아이디어 |
|------|-------------|
| [VectorMapNet](https://y30n9ju1v.github.io/posts/papers/vectormapnet-end-to-end-vectorized-hd-map-learning) | 카메라·LiDAR로 HD 맵을 벡터 polyline으로 직접 예측하는 최초 end-to-end 프레임워크, 2단계(element detection → polyline 생성) (ICML 2023) |
| [MapTR](https://y30n9ju1v.github.io/posts/papers/maptr-structured-modeling-online-vectorized-hd-map-construction) | Unified point set 동등 손실 + 계층적 이분 매칭으로 순열 모호성 제거, 5.1FPS 실시간 처리 (ICLR 2023) |
| [StreamMapNet](https://y30n9ju1v.github.io/posts/papers/streammapnet-streaming-mapping-network-vectorized-online-hd-map-construction) | Streaming 방식 temporal BEV 융합 + Instance-level cross-attention으로 100×50m 범위 지원, 데이터 누출 없는 공정 벤치마크 제시 (2023) |

> **흐름**: 오프라인 HD 맵 구축 비용을 없애는 **온라인 맵 생성** 파이프라인으로,  
> UniAD·VAD 같은 통합 계획 모델이 사용하는 맵 표현을 직접 대체합니다.

---

### 2-6. 통합 예측·계획

BEV 인식 위에 **모션 예측과 경로 계획**을 엔드-투-엔드로 통합합니다.

```
BEVFormer (인식)
    │
    ▼
UniAD (2023) ─── 5-task unified: 추적→맵→모션→점유→계획
    │
    ▼
VAD (2023) ────── 벡터화 표현으로 경량화 + 명시적 안전 제약
```

| 논문 | 핵심 아이디어 |
|------|-------------|
| [UniAD](https://y30n9ju1v.github.io/posts/papers/uniad-planning-oriented-autonomous-driving) | 계획 지향 통합 프레임워크, 5개 태스크를 Query로 연결 (CVPR 2023) |
| [VAD](https://y30n9ju1v.github.io/posts/papers/vad) | 래스터 맵 → 벡터화 표현, 인스턴스 수준 계획 제약 (2023) |

### 2-7. 평가 & 벤치마크

```
[강화학습 원류]
DQN (2013) ──► Value-based, 이산 행동 공간 (Atari)
PPO (2017) ──► Policy Gradient, 연속 행동 공간·CARLA AV policy 학습 표준

[클로즈드루프 벤치마크]
nuPlan (2021) ─► 실제 데이터 기반 클로즈드루프 계획 평가
NAVSIM (2024) ─► 비반응형 시뮬로 오픈/클로즈드루프 격차 해소
Bench2Drive (2024) ─► CARLA 기반 44개 시나리오 다중 능력 평가
```

| 논문 | 역할 |
|------|------|
| [DQN](https://y30n9ju1v.github.io/posts/papers/dqn-playing-atari-with-deep-reinforcement-learning) | Experience Replay + CNN Q-learning, value-based 강화학습 원형 (DeepMind 2013) |
| [PPO](https://y30n9ju1v.github.io/posts/papers/proximal-policy-optimization) | Clipped surrogate objective로 안정적 policy gradient, CARLA 기반 AV policy 학습 표준 (OpenAI 2017) |
| [NAVSIM](https://y30n9ju1v.github.io/posts/papers/navsim) | 실제 데이터 기반 비반응형 시뮬 벤치마크, PDMS 지표 (NeurIPS 2024) |
| [Bench2Drive](https://y30n9ju1v.github.io/posts/papers/bench2drive-multi-ability-benchmarking-e2e-autonomous-driving) | CARLA 기반 44개 시나리오 × 5경로로 E2E-AD 다중 능력 폐쇄 루프 평가, 공식 학습 데이터셋 제공 (NeurIPS 2024) |

### 2-8. World Model & 생성형 시뮬레이션

인식·계획 파이프라인을 넘어 **세계 자체를 생성 모델로 학습**하거나 **실제 주행 데이터로 다양한 시나리오를 합성**하는 패러다임 전환입니다.

```
DQN (2013) / PPO (2017) ─── 강화학습 원형
    │
    ▼
UniAD / VAD (E2E 계획)
    │  "어떻게 주행할지"를 학습
    ▼
GAIA-1 (2023) ─── "세상이 어떻게 작동하는지"를 학습
    ├─ 비디오 + 텍스트 + 액션 → 다음 토큰 예측 (GPT 방식)
    ├─ LLM 스케일링 법칙이 AV World Model에도 적용됨 확인
    └─ Neural Simulator로서 HUGSIM 방향의 생성형 대안 제시

DDPM (2020) ─── Diffusion 생성 모델 기반 기술
    ├─ 노이즈 점진적 추가/제거 Markov chain 기반 학습
    ├─ denoising score matching과의 동등성 발견
    └─ DriveDreamer·MagicDrive·DriveArena의 직접 기반
         └─► LDM (2022) ─── Latent Space Diffusion + Cross-Attention
                  └─► ControlNet (2023) ─── Zero Convolution으로 LDM에 공간 조건 주입
                          ├─ 원본 모델 freeze + 인코더 복사본에 조건 학습
                          └─ 엣지·깊이·포즈·세그멘테이션 등 다양한 공간 조건 지원

DriveDreamer (2023) ─── 실제 주행 데이터 기반 제어 가능한 비디오 생성
    ├─ 구조화된 트래픽 제약(HD맵, 3D 박스)으로 Diffusion 조건화
    ├─ 2단계 생성: 희소 제약 → 조건 이미지 → 멀티뷰 비디오
    └─ 생성 데이터로 E2E 모델 학습 데이터 증강 효과 검증

MagicDrive (2024) ─── 3D 기하 조건 기반 멀티카메라 스트리트뷰 생성
    ├─ BEV 맵 + 3D 바운딩 박스 + 카메라 포즈를 독립 인코딩
    ├─ Cross-view attention으로 카메라 간 일관성 보장
    └─ 생성 데이터로 BEV Segmentation·3D Detection 성능 향상

DriveArena (2024) ─── 최초의 폐루프 생성형 시뮬레이션 플랫폼
    ├─ Traffic Manager (OSM 기반 실시간 교통 시뮬레이션)
    ├─ World Dreamer (조건부 Diffusion 기반 포토리얼 이미지 생성)
    └─ 에이전트 행동 → 교통 상태 갱신 → 이미지 재생성의 폐루프 구현
```

| 논문 | 핵심 아이디어 |
|------|-------------|
| [DDPM](https://y30n9ju1v.github.io/posts/papers/denoising-diffusion-probabilistic-models) | Markov chain 기반 노이즈 추가/제거로 고품질 이미지 합성, DriveDreamer·MagicDrive의 직접 기반 (NeurIPS 2020) |
| [ControlNet](https://y30n9ju1v.github.io/posts/papers/controlnet-adding-conditional-control-to-text-to-image-diffusion-models) | Zero Convolution으로 사전 훈련 LDM에 공간 조건(엣지·깊이·포즈) 주입, 소규모 데이터로 안정적 학습 (ICCV 2023) |
| [GAIA-1](https://y30n9ju1v.github.io/posts/papers/gaia-1) | 비디오·텍스트·액션을 토큰으로 통합한 생성형 AV World Model, 스케일링 법칙 검증 (Wayve 2023) |
| [DriveDreamer](https://y30n9ju1v.github.io/posts/papers/drivedreamer) | 실제 주행 데이터로 학습한 제어 가능한 Diffusion 기반 비디오 생성, 합성 데이터로 E2E 성능 향상 (ECCV 2024) |
| [MagicDrive](https://y30n9ju1v.github.io/posts/papers/magicdrive-street-view-generation-3d-geometry-control) | 3D 바운딩 박스·BEV 맵·카메라 포즈 조건으로 멀티카메라 스트리트뷰 생성, Cross-view attention으로 일관성 보장 (ICLR 2024) |
| [DriveArena](https://y30n9ju1v.github.io/posts/papers/drivearena) | Traffic Manager + World Dreamer 조합으로 폐루프 생성형 시뮬레이션 구현, 실제 도로 기반 photo-realistic 상호작용 평가 (2024) |

---

## 3. 3D 장면 표현: NeRF → 3DGS → 자율주행 적용

### 3-1. 기반 기술

```
NeRF (2020) ─────────────────────── MLP로 5D 연속 함수 표현
    │ 유한 장면만 표현 가능, 무한 야외 장면에 취약
    ├─► EmerNeRF (2024) ──────────── 정적/동적 필드 자기지도 분리, AV 장면 NeRF 적용 교두보
    ▼
Instant-NGP (2022) ───────────────── Multiresolution Hash Encoding, 1000배 속도 향상
    │ 학습 시간: 수십 초 ~ 수 분
    ▼
3D Gaussian Splatting (2023) ─────── 3D Gaussian 파티클 + 실시간 래스터화
    │ SIGGRAPH 2023, 30fps+ @ 1080p
    ▼
4D Gaussian Splatting (2024) ─────── Canonical 3DGS + Deformation Field로 동적 장면 확장
    │ ICLR 2024, 82fps @ 800×800 (합성), HexPlane 시공간 인코딩
    ▼
```

| 논문 | 핵심 아이디어 |
|------|-------------|
| [NeRF](https://y30n9ju1v.github.io/posts/papers/nerf-representing-scenes-as-neural-radiance-fields-for-view-synthesis) | 희소 이미지로부터 체적 렌더링으로 새로운 시점 합성 (ECCV 2020) |
| [EmerNeRF](https://y30n9ju1v.github.io/posts/papers/emernerf) | 정적/동적 필드 분리 + 씬 플로우를 완전 자기지도로 창발, GT 어노테이션 불필요 (ICLR 2024) |
| [Instant-NGP](https://y30n9ju1v.github.io/posts/papers/instant-ngp-multiresolution-hash-encoding) | Multiresolution Hash Encoding으로 NeRF 학습 속도 1000배 향상 (SIGGRAPH 2022) |
| [3DGS](https://y30n9ju1v.github.io/posts/papers/3d-gaussian-splatting) | Gaussian 파티클 + tile-based 래스터화로 실시간 렌더링 (SIGGRAPH 2023) |
| [4D-GS](https://y30n9ju1v.github.io/posts/papers/4d-gaussian-splatting) | Canonical 3DGS + Gaussian Deformation Field로 실시간 동적 NVS (ICLR 2024) |

### 3-2. 3DGS 확장: 렌더링 품질 향상

3DGS의 **래스터화 한계**(이차 반사·굴절·왜곡 카메라 미지원)를 극복하는 두 방향의 확장이 동시에 진행됩니다.

```
3DGS (2023)
    ├─► 3D Gaussian Ray Tracing (2024) ─── BVH + OptiX 레이 트레이싱, secondary rays
    ├─► 3DGUT (2025) ──────────────────── Unscented Transform으로 왜곡 카메라 지원
    └─► DIFIX3D+ (2025) ───────────────── 단일 단계 Diffusion으로 아티팩트 제거
```

| 논문 | 핵심 아이디어 |
|------|-------------|
| [3D Gaussian Ray Tracing](https://y30n9ju1v.github.io/posts/papers/3d-gaussian-ray-tracing) | GPU BVH 레이 트레이싱으로 반사·굴절·그림자 지원 (SIGGRAPH Asia 2024) |
| [3DGUT](https://y30n9ju1v.github.io/posts/papers/3dgut-enabling-distorted-cameras-and-secondary-rays-in-gaussian-splatting) | Unscented Transform으로 어안렌즈·롤링셔터 래스터화 (NVIDIA 2025) |
| [DIFIX3D+](https://y30n9ju1v.github.io/posts/papers/difix3d-plus) | Diffusion 모델로 NeRF/3DGS 아티팩트 후처리 제거 (NVIDIA 2025) |

### 3-3. 자율주행 장면에 3DGS 적용

정적 배경과 동적 객체를 분리하여 자율주행 장면을 재구성합니다. **4D-GS의 deformation field 개념**이 이 단계의 핵심 기반 기술입니다.

```
4D-GS (동적 Gaussian 표현 기반)
    ├─► Street Gaussians (2024) ── 트래킹 기반 차량 자세 최적화 + 구면 고조파 분해 조명
    ├─► DrivingGaussian (2024) ─── 정적 Incremental 3DGS + 동적 Composite Graph
    └─► OmniRe (2025) ─────────── SMPL+Deformable+Rigid 통합 Gaussian 씬 그래프, holistic 재구성
```

| 논문 | 핵심 아이디어 |
|------|-------------|
| [Street Gaussians](https://y30n9ju1v.github.io/posts/papers/street-gaussians-modeling-dynamic-urban-scenes) | 트래킹 기반 차량 자세 최적화 + 구면 고조파 조명 분해로 동적 도시 장면 재구성 (ECCV 2024) |
| [DrivingGaussian](https://y30n9ju1v.github.io/posts/papers/driving-gaussian-composite-gaussian-splatting) | Composite 3DGS로 서라운드 동적 장면 재구성 (CVPR 2024) |
| [OmniRe](https://y30n9ju1v.github.io/posts/papers/omnire-omni-urban-scene-reconstruction) | SMPL+Deformable+Rigid 통합 Gaussian 씬 그래프, 차량·보행자·자전거 holistic 재구성 (NVIDIA 2025) |

---

## 4. 교차점: Neural Simulation

자율주행 **인식·계획 스택**, **생성형 World Model**, **3DGS 기반 장면 표현**이 한데 모여  
**포토리얼리스틱 클로즈드루프 시뮬레이터**가 탄생합니다.

```
[재구성 기반 시뮬레이터]                    [생성형 시뮬레이터]
 실제 데이터 기반 LiDAR 시뮬                  GAIA-1 / DriveDreamer
  → LiDARsim (레이캐스팅 + ML raydrop)        → 비디오·텍스트·액션 토큰 기반
 NeRF 장면 재구성                            MagicDrive / DriveArena
  → UniSim (카메라+LiDAR 동시 합성)           → 조건부 Diffusion + 폐루프
  → NeuRAD (5개 데이터셋 범용)
 3DGS 장면 재구성
  → 4D-GS / Street Gaussians
  → DrivingGaussian / OmniRe
  → HUGSIM (실시간 클로즈드루프)
         │                                           │
         └──────────────┬────────────────────────────┘
                        ▼
           [클로즈드루프 평가 프레임워크]

            자율주행 에이전트 (UniAD, VAD, TransFuser)
            + 평가 기준 (nuPlan, NAVSIM, Bench2Drive)
```

| 논문 | 역할 |
|------|------|
| [LiDARsim](https://y30n9ju1v.github.io/posts/papers/lidarsim-realistic-lidar-simulation-by-leveraging-the-real-world) | 실제 주행 데이터 기반 3D 자산 구축 + 레이캐스팅 + ML raydrop 학습으로 현실적 LiDAR 시뮬, UniSim의 선행 연구 (CVPR 2020) |
| [UniSim](https://y30n9ju1v.github.io/posts/papers/unisim-neural-closed-loop-sensor-simulator) | NeRF 기반 카메라+LiDAR 동시 시뮬, 장면 편집 지원 클로즈드루프 시뮬레이터 (CVPR 2023) |
| [NeuRAD](https://y30n9ju1v.github.io/posts/papers/neurad-neural-rendering-for-autonomous-driving) | 5개 AD 데이터셋 범용 NeRF 시뮬, rolling shutter·ray drop·beam divergence 통합 모델링 (2024) |
| [HUGSIM](https://y30n9ju1v.github.io/posts/papers/hugsim-real-time-photorealistic-closed-loop-simulator) | 3DGS 기반 실시간 포토리얼리스틱 클로즈드루프 AV 시뮬레이터 (CVPR 2024) |
| [GAIA-1](https://y30n9ju1v.github.io/posts/papers/gaia-1) | 비디오·텍스트·액션 토큰 기반 생성형 World Model, 신경 시뮬레이터 역할 (Wayve 2023) |
| [DriveDreamer](https://y30n9ju1v.github.io/posts/papers/drivedreamer) | 실제 주행 데이터 기반 Diffusion 비디오 생성, 제어 가능한 합성 데이터 생성 (ECCV 2024) |
| [MagicDrive](https://y30n9ju1v.github.io/posts/papers/magicdrive-street-view-generation-3d-geometry-control) | 3D 기하 조건 기반 멀티카메라 스트리트뷰 생성, Cross-view attention으로 카메라 간 일관성 보장 (ICLR 2024) |
| [DriveArena](https://y30n9ju1v.github.io/posts/papers/drivearena) | Traffic Manager + World Dreamer 폐루프 생성형 시뮬레이션, 에이전트 반응형 photo-realistic 평가 (2024) |

---

## 5. 논문 연도별 타임라인

```
2013  DQN ──────────────────────────── 딥 강화학습의 출발 (value-based)
2015  ResNet ──────────────────────── Skip Connection으로 Degradation Problem 해결, 수백 레이어 학습 가능 · ILSVRC 2015 1위 (CVPR 2016 발표)
2017  Transformer ──────────────────── Self-Attention만으로 RNN·CNN 대체, 현대 딥러닝 패러다임 전환 (NeurIPS)
      CARLA ───────────────────────── 자율주행 오픈 시뮬레이터
      PointNet ────────────────────── 포인트 클라우드 직접 처리 최초 딥러닝, MaxPooling 순열 불변성 (CVPR)
      PointNet++ ──────────────────── FPS + Ball Query + 계층적 Set Abstraction, 로컬 기하 학습 + 밀도 불균일 강인성 (NeurIPS)
      PPO ─────────────────────────── Clipped Policy Gradient, CARLA AV policy 학습 표준 (OpenAI)
2019  PointPillars ───────────────── Pillar + PointNet + 2D CNN, 62Hz 실시간 LiDAR 탐지 (CVPR)
2020  NeRF ──────────────────────── 신경 장면 표현
      nuScenes, Waymo ─────────────── 멀티모달 데이터셋
      Lift-Splat-Shoot ────────────── Latent Depth로 BEV, E2E 모션 플래닝
      DETR ────────────────────────── 이분 매칭 기반 집합 예측, NMS·anchor 제거, object query 패러다임 확립 (ECCV)
      DDPM ────────────────────────── Diffusion 생성 모델 원본, DriveDreamer·MagicDrive 기반 (NeurIPS)
      LiDARsim ───────────────────── 실제 데이터 기반 LiDAR 시뮬, 레이캐스팅 + ML raydrop (CVPR)
2021  ViT ─────────────────────────── 이미지를 패치 시퀀스로 처리하는 순수 Transformer, BEV 인식 backbone 표준 (ICLR)
      CLIP ────────────────────────── 4억 쌍 대조 학습으로 이미지·텍스트 공간 정렬, zero-shot 전이 + 생성 모델 텍스트 조건화 기반 (ICML)
      DETR3D ──────────────────────── top-down 3D query 기반 멀티뷰 탐지
      CenterPoint ────────────────── LiDAR 중심점 탐지 + velocity 기반 1ms 추적 (CVPR)
      nuPlan ──────────────────────── 클로즈드루프 ML 계획 벤치마크 (NeurIPS)
2022  LDM ────────────────────────── Latent Space Diffusion + Cross-Attention, Stable Diffusion 기반 (CVPR)
      MonoScene ───────────────────── 단일 RGB → 3D SSC 최초, FLoSP + 3D CRP (CVPR)
      Instant-NGP ────────────────── Hash Encoding으로 NeRF 1000배 가속
      BEVDepth ───────────────────── LiDAR 감독으로 깊이 신뢰도 개선
      BEVFormer ──────────────────── 멀티카메라 BEV Transformer
      TransFuser ─────────────────── 카메라+LiDAR 융합 E2E
2023  BEVFusion ───────────────────── BEV 공간 멀티센서 융합, 멀티태스크
      3DGS ───────────────────────── 실시간 3D Gaussian 렌더링 (SIGGRAPH)
      4D-GS ───────────────────────── Canonical 3DGS + Deformation Field, 동적 장면 실시간 NVS (ICLR 2024 발표)
      UniAD ───────────────────────── 계획 지향 통합 AV (CVPR)
      VAD ─────────────────────────── 벡터화 장면 표현으로 E2E 계획 경량화
      UniSim ──────────────────────── NeRF 기반 멀티센서 클로즈드루프 시뮬 (CVPR)
      Occ3D ───────────────────────── 대규모 3D 점유 예측 벤치마크 (NeurIPS)
      TPVFormer ───────────────────── Tri-Perspective View, 카메라만으로 3D 시맨틱 점유 예측 (CVPR)
      SurroundOcc ─────────────────── 2D-3D 공간 어텐션 + 볼류메트릭 쿼리 멀티카메라 밀집 점유 예측
      GAIA-1 ──────────────────────── 생성형 AV World Model (GPT 방식, Wayve)
      DriveDreamer ────────────────── 실제 주행 데이터 기반 제어 가능한 비디오 Diffusion
      VectorMapNet ────────────────── end-to-end 벡터화 HD 맵 최초 end-to-end 파이프라인 (ICML)
      MapTR ───────────────────────── unified point set 동등 손실 기반 실시간 벡터화 HD 맵 (ICLR)
      StreamMapNet ────────────────── streaming 시간 정보 통합 + 100×50m 넓은 범위 HD 맵
      ControlNet ──────────────────── Zero Convolution으로 LDM에 공간 조건(엣지·깊이·포즈) 주입, 소규모 데이터 학습 (ICCV)
2024  MagicDrive ──────────────────── 3D 기하 조건 기반 멀티카메라 스트리트뷰 생성 (ICLR)
      EmerNeRF ────────────────────── 자기지도 동적 장면 분해 NeRF, GT 어노테이션 불필요 (ICLR)
      GaussianWorld ───────────────── 3D Gaussian World Model 기반 스트리밍 occupancy 예측, 추가 연산 없이 mIoU +2%
      DriveArena ──────────────────── Traffic Manager + World Dreamer 폐루프 생성형 시뮬
      3D Gaussian Ray Tracing ────── BVH + OptiX 레이 트레이싱, secondary rays (SIGGRAPH Asia)
      Street Gaussians ────────────── 트래킹 기반 동적 도시 장면 3DGS (ECCV)
      DrivingGaussian ─────────────── Composite 3DGS로 서라운드 동적 장면 재구성 (CVPR)
      NeuRAD ──────────────────────── 5개 데이터셋 범용 AD NeRF 시뮬, rolling shutter·raydrop 모델링
      HUGSIM ──────────────────────── 3DGS 기반 실시간 포토리얼 클로즈드루프 시뮬 (CVPR)
      NAVSIM ──────────────────────── 실제 데이터 기반 비반응형 AV 벤치마크 (NeurIPS)
      Bench2Drive ─────────────────── CARLA 기반 44개 시나리오 다중 능력 E2E-AD 벤치마크 (NeurIPS)
2025  3DGUT ───────────────────────── Unscented Transform으로 어안렌즈·롤링셔터 지원 (NVIDIA)
      DIFIX3D+ ────────────────────── Diffusion 기반 NeRF/3DGS 아티팩트 후처리 제거 (NVIDIA)
      OmniRe ──────────────────────── SMPL+Deformable+Rigid 통합 Gaussian 씬 그래프, holistic 재구성 (NVIDIA)
```

---

## 6. 추천 읽기 순서

### 자율주행 스택에 집중한다면
0. [ResNet](https://y30n9ju1v.github.io/posts/papers/resnet-deep-residual-learning-for-image-recognition) — Skip Connection·잔차 학습·깊은 CNN backbone 이해 (ViT 이전 표준)
   → [Transformer](https://y30n9ju1v.github.io/posts/papers/attention-is-all-you-need) — Self-Attention·Multi-Head Attention·Positional Encoding 이해 (필수 선행)
   → [ViT](https://y30n9ju1v.github.io/posts/papers/vit-an-image-is-worth-16x16-words) — Transformer를 이미지에 적용, ResNet을 대체한 BEV 인식 계열 표준 backbone 이해
1. [nuScenes](https://y30n9ju1v.github.io/posts/papers/nuscenes-multimodal-dataset-autonomous-driving) — 데이터 기반 이해
2. [Lift-Splat-Shoot](https://y30n9ju1v.github.io/posts/papers/lift-splat-shoot) → [BEVDepth](https://y30n9ju1v.github.io/posts/papers/bevdepth) → [BEVFormer](https://y30n9ju1v.github.io/posts/papers/bevformer) — BEV 카메라 인식 계보
3. [DETR](https://y30n9ju1v.github.io/posts/papers/detr-end-to-end-object-detection-with-transformers) — object query 패러다임 원형 (NMS·anchor 제거, 이분 매칭)
   → [DETR3D](https://y30n9ju1v.github.io/posts/papers/detr3d-3d-object-detection-multi-view-images) — top-down 3D query 계보
   → [VectorMapNet](https://y30n9ju1v.github.io/posts/papers/vectormapnet-end-to-end-vectorized-hd-map-learning) → [MapTR](https://y30n9ju1v.github.io/posts/papers/maptr-structured-modeling-online-vectorized-hd-map-construction) → [StreamMapNet](https://y30n9ju1v.github.io/posts/papers/streammapnet-streaming-mapping-network-vectorized-online-hd-map-construction) — HD 맵 계보
4. [PointNet](https://y30n9ju1v.github.io/posts/papers/pointnet-deep-learning-on-point-sets-for-3d-classification-and-segmentation) → [PointNet++](https://y30n9ju1v.github.io/posts/papers/pointnet-plus-plus-deep-hierarchical-feature-learning) → [PointPillars](https://y30n9ju1v.github.io/posts/papers/pointpillars-fast-encoders-object-detection-point-clouds) → [CenterPoint](https://y30n9ju1v.github.io/posts/papers/centerpoint-center-based-3d-object-detection-and-tracking) — LiDAR 탐지 계보
5. [MonoScene](https://y30n9ju1v.github.io/posts/papers/monoscene-monocular-3d-semantic-scene-completion) — 카메라 전용 3D SSC 패러다임 확립
   → [Occ3D](https://y30n9ju1v.github.io/posts/papers/occ3d-large-scale-3d-occupancy-prediction-benchmark) — 점유 예측 벤치마크
   → [TPVFormer](https://y30n9ju1v.github.io/posts/papers/tpvformer-tri-perspective-view-3d-semantic-occupancy) — BEV를 세 평면으로 일반화, 카메라 전용 3D 점유 예측
   → [SurroundOcc](https://y30n9ju1v.github.io/posts/papers/surroundocc) — 2D-3D 공간 어텐션 + 볼류메트릭 쿼리 멀티카메라 밀집 점유 예측
   → [GaussianWorld](https://y30n9ju1v.github.io/posts/papers/gaussianworld-gaussian-world-model-for-streaming-3d-occupancy-prediction) — 3DGS World Model 기반 스트리밍 occupancy, 추가 연산 없이 시간적 모델링
6. [TransFuser](https://y30n9ju1v.github.io/posts/papers/transfuser) → [BEVFusion](https://y30n9ju1v.github.io/posts/papers/bevfusion-multi-task-multi-sensor-fusion) — 센서 융합 E2E
7. [UniAD](https://y30n9ju1v.github.io/posts/papers/uniad-planning-oriented-autonomous-driving) → [VAD](https://y30n9ju1v.github.io/posts/papers/vad) — 통합 계획
8. [DQN](https://y30n9ju1v.github.io/posts/papers/dqn-playing-atari-with-deep-reinforcement-learning) → [PPO](https://y30n9ju1v.github.io/posts/papers/proximal-policy-optimization) — 강화학습 원류 (DQN: value-based, PPO: policy gradient)
9. [nuPlan](https://y30n9ju1v.github.io/posts/papers/nuplan) → [NAVSIM](https://y30n9ju1v.github.io/posts/papers/navsim) → [Bench2Drive](https://y30n9ju1v.github.io/posts/papers/bench2drive-multi-ability-benchmarking-e2e-autonomous-driving) — 평가 체계
10. [CLIP](https://y30n9ju1v.github.io/posts/papers/clip-learning-transferable-visual-models-from-natural-language-supervision) — 이미지·텍스트 대조 학습, zero-shot 전이 + 생성 모델 텍스트 조건화 기반
11. [DDPM](https://y30n9ju1v.github.io/posts/papers/denoising-diffusion-probabilistic-models) — Diffusion 생성 모델 기반
12. [LDM](https://y30n9ju1v.github.io/posts/papers/high-resolution-image-synthesis-with-latent-diffusion-models) — DDPM의 픽셀 공간 한계 극복, Cross-Attention 조건부 생성 (DriveDreamer·MagicDrive 직접 기반)
    → [ControlNet](https://y30n9ju1v.github.io/posts/papers/controlnet-adding-conditional-control-to-text-to-image-diffusion-models) — Zero Convolution으로 LDM에 공간 조건(엣지·깊이·포즈) 주입, 조건부 생성 제어 강화 (ICCV 2023)
13. [GAIA-1](https://y30n9ju1v.github.io/posts/papers/gaia-1) — 생성형 World Model로의 패러다임 전환
14. [DriveDreamer](https://y30n9ju1v.github.io/posts/papers/drivedreamer) — 실제 데이터 기반 합성 데이터 생성
15. [MagicDrive](https://y30n9ju1v.github.io/posts/papers/magicdrive-street-view-generation-3d-geometry-control) — 3D 기하 조건 기반 멀티카메라 생성 (CLIP text embedding 활용)
16. [DriveArena](https://y30n9ju1v.github.io/posts/papers/drivearena) — 생성형 폐루프 시뮬레이션의 통합

### 3D 장면 표현에 집중한다면
1. [NeRF](https://y30n9ju1v.github.io/posts/papers/nerf-representing-scenes-as-neural-radiance-fields-for-view-synthesis) — 신경 렌더링 원리
2. [Instant-NGP](https://y30n9ju1v.github.io/posts/papers/instant-ngp-multiresolution-hash-encoding) — NeRF 가속 핵심 기법
3. [3DGS](https://y30n9ju1v.github.io/posts/papers/3d-gaussian-splatting) — 실시간 렌더링 혁신
4. [EmerNeRF](https://y30n9ju1v.github.io/posts/papers/emernerf) — 자기지도 동적 장면 분해 (NeRF 기반 AV 적용 교두보)
5. [4D-GS](https://y30n9ju1v.github.io/posts/papers/4d-gaussian-splatting) — 동적 장면으로 확장
6. [3D Gaussian Ray Tracing](https://y30n9ju1v.github.io/posts/papers/3d-gaussian-ray-tracing) + [3DGUT](https://y30n9ju1v.github.io/posts/papers/3dgut-enabling-distorted-cameras-and-secondary-rays-in-gaussian-splatting) — 렌더링 품질 확장
7. [Street Gaussians](https://y30n9ju1v.github.io/posts/papers/street-gaussians-modeling-dynamic-urban-scenes) + [DrivingGaussian](https://y30n9ju1v.github.io/posts/papers/driving-gaussian-composite-gaussian-splatting) + [OmniRe](https://y30n9ju1v.github.io/posts/papers/omnire-omni-urban-scene-reconstruction) — AV 적용 (OmniRe: holistic 통합)
8. [GaussianWorld](https://y30n9ju1v.github.io/posts/papers/gaussianworld-gaussian-world-model-for-streaming-3d-occupancy-prediction) — 3DGS를 occupancy 예측에 연결, World Model 기반 스트리밍 시간 모델링

### Neural Simulation에 집중한다면
1. [LiDARsim](https://y30n9ju1v.github.io/posts/papers/lidarsim-realistic-lidar-simulation-by-leveraging-the-real-world) — 실제 데이터 기반 LiDAR 시뮬의 선구, raydrop ML 학습 (CVPR 2020)
2. [UniSim](https://y30n9ju1v.github.io/posts/papers/unisim-neural-closed-loop-sensor-simulator) — NeRF 기반 카메라+LiDAR 동시 시뮬, 장면 편집 지원
3. [NeuRAD](https://y30n9ju1v.github.io/posts/papers/neurad-neural-rendering-for-autonomous-driving) — 범용 AD NeRF 시뮬 (다중 데이터셋, 센서 물리 모델링)
4. [HUGSIM](https://y30n9ju1v.github.io/posts/papers/hugsim-real-time-photorealistic-closed-loop-simulator) — 3DGS 기반 실시간 클로즈드루프 시뮬레이터 통합

### 두 분야의 교차점을 빠르게 파악한다면
[Transformer](https://y30n9ju1v.github.io/posts/papers/attention-is-all-you-need) → [ViT](https://y30n9ju1v.github.io/posts/papers/vit-an-image-is-worth-16x16-words) → [BEVFormer](https://y30n9ju1v.github.io/posts/papers/bevformer) → [3DGS](https://y30n9ju1v.github.io/posts/papers/3d-gaussian-splatting) → [4D-GS](https://y30n9ju1v.github.io/posts/papers/4d-gaussian-splatting) → [Street Gaussians](https://y30n9ju1v.github.io/posts/papers/street-gaussians-modeling-dynamic-urban-scenes) → [HUGSIM](https://y30n9ju1v.github.io/posts/papers/hugsim-real-time-photorealistic-closed-loop-simulator) → [NAVSIM](https://y30n9ju1v.github.io/posts/papers/navsim)

### World Model 계보를 따라간다면
**RL 원류**: [DQN](https://y30n9ju1v.github.io/posts/papers/dqn-playing-atari-with-deep-reinforcement-learning) → [PPO](https://y30n9ju1v.github.io/posts/papers/proximal-policy-optimization) → [UniAD](https://y30n9ju1v.github.io/posts/papers/uniad-planning-oriented-autonomous-driving) → [GAIA-1](https://y30n9ju1v.github.io/posts/papers/gaia-1)

**생성형 계보**: [Transformer](https://y30n9ju1v.github.io/posts/papers/attention-is-all-you-need) → [DDPM](https://y30n9ju1v.github.io/posts/papers/denoising-diffusion-probabilistic-models) → [LDM](https://y30n9ju1v.github.io/posts/papers/high-resolution-image-synthesis-with-latent-diffusion-models) → [DriveDreamer](https://y30n9ju1v.github.io/posts/papers/drivedreamer) → [MagicDrive](https://y30n9ju1v.github.io/posts/papers/magicdrive-street-view-generation-3d-geometry-control) → [DriveArena](https://y30n9ju1v.github.io/posts/papers/drivearena) → [UniSim](https://y30n9ju1v.github.io/posts/papers/unisim-neural-closed-loop-sensor-simulator) → [HUGSIM](https://y30n9ju1v.github.io/posts/papers/hugsim-real-time-photorealistic-closed-loop-simulator)

**3DGS World Model 계보**: [3DGS](https://y30n9ju1v.github.io/posts/papers/3d-gaussian-splatting) → [4D-GS](https://y30n9ju1v.github.io/posts/papers/4d-gaussian-splatting) → [GaussianWorld](https://y30n9ju1v.github.io/posts/papers/gaussianworld-gaussian-world-model-for-streaming-3d-occupancy-prediction) — 3D Gaussian이 장면 표현을 넘어 World Model로 진화하는 계보
