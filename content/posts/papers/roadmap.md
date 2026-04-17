---
title: "논문 로드맵: 자율주행 & 3D 장면 표현"
date: 2026-04-17T09:00:00+09:00
draft: false
categories: ["Papers"]
tags: ["Roadmap", "Autonomous Driving", "3DGS", "NeRF", "Overview"]
---

이 포스트는 블로그에 정리된 논문들이 서로 어떻게 연결되는지를 보여주는 로드맵입니다.  
크게 두 줄기—**자율주행 스택**과 **3D 장면 표현**—가 최근 **Neural Simulation**이라는 교차점에서 만납니다.

---

## 전체 흐름 요약

```
[데이터/시뮬레이터]
 nuScenes / Waymo / CARLA / nuPlan
         │
         ▼
[인식·표현: BEV]
 OFT → Lift-Splat-Shoot → BEVDepth → BEVFormer
         │
         ▼
[센서 융합 & E2E]
 TransFuser / BEVFusion
         │
         ▼
[예측·계획]
 UniAD → VAD
         │
         ▼
[평가 & 강화학습]
 DQN (2013) / NAVSIM / nuPlan
         │
         ▼
[World Model]
 GAIA-1 (2023)
         │
         ▼
[3D 장면 표현]
 NeRF
  └─► Mip-NeRF 360
       └─► Instant-NGP
            └─► 3DGS ──────────────────┐
                 ├─► 4D-GS             │
                 ├─► 3DGS-RT           │
                 ├─► 3DGUT             │
                 ├─► DIFIX3D+          │
                 ├─► DrivingGaussian   │
                 └─► HUGS              │
                                       ▼
                          [Neural Simulation]
                           UniSim (2023) — NeRF 기반 멀티센서
                           HUGSIM (2024) — 3DGS 기반 실시간
```

---

## 1. 기반: 데이터셋 & 시뮬레이터

자율주행 연구는 **대규모 실제 데이터**와 **시뮬레이터** 위에서 시작됩니다.

| 논문 | 역할 |
|------|------|
| [nuScenes](./nuscenes-multimodal-dataset-autonomous-driving) | 카메라·LiDAR·레이더 통합 멀티모달 데이터셋 (CVPR 2020) |
| [Waymo Open Dataset](./waymo-open-dataset) | 대규모 카메라+LiDAR 데이터, 15배 지리 다양성 (CVPR 2020) |
| [CARLA](./CARLA-An-Open-Urban-Driving-Simulator) | 오픈소스 도시 주행 시뮬레이터, 모방/강화학습 벤치마크 (CoRL 2017) |
| [nuPlan](./nuPlan) | 클로즈드루프 ML 계획 벤치마크, 1500시간 실제 데이터 (NeurIPS 2021) |

> **흐름**: nuScenes·Waymo가 **인식 연구의 표준 평가셋**이 되고,  
> CARLA는 **폐쇄 루프 시뮬레이션 플랫폼**으로, nuPlan은 **계획 벤치마크**로 자리잡습니다.

---

## 2. 자율주행 스택: 인식 → 예측 → 계획

### 2-1. 인식: BEV 표현

카메라 이미지를 **Bird's-Eye-View(BEV)** 공간으로 변환하는 것이 핵심 과제입니다.

```
단안 카메라                     다중 카메라
OFT (2018) ─────────────────► Lift-Splat-Shoot (2020) ──► BEVDepth (2022) ──► BEVFormer (2022)
(직교 피처 변환)                 (Latent Depth Distribution)  (명시적 깊이 감독)   (Spatial + Temporal Attention)
```

| 논문 | 핵심 아이디어 |
|------|-------------|
| [OFT](./OFT-monocular-3d-object-detection) | 단안 카메라 피처를 직교 BEV 공간으로 투영 (Cambridge, 2018) |
| [Lift-Splat-Shoot](./lift-splat-shoot) | 깊이 분포로 frustum 포인트 클라우드 생성 후 BEV Pillar Pooling, end-to-end 모션 플래닝 (ECCV 2020) |
| [BEVDepth](./bevdepth) | LiDAR 깊이 감독으로 LSS의 깊이 추정 신뢰도 개선 (AAAI 2023) |
| [BEVFormer](./BEVFormer) | 다중 카메라 + 시간 정보를 Transformer로 BEV 통합 (ECCV 2022) |

### 2-2. 센서 융합 & E2E 주행

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
| [TransFuser](./TransFuser) | Transformer self-attention으로 카메라+LiDAR 전역 융합 (TPAMI 2022) |
| [BEVFusion](./bevfusion-multi-task-multi-sensor-fusion) | BEV 공간 멀티센서 융합으로 탐지+분할 멀티태스크 동시 수행, Efficient BEV Pooling (CVPR 2023) |

### 2-3. 통합 예측·계획

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
| [UniAD](./uniad-planning-oriented-autonomous-driving) | 계획 지향 통합 프레임워크, 5개 태스크를 Query로 연결 (CVPR 2023) |
| [VAD](./VAD) | 래스터 맵 → 벡터화 표현, 인스턴스 수준 계획 제약 (2023) |

### 2-4. 평가 & 강화학습 기반

```
DQN (2013) ──► 강화학습 기반 에이전트의 원형 (Atari → AV policy)
NAVSIM (2024) ─► 비반응형 시뮬레이션으로 오픈/클로즈드루프 격차 해소
nuPlan (2022) ─► 클로즈드루프 계획 벤치마크
```

| 논문 | 역할 |
|------|------|
| [DQN](./DQN-playing-atari-with-deep-reinforcement-learning) | Experience Replay + CNN Q-learning, 강화학습 E2E의 출발점 (DeepMind 2013) |
| [NAVSIM](./NAVSIM) | 실제 데이터 기반 비반응형 시뮬 벤치마크, PDMS 지표 (NeurIPS 2024) |

### 2-5. World Model: 생성형 자율주행

인식·계획 파이프라인을 넘어 **세계 자체를 생성 모델로 학습**하는 패러다임 전환입니다.

```
DQN (강화학습 원형)
    │
    ▼
UniAD / VAD (E2E 계획)
    │  "어떻게 주행할지"를 학습
    ▼
GAIA-1 (2023) ─── "세상이 어떻게 작동하는지"를 학습
    ├─ 비디오 + 텍스트 + 액션 → 다음 토큰 예측 (GPT 방식)
    ├─ LLM 스케일링 법칙이 AV World Model에도 적용됨 확인
    └─ Neural Simulator로서 HUGSIM 방향의 생성형 대안 제시
```

| 논문 | 핵심 아이디어 |
|------|-------------|
| [GAIA-1](./GAIA-1) | 비디오·텍스트·액션을 토큰으로 통합한 생성형 AV World Model, 스케일링 법칙 검증 (Wayve 2023) |

---

## 3. 3D 장면 표현: NeRF → 3DGS → 자율주행 적용

### 3-1. 기반 기술

```
NeRF (2020) ─────────────────────── MLP로 5D 연속 함수 표현
    │ 유한 장면만 표현 가능, 무한 야외 장면에 취약
    ▼
Mip-NeRF 360 (2022) ──────────────── 비선형 장면 파라미터화로 무한 장면 표현, 57% MSE 감소
    │ 느린 렌더링 속도가 한계
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
| [NeRF](./nerf-representing-scenes-as-neural-radiance-fields-for-view-synthesis) | 희소 이미지로부터 체적 렌더링으로 새로운 시점 합성 (ECCV 2020) |
| [Mip-NeRF 360](./mip-nerf-360) | contract 함수 기반 비선형 파라미터화로 무한 야외 장면 표현, MSE 57% 감소 (CVPR 2022) |
| [Instant-NGP](./instant-ngp-multiresolution-hash-encoding) | Multiresolution Hash Encoding으로 NeRF 학습 속도 1000배 향상 (SIGGRAPH 2022) |
| [3DGS](./3d-gaussian-splatting) | Gaussian 파티클 + tile-based 래스터화로 실시간 렌더링 (SIGGRAPH 2023) |
| [4D-GS](./4d-gaussian-splatting) | Canonical 3DGS + Gaussian Deformation Field로 실시간 동적 NVS (ICLR 2024) |

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
| [3D Gaussian Ray Tracing](./3d-gaussian-ray-tracing) | GPU BVH 레이 트레이싱으로 반사·굴절·그림자 지원 (SIGGRAPH Asia 2024) |
| [3DGUT](./3dgut-enabling-distorted-cameras-and-secondary-rays-in-gaussian-splatting) | Unscented Transform으로 어안렌즈·롤링셔터 래스터화 (NVIDIA 2025) |
| [DIFIX3D+](./difix3d-plus) | Diffusion 모델로 NeRF/3DGS 아티팩트 후처리 제거 (NVIDIA 2025) |

### 3-3. 자율주행 장면에 3DGS 적용

정적 배경과 동적 객체를 분리하여 자율주행 장면을 재구성합니다. **4D-GS의 deformation field 개념**이 이 단계의 핵심 기반 기술입니다.

```
4D-GS (동적 Gaussian 표현 기반)
    ├─► DrivingGaussian (2024) ─── 정적 Incremental 3DGS + 동적 Composite Graph
    └─► HUGS (2024) ──────────── 의미론적(semantic) + 동적 통합 이해 (RGB만으로)
```

| 논문 | 핵심 아이디어 |
|------|-------------|
| [DrivingGaussian](./driving-gaussian-composite-gaussian-splatting) | Composite 3DGS로 서라운드 동적 장면 재구성 (CVPR 2024) |
| [HUGS](./hugs-holistic-urban-3d-scene-understanding) | 외관·의미·모션 통합 3DGS 이해, LiDAR 불필요 (2024) |

---

## 4. 교차점: Neural Simulation

자율주행 **인식·계획 스택**, **생성형 World Model**, **3DGS 기반 장면 표현**이 한데 모여  
**포토리얼리스틱 클로즈드루프 시뮬레이터**가 탄생합니다.

```
NeRF 기반 장면 재구성 (UniSim)
3DGS 기반 장면 재구성 (4D-GS, DrivingGaussian, HUGS)
         +
자율주행 에이전트 (UniAD, VAD, TransFuser)
         +
클로즈드루프 평가 프레임워크 (nuPlan, NAVSIM)
         +
생성형 World Model (GAIA-1)
         │
         ▼
    UniSim (2023)                        HUGSIM (2024)                    GAIA-1 (2023)
    ─ NeRF 기반 멀티센서 시뮬레이터       ─ 3DGS 기반 포토리얼 시뮬레이터   ─ 생성형 신경 시뮬레이터
    ─ 카메라 + LiDAR 동시 합성            ─ 실시간 클로즈드루프 평가         ─ 무한 시나리오 생성
    ─ 장면 편집 (액터 추가/제거)           ─ 70+ 시퀀스 벤치마크             ─ 텍스트/액션 제어 가능
```

| 논문 | 역할 |
|------|------|
| [UniSim](./unisim-neural-closed-loop-sensor-simulator) | NeRF 기반 카메라+LiDAR 동시 시뮬, 장면 편집 지원 클로즈드루프 시뮬레이터 (CVPR 2023) |
| [HUGSIM](./hugsim-real-time-photorealistic-closed-loop-simulator) | 3DGS 기반 실시간 포토리얼리스틱 클로즈드루프 AV 시뮬레이터 (CVPR 2024) |
| [GAIA-1](./GAIA-1) | 비디오·텍스트·액션 토큰 기반 생성형 World Model, 신경 시뮬레이터 역할 (Wayve 2023) |

---

## 5. 논문 연도별 타임라인

```
2013  DQN ──────────────────────────── 딥 강화학습의 출발
2017  CARLA ───────────────────────── 자율주행 오픈 시뮬레이터
2018  OFT ────────────────────────── 단안 BEV 탐지
2020  NeRF ──────────────────────── 신경 장면 표현
      nuScenes, Waymo ─────────────── 멀티모달 데이터셋
      Lift-Splat-Shoot ────────────── Latent Depth로 BEV, E2E 모션 플래닝
2022  Mip-NeRF 360 ───────────────── 무한 야외 장면을 위한 NeRF 확장
      Instant-NGP ────────────────── Hash Encoding으로 NeRF 1000배 가속
      BEVDepth ───────────────────── LiDAR 감독으로 깊이 신뢰도 개선
      BEVFormer ──────────────────── 멀티카메라 BEV Transformer
      TransFuser ─────────────────── 카메라+LiDAR 융합 E2E
      nuPlan ──────────────────────── 클로즈드루프 계획 벤치마크
2023  BEVFusion ───────────────────── BEV 공간 멀티센서 융합, 멀티태스크
      3DGS ───────────────────────── 실시간 3D Gaussian 렌더링
      UniAD ───────────────────────── 계획 지향 통합 AV
      VAD ──────────────────────────── 벡터화 장면 표현
      UniSim ──────────────────────── NeRF 기반 멀티센서 클로즈드루프 시뮬
      GAIA-1 ──────────────────────── 생성형 AV World Model (LLM 방식)
2024  4D-GS ───────────────────────── 동적 장면 실시간 3DGS (ICLR)
      3D Gaussian Ray Tracing ────── 레이 트레이싱 3DGS
      DrivingGaussian ─────────────── 동적 AV 장면 3DGS
      HUGS ────────────────────────── 의미론적 도시 3DGS
      HUGSIM ──────────────────────── 포토리얼 클로즈드루프 시뮬
      NAVSIM ──────────────────────── 비반응형 AV 벤치마크
2025  3DGUT ───────────────────────── 왜곡 카메라 3DGS
      DIFIX3D+ ────────────────────── Diffusion 기반 3D 아티팩트 제거
```

---

## 6. 추천 읽기 순서

### 자율주행 스택에 집중한다면
1. [nuScenes](./nuscenes-multimodal-dataset-autonomous-driving) — 데이터 기반 이해
2. [OFT](./OFT-monocular-3d-object-detection) → [Lift-Splat-Shoot](./lift-splat-shoot) → [BEVDepth](./bevdepth) → [BEVFormer](./BEVFormer) — BEV 인식 계보
3. [TransFuser](./TransFuser) → [BEVFusion](./bevfusion-multi-task-multi-sensor-fusion) — 센서 융합 E2E
4. [UniAD](./uniad-planning-oriented-autonomous-driving) → [VAD](./VAD) — 통합 계획
5. [nuPlan](./nuPlan) → [NAVSIM](./NAVSIM) — 평가 체계
6. [GAIA-1](./GAIA-1) — 생성형 World Model로의 패러다임 전환

### 3D 장면 표현에 집중한다면
1. [NeRF](./nerf-representing-scenes-as-neural-radiance-fields-for-view-synthesis) — 신경 렌더링 원리
2. [Mip-NeRF 360](./mip-nerf-360) — 무한 야외 장면으로 확장
3. [Instant-NGP](./instant-ngp-multiresolution-hash-encoding) — NeRF 가속 핵심 기법
4. [3DGS](./3d-gaussian-splatting) — 실시간 렌더링 혁신
5. [4D-GS](./4d-gaussian-splatting) — 동적 장면으로 확장
6. [3D Gaussian Ray Tracing](./3d-gaussian-ray-tracing) + [3DGUT](./3dgut-enabling-distorted-cameras-and-secondary-rays-in-gaussian-splatting) — 렌더링 품질 확장
7. [DrivingGaussian](./driving-gaussian-composite-gaussian-splatting) + [HUGS](./hugs-holistic-urban-3d-scene-understanding) — AV 적용
8. [UniSim](./unisim-neural-closed-loop-sensor-simulator) — NeRF 기반 멀티센서 시뮬레이터
9. [HUGSIM](./hugsim-real-time-photorealistic-closed-loop-simulator) — 시뮬레이터 통합

### 두 분야의 교차점을 빠르게 파악한다면
[3DGS](./3d-gaussian-splatting) → [4D-GS](./4d-gaussian-splatting) → [DrivingGaussian](./driving-gaussian-composite-gaussian-splatting) → [HUGSIM](./hugsim-real-time-photorealistic-closed-loop-simulator) → [NAVSIM](./NAVSIM)

### World Model 계보를 따라간다면
[DQN](./DQN-playing-atari-with-deep-reinforcement-learning) → [UniAD](./uniad-planning-oriented-autonomous-driving) → [GAIA-1](./GAIA-1) → [UniSim](./unisim-neural-closed-loop-sensor-simulator) → [HUGSIM](./hugsim-real-time-photorealistic-closed-loop-simulator)
