---
title: "Scalability in Perception for Autonomous Driving: Waymo Open Dataset"
date: 2026-04-10T09:00:00+09:00
draft: false
categories: ["Papers", "Autonomous Driving", "Benchmark & Dataset"]
tags: ["Autonomous Driving", "Dataset", "LiDAR", "3D Object Detection", "Multi-Object Tracking"]
---

## 개요

- **저자**: Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurélien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo 외 다수 (Waymo LLC, Google LLC)
- **발행년도**: 2020 (arXiv: 1912.04838v7)
- **주요 내용**: 자율주행 연구를 위한 대규모 멀티모달 카메라-LiDAR 데이터셋인 Waymo Open Dataset을 소개한다. 기존 데이터셋보다 15배 이상 지리적으로 다양하며, 1150개 장면에 약 1200만 개의 3D LiDAR 주석과 약 1200만 개의 카메라 주석을 포함한다.

## 목차

- Chapter 1: Introduction
- Chapter 2: Related Work
- Chapter 3: Waymo Open Dataset (센서 사양, 좌표계, 그라운드 트루스 레이블, 센서 데이터, 데이터셋 분석)
- Chapter 4: Tasks (객체 검출, 객체 추적)
- Chapter 5: Experiments (베이스라인 3D 검출, 2D 검출, 멀티 객체 추적, 도메인 갭, 데이터셋 크기)
- Chapter 6: Conclusion

## Chapter 1: Introduction

**요약**

자율주행 기술 발전을 가속하기 위해 Waymo는 가장 크고 다양한 멀티모달 자율주행 데이터셋을 공개한다. 기존 자율주행 데이터셋들은 규모와 지리적 다양성이 제한적이어서, 학습 도메인과 운영 환경 사이의 일반화 문제가 있었다. 이 논문은 여러 고해상도 카메라와 고품질 LiDAR 스캐너로 수집된 데이터로 구성된 새로운 대규모 데이터셋을 소개한다.

**핵심 개념**

- **Waymo Open Dataset**: 1150개 장면(각 20초), 산업용 LiDAR 5대 + 고해상도 카메라 5대로 구성
- **규모**: 약 1200만 개의 3D LiDAR 박스 주석, 약 1200만 개의 카메라 박스 주석, 113k LiDAR 추적 트랙, 250k 카메라 이미지 트랙
- **지리적 다양성**: 샌프란시스코, 피닉스, 마운틴뷰 3개 도시에서 수집, 기존 최대 카메라+LiDAR 데이터셋 대비 15배 더 다양한 지역 커버리지

## Chapter 2: Related Work

**요약**

자율주행 관련 공개 데이터셋들을 비교 분석한다. KITTI(22개 장면), nuScenes(1000개 장면), Argoverse(113개 장면) 등 기존 데이터셋들과 비교하여 Waymo Open Dataset의 차별점을 제시한다. 특히 지도 정보, 3D 주석 수, 방문 면적 등 여러 측면에서 기존 데이터셋을 압도한다.

**핵심 개념**

- **KITTI**: 22개 장면, 80K 3D 박스, LiDAR 1대 — 자율주행 벤치마크의 표준이지만 규모가 작음
- **nuScenes**: 1000개 장면, 1.4M 3D 박스, 방문 면적 5km² — 지도 정보 포함하지만 지리적 다양성 제한
- **Argoverse**: 113개 장면, 993K 3D 박스 — 상세 HD 맵 제공하지만 LiDAR 1대 사용
- **Waymo Open Dataset**: 1150개 장면, 12M 3D 박스, 방문 면적 76km² — 압도적인 규모와 지리적 다양성

## Chapter 3: Waymo Open Dataset

### 3.1 센서 사양 (Sensor Specifications)

**요약**

데이터 수집에는 5대의 고품질 LiDAR와 5대의 고해상도 핀홀 카메라가 사용된다. LiDAR는 전방(TOP), 전면(FRONT), 측면(SIDE_LEFT, SIDE_RIGHT)에 배치되며, 카메라는 전방(F), 전방좌(FL), 전방우(FR), 측면좌(SL), 측면우(SR)에 배치된다.

**핵심 개념**

- **TOP LiDAR**: 수직 FOV [-17.6°, +2.4°], 범위 75m (제한), 초당 2번 회전
- **FRONT/SIDE LiDAR**: 수직 FOV [-90°, 30°], 범위 20m
- **카메라 이미지**: 전방 카메라 1920×1280, 측면 카메라 1920×1040, 수평 FOV ±25.2°
- **롤링 셔터**: 카메라는 롤링 셔터 방식으로 촬영되어 LiDAR와의 동기화를 위한 프로젝션 보정이 필요

### 3.2 좌표계 (Coordinate Systems)

**요약**

데이터셋에서 사용되는 4가지 좌표계를 정의한다. 모든 좌표계는 오른손 법칙을 따르며, 데이터셋 내 임의의 두 프레임 사이의 변환 정보가 포함된다.

**핵심 개념**

- **Global frame**: 동북방향 좌표계, Z축이 중력 반대 방향
- **Vehicle frame**: 차량 중심, X축 전방, Y축 좌측, Z축 상방
- **Sensor frame**: 각 센서별 독립 좌표계, 차량 프레임으로의 변환 행렬(extrinsics) 제공
- **Image frame**: 각 카메라 이미지의 2D 좌표계

**LiDAR 구형 좌표 변환 수식**

$$\text{range} = \sqrt{x^2 + y^2 + z^2}$$

**수식 설명**
- **range**: LiDAR 센서로부터 포인트까지의 거리 (단위: m)
- **x, y, z**: LiDAR Cartesian 좌표계에서의 3D 위치값

$$\text{azimuth} = \text{atan2}(y, x)$$

**수식 설명**
- **azimuth**: 수평면에서의 각도 (방위각). atan2는 x, y를 이용해 -π ~ π 범위의 각도를 계산하는 함수

$$\text{inclination} = \text{atan2}(z, \sqrt{x^2 + y^2})$$

**수식 설명**
- **inclination**: 수평면으로부터의 수직 각도 (앙각). z값과 수평거리의 비를 이용해 계산

### 3.3 그라운드 트루스 레이블 (Ground Truth Labels)

**요약**

차량, 보행자, 자전거, 표지판에 대한 고품질 수동 주석을 제공한다. LiDAR 레이블은 7-DOF 3D 바운딩 박스(위치, 크기, 방향)로, 카메라 레이블은 4-DOF 2D 바운딩 박스로 구성된다.

**핵심 개념**

- **7-DOF 3D 바운딩 박스**: (cx, cy, cz, l, w, h, θ) — 중심 좌표, 길이/너비/높이, 방향각
- **추적 ID**: 모든 그라운드 트루스 박스에 추적 ID가 부여되어 시간에 따른 동일 객체 매칭 가능
- **난이도 레벨**: LEVEL_1(쉬움), LEVEL_2(누적, LEVEL_1 포함) — KITTI와 유사한 2단계 난이도 시스템
- **레이블 품질**: 전문 주석자가 제작하고 다중 검증 단계를 거쳐 높은 품질 보장

### 3.4 센서 데이터 (Sensor Data)

**요약**

LiDAR 데이터는 레인지 이미지 형식으로 인코딩되어 제공된다. 각 LiDAR 포인트는 range, intensity, elongation 및 vehicle pose를 포함한다. 카메라 데이터는 JPEG 압축 이미지로 제공되며, 롤링 셔터 보정 정보가 포함된다.

**카메라-LiDAR 동기화 수식**

$$\text{sync\_accuracy} = \text{camera\_center\_time} - \text{frame\_start\_time} - \text{camera\_center\_offset} / 360° \times 0.1s$$

**수식 설명**
- **camera_center_time**: 이미지 중심 픽셀의 노출 시간
- **frame_start_time**: 해당 데이터 프레임의 시작 시간
- **camera_center_offset**: 각 카메라 센서 프레임의 +x 축 오프셋 (예: FRONT 카메라 90°, FRONT_LEFT 카메라 90°+45°)
- 동기화 오차는 [-6ms, 7ms] 범위, 99.7% 신뢰도

**핵심 개념**

- **레인지 이미지**: LiDAR 포인트를 이미지 형태로 표현, 각 픽셀이 하나의 LiDAR 리턴에 해당
- **Elongation**: 레이저 펄스의 시간 폭 연장 — 먼지, 비, 안개 같은 부유물 분류에 유용
- **Rolling shutter projection**: 롤링 셔터로 촬영된 카메라 이미지에 LiDAR 포인트를 정확히 매핑하는 기법

### 3.5 데이터셋 분석 (Dataset Analysis)

**요약**

데이터셋은 교외/도심 환경, 낮/밤/새벽 등 다양한 시간대에서 수집된 장면으로 구성된다. 지리적 커버리지 지표로 150m 가시거리에서의 희석된 에고 포즈의 합집합 면적을 사용한다.

**핵심 개념**

- **지리적 커버리지**: 피닉스 40km², 샌프란시스코 36km² 커버 — 기존 데이터셋 대비 15.2배 우수
- **시간대 다양성**: 낮(Day), 밤(Night), 새벽(Dawn) 등 다양한 조명 조건 포함
- **훈련/검증/테스트 분할**: 훈련 798개, 검증 202개, 테스트 150개 시퀀스

## Chapter 4: Tasks

### 4.1 객체 검출 (Object Detection)

**요약**

2D 및 3D 객체 검출 태스크를 정의하며, 새로운 평가 지표인 APH(Average Precision Heading)를 도입한다. 기존 AP는 heading 정보를 반영하지 못하는 한계가 있었다.

**AP 및 APH 수식**

$$\text{AP} = 100 \int_0^1 \max\{p(r') \mid r' \geq r\} \, dr$$

**수식 설명**
- **AP (Average Precision)**: 재현율(r) 전 구간에 걸쳐 최대 정밀도를 적분한 값 (0~100 스케일)
- **p(r)**: P/R 커브 — r은 재현율(recall), p는 정밀도(precision)
- **∫**: 재현율 0부터 1까지 적분 → 면적을 구해 전체 성능을 하나의 숫자로 표현

$$\text{APH} = 100 \int_0^1 \max\{h(r') \mid r' \geq r\} \, dr$$

**수식 설명**
- **APH (Average Precision weighted by Heading)**: heading 정확도로 가중된 AP
- **h(r)**: heading 정확도 가중치가 적용된 정밀도. 가중치는 $\min(\{|\theta - \hat{\theta}|, 2\pi - |\theta - \hat{\theta}|\}) / \pi$
  - **θ**: 예측된 heading 각도
  - **$\hat{\theta}$**: 실제 ground truth heading 각도
  - heading이 완벽히 맞으면 가중치 1, 180° 틀리면 가중치 0

**핵심 개념**

- **IoU 기반 매칭**: 차량/보행자 0.7 IoU, 자전거 0.5 IoU로 매칭 — 예측과 ground truth 박스의 겹침 비율
- **헝가리안 매칭**: 예측-ground truth 쌍 매칭에 헝가리안 알고리즘 사용
- **2D 카메라 검출**: LiDAR 데이터 사용 없이 단일 카메라 이미지만으로 2D 바운딩 박스 예측

### 4.2 객체 추적 (Object Tracking)

**요약**

멀티 객체 추적(MOT)은 장면 내 객체들의 identity, 위치, 속성을 시간에 따라 추적하는 태스크이다. MOTA와 MOTP를 최종 평가 지표로 사용한다.

**MOTA 및 MOTP 수식**

$$\text{MOTA} = 100 - 100 \frac{\sum_t (m_t + \text{fp}_t + \text{mme}_t)}{\sum_t g_t}$$

**수식 설명**
- **MOTA (Multiple Object Tracking Accuracy)**: 추적 정확도 (높을수록 좋음, 최대 100)
- **$m_t$**: 시간 t에서의 miss (탐지 실패)
- **$\text{fp}_t$**: 시간 t에서의 false positive (오탐지)
- **$\text{mme}_t$**: 시간 t에서의 mismatch (ID 전환 오류)
- **$g_t$**: 시간 t에서의 ground truth 객체 수

$$\text{MOTP} = 100 \frac{\sum_{i,t} d_t^i}{\sum_t c_t}$$

**수식 설명**
- **MOTP (Multiple Object Tracking Precision)**: 매칭된 쌍들의 위치 정밀도
- **$d_t^i$**: 시간 t에서 i번째 매칭 쌍의 거리 (1 - IoU)
- **$c_t$**: 시간 t에서 매칭된 쌍의 수

**핵심 개념**

- **mismatch**: ground truth 객체가 이전과 다른 트랙에 매칭되면 1로 계산 — ID 전환 오류 측정
- **MOTA 선택 기준**: 모든 점수 임계값에서 MOTA를 계산하고 최고값을 최종 지표로 사용

## Chapter 5: Experiments

### 5.1 3D 객체 검출 베이스라인

**요약**

PointPillars 모델을 재구현하여 3D 객체 검출 베이스라인을 제시한다. 단일 프레임 LiDAR 데이터를 사용하며, 차량과 보행자에 대한 성능을 평가한다.

**핵심 개념**

- **PointPillars**: LiDAR 포인트 클라우드를 Birds Eye View (BEV) 가상 이미지로 변환 후 CNN 처리
- **Voxel 크기**: 차량/보행자 0.33m, 그리드 범위 [-85m, 85m] (X), [-85m, 85m] (Y), [-3m, 3m] (Z)
- **베이스라인 성능**: 차량 APH 79.1 / 71.0 (LEVEL_1/2), 보행자 APH 56.1 / 51.1 (LEVEL_1/2)
- **2D 검출**: ResNet-101 기반 Faster R-CNN, COCO 데이터셋 사전학습 후 파인튜닝

### 5.3 도메인 갭 (Domain Gap)

**요약**

샌프란시스코(SF)와 피닉스+마운틴뷰(SUB)의 데이터를 분리하여 도메인 갭 실험을 수행한다. 데이터셋 간 도메인 갭이 두드러지게 존재함을 확인한다.

**핵심 개념**

- **도메인 갭**: SF 데이터로 학습 후 SF 평가 vs. SUB 데이터 평가 시 APH 차이 7.6 발생
- **보행자 도메인 갭**: SUB(피닉스+마운틴뷰)에 보행자 수가 적어 SF 대비 큰 차이 발생
- **반지도학습/비지도학습 기회**: 도메인 갭 해소를 위한 반지도학습, 비지도 도메인 적응 연구 기회 제공

### 5.4 데이터셋 크기 효과 (Dataset Size)

**요약**

훈련 데이터 크기에 따른 성능 변화를 분석한다. 10%~100% 데이터로 학습하며 Validation set에서 평가한다.

**핵심 개념**

- 데이터가 많을수록 꾸준히 성능 향상 — 10% → 100% 사용 시 차량 APH 29.7 → 49.4 (LEVEL_2)
- 대규모 데이터셋이 자율주행 모델 성능 향상에 직접적으로 기여함을 실증

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **LiDAR 레인지 이미지** | LiDAR 포인트를 2D 이미지 형태로 표현, 표준 3D 포인트셋 외 다른 입력 형식 연구 가능 |
| **APH (Average Precision Heading)** | 방향각 정확도를 가중치로 사용하는 새로운 검출 지표 |
| **지리적 커버리지 지표** | 150m 가시거리 희석 에고 포즈의 합집합 면적으로 측정 |
| **도메인 갭** | 도시마다 환경, 보행자 분포 등이 달라 학습-평가 도메인 불일치 발생 |
| **Rolling Shutter 보정** | 카메라 롤링 셔터 효과를 LiDAR와 정확히 동기화하는 기법 |
| **7-DOF 3D 바운딩 박스** | 3D 위치(3), 크기(3), 방향각(1)으로 객체를 표현하는 방식 |
| **MOTA/MOTP** | 다중 객체 추적의 정확도(탐지+ID 유지)와 위치 정밀도를 측정하는 지표 |

## 결론 및 시사점

Waymo Open Dataset은 기존 자율주행 데이터셋의 한계(규모, 지리적 다양성, 멀티모달 정합성)를 극복한 대규모 공개 데이터셋이다.

**주요 기여**
- 1150개 장면, 20초씩, 10Hz 수집 — 약 1200만 개의 3D/2D 주석
- 3개 도시(샌프란시스코, 피닉스, 마운틴뷰)에 걸친 76km² 지리적 커버리지
- APH라는 새로운 평가 지표 도입으로 방향 예측 정확도까지 반영
- LiDAR-카메라 간 정밀 동기화 및 롤링 셔터 보정 정보 제공

**실무적 시사점**
- **도메인 적응 연구**: 도시 간 도메인 갭이 확인되어 반지도학습/비지도 도메인 적응 연구의 필요성 제기
- **데이터 효율**: 데이터 크기와 성능이 비례하여 더 많은 레이블 데이터 확보의 중요성 재확인
- **센서 퓨전**: LiDAR와 카메라 레이블의 연계로 멀티모달 융합 연구 기반 제공
- **향후 계획**: 지도 정보, 비레이블 데이터 추가, 행동 예측/계획 등 다양한 태스크 지원 예정
