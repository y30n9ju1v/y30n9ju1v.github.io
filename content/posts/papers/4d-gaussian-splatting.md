---
title: "4D Gaussian Splatting for Real-Time Dynamic Scene Rendering"
date: 2026-04-14T16:00:00+09:00
draft: false
categories: ["Papers"]
tags: ["4DGS", "Gaussian Splatting", "Dynamic Scene", "Novel View Synthesis", "Real-Time Rendering", "HexPlane", "Deformation Field"]
---

## 개요

- **저자**: Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, Xinggang Wang
- **소속**: Huazhong University of Science and Technology, Huawei Inc.
- **발행년도**: 2023 (arXiv:2310.08528, ICLR 2024)
- **프로젝트 페이지**: https://guanjunwu.github.io/4dgs/
- **주요 내용**: 정적 장면에 특화된 3D Gaussian Splatting(3DGS)을 동적 장면으로 확장한 **4D Gaussian Splatting(4D-GS)** 제안. 하나의 정규(canonical) 3D Gaussian 집합을 유지하면서, 각 타임스텝마다 **Gaussian Deformation Field Network**로 위치·회전·크기 변형을 예측하는 방식. 합성 데이터셋에서 82 FPS @ 800×800, 실제 데이터셋에서 30 FPS @ 1352×1014를 달성하여 실시간 동적 장면 렌더링의 새 기준을 제시.

## 목차

- Section 1: Introduction — 동적 장면 렌더링의 과제와 4D-GS 핵심 아이디어
- Section 2: Related Works — NVS, 동적 NeRF, Point Cloud 기반 렌더링
- Section 3: Preliminary — 3DGS 표현과 동적 NeRF 수식 정리
- Section 4: Method — 4D-GS 프레임워크, Gaussian Deformation Field Network, 최적화
- Section 5: Experiment — 합성/실제 데이터셋 평가, 소거 실험, 토론, 한계
- Section 6: Conclusion — 결론

---

## Section 1: Introduction

**요약**

동적 장면의 Novel View Synthesis(NVS)는 두 가지 상충되는 요구를 동시에 만족해야 합니다:
1. **정확성**: 복잡한 모션을 정밀하게 모델링
2. **효율성**: 실시간 렌더링 속도 유지

기존 방법들의 한계:
- **NeRF 기반 동적 방법** (HyperNeRF, TiNeuVox 등): 높은 품질이지만 학습/렌더링 속도 느림
- **3D-GS (정적)**: 실시간 렌더링 가능하지만 동적 장면에 적용하면 각 프레임마다 독립적인 3DGS가 필요 → 메모리/저장 비용 선형 증가

4D-GS의 핵심 아이디어:
```
하나의 Canonical 3D Gaussians G
         +
Gaussian Deformation Field F (타임스텝 t에 따른 변형 예측)
         ↓
Deformed 3D Gaussians G' = ΔG + G  (각 타임스텝)
         ↓
실시간 Splatting 렌더링
```

**메모리 복잡도**: O(N + F) — N은 Gaussian 수, F는 Deformation Field 파라미터 수. 3D-GS를 프레임별로 복제하는 O(tN) 대비 대폭 절감.

**핵심 개념**

- **Canonical Space**: 모든 타임스텝의 기준이 되는 정규 공간. 객체의 "기본 형태"가 이 공간에 저장됨. 각 타임스텝의 변형은 이 기본 형태로부터의 차이(deformation)로 표현
- **Deformation Field**: 입력 위치와 시간 t를 받아 해당 시점의 변형량(Δ위치, Δ회전, Δ크기)을 출력하는 함수. NeRF의 world-to-canonical 매핑을 Gaussian에 적용한 개념

---

## Section 2: Related Works

**요약**

동적 NVS 방법의 세 가지 흐름:

| 접근 방식 | 예시 | 특징 | 한계 |
|---------|------|------|------|
| Canonical 매핑 | HyperNeRF, NeRF-based | 정규 공간→타임스텝 역매핑 | 렌더링 속도 느림 |
| Time-aware 볼륨 렌더링 | TiNeuVox, K-Planes | 4D 복셀/플레인 기반 | 메모리 소비 큼 |
| Flow 기반 | NSFF | 프레임 간 광학 흐름으로 합성 | 장거리 변형에 취약 |

3D-GS(정적)는 명시적 표현과 미분 가능 splatting으로 실시간 렌더링을 달성하지만, 동적 장면에는 바로 적용 불가. **DynamicGaussian**은 각 3D Gaussian을 프레임별로 독립 추적하여 메모리가 선형 증가하는 반면, 4D-GS는 deformation field로 이를 해결합니다.

---

## Section 3: Preliminary

**요약**

### 3.1 3D Gaussian Splatting 복습

각 3D Gaussian은 공분산 행렬 Σ와 중심점 X로 표현됩니다:

$$G(X) = e^{-\frac{1}{2} X^T \Sigma^{-1} X}$$

**수식 설명**:
- **$G(X)$**: 위치 X에서의 Gaussian 밀도값 (0~1 사이)
- **$X$**: 중심점으로부터의 상대 위치 벡터
- **$\Sigma$**: 공분산 행렬 — Gaussian의 방향과 크기를 결정하는 3×3 행렬
- **$e^{-\frac{1}{2}(\cdot)}$**: 중심에서 멀어질수록 밀도가 지수적으로 감소
- 의미: 3D 공간에서 중심 X를 기준으로 타원형으로 분포하는 밀도 함수

공분산 행렬은 최적화를 위해 스케일 행렬 S와 회전 행렬 R로 분해됩니다:

$$\Sigma = RSS^T R^T$$

**수식 설명**:
- **$R$**: 3D 회전 행렬 (Gaussian의 방향)
- **$S$**: 대각 스케일 행렬 (Gaussian의 x/y/z 축 크기)
- 분해 이유: Σ를 직접 최적화하면 양정치(positive definite) 제약 위반 가능 → R, S로 분리하면 항상 유효한 Σ 생성

카메라 좌표계에서의 투영 공분산:

$$\Sigma' = JW\Sigma W^T J^T$$

**수식 설명**:
- **$W$**: 월드→카메라 변환 행렬
- **$J$**: affine 근사의 야코비안 행렬 (투영 변환의 선형화)
- **$\Sigma'$**: 2D 이미지 평면에서의 공분산 (타원형 splat의 모양 결정)

렌더링 블렌딩 수식:

$$\hat{C} = \sum_{i \in N} c_i \alpha_i \prod_{j=1}^{i-1}(1-\alpha_j)$$

**수식 설명**:
- **$\hat{C}$**: 최종 픽셀 색상
- **$c_i$**: i번째 Gaussian의 색상 (Spherical Harmonics로 표현)
- **$\alpha_i$**: i번째 Gaussian의 불투명도 × 2D Gaussian 밀도값
- **$\prod_{j=1}^{i-1}(1-\alpha_j)$**: i번째 Gaussian에 도달하기까지 앞 Gaussian들을 통과한 빛의 투과율
- 의미: 깊이 순으로 정렬된 Gaussian들의 알파 합성 (앞의 불투명한 Gaussian이 뒤를 가림)

### 3.2 동적 NeRF 수식

동적 NeRF는 8D 입력 (위치 x, 방향 d, 시간 t, 외관 λ)을 색상과 밀도로 매핑합니다:

$$c, \sigma = \mathcal{M}(\mathbf{x}, d, t, \lambda)$$

deformation 기반 방법은 world-to-canonical 매핑을 사용합니다:

$$c, \sigma = \text{NeRF}(\mathbf{x} + \Delta\mathbf{x}, d, \lambda)$$

**수식 설명**:
- **$\Delta\mathbf{x}$**: deformation network φ가 예측하는 위치 변위 벡터
- **$\mathbf{x} + \Delta\mathbf{x}$**: 타임스텝 t에서 canonical space로 역매핑된 위치
- 의미: 움직이는 물체의 현재 위치(x)를 정규 공간의 위치(x+Δx)로 변환해 NeRF에 쿼리

---

## Section 4: Method

**요약**

4D-GS의 전체 파이프라인:

```
입력: 뷰 행렬 M=[R,T], 타임스텝 t
  +
Canonical 3D Gaussians G (위치 x,y,z)
         │
         ▼
┌─────────────────────────────────────┐
│  Gaussian Deformation Field Network │
│                                      │
│  [Spatial-Temporal Structure Encoder H]│
│   ├─ HexPlane R_l(i,j) × 6개        │
│   └─ Tiny MLP φ_d                   │
│           ↓                          │
│   [Multi-head Gaussian Deformation  │
│    Decoder D]                        │
│   ├─ Position head φ_x → Δx,Δy,Δz  │
│   ├─ Rotation head φ_r → Δr         │
│   └─ Scaling head φ_s → Δs          │
└─────────────────────────────────────┘
         │
         ▼
Deformed Gaussians G' = {X', r', s', σ, C}
         │
         ▼
Splatting S → Rendered Image Ĩ
```

### 4.1 4D Gaussian Splatting 프레임워크

주어진 뷰와 타임스텝 t에서 새로운 뷰 이미지 Ĩ는:

$$\tilde{I} = \mathcal{S}(M, \mathcal{G}'), \quad \mathcal{G}' = \Delta\mathcal{G} + \mathcal{G}$$

**수식 설명**:
- **$\mathcal{S}$**: 미분 가능한 3DGS Splatting 렌더러
- **$\mathcal{G}$**: Canonical 3D Gaussians (학습 파라미터)
- **$\Delta\mathcal{G}$**: Gaussian Deformation Field F(G, t)가 예측한 변형량
- **$\mathcal{G}'$**: 타임스텝 t에서의 변형된 Gaussians
- 의미: Canonical Gaussians에 변형을 더해 현재 타임스텝의 Gaussians를 만들고 렌더링

### 4.2 Gaussian Deformation Field Network

**Spatial-Temporal Structure Encoder H**

인접한 3D Gaussian들은 비슷한 공간·시간적 특징을 가집니다. 이를 효율적으로 인코딩하기 위해 **HexPlane** 구조에서 착안한 6개의 다중해상도 플레인 모듈 $R_l(i, j)$을 사용합니다:

$$f_h = \bigcup_l \prod_i \text{interp}(R_l(i,j))$$

$$\{(i,j)\} \in \{(x,y), (x,z), (y,z), (x,t), (y,t), (z,t)\}$$

**수식 설명**:
- **$R_l(i,j)$**: 두 축 i, j로 구성된 2D 플레인의 l번째 해상도 피처 맵 (총 6개 조합)
- **$\text{interp}$**: 연속 좌표에서 플레인 피처를 bi-linear 보간으로 쿼리
- **$\prod_i$**: 각 좌표 쌍의 피처를 element-wise 곱 (Hadamard product)으로 결합
- **$\bigcup_l$**: 모든 해상도 레벨의 피처를 연결(concatenate)
- **$(x,y), (x,z), (y,z)$**: 공간 플레인 3개 — Gaussian의 3D 위치 인코딩
- **$(x,t), (y,t), (z,t)$**: 시공간 플레인 3개 — 시간에 따른 위치 변화 인코딩
- 의미: 6개의 2D 플레인으로 4D(x,y,z,t) 시공간 공간을 분해하여 메모리 효율적으로 인코딩

이후 Tiny MLP φ_d로 특징 집계: $f_d = \phi_d(f_h)$

**Multi-head Gaussian Deformation Decoder D**

인코딩된 피처 f_d로부터 각 Gaussian의 변형량을 예측합니다:

$$(\mathcal{X}', r', s') = (\mathcal{X} + \Delta r, r + \Delta r, s + \Delta s)$$

**수식 설명**:
- **$\phi_x(f_d)$** → **$\Delta\mathcal{X} = (\Delta x, \Delta y, \Delta z)$**: 위치 변위 (Position Head)
- **$\phi_r(f_d)$** → **$\Delta r$**: 회전 변위 (Rotation Head)
- **$\phi_s(f_d)$** → **$\Delta s$**: 크기 변위 (Scaling Head)
- 의미: 3개의 독립적인 MLP 헤드가 각각 위치/회전/크기 변형을 담당하여 복잡한 동작 모델링

### 4.3 최적화

**초기화**: SfM(Structure from Motion) 포인트 클라우드로 3D Gaussian 초기화 → 3000 iter 정적 워밍업 후 deformation field 학습 합류.

**손실 함수**:

$$\mathcal{L} = |\tilde{I} - I| + \mathcal{L}_{tv}$$

**수식 설명**:
- **$|\tilde{I} - I|$**: 렌더링 이미지와 Ground Truth 간의 L1 재구성 손실
- **$\mathcal{L}_{tv}$**: Total Variation 손실 — 그리드 기반 표현의 공간적 평활도 강제
- 의미: 렌더링 품질을 높이면서 deformation field가 급격히 변하지 않도록 정규화

---

## Section 5: Experiment

**요약**

### 5.1 실험 설정

- **합성 데이터셋**: D-NeRF [42] (단안 카메라, 50~200 프레임, 동적 물체)
- **실제 데이터셋 1**: HyperNeRF [39] (1~2대 카메라, 자유로운 카메라 이동)
- **실제 데이터셋 2**: Neu3D [25] (15~20대 카메라, 복잡한 카메라 모션)
- 하드웨어: RTX 3090 단일 GPU

### 5.2 결과

**합성 데이터셋** (Table 1, 800×800 해상도):

| 모델 | PSNR↑ | FPS↑ | 학습시간↓ | 저장용량↓ |
|------|-------|------|---------|---------|
| TiNeuVox-B | 32.67 | 1.5 | 28분 | 48 MB |
| 3D-GS (정적) | 23.19 | 170 | 10분 | 10 MB |
| V4D | 31.34 | 2.08 | 6분 | 377 MB |
| **4D-GS (Ours)** | **34.05** | **82** | **8분** | **18 MB** |

→ **품질(PSNR) 1위 + FPS 2위 + 저장용량 최소 수준** 동시 달성

**실제 데이터셋** (HyperNeRF, 960×540):
- PSNR 25.2, MS-SSIM 0.845로 2위
- 학습 30분, FPS 34로 속도 면에서 경쟁력

**실제 데이터셋** (Neu3D, 1352×1014):
- PSNR 31.15, LPIPS 0.049
- 30 FPS로 유일하게 실시간 달성하면서 경쟁력 있는 품질

### 5.3 소거 실험 (Ablation Study)

| 설정 | PSNR | FPS | 의미 |
|------|------|-----|------|
| HexPlane 제거 | 27.05 | 140 | 공간-시간 인코딩이 품질에 핵심 |
| 초기화 없음 | 31.91 | 79 | 워밍업 없으면 수렴 불안정 |
| φ_d 제거 | 26.67 | 82 | 피처 집계 MLP 없으면 품질 급락 |
| φ_r 제거 | 33.08 | 83 | 회전 헤드가 성능에 중요 |
| **전체 모델** | **34.05** | **82** | 모든 컴포넌트가 기여 |

### 5.4 토론

**3D Gaussian으로 추적**: deformation field가 각 Gaussian의 궤적을 모델링하므로 3D 추적 가능 (Figure 7). FFFDNeRF보다 모노큘러 환경에서도 저메모리로 추적 수행.

**4D Gaussian으로 합성**: 학습된 deformation space에서 동일한 공간에 여러 타임스텝의 Gaussians를 배치하여 4D 합성 가능 (Figure 8).

**렌더링 속도 vs Gaussian 수**: Gaussian이 30,000개 이하이면 90 FPS까지 가능. 수가 늘수록 속도는 선형 감소 (Figure 9).

### 5.5 한계

1. **대규모 모션**: 배경 포인트 부재, 부정확한 카메라 포즈에서 최적화 어려움
2. **모노큘러 분리**: 정적/동적 Gaussian의 분리 지도 없음
3. **도시 규모 확장**: 대량의 Gaussian에 의한 HexPlane 쿼리 연산량 증가

---

## Section 6: Conclusion

**요약**

4D Gaussian Splatting은 실시간 동적 장면 렌더링을 위한 효율적인 명시적 표현입니다. 핵심 기여:

1. **Spatial-Temporal Structure Encoder**: HexPlane 기반으로 인접 Gaussian들의 공간·시간 피처를 효율적으로 인코딩
2. **Multi-head Deformation Decoder**: 위치·회전·크기를 별도 헤드로 예측해 정확한 변형 모델링
3. **O(N+F) 메모리 복잡도**: 프레임 수에 무관한 메모리 사용량
4. **확장성**: 3D 추적, 4D 합성 등 downstream 태스크로 확장 가능

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **Canonical 3D Gaussians G** | 모든 타임스텝의 기준이 되는 정규 Gaussian 집합. 학습 중 유지되는 단일 표현 |
| **Gaussian Deformation Field F** | 타임스텝 t마다 각 Gaussian의 위치·회전·크기 변형량(Δ)을 예측하는 신경망 |
| **HexPlane** | (x,y), (x,z), (y,z), (x,t), (y,t), (z,t) 6개의 2D 플레인으로 4D 시공간을 분해하는 효율적 인코딩 방법 |
| **Multi-head Decoder** | Position/Rotation/Scaling 헤드를 분리하여 독립적으로 변형 예측 |
| **메모리 복잡도 O(N+F)** | N(Gaussian 수) + F(Field 파라미터 수)로 프레임 수 t에 무관. 동적 방법 중 최저 수준 |
| **Total Variation Loss** | 인접 그리드 값의 차이를 최소화하여 HexPlane 피처가 급격히 변하지 않도록 정규화 |

---

## 결론 및 시사점

4D-GS는 **3DGS(2023)**와 **DrivingGaussian(2024)** 사이의 핵심 연결 고리입니다.

**3DGS → 4D-GS → DrivingGaussian의 흐름**:
- **3DGS**: 정적 장면 실시간 렌더링 달성
- **4D-GS**: 단일 deformation field로 동적 물체 모델링 → 메모리 효율적 동적 NVS
- **DrivingGaussian**: 자율주행 장면에서 정적 배경(Incremental 3DGS) + 동적 차량(Composite Gaussian Graph) 분리 모델링

**자율주행 시뮬레이션에서의 의미**:
- HUGSIM 같은 포토리얼리스틱 시뮬레이터는 동적 에이전트를 재현해야 하는데, 4D-GS의 deformation field 개념이 그 기반 기술이 됨
- 실시간 렌더링(30+ FPS) 요건을 만족하면서도 높은 재구성 품질 유지
- 향후 자율주행 World Model(GAIA-1 등)과 결합 시 포토리얼리스틱 신경 시뮬레이터 구축 가능

> **로드맵 상의 위치**: `3DGS (2023)` → **4D-GS (ICLR 2024)** → `DrivingGaussian / HUGS (2024)` → `HUGSIM (2024)`
