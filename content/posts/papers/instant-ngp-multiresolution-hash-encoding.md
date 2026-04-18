---
title: "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding"
date: 2026-04-17T08:00:00+09:00
draft: false
categories: ["Papers", "Novel View Synthesis"]
tags: ["NeRF", "Novel View Synthesis", "Neural Rendering", "Hash Encoding", "Real-Time Rendering"]
---

## 개요
- **저자**: Thomas Müller, Alex Evans, Christoph Schied, Alexander Keller (NVIDIA)
- **발행년도**: 2022
- **게재**: ACM Transactions on Graphics, Vol. 41, No. 4, Article 102 (SIGGRAPH 2022)
- **주요 내용**: Multiresolution Hash Encoding이라는 새로운 입력 인코딩 방식을 제안하여, Neural Graphics Primitives(NeRF, SDF, NRC, Gigapixel Image)를 수 초~수 분 내에 훈련 가능하게 만든 논문. GPU 병렬성을 최대한 활용한 CUDA 구현을 통해 기존 대비 수십~수천 배의 속도 향상을 달성함.

## 한계 극복

이 논문은 기존 Neural Graphics Primitives의 훈련 속도 문제를 해결하기 위해 작성되었습니다.

- **기존 한계 1 — 느린 훈련 속도**: NeRF와 같은 신경망 기반 3D 표현은 단일 장면을 학습하는 데 수 시간이 필요했음 (mip-NeRF, NSVF 등)
- **기존 한계 2 — 큰 네트워크 크기**: 표현력을 높이려면 네트워크가 커져야 하고, 그럴수록 훈련/추론이 더 느려지는 딜레마 존재
- **기존 한계 3 — Task-specific 데이터 구조**: Octree 등 기존 적응형 인코딩은 task-specific 구조 수정이 필요하여 범용성이 떨어짐
- **이 논문의 접근 방식**: 학습 가능한 Feature Vector를 저장하는 Multiresolution Hash Table을 보조 데이터 구조로 도입. 네트워크는 작게 유지하면서 표현력은 해시 테이블이 담당. 완전히 CUDA로 구현하여 GPU 병렬성을 극대화함.

## 목차
- Section 1: Introduction
- Section 2: Background and Related Work
- Section 3: Multiresolution Hash Encoding
- Section 4: Implementation
- Section 5: Experiments (Gigapixel Image, SDF, NRC, NeRF)
- Section 6: Discussion and Future Work
- Section 7: Conclusion
- Appendix A-E: Smooth Interpolation, NGLOD Implementation, SDF Training, Baseline MLPs, Accelerated NeRF Ray Marching

---

## Section 1: Introduction

**요약**

컴퓨터 그래픽스에서 Neural Graphics Primitives(NeRF, SDF, NRC 등)는 강력한 3D 표현 방식이지만, 훈련 비용이 크고 평가가 느리다는 문제가 있었습니다. 이 논문은 "입력 인코딩"의 설계를 바꿔 이 문제를 해결합니다.

핵심 아이디어는 입력 좌표 **x**를 직접 신경망에 넣는 대신, 먼저 **해시 테이블에서 특징 벡터를 조회(lookup)**하여 신경망에 전달하는 것입니다. 이를 통해:
- 네트워크 크기를 10~100배 줄일 수 있음
- 더 적은 FLOPs로 동등하거나 더 높은 품질 달성
- GPU 메모리 대역폭을 효율적으로 활용

**핵심 개념**
- **Neural Graphics Primitives**: 신경망으로 표현되는 그래픽스 함수들 — 이미지, 3D 형상(SDF), 라디언스 필드(NeRF), 조명 캐시(NRC) 등
- **입력 인코딩(Input Encoding)**: 입력 **x**를 더 표현력 있는 특징 공간으로 변환하는 함수 enc(**x**; θ). 네트워크가 고주파 신호를 학습하는 데 도움을 줌
- **Adaptivity**: 장면의 세밀한 부분에 자동으로 더 많은 파라미터를 할당하는 능력

---

## Section 2: Background and Related Work

**요약**

기존 입력 인코딩 방식들을 정리하고, 이 논문이 어떻게 다른지 설명합니다.

**핵심 개념**
- **One-hot / Dense Grid Encoding**: 단순히 그리드 셀 인덱스를 one-hot으로 표현. 파라미터 수가 기하급수적으로 증가하여 고해상도 불가
- **Frequency Encoding (NeRF Positional Encoding)**: 사인/코사인 함수를 이용해 좌표를 주파수 공간으로 변환. 고주파 학습을 가능하게 하지만 큰 MLP가 필요
- **Parametric Encodings**: 학습 가능한 파라미터를 보조 데이터 구조(그리드, 트리 등)에 저장하고, 이를 신경망과 함께 학습
  - **Dense Grid** (ACORN): 전 해상도 그리드에 특징 저장. 정밀하지만 메모리 폭발적 증가
  - **Octree 기반** (NGLOD): 표면 근처만 세밀하게 저장. 정확하지만 SDF 학습에 특화, 범용성 부족
  - **이 논문**: 해시 테이블 + 다중 해상도로 위 두 방식의 장점을 결합

---

## Section 3: Multiresolution Hash Encoding

**요약**

이 논문의 핵심 기여입니다. 입력 좌표 **x** ∈ ℝ^d를 여러 해상도의 격자에 매핑하고, 각 격자의 꼭짓점에서 학습 가능한 특징 벡터를 조회한 뒤, 이를 연결(concatenate)하여 신경망에 입력합니다.

**전체 파이프라인 (5단계)**

1. **Hashing of voxel vertices**: 각 해상도 레벨 ℓ에서 입력 좌표 **x** 주변의 격자 꼭짓점들을 해시 함수로 해시 테이블 인덱스로 변환
2. **Lookup**: 해시 테이블에서 해당 인덱스의 F차원 특징 벡터를 읽어옴
3. **Linear interpolation**: **x**의 격자 내 상대 위치에 따라 d선형(trilinear) 보간
4. **Concatenation**: 모든 L개 해상도 레벨의 특징 벡터와 보조 입력 ξ를 연결
5. **Neural Network**: 연결된 벡터를 MLP에 입력하여 최종 출력 m(**y**; Φ) 계산

**핵심 개념**

- **해상도 레벨 (Resolution Levels)**: 가장 거친 해상도 N_min부터 가장 세밀한 N_max까지 L개의 레벨이 기하급수적으로 증가

- **성장 인자 b**:

$$b = \exp\left(\frac{\ln N_{\max} - \ln N_{\min}}{L-1}\right)$$

**수식 설명**:
- **$b$**: 레벨 간 해상도 성장 비율 (보통 1.26 ~ 2.0)
- **$N_{\min}$**: 가장 거친 해상도 (예: 16)
- **$N_{\max}$**: 가장 세밀한 해상도 (예: 512 ~ 524288)
- **$L$**: 총 레벨 수 (보통 16)
- 이 수식으로 각 레벨의 해상도 $N_\ell = \lfloor N_{\min} \cdot b^\ell \rfloor$가 결정됨

- **해시 함수**: 격자 꼭짓점 정수 좌표 **x**⌊⌋를 해시 테이블 인덱스로 변환

$$h(\mathbf{x}) = \left(\bigoplus_{i=1}^{d} x_i \pi_i\right) \bmod T$$

**수식 설명**:
- **$h(\mathbf{x})$**: 최종 해시 인덱스 (0 ~ T-1 범위)
- **$\oplus$**: 비트 단위 XOR 연산
- **$x_i$**: i번째 차원의 격자 꼭짓점 정수 좌표
- **$\pi_i$**: 각 차원마다 다른 큰 소수 (π₁=1, π₂=2654435761, π₃=805459861). 차원 독립성을 위한 pseudo-random 순열
- **$T$**: 해시 테이블 크기 (튜닝 가능한 하이퍼파라미터, 보통 2^14 ~ 2^24)
- XOR 기반이라 계산이 매우 빠름 (곱셈/나눗셈 불필요)

- **d선형 보간 (d-linear interpolation)**: 각 레벨에서 꼭짓점의 특징 벡터를 **x**의 격자 내 상대 위치 w = **x** - ⌊**x**⌋로 가중 보간

$$\text{보간 가중치: } \mathbf{w}_\ell = \mathbf{x} - \lfloor \mathbf{x}_\ell \rfloor$$

- **해시 충돌 (Hash Collision)**: 서로 다른 공간 위치가 같은 해시 인덱스로 매핑되는 현상. 충돌된 두 위치의 그래디언트가 평균됨. 그러나 다중 해상도 구조 덕분에 충돌은 다른 레벨에서 분산되어 자연스럽게 해결됨 (명시적 충돌 처리 불필요)

- **하이퍼파라미터 요약**:

| 파라미터 | 기호 | 기본값 |
|---------|------|--------|
| 레벨 수 | L | 16 |
| 해시 테이블 크기 | T | 2^14 ~ 2^24 |
| 레벨당 특징 차원 | F | 2 |
| 가장 거친 해상도 | N_min | 16 |
| 가장 세밀한 해상도 | N_max | 512 ~ 524288 |

**해시 충돌의 암묵적 해결 원리**

충돌이 발생해도 잘 동작하는 이유:
1. **다중 해상도**: 거친 레벨은 충돌이 적고 (격자 수 < T), 세밀한 레벨은 충돌이 많지만 각 레벨이 서로 다른 장면 스케일을 담당하여 상호 보완
2. **그래디언트 평균화**: 같은 해시를 공유하는 위치들의 그래디언트가 평균됨 → 더 중요한(방문 빈도 높은) 위치의 그래디언트가 지배적
3. **확률적 분산**: 충돌이 동시에 모든 레벨에서 발생할 확률이 통계적으로 매우 낮음

---

## Section 4: Implementation

**요약**

NVIDIA CUDA로 전체 파이프라인을 완전히 구현하여 GPU 병렬성을 극대화했습니다. 작은 MLP와 효율적인 메모리 접근 패턴이 핵심입니다.

**핵심 개념**

- **Fully-fused CUDA Kernels**: 해시 인코딩 → MLP forward/backward를 하나의 CUDA 커널로 융합. 메모리 왕복을 최소화하여 latency 감소
- **tiny-cuda-nn**: NVIDIA의 소형 신경망 추론 프레임워크. 16개 뉴런 단위로 설계되어 GPU 연산 효율 극대화
- **네트워크 크기**: 2개의 hidden layer, 각 64개 뉴런 (NeRF 기준). 인코딩이 표현력을 담당하므로 작은 네트워크로 충분
- **Half-precision (fp16)**: 해시 테이블 조회에 fp16 원자적 연산 사용. 메모리 대역폭 절감
- **온라인 적응성**: 입력 분포가 바뀌어도 (NRC의 경우 실시간 렌더링 중) 자동으로 현재 중요한 영역에 적응

---

## Section 5: Experiments

**요약**

4가지 Neural Graphics Primitives에 Multiresolution Hash Encoding을 적용하고 기존 방법들과 비교합니다.

### 5.1 Gigapixel Image Approximation

2D 좌표 → RGB 매핑을 학습하여 기가픽셀 이미지를 신경망으로 표현하는 태스크.

- **비교 대상**: ACORN (Martel et al. 2021)
- **결과**: ACORN은 36.9h 훈련 후 PSNR 38.59 dB. 우리 방법은 **2.5분** 만에 동등한 품질 달성, 10~100× 적은 파라미터 사용

### 5.2 Signed Distance Functions (SDF)

3D 형상을 SDF로 표현. f(**x**) = 0이 표면을 나타냄.

- **비교 대상**: NGLOD (Takikawa et al. 2021), Frequency encoding
- **결과**: NGLOD보다 IoU가 유사하거나 높으면서, 훈련 시간은 **수 초** (NGLOD는 수 분)
- 시각적으로 약간의 "grain" microstructure가 있으나 수치는 경쟁적

### 5.3 Neural Radiance Caching (NRC)

실시간 렌더링 중 간접 조명을 캐싱하는 신경망. 1ms/frame 예산.

- **특징**: 온라인 학습 (렌더링 중 실시간 학습), 1ms 예산 내에서 최대한 정확해야 함
- **결과**: 기존 triangle wave encoding 대비 **시각적 품질 크게 향상** (그림자, 세부 조명 표현 개선). FPS는 147 → 133으로 소폭 감소

### 5.4 Neural Radiance and Density Fields (NeRF)

2D 이미지들로부터 3D 볼륨을 재구성하는 태스크.

- **비교 대상**: mip-NeRF (수 시간 훈련), NSVF (수 시간 훈련)
- **결과** (Synthetic NeRF 데이터셋):

| 방법 | 훈련 시간 | PSNR (avg) |
|------|---------|-----------|
| Ours: Hash | 1초 | 21.2 dB |
| Ours: Hash | 1분 | 33.1 dB |
| Ours: Hash | 5분 | 33.2 dB |
| mip-NeRF | ~수 시간 | 33.09 dB |
| NSVF | ~수 시간 | 31.0 dB |

- 5분 훈련으로 수 시간 훈련한 mip-NeRF와 동등한 품질 달성
- Hash encoding 자체가 20~60× 속도 향상의 원인

**수식 — NeRF Volume Rendering**

$$C(\mathbf{r}) = \int_{t_n}^{t_f} T(t)\, \sigma(\mathbf{r}(t))\, \mathbf{c}(\mathbf{r}(t), \mathbf{d})\, dt$$

$$T(t) = \exp\left(-\int_{t_n}^{t} \sigma(\mathbf{r}(s))\, ds\right)$$

**수식 설명**:
- **$C(\mathbf{r})$**: 광선 **r**의 최종 렌더링 색상
- **$\sigma(\mathbf{r}(t))$**: 위치 **r**(t)에서의 밀도 (불투명도). 클수록 빛이 잘 통과 못 함
- **$\mathbf{c}(\mathbf{r}(t), \mathbf{d})$**: 위치 **r**(t)에서 방향 **d**로 보이는 색상
- **$T(t)$**: 광선이 $t_n$에서 $t$까지 도달하는 투과율. 앞에 물체가 있으면 작아짐
- 직관: 광선이 3D 공간을 통과하면서, 각 지점의 색상을 밀도와 투과율로 가중합산

---

## Section 6: Discussion and Future Work

**요약**

논문의 설계 선택들에 대한 심층 분석과 미래 방향을 논의합니다.

**핵심 개념**

- **Concatenation vs. Reduction**: 각 레벨의 특징 벡터를 합산(sum)이 아닌 연결(concatenate)하는 이유 — GPU 병렬처리 효율, 정보 손실 방지
- **해시 함수 선택**: 논문에서 사용한 XOR 기반 해시가 PCG32, space-filling curve 등 대안보다 품질/속도 균형이 좋음
- **Generative Setting**: GAN 등 생성 모델과 결합 시 해시 충돌 처리가 어려울 수 있음 (특징이 규칙 격자에 정렬되지 않기 때문)
- **Attention Mechanism과의 연결**: 주파수 인코딩이 Transformer의 attention에서 유래했듯, 해시 인코딩도 attention 기반 모델에 적용 가능성 있음
- **Microstructure Artifact**: 해시 충돌로 인한 미세 grain 패턴이 SDF에서 약간 보임 → 해시 테이블 조회에 추가적인 smoothness prior 적용 필요성

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **Multiresolution Hash Encoding** | L개의 해상도 레벨 × 각 레벨의 해시 테이블에서 특징 벡터를 조회하고 연결하는 입력 인코딩 |
| **Hash Table (T)** | 학습 가능한 F차원 특징 벡터 T개를 저장하는 배열. Adam optimizer로 학습됨 |
| **Spatial Hashing** | XOR + 소수 곱 기반 해시 함수로 격자 꼭짓점 좌표 → 테이블 인덱스 변환 |
| **Implicit Collision Resolution** | 해시 충돌을 명시적으로 처리하지 않고 그래디언트 평균화로 자연스럽게 해결 |
| **d-linear Interpolation** | 격자 내 상대 위치 기반 보간으로 연속적이고 미분 가능한 인코딩 보장 |
| **Fully-fused CUDA Kernels** | 인코딩 + MLP를 단일 CUDA 커널로 융합하여 메모리 왕복 최소화 |
| **Adaptivity** | 장면의 중요한 영역에 자동으로 더 많은 파라미터가 집중되는 자기 적응 능력 |
| **Neural Graphics Primitives** | NeRF(볼륨 렌더링), SDF(형상 표현), NRC(조명 캐싱), Gigapixel Image(이미지 압축) 4가지 응용 |

**기존 방법들과의 비교 요약**

| 방법 | 파라미터 | 훈련 속도 | 품질 | 범용성 |
|------|---------|---------|------|--------|
| Frequency Encoding (NeRF) | 적음 | 수 시간 | 보통 | 높음 |
| Dense Grid | 매우 많음 | 빠름 | 높음 | 낮음 |
| Octree (NGLOD) | 중간 | 중간 | 높음 (SDF) | 낮음 (SDF 특화) |
| **Multiresolution Hash (Ours)** | **중간** | **수 초~분** | **높음** | **높음** |

---

## 결론 및 시사점

**논문의 결론**

Multiresolution Hash Encoding은 단순하고 범용적인 입력 인코딩 방식으로, 어떤 Neural Graphics Primitives에도 적용 가능합니다. 핵심은:

1. **속도**: 기존 방법 대비 수십~수천 배 빠른 훈련 (수 시간 → 수 초)
2. **품질**: 동등하거나 더 높은 품질 (일부 view-dependent 효과 제외)
3. **범용성**: NeRF, SDF, NRC, Gigapixel Image 모두 동일한 구현으로 처리
4. **단순성**: 해시 테이블 크기 T만 조정하면 품질/속도/메모리 트레이드오프 제어 가능

**실무적 시사점**

- **NeRF 실용화의 전환점**: Instant-NGP는 NeRF를 연구 도구에서 실용적 도구로 전환시킨 논문. 이후 3D Gaussian Splatting(2023)과 함께 real-time 3D reconstruction 분야의 양대 산맥이 됨
- **후속 연구에 미친 영향**: 
  - 3DGS(2023)는 Instant-NGP의 한계(view-dependent effect, 실시간 렌더링)를 극복하기 위해 rasterization 기반 접근 채택
  - NeRF 기반 자율주행 시뮬레이터(EmerNeRF, OmniRe 등)에서 Hash Encoding 기본 채택
  - HuGSim, Difix3D+ 등 최신 논문들도 이 방법의 영향권
- **설계 철학**: "큰 신경망이 모든 것을 학습하게 하는" 방식 대신, "데이터 구조와 신경망의 역할 분담" — 데이터 구조가 위치 정보를, 신경망이 함수 표현을 담당

**한계**

- view-dependent 효과(반사, 광택)가 있는 장면에서 mip-NeRF 대비 품질 열세
- 해시 충돌로 인한 grain artifact (SDF에서 두드러짐)
- Generative 모델과 결합 시 충돌 처리 추가 연구 필요
- 훈련 시간이 빠르지만 실시간 렌더링은 여전히 3DGS보다 느림 (~10 FPS vs. >30 FPS)
