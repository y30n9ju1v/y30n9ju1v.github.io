---
title: "CLIP: Learning Transferable Visual Models From Natural Language Supervision"
date: 2026-04-29T00:00:00+09:00
draft: false
categories: ["Papers"]
tags: ["CLIP", "Contrastive Learning", "Zero-Shot Transfer", "Multi-Modal", "Vision-Language", "OpenAI", "Image Classification", "Transfer Learning"]
---

## 개요

- **저자**: Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever (OpenAI)
- **발행년도**: 2021
- **arXiv**: 2103.00020
- **주요 내용**: 인터넷에서 수집한 4억 개의 (이미지, 텍스트) 쌍으로 이미지 인코더와 텍스트 인코더를 대조 학습(contrastive learning)으로 동시 훈련. 클래스 이름을 자연어로 표현하여 추가 훈련 없이 30개 이상의 다양한 시각 태스크에 zero-shot으로 전이. ImageNet에서 ResNet-50 수준의 zero-shot 성능 달성.

## 한계 극복

- **기존 한계 1 — 고정된 클래스 집합**: 기존 컴퓨터 비전 모델은 사전 정의된 클래스만 예측 가능. 새로운 개념을 인식하려면 레이블된 데이터를 추가로 수집하고 재학습해야 함.
- **기존 한계 2 — 크라우드소싱 레이블의 비용과 병목**: ImageNet 같은 고품질 데이터셋 구축에는 막대한 인력과 시간이 필요. 인터넷에 존재하는 방대한 이미지-텍스트 쌍을 활용하지 못함.
- **기존 한계 3 — 낮은 zero-shot 전이 성능**: 기존 자연어 감독 방식(Visual N-Grams 등)은 ImageNet에서 11.5% 정확도에 불과. 규모와 학습 방식 모두 한계.
- **이 논문의 접근 방식**: "어떤 텍스트가 어떤 이미지와 쌍을 이루는가"를 예측하는 대조적 사전 훈련(contrastive pre-training) + 클래스 이름을 텍스트 템플릿으로 변환하는 zero-shot 분류기 합성.

## 목차

- Section 1: Introduction and Motivating Work
- Section 2: Approach
  - 2.1 Natural Language Supervision
  - 2.2 Creating a Sufficiently Large Dataset
  - 2.3 Selecting an Efficient Pre-Training Method
  - 2.4 Choosing and Scaling a Model
  - 2.5 Training
- Section 3: Experiments
  - 3.1 Zero-Shot Transfer
  - 3.2 Representation Learning
  - 3.3 Robustness to Natural Distribution Shift
- Section 4: Comparison to Human Performance
- Section 5: Data Overlap Analysis
- Section 6: Limitations
- Section 7: Broader Impacts

---

## Section 1: Introduction and Motivating Work

**요약**

NLP에서는 GPT, BERT처럼 원시 텍스트로 사전 훈련한 모델이 다양한 태스크에 zero-shot 또는 few-shot으로 전이되어 큰 성공을 거뒀다. 컴퓨터 비전에서도 같은 패러다임이 가능할까?

이미지에 붙은 텍스트(캡션, 해시태그, 설명)는 인터넷에 무한히 존재한다. 이를 감독 신호로 사용하면 1000개 클래스에 고정된 ImageNet 방식의 한계를 넘어 개방형 어휘(open vocabulary)로 시각 개념을 학습할 수 있다.

**핵심 개념**

- **Natural Language Supervision**: 레이블 대신 자유로운 텍스트를 감독 신호로 사용. 어노테이션 병목 없이 인터넷 규모의 데이터를 활용.
- **Zero-shot transfer**: 특정 태스크의 학습 데이터 없이, 클래스 이름을 텍스트로 표현하는 것만으로 분류 수행.
- **Task-agnostic pre-training**: 하나의 모델이 OCR, 행동 인식, 지리 위치 추정 등 이질적인 태스크를 모두 수행.

---

## Section 2: Approach

### 2.1 Natural Language Supervision

**요약**

핵심 아이디어는 이미지와 텍스트를 **함께** 학습하는 것이 아니라, 이미지-텍스트 쌍의 **매칭**을 학습하는 것이다. 자연어는 단순 레이블보다 훨씬 풍부한 감독 신호를 제공하며, 새로운 시각 개념을 표현하는 데 유연하다.

---

### 2.2 Creating a Sufficiently Large Dataset (WIT)

**요약**

기존 데이터셋(MS-COCO ~10만, YFCC100M ~1억)은 현대 기준으로 작다. OpenAI는 인터넷의 다양한 공개 출처에서 **4억 개의 (이미지, 텍스트) 쌍**을 수집하여 **WIT(WebImageText)** 데이터셋을 구축했다.

- 500,000개의 쿼리(영어 Wikipedia의 단어 + bi-gram + WordNet synset)로 검색
- 각 쿼리당 최대 20,000쌍 수집
- 전체 단어 수가 GPT-2 학습 데이터(WebText)와 유사한 수준

---

### 2.3 Selecting an Efficient Pre-Training Method

**요약**

초기에는 VirTex처럼 이미지 CNN과 텍스트 Transformer를 함께 훈련해 캡션을 예측하는 방식을 시도했다. 그러나 정확한 단어를 예측하는 것은 지나치게 어려운 과제였고, 그림 2에서 보듯 bag-of-words 예측보다 3배 느리게 학습되었다.

CLIP은 **정확한 단어 예측 대신 어떤 텍스트가 어떤 이미지와 쌍인지만 예측**하는 대조적 목표(contrastive objective)로 전환했다. 이것이 zero-shot ImageNet 전이 속도를 추가로 4배 향상시켰다.

**핵심 개념: 대조 학습 (Contrastive Learning)**

배치 크기 $N$의 (이미지, 텍스트) 쌍이 주어졌을 때, CLIP은 $N \times N$개의 가능한 쌍 중 실제로 매칭된 $N$개를 찾도록 학습한다.

$$\text{logits} = \frac{I_e \cdot T_e^\top}{\tau}$$

**수식 설명**

- **$I_e$**: 이미지 임베딩 행렬 ($N \times d_e$). 이미지 인코더 출력을 L2 정규화한 것.
- **$T_e$**: 텍스트 임베딩 행렬 ($N \times d_e$). 텍스트 인코더 출력을 L2 정규화한 것.
- **$I_e \cdot T_e^\top$**: $N \times N$ 코사인 유사도 행렬. $(i,j)$ 원소는 $i$번째 이미지와 $j$번째 텍스트의 유사도.
- **$\tau$**: 학습 가능한 온도 파라미터 (temperature). 유사도 분포의 날카로움을 제어.
  - 작은 $\tau$: 정답 쌍과 오답 쌍의 차이를 강하게 강조
  - 로그 스케일로 파라미터화하여 학습 중 자동 최적화

손실 함수는 행 방향(이미지→텍스트)과 열 방향(텍스트→이미지) 각각의 cross-entropy를 평균한 **symmetric cross-entropy loss**:

$$\mathcal{L} = \frac{1}{2}\left(\mathcal{L}_\text{img} + \mathcal{L}_\text{txt}\right)$$

- **$\mathcal{L}_\text{img}$**: 각 이미지에 대해 $N$개 텍스트 중 올바른 텍스트를 찾는 cross-entropy
- **$\mathcal{L}_\text{txt}$**: 각 텍스트에 대해 $N$개 이미지 중 올바른 이미지를 찾는 cross-entropy
- **대각 원소가 정답**: logits 행렬의 대각선 원소들이 실제 (이미지, 텍스트) 쌍에 해당

이 방식은 **InfoNCE loss** (또는 multi-class N-pair loss)로 알려진 대조 학습 목표의 변형이다.

---

### 2.4 Choosing and Scaling a Model

**요약**

이미지 인코더로 두 가지 아키텍처를 실험:

- **ResNet 계열**: ResNet-50을 기반으로 ResNet-D 개선 + 안티앨리어싱 풀링 + attention pooling 적용. RN50, RN101, RN50x4, RN50x16, RN50x64 5종.
- **Vision Transformer (ViT) 계열**: ViT-B/32, ViT-B/16, ViT-L/14 3종. patch+position 임베딩 전에 layer normalization 추가.

텍스트 인코더: 63M 파라미터 12-layer 512-wide Transformer, 49,152 vocab BPE, 최대 76 토큰. [EOS] 토큰의 최상위 레이어 activations를 텍스트 표현으로 사용.

최고 성능 모델: **ViT-L/14@336px** (336px 해상도로 추가 1 epoch fine-tune).

---

### 2.5 Training

**요약**

- 5개 ResNet + 3개 ViT, 총 8개 모델을 동시 훈련
- Adam optimizer, cosine learning rate decay, 32,768의 매우 큰 배치 크기
- 혼합 정밀도(mixed-precision) 훈련, gradient checkpointing
- 가장 큰 ResNet(RN50x64): 592개 V100 GPU에서 18일 훈련
- 가장 큰 ViT(ViT-L/14): 256개 V100 GPU에서 12일 훈련
- 온도 파라미터 $\tau$: 0.07로 초기화, 100 이하로 클리핑

---

## Section 3: Experiments

### 3.1 Zero-Shot Transfer

**요약**

**Zero-shot 분류 방법**:

1. 데이터셋의 모든 클래스 이름을 프롬프트 템플릿에 삽입: `"A photo of a {label}."`
2. 텍스트 인코더로 각 클래스 텍스트를 임베딩 → 텍스트 임베딩 집합 생성
3. 테스트 이미지를 이미지 인코더로 임베딩
4. 이미지 임베딩과 각 클래스 텍스트 임베딩의 코사인 유사도를 계산
5. 가장 유사한 클래스를 예측 레이블로 선택

**Prompt Engineering & Ensembling**

단순히 클래스 이름만 사용하면 다의어(polysemy) 문제가 발생한다. 예: "crane"이 건설 크레인인지 두루미인지 구분 불가.

`"A photo of a {label}."` 템플릿만으로도 ImageNet 정확도가 1.3%p 향상. 태스크에 맞는 문맥 추가로 추가 향상:
- 애완동물 분류: `"A photo of a {label}, a type of pet."`
- 위성 이미지: `"a satellite photo of a {label}."`
- 80개 다양한 프롬프트 앙상블 → ImageNet 추가 3.5%p 향상

프롬프트 엔지니어링 + 앙상블로 평균 **약 5%p 향상** (4배 많은 compute와 동등한 효과).

**주요 결과**

| 데이터셋 | Visual N-Grams | Zero-Shot CLIP |
|---------|---------------|---------------|
| ImageNet | 11.5% | **76.2%** |
| aYahoo | 72.4% | **98.4%** |
| SUN | 23.0% | **58.5%** |

- ImageNet에서 원본 ResNet-50과 동일한 76.2% 달성 (1.28M 학습 데이터 없이)
- top-5 정확도 95% (Inception-V4 수준)
- Zero-shot CLIP이 ResNet-50 위의 linear probe보다 16개 데이터셋에서 우수

**Zero-shot이 약한 태스크**:
- 세밀한 분류 (자동차 모델, 꽃 종류, 항공기 변형)
- 추상적 계수 (CLEVRCounts)
- 위성 이미지 (EuroSAT)
- 전문 의료 이미지 (PatchCamelyon)
- 거리 추정 (KITTI Distance)

---

### 3.2 Representation Learning

**요약**

Linear probe (CLIP 특징 위에 선형 분류기 추가 학습) 평가에서 CLIP은 27개 데이터셋 평균 기준 모든 기존 모델을 계산 효율성 면에서 압도한다.

- CLIP ResNet-50x64: 최고 성능 기존 모델(Noisy Student EfficientNet-L2)를 전반적 점수와 계산 효율성 모두에서 소폭 초과
- ViT 모델이 ResNet 대비 약 3배 계산 효율 우수

---

### 3.3 Robustness to Natural Distribution Shift

**요약**

ImageNet에서 학습된 모델들은 distribution shift에 매우 취약하다. ResNet-101은 7개 자연 분포 이동 데이터셋에서 ImageNet 대비 5배 많은 실수를 범한다.

Zero-shot CLIP은 이 **robustness gap을 최대 75%까지 줄인다**.

| 데이터셋 | ResNet-101 | Zero-Shot CLIP | 개선 |
|---------|-----------|---------------|------|
| ImageNet | 76.2% | 76.2% | 0% |
| ImageNet-R | 37.7% | **88.9%** | +51.2%p |
| ObjectNet | 32.6% | **72.3%** | +39.7%p |
| ImageNet-Sketch | 25.2% | **60.2%** | +35.0%p |
| ImageNet-A | 2.7% | **77.1%** | +74.4%p |

**이유**: Zero-shot 모델은 특정 데이터셋의 spurious correlation을 학습할 수 없어 본질적으로 robust. 반면 supervised adaptation은 ImageNet 분포에 과적합되어 다른 분포에서 크게 성능이 하락.

---

## Section 4: Comparison to Human Performance

**요약**

Oxford IIT Pets 데이터셋에서 인간과 CLIP 비교:

| 설정 | 정확도 |
|------|--------|
| Zero-shot Human | 53.7% |
| **Zero-shot CLIP** | **93.5%** |
| One-shot Human | 75.7% |
| Two-shot Human | 75.7% |

CLIP의 zero-shot이 인간의 zero-shot을 크게 앞선다. 인간이 어렵다고 느끼는 이미지(불확실성이 높은 경우)는 CLIP도 어려워하는 경향이 있다. 한편 인간은 1장의 예시만으로도 75.7%로 크게 향상되지만, CLIP의 few-shot 전이는 counter-intuitive하게 zero-shot보다 낮아지는 현상이 발생한다.

---

## Section 5: Data Overlap Analysis

**요약**

WIT 사전 훈련 데이터와 평가 데이터셋 간의 중복 여부 분석. 35개 데이터셋 중 중앙값 2.2%, 평균 3.2% 중복. 중복으로 인한 정확도 향상은 통계적으로 유의미한 경우가 6개뿐이며, 최대 0.6% (Birdsnap). 실질적으로 중복이 결과에 미치는 영향은 미미하다.

---

## Section 6: Limitations

**요약**

- **태스크 특화 성능 부족**: zero-shot CLIP은 ResNet-50 위의 linear probe 수준이며, 태스크별 SOTA보다 훨씬 낮음.
- **약한 few-shot 전이**: zero-shot → few-shot으로 전환 시 성능이 오히려 하락하는 현상. 인간과 달리 prior knowledge를 few-shot 예시에 통합하는 능력 부족.
- **세밀한 분류 취약**: 자동차 모델, 꽃 종류 등 세밀한 구분이 필요한 태스크에서 취약.
- **추상적 태스크 불가**: 개수 세기, 거리 추정 등.
- **데이터 효율 낮음**: 동등 성능에 약 1000배 더 많은 compute 필요. CLIP보다 CLIP의 특징 위 linear probe가 더 효율적이라는 역설.
- **사회적 편향**: 인터넷 텍스트로 학습하므로 사회 편향이 내재화됨.

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **Contrastive Pre-training** | $N$개 (이미지, 텍스트) 쌍에서 $N \times N$ 유사도 행렬을 학습. 정답 대각 원소를 최대화 |
| **Symmetric Cross-Entropy Loss** | 이미지→텍스트 방향과 텍스트→이미지 방향 loss를 각각 계산하여 평균 |
| **Temperature $\tau$** | 유사도 점수의 날카로움을 제어하는 학습 파라미터 |
| **WIT (WebImageText)** | 인터넷에서 수집한 4억 쌍의 (이미지, 텍스트) 사전 훈련 데이터셋 |
| **Zero-Shot Classifier 합성** | 클래스 이름을 텍스트 인코더에 통과시켜 분류기 가중치를 동적으로 생성 |
| **Prompt Engineering** | `"A photo of a {label}."` 같은 텍스트 템플릿으로 다의어 문제 해소 및 성능 향상 |
| **Prompt Ensembling** | 80개 다양한 프롬프트의 텍스트 임베딩을 평균하여 robust한 분류기 구성 |
| **Linear Probe** | CLIP 특징 위에 선형 분류기만 추가 학습. 표현 품질 평가 지표. |
| **Effective Robustness** | 분포 이동 하에서 in-distribution 정확도 예측값 대비 실제 out-of-distribution 정확도의 개선량 |

## 결론 및 시사점

CLIP은 "자연어로 시각 세계를 기술"하는 방식이 레이블 기반 패러다임의 근본적 한계를 돌파한다는 것을 증명했다. 400M 쌍의 약한 감독만으로 30개 이상의 이질적 태스크에 zero-shot 전이되며, distribution shift에 대한 강인성이 특히 두드러진다.

**자율주행·합성 데이터 관점 시사점**

- **개방형 어휘 인식**: 자율주행 시나리오의 롱테일 객체(공사 장비, 특이 차량 등)를 추가 레이블 없이 인식하는 기반으로 활용. MagicDrive, DriveArena 등에서 텍스트 조건으로 장면을 제어할 때 CLIP text embedding이 핵심 역할.
- **Robust 특징**: CLIP 특징은 distribution shift에 강하여 시뮬레이터→실세계 도메인 갭 완화에 유리. 합성 데이터로 훈련한 모델의 실세계 전이 성능 향상에 활용 가능.
- **Zero-shot 평가**: 새로운 시나리오나 엣지 케이스에 대해 레이블 없이 CLIP으로 빠르게 성능 예측 가능.
- **한계**: 정밀한 거리 추정, 세밀한 차량 분류 등 자율주행 핵심 태스크에서는 domain-specific fine-tuning이 여전히 필요.
