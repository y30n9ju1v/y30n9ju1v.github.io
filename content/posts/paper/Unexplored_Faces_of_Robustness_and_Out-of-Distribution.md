+++
title = 'Unexplored Faces of Robustness and Out of Distribution: Covariate Shifts in Environment and Sensor Domains'
date = 2024-07-31T07:59:16+09:00
draft = true
+++

이 글은 https://arxiv.org/abs/2404.15882 을 번역 및 요약한 글입니다.
이 논문은 서울대 김형신 교수님 연구실에서 작성된 것으로 CVPR 2024에 게시되었습니다.

## Abstract
Computer vision applications predict on digital images acquired by a camera from physical scenes through light.
However, conventional robustness benchmarks rely on perturbations in digitized images, diverging from distribution shifts occurring in the image acquisition process.
To bridge this gap, we introduce a new distribution shift dataset, ImageNet-ES, comprising variations in environmental and camera sensor factors by directly capturing 202k images with a real camera in a controllable testbed.
With the new dataset, we evaluate out-of-distribution (OOD) detection and model robustness.

컴퓨터 비전 애플리케이션은 물리적 장면에서 카메라로 획득한 디지털 이미지에 대해 예측을 수행합니다.
그러나 기존의 견고성 벤치마크는 디지털 이미지의 변형에 의존하며, 이는 이미지 획득 과정에서 발생하는 분포 변화와는 다릅니다.
이 격차를 해소하기 위해 우리는 환경 및 카메라 센서 요인의 변화를 직접 제어할 수 있는 테스트베드에서 실제 카메라로 202,000장의 이미지를 캡처하여 새로운 분포 변화 데이터셋인 ImageNet-ES를 소개합니다.
이 새로운 데이터셋을 통해 우리는 분포 외 검출(OOD)과 모델 견고성을 평가합니다.

We also observe that the model becomes more robust in both ImageNet-C and -ES by learning environment and sensor variations in addition to existing digital augmentations.
Lastly, our results suggest that effective shift mitigation via camera sensor control can significantly improve performance without increasing model size.

또한, 모델이 기존의 디지털 증강 외에 환경 및 센서 변화를 학습함으로써 ImageNet-C 및 ImageNet-ES에서 더 견고해진다는 것을 관찰했습니다.
마지막으로, 카메라 센서 제어를 통한 효과적인 변화 완화가 모델 크기를 늘리지 않고도 성능을 크게 향상시킬 수 있음을 시사합니다.

## 1. Introduction
Existing robustness benchmarks evaluate the resilience of model predictions against perturbations in digitized images [7, 10, 20, 25].
Various techniques, such as domain generalization/adaptation and out-of-distribution (OOD) detection, have refined deep learning models to handle distribution shifts [1, 6, 11, 13, 15, 19, 21, 22, 26, 30, 31, 37, 42].

기존의 견고성 벤치마크는 디지털 이미지에서 발생하는 변형에 대한 모델 예측의 탄력성을 평가합니다[7, 10, 20, 25].
도메인 일반화/적응 및 분포 외 검출(OOD)과 같은 다양한 기술들이 분포 변화에 대처할 수 있도록 딥러닝 모델을 개선해 왔습니다[1, 6, 11, 13, 15, 19, 21, 22, 26, 30, 31, 37, 42].

However, the implications of distribution shifts resulting from the image acquisition process (i.e. eyes), caused by variations in real-world light and camera sensor operations, remain unexplored.
The absence of a benchmark introduces uncertainty regarding the generalizability of observed robustness in synthetic data to real-world applications.
Moreover, the synergistic interplay between the camera sensor and the model has not been investigated.
Therefore, current approaches may risk inefficiency by attempting to address eyesight/light problems through over-training the brain.

그러나 실제 세계의 빛과 카메라 센서 작동의 변동으로 인한 이미지 획득 과정(즉, 눈)에서 발생하는 분포 변화의 의미는 아직 탐구되지 않았습니다.
벤치마크의 부재는 합성 데이터에서 관찰된 견고성이 실제 응용에 일반화될 수 있는지에 대한 불확실성을 초래합니다.
또한, 카메라 센서와 모델 간의 시너지 효과도 조사되지 않았습니다.
따라서, 현재 접근 방식은 눈/빛 문제를 과도하게 뇌를 훈련시키는 방식으로 해결하려 함으로써 비효율성을 초래할 위험이 있습니다.

This work aims to narrow the gap between synthetic and real-world data by investigating the impact of environmental and camera sensor factors.
Instead of relying on digital perturbation, we construct a controllable testbed, ESStudio.
This testbed allows us to directly capture images using a physical camera with varying sensor parameters (ISO, shutter speed and aperture) and different light conditions (on/off), resulting in a novel dataset called ImageNet-ES.

이 연구는 환경 및 카메라 센서 요인의 영향을 조사하여 합성 데이터와 실제 데이터 간의 격차를 좁히는 것을 목표로 합니다.
디지털 변형에 의존하는 대신, 우리는 제어 가능한 테스트베드인 ES-Studio를 구축했습니다.
이 테스트베드를 통해 ISO, 셔터 속도 및 조리개와 같은 다양한 센서 매개변수와 다른 조명 조건(켜짐/꺼짐)에서 실제 카메라를 사용하여 직접 이미지를 캡처할 수 있습니다.
그 결과, ImageNet-ES라는 새로운 데이터셋이 만들어졌습니다.

With the ImageNet-ES dataset, we conduct an extensive empirical study on OOD detection and domain generalization.
Furthermore, we explore the potential of camera sensor control in addressing real-world distribution shifts.
Our study unveils a series of noteworthy findings as follows:

ImageNet-ES 데이터셋을 통해 우리는 OOD(Out-of-Distribution) 검출 및 도메인 일반화에 대한 광범위한 실증 연구를 수행합니다.
더 나아가, 실제 분포 변화를 해결하기 위한 카메라 센서 제어의 가능성을 탐구합니다.
우리의 연구는 다음과 같은 주목할 만한 일련의 발견을 공개합니다:

OOD definition: Covariate-shifted data (C-OOD) have been categorized entirely as either OOD or in-distribution (ID). However, C-OOD data in ImageNet-ES exhibit widespread OOD scores in most metrics, including both ID and OOD. Model-Specific OOD (MS-OOD) [1] is
more proper for fine-grained labeling of our C-OOD data.
• OOD detection: State-of-the-art (SOTA) OOD detection methods, focusing on distinguishing semantic shifts, fal- ter in ImageNet-ES. OOD detection should be improved
to incorporate real-world covariate shifts together.
• Domain generalization: Existing digital augmenta- tions do not incorporate distribution shifts in ImageNet- ES. Learning environment/sensor-domain perturbations in ImageNet-ES with existing augmentations improves
model robustness, even in conventional benchmarks.
• Potential of sensor control: Camera sensor control can significantly improve prediction accuracy by mitigating distribution shifts. With sensor control, EfficientNet can perform comparably to much heavier transformer models.
• Direction of sensor control: High-quality images in terms of model prediction do not necessarily align with human aesthetics but rather with what the model learns from training data. Sensor control should be grounded in
the features that the model (not the human) prefers. Overall, future research on OOD detection and model ro- bustness requires more thorough evaluations, including en- vironmental and camera sensor variations. Furthermore, it is valuable to explore camera sensor control so that acquired images contain more features preferred by the model.

### 주요 발견 사항

1. **OOD 정의:**
   - 공변량 변화 데이터(C-OOD)는 완전히 OOD 또는 인-디스트리뷰션(ID)으로 분류되었습니다. 그러나 ImageNet-ES의 C-OOD 데이터는 대부분의 지표에서 ID와 OOD를 모두 포함하는 광범위한 OOD 점수를 보입니다.
   - Model-Specific OOD (MS-OOD) [1]는 C-OOD 데이터를 세밀하게 라벨링하는 데 더 적합합니다.

2. **OOD 검출:**
   - 최신의 OOD 검출 방법들은 의미적 변화를 구분하는 데 중점을 두지만, ImageNet-ES에서는 실패합니다.
   - OOD 검출은 실제 공변량 변화를 통합하도록 개선되어야 합니다.

3. **도메인 일반화:**
   - 기존의 디지털 증강은 ImageNet-ES의 분포 변화를 통합하지 않습니다.
   - 기존 증강과 함께 ImageNet-ES에서 환경/센서 도메인 변화를 학습하면, 기존 벤치마크에서도 모델 견고성이 향상됩니다.

4. **센서 제어의 잠재력:**
   - 카메라 센서 제어는 분포 변화를 완화하여 예측 정확도를 크게 향상시킬 수 있습니다.
   - 센서 제어를 통해 EfficientNet은 훨씬 더 무거운 트랜스포머 모델과 유사한 성능을 발휘할 수 있습니다.

5. **센서 제어의 방향:**
   - 모델 예측 측면에서 고품질 이미지는 반드시 인간의 미적 감각과 일치하지 않으며, 오히려 모델이 학습한 데이터에서 선호하는 특징과 일치합니다.
   - 센서 제어는 인간이 아닌 모델이 선호하는 특징에 기반해야 합니다.

향후 OOD 검출 및 모델 견고성 연구는 환경 및 카메라 센서 변동을 포함한 더 철저한 평가가 필요합니다. 또한, 획득한 이미지가 모델이 선호하는 특징을 더 많이 포함하도록 카메라 센서 제어를 탐구하는 것이 가치가 있습니다.

## 2. Related Work
### 2.1. Robustness Benchmarks
A number of benchmarks have employed various digital perturbations to assess image classifier robustness or OOD detection methods. Notably, ImageNet-C and -P [10] sim- ulate environmental and adversarial perturbations through blur, noise, brightness, etc. ImageNet-A and -O [14] limit spurious cues using adversarial perturbations. Several datasets utilize visual renditions to change real scenes, such asart,cartoons,patterns,toys,paintings,etc. [13,29,36]. SVSF [13] or ImageNet-E [20] changes camera views or image compositions.

여러 벤치마크는 이미지 분류기의 견고성 또는 OOD 검출 방법을 평가하기 위해 다양한 디지털 변형을 사용해 왔습니다. 특히, ImageNet-C와 -P [10]는 블러, 노이즈, 밝기 등을 통해 환경적 및 적대적 변형을 시뮬레이션합니다. ImageNet-A와 -O [14]는 적대적 변형을 사용하여 불필요한 단서를 제한합니다. 여러 데이터셋은 예술, 만화, 패턴, 장난감, 그림 등 시각적 표현을 활용하여 실제 장면을 변경합니다 [13, 29, 36]. SVSF [13] 또는 ImageNet-E [20]는 카메라 뷰 또는 이미지 구성을 변경합니다.

While these benchmarks aim to incorporate real-world distribution shifts, such as camera framing, their approaches are limited to the indirect simulation of actual shifts via perturbing already-acquired digital images. Recent studies have highlighted that SOTA OOD detection methods face challenges due to a lack of knowledge about the real-world OOD distributions [28] and experience performance degra- dation in near-OOD, shifted benchmarks [18].

이러한 벤치마크는 카메라 프레이밍과 같은 실제 분포 변화를 포함하려고 하지만, 이미 획득한 디지털 이미지를 변형하여 실제 변화를 간접적으로 시뮬레이션하는 데 한정됩니다. 최근 연구에 따르면, 최신의 OOD 검출 방법은 실제 세계의 OOD 분포에 대한 지식 부족으로 인해 어려움을 겪고 있으며, 근접 OOD 및 변화된 벤치마크에서 성능 저하를 경험합니다 [28, 18].

### 2.2. Out-of-Distribution (OOD) Detection
Out-of-distribution (OOD) detection is the task of identify- ing test data that come from a distribution different from the distribution of training data, due to either semantic shift (S-OOD) or covariate shift (C-OOD) [18].

분포 외 검출(OOD)은 의미적 변화(S-OOD) 또는 공변량 변화(C-OOD)로 인해 훈련 데이터의 분포와 다른 분포에서 온 테스트 데이터를 식별하는 작업입니다 [18].

OOD studies have focused on detecting samples with semantic shifts (S-OOD) that do not belong to any of the classes present in the training set. A number of methods de- termine the OOD score based on the decision-making com- ponent of classifiers [11, 15, 21, 22]. These techniques are more robust when class-agnostic information needs to be carefully considered, but vulnerable to significant semantic shifts or overconfidence issues [30, 37]. To alleviate these problems, other methods calculate the OOD score based on features the model learned [6, 19, 30, 31, 37].

OOD 연구는 주로 훈련 세트에 존재하지 않는 클래스에 속하는 샘플의 의미적 변화(S-OOD)를 감지하는 데 중점을 두고 있습니다. 여러 방법은 분류기의 의사 결정 구성 요소를 기반으로 OOD 점수를 결정합니다 [11, 15, 21, 22]. 이러한 기술은 클래스 비종속 정보를 신중하게 고려해야 할 때 더 견고하지만, 의미적 변화나 과신 문제에 취약합니다 [30, 37]. 이러한 문제를 완화하기 위해 다른 방법들은 모델이 학습한 특징을 기반으로 OOD 점수를 계산합니다 [6, 19, 30, 31, 37].

However, prior work has relatively unexplored how to handle covariate-shifted (C-OOD) samples. A handful of studies have considered entire C-OOD examples as in- distribution (ID) to enhance classifier robustness against co- variate shifts [39, 40, 42]. Some studies have taken oppo- site approaches, treating all C-OOD samples as OOD to make OOD detection more generalizable to non-semantic shifts [16]. To address the problem of the rough treatment of entire covariate-shifted data as ID or OOD, more re- cent studies provide fine-grained categorization of C-OOD samples into ID and OOD, based on their own defini- tions [1, 33]. Notably, Averly and Chao have proposed a unified criterion that incorporates both S-OOD and C- OOD data based on model prediction results, called Model- Specific OOD (MS-OOD) [1].

그러나 이전 연구들은 공변량 변화가 있는 (C-OOD) 샘플을 처리하는 방법에 대해 상대적으로 탐구되지 않았습니다. 일부 연구들은 공변량 변화 예제를 인디스트리뷰션(ID)으로 간주하여 분류기의 공변량 변화에 대한 견고성을 향상시키려 했습니다 [39, 40, 42]. 반면, 일부 연구들은 모든 C-OOD 샘플을 OOD로 처리하여 OOD 검출을 비의미적 변화에 대해 더 일반화하려고 했습니다 [16]. 전체 공변량 변화 데이터를 ID 또는 OOD로 대략적으로 처리하는 문제를 해결하기 위해, 최근 연구들은 자체 정의를 바탕으로 C-OOD 샘플을 세밀하게 ID와 OOD로 분류하는 방법을 제공합니다 [1, 33]. 특히, Averly와 Chao는 모델 예측 결과를 기반으로 S-OOD와 C-OOD 데이터를 모두 포함하는 통합 기준인 Model-Specific OOD (MS-OOD)를 제안했습니다 [1].

### 2.3. Domain Generalization
Domain generalization focuses on improving the robustness of models to distribution shifts in testing domains. To this end, Hendrycks et al. identified that using larger models and artificial data augmentations (called DeepAugment) can im- prove model robustness [13]. While many augmentation techniques [5, 12, 41] have shown to improve the robust- ness, their evaluation scope is limited to digital corruptions. More recently, foundation models have demonstrated suc- cess in learning effective feature representations through ar- chitectural changes [24], discriminative self-supervised pre- training [9, 26, 44], or large uncurated data [26].

도메인 일반화는 테스트 도메인에서 분포 변동에 대한 모델의 견고성을 향상시키는 데 중점을 둡니다. 이를 위해 Hendrycks 등은 더 큰 모델과 인공 데이터 증강(DeepAugment라 불림)을 사용하면 모델의 견고성을 향상시킬 수 있음을 확인했습니다 [13]. 많은 증강 기법들이 [5, 12, 41] 견고성을 향상시키는 것으로 나타났지만, 그들의 평가 범위는 디지털 손상에 한정되어 있습니다. 최근에는 아키텍처 변화 [24], 변별적 자기 지도 사전 학습 [9, 26, 44], 또는 대규모 비정제 데이터 [26]를 통해 효과적인 특징 표현을 학습하는 데 성공한 기반 모델들이 등장했습니다.

However, prior work has focused on digital distribution shifts (e.g. pixelate or gaussian noise etc.), scene and cam- era composition shifts. On the other hand, our ImageNet-ES addresses other types of distributional shifts, such as those arising from the image acquisition process.

그러나 이전 연구들은 디지털 분포 변동(예: 픽셀화 또는 가우시안 노이즈 등), 장면 및 카메라 구성 변동에 초점을 맞추었습니다. 반면, 우리의 ImageNet-ES는 이미지 획득 과정에서 발생하는 분포 변동과 같은 다른 유형의 분포 변동을 다룹니다.

## 3. Background
