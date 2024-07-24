+++
title = 'On Testing Machine Learning Programs'
date = 2024-07-23T20:17:51+09:00
draft = false
+++

이 글은 https://arxiv.org/abs/1812.02257 을 번역 및 요약한 글입니다.

## Abstract
Given this growing importance of ML-based systems in our daily lives, it is becoming utterly important to ensure their reliability.

우리 일상 생활에서 ML 기반 시스템의 중요성이 커짐에 따라 이들의 신뢰성을 보장하는 것이 매우 중요해지고 있습니다.

Recently, software researchers have started adapting concepts from the software testing domain (e.g., code coverage, mutation testing, or property-based testing) to help ML engineers detect and correct faults in ML programs.

최근 소프트웨어 연구자들은 ML 엔지니어가 ML 프로그램의 결함을 탐지하고 수정하는 데 도움을 줄 수 있도록 소프트웨어 테스트 도메인의 개념(예: 코드 커버리지, 변이 테스트 또는 속성 기반 테스트)을 적응시키기 시작했습니다.

This paper reviews current existing testing practices for ML programs.

이 논문은 현재 존재하는 ML 프로그램 테스트 관행을 검토합니다.

## Introduction

detecting and correcting faults in ML programs is still very challenging as evidenced by the recent Uber’s car incident that resulted in the death of a pedestrian.

최근 보행자가 사망한 우버의 자동차 사고에서 알 수 있듯이 ML 프로그램의 결함을 감지하고 수정하는 것은 여전히 매우 어렵습니다.

The main reason behind the difficulty to test ML programs is the shift in the development paradigm induced by ML and AI.

ML 프로그램을 테스트하는 데 어려움이 있는 주된 이유는 ML과 AI로 인해 개발 패러다임이 변화했기 때문입니다.

Traditionally, software systems are constructed deductively, by writing down the rules that govern the behavior of the system as program code.
However, with ML, these rules are inferred from training data (i.e., they are generated inductively).

전통적으로 소프트웨어 시스템은 시스템의 동작을 프로그래밍 코드로 작성하여 연역적으로 구축됩니다.
그러나 ML의 경우 이러한 규칙은 훈련 데이터에서 추론됩니다(즉, 귀납적으로 생성됨).

This paradigm shift in application development makes it difficult to reason about the behavior of software systems with ML components, resulting in systems that are intrinsically challenging to test and verify, given that they do not have (complete) specifications or even source code corresponding to some of their critical behaviors.

애플리케이션 개발의 이러한 패러다임 전환으로 인해 ML 구성 요소를 포함하는 소프트웨어 시스템의 동작을 추론하기 어려워졌습니다. 결과적으로 이러한 시스템은 완전한 사양이나 일부 중요한 동작에 해당하는 소스 코드가 없기 때문에 본질적으로 테스트 및 검증이 어렵습니다.

Compared with traditional software, the dimension and potential testing space of ML programs is much larger.
Current existing software development techniques must be revisited and adapted to this new reality.

전통적인 소프트웨어와 비교할 때, ML 프로그램의 차원과 잠재적인 테스트 공간은 훨씬 더 큽니다.
현재 기존의 소프트웨어 개발 기술을 재검토하고 이 새로운 현실에 맞게 조정해야 합니다.

In this paper, we survey existing testing practices that have been proposed for ML programs, explaining the context in which they can be applied and their expected outcome.
We also identify gaps in the literature related to the testing of ML programs and suggest future research directions for the scientific community.
This paper makes the following contributions:

* We present and explain challenges related to the testing of ML programs that use differentiable models.
* We provide a comprehensive review of current software testing practices for ML programs.
* We identify gaps in the literature related to the testing of ML programs and provide future research directions for the scientific community.

이 논문에서는 ML 프로그램을 위해 제안된 기존 테스트 관행을 조사하고, 이를 적용할 수 있는 맥락과 예상되는 결과를 설명합니다.
또한 ML 프로그램의 테스트와 관련된 문헌의 격차를 식별하고 과학 커뮤니티를 위한 향후 연구 방향을 제안합니다.
이 논문은 다음과 같은 기여를 합니다:

* 우리는 미분 가능한 모델을 사용하는 ML 프로그램 테스트와 관련된 문제를 제시하고 설명합니다.
* 우리는 ML 프로그램에 대한 현재 소프트웨어 테스트 관행을 포괄적으로 검토합니다.
* 우리는 ML 프로그램의 테스트와 관련된 문헌의 격차를 식별하고 과학 커뮤니티를 위한 향후 연구 방향을 제공합니다.

## 2. Background on Machine Learning Model
There are two main source of faults in ML programs : the data and the model.
For each of these two dimensions (i.e., data and model), there can be errors both at the conceptual and implementation levels, which makes fault location very challenging.

ML 프로그램의 주요 결함 원인은 데이터와 모델 두 가지입니다.
이 두 가지 차원(즉, 데이터와 모델) 각각에서 개념적 및 구현 수준 모두에서 오류가 발생할 수 있으며, 이는 결함 위치를 찾는 것을 매우 어렵게 만듭니다.

Approaches that rely on tweaking variables and watching signals from execution are generally ineffective because of the exponential increase of the number of potential faults locations.

변수 조정 및 실행 신호 관찰에 의존하는 접근 방식은 잠재적 결함 위치의 수가 기하급수적으로 증가하기 때문에 일반적으로 효과적이지 않습니다.

## 3. Research Trends in ML Application Testing
### 3.2 Approaches that aim to detect conceptual and implementation errors in ML models
#### 3.2.1 Approaches that aim to detect conceptual errors in ML models
Approaches in this category assume that the models are implemented into programs without errors and focus on providing mechanisms to detect potential errors in the calibration of the models.

이 범주의 접근 방식은 모델이 오류 없이 프로그램에 구현된 것으로 가정하고, 모델의 보정에서 발생할 수 있는 잠재적 오류를 감지하는 메커니즘을 제공하는 데 중점을 둡니다.

These approaches can be divided in two groups: black-box and white-box approaches [9].
Black-box approaches are testing approaches that do not need access to the internal implementation details of the model under test.
These approaches focus on ensuring that the model under test predicts the target value with a high accuracy, without caring about its internal learned parameters.

이러한 접근 방식은 블랙박스 접근 방식과 화이트박스 접근 방식의 두 그룹으로 나눌 수 있습니다.
블랙박스 접근 방식은 테스트 중인 모델의 내부 구현 세부 사항에 접근할 필요가 없는 테스트 접근 방식입니다.
이러한 접근 방식은 내부 학습 매개변수를 신경 쓰지 않고 테스트 중인 모델이 목표 값을 높은 정확도로 예측하는지 확인하는 데 중점을 둡니다.

White-box testing approaches on the other hand take into account the internal implementation logic of the model.
The goal of these approaches is to cover a maximum of specific spots (e.g., neurons) in models. 

반면, 화이트박스 테스트 접근 방식은 모델의 내부 구현 논리를 고려합니다.
이러한 접근 방식의 목표는 모델에서 특정 지점(예: 뉴런)을 최대한 많이 커버하는 것입니다.

##### A: Black-box testing approaches for ML models.
The common denominator to black-box testing approaches is the generation of adversarial data set that is used to test the ML models.
These approaches leverage statistical analysis techniques to devise a multidimensional random process that can generate data with the same statistical characteristics as the input data of the model.
More specifically, they construct generative models that can fit a probability distribution that best describes the input data. 

블랙박스 테스트 접근 방식의 공통 분모는 ML 모델을 테스트하는 데 사용되는 적대적 데이터 세트를 생성하는 것입니다.
이러한 접근 방식은 통계 분석 기법을 활용하여 모델의 입력 데이터와 동일한 통계적 특성을 가진 데이터를 생성할 수 있는 다차원 랜덤 프로세스를 고안합니다.
더 구체적으로, 이들은 입력 데이터를 가장 잘 설명하는 확률 분포에 맞출 수 있는 생성 모델을 구축합니다.

The advantage of this approach is that the synthetic data that is used to test the model is independent from the ML model, but statistically close to its input data.

이 접근 방식의 장점은 모델을 테스트하는 데 사용되는 합성 데이터가 ML 모델과 독립적이지만 통계적으로 입력 데이터와 유사하다는 것입니다.

Adversarial machine learning is an emerging technique that aims to assess the robustness the machine learning models based on the generation of adversarial examples.
it is important to test the robustness of ML models to such variations in input data.

적대적 머신 러닝은 적대적 예제를 생성하여 머신 러닝 모델의 강인성을 평가하는 새로운 기법입니다.
입력 데이터의 이러한 변동에 대한 ML 모델의 강인성을 테스트하는 것이 중요합니다.

Several mechanisms exist for the creation of adversarial examples, such as : making small modifications to the input pixels[10], applying spatial transformations[11], or simple guess-and-check to find misclassified [12].

적대적 예제를 생성하기 위한 여러 메커니즘이 존재합니다.
예를 들어, 입력 픽셀을 약간 수정하거나[10], 공간 변환을 적용하거나[11], 잘못 분류된 항목을 찾기 위해 간단히 추측하고 확인하는 방법[12]이 있습니다.

Recent results [13] [14] involving adversarial evasion attacks against deep neural network models have demonstrated the effectiveness of adversarial examples in revealing weaknesses in models.
Multiple DNN-based image classifiers that achieved state-of-the-art performance levels on randomly selected dataset where found to perform poorly on synthetic images generated by adding humanly imperceptible perturbations.

최근 딥 뉴럴 네트워크 모델에 대한 적대적 회피 공격을 포함한 연구 결과[13][14]는 적대적 예제가 모델의 약점을 드러내는 데 효과적임을 보여주었습니다. 무작위로 선택된 데이터 세트에서 최첨단 성능 수준을 달성한 여러 DNN 기반 이미지 분류기가 사람의 눈에 감지되지 않는 작은 변동을 추가하여 생성된 합성 이미지에서 성능이 저조한 것으로 나타났습니다.

One major limitation of these black-box testing techniques is the representativeness of the generated adversarial examples.
In fact, many adversarial models that generate synthetic images often apply only tiny, undetectable, and imperceptible perturbations, since any visible change would require manual inspection to ensure the correctness of the model’s decision.

이러한 블랙박스 테스트 기술의 주요 한계는 생성된 적대적 예제의 대표성입니다.
실제로 합성 이미지를 생성하는 많은 적대적 모델은 작은, 감지되지 않는, 눈에 띄지 않는 변동만을 적용합니다.
이는 모델의 결정의 올바름을 보장하기 위해 수동 검사가 필요하기 때문입니다.

##### B: White-box testing approaches for ML models.

Pei et al. proposed DeepXplore [15], the first white-box approach for systematically testing deep learning models.
DeepXplore is capable of automatically identifying erroneous behaviors in deep learning models without the need of manual labelling.
The technique makes use of a new metric named neuron coverage, which estimates the amount of neural network’s logic explored by a set of inputs.
This neuron coverage metric computes the rate of activated neurons in the neural network. 

Pei et al.은 딥 러닝 모델을 체계적으로 테스트하기 위한 최초의 화이트박스 접근 방식인 DeepXplore를 제안했습니다 [15].
DeepXplore는 수동 라벨링 없이 딥 러닝 모델의 오류 동작을 자동으로 식별할 수 있습니다.
이 기술은 뉴런 커버리지라는 새로운 메트릭을 사용하며, 이는 입력 세트에 의해 탐색된 신경망의 논리 양을 추정합니다.
이 뉴런 커버리지 메트릭은 신경망에서 활성화된 뉴런의 비율을 계산합니다.

Differential testing is a pseudo-oracle testing approach that has been successfully applied to traditional software that do not have a reference test oracle [16].
It is based on the intuition that any divergence between programs’ behaviors, solving the same problem, on the same input data is a probably due to an error.

차별적 테스트는 참조 테스트 오라클이 없는 전통적인 소프트웨어에 성공적으로 적용된 의사 오라클 테스트 접근 방식입니다 [16].
이는 동일한 입력 데이터에 대해 동일한 문제를 해결하는 프로그램 간의 동작 차이가 오류로 인해 발생할 가능성이 높다는 직관에 기반합니다.

DeepXplore leverages a group of similar deep neural networks that solve the same problem. 
Applying differential testing to deep learning with the aim of finding a large number of difference-inducing inputs while maximizing neuron coverage can be formulated as a joint optimization problem.

DeepXplore는 동일한 문제를 해결하는 유사한 딥 뉴럴 네트워크 그룹을 활용합니다.
뉴런 커버리지를 최대화하면서 차이 유발 입력을 많이 찾는 것을 목표로 하는 딥 러닝에 차별적 테스트를 적용하는 것은 공동 최적화 문제로 공식화될 수 있습니다.

Ma et al.[17] generalized the concept of neuron coverage by proposing DeepGauge, a set of multi-granularity testing criteria for Deep Learning systems.
DeepGauge measures the testing quality of test data (whether it being genuine or synthetic) in terms of its capacity to trigger both major function regions as well as the corner-case regions of DNNs(Deep Neural Networks).
It separates DNNs testing coverage in two different levels.

Ma et al.[17]은 DeepGauge라는 딥 러닝 시스템을 위한 다중 세분화 테스트 기준 세트를 제안하여 뉴런 커버리지 개념을 일반화했습니다.
DeepGauge는 테스트 데이터(진짜든 합성이든)의 테스트 품질을 DNN(딥 뉴럴 네트워크)의 주요 기능 영역뿐만 아니라 코너 케이스 영역을 유발하는 능력 측면에서 측정합니다.
이는 DNN 테스트 커버리지를 두 가지 다른 수준으로 나눕니다.

At the neuron-level, the first criterion is k-multisection neuron coverage, where the range of values observed during training sessions for each neuron are divided into k sections to assess the relative frequency of returning a value belonging to each section.
They introduced the concept of neuron boundary coverage to measure how well the test datasets can push activation values to go above and below a pre-defined bound (i.e., covering the upper boundary and the lower boundary values).

뉴런 수준에서 첫 번째 기준은 k-분할 뉴런 커버리지로, 각 뉴런에 대해 훈련 세션 동안 관찰된 값 범위를 k 개의 섹션으로 나누어 각 섹션에 속하는 값을 반환하는 상대적 빈도를 평가합니다.
그들은 테스트 데이터 세트가 활성화 값을 사전에 정의된 경계를 초과하거나 하회하도록 밀어붙이는 능력을 측정하기 위해 뉴런 경계 커버리지 개념을 도입했습니다(즉, 상한 값과 하한 값을 커버하는 것).

At layer-level, the authors leveraged recent findings that empirically showed the potential usefulness of discovered patterns within the hyperactive neurons, which render relatively larger outputs.

계층 수준에서, 저자들은 상대적으로 더 큰 출력을 생성하는 과활성 뉴런 내에서 발견된 패턴의 잠재적 유용성을 경험적으로 보여주는 최근 연구 결과를 활용했습니다.

In their empirical evaluation, Ma et al. showed that DeepGauge scales well to practical sized DNN models (e.g., VGG-19, ResNet-50) and that it could capture erroneous behavior introduced by four state-of-the-art adversarial data generation algorithms (i.e., Fast Gradient Sign Method (FGSM) [10], Basic Iterative Method (BIM) [18], Jacobian-based Saliency Map Attack (JSMA)[19] and Carlini/Wagner attack (CW) [13]).
Therefore, a higher coverage of their criteria potentially plays a substantial role, in improving the detection of errors in the DNNs.

경험적 평가에서 Ma et al.은 DeepGauge가 실용적인 크기의 DNN 모델(VGG-19, ResNet-50 등)에 잘 확장되며, 네 가지 최첨단 적대적 데이터 생성 알고리즘(Fast Gradient Sign Method (FGSM) [10], Basic Iterative Method (BIM) [18], Jacobian-based Saliency Map Attack (JSMA) [19], Carlini/Wagner attack (CW) [13])에 의해 도입된 오류 동작을 포착할 수 있음을 보여주었습니다.
따라서 그들의 기준에 대한 더 높은 커버리지는 DNN에서 오류 감지를 개선하는 데 상당한 역할을 할 수 있습니다.

Building on the pioneer work of Pei et al., Tian et al. proposed DeepTest [20], a tool for automated testing of DNN-driven autonomous cars.
In DeepTest, Tian et al. expanded the notion of neuron coverage proposed by Pei et al. for CNNs (Convolutional Neural Networks), to other types of neural networks, including RNNs (Recurrent Neural Networks).
Moreover, instead of randomly injecting perturbations in input image data, DeepTest focuses on generating realistic synthetic images by applying realistic image transformations like changing brightness, contrast, translation, scaling, horizontal shearing, rotation, blurring, fog effect, and rain effect, etc.
They also mimic different real-world phenomena like camera lens distortions, object movements, different weather conditions, etc.
They argue that generating inputs that maximize neuron coverage cannot test the robustness of trained DNN unless the inputs are likely to appear in the real-world.

Pei et al.의 선구적인 작업을 기반으로 Tian et al.은 DNN 기반 자율 주행 자동차의 자동 테스트를 위한 도구인 DeepTest를 제안했습니다 [20].
DeepTest에서 Tian et al.은 Pei et al.이 CNN(Convolutional Neural Networks)을 위해 제안한 뉴런 커버리지 개념을 RNN(Recurrent Neural Networks)을 포함한 다른 유형의 신경망으로 확장했습니다.
또한 입력 이미지 데이터에 무작위로 변동을 주입하는 대신, DeepTest는 밝기, 대비, 이동, 스케일링, 수평 전단, 회전, 흐림, 안개 효과, 비 효과 등과 같은 현실적인 이미지 변환을 적용하여 현실적인 합성 이미지를 생성하는 데 중점을 둡니다.
또한 카메라 렌즈 왜곡, 객체 이동, 다양한 날씨 조건 등 다양한 실제 현상을 모방합니다.
그들은 뉴런 커버리지를 최대화하는 입력을 생성하는 것이 입력이 실제 세계에서 나타날 가능성이 없으면 훈련된 DNN의 강인성을 테스트할 수 없다고 주장합니다.

DeepTest leverages metamorphic relations (MRs) to create a test oracle that allows it to identify erroneous behaviors without requiring multiple DNNs or manual labeling.
Metamorphic testing [21] is another pseudo-oracle software testing technique that allows identifying erroneous behaviors by detecting violations of domain-specific metamorphic relations (MR).

DeepTest는 메타모픽 관계(MR)를 활용하여 여러 DNN이나 수동 레이블링 없이 오류 동작을 식별할 수 있는 테스트 오라클을 생성합니다.
메타모픽 테스트 [21]는 도메인별 메타모픽 관계(MR) 위반을 감지하여 오류 동작을 식별할 수 있는 또 다른 의사 오라클 소프트웨어 테스트 기법입니다.

DeepRoad [22] continued the same line of work as DeepTest, designing a systematic mechanism for the automatic generation of test cases for DNNs used in autonomous driving cars.
Data sets capturing complex real-world driving situations is generated and Metamorphic Testing is applied to map each data point into the predicted continuous output.
However, DeepRoad differentiates from DeepTest in the approach used to generate new test images.
DeepRoad relies on a Generative Adversarial Network (GAN)-based method to provide realistic snowy and rainy scenes, which can hardly be distinguished from original scenes and cannot be generated by DeepTest using simple affine transformations.

DeepRoad [22]는 DeepTest와 동일한 작업을 계속 진행하여 자율 주행 자동차에 사용되는 DNN을 위한 테스트 케이스의 자동 생성을 위한 체계적인 메커니즘을 설계했습니다.
복잡한 실제 주행 상황을 포착하는 데이터 세트를 생성하고, 각 데이터 포인트를 예측 연속 출력으로 매핑하기 위해 메타모픽 테스트를 적용합니다.
그러나 DeepRoad는 새로운 테스트 이미지를 생성하는 접근 방식에서 DeepTest와 다릅니다.
DeepRoad는 생성적 적대 신경망(GAN) 기반 방법을 사용하여 현실적인 눈과 비 장면을 제공하는데, 이는 원본 장면과 거의 구분되지 않으며 DeepTest가 단순한 아핀 변환을 사용하여 생성할 수 없는 장면입니다.

Zhang et al. argue that DeepTest synthetic image transformations, such as adding blurring/fog/rain effect filters, cannot simulate complex weather conditions.
To solve this lack of realism in generated data, DeepRoad leveraged a recent unsupervised DNN-based method (i.e., UNIT) which is based on GANs and VAEs, to perform image-to-image transformations.
Evaluation results show that the generative model used by DeepRoad successfully generates realistic scenes, allowing for the detection of thousands of behavioral inconsistencies in well-known autonomous driving systems.

Zhang et al.은 DeepTest의 합성 이미지 변환(예: 흐림/안개/비 효과 필터 추가)이 복잡한 날씨 조건을 시뮬레이션할 수 없다고 주장합니다.
생성된 데이터의 현실성 부족을 해결하기 위해 DeepRoad는 GAN 및 VAE를 기반으로 하는 최근 비지도 DNN 기반 방법(즉, UNIT)을 활용하여 이미지 간 변환을 수행했습니다.
평가 결과, DeepRoad에서 사용된 생성 모델은 수천 개의 행동 불일치를 잘 알려진 자율 주행 시스템에서 성공적으로 감지할 수 있도록 현실적인 장면을 생성했습니다.

Despite the relative success of DeepXplore, DeepTest, and DeepRoad, in increasing the test coverage of neural networks, Ma et al. [23] remarked that the runtime state space is very large when each neuron output is considered as a state, which can lead to a combinatorial explosion.
To help address this issue, they proposed DeepCT, which is a new testing method that adapts combinatorial testing (CT) techniques to deep learning models, in order to reduce the testing space.
CT [24] has been successfully applied to test traditional software requiring many configurable parameters.

DeepXplore, DeepTest 및 DeepRoad가 신경망의 테스트 커버리지를 증가시키는 데 상대적인 성공을 거두었음에도 불구하고, Ma et al. [23]은 각 뉴런 출력을 상태로 간주할 때 런타임 상태 공간이 매우 커져 조합 폭발을 초래할 수 있다고 언급했습니다.
이 문제를 해결하기 위해 그들은 DeepCT를 제안했는데, 이는 조합 테스트(CT) 기법을 딥 러닝 모델에 적용하여 테스트 공간을 줄이는 새로운 테스트 방법입니다.
CT [24]는 많은 구성 가능한 매개변수가 필요한 전통적인 소프트웨어를 테스트하는 데 성공적으로 적용되었습니다.

Ma et al. [23] conducted an empirical study, comparing the 2-way CT cases with random testing in terms of the number of adversarial examples detected.
They observed that random testing was ineffective even when a large number of tests were generated.

Ma et al. [23]은 2-웨이 CT 케이스와 무작위 테스트를 비교하여 감지된 적대적 예제의 수를 경험적으로 연구했습니다.
그들은 많은 수의 테스트가 생성되었음에도 불구하고 무작위 테스트가 효과적이지 않다는 것을 관찰했습니다.

Sun et al. [26] examined the effectiveness of the neuron coverage metric introduced by DeepXplore and report that a 100% neuron coverage can be easily achieved by a few test data points while missing multiples incorrect behaviors of the model.
To illustrate this fact, they showed how 25 randomly selected images from the MNIST test set yield a close to 100% neuron coverage for an MNIST classifier.
Thereby, they argue that testing DNNs should take into account the semantic relationships between neurons in adjacent layers in the sense that deeper layers use previous neurons’ information represented by computed features and summarize them in more complex features. 

Sun et al. [26]은 DeepXplore에서 도입한 뉴런 커버리지 메트릭의 효과를 조사했으며, 100% 뉴런 커버리지는 몇 개의 테스트 데이터 포인트로 쉽게 달성할 수 있지만 모델의 여러 잘못된 동작을 놓칠 수 있다고 보고했습니다.
이 사실을 설명하기 위해, 그들은 MNIST 테스트 세트에서 임의로 선택한 25개의 이미지가 MNIST 분류기에 대해 거의 100% 뉴런 커버리지를 제공하는 방법을 보여주었습니다.
따라서, 그들은 DNN을 테스트할 때 인접 계층의 뉴런 간의 의미론적 관계를 고려해야 한다고 주장합니다.
이는 더 깊은 계층이 계산된 특징으로 표현된 이전 뉴런의 정보를 사용하고 이를 더 복잡한 특징으로 요약하기 때문입니다.

To propose a solution to this problem, they adapted the concept of Modified Condition/Decision Coverage (MC/DC)[27] developed by NASA.
They propose a testing approach that consists of a set of four criteria inspired by MC/DC and a test cases generator based on linear programming (LP). 

이 문제에 대한 해결책을 제안하기 위해, 그들은 NASA가 개발한 수정된 조건/결정 커버리지(MC/DC) [27] 개념을 적응했습니다.
그들은 MC/DC에서 영감을 받은 네 가지 기준 세트와 선형 프로그래밍(LP)을 기반으로 한 테스트 케이스 생성기를 포함하는 테스트 접근 방식을 제안합니다.

Sun et al. [28] also applied concolic testing [29] to DNNs. Concolic testing combines concrete executions and symbolic analysis to explore the execution paths of a program that are hard to cover by blind test cases generation techniques such as random testing.
The authors first formulate an objective function that contains a set of existing DNN-related coverage requirements using Quantified Linear Arithmetic over Rationals (QLAR). Then, their proposed method incrementally finds inputs data that satisfy each test coverage requirement in the objective.
The process finishes by providing a test suite that helps reaching a satisfactory level of coverage.

Sun et al. [28]은 또한 DNN에 콘콜릭 테스트 [29]를 적용했습니다. 콘콜릭 테스트는 구체적 실행과 기호적 분석을 결합하여 무작위 테스트와 같은 맹목적인 테스트 케이스 생성 기법으로는 커버하기 어려운 프로그램의 실행 경로를 탐색합니다.
저자들은 먼저 정수에 대한 정량적 선형 산술(QLAR)을 사용하여 기존 DNN 관련 커버리지 요구 사항 집합을 포함하는 목적 함수를 공식화합니다. 그런 다음, 제안된 방법은 목적 내의 각 테스트 커버리지 요구 사항을 충족하는 입력 데이터를 점진적으로 찾습니다.
이 과정은 만족스러운 수준의 커버리지를 달성하는 데 도움이 되는 테스트 스위트를 제공함으로써 완료됩니다.
