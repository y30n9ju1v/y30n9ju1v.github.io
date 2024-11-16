+++
title = 'DeepGauge 2'
date = 2024-08-12T19:12:06+09:00
draft = true
+++

이 글은 DeepGauge: Multi-Granularity Testing Criteria for Deep Learning Systems (https://arxiv.org/abs/1803.07519)을 번역 및 요약한 글입니다.
[DeepGauge 1]({{< ref "DeepGauge-1" >}}) 에서 이어집니다.

## 4. Experiments
We implement DeepGauge on Keras 2.1.3 [10] with TensorFlow 1.5.0 backend [1], and apply the proposed testing criteria to DNNs for evaluation in this section.

우리는 Keras 2.1.3 [10]과 TensorFlow 1.5.0 백엔드 [1]에서 DeepGauge를 구현하였으며, 이 섹션에서 제안된 테스트 기준을 DNN(딥 뉴럴 네트워크)에 적용하여 평가를 수행합니다.

### 4.1 Evaluation Subjects
Datasets and DNN Models. We select two popular publicly-available datasets, i.e., MNIST [32] and ImageNet [42] (see Table 1) for eval- uation. MNIST is for handwritten digits recognition, containing 70,000 input data in total, of which 60,000 are training data and 10,000 are test data. On MNIST, we use three pre-trained LeNet family models (LeNet-1, LeNet-4, and LeNet-5) [32] for analysis.

데이터셋 및 DNN 모델. 우리는 평가를 위해 두 가지의 널리 사용되는 공개 데이터셋인 MNIST [32]와 ImageNet [42]를 선택했습니다(표 1 참조). MNIST는 손글씨 숫자 인식용 데이터셋으로, 총 70,000개의 입력 데이터를 포함하고 있으며, 이 중 60,000개는 훈련 데이터이고 10,000개는 테스트 데이터입니다. MNIST에서 우리는 분석을 위해 세 가지 사전 학습된 LeNet 계열 모델(LeNet-1, LeNet-4, LeNet-5) [32]을 사용합니다.

![Table 1](/posts/paper/DeepGauge/table1.png)

To further demonstrate the usefulness of our criteria towards larger scale real-world DL systems, we also select ImageNet, a large set of general image dataset (i.e., ILSVRC-2012 [42]) for classification containing more than 1.4 million training data and 50,000 test data from 1,000 categories. The DNNs we used for ImageNet are pre- trained VGG-19 [44] and ResNet-50 [24] models, both of which are relatively large in size and obtain competitive records in the ILSVRC competition [42], containing more than 16,000 and 94,000 neurons, and 25 and 176 layers, respectively. As a DNN testing criterion towards future industry level application, we believe the scalability up-to ImageNet-like or even larger data size and model size is almost indispensable.

우리의 기준이 대규모 실세계 딥러닝 시스템에서 얼마나 유용한지를 더욱 입증하기 위해, 우리는 또한 분류를 위해 1,000개 카테고리에서 140만 개 이상의 훈련 데이터와 50,000개의 테스트 데이터를 포함하는 대규모 일반 이미지 데이터셋인 ImageNet (즉, ILSVRC-2012 [42])을 선택했습니다. ImageNet에 사용된 DNN은 사전 학습된 VGG-19 [44]와 ResNet-50 [24] 모델이며, 이들 모델은 모두 상대적으로 큰 규모이며 ILSVRC 대회 [42]에서 경쟁력 있는 성적을 기록한 바 있습니다. 각각 16,000개 이상의 뉴런과 94,000개 이상의 뉴런, 그리고 25층과 176층으로 구성되어 있습니다. 미래의 산업 수준 응용을 위한 DNN 테스트 기준으로서, ImageNet과 같은 또는 그보다 더 큰 데이터 크기와 모델 크기에 대한 확장 가능성은 거의 필수적이라고 믿습니다.

Adversarial Test Input Generation. Besides using original test data accompanied in the corresponding dataset for coverage evalu- ation, we further explore four state-of-the-art adversarial test input generation techniques (i.e., FGSM [20], BIM [31], JSMA [37], and CW [8]) for comparative study. Each of the adversarial techniques generates tests to detect DNN’s potential defects through the minor perturbations on a given input, described as follows:

적대적 테스트 입력 생성. 커버리지 평가를 위해 해당 데이터셋에 포함된 원본 테스트 데이터를 사용하는 것 외에도, 우리는 네 가지 최신 적대적 테스트 입력 생성 기술(FGSM [20], BIM [31], JSMA [37], CW [8])을 비교 연구를 위해 추가로 탐구합니다. 각각의 적대적 기술은 주어진 입력에 대해 미세한 변화를 가하여 DNN의 잠재적 결함을 탐지하는 테스트를 생성합니다. 이들 기술은 다음과 같이 설명됩니다:

FGSM crafts adversarial examples using loss function J(Θ,x,y) with respect to the input feature vector, where Θ denotes the model parameters, x is the input, and y is the output label of x, the adversarial example is generated as: x∗ = x+ε sign(∇x J (Θ, x, y)).

FGSM(Fast Gradient Sign Method)은 입력 특징 벡터에 대해 손실 함수  J(\Theta, x, y) 를 사용하여 적대적 예제를 생성합니다. 여기서  \Theta 는 모델 파라미터를 나타내고,  x 는 입력,  y 는  x 의 출력 레이블입니다. 적대적 예제는 다음과 같이 생성됩니다:
x^* = x + \epsilon \cdot \text{sign}(\nabla_x J(\Theta, x, y))

BIM applies adversarial noise η many times iteratively with a small parameter ε , rather than one η with one ε at a time, which gives a recursive formula: x∗0 = x and xi∗ = clipx,ε(xi∗−1 + ε sign(∇xi∗−1 J(Θ,xi∗−1,y))), where clipx,ε(·) denotes a clipping of the values of the adversarial sample such that they are within an ε-neighborhood of the original input x.

BIM(Basic Iterative Method)은 적대적 노이즈 \(\eta\)를 한 번에 하나의 \(\epsilon\)으로 적용하는 대신, 작은 파라미터 \(\epsilon\)을 사용하여 여러 번 반복적으로 적용합니다. 이를 통해 다음과 같은 재귀 공식을 제공합니다: \(x^*_0 = x\) 그리고 \(x^*_i = \text{clip}_{x,\epsilon}(x^*_{i-1} + \epsilon \cdot \text{sign}(\nabla_{x^*_{i-1}} J(\Theta, x^*_{i-1}, y)))\), 여기서 \(\text{clip}_{x,\epsilon}(\cdot)\)은 적대적 샘플의 값이 원본 입력 \(x\)의 \(\epsilon\)-이웃 안에 있도록 제한하는 클리핑 연산을 나타냅니다.

JSMA is proposed for targeted misclassification. For an input x and a neural network F , the output of class j is denoted as F j (x). To achieve a target misclassification class t , Ft (x) is increased while the probabilities Fj (x) of all other classes j , t decrease, until t = arg maxj Fj (x).

JSMA(JSMA는 목표된 오분류를 위해 제안되었습니다. 입력 \(x\)와 신경망 \(F\)에 대해, 클래스 \(j\)의 출력은 \(F^j(x)\)로 표시됩니다. 목표 오분류 클래스 \(t\)를 달성하기 위해, \(F^t(x)\)는 증가시키고 다른 모든 클래스 \(j\)에 대한 확률 \(F^j(x)\)는 감소시킵니다. 이를 통해 \(t = \arg\max_j F^j(x)\)가 될 때까지 진행합니다.

Carlini/Wagner (CW): Carlini and Wagner recently proposed new optimization-based attack technique which is arguably the most effective in terms of the adversarial success rates achieved with minimal perturbation [8]. In principle, the CW attack is to approximate the solution to the following optimization problem:

Carlini/Wagner (CW): Carlini와 Wagner는 최근 새로운 최적화 기반 공격 기법을 제안했으며, 이는 최소한의 변형으로 달성된 적대적 성공률 측면에서 가장 효과적인 것으로 평가받고 있습니다 [8]. 원칙적으로, CW 공격은 다음 최적화 문제의 해를 근사하는 것을 목표로 합니다.

where L is a loss function to measure the distance between the prediction and the ground truth, and the constant λ is to balance the two loss contributions. In this paper, we adopt the CW∞, where each pixel is allowed to be changed by up to a limit.

여기서 \(L\)은 예측과 실제 값 간의 거리를 측정하는 손실 함수이며, 상수 \(\lambda\)는 두 손실 기여도를 균형 있게 조정하기 위해 사용됩니다. 본 논문에서는 각 픽셀이 일정 한도까지 변경될 수 있는 CW\(_\infty\)를 채택합니다.

Figure 2 shows examples of the generated tests of the four ad- versarial techniques on the sampled data from MNIST test set. In this example, we could see that compared with FGSM and BIM, JSMA and CW perturb fewer pixels on the sampled test input. Fur- thermore, given the same input data but different DNNs, the same technique would often generate different adversarial test results. For example, given the input image 7, JSMA generates different re- sults on DNNs (i.e., LeNet-1, LeNet-4 and LeNet-5). In other words, the studied adversarial techniques are often DNN dependent.

그림 2는 MNIST 테스트 세트에서 샘플링된 데이터에 대해 네 가지 적대적 기술로 생성된 테스트 예제를 보여줍니다. 이 예제에서 볼 수 있듯이, FGSM과 BIM에 비해 JSMA와 CW는 샘플된 테스트 입력에서 더 적은 픽셀을 변형시킵니다. 또한 동일한 입력 데이터를 사용하더라도 서로 다른 DNN에 대해 동일한 기술이 종종 다른 적대적 테스트 결과를 생성합니다. 예를 들어, 입력 이미지 '7'에 대해 JSMA는 서로 다른 DNN(LeNet-1, LeNet-4, LeNet-5)에서 다른 결과를 생성합니다. 즉, 연구된 적대적 기술은 종종 DNN에 따라 달라집니다.

### 4.3 Experimental Results
In our experiments, we have seen useful testing feedbacks from multiple perspectives with each testing criterion, showing some unique portrayal of the runtime behavior of DNNs. We first describe some obtained results and then summarize our findings.

우리의 실험에서 각 테스트 기준을 통해 다양한 관점에서 유용한 테스트 피드백을 얻을 수 있었으며, 이는 DNN의 런타임 동작에 대한 독특한 묘사를 보여줍니다. 먼저 얻은 결과를 설명한 후, 우리의 발견을 요약합니다.

#### 4.3.1 MNIST and ImageNet
#### 4.3.2 Findings and Remarks

### 4.4 Comparison with DeepXplore’s Neuron Coverage (DNC)
### 4.5 Threats to Validity and Discussion
The selection of evaluation subjects (i.e., dataset and DNN mod- els) could be a threat to validity. We try to counter this by using the commonly-studied MNIST dataset and the practical large-scale dataset ImageNet; for each studied dataset, we use the well-known pre-trained models of different sizes and complexity ranging from 52 neurons up to more than 90,000 neurons. Even though, some of our results might not generalize to other datasets and DNN models. Another threat could be caused by the configurable hyper- parameters in the coverage criteria definition. As a countermeasure while considering the limited computational resources, we evaluate each criterion with different settings, and analyze the influence of the parameters on criteria accuracy. Even though, it might still not cover the best parameter use-cases. For example, our evaluation studied k = 1, 000 and k = 10, 000 for k -multisection neuron cover- age. We leave the optimized hyper-parameter selection in our future work. Further threat could be caused by the quality of training data used for distribution (i.e., the interval range) analysis of neuron output. In this paper, we consider publicly available well-pretrained DNN models accompanied by training data with good quality.

평가 대상(즉, 데이터셋 및 DNN 모델)의 선택은 타당성에 대한 위협이 될 수 있습니다. 이를 방지하기 위해, 우리는 일반적으로 연구되는 MNIST 데이터셋과 실용적인 대규모 데이터셋인 ImageNet을 사용했습니다. 각 데이터셋에 대해, 52개의 뉴런에서 90,000개 이상의 뉴런에 이르는 다양한 크기와 복잡성을 가진 잘 알려진 사전 학습 모델을 사용했습니다. 그럼에도 불구하고, 우리의 일부 결과는 다른 데이터셋과 DNN 모델에 일반화되지 않을 수 있습니다. 또 다른 위협은 커버리지 기준 정의에서 사용되는 구성 가능한 하이퍼파라미터로 인해 발생할 수 있습니다. 제한된 계산 자원을 고려하여 각 기준을 다양한 설정으로 평가하고, 기준의 정확성에 대한 파라미터의 영향을 분석하는 방법으로 대응하고자 했습니다. 그럼에도 불구하고, 여전히 최적의 파라미터 사용 사례를 완전히 다루지 못할 수 있습니다. 예를 들어, 우리의 평가에서는 k-멀티섹션 뉴런 커버리지에 대해 \(k = 1,000\)과 \(k = 10,000\)을 연구했습니다. 최적의 하이퍼파라미터 선택은 향후 연구로 남겨둡니다. 또 다른 위협은 뉴런 출력의 분포(즉, 구간 범위) 분석에 사용된 훈련 데이터의 품질로 인해 발생할 수 있습니다. 본 논문에서는 공공적으로 이용 가능한, 잘 사전 학습된 DNN 모델과 품질이 좋은 훈련 데이터를 고려했습니다.

For adversarial test generation, we select four popular state-of- the-practice techniques to simulate defects from different sources and granularity. We either follow the authors’ suggested settings or use their default settings. Moreover, to make comprehensive com- parisons with DeepXplore’s neuron coverage (DNC), we evaluate DNC with multiple threshold settings.

적대적 테스트 생성에 있어, 우리는 다양한 출처와 세분화에서 결함을 시뮬레이션하기 위해 널리 사용되는 네 가지 최신 기법을 선택했습니다. 각 기법에 대해서는 저자들이 제안한 설정을 따르거나 기본 설정을 사용했습니다. 또한 DeepXplore의 뉴런 커버리지(DNC)와의 포괄적인 비교를 위해, DNC를 여러 임계값 설정으로 평가했습니다.

## 5. Related Work
### 5.1 Testing of DL Systems
Traditional practices in measuring machine learning systems mainly rely on probing their accuracy on test inputs which are randomly drawn from manually labeled datasets and ad hoc simulations [54]. However, such black-box testing methodology may not be able to find various kinds of corner-case behaviors that may induce un- expected errors [19]. Wicker et al. [53] recently proposed a Scale Invariant Feature Transform feature guided black-box testing and showed its competitiveness with CW and JSMA along this direction.

전통적인 머신러닝 시스템 성능 측정 방식은 주로 수동으로 라벨링된 데이터셋에서 무작위로 추출된 테스트 입력에 대한 정확도를 평가하고, 임시 시뮬레이션에 의존합니다 [54]. 그러나 이러한 블랙박스 테스트 방법론은 예기치 않은 오류를 유발할 수 있는 다양한 코너 케이스 동작을 발견하지 못할 수 있습니다 [19]. Wicker 등 [53]은 최근에 Scale Invariant Feature Transform (SIFT) 특성에 기반한 블랙박스 테스트를 제안했으며, 이 방법이 CW와 JSMA와의 비교에서 경쟁력이 있음을 보였습니다.

Pei et al. [38] proposed a white-box differential testing algorithm for systematically finding inputs that can trigger inconsistencies between multiple DNNs. They introduced neuron coverage for measuring how much of the internal logic of a DNN has been tested.However, it still exhibits several caveats as discussed in Sec- tion 4.4. DeepTest [49] investigates a basic set of image transforma- tions (e.g., scaling, shearing, and rotation) from OpenCV and shows that they are useful to detect defects in DNN-driven autonomous cars. Along this direction, DeepRoad [59] uses input image scene transformation and shows its potentiality with two scenes (i.e., snowy and rainy) for autonomous driving testing. The scene trans- formation is obtained through training a generative adversarial network (GAN) with a pair of collected training data that cover the statistical features of the two target scenes.

Pei 등 [38]은 여러 DNN 간의 불일치를 유발할 수 있는 입력을 체계적으로 찾기 위해 화이트박스 차등 테스트 알고리즘을 제안했습니다. 그들은 DNN의 내부 논리가 얼마나 테스트되었는지를 측정하기 위해 뉴런 커버리지를 도입했습니다. 그러나 이 방법은 여전히 4.4절에서 논의된 몇 가지 단점을 가지고 있습니다. DeepTest [49]는 OpenCV의 기본적인 이미지 변환 세트(예: 스케일링, 전단, 회전)를 조사하여, 이러한 변환이 DNN 기반 자율 주행 자동차의 결함을 감지하는 데 유용함을 보여주었습니다. 이 방향에서 DeepRoad [59]는 입력 이미지 장면 변환을 사용하여 자율 주행 테스트에서 두 가지 장면(즉, 눈이 내리는 장면과 비가 내리는 장면)에서의 잠재력을 보여줍니다. 장면 변환은 두 가지 대상 장면의 통계적 특징을 포함하는 수집된 훈련 데이터 쌍을 사용하여 생성적 적대 신경망(GAN)을 훈련시킴으로써 얻어집니다.

Compared with traditional software, the dimension and potential testing space of a DNN is often quite large. DeepCT [35] adapts the concept of combinatorial testing, and proposes a set of coverage based on the neuron input interaction for each layer of DNNs, to guide test generation towards achieving reasonable defect detec- tion ability with a relatively small number of tests. Inspired by the MC/DC test criteria in traditional software [29], Sun et al. [47] proposed a set of adapted MC/DC test criteria for DNNs, and show that generating tests guided by the proposed criteria on small scale neural networks (consisting of Dense layers with no more than 5 hidden layers and 400 neurons) exhibits higher defect detection ability than random testing. However, whether MC/DC criteria scale to real-world-sized DL systems still needs further investiga- tion. Instead of observing the runtime internal behaviors of DNNs, DeepMutation [34] proposes to mutate DNNs (i.e., injecting faults either from the source level or model level) to evaluate the test data quality, which could potentially be useful for test data prioritization in respect of robustness on a given DNN.

전통적인 소프트웨어와 비교할 때, DNN의 차원과 잠재적인 테스트 공간은 종종 매우 큽니다. DeepCT [35]는 조합적 테스트 개념을 적용하여, 각 DNN 레이어의 뉴런 입력 상호작용을 기반으로 한 커버리지 세트를 제안하고, 상대적으로 적은 수의 테스트로 합리적인 결함 탐지 능력을 달성할 수 있도록 테스트 생성 가이드를 제공합니다. 전통적인 소프트웨어의 MC/DC 테스트 기준 [29]에서 영감을 받아, Sun 등 [47]은 DNN을 위한 적응된 MC/DC 테스트 기준을 제안하였고, 제안된 기준에 따라 소규모 신경망(숨겨진 레이어가 5개 이하이고 400개의 뉴런으로 구성된 Dense 레이어)에서 생성된 테스트가 무작위 테스트보다 더 높은 결함 탐지 능력을 보인다고 밝혔습니다. 그러나 MC/DC 기준이 실제 크기의 딥러닝 시스템에 적용될 수 있는지 여부는 여전히 추가 조사가 필요합니다. DNN의 런타임 내부 동작을 관찰하는 대신, DeepMutation [34]은 DNN을 변형(즉, 소스 레벨 또는 모델 레벨에서 결함을 주입)하여 테스트 데이터 품질을 평가하는 방법을 제안하며, 이는 주어진 DNN의 견고성과 관련하여 테스트 데이터 우선순위를 지정하는 데 유용할 수 있습니다.

Our work of proposing multi-granularity testing coverage for DL systems is mostly orthogonal to the existing work. Compared with the extensive study on traditional software testing, testing DL is still at an early stage. Most existing work on DL testing lacks some suitable criteria to understand and guide the test generation process. Since test generation guided by coverage criteria (e.g., statement coverage, branch coverage) towards the exploration of diverse soft- ware states for defect detection has become the de facto standard in traditional software testing [5, 16, 17, 33], the study to design suitable testing criteria for DL is desperately demanding. This paper makes an early attempt towards this direction by proposing a set of testing criteria. Our criteria not only can differentiate state-of-the- art adversarial test generation techniques, but also potentially be useful for the measurement of test suite diversity by analyzing the DNNs’ internal states from multiple portrayals. We believe that our proposed criteria set up an important cornerstone and bring a new opportunity to design more effective automated testing techniques guided by testing criteria for DL systems.

딥러닝 시스템을 위한 다중 세분화 테스트 커버리지를 제안하는 우리의 연구는 기존 연구와 대부분 독립적입니다. 전통적인 소프트웨어 테스트에 대한 광범위한 연구와 비교할 때, 딥러닝 테스트는 여전히 초기 단계에 있습니다. 대부분의 기존 딥러닝 테스트 연구는 테스트 생성 과정을 이해하고 가이드할 수 있는 적절한 기준이 부족합니다. 결함 탐지를 위한 다양한 소프트웨어 상태를 탐색하기 위해 커버리지 기준(예: 문장 커버리지, 분기 커버리지)에 따라 테스트를 생성하는 것이 전통적인 소프트웨어 테스트의 사실상 표준이 된 것처럼 [5, 16, 17, 33], 딥러닝을 위한 적절한 테스트 기준을 설계하는 연구가 절실히 필요합니다. 본 논문은 이 방향으로의 초기 시도로서, 일련의 테스트 기준을 제안합니다. 우리의 기준은 최신 적대적 테스트 생성 기술을 구별할 수 있을 뿐만 아니라, DNN의 내부 상태를 여러 관점에서 분석하여 테스트 스위트 다양성을 측정하는 데도 유용할 가능성이 있습니다. 우리는 제안된 기준이 중요한 초석을 마련하고, 딥러닝 시스템을 위한 테스트 기준에 따라 보다 효과적인 자동화된 테스트 기술을 설계할 수 있는 새로운 기회를 제공할 것이라고 믿습니다.

### 5.2 Verification of DL Systems
Formal methods can provide formal guarantees about safety and robustness of verified DL systems [22, 28, 39, 40, 50, 53]. The main concern of formal methods are their scalability for real-world-sized (e.g., 100, 000 neurons or even more) DL systems.

형식적 방법(Formal methods)은 검증된 딥러닝 시스템의 안전성과 견고성에 대한 형식적인 보장을 제공할 수 있습니다 [22, 28, 39, 40, 50, 53]. 형식적 방법의 주요 우려 사항은 실제 크기의 딥러닝 시스템(예: 100,000개 이상의 뉴런)에 대한 확장성입니다.

The early work in [40] provided an abstraction-refinement ap- proach to checking safety properties of multi-layer perceptrons. Their approach has been applied to verify a network with only 6 neurons. DLV [53] can verify local robustness of DL systems w.r.t. a set of user specified manipulations. Reluplex [28] is a sound and complete SMT-based approach to verifying safety and robustness of DL systems with ReLU activation functions. The networks verified by Reluplex in [28] have 8 layers and 300 ReLU nodes. DeepSafe [22] uses Reluplex as its verification engine and has the same scala- bility problem as Reluplex. AI2 [50] is a sound analyzer based on abstract interpretation that can reason about safety and robustness of DL systems. It trades precision for scalability and scales better than Reluplex. The precision of AI2 depends on abstract domains used in the verification, and it might fail to prove a property when it actually holds. VERIVIS [39] can verify safety properties of DL systems when attackers are constrained to modify the inputs only through given transformation functions. However, real-world trans- formations can be much more complex than the transformation functions considered in the paper.

초기 연구 [40]에서는 다층 퍼셉트론의 안전 속성을 확인하기 위해 추상화-정제 접근법을 제시했습니다. 이 접근법은 단 6개의 뉴런을 가진 네트워크를 검증하는 데 적용되었습니다. DLV [53]는 사용자 지정 조작에 대해 딥러닝 시스템의 국부적 견고성을 검증할 수 있습니다. Reluplex [28]는 ReLU 활성화 함수를 사용하는 딥러닝 시스템의 안전성과 견고성을 검증하기 위한 완전하고 정확한 SMT 기반 접근법입니다. Reluplex로 검증된 네트워크 [28]는 8개의 레이어와 300개의 ReLU 노드를 가지고 있습니다. DeepSafe [22]는 Reluplex를 검증 엔진으로 사용하며, Reluplex와 동일한 확장성 문제를 가지고 있습니다. AI2 [50]는 추상 해석에 기반한 정확한 분석기로, 딥러닝 시스템의 안전성과 견고성을 추론할 수 있습니다. AI2는 정확성을 희생하여 확장성을 확보하며, Reluplex보다 더 잘 확장됩니다. AI2의 정확성은 검증에 사용된 추상 도메인에 따라 달라지며, 실제로 속성이 유지될 때도 증명에 실패할 수 있습니다. VERIVIS [39]는 공격자가 주어진 변환 함수만을 통해 입력을 수정할 수 있는 경우, 딥러닝 시스템의 안전 속성을 검증할 수 있습니다. 그러나 실제 변환은 논문에서 고려된 변환 함수보다 훨씬 더 복잡할 수 있습니다.

### 5.3 Attacks and Defenses of DL Systems
A plethora of research has shown that deep learning systems can be fooled by applying carefully crafted adversarial perturbation added to the original input [6–9, 20, 48, 55, 60], many of which are based on gradient or optimization techniques. However, it still lacks extensive study on how these adversarial techniques differentiate in terms of DNNs’ internal states. In this study, we make an early attempt towards such a direction based on our proposed criteria.

다양한 연구에서 원본 입력에 정교하게 설계된 적대적 변화를 추가하여 딥러닝 시스템을 속일 수 있다는 것이 입증되었습니다 [6–9, 20, 48, 55, 60]. 이들 중 많은 기술이 그래디언트 또는 최적화 기법을 기반으로 합니다. 그러나 이러한 적대적 기술들이 DNN의 내부 상태와 관련하여 어떻게 차별화되는지에 대한 광범위한 연구는 여전히 부족합니다. 본 연구에서는 제안된 기준을 바탕으로 이러한 방향에서 초기 시도를 수행합니다.

With the rapid development of adversarial attack techniques, extensive studies have been performed to circumvent adversarial attacks. Galloway et al. [18] recently observe that low-precision DNNs exhibit improved robustness against some adversarial at- tacks. This is primarily due to the stochastic quantization in neural network weights. Ensemble adversarial training [51], GAN based approaches [43, 45], random resizing and random padding [56], game theory [13], and differentiable certificate [41] methods are all investigated to defend against adversarial examples. By applying image transformations, such as total variance minimization and image quilting, very effective defenses can be achieved when the network is trained on the aforementioned transformed images [23]. For more extensive discussion on current state-of-the-art defense techniques, we refer readers to [4].

적대적 공격 기술의 급속한 발전과 함께, 적대적 공격을 회피하기 위한 광범위한 연구가 수행되었습니다. Galloway 등 [18]은 최근 저정밀도 DNN이 일부 적대적 공격에 대해 향상된 견고성을 보인다는 것을 관찰했습니다. 이는 주로 신경망 가중치에서의 확률적 양자화에 기인합니다. 앙상블 적대적 학습 [51], GAN 기반 접근법 [43, 45], 무작위 크기 조정 및 무작위 패딩 [56], 게임 이론 [13], 그리고 미분 가능한 증명서 [41] 방법 등이 적대적 예제에 대응하기 위해 연구되었습니다. 총 변동 최소화 및 이미지 퀼팅과 같은 이미지 변환을 적용하면, 앞서 언급한 변환된 이미지로 네트워크를 훈련시켰을 때 매우 효과적인 방어를 달성할 수 있습니다 [23]. 현재 최신 방어 기술에 대한 더 광범위한 논의는 [4]를 참조하시기 바랍니다.

Our proposed testing criteria enable the quantitative measure- ment of different adversarial attack techniques from the software engineering perspective. This could be potentially helpful for un- derstanding and interpreting DNNs’ behaviors, based on which more effective DNN defense technique could be designed. In future work, it would be also interesting to examine how to integrate the proposed testing criteria into the DL development life cycle towards building high quality DL systems.

우리가 제안한 테스트 기준은 소프트웨어 엔지니어링 관점에서 다양한 적대적 공격 기술을 정량적으로 측정할 수 있게 합니다. 이는 DNN의 동작을 이해하고 해석하는 데 도움이 될 수 있으며, 이를 바탕으로 더 효과적인 DNN 방어 기술을 설계할 수 있을 것입니다. 향후 연구에서는 제안된 테스트 기준을 딥러닝 개발 생애 주기에 통합하여 고품질의 딥러닝 시스템을 구축하는 방법을 조사하는 것도 흥미로울 것입니다.

## 6. Conclusion and Future Work
The wide adoption of DL systems, especially in many safety-critical areas, has posed a severe threat to its quality and generalization property. To effectively measure the testing adequacy and lay down the foundation to design effective DL testing techniques, we pro- pose a set of testing criteria for DNNs. Our experiments on two well-known datasets, five DNNs with diverse complexity, and four state-of-the-art adversarial testing techniques show that the tests generated by the adversarial techniques incur obvious increases of the coverage in terms of the metrics defined in the paper. This demonstrates that DeepGauge could be a useful indicator for evalu- ating testing adequacy of DNNs.

딥러닝 시스템의 널리 채택, 특히 많은 안전 필수 분야에서, 그 품질과 일반화 특성에 심각한 위협이 제기되고 있습니다. 테스트 적합성을 효과적으로 측정하고 효과적인 딥러닝 테스트 기술 설계를 위한 기반을 마련하기 위해, 우리는 DNN을 위한 일련의 테스트 기준을 제안합니다. 두 개의 잘 알려진 데이터셋, 다양한 복잡성을 가진 다섯 개의 DNN, 그리고 네 가지 최신 적대적 테스트 기술에 대한 실험 결과, 적대적 기술로 생성된 테스트는 이 논문에서 정의한 지표 측면에서 커버리지가 명백히 증가하는 것을 보여줍니다. 이는 DeepGauge가 DNN의 테스트 적합성을 평가하는 유용한 지표가 될 수 있음을 입증합니다.

To the best of our knowledge, our work is among the early studies to propose testing criteria for DL systems. We expect that the proposed testing criteria could be particularly amenable to DL testing in the wild. In the next step, we will continue to explore alternative testing criteria for DNNs, such as the combination of both hyperactive and hypoactive neurons. We also plan to study the proposed testing criteria guided automated test generation techniques for DNNs. We hope that our study not only provides an avenue to illuminate the nature and mechanism of DNNs, but also lays the foundation towards understanding and building generic and robust DL systems.

우리가 아는 한, 우리의 연구는 딥러닝 시스템에 대한 테스트 기준을 제안한 초기 연구 중 하나입니다. 제안된 테스트 기준이 실제 환경에서의 딥러닝 테스트에 특히 적합할 것으로 기대합니다. 다음 단계에서는 과활성 뉴런과 저활성 뉴런의 조합과 같은 대안적인 DNN 테스트 기준을 계속 탐구할 예정입니다. 또한 제안된 테스트 기준을 활용한 DNN의 자동화된 테스트 생성 기술에 대한 연구도 계획하고 있습니다. 우리의 연구가 DNN의 본질과 메커니즘을 밝히는 하나의 길을 제공할 뿐만 아니라, 일반적이고 견고한 딥러닝 시스템을 이해하고 구축하는 데 기초를 마련할 수 있기를 바랍니다.


