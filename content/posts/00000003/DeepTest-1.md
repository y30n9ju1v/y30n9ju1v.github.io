+++
title = 'DeepTest 1'
date = 2024-07-28T12:46:41+09:00
draft = false
+++

이 글은 DeepTest: Automated Testing of Deep-Neural-Network-driven Autonomous Cars (https://arxiv.org/abs/1708.08559)을 번역 및 요약한 글입니다.

## Abstract

despite their spectacular progress, DNNs, just like traditional software, often demonstrate incorrect or unexpected corner-case behaviors that can lead to potentially fatal collisions.
Several such real-world accidents involving autonomous cars have already happened including one which resulted in a fatality.
Most existing testing techniques for DNN-driven vehicles are heavily dependent on the manual collection of test data under different driving conditions which become prohibitively expensive as the number of test conditions increases.

전통적인 소프트웨어와 마찬가지로, DNN도 때때로 잘못된 또는 예상치 못한 코너 케이스 행동을 보이며 이는 잠재적으로 치명적인 충돌로 이어질 수 있습니다.
실제로 몇몇 자율주행차 사고가 이미 발생했으며, 그 중 하나는 치명적인 사고로 이어졌습니다.
현재 DNN 기반 차량의 대부분의 테스트 기술은 다양한 주행 조건에서 테스트 데이터를 수집하는 데 크게 의존하고 있어 테스트 조건이 증가함에 따라 비용이 급격히 증가합니다.

In this paper, we design, implement, and evaluate DeepTest, a systematic testing tool for automatically detecting erroneous behaviors of DNN-driven vehicles that can potentially lead to fatal crashes.
First, our tool is designed to automatically generated test cases leveraging real-world changes in driving conditions like rain, fog, lighting conditions, etc.
DeepTest systematically explore different parts of the DNN logic by generating test inputs that maximize the numbers of activated neurons.
DeepTest found thousands of erroneous behaviors under different realistic driving conditions (e.g., blurring, rain, fog, etc.) many of which lead to potentially fatal crashes in three top performing DNNs in the Udacity self-driving car challenge.

이 논문에서는 치명적인 충돌로 이어질 수 있는 DNN 기반 차량의 오류 동작을 자동으로 감지하는 체계적인 테스트 도구인 DeepTest를 설계, 구현 및 평가합니다.
먼저, 우리 도구는 비, 안개, 조명 조건 등의 실제 주행 조건의 변화를 활용하여 테스트 케이스를 자동으로 생성하도록 설계되었습니다.
DeepTest는 활성화된 뉴런의 수를 최대화하는 테스트 입력을 생성하여 DNN 로직의 다양한 부분을 체계적으로 탐색합니다.
DeepTest는 Udacity 자율주행차 챌린지에서 상위 성능을 보인 세 개의 DNN에서 흐림, 비, 안개 등의 다양한 현실적인 주행 조건에서 수천 건의 오류 동작을 발견했으며, 이 중 많은 부분이 잠재적으로 치명적인 충돌로 이어질 수 있음을 확인했습니다.

## 1. Introduction
despite the tremendous progress, just like traditional software, DNN-based software, including the ones used for autonomous driving, often demonstrate incorrect/unexpected corner-case behaviors that can lead to dangerous consequences like a fatal collision.
Several such real-world cases have already been reported (see Table 1).

전통적인 소프트웨어와 마찬가지로, 자율주행을 포함한 DNN 기반 소프트웨어도 종종 잘못되거나 예상치 못한 코너 케이스 동작을 보여 치명적인 충돌과 같은 위험한 결과를 초래할 수 있습니다.
Table 1에 나와 있듯이, 실제로 몇몇 이러한 사례가 이미 보고되었습니다. 

![Table 1](/posts/paper/DeepTest/table1.png)

At a conceptual level, these erroneous corner-case behaviors in DNN-based software are analogous to logic bugs in traditional software. 
However, this is a challenging problem, as noted by large software companies like Google and Tesla that have already deployed machine learning techniques in several production-scale systems including self-driving car, speech recognition, image search, etc. [22, 73].

개념적 수준에서, DNN 기반 소프트웨어의 이러한 오류 코너 케이스 동작은 전통적인 소프트웨어의 논리적 버그와 유사합니다.
그러나, 이것은 자율주행차, 음성 인식, 이미지 검색 등을 포함한 여러 생산 규모 시스템에 머신러닝 기술을 이미 배포한 구글과 테슬라와 같은 대형 소프트웨어 회사들이 지적했듯이 어려운 문제입니다 [22, 73].

unlike traditional software where the program logic is manually written by the software developers, DNN-based software automatically learns its logic from a large amount of data with minimal human guidance.
In addition, the logic of a traditional program is expressed in terms of control flow statements while DNNs use weights for edges between different neurons and nonlinear activation functions for similar purposes.

소프트웨어 개발자가 프로그램 로직을 수동으로 작성하는 전통적인 소프트웨어와 달리, DNN 기반 소프트웨어는 최소한의 인간 지도만으로 대량의 데이터에서 자동으로 로직을 학습합니다.
또한, 전통적인 프로그램의 로직은 제어 흐름 문장으로 표현되는 반면, DNN은 서로 다른 뉴런 간의 가중치와 유사한 목적을 위한 비선형 활성화 함수를 사용합니다.

traditional software testing techniques for systematically exploring different parts of the program logic by maximizing branch/code coverage is not very useful for DNN-based software as the logic is not encoded using control flow [70].
Next, DNNs are fundamentally different from the models (e.g., finite state machines) used for modeling and testing traditional programs.
Unlike the traditional models, finding inputs that will result in high model coverage in a DNN is significantly more challenging due to the non-linearity of the functions modeled by DNNs.
Moreover, the Satisfiability Modulo Theory (SMT) solvers that have been quite successful at generating high-coverage test inputs for traditional software are known to have trouble with formulas involving floating-point arithmetic and highly nonlinear constraints, which are commonly used in DNNs.

분기/코드 커버리지를 최대화하여 프로그램 로직의 다양한 부분을 체계적으로 탐색하는 전통적인 소프트웨어 테스트 기술은 제어 흐름을 사용하여 로직이 인코딩되지 않는 DNN 기반 소프트웨어에는 그다지 유용하지 않습니다 [70].
다음으로, DNN은 전통적인 프로그램을 모델링하고 테스트하는 데 사용되는 모델(예: 유한 상태 기계)과 근본적으로 다릅니다.
전통적인 모델과 달리, DNN에서 높은 모델 커버리지를 달성할 입력을 찾는 것은 DNN이 모델링하는 함수의 비선형성 때문에 훨씬 더 어렵습니다.
게다가, 전통적인 소프트웨어에 대해 높은 커버리지 테스트 입력을 생성하는 데 성공적인 Satisfiability Modulo Theory(SMT) 솔버는 부동 소수점 산술 및 DNN에서 일반적으로 사용되는 고도의 비선형 제약 조건을 포함하는 수식에서 문제를 겪는 것으로 알려져 있습니다.

We empirically demonstrate that changes in neuron coverage are statistically correlated with changes in the actions of self-driving cars (e.g., steering angle).
Therefore, neuron coverage can be used as a guidance mechanism for systemically exploring different types of car behaviors and identify erroneous behaviors.
Next, we demonstrate that different image transformations that mimic real-world differences in driving conditions like changing contrast/brightness, rotation of the camera result in activation of different sets of neurons in the self-driving car DNNs.
We show that by combining these image transformations, the neuron coverage can be increased by 100% on average compared to the coverage achieved by manual test inputs.
Finally, we use transformation-specific metamorphic relations between multiple executions of the tested DNN (e.g., a car should behave similarly under different lighting conditions) to automatically detect erroneous corner case behaviors.

뉴런 커버리지의 변화가 자율주행차의 행동(예: 조향 각도)의 변화와 통계적으로 상관관계가 있음을 경험적으로 입증합니다.
따라서 뉴런 커버리지는 다양한 차량 동작을 체계적으로 탐색하고 오류 동작을 식별하는 지침 메커니즘으로 사용될 수 있습니다.
다음으로, 대비/밝기 변화, 카메라 회전 등 실제 주행 조건의 차이를 모방한 다양한 이미지 변환이 자율주행차 DNN에서 서로 다른 뉴런 세트를 활성화함을 입증합니다.
이러한 이미지 변환을 결합함으로써 수동 테스트 입력으로 달성한 커버리지에 비해 뉴런 커버리지를 평균 100% 증가시킬 수 있음을 보여줍니다.
마지막으로, 테스트된 DNN의 여러 실행 간의 변환별 변형 관계(예: 자동차는 서로 다른 조명 조건에서 유사하게 작동해야 함)를 사용하여 오류 코너 케이스 동작을 자동으로 감지합니다.

The key contributions of this paper are:
* We present a systematic technique to automatically synthesize
test cases that maximizes neuron coverage in safety-critical DNN-based systems like autonomous cars.
We empirically demonstrate that changes in neuron coverage correlate with changes in an autonomous car’s behavior.
* We demonstrate that different realistic image transformations like changes in contrast, presence of fog, etc. can be used to generate synthetic tests that increase neuron coverage.
We leverage transformation-specific metamorphic relations to automatically detect erroneous behaviors.
Our experiments also show that the synthetic images can be used for retraining and making DNNs more robust to different corner cases.
* We implement the proposed techniques in DeepTest, to the best of our knowledge, the first systematic and automated testing tool for DNN-driven autonomous vehicles.
We use DeepTest to systematically test three top performing DNN models from the Udacity driving challenge.
DeepTest found thousands of erroneous behaviors in these systems many of which can lead to potentially fatal collisions as shown in Figure 1.
* We have made the erroneous behaviors detected by DeepTest available at https://deeplearningtest.github.io/deepTest/.
We also plan to release the generated test images and the source of DeepTest for public use.

이 논문의 주요 기여는 다음과 같습니다:
* 우리는 자율주행차와 같은 안전 필수 DNN 기반 시스템에서 뉴런 커버리지를 최대화하는 테스트 케이스를 자동으로 합성하는 체계적인 기술을 제시합니다.
뉴런 커버리지의 변화가 자율주행차의 행동 변화와 상관관계가 있음을 경험적으로 입증합니다.
* 대비 변화, 안개 등의 다양한 현실적인 이미지 변환을 사용하여 뉴런 커버리지를 증가시키는 합성 테스트를 생성할 수 있음을 입증합니다.
우리는 변환별 변형 관계를 활용하여 오류 동작을 자동으로 감지합니다.
우리의 실험은 합성 이미지를 사용하여 DNN을 재훈련하고 다양한 코너 케이스에 대해 더욱 견고하게 만들 수 있음을 보여줍니다.
* 우리는 제안된 기술을 DeepTest에 구현하였으며, 이는 우리가 알고 있는 한 DNN 기반 자율주행차를 위한 최초의 체계적이고 자동화된 테스트 도구입니다.
DeepTest를 사용하여 Udacity 주행 챌린지에서 상위 성능을 보인 세 개의 DNN 모델을 체계적으로 테스트했습니다.
DeepTest는 이 시스템들에서 수천 건의 오류 동작을 발견했으며, 그 중 많은 부분이 잠재적으로 치명적인 충돌로 이어질 수 있음을 확인했습니다 (그림 1 참조).
* 우리는 DeepTest에 의해 감지된 오류 동작을 https://deeplearningtest.github.io/deepTest/에서 제공했습니다.
또한 생성된 테스트 이미지와 DeepTest의 소스를 공개할 계획입니다.

![Figure 1](/posts/paper/DeepTest/figure1.png)

## 2. Background
### 2.1 Deep Learning for Autonomous Driving
A typical feed-forward DNN is composed of multiple processing layers stacked together to extract different representations of the input [30].
Each layer of the DNN increasingly abstracts the input, e.g., from raw pixels to semantic concepts.
For example, the first few layers of an autonomous car DNN extract low-level features such as edges and directions, while the deeper layers identify objects like stop signs and other cars, and the final layer outputs the steering decision (e.g., turning left or right).

일반적인 피드포워드 DNN은 입력의 다양한 표현을 추출하기 위해 여러 처리 레이어가 함께 쌓여 구성됩니다 [30].
DNN의 각 레이어는 점점 더 입력을 추상화합니다.
예를 들어, 자율주행차 DNN의 첫 몇 개 레이어는 엣지와 방향과 같은 저수준 특징을 추출하고, 더 깊은 레이어는 정지 신호나 다른 차량과 같은 객체를 식별하며, 최종 레이어는 조향 결정(예: 좌회전 또는 우회전)을 출력합니다.

Figure 2 illustrates a basic DNN in the perception module of a self-driving car.
Essentially, the DNN is a sequence of linear transformations (e.g., dot product between the weight parameters θ of each edge and the output value of the source neuron of that edge) and nonlinear activations (e.g., ReLU in each neuron).

그림 2는 자율주행차의 인식 모듈에 있는 기본적인 DNN을 보여줍니다.
기본적으로 DNN은 일련의 선형 변환(예: 각 엣지의 가중치 파라미터 θ와 해당 엣지의 소스 뉴런 출력 값 사이의 내적)과 비선형 활성화(예: 각 뉴런의 ReLU)로 구성됩니다.

![Figure 2](/posts/paper/DeepTest/figure2.png)

## 3. METHODOLOGY
To develop an automated testing methodology for DNN-driven autonomous cars we must answer the following questions.
(i) How do we systematically explore the input-output spaces of an autonomous car DNN? (ii) How can we synthesize realistic inputs to automate such exploration? (iii) How can we optimize the exploration process? (iv) How do we automatically create a test oracle that can detect erroneous behaviors without detailed manual specifications? We briefly describe how DeepTest addresses each of these questions below.

DNN 기반 자율주행차를 위한 자동화된 테스트 방법론을 개발하기 위해 다음 질문에 답해야 합니다.
(i) 자율주행차 DNN의 입력-출력 공간을 어떻게 체계적으로 탐색할 수 있는가? (ii) 그러한 탐색을 자동화하기 위해 현실적인 입력을 어떻게 합성할 수 있는가? (iii) 탐색 과정을 어떻게 최적화할 수 있는가? (iv) 상세한 수동 명세 없이 오류 동작을 감지할 수 있는 테스트 오라클을 어떻게 자동으로 생성할 수 있는가? 아래에서 DeepTest가 이러한 질문 각각을 어떻게 해결하는지 간략히 설명합니다.

### 3.1 Systematic Testing with Neuron Coverage
The input-output space (i.e., all possible combinations of inputs and outputs) of a complex system like an autonomous vehicle is too large for exhaustive exploration.
Therefore, we must devise a systematic way of partitioning the space into different equivalence classes and try to cover all equivalence classes by picking one sample from each of them.
In this paper, we leverage neuron coverage [70] as a mechanism for partitioning the input space based on the assumption that all inputs that have similar neuron coverage are part of the same equivalence class (i.e., the target DNN behaves similarly for these inputs).

"자율 주행 차량과 같은 복잡한 시스템의 입력-출력 공간(즉, 가능한 모든 입력과 출력의 조합)은 너무 커서 철저한 탐사가 불가능합니다.
따라서 우리는 공간을 여러 동등성 클래스로 분할하고 각 클래스에서 하나의 샘플을 선택하여 모든 동등성 클래스를 포괄하려는 체계적인 방법을 고안해야 합니다.
이 논문에서는 뉴런 커버리지[70]를 입력 공간을 분할하는 메커니즘으로 활용하며, 유사한 뉴런 커버리지를 가진 모든 입력이 동일한 동등성 클래스의 일부라고 가정합니다(즉, 대상 DNN이 이러한 입력에 대해 유사하게 동작함)."

Neuron coverage is defined as the ratio of unique neurons that get activated for given input(s) and the total number of neurons in a DNN.
An individual neuron is considered activated if the neuron’s output (scaled by the overall layer’s outputs) is larger than a DNN-wide threshold.
In this paper, we use 0.2 as the neuron activation threshold for all our experiments.

뉴런 커버리지는 주어진 입력에 대해 활성화되는 고유 뉴런의 수와 DNN의 전체 뉴런 수의 비율로 정의됩니다.
개별 뉴런은 뉴런의 출력(레이어의 전체 출력에 의해 스케일링된 값)이 DNN 전체의 임계값보다 큰 경우 활성화된 것으로 간주됩니다.
이 논문에서는 모든 실험에 대해 뉴런 활성화 임계값으로 0.2를 사용합니다.

Pei et al. defined neuron coverage only for CNNs [70].
We further generalize the definition to include RNNs.
Neurons, depending on the type of the corresponding layer, may produce different types of output values (i.e. single value and multiple values organized in a multidimensional array).
We describe how we handle such cases in detail below.

Pei 등은 뉴런 커버리지를 CNN에 대해서만 정의했습니다 [70].
우리는 이를 확장하여 RNN도 포함하도록 정의를 일반화합니다.
뉴런은 해당 레이어의 유형에 따라 단일 값 또는 다차원 배열로 구성된 여러 값을 생성할 수 있습니다.
이러한 경우를 처리하는 방법에 대해서는 아래에 자세히 설명합니다.

For all neurons in fully-connected layers, we can directly compare their outputs against the neuron activation threshold as these neurons output a single scalar value.
By contrast, neurons in convolutional layers output multidimensional feature maps as each neuron outputs the result of applying a convolutional kernel across the input space [45].
For example, the first layer in Figure 3.1 illustrates the application of one convolutional kernel (of size 3×3) to the entire image (5×5) that produces a feature map of size 3×3 in the succeeding layer.
In such cases, we compute the average of the output feature map to convert the multidimensional output of a neuron into a scalar and compare it to the neuron activation threshold.

완전 연결 층의 모든 뉴런에 대해서는 이들의 출력이 단일 스칼라 값이므로 뉴런 활성화 임계값과 직접 비교할 수 있습니다.
반면에, 컨볼루션 층의 뉴런은 입력 공간에 컨볼루션 커널을 적용한 결과로 다차원 특성 맵을 출력합니다 [45].
예를 들어, 그림 3.1의 첫 번째 층은 전체 이미지(5×5)에 하나의 컨볼루션 커널(크기 3×3)을 적용하여 후속 층에서 크기 3×3의 특성 맵을 생성하는 과정을 보여줍니다.
이러한 경우, 뉴런의 다차원 출력을 스칼라 값으로 변환하기 위해 출력 특성 맵의 평균을 계산하고 이를 뉴런 활성화 임계값과 비교합니다.

For RNN/LSTM with loops, the intermediate neurons are unrolled to produce a sequence of outputs (Figure 3.2).
We treat each neuron in the unrolled layers as a separate individual neuron for the purpose of neuron coverage computation.

루프가 있는 RNN/LSTM의 경우 중간 뉴런들이 펼쳐져서 일련의 출력을 생성합니다 (그림 3.2).
뉴런 커버리지 계산을 위해, 펼쳐진 각 층의 뉴런을 별개의 개별 뉴런으로 취급합니다.

![Figure 3](/posts/paper/DeepTest/figure3.png)

### 3.2 Increasing Coverage with Synthetic Images
Generating arbitrary inputs that maximize neuron coverage may not be very useful if the inputs are not likely to appear in the real-world even if these inputs potentially demonstrate buggy behaviors.
Therefore, DeepTest focuses on generating realistic synthetic images by applying image transformations on seed images and mimic different real-world phenomena like camera lens distortions, object movements, different weather conditions, etc.
To this end, we investigate nine different realistic image transformations (changing brightness, changing contrast, translation, scaling, horizontal shearing, rotation, blurring, fog effect, and rain effect).
These transformations can be classified into three groups: linear, affine, and convolutional.

임의의 입력을 생성하여 뉴런 커버리지를 최대화하는 것은 해당 입력이 실제 세계에서 발생할 가능성이 낮다면 유용하지 않을 수 있습니다.
이러한 입력이 버그가 있는 동작을 보여줄 수 있다고 하더라도 말입니다.
따라서 DeepTest는 시드 이미지에 이미지 변환을 적용하여 카메라 렌즈 왜곡, 객체 이동, 다양한 날씨 조건 등 실제 세계의 다양한 현상을 모방함으로써 현실적인 합성 이미지를 생성하는 데 중점을 둡니다.
이를 위해, 우리는 아홉 가지 다른 현실적인 이미지 변환(밝기 변경, 대비 변경, 평행 이동, 스케일링, 수평 기울이기, 회전, 블러, 안개 효과, 비 효과)을 조사합니다.
이러한 변환은 선형, 아핀, 그리고 컨볼루션으로 세 그룹으로 분류될 수 있습니다.

Adjusting brightness and contrast are both linear transformations.
The brightness of an image depends on how large the pixel values are for that image.
An image’s brightness can be adjusted by adding/subtracting a constant parameter β to each pixel’s current value.
Contrast represents the difference in brightness between different pixels in an image.
One can adjust an image’s contrast by multiplying each pixel’s value by a constant parameter α .

밝기와 대비 조정은 모두 선형 변환입니다.
이미지의 밝기는 해당 이미지의 픽셀 값이 얼마나 큰지에 따라 달라집니다.
이미지를 밝게 하려면 각 픽셀의 현재 값에 일정한 매개변수 β를 더하거나 빼는 방식으로 조정할 수 있습니다.
대비는 이미지의 서로 다른 픽셀 간의 밝기 차이를 나타냅니다.
이미지를 대비를 조정하려면 각 픽셀 값을 일정한 매개변수 α로 곱하면 됩니다.

Translation, scaling, horizontal shearing, and rotation are all different types of affine transformations.
An affine transformation is a linear mapping between two images that preserves points, straight lines, and planes [5].
Affine transforms are often used in image processing to fix distortions resulting from camera angle variations.
In this paper, we leverage affine transformations for the inverse case, i.e., to simulate different real-world camera perspectives or movements of objects and check how robust the self-driving DNNs are to those changes.

평행 이동, 스케일링, 수평 기울이기, 회전은 모두 다양한 유형의 아핀 변환입니다.
아핀 변환은 두 이미지 간의 선형 매핑으로, 점, 직선, 평면을 보존합니다 [5].
아핀 변환은 이미지 처리에서 카메라 각도 변화로 인한 왜곡을 수정하는 데 자주 사용됩니다.
이 논문에서는 아핀 변환을 역으로 활용하여 실제 세계의 다양한 카메라 시점이나 객체의 움직임을 시뮬레이션하고, 자율 주행 DNN이 이러한 변화에 얼마나 견고한지를 확인합니다.

We list the transformation matrices for the four types of affine transformations (translation, scale, shear, and rotation) used in this paper in Table 2.

이 논문에서 사용된 네 가지 유형의 아핀 변환(평행 이동, 스케일, 기울이기, 회전)의 변환 행렬은 표 2에 나와 있습니다.

![Table 2](/posts/paper/DeepTest/table2.png)

Blurring and adding fog/rain effects are all convolutional transformations, i.e., they perform the convolution operation on the input pixels with different transform-specific kernels.
We use four different types of blurring filters: averaging, Gaussian, median, and bilateral [3].
We compose multiple filters provided by Adobe Photoshop on the input images to simulate realistic fog and rain effects [1, 2].

블러 처리와 안개/비 효과 추가는 모두 컨볼루션 변환입니다.
즉, 입력 픽셀에 대해 변환에 특화된 커널을 사용하여 컨볼루션 연산을 수행합니다.
우리는 네 가지 다른 유형의 블러 필터를 사용합니다: 평균, 가우시안, 미디언, 바이레터럴 [3].
현실적인 안개와 비 효과를 시뮬레이션하기 위해 Adobe Photoshop에서 제공하는 여러 필터를 입력 이미지에 적용합니다 [1, 2].

### 3.3 Combining Transformations to Increase Coverage

As the individual image transformations increase neuron coverage, one obvious question is whether they can be combined to further increase the neuron coverage.
Our results demonstrate that different image transformations tend to activate different neurons, i.e., they can be stacked together to further increase neuron coverage.
However, the state space of all possible combinations of different transformations is too large to explore exhaustively.
We provide a neuron-coverage-guided greedy search technique for efficiently finding combinations of image transformations that result in higher coverage (see Algorithm 1).

개별 이미지 변환이 뉴런 커버리지를 증가시키는 만큼, 이들을 결합하여 뉴런 커버리지를 더욱 증가시킬 수 있는지에 대한 질문이 자연스럽게 제기됩니다.
우리의 결과는 서로 다른 이미지 변환이 서로 다른 뉴런을 활성화하는 경향이 있음을 보여줍니다.
즉, 이들을 함께 쌓아 올려서 뉴런 커버리지를 더욱 증가시킬 수 있습니다.
그러나 서로 다른 변환의 가능한 모든 조합의 상태 공간은 철저히 탐색하기에는 너무 큽니다.
우리는 뉴런 커버리지에 기반한 탐욕적 검색 기법을 제공하여 더 높은 커버리지를 가져오는 이미지 변환 조합을 효율적으로 찾습니다(알고리즘 1 참조).

![Algorithm 1](/posts/paper/DeepTest/algorithm1.png)

The algorithm takes a set of seed images I, a list of transformations T and their corresponding parameters as input.
The key idea behind the algorithm is to keep track of the transformations that successfully increase neuron coverage for a given image and prioritize them while generating more synthetic images from the given image.
This process is repeated in a depth-first manner to all images.

알고리즘은 시드 이미지 집합 \( I \), 변환 목록 \( T \) 및 해당 매개변수를 입력으로 받습니다.
알고리즘의 핵심 아이디어는 주어진 이미지에 대해 뉴런 커버리지를 성공적으로 증가시키는 변환을 추적하고, 이를 우선시하여 주어진 이미지에서 더 많은 합성 이미지를 생성하는 것입니다.
이 과정은 깊이 우선 방식으로 모든 이미지에 반복됩니다.

### 3.4 Creating a Test Oracle with Metamorphic Relations
avoid this issue, we leverage metamorphic relations [33] between the car behaviors across different synthetic images.
The key insight is that even though it is hard to specify the correct behavior of a self-driving car for every transformed image, one can define relationships between the car’s behaviors across certain types of transformations.
For example, the autonomous car’s steering angle should not change significantly for the same image under any lighting/weather conditions, blurring, or any affine transformations with small parameter values.
Thus, if a DNN model infers a steering angle θo for an input seed image Io and a steering angle θt for a new synthetic image It , which is generated by applying the transformation t on Io , one may define a simple metamorphic relation where θo and θt are identical.

자율 주행 차량과 같은 복잡한 DNN 기반 시스템을 테스트할 때의 주요 도전 과제 중 하나는 시스템의 동작을 검사할 수 있는 시스템 사양을 수동으로 작성하는 것입니다.
이러한 시스템에 대한 자세한 사양을 작성하는 것은 본질적으로 인간 운전자의 논리를 재현하는 것을 포함하기 때문에 어렵습니다.
이 문제를 피하기 위해 우리는 다양한 합성 이미지에서 차량 동작 간의 변형 관계 [33]를 활용합니다.
핵심 통찰은 변형된 모든 이미지에 대해 자율 주행 차량의 정확한 동작을 명시하기는 어렵지만 특정 유형의 변환 간의 차량 동작 간의 관계를 정의할 수 있다는 것입니다.
예를 들어, 자율 주행 차량의 조향 각도는 조명/날씨 조건, 블러 처리 또는 작은 매개변수 값을 가진 아핀 변환에서 동일한 이미지에 대해 크게 변하지 않아야 합니다.
따라서, DNN 모델이 입력 시드 이미지 \( I_o \)에 대해 조향 각도 \( \theta_o \)를 추론하고, \( I_o \)에 변환 \( t \)를 적용하여 생성된 새로운 합성 이미지 \( I_t \)에 대해 조향 각도 \( \theta_t \)를 추론하면, \( \theta_o \)와 \( \theta_t \)가 동일한 간단한 변형 관계를 정의할 수 있습니다.

To minimize false positives, we relax our metamorphic relations and allow variations within the error ranges of the original input images.
We observe that the set of outputs predicted by a DNN model for the original images, say { \( θ_{o1} \), \(θ_{o2}\), ...., \(θ_{on}\) }, in practice, result in a small but non-trivial number of errors w.r.t. their respective manual labels \( \{\hat{\theta}_1, \hat{\theta}_2, \ldots, \hat{\theta}_n\} \).
Such errors are usually measured using MeanSquaredError(MSE),where \(MSE_{orig} = \frac{1}{n} \sum_{i=1}^{n} (\hat{\theta}_i - \theta_{oi})^2 \).
Leveraging this property, we redefine a new metamorphic relation as:

\[ (\hat{\theta}_i - \theta_{ti})^2 \leq \lambda \cdot MSE_{orig}           \qquad\qquad(2)\]

거짓 양성을 최소화하기 위해, 우리는 변형 관계를 완화하고 원래 입력 이미지의 오류 범위 내에서 변화를 허용합니다.
DNN 모델이 원래 이미지에 대해 예측한 출력 집합 { \( θ_{o1} \), \(θ_{o2}\), ...., \(θ_{on}\) } 이 실제로 해당 수동 레이블\( \{\hat{\theta}_1, \hat{\theta}_2, \ldots, \hat{\theta}_n\} \)과 비교할 때 작지만 비트리비얼한 수의 오류를 발생시키는 것을 관찰합니다.
이러한 오류는 일반적으로 평균 제곱 오차(MSE)를 사용하여 측정되며, \(MSE_{orig} = \frac{1}{n} \sum_{i=1}^{n} (\hat{\theta}_i - \theta_{oi})^2 \) 으로 계산됩니다.

이 속성을 활용하여, 우리는 새로운 변형 관계를 다음과 같이 재정의합니다:

\[ (\hat{\theta}_i - \theta_{ti})^2 \leq \lambda \cdot MSE_{orig}           \qquad\qquad(2)\]

The above equation assumes that the errors produced by a model for the transformed images as input should be within a range of λ times the MSE produced by the original image set.
Here, λ is a configurable parameter that allows us to strike a balance between the false positives and false negatives.

위의 식은 변환된 이미지를 입력으로 사용할 때 모델이 생성하는 오류가 원래 이미지 세트에서 생성된 MSE의 λ 배 범위 내에 있어야 함을 가정합니다.
여기서 λ는 거짓 양성과 거짓 음성 사이의 균형을 맞추기 위해 조정할 수 있는 매개변수입니다.

## 4. IMPLEMENTATION
Autonomous driving DNNs. We evaluate our techniques on three DNN models that won top positions in the Udacity self-driving challenge [15].
We choose these three models as their implementations are based on the Keras framework [34] that our current prototype of DeepTest supports. The details of the DNN models and dataset are summarized in Table 3.

자율 주행 딥러닝 네트워크(DNN). 우리는 Udacity 자율 주행 챌린지에서 상위 순위를 차지한 세 가지 DNN 모델을 대상으로 우리의 기술을 평가합니다 [15].
이 세 가지 모델을 선택한 이유는 현재 DeepTest 프로토타입이 지원하는 Keras 프레임워크 [34]를 기반으로 구현되었기 때문입니다. DNN 모델과 데이터셋의 세부 사항은 표 3에 요약되어 있습니다.

![Table 3](/posts/paper/DeepTest/table3.png)

Image transformations.
In the experiments for RQ2 and RQ3, we leverage seven different types of simple image transformations: translation, scaling, horizontal shearing, rotation, contrast adjustment, brightness adjustment, and blurring.
We use OpenCV to implement these transformations [7].
For RQ2 and RQ3 described in Section 5, we use 10 parameters for each transformation as shown in Table 4.

이미지 변환.
RQ2 및 RQ3 실험에서는 일곱 가지 간단한 이미지 변환을 활용합니다: 평행 이동, 스케일링, 수평 기울이기, 회전, 대비 조정, 밝기 조정 및 블러링.
이러한 변환은 OpenCV를 사용하여 구현합니다 [7].
섹션 5에 설명된 RQ2 및 RQ3의 경우, 각 변환에 대해 표 4에 나타난 10개의 매개변수를 사용합니다.

![Table 4](/posts/paper/DeepTest/table4.png)
