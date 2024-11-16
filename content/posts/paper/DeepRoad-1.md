+++
title = 'DeepRoad 1'
date = 2024-10-27T13:30:01+09:00
draft = true
+++

이 글은 DeepRoad: GAN-based Metamorphic Autonomous Driving System Testing (https://arxiv.org/abs/1802.02295)을 번역 및 요약한 글입니다.

## Abstract
While Deep Neural Networks (DNNs) have established the fundamentals of DNN-based autonomous driving systems, they may exhibit erroneous behaviors and cause fatal accidents.
To resolve the safety issues of autonomous driving systems, a recent set of testing techniques have been designed to automatically generate test cases, e.g., new input images transformed from the original ones.
Unfortunately, many such generated input images often render inferior authenticity, lacking accurate semantic information of the driving scenes and hence compromising the resulting efficacy and reliability.

딥 신경망(DNN)은 DNN 기반 자율주행 시스템의 기초를 확립했지만, 잘못된 동작을 보일 수 있으며 치명적인 사고를 유발할 수 있습니다.
자율주행 시스템의 안전 문제를 해결하기 위해 최근 자동으로 테스트 케이스를 생성하는 여러 테스트 기법들이 설계되었습니다.
예를 들어, 원본 이미지를 변형하여 새로운 입력 이미지를 생성하는 방법입니다.
그러나 이렇게 생성된 입력 이미지 중 많은 경우는 진정성이 떨어지며, 운전 장면에 대한 정확한 의미 정보를 부족하게 만들어 결과적인 효과성과 신뢰성을 저해할 수 있습니다.

In this paper, we propose DeepRoad, an unsupervised framework to automatically generate large amounts of accurate driving scenes to test the consistency of DNN-based autonomous driving systems across different scenes.
In particular, DeepRoad delivers driving scenes with various weather conditions (including those with rather extreme conditions) by applying the Generative Adversarial Networks (GANs) along with the corresponding real-world weather scenes.
Moreover, we have implemented DeepRoad to test three well-recognized DNN-based autonomous driving systems.
Experimental results demonstrate that DeepRoad can detect thousands of behavioral inconsistencies for these systems.

이 논문에서는 DNN 기반 자율주행 시스템이 다양한 주행 장면에서 일관성을 유지하는지를 테스트하기 위해, 대량의 정확한 주행 장면을 자동으로 생성하는 비지도 학습 기반의 프레임워크인 DeepRoad를 제안합니다.
특히 DeepRoad는 생성적 적대 신경망(GAN)을 사용하여 실제 기상 조건에 맞춘 다양한 날씨 조건(극한의 날씨 조건을 포함한)을 가진 주행 장면을 제공합니다.
또한, DeepRoad를 구현하여 세 가지로 잘 알려진 DNN 기반 자율주행 시스템을 테스트하였으며, 실험 결과 DeepRoad가 이러한 시스템들의 수천 개의 동작 불일치를 감지할 수 있음을 입증했습니다.

## 1. Introduction
Typically, autonomous driving refers to utilizing sensors (cameras, LiDAR, RADAR, GPS, etc) to automatically control vehicles without human intervention.
The recent advances in Deep Neural Networks (DNNs) enables autonomous driving systems to adapt their driving behaviors according to the dynamic environments.
In particular, an end-to-end supervised learning framework is made possible to train a DNN for predicting driving behaviors (e.g., steering angles) by inputing driving scenes (e.g., images), using the ⟨driving scene, driving behavior⟩pairs as the training data.
For instance, DAVE-2 [13], released by NVIDIA in 2016, can predict steering angles based on only driving scenes captured by a single front-centered camera of autonomous cars.

일반적으로 자율주행은 센서(카메라, LiDAR, RADAR, GPS 등)를 이용하여 사람의 개입 없이 차량을 자동으로 제어하는 것을 의미합니다.
최근 딥러닝 신경망(DNN)의 발전은 자율주행 시스템이 동적인 환경에 맞춰 운전 행동을 적응시킬 수 있게 만들었습니다.
특히, 주행 장면(예: 이미지)과 주행 행동(예: 조향각) 쌍을 학습 데이터로 사용하여 DNN을 훈련시키는 종단 간 지도 학습 프레임워크가 가능해졌습니다.
예를 들어, NVIDIA가 2016년에 출시한 DAVE-2 [13]는 자율주행 차량의 전면 중앙에 위치한 단일 카메라로 촬영된 주행 장면만을 기반으로 조향각을 예측할 수 있습니다.

Recent testing techniques [23; 28] demonstrate that adding error-inducing inputs to the training datasets can help improve the reliability and accuracy of existing autonomous driving models.
For example, the most recent DeepTest work [28] designs systematic ways to automatically generate test cases, seeking to mimic real-world driving scenes.
Its main methodology is to transform training driving scenes by applying simple affine transformations and various effect filters such as blurring/fog/rain to the original image data, and then check if autonomous driving systems perform consistently among the original and transformed scenes.
With large amounts of original and transformed driving scenes, DeepTest can detect various erroneous inconsistent driving behaviors for some well-performed open-source autonomous driving models, in a cheap and quick manner.

최근 테스트 기법들 [23; 28]은 오류를 유발하는 입력 데이터를 학습 데이터셋에 추가하는 것이 기존 자율주행 모델의 신뢰성과 정확성을 향상시키는 데 도움이 된다는 것을 입증했습니다.
예를 들어, 가장 최신의 DeepTest 연구 [28]는 실제 주행 장면을 모방하려는 목적으로 테스트 케이스를 자동으로 생성하는 체계적인 방법을 설계했습니다.
이 방법론의 핵심은 원본 이미지 데이터에 간단한 아핀 변환과 블러링, 안개, 비와 같은 다양한 효과 필터를 적용하여 주행 장면을 변형한 후, 자율주행 시스템이 원본 장면과 변형된 장면에서 일관되게 동작하는지 확인하는 것입니다.
DeepTest는 대량의 원본 및 변형된 주행 장면을 통해 일부 잘 작동하는 오픈소스 자율주행 모델에서 다양한 오류와 불일치된 주행 행동을 저렴하고 빠르게 감지할 수 있습니다.

However, it is observed that the methodologies applied in DeepTest to generate test cases cannot accurately reflect the real-world driving scenes.
Specifically, real-world driving scenes can rarely be affine-transformed and captured by the cameras of autonomous driving systems; the blurring/ fog/rain effects made by simply simulating the corresponding effects also appear to be unrealistic which compromises the efficacy and reliability of DeepTest.
For instance, Figure 1a shows the fog effect transformation applied in DeepTest.
It can be observed that Figure 1a is distorted.
In particular, it appears to be synthesized by simply dimming the original image and mixing it with the scrambled “smoke” effect.
In addition, Figure 1b shows the rain effect transformation applied in DeepTest.
Similarly, DeepTest simply simulates rain by adding a group of lines over the original image.
This rain effect transformation is even more distorted because usually when it rains, the camera tends to be wet and the image is highly possible to be blurred.
The fact that few test cases in DeepTest appear authentic to reflect the real-world driving scenes makes it difficult to determine whether the erroneous driving behaviors are caused by the flaws of the DNN-based models or the inadequacy of the testing technique itself.
Furthermore, there are many potential driving scenes that can not be easily simulated with simple image processing.
For instance, the snowy road condition requires different sophisticated transformations for the road and the roadside objects (such as trees).

그러나 DeepTest에서 테스트 케이스를 생성하는 데 사용된 방법론은 실제 주행 장면을 정확하게 반영하지 못한다는 것이 관찰되었습니다.
구체적으로, 실제 주행 장면은 자율주행 시스템의 카메라에 의해 거의 아핀 변환될 수 없으며, 단순히 대응 효과를 시뮬레이션한 블러링, 안개, 비 효과는 비현실적으로 보여 DeepTest의 효과성과 신뢰성을 저하시킵니다.
예를 들어, 그림 1a는 DeepTest에서 적용된 안개 효과 변환을 보여줍니다.
그림 1a에서 왜곡된 점을 확인할 수 있는데, 이는 단순히 원본 이미지를 어둡게 하고 혼란스러운 ‘연기’ 효과를 혼합하여 합성된 것으로 보입니다.
또한, 그림 1b는 DeepTest에서 적용된 비 효과 변환을 보여줍니다.
이와 유사하게, DeepTest는 단순히 원본 이미지 위에 여러 줄을 추가하여 비를 시뮬레이션합니다.
이 비 효과 변환은 더욱 왜곡되어 보이는데, 비가 올 때는 보통 카메라가 젖고 이미지가 흐려질 가능성이 높기 때문입니다.
DeepTest에서 생성된 테스트 케이스 중 실제 주행 장면을 반영하는 것이 거의 없다는 사실은 DNN 기반 모델의 결함으로 인해 잘못된 주행 행동이 발생한 것인지, 또는 테스트 기법 자체의 부적절성 때문인지 판단하기 어렵게 만듭니다.
더욱이, 단순한 이미지 처리로 쉽게 시뮬레이션할 수 없는 잠재적인 주행 장면들이 많이 존재합니다.
예를 들어, 눈 덮인 도로 상황에서는 도로와 도로변의 나무와 같은 객체에 대해 서로 다른 복잡한 변환이 필요합니다.

In order to automatically synthesize large amounts of authentic driving scenes for testing DNN-based autonomous driving systems, in this paper, we propose an unsupervised framework, namely DeepRoad, that employs a Generative Adversarial Network (GAN)-based technique [15] to deliver authentic driving scenes with various weather conditions which are rather difficult to be collected manually.
Specifically, DeepRoad develops a metamorphic testing module for DNN-based autonomous systems, where the metamorphic relations are defined such that no matter how the driving scenes are synthesized to cope with different weather conditions, the driving behaviors are expected to be consistent with those under the corresponding original driving scenes.
At this point, DeepRoad enables us to test the accuracy and reliability of existing DNN-based autonomous driving systems under different extreme weather scenarios, including heavy snow and hard rain, and can greatly complement the existing autonomous driving system testing approaches (such as DeepTest).
For instance, Figure 2 presents the snowy and rainy scenes generated by DeepRoad (from fine scenes), which can hardly be distinguished from genuine ones and cannot be generated using simple transformations.

DNN 기반 자율주행 시스템을 테스트하기 위해 대량의 실제적인 주행 장면을 자동으로 합성하기 위해, 본 논문에서는 GAN(생성적 적대 신경망) 기반 기법 [15]을 사용하는 비지도 학습 프레임워크인 DeepRoad를 제안합니다.
DeepRoad는 수작업으로 수집하기 어려운 다양한 기상 조건의 실제적인 주행 장면을 제공합니다.
구체적으로, DeepRoad는 DNN 기반 자율주행 시스템을 위한 변환 테스트 모듈을 개발했으며, 이 모듈에서 변환 관계는 주행 장면이 다양한 날씨 조건에 맞게 합성되더라도, 해당 주행 장면에 대응하는 원본 주행 장면에서 기대되는 주행 행동과 일관되어야 한다는 원칙으로 정의됩니다.
이 시점에서 DeepRoad는 기존의 DNN 기반 자율주행 시스템이 폭설이나 폭우 같은 극한 날씨 시나리오에서 얼마나 정확하고 신뢰성 있게 동작하는지를 테스트할 수 있게 하며, 기존의 자율주행 시스템 테스트 접근 방식(예: DeepTest)을 크게 보완할 수 있습니다.
예를 들어, 그림 2는 DeepRoad가 생성한 눈 덮인 장면과 비 내리는 장면을 보여주는데, 이 장면들은 실제와 거의 구분이 불가능하며 단순한 변환으로는 생성할 수 없습니다.

Although our DeepRoad approach is general, and can be used to simulate various weather conditions, in this work, we first synthesize driving scenes under heavy snow and hard rain.
In particular, based on the GAN technique, we collect images with the two extreme weather conditions from Youtube videos to transform real-world driving scenes and deliver them with the corresponding weather conditions.
Subsequently, these synthesized driving scenes are used to test three well-recognized Udacity DNN-based autonomous driving systems [8].
The experimental results reveal that DeepRoad can effectively detect thousands of behavioral inconsistencies of different levels for these systems, indicating a promising future for testing autonomous driving systems via GAN-based road scene transformation.

비록 DeepRoad 접근 방식은 일반적으로 다양한 날씨 조건을 시뮬레이션할 수 있지만, 이 연구에서는 먼저 폭설과 폭우 상황에서 주행 장면을 합성했습니다.
구체적으로, GAN 기법을 기반으로 유튜브 비디오에서 두 가지 극한 기상 조건을 가진 이미지를 수집하여 실제 주행 장면을 해당 기상 조건으로 변환했습니다.
그런 다음, 이렇게 합성된 주행 장면을 사용하여 세 가지로 잘 알려진 Udacity DNN 기반 자율주행 시스템 [8]을 테스트했습니다.
실험 결과, DeepRoad는 이러한 시스템에서 서로 다른 수준의 수천 가지 동작 불일치를 효과적으로 감지할 수 있음을 보여주었으며, 이는 GAN 기반 도로 장면 변환을 통해 자율주행 시스템을 테스트하는 데 있어 유망한 가능성을 시사합니다.

The contributions of this paper are as follows.
* **Idea.** We propose the first GAN-based metamorphic testing approach, namely DeepRoad, to generate authentic driving scenes with various weather conditions for detecting autonomous driving system inconsistencies.
* **Implementation.** We implement the proposed approach based on Pytorch and Python to synthesize driving scenes under heavy snow and hard rain based on training data collected from Youtube videos.
* **Evaluation.** We use DeepRoad to test well-recognized DNN-based autonomous driving models and successfully detect thousands of inconsistent driving behaviors for them.

본 논문의 기여는 다음과 같습니다.
* 아이디어: 자율주행 시스템의 불일치를 감지하기 위해 다양한 날씨 조건에서 실제적인 주행 장면을 생성하는 첫 번째 GAN 기반 변환 테스트 접근 방식인 DeepRoad를 제안합니다.
* 구현: 제안한 접근 방식을 Pytorch와 Python을 기반으로 구현하여, 유튜브 비디오에서 수집한 데이터를 바탕으로 폭설과 폭우 조건에서 주행 장면을 합성합니다.
* 평가: DeepRoad를 사용하여 잘 알려진 DNN 기반 자율주행 모델을 테스트하였으며, 수천 가지의 불일치된 주행 행동을 성공적으로 감지했습니다.

## 2. Background
Nowadays, DNN-based autonomous driving systems have been rapidly evolving [24; 13].
For example, many major car manufacturers (including Tesla, GM, Volvo, Ford, BMV, Honda, and Daimler) and IT companies (including Waymo/Google, Uber, and Baidu) are working on building and testing various DNN-based autonomous driving systems.
In DNN-based autonomous driving systems, the neural network models take the driving scenes captured by the sensors (LiDar, Radar, cameras, etc.) as input and output the driving behaviors (e.g., steering and braking control decisions).
In this work, we mainly focus on DNN-based autonomous driving systems with camera inputs and steering angle outputs.
To date, feed-forward Convolutional Neural Network (CNN) [17] and Recurrent Neural Network (RNN) [25] are the most widely used DNNs for autonomous driving systems.
Figure 3 shows an example CNN-based autonomous driving system.
Shown in the figure, the system consists of an input (the camera image inputs) and an output layer (the steering angle), as well as multiple hidden layers.
The use of convolution hidden layers allows weight sharing across multiple connections and can greatly save the training efforts; furthermore, its local-to-global recognition process actually coincides with the manual object recognition process.

오늘날 DNN 기반 자율주행 시스템은 빠르게 발전하고 있습니다 [24; 13].
예를 들어, Tesla, GM, Volvo, Ford, BMV, Honda, Daimler와 같은 주요 자동차 제조업체들과 Waymo/Google, Uber, Baidu와 같은 IT 기업들이 다양한 DNN 기반 자율주행 시스템을 구축하고 테스트하는 작업을 진행 중입니다.
DNN 기반 자율주행 시스템에서 신경망 모델은 센서(LiDar, Radar, 카메라 등)로 캡처된 주행 장면을 입력으로 받아, 주행 행동(예: 조향 및 제동 제어 결정)을 출력합니다.
이 연구에서는 주로 카메라 입력과 조향각 출력을 사용하는 DNN 기반 자율주행 시스템에 초점을 맞추고 있습니다.
현재까지 피드포워드 합성곱 신경망(CNN) [17]과 순환 신경망(RNN) [25]이 자율주행 시스템에 가장 널리 사용되는 DNN입니다.
그림 3은 CNN 기반 자율주행 시스템의 예를 보여줍니다.
그림에서 볼 수 있듯이, 이 시스템은 입력층(카메라 이미지 입력)과 출력층(조향각), 그리고 여러 개의 은닉층으로 구성됩니다.
합성곱 은닉층의 사용은 여러 연결 간 가중치를 공유할 수 있게 하여 훈련 노력을 크게 절감할 수 있으며, 국소적인 인식에서 전역적인 인식으로 이어지는 과정이 실제로 수동 객체 인식 과정과 일치합니다.

DNN-based autonomous driving systems are essentially software systems, which are error-prone and can lead to tragedies.
For example, a Tesla Model S plowed into a fire truck at 65 mph while using Autopilot system [6].
To ensure the quality of software systems, many software testing techniques have been proposed in the literature [12; 21], where typically, a set of specific test cases are generated to test if the software programs perform as expected.
The process of determining whether the software performs as expected upon the given test inputs is known as the test oracle problem [12].
Despite the abundance of traditional software testing techniques, they cannot be directly applied for DNN-based systems since the logics of DNN-based softwares are learned from data with minimal human interference (like a blackbox) while the logics of traditional software programs are manually created.

DNN 기반 자율주행 시스템은 본질적으로 소프트웨어 시스템이며, 오류가 발생하기 쉬워 비극적인 사고를 초래할 수 있습니다.
예를 들어, Tesla Model S가 Autopilot 시스템을 사용 중에 시속 65마일로 소방차에 충돌한 사례가 있습니다 [6].
소프트웨어 시스템의 품질을 보장하기 위해 많은 소프트웨어 테스트 기법들이 문헌에서 제안되었습니다 [12; 21].
일반적으로 특정 테스트 케이스 세트를 생성하여 소프트웨어 프로그램이 예상대로 작동하는지 테스트합니다.
주어진 테스트 입력에 대해 소프트웨어가 예상대로 작동하는지 여부를 결정하는 과정은 테스트 오라클 문제라고 불립니다 [12].
전통적인 소프트웨어 테스트 기법이 풍부하게 존재하지만, DNN 기반 시스템에는 직접적으로 적용하기 어렵습니다.
그 이유는 DNN 기반 소프트웨어의 논리는 최소한의 인간 개입으로 데이터에서 학습되는 반면, 전통적인 소프트웨어 프로그램의 논리는 수동으로 만들어지기 때문입니다.
DNN 기반 소프트웨어는 마치 블랙박스처럼 동작하기 때문에 테스트 접근법에 차이가 있습니다.

Recently, researchers have proposed various techniques to test DNN-based autonomous driving systems, e.g., DeepXplore [23] and DeepTest [28].
DeepXplore aims to automatically generate input images that can differentiate the behaviors of different DNN-based systems.
However, it cannot be directly used to test one DNN-based autonomous driving system in isolation.
The more recent DeepTest work utilizes some simple affine transformations and blurring/fog/rain effect filters to synthesize test cases to detect the inconsistent driving behaviors derived from the original and synthesized images.
Although DeepTest can be applied to test any DNN-based driving system, the synthesized images may be unrealistic, and it cannot simulate complex weather conditions (e.g., snowy scenes).

최근 연구자들은 DNN 기반 자율주행 시스템을 테스트하기 위한 다양한 기법들을 제안했습니다.
예를 들어, DeepXplore [23]와 DeepTest [28]가 있습니다.
DeepXplore는 서로 다른 DNN 기반 시스템의 동작을 구분할 수 있는 입력 이미지를 자동으로 생성하는 것을 목표로 하지만, 단일 DNN 기반 자율주행 시스템을 단독으로 테스트하는 데는 직접적으로 사용될 수 없습니다.
더 최근의 연구인 DeepTest는 간단한 아핀 변환과 블러링, 안개, 비 효과 필터를 사용하여 테스트 케이스를 합성하고, 원본 이미지와 합성된 이미지에서 발생하는 불일치된 주행 행동을 감지하는 방법을 제안합니다.
DeepTest는 어떤 DNN 기반 자율주행 시스템에도 적용할 수 있지만, 합성된 이미지가 비현실적일 수 있으며 복잡한 기상 조건(예: 눈 덮인 장면)을 시뮬레이션할 수 없습니다.

## 3. Approach
### 3.1 Metamorphic DNN Testing

## 4. Experimental
### 4.1 Data
We use a real-world dataset released by Udacity [9] as a baseline to check the inconsistency of autonomous driving systems.
From the dataset, we select two episodes of high-way driving video where obvious changes of lighting and road conditions can be observed among frames.
To train our UNIT model, we also collect images of extreme scenarios from Youtube.
In the experiments, we select snow and hard rain, two extreme weather conditions to transform real-world driving images.
To make the variance of collected images relatively large, we only search for videos which is longer than 20mins.
In the scenario of hard rain, the video would record wipers swiping windows, and in the data preprocessing phase, we manually check and filter out those images.
Note that all images used in the experiments are cropped and resized to 240 ×320, and we have performed down-sampling for Youtube videos to skip consecutive frames with close contents.
The detailed information is present in Table 1.

우리는 자율주행 시스템의 일관성 검사를 위한 기준으로 Udacity [9]에서 공개한 실제 데이터셋을 사용합니다.
이 데이터셋에서 프레임 간 명확한 조명 및 도로 조건 변화가 관찰되는 고속도로 주행 비디오 두 에피소드를 선택했습니다.
또한, UNIT 모델을 학습시키기 위해 유튜브에서 극한 상황의 이미지를 수집했습니다.
실험에서는 눈과 폭우, 두 가지 극한 날씨 조건을 선택하여 실제 주행 이미지를 변환했습니다.
수집된 이미지의 분산을 상대적으로 크게 하기 위해 20분 이상 길이의 비디오만을 검색했습니다.
폭우 시나리오에서는 와이퍼가 창문을 닦는 장면이 기록되며, 데이터 전처리 단계에서 수동으로 확인하여 해당 이미지를 필터링했습니다.
실험에 사용된 모든 이미지는 240 × 320으로 잘리고 크기 조정이 이루어졌으며, 유튜브 비디오의 경우 연속된 프레임에서 유사한 내용이 담긴 경우를 건너뛰기 위해 다운샘플링을 수행했습니다.
자세한 정보는 표 1에 나와 있습니다.

### 4.2 Models
We evaluate our framework on three DNN-based autonomous driving models which are released by Udacity [9]: Autumn [2], Chauffeur [3], and Rwightman [4].
We choose these three models as their pre-trained model are public and can be evaluated directly on the synthesized datasets.
To be specific, the model details of Rwightman are not publicly released, however, just like black-box testing, our approach aims to detect the inconsistencies of the model instead of localizing software faults, hence, we still use Rwightman for the evaluation.
Autumn.
Autumn is composed by a data preprocessing module and a CNN.
Specifically, Autumn first computes the optical flow of input images and input them to a CNN to predict the steering angles.
The architecture of Autumn is: three 5x5 conv layers with stride 2 pluses two 3x3 conv layers and followed by five fully-connected layers with dropout.
The model is implemented by OpenCV, Tensorflow and Keras.
Chauffeur.
Chauffeur consists of one CNN and one RNN module with LSTM.
The work flow is that CNN firstly extracts the features of input images and then utilizes RNN to predict the steering angle from previous 100 consecutive images.
This model is also implemented by Tensorflow and Keras.

우리는 Udacity [9]에서 공개한 DNN 기반 자율주행 모델 세 가지, Autumn [2], Chauffeur [3], Rwightman [4]을 사용하여 프레임워크를 평가합니다.
이 세 모델을 선택한 이유는 사전 학습된 모델이 공개되어 있어 합성 데이터셋에서 직접 평가할 수 있기 때문입니다.
Rwightman 모델의 세부 사항은 공개되지 않았지만, 블랙박스 테스트와 같이 모델의 불일치를 감지하는 것이 목표이지 소프트웨어 결함을 찾는 것이 목표가 아니므로 Rwightman을 평가에 포함했습니다.

**Autumn** Autumn은 데이터 전처리 모듈과 CNN으로 구성됩니다.
구체적으로, Autumn은 입력 이미지의 광류(optical flow)를 계산한 후 이를 CNN에 입력하여 조향각을 예측합니다.
Autumn의 구조는: 세 개의 5x5 컨볼루션 레이어(스트라이드 2), 두 개의 3x3 컨볼루션 레이어, 그 뒤에 다섯 개의 드롭아웃이 있는 완전 연결 레이어로 이루어져 있습니다.
이 모델은 OpenCV, Tensorflow 및 Keras로 구현되었습니다.

**Chauffeur** Chauffeur는 CNN과 LSTM이 포함된 RNN 모듈로 구성됩니다.
작동 방식은 CNN이 입력 이미지의 특징을 먼저 추출한 후, RNN이 이전 100개의 연속된 이미지에서 조향각을 예측하는 것입니다.
이 모델 또한 Tensorflow와 Keras로 구현되었습니다.

### 4.3 Metric

### 4.4 Results
**Quality of generated images** We first present several Youtube frames as ground truth in Figure 6 to help readers check the quality of generated images.
In Figure 7, we list real and GAN-generated images pairs, where the two rows present the transformation of Udacity dataset to snowy and rainy scenes, respectively, and the odd and even columns present original and GAN-generated images, respectively.
Qualitatively, the GAN-generated images are visually similar to the images collected from Youtube and they also can keep the major semantic information (such as the shape of tree and road) of the original images.
Interestingly, in the first snowy image in Figure 7, the sky is relatively dark and GAN can successfully render the snow texture and the light in front of the car.
In the second column, the sharpness of rainy images are relatively low and this is consistent to the real scene showed in Figure 6.
Our results are consistent with the original UNIT work [19], and further demonstrate the effectiveness of UNIT for image transformation.

생성된 이미지의 품질에 대해 먼저 독자들이 생성된 이미지의 품질을 확인할 수 있도록 유튜브에서 추출한 여러 프레임을 그림 6에 “ground truth”로 제시합니다.
그림 7에서는 실제 이미지와 GAN으로 생성된 이미지의 쌍을 나열하며, 두 행은 Udacity 데이터셋을 각각 눈 내린 장면과 비 내리는 장면으로 변환한 예를, 홀수 및 짝수 열은 각각 원본 이미지와 GAN으로 생성된 이미지를 나타냅니다.
정성적으로 볼 때, GAN으로 생성된 이미지는 유튜브에서 수집한 이미지와 시각적으로 유사하며 원본 이미지의 주요 의미 정보(예: 나무와 도로의 형태)를 유지합니다.
흥미롭게도, 그림 7의 첫 번째 눈 내린 이미지에서 하늘이 비교적 어둡고, GAN이 눈의 질감과 차량 전방의 빛을 성공적으로 렌더링한 것을 확인할 수 있습니다.
두 번째 열에서 비 내리는 이미지의 선명도는 비교적 낮으며 이는 그림 6에 나타난 실제 장면과 일치합니다.
우리의 결과는 원래 UNIT 연구 [19]와 일치하며, 이미지 변환에 대한 UNIT의 효과를 추가로 입증합니다.

Inconsistency of autonomous driving models We further present examples for the detected inconsistent autonomous driving behaviors in Figure 8.
In the figure, each row shows the scenes of snow and rain, respectively.
In each sub-figure, the blue caption indicates the model name, while the red and green captions indicate the predicted steering angles on the real and synthesized images, respectively.
The curves visualize the predictions which help check the differences.
From the figure we can observe that model Autumn (the first two columns) has the highest inconsistency number on both scenes; in contrast, model Rwightman (the last two columns) is the most stable model under different scenes.
This figure shows that DeepRoad is able to find inconsis- tent behaviors under different road scenes for real-world autonomous driving systems.
For example, a model like Autumn or Chauffeur [1] (they are both ranked higher than Rwightman in the Udacity challenge) may work per- fectly in a fine day but can crash into the curbside (or even worse, the oncoming cars) in a rainy or snowy day (shown in Figure 8).

자율주행 모델의 불일치에 대해 그림 8에 감지된 자율주행 동작의 불일치 예시를 추가로 제시합니다.
그림에서 각 행은 각각 눈과 비의 장면을 보여줍니다.
각 소그림에서 파란색 캡션은 모델 이름을 나타내고, 빨간색과 초록색 캡션은 각각 실제 이미지와 합성 이미지에서 예측된 조향각을 나타냅니다.
곡선은 예측 결과를 시각화하여 차이를 확인할 수 있도록 돕습니다.
그림에서 알 수 있듯이 모델 Autumn(첫 번째와 두 번째 열)은 두 장면 모두에서 가장 높은 불일치 수치를 나타내며, 반면 모델 Rwightman(마지막 두 열)은 다양한 장면에서도 가장 안정적인 모델입니다.
이 그림은 DeepRoad가 실제 자율주행 시스템의 다양한 도로 장면에서 불일치한 동작을 찾아낼 수 있음을 보여줍니다.
예를 들어, Autumn이나 Chauffeur [1] 모델은 Udacity 챌린지에서 Rwightman보다 높은 순위를 차지했음에도 맑은 날에는 완벽하게 작동할 수 있지만, 비나 눈이 오는 날에는 도로 연석에 충돌하거나(심지어는 반대 차선 차량과 충돌할 위험도 있음) 문제가 발생할 수 있습니다(그림 8 참조).

Table 2 presents the detailed number of detected inconsis- tent behaviors under different weather conditions and error bounds for each studied autonomous driving model on the Udacity dataset.
For example, when using the error bound of 10◦and the rainy scenes, DeepRoad detects 5279, 710, and 656 inconsistent behaviors for Autumn, Chauffeur, and Rwightman, respectively.
From the table we can observe that the inconsistency number of Autumn is the highest un- der both weather conditions.
We think one potential reason is that Autumn is purely based on CNN, and does not utilized prior history information (e.g., via RNN), and thus may not always perform well in all road scenes.
On the other hand, Rwightman performs the most consistently than the other two models under all error bounds.
This result presents a very interesting phenomenon – DeepRoad can not only de- tect thousands of inconsistent behaviors of the studied au- tonomous driving systems, but can also measure different au- tonomous systems in terms of their robustness.
For example, with the original Udacity dataset, it is hard to find the limita- tions of autonomous driving systems like Autumn.

표 2는 Udacity 데이터셋을 사용하여 분석한 자율주행 모델별로 다양한 날씨 조건과 오류 한도에서 감지된 불일치 동작의 상세한 수치를 제시합니다.
예를 들어, 오류 한도를 10°로 설정하고 비 오는 장면을 분석했을 때, DeepRoad는 Autumn, Chauffeur, Rwightman 모델에서 각각 5279, 710, 656건의 불일치 동작을 감지했습니다.
표에서 확인할 수 있듯이, Autumn의 불일치 수치는 두 가지 날씨 조건 모두에서 가장 높습니다.
이는 Autumn 모델이 CNN 기반으로만 구성되어 있어 RNN과 같은 이전 이력 정보를 활용하지 않기 때문에 모든 도로 장면에서 항상 일관된 성능을 발휘하지 못할 가능성이 있다고 생각됩니다.
반면, Rwightman 모델은 모든 오류 한도에서 다른 두 모델보다 일관된 성능을 보였습니다.
이 결과는 DeepRoad가 자율주행 시스템의 수천 건의 불일치 동작을 감지할 수 있을 뿐만 아니라, 각 시스템의 견고성을 측정할 수 있다는 흥미로운 현상을 보여줍니다.
예를 들어, 원래의 Udacity 데이터셋만으로는 Autumn과 같은 자율주행 시스템의 한계를 발견하기 어렵습니다.

## 5. Related work
Testing and verification of DNN-based autonomous driving systems.
Different from traditional testing practices for DNN models [29; 20], a recent set of approaches (such as DeepXplore [23] and DeepTest [28]) utilize differential and metamorphic testing algorithms for identifying inputs that trigger inconsistencies among different DNN models, or among the original and transformed driving scenes.
Although such approaches have successfully found various autonomous driving system issues, there still lack approaches that can test DNN-based autonomous driving system with realistic synthesized driving scenes.

DNN 기반 자율주행 시스템의 테스트와 검증은 기존 DNN 모델의 테스트 방식과 다릅니다 [29; 20].
최근의 접근 방식(예: DeepXplore [23] 및 DeepTest [28])은 차별 및 변환 기반 테스트 알고리즘을 활용하여 서로 다른 DNN 모델 간 또는 원본과 변환된 주행 장면 간의 불일치를 유발하는 입력을 식별합니다.
이러한 방법들은 자율주행 시스템의 다양한 문제를 성공적으로 찾아냈으나, 현실적으로 합성된 주행 장면을 사용하여 DNN 기반 자율주행 시스템을 테스트할 수 있는 접근 방식은 여전히 부족한 실정입니다.

GAN-based virtual/real scene adaption.
GAN-based domain adaption has been recently shown to be effective in virtual-to-real and real-to-virtual scene adaption [32; 18].
DU-drive [32] proposes an unsupervised real to virtual domain unification framework for end-to-end driving.
Their key insight is the raw image may contain nuisance details which are not related to the prediction of steering angles, and a corresponding virtual scene can ignore these details and also address the domain shift problem.
SG-GAN [18] is designed to automatically transfer the scene annotation in virtual-world to facilitate real-world visual tasks.
In that work, a semanticaware discriminator is proposed for validating the fidelity of rendered image w.r.t each semantic region.

GAN 기반 가상/실제 장면 적응은 최근 가상에서 실제로, 또는 실제에서 가상으로 장면을 적응시키는 데 효과적인 것으로 입증되었습니다 [32; 18].
DU-drive [32]는 자율주행을 위한 비지도 학습 기반의 실제-가상 도메인 통합 프레임워크를 제안하며, 주요 개념은 원본 이미지가 조향각 예측에 불필요한 세부 정보를 포함할 수 있다는 점입니다.
해당 가상 장면은 이러한 불필요한 정보를 무시하고 도메인 변환 문제를 해결할 수 있습니다.
또한, SG-GAN [18]은 가상 세계의 장면 주석을 실제 시각적 작업에 활용할 수 있도록 자동으로 전이하는 기능을 갖춘 모델입니다.
이 연구에서는 각 의미 영역과 관련하여 렌더링된 이미지의 신뢰성을 검증하는 의미 인식 판별자를 제안했습니다.

Metamorphic testing.
Metamorphic testing is a classical software testing method that identify software bugs [33; 14; 27].
Its core idea is to detect violations of domain-specific metamorphic relations defined across outputs from multiple runs of the program with different inputs.
Metamorphic testing has been applied for testing machine learning classifiers [22; 30; 31].
In this paper, DeepRoad develops a specific GAN-based metamorphic testing module for DNN-based autonomous systems, where the metamorphic relations are defined such that regardless of how the driving scenes are synthesized to cope with weather conditions, the driving behaviors are expected to be consistent with those under the corresponding original driving scenes.

변환 기반 테스트(Metamorphic Testing)는 소프트웨어 버그를 식별하기 위한 전통적인 소프트웨어 테스트 방법입니다 [33; 14; 27].
이 방법의 핵심 아이디어는 프로그램의 여러 실행에서 다른 입력으로 생성된 출력 간의 도메인별 변환 관계의 위반을 감지하는 것입니다.
변환 기반 테스트는 머신러닝 분류기 테스트에도 적용된 바 있습니다 [22; 30; 31].
본 논문에서는 DeepRoad가 DNN 기반 자율주행 시스템을 위해 특정한 GAN 기반 변환 테스트 모듈을 개발했으며, 날씨 조건에 맞춰 주행 장면이 어떻게 합성되더라도 주행 동작이 해당 원본 장면과 일관성을 유지해야 한다는 관계를 정의했습니다.


