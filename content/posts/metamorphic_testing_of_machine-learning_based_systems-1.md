+++
title = 'Metamorphic Testing of Machine Learning Based Systems 1'
date = 2024-09-25T20:21:04+09:00
draft = false
+++

Testing machine learning (ML)-based systems requires different approaches compared to traditional software. With traditional software, the specification and its relation to the implementation is typically quite explicit: “When the user types a valid username and matching password, they are successfully logged in”. Very simple to understand, deterministic, and easy to write a test case for.

머신러닝(ML) 기반 시스템을 테스트하는 방식은 전통적인 소프트웨어 테스트와 다릅니다. 전통적인 소프트웨어의 경우, 명세와 구현의 관계가 명확하게 정의되어 있습니다. 예를 들어, “사용자가 올바른 사용자 이름과 비밀번호를 입력하면 성공적으로 로그인된다”는 식으로 단순하고 결정론적이며 테스트 사례를 작성하기도 쉽습니다.

ML-based systems are quite different. Instead of clearly defined inputs and logical flows based on explicit programming statements, a ML-based system is based on potentially huge input spaces with probabilistic outcomes from largely black-box components (models). In this article, I take a look at metamorphic testing, which is a technique that has become increasingly popular to address some of the ML-based systems testing challenge. I will go through some of the latest research, and present examples from different application domains.

ML 기반 시스템은 전통적인 소프트웨어와 매우 다릅니다. 명확한 입력과 논리적 흐름 대신, 잠재적으로 방대한 입력 공간과 블랙박스 모델에서 나오는 확률적 결과를 기반으로 합니다. 이 글에서는 이러한 ML 기반 시스템 테스트 문제를 해결하기 위해 점점 더 인기를 얻고 있는 메타모픽 테스트 기법을 살펴봅니다. 최신 연구를 검토하고 다양한 응용 분야의 예시를 제시할 것입니다.

Metamorphic Testing
Metamorphic Testing (MMT) was originally proposed quite a while back, at least up to (Chen1998). Having worked a long time with software testing research, I always viewed MMT as a curiosity with few real use cases. With ML-based systems, however, it seems to have found its niche nicely.

메타모픽 테스트(MMT)는 원래 오래전에 제안된 기법으로, 최소한 (Chen1998) 시점까지 거슬러 올라갑니다. 오랫동안 소프트웨어 테스트 연구를 해온 저로서는 MMT를 실제 사용 사례가 거의 없는 흥미로운 기법으로 여겼습니다. 하지만 ML 기반 시스템에서는 MMT가 그만의 적절한 역할을 찾은 것으로 보입니다.

The general idea of MMT is to describe the system functionality in terms of generic relations between inputs, the generic transformations of those inputs and their outputs, rather than as mappings of specific inputs to specific outputs.

MMT의 일반적인 아이디어는 시스템 기능을 특정 입력과 특정 출력 간의 매핑이 아닌, 입력과 입력 변환 및 출력 간의 일반적인 관계로 설명하는 것입니다.

One typical example used for metamorphic testing in the past has been from testing search engines (e.g., Zhou2016). As search engines are these days practically natural language processing (NLP)/ML-based systems, they also fit the topic of this article well. To illustrate the concept, I ran two queries on Google (in October 2020):

과거 메타모픽 테스트의 일반적인 예로 검색 엔진 테스트가 있습니다(e.g., Zhou2016). 현재 검색 엔진은 사실상 자연어 처리(NLP) 및 ML 기반 시스템이기 때문에 이 글의 주제와 잘 맞습니다. 이 개념을 설명하기 위해, 저는 2020년 10월에 구글에서 두 가지 검색 쿼리를 실행했습니다.

The first query is just one word “car”. The second query adds another word to the first query, “autonomous”. So the query now becomes “autonomous car”. This addition of a restrictive search keyword is an example of an input transformation (or “morphing”, in the spirit or metamorphic testing):

첫 번째 쿼리는 단어 하나인 “car”입니다. 두 번째 쿼리는 첫 번째 쿼리에 “autonomous”라는 단어를 추가하여 “autonomous car”가 됩니다. 이렇게 제한적인 검색 키워드를 추가하는 것은 입력 변환(혹은 메타모픽 테스트의 "변형")의 예입니다.

And to perform a check on the test results (a test oracle), we define a matching relation that should hold when the input transformation is applied:

그리고 테스트 결과를 확인하기 위해, 입력 변환이 적용될 때 성립해야 하는 일치 관계를 정의합니다(테스트 오라클).

In this case, adding the restrictive search term (“autonomous”) to the previous query (“car”) changes the result set, restricting it to a smaller set. From 8.3 billion results to 159 million. The metamorphic test would not specify these exact values, but rather the relation “restricting the query leads to fewer search results”. And one could generate several (seed) inputs (search queries), associated restrictive keywords for transformations, and run the query and check the metamorphic relation holds (restricting the query produces fewer results). For more details on MMT with search engines, see (Zhou2016).

이 경우, 제한적인 검색어(“autonomous”)를 기존 쿼리(“car”)에 추가하면 결과 집합이 83억 개에서 1억 5,900만 개로 줄어들어 더 작은 집합으로 제한됩니다. 메타모픽 테스트는 이러한 정확한 값이 아닌 “쿼리 제한 → 검색 결과 감소”라는 관계를 정의합니다. 여러 초기 입력(검색 쿼리)과 변환을 위한 제한 키워드를 생성하고, 쿼리를 실행하여 메타모픽 관계(쿼리 제한 시 결과 감소)가 유지되는지 확인할 수 있습니다. 검색 엔진 관련 MMT에 대한 자세한 내용은 (Zhou2016)을 참조하십시오.

The above is an example of what metamorphic testing refers to. You transform (morph) your inputs in some way, while at the same time defining a relation that should hold from the previous input (and its output) to the new morphed input (and its output). The key concepts / terms are:
morph/transform: modify a seed input in a way that your defined metamorphic relations should hold
metamorphic relation: the defined transformation of the input should have a known/measurable effect on the output. Checking that this relation holds after the transformation is the test oracle of metamorphic testing. (Test oracle is a general term for a mechanism to give a verdict on test result)
seed inputs: the inputs that are used as initial inputs for the tests, to be transformed. If you know the output of the seed input, you may use it to define a stricter relation (output should be correct). But even without the seed output, you can still define a relation check, but it might be a bit more relaxed (output should be similar but you don’t know if it is correct).

위 내용은 변형 테스트(metamorphic testing)의 예시입니다. 입력값을 어떤 방식으로 변형(변환)하고, 이전 입력(및 그 출력)과 새로운 변형된 입력(및 그 출력) 간의 관계를 정의하는 것입니다. 핵심 개념/용어는 다음과 같습니다:
morph/transform: 정의한 변형 관계가 유지되도록 시드 입력을 수정하는 것

metamorphic relation: 정의된 입력 변형이 출력에 대해 알려진 또는 측정 가능한 영향을 미쳐야 합니다. 변형 후 이 관계가 유지되는지 확인하는 것이 변형 테스트의 테스트 오라클입니다. (테스트 오라클은 테스트 결과에 대해 판정을 내리는 메커니즘을 나타내는 일반적인 용어입니다.)

seed inputs: 테스트의 초기 입력으로 사용되는 입력값으로, 변형될 대상입니다. 시드 입력의 출력을 알고 있다면, 더 엄격한 관계(출력이 정확해야 함)를 정의할 수 있습니다. 하지만 시드 출력이 없더라도 관계 검사를 정의할 수 있으며, 이 경우 출력이 정확한지는 모르지만 유사해야 한다는 식으로 다소 느슨하게 설정할 수 있습니다.

More generally metamorphic testing refers to defining such transformations, and observing their impact (metamorphic relations) on the result. The effectiveness and applicability then depends on how well and extensively these can be defined. I will present more concrete examples in the following sections.

좀 더 일반적으로 변형 테스트는 이러한 변환을 정의하고, 그 변환이 결과에 미치는 영향을 (변형 관계) 관찰하는 것을 의미합니다. 이러한 변환을 얼마나 잘, 그리고 폭넓게 정의할 수 있는지가 변형 테스트의 효과성과 적용 가능성을 결정합니다. 다음 섹션에서는 더 구체적인 예를 제시하겠습니다.

Problem Space
Why would you want to use metamorphic testing? I will illustrate this with an example for autonomous cars. Autonomous cars are recently going through a lot of development, getting a lot of funding, have safety-critical requirements, and are highly dependent on machine-learning. Which is maybe also why they have received so much attention in MMT research. Makes great examples.

왜 변형 테스트를 사용해야 할까요? 이를 자율주행 자동차의 예를 들어 설명하겠습니다. 최근 자율주행 자동차는 많은 개발이 이루어지고 있으며, 많은 자금이 투자되고, 안전이 중요한 요구 사항으로 부각되고 있으며, 머신러닝에 크게 의존하고 있습니다. 이러한 이유로 MMT(변형 테스트) 연구에서도 많은 주목을 받고 있으며, 훌륭한 예시로 자주 사용됩니다.

For example, the Tesla Autopilot collects data (or did when I was writing this..) from several front-, rear-, and side-cameras, a radar, and 12 ultrasonic sensors. At each moment in time, it must be able to process all this data, along with previous measurements, and come up with reasoning fulfilling highest safety-standards. Such real-world input-spaces are incredibly large. Consider the two pictures I took just a few days apart, near my previous office:

For example, the Tesla Autopilot collects data (or did when I was writing this..) from several front-, rear-, and side-cameras, a radar, and 12 ultrasonic sensors. At each moment in time, it must be able to process all this data, along with previous measurements, and come up with reasoning fulfilling highest safety-standards. Such real-world input-spaces are incredibly large. Consider the two pictures I took just a few days apart, near my previous office:

예를 들어, 테슬라 오토파일럿은 여러 개의 전방, 후방, 측면 카메라, 레이더, 그리고 12개의 초음파 센서로부터 데이터를 수집합니다(제가 이 글을 작성할 당시에도 그랬습니다). 매 순간 이 모든 데이터와 이전 측정값을 처리하여, 최고 수준의 안전 기준을 충족하는 판단을 내려야 합니다. 이러한 현실 세계의 입력 공간은 엄청나게 넓습니다. 며칠 전, 이전 사무실 근처에서 촬영한 두 장의 사진을 생각해 보세요.

Just in these two pictures there are many variations visible. Snow/no snow, shadows/no shadows, road markers / no markers, connecting roads visible, parking lots visible, other cars, and so on. Yet in all such conditions one would be expected to be able to navigate, safely. To illustrate the problem a bit more, here are some example variants in that domain that quickly come to mind:

이 두 장의 사진만 보더라도 눈이 있음/없음, 그림자가 있음/없음, 도로 표시가 있음/없음, 연결 도로가 보임, 주차장이 보임, 다른 차량이 있음 등 다양한 변화를 볼 수 있습니다. 그러나 이러한 모든 상황에서도 안전하게 주행할 수 있어야 합니다. 문제를 조금 더 설명하기 위해, 이와 같은 도메인에서 빠르게 떠오르는 몇 가지 예시 변형을 소개하겠습니다:

Besides these, one can easily expand this to different locations, road shapes, object types, bridges, trains, ... Other sensors have other considerations, every location is different, and so on.

이 외에도 다양한 위치, 도로 형태, 객체 유형, 다리, 기차 등으로 쉽게 확장할 수 있습니다. 각 센서마다 고려해야 할 사항이 다르고, 모든 위치는 서로 다르기 때문에 상황이 더욱 복잡해집니다.

In different domains of ML-based system applications, one would need to be able to identify similar problem scenarios, and their relevant combinations, to be able to test them. Manually building test sets to cover all this is (for me) an unrealistic effort.

다양한 머신러닝 기반 시스템 응용 도메인에서는 이와 유사한 문제 시나리오와 그에 관련된 조합을 식별하고 이를 테스트할 수 있어야 합니다. 이러한 모든 경우를 수동으로 테스트 세트로 구성하는 것은 (제 생각에) 현실적으로 불가능한 작업입니다.

Metamorphic Testing with Autonomous Cars
Metamorphic testing can help in better covering domains such as the above autonomous cars problem space. As the interest is high, many approaches for this have also been presented, and I will describe a few of those here.

변형 테스트는 위에서 언급한 자율주행 자동차와 같은 문제 영역을 더 잘 포괄할 수 있도록 도와줍니다. 이와 같은 문제에 대한 관심이 높아 다양한 접근법이 제시되었으며, 여기서 몇 가지를 설명하겠습니다.

Covering Image Variations
The DeepTest work in (Tian2018) uses transformations on real images captured from driving cars to produce new images. In this case, the metamorphic attributes are:
Seed inputs: Real images from car cameras.
Metamorphic transformations: moving, tilting, blurring, scaling, zooming, adding fog, adding rain, etc. on he original images
Metamorphic relation: the autonomous driving decisions should show minimal divergence on the same input images after the transformations.

(Tian2018)의 DeepTest 연구에서는 주행 중 촬영된 실제 이미지에 변환을 적용하여 새로운 이미지를 생성합니다. 이 경우 변형 속성(metamorphic attributes)은 다음과 같습니다:
시드 입력: 자동차 카메라에서 촬영한 실제 이미지  
변형 변환: 원본 이미지에 이동, 기울이기, 흐리기, 확대/축소, 줌, 안개 추가, 비 추가 등의 변환 적용  
변형 관계: 변환된 후에도 동일한 입력 이미지에 대해 자율주행 시스템의 판단이 최소한의 차이를 보여야 함  

The following illustrates this with some simple examples using the road image from outside my previous office. In the following, I “transformed” the image by simply rotating the camera a bit at the location. I then added the arrows to illustrate how a system should “predict” a path that should be taken. The arrow here is manually added, and intended to be only illustrative:

다음은 이전 사무실 밖에서 촬영한 도로 이미지를 사용하여 간단한 예시를 보여줍니다. 여기서 저는 카메라를 해당 위치에서 약간 회전시켜 이미지를 “변환”했습니다. 그런 다음 시스템이 예측해야 할 경로를 나타내기 위해 화살표를 추가했습니다. 이 화살표는 수동으로 추가한 것으로, 단순히 설명을 위한 것입니다:

And the same, but with the snowy ground (two transformations in the following compared to the above; added snow + rotation):

그리고 동일한 예시를 눈 덮인 도로에 적용했습니다(위 이미지와 비교했을 때 두 가지 변환이 적용되었습니다: 눈 추가 + 회전).

Of course, no-one would expect to manually create any large number of such images (or transformations). Instead, automated transformation tools can be used. For example, there are several libraries for image augmentation, originally created to help increase training dataset sizes in machine learning. The following illustrates a few such augmentations run on the original non-snow image from above:

물론, 이렇게 많은 수의 이미지(또는 변환)를 수동으로 생성하는 것은 기대하기 어렵습니다. 대신, 자동화된 변환 도구를 사용할 수 있습니다. 예를 들어, 원래는 머신러닝에서 훈련 데이터셋 크기를 늘리기 위해 개발된 여러 이미지 증강 라이브러리가 있습니다. 아래는 눈이 없는 원본 이미지에 적용된 몇 가지 이미지 증강 예시를 보여줍니다:

All these augmented / transformed images were generated from the same original source image shown before, using the Python imgaug image augmentation library. Some could maybe be improved with more advanced augmentation methods, but most are already quite useful.

이 모든 증강/변환 이미지는 앞서 보여준 동일한 원본 이미지에서 Python의 `imgaug` 이미지 증강 라이브러리를 사용하여 생성되었습니다. 일부 이미지는 더 고급 증강 방법을 사용하면 개선될 수 있겠지만, 대부분 이미 충분히 유용합니다.

Once those transformations are generated, the metamorphic relations on the generated images can be checked. For example, the system should propose a very similar driving path, with minimal differences across all transformations on acceleration, steering, etc. Or more complex checks if such can be defined, such as defining a known reference path (if such exists).

이러한 변환이 생성되면, 생성된 이미지에 대해 변형 관계를 확인할 수 있습니다. 예를 들어, 시스템은 가속, 조향 등 모든 변환에서 최소한의 차이로 매우 유사한 주행 경로를 제안해야 합니다. 또는, 정의할 수 있다면, 알려진 참조 경로(존재할 경우)를 기준으로 더 복잡한 검사를 수행할 수도 있습니다.

Again, this process of transformation and checking metamorphic relations is what MMT is about. It helps achieve higher coverage and thus confidence by automating some of the testing process for complex systems, where scaling to the large input spaces is otherwise difficult.

다시 말해, 이러한 변환 과정과 변형 관계를 확인하는 과정이 바로 MMT의 핵심입니다. 이는 복잡한 시스템에서 입력 공간이 매우 넓어 수동으로 테스트하기 어려운 경우, 테스트 과정을 자동화하여 더 높은 커버리지와 신뢰성을 확보하는 데 도움이 됩니다.

GAN-based Transformations with MMT
A more advanced approach to generate input transformations is to apply different ML-based techniques to build the transformations themselves. In image augmentation, one such method is Generative Adversarial Networks (GANs). An application of GANs to autonomous cars is presented in (Zhang2018). In their work, GANs are trained to transform images with different weather conditions. For example, taking a sunny image of a road, and transforming this into a rainy or foggy image.

입력 변환을 생성하는 더 발전된 접근 방식은 다양한 머신러닝 기반 기술을 적용하여 변환 자체를 구축하는 것입니다. 이미지 증강에서 이러한 방법 중 하나가 생성적 적대 신경망(GANs)입니다. 자율주행 자동차에 GANs를 적용한 연구는 (Zhang2018)에서 제시되었습니다. 이 연구에서는 GANs를 훈련시켜 서로 다른 날씨 조건으로 이미지를 변환합니다. 예를 들어, 맑은 날 도로 이미지를 비 오는 날이나 안개가 낀 이미지로 변환하는 것입니다.

The argument is that GAN generated weather effects and manipulations are more realistic than more traditional synthetic transformations. (Zhang2018) uses the NVidia UNIT (Liu2017) toolkit to train and apply the GAN models, using input such as YouTube videos for training.

논점은 GAN이 생성한 날씨 효과와 조작이 기존의 합성 변환보다 더 현실적이라는 것입니다. (Zhang2018) 연구에서는 NVidia UNIT (Liu2017) 툴킷을 사용하여 GAN 모델을 훈련시키고 적용하며, 유튜브 동영상과 같은 데이터를 입력으로 사용하여 훈련을 진행했습니다.

Images illustrating the GAN results are available on the UNIT website, as well as in higher resolution in their Google Photos album. I recommend having a look, it is quite interesting. The smaller images on the UNIT website look very convincing, but looking more closely in the bigger images in the photo albums reveals some limitations. However, the results are quite impressive, and this was a few years ago. I expect the techniques to improve further over time. In general, using machine learning to produce transformations appears to be a very promising area in MMT.

GAN 결과를 보여주는 이미지는 UNIT 웹사이트와 Google Photos 앨범에서 확인할 수 있습니다. 한 번 확인해 보시길 권장합니다. 매우 흥미롭습니다. UNIT 웹사이트의 작은 이미지는 매우 그럴듯해 보이지만, 사진 앨범의 큰 이미지를 자세히 살펴보면 몇 가지 한계점이 드러납니다. 그럼에도 불구하고, 결과는 상당히 인상적이며, 이는 몇 년 전의 연구입니다. 이러한 기술은 시간이 지나면서 더욱 발전할 것으로 기대됩니다. 일반적으로, 머신러닝을 사용하여 변환을 생성하는 것은 변형 테스트(MMT)에서 매우 유망한 분야로 보입니다.

LIDAR Transformation
Besides cameras, there are many possible sensors a system can also use. In autonomous cars, one such system is LIDAR, measuring distances to objects using laser-based sensors. A study of applying metamorphic testing on LIDAR data in the Baidu Apollo autonomous car system is presented in (Zhou2019).

카메라 외에도 시스템에서 사용할 수 있는 다양한 센서가 존재합니다. 자율주행 자동차의 경우, 레이저 기반 센서를 사용하여 물체와의 거리를 측정하는 LIDAR가 그 중 하나입니다. LIDAR 데이터를 변형 테스트에 적용한 연구는 (Zhou2019)에서 Baidu Apollo 자율주행 자동차 시스템을 대상으로 제시되었습니다.

The system first identifies a region of interest (ROI), the “drivable” area. It then identifies and tracks objects in this area. The system consists of multiple components:
Object segmentation and bounds identification: Find and identify obstacles in ROI
Object tracking: Tracking the obstacles (movement)
Sequential type fusion: To smooth the obstacle types over time (make more consistent classifications over time by using also time related data)

시스템은 먼저 "주행 가능한" 영역인 관심 영역(ROI)을 식별합니다. 그런 다음 이 영역 내의 객체를 식별하고 추적합니다. 이 시스템은 다음과 같은 여러 구성 요소로 이루어져 있습니다:
- **객체 분할 및 경계 식별**: ROI 내 장애물을 찾고 식별  
- **객체 추적**: 장애물의 움직임 추적  
- **순차적 유형 융합**: 시간 관련 데이터를 사용하여 장애물 유형을 시간에 따라 더 일관되게 분류하도록 하는 과정 (장애물 유형을 시간이 지남에 따라 부드럽게 조정)  

The (Zhou2019) study focuses on metamorphic testing of the object identification component, specifically on robustness of classification vs misclassification in minor variations of the LIDAR point cloud. The LIDAR point cloud in this case is simply collection of measurement points the LIDAR system reports seeing. These clouds can be very detailed, and the number of measured points very large (Zhou2019).

(Zhou2019) 연구는 객체 식별 구성 요소에 대한 변형 테스트에 초점을 맞추고 있으며, 특히 LIDAR 포인트 클라우드의 작은 변동에 따른 분류의 강건성과 오분류 여부를 검증합니다. 이 경우 LIDAR 포인트 클라우드는 LIDAR 시스템이 감지한 측정 포인트의 집합을 의미합니다. 이러한 클라우드는 매우 세밀할 수 있으며, 측정된 포인트의 수가 매우 많을 수 있습니다 (Zhou2019).

The following figures illustrates this scenario (see (Zhou2019) for the realistic LIDAR images from actual cars, I just use my own drawings here to illustrate the general idea. I marked the ROI in a darker color, and added some dots in circular fashion to illustrate the LIDAR scan. The green box illustrates a bigger obstacle (e.g., a car), and the smaller red box illustrates a smaller obstacle (e.g., a pedestrian):

다음 그림은 이 시나리오를 설명합니다 (실제 차량의 현실적인 LIDAR 이미지는 (Zhou2019)를 참조하시기 바랍니다. 여기서는 일반적인 개념을 설명하기 위해 제가 직접 그린 그림을 사용했습니다). ROI는 더 짙은 색으로 표시했으며, LIDAR 스캔을 나타내기 위해 원형으로 점을 추가했습니다. 초록색 상자는 더 큰 장애물(예: 자동차)을, 작은 빨간색 상자는 더 작은 장애물(예: 보행자)을 나타냅니다.

The metamorphic relations and transformations in this case are:
Metamorphic relation: same obstacles (objects) should be identified both before and after adding small amounts of noise to the LIDAR point cloud.
Transformation: add noise (points to the LIDAR point cloud)
Seed inputs: actual LIDAR measurements from cars

이 경우 변환 및 변환 관계는 다음과 같습니다:

- **변환 관계 (Metamorphic relation)**: LIDAR 포인트 클라우드에 소량의 노이즈를 추가하기 전후로 동일한 장애물(객체)이 식별되어야 합니다.
- **변환 (Transformation)**: LIDAR 포인트 클라우드에 노이즈(포인트) 추가
- **초기 입력 (Seed inputs)**: 자동차의 실제 LIDAR 측정값

The following figure illustrates this type of metamorphic transformation, with the added points marked in red. I simply added them in a random location, outside the ROI in this case, as this was the example also in (Zhou2019):

다음 그림은 이러한 유형의 변환 관계를 보여주며, 추가된 포인트는 빨간색으로 표시되어 있습니다. 이 예에서는 추가된 포인트를 ROI(관심 영역) 외부의 임의 위치에 추가했으며, 이는 (Zhou2019)에서도 사용된 예시입니다.

The above is a very simple transformation and metamorphic relation to check, but I find often the simple ones work the best.

위 내용은 매우 간단한 변환과 변환 관계로, 검증하기 쉽습니다. 하지만 종종 이러한 단순한 방법들이 가장 효과적이라는 것을 발견하곤 합니다.

In summary, the MMT approach here takes existing LIDAR data, and adds some noise to it, in form of added LIDAR data points. In relation to the real world, such noise is described in (Zhou2019) as potentially insects, dust, or sensor noise. The amount of added noise is also described as a very small percentage of the overall points, to make it more realistic.

요약하자면, 여기서 사용된 MMT 접근법은 기존의 LIDAR 데이터를 가져와 LIDAR 데이터 포인트를 추가하는 형태로 노이즈를 더하는 것입니다. 현실 세계와 관련하여, 이러한 노이즈는 (Zhou2019)에서 잠재적으로 곤충, 먼지, 또는 센서 노이즈로 설명됩니다. 추가된 노이즈의 양은 전체 포인트의 매우 작은 비율로, 더 현실적이도록 설정됩니다.

The metamorphic experiments in (Zhou2019) show how adding a small number of points outside the ROI area in the point cloud was enough to cause the classifier (metamorphic relation check) to fail.

(Zhou2019)의 변환 실험에서는 포인트 클라우드의 ROI 영역 외부에 소량의 포인트를 추가하는 것만으로도 분류기(변환 관계 검사)가 실패하는 결과를 초래할 수 있음을 보여줍니다.

As a result, (Zhou2019) report discussing with the Baidu Apollo team about their findings, getting acknowledgement for the issues, and how the Baidu team incorporated some of the test data into their training dataset. This can be a useful approach, since metamorphic testing can be seen as a way to generate new data that could be used for training. However, I think one should not simply discard the tests in either case, even if re-using some of the data for further ML-model training. More on this later.

결과적으로, (Zhou2019)에서는 Baidu Apollo 팀과 그들의 연구 결과에 대해 논의하였고, 문제를 인정받았으며, Baidu 팀이 일부 테스트 데이터를 그들의 학습 데이터셋에 포함시켰음을 보고하고 있습니다. 이는 변환 테스트를 새로운 데이터를 생성하는 방법으로 볼 수 있기 때문에 유용한 접근 방식이 될 수 있습니다. 하지만, 일부 데이터를 추가적인 ML 모델 학습에 재사용하더라도 테스트 자체를 간과해서는 안 된다고 생각합니다. 이에 대해서는 나중에 더 자세히 설명하겠습니다.

