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

Adversarial Inputs and Relations Across Time
Adversarial Inputs
A specific type of transformation that is often separately discussed in machine learning is that of adversarial inputs, which is extensively described in (Goodfellow2018). In general, an adversarial input aims to trick the machine learning algorithm to make a wrong classification. An example from (Goodfellow2018) is to fool an autonomous car (surprise) to misclassify a stop sign and potentially lead to an accident or other issues.
기계 학습에서 종종 별도로 논의되는 특정 유형의 변환은 적대적 입력(Adversarial Inputs)으로, 이는 (Goodfellow2018)에서 자세히 설명되고 있습니다. 일반적으로 적대적 입력은 기계 학습 알고리즘을 속여 잘못된 분류를 하도록 유도하는 것을 목표로 합니다. (Goodfellow2018)의 예로는 자율 주행 자동차가 정지 표지판을 잘못 분류하여 사고나 다른 문제를 일으킬 수 있는 경우를 들 수 있습니다.

Generating such adversarial inputs can be seen as one example of a metamorphic transformation, with an associated relation that the output should not change, or change should be minimal, due to adversarial inputs.
이러한 적대적 입력을 생성하는 것은 변환 관계의 한 예로 볼 수 있으며, 관련된 변환 관계로는 적대적 입력으로 인해 출력이 변하지 않거나 최소한의 변화만 있어야 한다는 것입니다.

Typically such adversarial testing requires specifically tailored data to trigger such misclassification. In a real-world driving scenario, where the car sensors are not tampered with, it might be harder to produce such purely adversarial effects. However, there are some studies and approaches, such as (Zhou2020) considering this for real-world cases. More on this in a bit.
일반적으로 이러한 적대적 테스트는 잘못된 분류를 유발하기 위해 특별히 설계된 데이터를 필요로 합니다. 실제 운전 시나리오에서 차량 센서가 변조되지 않은 경우, 순수하게 적대적 효과를 만들어내는 것은 더 어려울 수 있습니다. 그러나 (Zhou2020)과 같은 연구에서는 이러한 문제를 실제 환경에서도 고려하고 있습니다. 이에 대해서는 잠시 후에 더 자세히 설명하겠습니다.

Beyond autonomous cars, digitally altered or tailored adversarial inputs might be a bigger issue. For example, in domains such as cyber-security log analysis, or natural language processing, where providing customized input data could be easier. I have not seen practical examples of this from the real world, but I expect once the techniques mature and become more easily available, more practical sightings would surface.
자율 주행 자동차를 넘어, 디지털로 변조되거나 조작된 적대적 입력은 더 큰 문제로 부각될 수 있습니다. 예를 들어, 맞춤형 입력 데이터를 제공하기가 비교적 쉬운 사이버 보안 로그 분석이나 자연어 처리와 같은 분야에서는 이러한 위험이 더 커질 수 있습니다. 실제 사례는 아직 보지 못했지만, 기술이 성숙하고 더 쉽게 접근할 수 있게 되면, 실질적인 사례가 더 많이 나타날 것으로 예상됩니다.

Much of the work on adversarial elements, such as (Goodfellow2018), have examples of adversarially modified single inputs (images). Real systems are often not so simple. For example, as a car drives, the images (as well as other sensor data), and the decisions that need to be made based on that data, change continuously. This is what the (Zhou2020) paper discusses for autonomous cars.
(Goodfellow2018)과 같은 연구에서는 주로 단일 입력(이미지)에 대한 적대적 변형 예제를 다루고 있습니다. 하지만 실제 시스템은 그리 단순하지 않습니다. 예를 들어, 차량이 주행할 때 이미지를 비롯한 다양한 센서 데이터와 해당 데이터를 기반으로 내려야 하는 결정은 끊임없이 변합니다. 이는 (Zhou2020) 논문에서 자율 주행 자동차를 위해 논의되고 있는 내용입니다.

Relations Across Time
In many cases, besides singular inputs, sequences of input over time are more relevant for the ML-based system. Driving past a sign (or a digital billboard..), the system has to cope with all the sensor data at all the different positions in relation to the environment over time. In this case, the camera viewing angle. For other sensors (LIDAR etc), the input and thus output data would change in a similar manner over time.
많은 경우 단일 입력 외에도 시간에 따른 입력 시퀀스가 ML 기반 시스템에 더 중요한 역할을 합니다. 예를 들어, 표지판이나 디지털 광고판을 지나칠 때, 시스템은 시간에 따라 환경과의 다양한 위치 관계에서 수집되는 모든 센서 데이터를 처리해야 합니다. 이 경우 카메라의 시야각이 중요한 요소가 됩니다. 다른 센서들(LIDAR 등)도 마찬가지로, 입력 데이터와 그에 따른 출력 데이터가 시간에 따라 유사한 방식으로 변화하게 됩니다.

Following is an example of what might be two frames a short time apart. In a real video stream there would be numerous changes and images (and other inputs) per second:
다음은 짧은 시간 간격으로 두 프레임이 있을 수 있는 예입니다. 실제 비디오 스트림에서는 매 초마다 많은 변화와 이미지(및 기타 입력)가 발생합니다.

Not only does the angle change, but time as a context should be more generally considered in this type of testing (and implementation). Are we moving past the sign? Towards it? Passing it? Did we stop already? What else is in the scene? And so on.
각도뿐만 아니라, 이러한 유형의 테스트(및 구현)에서는 시간이라는 맥락도 보다 일반적으로 고려되어야 합니다. 예를 들어, 우리는 표지판을 지나가고 있는가? 표지판을 향해 다가가고 있는가? 표지판을 지나쳤는가? 이미 멈췄는가? 장면에 다른 요소는 무엇이 있는가? 등 다양한 질문들이 포함됩니다. 이러한 맥락적 요소들은 시스템의 이해와 정확한 판단을 위해 매우 중요합니다.

This topic is studied in (Zhou2020), which considers it from the viewpoint of adversarial input generation. In a real-world setting, you are less likely to have your image data directly manipulated, but may be susceptible to adversarial inputs on modified traffic signs, digital billboards, or similar. This is what they (Zhou2020) focus on.
이 주제는 (Zhou2020)에서 적대적 입력 생성의 관점에서 연구되고 있습니다. 현실 세계에서는 이미지 데이터가 직접적으로 조작될 가능성은 낮지만, 변형된 교통 표지판이나 디지털 광고판과 같은 적대적 입력에는 취약할 수 있습니다. (Zhou2020)은 이러한 실제 사례에 초점을 맞추고 있습니다.

The following example illustrates how any such modification would also need to change along with the images, over time (compared to calculating a single, specific altered input vs real-world physical data moving across time):
다음 예시는 단일 특정 변형 입력을 계산하는 것과 비교하여, 실제 세계의 물리적 데이터가 시간에 따라 이동할 때 모든 변형도 이미지와 함께 시간에 따라 변화해야 한다는 점을 보여줍니다.

This temporal aspect is important in more ways than just for adversarial inputs. For example, all the image augmentations (weather effects, etc) I discussed earlier would benefit from being applied in a realistic driving scenario (sequences of images) vs just a single image. This is what the cars have to deal with in the real world after all.
이러한 시간적 측면은 적대적 입력뿐만 아니라 여러 면에서 중요합니다. 예를 들어, 제가 이전에 논의했던 이미지 증강(날씨 효과 등)도 단일 이미지가 아닌 실제 운전 시나리오(연속된 이미지 시퀀스)에서 적용될 때 더 효과적일 것입니다. 실제 환경에서는 차량이 이러한 연속적인 변화에 대응해야 하기 때문입니다.

The test oracle in (Zhou2020) also considers the effect of the adversarial input from two different viewpoints: strength and probability. That is, how large deviations can you cause in the steering of the car with the adversarial changes, and how likely it is that you can cause these deviations with the adversarial input.
(Zhou2020)에서 테스트 오라클은 적대적 입력의 영향을 두 가지 관점에서 고려합니다: 강도와 확률입니다. 즉, 적대적 변화를 통해 차량의 조향에 얼마나 큰 편차를 유발할 수 있는지와 이러한 편차를 적대적 입력으로 얼마나 자주 유발할 수 있는지를 평가합니다.

Beyond cars and video streams, time series sequences are common in other domains as well. The drone scenarios discussed are one example. Other examples include processing linked paragraphs of text, longer periods of signal in a stock market, or basic sensor signals such as temperature and wind speed.
자동차와 비디오 스트림을 넘어, 시계열 데이터는 다른 분야에서도 흔하게 사용됩니다. 예를 들어, 드론 시나리오가 그 중 하나입니다. 또한 연결된 텍스트 단락 처리, 주식 시장에서의 장기간 신호 분석, 온도와 풍속 같은 기본적인 센서 신호 처리 등도 시계열 데이터의 예로 들 수 있습니다.

Minimizing the Test Set
While automating metamorphic testing can be quite straightforward (once you figure your domain relations and build working transformations…), the potential input space from which to choose, and the number of transformations and their combinations can quickly grow huge. For this reason, test selection in MMT is important, just as with other types of testing.
변환 관계를 정의하고 변환을 구축하면 변환 테스트를 자동화하는 것은 비교적 간단하지만, 선택할 수 있는 잠재적 입력 공간, 변환의 수, 그리고 그들의 조합이 급격히 증가할 수 있습니다. 이러한 이유로, 다른 테스트 유형과 마찬가지로 변환 테스트에서도 테스트 선택이 매우 중요합니다. 적절한 테스트 사례를 선택하는 것은 효율적인 검증을 위해 필수적입니다.

One approach to address this is presented in (Tian2018), which applies a greedy search strategy. Starting with a seed set of images and transformations, the transformations and their combinations are applied on the input (images), and the achieved neuron activation coverage is measured. If they increase coverage, the “good” combinations are added back to the seed set for following rounds, along with other inputs and transformations, as long as they provide some threshold of increased coverage. This iterates until defined ending thresholds (or number of experiments). Quite similar to more traditional testing approaches.
이를 해결하기 위한 한 가지 접근법은 (Tian2018)에서 제시된 탐욕적 탐색(greedy search) 전략입니다. 초기 이미지와 변환 집합을 사용하여 시작하고, 입력 이미지에 변환과 조합을 적용한 후 달성된 뉴런 활성화 범위를 측정합니다. 범위가 증가하면, “좋은” 조합은 다음 라운드를 위해 다른 입력 및 변환과 함께 초기 집합에 추가되며, 일정한 증가 임계값을 초과할 경우에만 계속 사용됩니다. 이 과정은 정의된 종료 임계값 또는 실험 횟수에 도달할 때까지 반복됩니다. 이는 전통적인 테스트 접근 방식과 유사합니다.

Coverage in (Tian2018) is measured in terms of activations of different neurons in the ML model. They build coverage criteria for different neural network architectures, such as convolutional neural nets, recurrent neural nets, and dense neural nets. Various other coverage criteria also have been proposed, that could be used, such as one in (Gerasimou2020) on evaluating the importance of different neurons in classification.
(Tian2018)에서는 커버리지를 ML 모델의 다양한 뉴런 활성화 측면에서 측정합니다. 그들은 합성곱 신경망(CNN), 순환 신경망(RNN), 밀집 신경망(Dense Neural Networks)과 같은 다양한 신경망 아키텍처에 대한 커버리지 기준을 구축했습니다. 또한 (Gerasimou2020)에서 제안된 분류에서 뉴런의 중요도를 평가하는 것과 같은, 사용할 수 있는 다양한 다른 커버리지 기준도 존재합니다.

When more and easily applicable tools become available for this type of ML-model coverage measurement, it would seem a very useful approach. However, I do not see people generally writing their own neural net coverage measurement tools.
이런 유형의 ML 모델 커버리지 측정을 위한 더 많은 도구가 쉽게 사용할 수 있게 된다면 매우 유용한 접근 방식이 될 것입니다. 그러나 일반적으로 사람들이 직접 신경망 커버리지 측정 도구를 개발하는 경우는 드문 것 같습니다. 도구의 개발과 적용이 어렵기 때문에, 이러한 측정 방법이 더 널리 사용되기 위해서는 표준화된 도구와 접근법이 필요합니다.

Relation to Traditional Software Testing
Besides test suite optimization, it is important to consider MMT more broadly in relation to overall software and system testing. MMT excels in testing and verifying many aspects of ML-based systems, which are more probabilistic and black-box in nature. At least to gain higher confidence / assurance in them.
테스트 스위트 최적화 외에도, 변환 테스트(MMT)를 전체 소프트웨어 및 시스템 테스트와 연관 지어 더 넓게 고려하는 것이 중요합니다. MMT는 확률적이고 블랙박스적인 특성을 가진 ML 기반 시스템의 여러 측면을 테스트하고 검증하는 데 강점을 가지고 있습니다. 이를 통해 시스템에 대한 신뢰도와 보증을 높이는 데 도움이 될 수 있습니다.

However, even in ML-based systems, the ML-part is not generally an isolated component working alone. Rather it consumes inputs, produces outputs, and uses ML models for processing complex datasets. The combinatorial, equivalence partitioning, and model-based methods I mentioned earlier are some examples of how the MMT based approaches can be applied together with the overall, more traditional, system testing.
그러나 ML 기반 시스템에서도 ML 부분은 일반적으로 독립적으로 작동하는 고립된 구성 요소가 아닙니다. 오히려 입력을 받아 출력으로 변환하고, 복잡한 데이터셋을 처리하기 위해 ML 모델을 사용합니다. 제가 이전에 언급한 조합 테스트, 등가 분할, 모델 기반 방법은 MMT 접근 방식을 전체적인, 보다 전통적인 시스템 테스트와 결합하여 적용할 수 있는 몇 가지 예입니다. 이러한 통합 접근 방식은 ML 모델의 신뢰성을 높이고 시스템 전체의 테스트 효과를 향상시키는 데 도움이 됩니다.

As I mentioned with the Baidu Apollo case and its LIDAR test data generation, one of the feedbacks was to use the metamorphic test data for further ML training. This in general seems like a useful idea, and it is always nice to get more training data. In my experience with building ML-based systems, and training related ML-models, everyone always wants more training data.
제가 Baidu Apollo 사례와 LIDAR 테스트 데이터 생성에 대해 언급했듯이, 피드백 중 하나는 변환 테스트 데이터를 추가적인 ML 학습에 사용하는 것이었습니다. 이는 일반적으로 유용한 아이디어로 보이며, 더 많은 학습 데이터를 확보하는 것은 언제나 좋은 일입니다. ML 기반 시스템을 구축하고 관련 ML 모델을 학습시키는 경험에 비추어 볼 때, 누구나 항상 더 많은 학습 데이터를 원하기 마련입니다. 추가적인 데이터를 확보하면 모델의 성능과 일반화 능력을 향상시키는 데 큰 도움이 됩니다.

However, I believe one should not simply dump all MMT test data into the training dataset. A trained model will learn from the given data, and can be tested for general accuracy on a split test set. This is the typical approach to test a specific ML-model in isolation. However, in practice, the classifications will not be 100% accurate, and some items will end up misclassified, or with low confidence scores. These further feed into the overall system, which may have unexpected reactions in combination with other inputs or processes. Running specific (MMT based or not) tests with specific inputs helps highlight exactly which data is causing issues, how this behaviour changes over time, and so on. If you just throw your MMT tests into the training set and forget it, you lose the benefit of this visibility.
하지만 변환 테스트(MMT) 데이터를 단순히 학습 데이터셋에 모두 추가해서는 안 된다고 생각합니다. 학습된 모델은 제공된 데이터로부터 학습하며, 분리된 테스트 세트를 사용하여 일반적인 정확도를 검증할 수 있습니다. 이는 특정 ML 모델을 독립적으로 테스트하는 일반적인 접근 방식입니다. 하지만 실제 환경에서는 분류가 100% 정확하지 않으며, 일부 항목은 오분류되거나 낮은 신뢰도 점수를 가질 수 있습니다. 이러한 결과는 전체 시스템에 다시 피드백되어 다른 입력이나 프로세스와 결합될 때 예상치 못한 반응을 유발할 수 있습니다.

특정 입력을 사용하여 특정 테스트(MMT 기반이든 아니든)를 실행하는 것은 정확히 어떤 데이터가 문제를 일으키고 있는지, 이 행동이 시간이 지남에 따라 어떻게 변화하는지를 파악하는 데 도움이 됩니다. MMT 테스트 데이터를 학습 세트에 무작정 추가하고 이를 잊어버리면, 이러한 가시성의 이점을 잃게 됩니다. 문제를 식별하고 해결할 수 있는 능력을 유지하기 위해서는 테스트 데이터를 잘 관리하고, 문제를 일으키는 데이터를 명확히 이해한 후에 신중하게 학습에 활용해야 합니다.

Besides MMT, and complimentary to it, other interesting approaches of tailoring traditional testing techniques for ML-based system testing exist. One specific approach is A/B testing (evaluating benefits of different options). In ML-based systems, this can also be a feedback loop from the human user, or operational system, back to testing and training. The Tesla Shadow Mode is one interesting example, where the autonomous ML-based system makes continuous driving decisions, but these decisions are never actually executed. Rather they are compared with the actual human driver choices in those situations, and this is used to refine the models. Similar approaches, where the system can learn from human corrections are quite common, such as tuning search-engine results and machine translations, based on human interactions with the system. You are changing / morphing the system here as well, but in a different way. This would also make an interesting seed input source for MMT, along with oracle data (e.g., driving path taken by human user) for the metamorphic relation.
Besides MMT, and complimentary to it, other interesting approaches of tailoring traditional testing techniques for ML-based system testing exist. One specific approach is A/B testing (evaluating benefits of different options). In ML-based systems, this can also be a feedback loop from the human user, or operational system, back to testing and training. The Tesla Shadow Mode is one interesting example, where the autonomous ML-based system makes continuous driving decisions, but these decisions are never actually executed. Rather they are compared with the actual human driver choices in those situations, and this is used to refine the models. Similar approaches, where the system can learn from human corrections are quite common, such as tuning search-engine results and machine translations, based on human interactions with the system. You are changing / morphing the system here as well, but in a different way. This would also make an interesting seed input source for MMT, along with oracle data (e.g., driving path taken by human user) for the metamorphic relation.
MMT와 보완적으로, ML 기반 시스템 테스트를 위해 기존의 테스트 기법을 변형하여 적용하는 흥미로운 접근 방식들도 있습니다. 그 중 하나가 A/B 테스트입니다. 이는 서로 다른 옵션의 효과를 평가하는 방식으로, ML 기반 시스템에서는 사용자나 운영 시스템으로부터의 피드백 루프를 통해 테스트와 학습에 다시 반영될 수 있습니다.

테슬라의 *Shadow Mode*가 그 좋은 예로, 자율 주행 ML 시스템이 지속적으로 운전 결정을 내리지만, 실제로 실행되지는 않습니다. 대신 이러한 결정은 당시 상황에서 인간 운전자의 실제 선택과 비교되어 모델을 개선하는 데 사용됩니다. 이와 유사하게, 시스템이 인간의 수정사항으로부터 학습하는 방식은 매우 흔합니다. 예를 들어, 검색 엔진 결과를 조정하거나, 인간 상호작용을 기반으로 기계 번역을 개선하는 경우가 있습니다. 이러한 방법들은 시스템을 변화시키거나 개선하는 것이지만, 다른 방식으로 변형하는 과정입니다.

이러한 피드백 루프는 MMT를 위한 흥미로운 초기 입력(seed input) 소스가 될 수 있습니다. 예를 들어, 인간 사용자가 선택한 운전 경로와 같은 오라클 데이터를 활용하여 변환 관계를 정의하고, 시스템이 인간의 결정을 학습하며 변형되는 방식을 테스트할 수 있습니다. 이러한 접근 방식은 단순한 데이터 추가 이상으로, 인간 피드백과 시스템 개선이 결합된 통합된 테스트 및 학습 체계를 구축하는 데 도움이 될 것입니다.

Conclusions
Testing machine learning based systems is a different challenge from more traditional systems. The algorithms and models do not come with explicit specifications of inputs and outputs that can be simply tested and verified. The potential space for both is often quite huge and noisy. Metamorphic testing is one useful technique to gain confidence in their operation with a reasonable effort. Compared to traditional testing techniques, it is not a replacement but rather a complimentary approach.
기계 학습 기반 시스템 테스트는 기존 시스템 테스트와는 다른 도전 과제입니다. 알고리즘과 모델은 명확한 입력 및 출력 명세서를 제공하지 않으며, 간단히 테스트하고 검증할 수 없습니다. 입력과 출력의 잠재적 공간은 매우 크고 복잡하며, 노이즈도 많습니다. 변환 테스트(Metamorphic Testing)는 적절한 노력으로 이러한 시스템의 동작에 대한 신뢰를 얻을 수 있는 유용한 기법입니다. 기존 테스트 기법에 비해 대체가 아닌 보완적인 접근 방식으로, 전통적인 테스트 방법과 함께 사용하여 ML 기반 시스템의 다양한 문제를 더 효과적으로 탐지하고 검증할 수 있습니다.

I presented several examples of applying MMT to different domains in this article. While applications in different domains require different considerations, I believe some generally useful guidelines can be derived to help perform MMT over ML-based systems:
이 글에서 저는 다양한 분야에 MMT를 적용한 여러 사례를 제시했습니다. 각 분야에서의 적용에는 서로 다른 고려 사항이 필요하지만, ML 기반 시스템에 MMT를 효과적으로 수행할 수 있도록 도와줄 몇 가지 일반적인 유용한 지침을 도출할 수 있다고 생각합니다.

metamorphic transformations: these do not have to be hugely complex, but rather simple ones can bring good benefits, such as the addition of a few random points to the LIDAR cloud. Consider how the same input could change in its intended usage environment, and how such change can be implemented with least (or reasonable) effort as a transformation.
변환 관계: 변환은 반드시 복잡할 필요는 없습니다. 예를 들어, LIDAR 클라우드에 몇 개의 랜덤 포인트를 추가하는 것처럼 간단한 변환도 충분한 효과를 가져올 수 있습니다. 동일한 입력이 의도된 사용 환경에서 어떻게 변화할 수 있는지를 고려하고, 이러한 변화를 최소한의(또는 합리적인) 노력으로 구현할 수 있는 변환 방법을 생각해보는 것이 중요합니다.

metamorphic relations: to build these relations, we need to ask how can we change the ML input, and what effect should it have on the output? Sometimes this requires deep domain expertise to identify most relevant changes, as in the medical domain example.
변환 관계: 이러한 관계를 구축하려면 ML 입력을 어떻게 변경할 수 있는지, 그리고 이러한 변경이 출력에 어떤 영향을 미쳐야 하는지를 고민해야 합니다. 이는 때로는 가장 관련성 높은 변화를 식별하기 위해 깊은 도메인 전문 지식이 필요할 수 있습니다. 예를 들어, 의료 분야의 경우, 입력 데이터의 특정 변경이 진단 결과나 모델 예측에 미치는 영향을 이해하기 위해서는 해당 분야의 전문 지식이 필수적입니다.

test oracles: These check that the performed transformation results in a acceptable (vs valid) output. Requires considerations such as how to represent the change (e.g., steering angle change, sentence structural change), possibly defining the probability of some error, the severity of the error, and a distance metric between the potential outputs after transformation (e.g., steering angle calculation). That is, the values are likely not fixed but in a continuous range.
테스트 오라클: 이는 수행된 변환이 허용 가능한(유효한) 출력을 생성했는지 확인합니다. 여기에는 변화의 표현 방법(예: 조향 각도 변화, 문장 구조 변화)과 잠재적 오류의 확률, 오류의 심각도, 변환 후 가능한 출력 간의 거리 측정 방법(예: 조향 각도 계산)과 같은 요소들을 고려해야 합니다. 즉, 출력 값이 고정된 것이 아니라 연속적인 범위 내에서 변화할 가능성이 크다는 점을 인식해야 합니다. 이를 통해 허용 가능한 범위 내에서 시스템의 일관성을 평가할 수 있습니다.

time relation: in many systems, the inputs and outputs are not singular but the overall system performance over time is important. This may also require asking the question of how time might be impacting the system, and how it should be considered in sequences of metamorphic relations. The idea of overall test scenarios as providers of a broader context, time related and otherwise, is useful to consider here.
시간 관계: 많은 시스템에서 입력과 출력이 단일 이벤트가 아닌, 전체 시스템의 시간에 따른 성능이 중요합니다. 이를 위해 시간이 시스템에 어떤 영향을 미치는지, 그리고 일련의 변환 관계에서 이를 어떻게 고려해야 하는지를 질문할 필요가 있습니다. 전체 테스트 시나리오를 통해 더 넓은 맥락(시간 관련 및 기타)을 제공하는 접근 방식을 생각해 보는 것이 유용합니다. 이렇게 하면 시스템이 시간의 흐름에 따라 어떻게 반응하고, 지속적으로 일관된 성능을 유지하는지 평가할 수 있습니다.

test data: can you use the user interactions with the system as an automated source of test inputs for transformations and metamorphic relations? Think Tesla Shadow mode, Google search results, and the inputs from the environment and the user, and use reactions to these inputs.
테스트 데이터: 사용자와 시스템의 상호작용을 자동화된 테스트 입력 소스로 활용하여 변환 및 변환 관계를 평가할 수 있습니다. 예를 들어, 테슬라의 *Shadow Mode*, 구글 검색 결과, 환경 및 사용자의 입력을 생각해 볼 수 있습니다. 이러한 상호작용에서 생성된 입력과 그에 대한 시스템의 반응을 사용하여 테스트를 수행할 수 있습니다. 이러한 데이터는 현실적인 시나리오를 반영하므로, 모델의 실제 성능을 평가하고 개선하는 데 유용한 자원이 될 수 있습니다. 이를 통해 실사용 데이터를 기반으로 시스템이 예상대로 작동하는지, 또는 특정 입력에 대해 예기치 않은 동작을 하는지 효과적으로 검증할 수 있습니다.

As discussed with some of the examples, an interesting trend I see is the move towards using ML-based algorithms to produce or enhance the (MMT-based) tests for ML-based systems. In the NLP domain this is shown by the use of BERT as a tool to build metamorphic transformations for testing natural language translations. In the autonomous cars domain by the use of GAN-based networks to create transformations between image properties, such as different weather elements and time of day.
논의된 예시에서 볼 수 있듯이, ML 기반 시스템을 테스트하기 위해 ML 알고리즘을 사용하거나 개선하는 추세가 흥미롭게 다가옵니다. NLP 분야에서는 BERT를 사용하여 자연어 번역 테스트를 위한 변환 관계를 구축하는 사례가 있으며, 자율 주행 자동차 분야에서는 GAN 기반 네트워크를 활용해 날씨 요소나 시간대와 같은 이미지 속성 간의 변환을 생성하는 경우가 있습니다. 이러한 접근 방식은 ML 모델이 생성한 데이터나 변환을 활용하여 ML 기반 시스템을 더 효과적으로 테스트할 수 있도록 지원합니다. 이는 기존의 테스트 데이터 생성 방식에 비해 더 다양한 시나리오와 조건을 고려할 수 있어 테스트의 범위와 품질을 향상시킬 수 있습니다.

Overall the ML field still seems to be advancing quite fast, with useful approaches already available also for MMT, and hopefully much more mature tooling in the next few years. Without good tool support for testing (data generation, model coverage measurement, etc), finding people with all this expertise (testing, machine learning, domain specifics, …), and implementing it all over again for every system, seems likely to be quite a challenge and sometimes a needlessly high effort without good support in tools and methods.
전반적으로 ML 분야는 여전히 빠르게 발전하고 있으며, 이미 MMT를 위한 유용한 접근법들이 존재하고 앞으로 몇 년 내에 더 성숙한 도구들이 많이 등장할 것으로 기대됩니다. 그러나 이러한 도구 지원(데이터 생성, 모델 커버리지 측정 등)이 부족할 경우, 테스트, 기계 학습, 도메인 전문 지식을 모두 갖춘 인재를 찾고, 각 시스템마다 이를 새로 구현하는 것은 상당한 도전 과제가 될 수 있으며, 종종 불필요하게 많은 노력이 요구될 수 있습니다. 효과적인 도구와 방법론의 지원이 없다면 이러한 작업은 매우 비효율적일 수 있으며, 따라서 보다 성숙하고 쉽게 활용할 수 있는 테스트 도구의 개발이 매우 중요합니다. 이를 통해 ML 기반 시스템의 품질 보증과 신뢰성을 높이는 데 큰 도움이 될 것입니다.
