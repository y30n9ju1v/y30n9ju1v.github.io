+++
title = 'Realistic 3D Simulators for Automotive'
date = 2024-10-17T13:58:55+09:00
draft = true
+++

이 글은 Realistic 3D Simulators for Automotive: A Review of Main Applications and Features (https://www.mdpi.com/1424-8220/24/18/5880)을 번역 및 요약한 글입니다.

## Abstract
Recent advancements in vehicle technology have stimulated innovation across the automotive sector, from Advanced Driver Assistance Systems (ADAS) to autonomous driving and motorsport applications.
Modern vehicles, equipped with sensors for perception, localization, navigation, and actuators for autonomous driving, generate vast amounts of data used for training and evaluating autonomous systems.
Real-world testing is essential for validation but is complex, expensive, and time-intensive, requiring multiple vehicles and reference systems.
To address these challenges, computer graphics-based simulators offer a compelling solution by providing high-fidelity 3D environments to simulate vehicles and road users.
These simulators are crucial for developing, validating, and testing ADAS, autonomous driving systems, and cooperative driving systems, and enhancing vehicle performance and driver training in motorsport.
This paper reviews computer graphics-based simulators tailored for automotive applications.
It begins with an overview of their applications and analyzes their key features.
Additionally, this paper compares five open-source (CARLA, AirSim, LGSVL, AWSIM, and DeepDrive) and ten commercial simulators.
Our findings indicate that open-source simulators are best for the research community, offering realistic 3D environments, multiple sensor support, APIs, co-simulation, and community support.
Conversely, commercial simulators, while less extensible, provide a broader set of features and solutions.

최근 차량 기술의 발전은 자동차 산업 전반에 걸쳐 혁신을 촉진하고 있으며, 이는 첨단 운전자 지원 시스템(ADAS)에서 자율주행 및 모터스포츠 응용 분야에 이르기까지 다양합니다.
최신 차량은 인식, 위치 측정, 내비게이션을 위한 센서와 자율주행을 위한 액추에이터를 갖추고 있으며, 자율 시스템을 훈련하고 평가하는 데 사용되는 방대한 데이터를 생성합니다.
실세계 테스트는 검증을 위해 필수적이지만, 복잡하고 비용이 많이 들며 시간 소모적입니다.
이는 여러 대의 차량과 기준 시스템을 필요로 하기 때문입니다.
이러한 과제를 해결하기 위해, 컴퓨터 그래픽 기반 시뮬레이터는 차량 및 도로 사용자를 시뮬레이션할 수 있는 고충실도의 3D 환경을 제공함으로써 유망한 해결책을 제시합니다.
이러한 시뮬레이터는 ADAS, 자율주행 시스템 및 협력 주행 시스템을 개발, 검증, 테스트하는 데 매우 중요하며, 모터스포츠에서 차량 성능을 향상시키고 운전자 교육을 강화하는 데에도 사용됩니다.
이 논문은 자동차 응용 분야에 맞춘 컴퓨터 그래픽 기반 시뮬레이터를 검토합니다.
먼저 이들의 응용 분야에 대한 개요를 제공하고 주요 기능을 분석합니다.
또한 CARLA, AirSim, LGSVL, AWSIM, DeepDrive와 같은 5개의 오픈소스 시뮬레이터와 10개의 상용 시뮬레이터를 비교합니다.
우리의 연구 결과, 오픈소스 시뮬레이터는 현실적인 3D 환경, 다중 센서 지원, API, 공동 시뮬레이션, 커뮤니티 지원을 제공함으로써 연구 커뮤니티에 가장 적합하다는 것을 나타냅니다.
반면 상용 시뮬레이터는 확장성은 낮지만, 더 폭넓은 기능과 솔루션을 제공합니다.

## 1. Introduction and Related Work
Nowadays, vehicles are equipped with technologies for perception, localization, actuation, and communication.
These include sensors, such as Light Detection and Ranging (LiDAR), radar, cameras, and Global Navigation Satellite System (GNSS).
With the progression of ADAS and autonomous driving technologies, advanced techniques such as sensor fusion are essential for integrating data collected from these sensors.
The ultimate objective is to attain Level 5 autonomous driving capabilities, as defined by the Society of Automotive Engineers (SAE) International standards [1].
Achieving this level of autonomy requires sophisticated algorithms capable of perception, positioning, and decision-making based on processed information.

오늘날 차량은 인식, 위치 측정, 작동 및 통신을 위한 기술을 갖추고 있습니다.
여기에는 라이다(LiDAR), 레이더, 카메라, 위성항법시스템(GNSS)과 같은 센서가 포함됩니다.
ADAS와 자율주행 기술이 발전함에 따라, 이러한 센서로부터 수집된 데이터를 통합하기 위한 고급 기술인 센서 융합이 필수적입니다.
궁극적인 목표는 국제 자동차 기술자 협회(SAE)에서 정의한 Level 5 자율주행 기능을 달성하는 것입니다.
이러한 수준의 자율성을 달성하려면, 처리된 정보를 기반으로 인식, 위치 측정, 의사결정을 수행할 수 있는 고도화된 알고리즘이 필요합니다.

Developing and testing autonomous driving systems or ADAS in real-world conditions poses significant challenges [2,3].
The complexity arises from the need to evaluate numerous scenarios requiring diverse environmental conditions, vehicle types, sensors, and processing techniques (Figure 1).
One of the most challenging applications is the implementation of cooperative driving features, such as cooperative positioning and maneuvering [4,5].
These applications require multiple vehicles, each equipped with advanced reference systems and costly sensors, to be tested effectively in real-world settings.
Another challenge is the development of solutions based on Artificial Intelligence (AI).
AI techniques play a critical role in the algorithms for autonomous driving, particularly in combining multi-sensory information.
However, training AI algorithms demands a substantial amount of data [6], which can be difficult and costly to acquire solely through real-world testing.
Simulated data can provide an effective way for pre-training an AI model, which can then be improved with real-world data.

자율주행 시스템이나 ADAS를 실제 환경에서 개발하고 테스트하는 것은 상당한 도전 과제를 수반합니다 [2,3].
이러한 복잡성은 다양한 환경 조건, 차량 유형, 센서 및 처리 기술을 필요로 하는 수많은 시나리오를 평가해야 한다는 점에서 발생합니다(그림 1).
가장 어려운 응용 중 하나는 협력 주행 기능, 예를 들어 협력적 위치 측정 및 기동의 구현입니다 [4,5].
이러한 응용 프로그램은 고급 기준 시스템과 고가의 센서를 장착한 여러 대의 차량이 실제 환경에서 효과적으로 테스트되어야 합니다.
또 다른 도전 과제는 인공지능(AI)을 기반으로 한 솔루션의 개발입니다.
AI 기술은 특히 다중 센서 정보를 결합하는 자율주행 알고리즘에서 중요한 역할을 합니다.
그러나 AI 알고리즘을 훈련하려면 막대한 양의 데이터가 필요하며 [6], 이를 실제 테스트만으로 확보하는 것은 어렵고 비용이 많이 듭니다.
시뮬레이션된 데이터는 AI 모델을 사전 훈련하는 데 효과적인 방법을 제공할 수 있으며, 이후 실제 데이터를 통해 개선될 수 있습니다.

Simulators based on advanced computer graphics (e.g., game engines) can offer a viable solution for automotive applications by providing realistic 3D environments with visuals and vehicle physics that closely resemble the real world.
They allow for testing under different conditions, including different numbers of vehicles and varying environmental factors such as weather and time of day.
Moreover, these simulators support multiple sensors for perception, safety, and localization.
Using these simulators allows generating large amounts of data that can be used, e.g., for prototyping new sensors, training neural networks or evaluating autonomous driving algorithms.

고급 컴퓨터 그래픽(예: 게임 엔진)을 기반으로 한 시뮬레이터는 현실 세계와 매우 유사한 비주얼과 차량 물리를 제공하는 사실적인 3D 환경을 통해 자동차 응용 분야에 실용적인 해결책을 제시할 수 있습니다.
이러한 시뮬레이터는 다양한 조건에서 테스트를 가능하게 하며, 차량의 수와 날씨, 시간대와 같은 다양한 환경 요소를 조정할 수 있습니다.
또한, 이러한 시뮬레이터는 인식, 안전, 위치 측정을 위한 여러 센서를 지원합니다.
이를 활용하면 새로운 센서 프로토타입 제작, 신경망 훈련 또는 자율주행 알고리즘 평가 등을 위한 대량의 데이터를 생성할 수 있습니다.

A survey proposed by Craighead et al.
[7] analyzed computer-based simulators for unmanned vehicles, including, full-scale and micro-size vehicles, surface and subsurface vehicles, ground vehicles, and aerial vehicles.
An analysis was made for commercially available and open-source simulators.

Craighead 등 [7]이 제안한 설문조사는 무인 차량을 위한 컴퓨터 기반 시뮬레이터를 분석했으며, 여기에는 대형 및 소형 차량, 수상 및 수중 차량, 지상 차량, 공중 차량이 포함되었습니다.
이 연구에서는 상용 시뮬레이터와 오픈소스 시뮬레이터에 대한 분석이 이루어졌습니다.

Rosique et al. [8] present a systematic review of perception systems and simulators for autonomous vehicles.
The paper analyzes the necessary perception systems and sensors, details sensors for accurate positioning like GNSS and sensor fusion techniques, and reviews simulators for autonomous driving, model-based development, game engines, robotics, and those specifically designed for autonomous vehicles.
A survey on simulators for self-driving cars was proposed in [9], comparing Matlab, CarSim, PreScan, Gazebo, and CARLA.
The paper analyzed how well they are at simulating and testing perception, mapping and localization, path planning and vehicle control for self-driving cars.
Similarly, a review of open-source simulators for autonomous driving is proposed in [10].
It categorizes simulators according to their main application, providing researchers with a taxonomy to find simulators more suitable for a particular use.
Therefore, it presents these categories of simulators: traffic flow, sensor data, driving policy, vehicle dynamics, and comprehensive simulator.

Rosique 등 [8]은 자율주행 차량을 위한 인식 시스템과 시뮬레이터에 대한 체계적인 리뷰를 제시합니다.
이 논문은 필수적인 인식 시스템과 센서를 분석하고, GNSS와 같은 정확한 위치 측정을 위한 센서 및 센서 융합 기술을 상세히 설명하며, 자율주행, 모델 기반 개발, 게임 엔진, 로봇 공학, 그리고 자율주행 차량을 위해 특별히 설계된 시뮬레이터를 검토합니다.
또한 [9]에서는 Matlab, CarSim, PreScan, Gazebo, CARLA를 비교하는 자율주행 차량 시뮬레이터에 대한 설문조사가 제안되었습니다.
이 논문은 자율주행 차량의 인식, 매핑 및 위치 측정, 경로 계획 및 차량 제어를 얼마나 잘 시뮬레이션하고 테스트할 수 있는지 분석했습니다.
이와 유사하게 [10]에서는 자율주행을 위한 오픈소스 시뮬레이터에 대한 리뷰가 제안되었으며, 주요 응용 프로그램에 따라 시뮬레이터를 분류하여 연구자들이 특정 용도에 더 적합한 시뮬레이터를 찾을 수 있는 분류 체계를 제공합니다.
따라서 시뮬레이터는 교통 흐름, 센서 데이터, 주행 정책, 차량 동역학, 종합 시뮬레이터의 범주로 나뉘어 제시됩니다.

This paper stands out from previous studies by specifically focusing on 3D computer graphics-based simulators for automotive applications.
The main contributions of this work are: a discussion of the main applications and requirements of the simulators for automotive contexts; a comprehensive compilation of five open-source [11–19] and ten commercial simulators [20–29]; a detailed analysis of the features and sensors supported by these simulators.

이 논문은 자동차 응용 분야를 위한 3D 컴퓨터 그래픽 기반 시뮬레이터에 특별히 집중함으로써 이전 연구와 차별화됩니다.
본 연구의 주요 기여는 다음과 같습니다: 자동차 분야에서 시뮬레이터의 주요 응용 및 요구 사항에 대한 논의; 5개의 오픈소스 시뮬레이터 [11-19]와 10개의 상용 시뮬레이터 [20-29]를 종합적으로 정리한 자료; 이러한 시뮬레이터가 지원하는 기능과 센서에 대한 상세한 분석.

## 2. Computer Graphics Simulators for Automotive: Requirements and Applications
As depicted in Figure 1, computer graphics simulators for automotive applications, particularly in autonomous driving, require two main components: realistic 3D environments (urban, suburban, highway) with varied conditions (weather, lighting, traffic), and accurate vehicle physics/motion simulation. This includes sensor models for perception, navigation, and positioning, essential for autonomous driving. Simulators should also support Vehicle-to-Everything (V2X) communication for vehicle interactions and employ traffic management systems for vehicle mobility. While computer graphics simulators excel in rendering realistic 3D environments and physics, they often lack V2X capabilities and complex traffic and mobility models. Dedicated simulators focusing on traffic, mobility, and V2X communications can complement computer graphics simulators through co-simulation.

그림 1에서 보여지듯이, 자율주행을 포함한 자동차 응용 분야를 위한 컴퓨터 그래픽 시뮬레이터는 두 가지 주요 구성 요소를 필요로 합니다: 다양한 조건(날씨, 조명, 교통)을 포함한 현실적인 3D 환경(도시, 교외, 고속도로)과 정확한 차량 물리/동작 시뮬레이션입니다. 여기에 자율주행에 필수적인 인식, 내비게이션 및 위치 측정을 위한 센서 모델이 포함됩니다. 또한, 시뮬레이터는 차량 간 상호작용을 위한 V2X(Vehicle-to-Everything) 통신을 지원하고, 차량 이동성을 관리하기 위한 교통 관리 시스템도 필요합니다. 컴퓨터 그래픽 시뮬레이터는 사실적인 3D 환경과 물리 구현에서 뛰어나지만, V2X 기능 및 복잡한 교통과 이동성 모델이 부족한 경우가 많습니다. 교통, 이동성 및 V2X 통신에 중점을 둔 전용 시뮬레이터는 컴퓨터 그래픽 시뮬레이터와의 공동 시뮬레이션을 통해 이러한 부족한 부분을 보완할 수 있습니다.

Computer graphics-based simulators offer a plethora of features for automotive applications by providing a realistic 3D environment, which allows supporting different types of technologies and sensors. The main applications of these simulators are as follows:

* Autonomous Driving and ADAS: prototyping, development and evaluation of autonomous driving systems [30–32].
Simulated data can be used for the development of sensors, new algorithms, and sensor fusion techniques.
Similarly, with autonomous driving, ADAS systems benefit from having a simulation tool that provides a multitude of scenarios in which advanced driver assistance features can be developed.
* AI: Simulators generate data, which is essential for developing new methods and training AI techniques for autonomous vehicles [30,33,34].
* Cooperative Driving: vehicles operating cooperatively, i.e., exchanging sensor data to enable cooperative positioning, perception, awareness, or cooperative maneuvering [4,35,36].
* V2X: communication between vehicles and other entities, which is essential for cooperative driving (maneuvering, perception, or positioning) [15,17].
* Motorsport: improve vehicle development (aerodynamics, chassis systems, steering systems, etc.), improve testing efficiency, and driver training [27,37,38].

컴퓨터 그래픽 기반 시뮬레이터는 현실적인 3D 환경을 제공하여 다양한 기술과 센서를 지원함으로써 자동차 응용 분야에 수많은 기능을 제공합니다. 이러한 시뮬레이터의 주요 응용 분야는 다음과 같습니다:

- **자율주행 및 ADAS**: 자율주행 시스템의 프로토타입 개발, 평가에 사용됩니다 [30–32]. 시뮬레이션 데이터를 통해 센서 개발, 새로운 알고리즘 및 센서 융합 기술 개발에 활용될 수 있으며, ADAS 시스템도 다양한 시나리오를 제공하는 시뮬레이션 도구를 통해 고급 운전자 지원 기능을 개발하는 데 도움을 줍니다.
- **인공지능(AI)**: 시뮬레이터는 데이터를 생성하여 자율주행 차량을 위한 새로운 방법과 AI 기술을 개발하고 훈련하는 데 필수적입니다 [30,33,34].
- **협력 주행**: 차량 간 센서 데이터를 교환하여 협력적 위치 측정, 인식, 상황 인지 또는 협력적 기동을 가능하게 합니다 [4,35,36].
- **V2X**: 차량과 다른 엔터티 간의 통신은 협력 주행(기동, 인식 또는 위치 측정)에 필수적입니다 [15,17].
- **모터스포츠**: 차량 개발(공기역학, 섀시 시스템, 조향 시스템 등)을 개선하고, 테스트 효율성과 운전자 교육을 향상시킵니다 [27,37,38].

## 3. Main Simulator Features
In this section, an analysis is made of the main features provided by computer graphics simulators for automotive applications.
Although each simulator is unique, with differentiated functionality, they share some features that are described next.

이 섹션에서는 자동차 응용 분야를 위한 컴퓨터 그래픽 시뮬레이터가 제공하는 주요 기능을 분석합니다.
각 시뮬레이터는 고유한 특성과 차별화된 기능을 가지고 있지만, 공통적으로 다음과 같은 기능을 공유합니다.

### 3.1. Open-Source vs. Closed-Source
Commercial simulators are typically closed-source, which limits their extensibility and restricts access to the code.
Also, these simulators are paid, which adds to the financial cost.
In contrast, open-source simulators offer significant advantages: they are freely accessible, fostering ease of extensibility and benefiting from community support for bug fixes.
Furthermore, the open nature of their source code enhances reproducibility, enabling researchers to validate and build upon each other’s findings more effectively.
Henceforth, we will compare commercial and open-source simulators, with a particular focus on the open-source ones.

상용 시뮬레이터는 일반적으로 소스 코드가 비공개되어 있어 확장성이 제한되고 코드 접근이 어렵습니다.
또한, 이러한 시뮬레이터는 유료이므로 추가적인 재정적 부담이 발생합니다.
반면, 오픈소스 시뮬레이터는 상당한 장점을 제공합니다.
무료로 접근할 수 있어 확장성이 용이하며, 커뮤니티 지원을 통해 버그 수정이 이루어집니다.
또한, 소스 코드가 공개되어 있어 재현성이 향상되며, 연구자들이 서로의 연구 결과를 검증하고 기반을 쌓아나가는 데 더 효과적입니다. 따라서, 우리는 상용 시뮬레이터와 오픈소스 시뮬레이터를 비교할 것이며, 특히 오픈소스 시뮬레이터에 중점을 둘 것입니다.

### 3.2. Game Engine
Simulators for automotive applications are typically built as extensions of already existing game engines.
The most well-known are the Unreal Engine and Unity.
These engines provide frameworks to facilitate game development, including rendering, physics and scripting.
In addition, they have advanced graphics capabilities, supporting 3D development, hence they are suitable for simulating automotive scenarios, with a realistic environment for vehicles, including roads and an actor’s interaction between vehicles and other road users.

자동차 응용 분야를 위한 시뮬레이터는 일반적으로 기존에 존재하는 게임 엔진의 확장으로 구축됩니다.
가장 잘 알려진 것은 언리얼 엔진(Unreal Engine)과 유니티(Unity)입니다.
이러한 엔진은 렌더링, 물리 및 스크립팅을 포함한 게임 개발을 용이하게 하는 프레임워크를 제공합니다.
또한, 이들 엔진은 3D 개발을 지원하는 고급 그래픽 기능을 갖추고 있어, 자동차 시나리오를 사실적인 환경에서 시뮬레이션하는 데 적합합니다.
여기에는 도로를 포함한 차량 환경과 차량 및 기타 도로 사용자 간의 상호작용이 포함됩니다.

### 3.3. Supported Sensors
Most modern vehicles are equipped with several sensors, especially the ones with ADAS and autonomous driving capabilities.
Various sensors gather data about the vehicle’s surroundings and internal state, for perception, localization (absolute and relative), safety and navigation.
In the following, a list of commonly found sensors, supported by computer graphics simulators, is presented.

대부분의 현대 차량, 특히 ADAS 및 자율주행 기능이 있는 차량에는 여러 센서가 장착되어 있습니다.
이러한 다양한 센서는 차량의 주변 환경과 내부 상태에 대한 데이터를 수집하여 인식, 위치 측정(절대 및 상대), 안전, 내비게이션에 활용됩니다.
다음은 컴퓨터 그래픽 시뮬레이터에서 지원되는 일반적인 센서 목록입니다.

#### 3.3.1. GNSS
GNSS provides absolute positioning by using a constellation of satellites and a receiver within the vehicle to estimate its position.
Computer graphics simulators have simplistic models for GNSS, usually providing the estimated positions using a simple noise model, like additive white Gaussian noise.
In automotive applications, GNSS is often combined with relative positioning techniques, like dead reckoning, to improve the accuracy.

GNSS는 위성 군과 차량 내 수신기를 사용하여 차량의 위치를 추정하는 방식으로 절대 위치 측정을 제공합니다.
컴퓨터 그래픽 시뮬레이터에서는 GNSS에 대한 단순한 모델을 사용하며, 보통 가산 백색 가우시안 잡음과 같은 간단한 잡음 모델을 통해 추정된 위치를 제공합니다.
자동차 응용 분야에서는 GNSS의 정확도를 향상시키기 위해 종종 상대 위치 측정 기법, 예를 들어 항법(dead reckoning)과 결합하여 사용됩니다.

#### 3.3.2. Inertial Measurement Unit (IMU)
IMUs are self-contained systems with a tri-axis accelerometer, gyroscope, and magnetometer to measure acceleration, angular velocity, and magnetic field.
IMUs usually perform on-board processing, combining raw data from all sensors into the estimation of the device’s attitude.
The representation of the device’s attitude (or orientation) can be provided in Euler angles or in quaternion form.

IMU(관성 측정 장치)는 3축 가속도계, 자이로스코프, 자력계를 포함한 독립적인 시스템으로, 가속도, 각속도, 자기장을 측정합니다.
IMU는 보통 온보드 처리를 수행하여 모든 센서에서 수집한 원시 데이터를 결합하여 장치의 자세를 추정합니다.
장치의 자세(또는 방향)는 오일러 각(Euler angles) 또는 쿼터니언(quaternion) 형식으로 제공될 수 있습니다.

#### 3.3.3. Encoder (Distance)
Encoders measure rotation angle or linear displacement and are often used as odometers to measure the traveled distance.

엔코더는 회전 각도 또는 선형 변위를 측정하며, 종종 주행 거리를 측정하는 주행계(odometer)로 사용됩니다.

#### 3.3.4. Light Detection and Ranging
LiDARs create a 3D map of its surroundings using a laser and a receiver.
It works by emitting a short laser pulse and recording the time it takes for the pulse to be reflected.
This allows the conversion of time into a target distance measurement, providing a 3D representation of the surrounding environment with high-resolution point clouds, performed, for example, in 3D mapping in outdoor environments.

라이다(LiDAR)는 레이저와 수신기를 사용하여 주변 환경의 3D 지도를 생성합니다.
짧은 레이저 펄스를 방출하고, 펄스가 반사되어 돌아오는 시간을 기록함으로써, 시간을 목표 거리 측정으로 변환할 수 있습니다.
이를 통해 고해상도의 포인트 클라우드로 주변 환경의 3D 표현을 제공하며, 예를 들어 야외 환경의 3D 맵핑에서 주로 사용됩니다.

#### 3.3.5. Radar
Radar sensors use radio waves to measure the distance, angle, and velocity of objects.
In automotive applications, radar can be used for numerous purposes, such as adaptive cruise control; collision warning and avoidance; automatic emergency brake; and blind spot detection.
In autonomous driving scenarios, radar sensors are essential to reliably detect objects and people and avoid collisions.

레이더 센서는 전파를 사용하여 물체의 거리, 각도, 속도를 측정합니다.
자동차 응용 분야에서 레이더는 적응형 크루즈 컨트롤, 충돌 경고 및 회피, 자동 긴급 제동, 사각지대 감지 등 다양한 목적으로 사용될 수 있습니다.
자율주행 시나리오에서는 레이더 센서가 물체와 사람을 신뢰성 있게 감지하고 충돌을 피하는 데 필수적입니다.

#### 3.3.6. Ultrasound
Ultrasound sensors use high-frequency sound waves to detect objects by measuring the time it takes for the sound waves to return after hitting an object.
Then, the distance to the object is calculated based on the speed of sound.
Ultrasound sensors are inexpensive when compared with radar and LiDAR but are limited to short-range operation; hence, they are primarily used in short-range detection applications.

초음파 센서는 고주파 음파를 사용하여 물체를 감지하며, 음파가 물체에 부딪힌 후 돌아오는 시간을 측정하여 물체까지의 거리를 계산합니다.
이때 거리는 음속을 기반으로 계산됩니다.
초음파 센서는 레이더나 라이다에 비해 저렴하지만, 짧은 거리에서만 작동할 수 있다는 한계가 있습니다.
따라서 주로 근거리 감지가 필요한 응용 분야에서 사용됩니다.

#### 3.3.7. Cameras
Cameras capture images of the environment and can be installed in several parts of the vehicles to assist in parking and to assist in autonomous navigation.
They can be of different types and have different purposes, namely: RGB cameras capture color images and are used for object detection and recognition, aiding in tasks like lane keeping and traffic sign recognition; depth cameras provide 3D data about the surroundings, essential for obstacle detection and autonomous navigation; Infrared (IR) cameras enable night vision and thermal imaging, improving visibility in low-light conditions and detecting pedestrians or animals; segmentation cameras use advanced algorithms to distinguish different elements in a scene, such as vehicles, pedestrians, and road markings, facilitating autonomous driving and ADAS; optical flow cameras detect and quantify the movement of objects in a scene by analyzing changes in pixel intensity over time; event cameras, also known as Dynamic Vision Sensors (DVSs), capture changes in a scene with high temporal resolution, enabling efficient motion detection and tracking.
The main drawbacks of cameras are their sensitivity to low-light environments, adverse weather conditions, and privacy concerns.

카메라는 주변 환경의 이미지를 캡처하며, 주차 보조 및 자율 주행을 지원하기 위해 차량의 여러 부분에 설치될 수 있습니다.
카메라는 여러 종류가 있으며, 각각 다른 목적을 가지고 있습니다.
RGB 카메라는 컬러 이미지를 캡처하여 물체 감지 및 인식을 위해 사용되며, 차선 유지나 교통 표지 인식과 같은 작업에 도움을 줍니다.
깊이 카메라는 주변 환경에 대한 3D 데이터를 제공하여 장애물 감지 및 자율 주행에 필수적입니다.
적외선(IR) 카메라는 야간 투시 및 열 이미지를 제공하여 어두운 환경에서 가시성을 개선하고 보행자나 동물을 감지하는 데 유용합니다.
분할(segmentation) 카메라는 차량, 보행자, 도로 표식 등 장면의 다양한 요소를 구분하는 고급 알고리즘을 사용하여 자율 주행 및 ADAS를 지원합니다.
광학 흐름(optical flow) 카메라는 픽셀 강도의 변화를 분석하여 장면 내 물체의 움직임을 감지하고 측정합니다.
이벤트 카메라, 또는 동적 비전 센서(DVS)는 높은 시간 해상도로 장면의 변화를 캡처하여 효율적인 움직임 감지 및 추적을 가능하게 합니다.
카메라의 주요 단점은 저조도 환경, 악천후에 대한 민감성, 그리고 프라이버시 우려입니다.

### 3.4. SIL and HIL
Software-In-the-Loop (SIL) enables the development of software components, allowing to test them in isolation from the hardware.
Simulators supporting SIL usually have a simulated Electronic Control Unit (ECU), including its software components as well as simulated sensor and actuator models, as the replacement of real hardware.

Software-In-the-Loop(SIL)은 소프트웨어 구성 요소를 개발하고, 이를 하드웨어와 분리된 상태에서 테스트할 수 있도록 합니다. SIL을 지원하는 시뮬레이터는 일반적으로 실제 하드웨어를 대체하는 방식으로 소프트웨어 구성 요소를 포함한 가상 전자 제어 장치(ECU)와 가상 센서 및 액추에이터 모델을 제공합니다.

Hardware-In-the-Loop (HIL) involves evaluating the interaction between software and hardware components in a simulated environment.
It allows for detecting and debugging hardware–software integration issues early in the development cycle.
For instance, HIL enables testing of real ECUs in a realistic simulated setting.
HIL tests are reproducible and can be automated, speeding up validation and testing processes.

Hardware-In-the-Loop(HIL)은 소프트웨어와 하드웨어 구성 요소 간의 상호작용을 시뮬레이션 환경에서 평가하는 방식입니다. 이를 통해 개발 초기 단계에서 하드웨어-소프트웨어 통합 문제를 감지하고 디버깅할 수 있습니다. 예를 들어, HIL은 실제 ECU를 현실감 있는 시뮬레이션 환경에서 테스트할 수 있게 해줍니다. HIL 테스트는 재현 가능하며 자동화할 수 있어 검증 및 테스트 프로세스를 가속화합니다.

Both SIL and HIL facilitate the evaluation of critical corner cases within a controlled environment.

SIL과 HIL 모두 통제된 환경에서 중요한 코너 케이스를 평가하는 데 유용합니다.

### 3.5. Co-Simulation
A simulator that supports co-simulation means it can be coupled with other simulation tools, e.g., tools that generate traffic and mobility [39–41], tools that support V2X communications [42,43], or autonomous driving stacks, such as Autoware [44] or Baidu Apollo [45].
This allows extending the capabilities of the computer graphics simulator to support new features that were not previously supported.

공동 시뮬레이션을 지원하는 시뮬레이터는 교통 및 이동성을 생성하는 도구 [39-41], V2X 통신을 지원하는 도구 [42,43], 또는 Autoware [44]나 Baidu Apollo [45]와 같은 자율주행 스택과 연동될 수 있습니다.
이를 통해 컴퓨터 그래픽 시뮬레이터의 기능을 확장하여 이전에 지원되지 않았던 새로운 기능을 구현할 수 있습니다.

### 3.6. ROS Integration
Robotic Operating System (ROS) is a framework for developing, testing, and deploying robotic systems.
It offers a standardized communications module with the publish/subscribe model.
By having ROS integration, the simulator is capable of interacting with the ROS modules to implement features such as sensor fusion for positioning, SLAM, navigation, and perception.

로봇 운영 체제(ROS)는 로봇 시스템을 개발, 테스트 및 배포하기 위한 프레임워크입니다.
ROS는 발행/구독 모델을 사용하는 표준화된 통신 모듈을 제공합니다.
시뮬레이터가 ROS 통합을 지원하면 ROS 모듈과 상호작용하여 위치 측정을 위한 센서 융합, SLAM(동시적 위치추정 및 지도작성), 내비게이션, 인식과 같은 기능을 구현할 수 있습니다.

### 3.7. Hardware Specifications
Since simulators are built on game engines, they require significant computing power due to their demands for rendering 3D graphics and simulating physics (moving objects in simulation, collision detection, gravity and other interactions).
AI systems within the engine help create Non-Player Character (NPC) behaviors and other intelligent behaviors of road actors.
Typically, the documentation for these tools provides detailed minimum and recommended system requirements, especially the Central Processing Unit (CPU) and Graphics Processing Unit (GPU) requirements.
Regarding the CPU, a multi-core processor is usually required as it provides better multi-tasking capabilities, running multiple tasks in parallel.
The CPU clock speed, usually defined in GHz, is also a requirement, and higher clock speeds improve performance.
A dedicated (discrete) GPU is essential for rendering complex graphics and high-resolution textures.
Some simulators include requirements for Application Programming Interfaces (APIs) like DirectX, OpenGL, or Vulkan, as these are fundamental for interacting with the GPU to create visual effects, handle complex calculations, and manage hardware resources effectively.
Additionally, some simulators include RAM and disk storage requirements, as these tools have numerous assets that require both RAM and disk space to run properly.

시뮬레이터는 게임 엔진을 기반으로 구축되기 때문에 3D 그래픽 렌더링과 물리 시뮬레이션(시뮬레이션 내의 물체 이동, 충돌 감지, 중력 및 기타 상호작용)을 처리하는 데 상당한 컴퓨팅 성능이 필요합니다.
엔진 내의 AI 시스템은 비플레이어 캐릭터(NPC) 동작 및 도로 사용자들의 지능적인 동작을 생성하는 데 도움을 줍니다.
이러한 도구의 문서에는 보통 중앙처리장치(CPU)와 그래픽처리장치(GPU)에 대한 최소 및 권장 시스템 요구 사항이 자세히 설명되어 있습니다.
CPU와 관련해서는 멀티태스킹 기능이 뛰어나며 여러 작업을 병렬로 실행할 수 있는 멀티코어 프로세서가 보통 요구됩니다.
CPU의 클록 속도는 보통 GHz로 정의되며, 더 높은 클록 속도는 성능 향상에 기여합니다.
복잡한 그래픽과 고해상도 텍스처를 렌더링하기 위해서는 전용(외장형) GPU가 필수입니다.
또한, 일부 시뮬레이터는 DirectX, OpenGL 또는 Vulkan과 같은 응용 프로그래밍 인터페이스(API)를 요구합니다.
이러한 API는 GPU와 상호작용하여 시각적 효과를 생성하고 복잡한 계산을 처리하며, 하드웨어 자원을 효율적으로 관리하는 데 필수적입니다.
추가로, 시뮬레이터는 RAM 및 디스크 저장 용량에 대한 요구 사항을 포함하기도 합니다.
이는 시뮬레이터에 필요한 수많은 자산을 원활하게 실행하기 위해 충분한 RAM과 디스크 공간이 필요하기 때문입니다.

## 4. Overview of Selected Simulators
In this section, we present the simulators selected for analysis in this paper.
The primary selection criteria were simulators’ ability to provide realistic 3D environments for automotive applications using game engines, creating high-fidelity environments for autonomous driving development and testing.
Both open-source and commercial options were considered to provide a holistic view of the available tools, particularly because open-source simulators are widely used both in academic and industrial research.
They are free to use and offer open code, which can be extended, adapted, and supported by the community.
Hence, open-source simulators are a suitable option for academic researchers and small teams with limited resources but are also used by the industry.
Open-source simulators were identified through a literature review [8–10].
Commercial simulators were selected because they are more readily available and offer greater diversity in terms of functionality.
They also have extended support, including patches and updates that enhance their reliability.
However, these simulators are paid solutions and can be quite expensive, making them more accessible to industry, which typically has larger budgets, compared to academic researchers.
Selected commercial simulators were found via literature review [7–10] and web search, in which we prioritized those offering more functionality according to the requirements and applications listed in Section 2.

이 섹션에서는 본 논문에서 분석할 시뮬레이터를 소개합니다.
주요 선정 기준은 게임 엔진을 사용하여 자동차 응용 분야에 현실적인 3D 환경을 제공하고, 자율주행 개발 및 테스트를 위한 고충실도의 환경을 생성할 수 있는지 여부였습니다.
오픈소스와 상용 옵션 모두를 고려하여 사용 가능한 도구에 대한 전체적인 관점을 제공하고자 했습니다.
특히 오픈소스 시뮬레이터는 학계와 산업 연구에서 널리 사용되고 있기 때문에 중요한 선택지로 고려되었습니다.
오픈소스 시뮬레이터는 무료로 사용할 수 있으며, 커뮤니티에서 확장, 수정 및 지원할 수 있는 오픈 코드를 제공하므로, 학술 연구자 및 제한된 자원을 가진 소규모 팀에게 적합한 옵션입니다.
그러나 산업에서도 오픈소스 시뮬레이터를 활용하고 있습니다.
오픈소스 시뮬레이터는 문헌 검토 [8–10]를 통해 확인되었습니다.
상용 시뮬레이터는 더 많은 기능을 제공하고, 패치 및 업데이트를 통해 신뢰성을 높이는 확장된 지원이 있기 때문에 선택되었습니다.
하지만 이러한 시뮬레이터는 유료이며, 상당히 비쌀 수 있어 학술 연구자들보다는 더 큰 예산을 가진 산업체에 더 적합합니다.
선정된 상용 시뮬레이터는 문헌 검토 [7–10]와 웹 검색을 통해 찾았으며, 섹션 2에서 나열된 요구 사항과 응용 프로그램에 따라 더 많은 기능을 제공하는 시뮬레이터를 우선적으로 선택했습니다.

The following subsections detail open-source simulators, highlighting their purposes, applications and main features.
These are described in greater detail due to their extensibility and reusability by the research community.
The last subsection introduces commercial simulators.

다음 세부 섹션에서는 오픈소스 시뮬레이터의 목적, 응용 분야 및 주요 기능을 중점적으로 설명합니다.
이러한 시뮬레이터들은 연구 커뮤니티에서 확장성과 재사용 가능성이 높기 때문에 더 자세히 다루어집니다.
마지막 섹션에서는 상용 시뮬레이터를 소개합니다.

### 4.1 CARLA
CARLA [11,12] (Figure 2) is an open-source simulator that was built specifically for autonomous driving research.
It was first proposed in 2017 and is still under development with community support and feature updates.
Being based on the Unreal Engine, CARLA provides a realistic 3D environment with dynamic traffic, pedestrians, and various weather conditions.
The autonomous driving sensor suite provides several sensors that users can configure, e.g., LiDAR, cameras, radar, GNSS, and IMU, among others.
To interact with the tool, CARLA provides an easy-to-use Python API for defining custom scenarios, controlling vehicles, and accessing sensor data.

CARLA [11,12]는 자율주행 연구를 위해 특별히 개발된 오픈소스 시뮬레이터입니다.
2017년에 처음 제안되었으며, 현재도 커뮤니티 지원과 기능 업데이트를 통해 계속 개발 중입니다.
CARLA는 언리얼 엔진을 기반으로 하여 사실적인 3D 환경을 제공하며, 동적인 교통, 보행자, 다양한 날씨 조건을 시뮬레이션할 수 있습니다.
자율주행 센서 스위트는 LiDAR, 카메라, 레이더, GNSS, IMU 등 여러 센서를 사용자가 구성할 수 있도록 지원합니다.
CARLA는 Python API를 통해 사용자에게 사용자 지정 시나리오 정의, 차량 제어, 센서 데이터 접근 등의 기능을 쉽게 사용할 수 있도록 제공합니다.

CARLA supports co-simulation, i.e., it can be used with other simulators.
It has native support for Simulation of Urban MObility (SUMO) [40], VISSIM [41], and CarSim [25].
SUMO and VISSIM are traffic and mobility simulators, which allows for managing traffic, while still being inside CARLA’s virtual environment.
The integration with CarSim [25] allows vehicle controls in CARLA to be forwarded to CarSim.
There are also custom co-simulation packages, developed by the research community.
For instance, ref. [46] enhances CARLA with V2X capabilities, and ref. [35] improves traffic and mobility.
ROS integration is enabled by a bridge that enables two-way communication between ROS and CARLA.
Another important aspect of CARLA is that it has an active community on GitHub providing help not only in solving bugs and identified issues but also providing help on how to use the tool.

CARLA는 공동 시뮬레이션을 지원하며, 다른 시뮬레이터와 함께 사용할 수 있습니다.
CARLA는 Simulation of Urban MObility(SUMO) [40], VISSIM [41], CarSim [25]에 대한 네이티브 지원을 제공합니다.
SUMO와 VISSIM은 교통 및 이동성 시뮬레이터로, CARLA의 가상 환경 내에서 교통을 관리할 수 있게 해줍니다.
CarSim과의 통합을 통해 CARLA에서의 차량 제어를 CarSim으로 전달할 수 있습니다.
또한, 연구 커뮤니티에서 개발된 맞춤형 공동 시뮬레이션 패키지도 존재합니다.
예를 들어, 참고문헌 [46]은 CARLA에 V2X 기능을 추가하고, 참고문헌 [35]는 교통 및 이동성을 개선합니다.
ROS 통합은 ROS와 CARLA 간의 양방향 통신을 가능하게 하는 브리지를 통해 활성화됩니다.
CARLA의 또 다른 중요한 측면은 GitHub에서 활발한 커뮤니티를 보유하고 있어 버그 해결 및 사용 방법에 대한 도움을 제공한다는 점입니다.

Being open-source and still actively supported and developed makes CARLA one of the most used open-source simulators by the research community to support cooperative perception [35], cooperative positioning [47] using LiDAR, sensor fusion applications with V2X capabilities [48], and HIL autonomous driving simulation [49].

CARLA는 오픈소스이며 여전히 활발히 지원 및 개발되고 있어, 연구 커뮤니티에서 가장 많이 사용되는 오픈소스 시뮬레이터 중 하나입니다.
CARLA는 협력적 인식 [35], LiDAR를 이용한 협력적 위치 측정 [47], V2X 기능을 포함한 센서 융합 응용 [48], HIL 자율주행 시뮬레이션 [49]을 지원하는 데 널리 활용되고 있습니다.

### 4.2 AirSim
AirSim [13,14] (Figure 3) was first introduced in 2017.
It is an open-source simulator for urban environments, with realistic physics and visual rendering for drones, cars, and other vehicles.
It differentiates from other simulators, since no other simulates aerial vehicles as well as land vehicles.
This tool was developed by Microsoft Research using Unreal Engine as the rendering platform but also has an experimental version running with Unity.
Supported sensors are a camera (RGB, infrared, optical flow), barometer, IMU, Global Positioning System (GPS), magnetometer, distance and LiDAR.
Users can interact with the tool via the provided APIs in Python and C++, as well as the ROS wrapper.
Another distinctive feature of AirSim is that it supports SIL and HIL, using gaming and flight controllers.
In SIL mode, the algorithms development can be achieved without needing physical hardware.
Conversely, in HIL mode, physical hardware can be evaluated within the simulation environment, as the simulator interfaces with the device hardware to obtain actuator signals.

AirSim [13,14]은 2017년에 처음 도입된 오픈소스 시뮬레이터로, 드론, 자동차 및 기타 차량을 위한 도시 환경에서 현실적인 물리적 동작과 시각적 렌더링을 제공합니다.
다른 시뮬레이터와의 차별점은 지상 차량뿐만 아니라 항공 차량도 시뮬레이션할 수 있다는 점입니다.
이 도구는 Microsoft Research에서 언리얼 엔진을 렌더링 플랫폼으로 개발했으며, Unity를 사용하는 실험적인 버전도 있습니다.
지원되는 센서로는 카메라(RGB, 적외선, 광학 흐름), 기압계, IMU, GPS, 자력계, 거리 측정기, LiDAR가 있습니다. 사용자는 Python과 C++로 제공되는 API뿐만 아니라 ROS 래퍼를 통해 도구와 상호작용할 수 있습니다.
AirSim의 또 다른 독특한 기능은 SIL(Software-In-the-Loop)과 HIL(Hardware-In-the-Loop)을 모두 지원한다는 점입니다.
SIL 모드에서는 물리적 하드웨어 없이 알고리즘 개발이 가능하고, HIL 모드에서는 시뮬레이터가 장치 하드웨어와 인터페이스하여 액추에이터 신호를 받아내어 물리적 하드웨어를 시뮬레이션 환경에서 평가할 수 있습니다.

Unfortunately Microsoft Research terminated this project in 2022, ending further development and support.
Their decision comes from the need to focus on the development of a new product, called Project AirSim, a commercial product that provides an end-to-end platform for developing and testing aerial autonomy through simulation.
Despite that, the code for AirSim is still openly available online [50].

안타깝게도 Microsoft Research는 2022년에 이 프로젝트를 종료하면서 AirSim의 추가 개발과 지원을 중단했습니다.
이 결정은 새로운 상업용 제품인 **Project AirSim** 개발에 집중하기 위한 것으로, 이 제품은 시뮬레이션을 통해 항공 자율성 개발 및 테스트를 위한 엔드투엔드 플랫폼을 제공합니다.
그럼에도 불구하고 AirSim의 코드는 여전히 온라인에서 공개적으로 이용 가능합니다 [50].

AirSim has been used by the research community as a simulation framework, e.g., for cooperative autonomous driving using 6G V2X [51], for ADAS, with a collision avoidance system [52], and for autonomous driving based on reinforced learning [53].

AirSim은 연구 커뮤니티에서 시뮬레이션 프레임워크로 널리 사용되었습니다.
예를 들어, **6G V2X**를 사용한 협력 자율주행 [51], 충돌 회피 시스템을 갖춘 **ADAS** [52], 그리고 강화 학습을 기반으로 한 자율주행 [53]과 같은 다양한 연구 분야에서 활용되었습니다.

### 4.3 LGSVL
LGSVL [15,16] (Figure 4) was developed aiming to improve autonomous vehicle de- velopment with high-fidelity simulation, exploring the Unity engine.
Similarly to CARLA and AirSim, LGSVL replicates the complexity of real-world environments with the simu- lated environment, supporting sensors such as cameras, radar, LiDAR, GPS, IMU, among others.
Supported features include SIL, HIL and ROS, as well as other integration options, particularly a communication channel that enables communication between the simulator and an autonomous driving stack, like Autoware [44] or Baidu Apollo [45].
This project was discontinued on January 1st, 2022, and no further updates or fixes are planned.
Despite that, the code repository is also available online in [15].

LGSVL [15,16]은 Unity 엔진을 활용하여 고충실도의 시뮬레이션을 통해 자율주행 차량 개발을 개선하기 위해 개발되었습니다.
CARLA 및 AirSim과 유사하게, LGSVL은 시뮬레이션 환경을 통해 실제 환경의 복잡성을 재현하며, 카메라, 레이더, LiDAR, GPS, IMU 등의 센서를 지원합니다.
지원되는 기능에는 SIL(Software-In-the-Loop), HIL(Hardware-In-the-Loop), ROS 통합과 자율주행 스택(예: Autoware [44] 또는 Baidu Apollo [45])과의 통신 채널을 통한 상호작용이 포함됩니다.
이 프로젝트는 2022년 1월 1일에 중단되었으며, 추가 업데이트나 수정은 예정되어 있지 않습니다.
그럼에도 불구하고 코드 저장소는 여전히 온라인에서 사용할 수 있습니다 [15].

LGSVL has been used by the research community for simulation and evaluation of autonomous driving systems.
For example, in [54], a roadside-assisted cooperative planning solution developed in Autoware was evaluated using LGSVL.
In [55], LGSVL was used to evaluate a camera-based perception system for autonomous driving.
LGSVL was also used to evaluate a system that performs behavior monitoring of autonomous vehicles to detect safety violations [56].
This is crucial to ensuring the reliability and safety of autonomous driving systems.

LGSVL은 자율주행 시스템의 시뮬레이션 및 평가를 위해 연구 커뮤니티에서 널리 사용되었습니다.
예를 들어, 참고문헌 [54]에서는 Autoware에서 개발된 도로 측 보조 협력 계획 솔루션이 LGSVL을 통해 평가되었습니다.
참고문헌 [55]에서는 자율주행을 위한 카메라 기반 인식 시스템이 LGSVL을 사용하여 평가되었습니다.
또한, LGSVL은 자율주행 차량의 안전 위반을 감지하기 위한 동작 모니터링 시스템을 평가하는 데도 사용되었습니다 [56].
이는 자율주행 시스템의 신뢰성과 안전성을 보장하는 데 중요한 요소입니다.

### 4.4 AWSIM

