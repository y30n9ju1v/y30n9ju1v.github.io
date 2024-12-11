+++
title = 'Performance evaluation of a ROS2 based Automated Driving System'
date = 2024-12-11T09:58:30+09:00
draft = false
+++

이 글은 Performance evaluation of a ROS2 based Automated Driving System (https://arxiv.org/abs/2411.11607)을 번역 및 요약한 글입니다.

## Abstract
Automated driving is currently a prominent area of scientific work.
자율 주행은 현재 과학 연구의 중요한 분야입니다.

In the future, highly automated driving and new Advanced Driver Assistance Systems will become reality.
미래에는 고도로 자동화된 주행과 새로운 첨단 운전자 지원 시스템(ADAS)이 현실이 될 것입니다.

While Advanced Driver Assistance Systems and automated driving functions for certain domains are already commercially available, ubiquitous automated driving in complex scenarios remains a subject of ongoing research.
특정 영역에 대한 ADAS 및 자동화된 주행 기능은 이미 상용화되었지만, 복잡한 시나리오에서의 유비쿼터스 자동화된 주행은 여전히 진행 중인 연구 주제입니다.

Contrarily to single-purpose Electronic Control Units, the software for automated driving is often executed on high performance PCs.
단일 목적 전자 제어 장치와 달리 자율 주행 소프트웨어는 종종 고성능 PC에서 실행됩니다.

The Robot Operating System 2 (ROS2) is commonly used to connect components in an automated driving system.
로봇 운영 체제 2 (ROS2)는 자율 주행 시스템의 구성 요소를 연결하는 데 일반적으로 사용됩니다.

Due to the time critical nature of automated driving systems, the performance of the framework is especially important.
자율 주행 시스템의 시간 제약적 특성으로 인해 프레임워크의 성능은 특히 중요합니다.

In this paper, a thorough performance evaluation of ROS2 is conducted, both in terms of timeliness and error rate.
본 논문에서는 적시성과 오류율 측면에서 ROS2에 대한 철저한 성능 평가를 수행합니다.

The results show that ROS2 is a suitable framework for automated driving systems.
결과는 ROS2가 자율 주행 시스템에 적합한 프레임워크임을 보여줍니다.

## 1. INTRODUCTION
Automated driving is a trending area of research, with a lot of effort from both academia and economy.
자율 주행은 학계와 산업계 모두에서 많은 노력을 기울이고 있는 트렌디한 연구 분야입니다.

Modern vehicles are equipped with many Advanced Driver Assistance Systems (ADAS) and even automated driving functions, such as highway pilots [Binder et al., 2016].
현대 자동차에는 고속도로 주행 보조와 같은 많은 첨단 운전자 지원 시스템(ADAS)과 자동화된 주행 기능이 탑재되어 있습니다 [Binder et al., 2016].

Despite the existence of these systems, automated driving higher than level 3 [SAE, 2021] is still a challenge, especially in complex and urban environments.
이러한 시스템이 존재함에도 불구하고, 레벨 3 이상의 자동화된 주행 [SAE, 2021]은 특히 복잡한 도시 환경에서 여전히 어려운 과제입니다.

In most commercial cars, there are many different Electronic Control Units (ECU), each for a specific purpose.
대부분의 상용 자동차에는 특정 목적을 위한 다양한 전자 제어 장치(ECU)가 있습니다.

Vehicle bus systems, most prominently CAN and FlexRay [Reif, 2011], allow these systems to communicate with each other.
차량 버스 시스템, 가장 대표적으로 CAN 및 FlexRay [Reif, 2011]를 통해 이러한 시스템이 서로 통신할 수 있습니다.

Research in automated driving shows that the tasks for this challenge are more complex and have high computational requirements.
자율 주행 연구에 따르면 이러한 과제에 대한 작업은 더 복잡하고 높은 계산 요구 사항을 가지고 있습니다.

For example, the evaluation of sensor values to detect objects, is often performed with neural networks [Spielberg et al., 2019].
예를 들어, 물체를 감지하기 위한 센서 값 평가는 종종 신경망을 사용하여 수행됩니다 [Spielberg et al., 2019].

Also, some of the tasks regarding automated driving are linked, e.g., object detection and localization.
또한 자율 주행과 관련된 일부 작업은 예를 들어 물체 감지 및 위치 파악과 같이 연결되어 있습니다.

For the development of automated driving systems (ADS), the Robot Operating System (ROS) is widely used [Reke et al., 2020], especially the Robot Operating System 2 (ROS2).
자율 주행 시스템(ADS) 개발을 위해 로봇 운영 체제(ROS)가 널리 사용되고 있으며 [Reke et al., 2020], 특히 로봇 운영 체제 2(ROS2)가 널리 사용됩니다.

The tasks regarding automated driving can be more efficiently performed on one or several central computation units, e.g., high performance PCs.
자율 주행과 관련된 작업은 하나 또는 여러 개의 중앙 계산 장치(예: 고성능 PC)에서 더 효율적으로 수행할 수 있습니다.

As the realization of automated driving consists of several subtasks, a modular software architecture is suitable.
자율 주행의 실현은 여러 하위 작업으로 구성되므로 모듈식 소프트웨어 아키텍처가 적합합니다.

An automated vehicle must sense and detect other objects, it must localize itself, and it must plan and control a trajectory.
자율 주행 차량은 다른 물체를 감지하고, 자신의 위치를 파악하고, 궤적을 계획하고 제어해야 합니다.

ROS2 facilitates simple communication between modules through a publish and subscribe pattern.
ROS2는 게시 및 구독 패턴을 통해 모듈 간의 간단한 통신을 용이하게 합니다.

As it was designed for the development of robots, it also provides a rich ecosystem of useful libraries for automated driving, such as probabilistic filters and planning algorithms.
로봇 개발을 위해 설계되었기 때문에 확률적 필터 및 계획 알고리즘과 같은 자율 주행에 유용한 라이브러리의 풍부한 생태계를 제공합니다.

ROS2 abstracts middleware communication across several levels in a high-level API.
ROS2는 고급 API에서 여러 수준에 걸쳐 미들웨어 통신을 추상화합니다.

The foundation for message exchange is a data distribution service (DDS), defined by the standard of the same name [Object Management Group, 2015].
메시지 교환의 기초는 동일한 이름의 표준으로 정의된 데이터 분산 서비스(DDS)입니다 [Object Management Group, 2015].

The connection between the DDS and ROS2 is abstracted using the ROS middleware interface (rmw).
DDS와 ROS2 간의 연결은 ROS 미들웨어 인터페이스(rmw)를 사용하여 추상화됩니다.

The core functionality of ROS2 is implemented in the ROS client library (rcl), which is based on the rmw.
ROS2의 핵심 기능은 rmw를 기반으로 하는 ROS 클라이언트 라이브러리(rcl)에서 구현됩니다.

Applications are normally implemented in language-specific wrappers of the rcl.
애플리케이션은 일반적으로 rcl의 언어별 래퍼로 구현됩니다.

The control of an automated vehicle imposes severe temporal and reliability requirements.
자율 주행 차량의 제어에는 엄격한 시간적 및 안정성 요구 사항이 적용됩니다.

The detection of obstacles and the planning of a path algorithm must be completed within a certain time frame.
장애물 감지 및 경로 계획 알고리즘은 특정 시간 프레임 내에 완료되어야 합니다.

Furthermore, information must not be lost.
또한 정보가 손실되어서는 안 됩니다.

Having a modular architecture with a distributed framework, such as ROS2, demands that the framework itself is performing efficiently.
ROS2와 같은 분산 프레임워크를 사용하는 모듈식 아키텍처를 사용하려면 프레임워크 자체가 효율적으로 수행되어야 합니다.

The high amount data that is necessary to be processed for automated driving, e.g., LIDAR point clouds and camera streams, makes this task even more challenging.
LIDAR 포인트 클라우드 및 카메라 스트림과 같이 자율 주행을 위해 처리해야 하는 많은 양의 데이터는 이 작업을 더욱 어렵게 만듭니다.

Therefore, in this paper, a performance evaluation of the ROS2 framework in an automated vehicle is presented.
따라서 본 논문에서는 자율 주행 차량에서 ROS2 프레임워크의 성능 평가를 제시합니다.

Particularly, the suitability of different middleware implementations for vehicular applications is investigated.
특히 차량 애플리케이션에 적합한 다양한 미들웨어 구현의 적합성을 조사합니다.

These implementations are compared in terms of latency and error susceptibility.
이러한 구현은 지연 시간 및 오류 민감성 측면에서 비교됩니다.

In this context, latency refers to the time elapsed from message transmission to reception.
이 맥락에서 지연 시간은 메시지 전송에서 수신까지 경과된 시간을 나타냅니다.

Besides, the error rate is quantified as packet loss.
또한 오류율은 패킷 손실로 정량화됩니다.

The scenarios for the analysis vary in terms of the number of components in the graph and the size and frequency of individual data packets.
분석 시나리오는 그래프의 구성 요소 수와 개별 데이터 패킷의 크기 및 빈도 측면에서 다릅니다.

All evaluations are performed on an actual on-board PC in an automated vehicle.
모든 평가는 자율 주행 차량의 실제 온보드 PC에서 수행됩니다.

## 2. RELATED WORK
Several architectures for automated driving based on ROS or ROS2 exist.
ROS 또는 ROS2를 기반으로 하는 자율 주행을 위한 여러 아키텍처가 존재합니다.

Some of the most prominent ones are Autoware.auto [The Autoware Foundation, 2023] and Apollo [Baidu Apollo consortium, 2023].
가장 눈에 띄는 것 중 일부는 Autoware.auto [The Autoware Foundation, 2023]와 Apollo [Baidu Apollo consortium, 2023]입니다.

These systems show that ROS2 is a suitable framework for developing an ADS.
이러한 시스템은 ROS2가 ADS를 개발하기에 적합한 프레임워크임을 보여줍니다.

However, due to their complexity, for research purposes, more lightweight approaches can lead to faster results and better performance.
그러나 복잡성으로 인해 연구 목적으로 더 가벼운 접근 방식을 사용하면 더 빠른 결과와 더 나은 성능을 얻을 수 있습니다.

An analysis of the performance of Autoware.auto yields good results, but this is not generalized to ROS2 [Li et al., 2022].
Autoware.auto의 성능 분석은 좋은 결과를 산출하지만 이는 ROS2에 일반화되지 않습니다 [Li et al., 2022].

In another publication, an alternative architecture for a ROS2 based automated vehicle is presented [Reke et al., 2020].
다른 출판물에서는 ROS2 기반 자율 주행 차량을 위한 대안 아키텍처가 제시됩니다 [Reke et al., 2020].

The system is described in detail and a performance evaluation is presented.
시스템에 대해 자세히 설명하고 성능 평가를 제공합니다.

This work indicates that ROS2 is suitable for real time operations.
이 작업은 ROS2가 실시간 작업에 적합함을 나타냅니다.

However, the Data Distribution Services (DDS) is not exchanged for analysis, and packet loss is not examined, either.
그러나 데이터 배포 서비스(DDS)는 분석을 위해 교환되지 않으며 패킷 손실도 검사되지 않습니다.

An assessment of the performance of ROS2 took place very early in the development stage [Maruyama et al., 2016].
ROS2의 성능 평가는 개발 단계에서 매우 일찍 수행되었습니다 [Maruyama et al., 2016].

Here, a comparative analysis is conducted between ROS1 and ROS2 to assess the potential positive impact of the novel concepts introduced in ROS2.
여기서는 ROS2에 도입된 새로운 개념의 잠재적인 긍정적 영향을 평가하기 위해 ROS1과 ROS2 간의 비교 분석을 수행합니다.

At that point of time, ROS2 does not exhibit superior performance compared to ROS1.
이 시점에서 ROS2는 ROS1에 비해 뛰어난 성능을 보이지 않습니다.

However, a notable improvement can be observed, particularly regarding the equal distribution of latencies across all subscribers.
그러나 특히 모든 구독자에게 지연 시간을 동일하게 분배하는 것과 관련하여 주목할 만한 개선 사항을 확인할 수 있습니다.

A different study investigates the real-time capabilities of ROS2 [Gutiérrez et al., 2018].
다른 연구에서는 ROS2의 실시간 기능을 조사합니다 [Gutiérrez et al., 2018].

The evaluation focuses on the ability of ROS2 to achieve soft real-time capabilities, indicating its potential for applications with timing constraints.
이 평가는 ROS2가 시간 제약이 있는 애플리케이션에 대한 잠재력을 나타내는 소프트 실시간 기능을 달성하는 능력에 중점을 둡니다.

The evaluation methodology primarily considers one-to-one communication, while more complex many-to-many scenarios are substituted with artificially generated workloads external to the ROS2 applications.
평가 방법론은 주로 일대일 통신을 고려하는 반면 더 복잡한 다대다 시나리오는 ROS2 애플리케이션 외부에서 인위적으로 생성된 워크로드로 대체됩니다.

This approach allows for an assessment of the performance of ROS2 in a controlled environment.
이 접근 방식을 통해 제어된 환경에서 ROS2의 성능을 평가할 수 있습니다.

In a more recent work, the performance of the three official DDS implementations (FastDDS, CycloneDDS, and RTI Connext) is compared, varying sending frequencies, packet sizes, and participants [Kronauer et al., 2021].
보다 최근의 연구에서는 전송 빈도, 패킷 크기 및 참가자를 변경하여 세 가지 공식 DDS 구현(FastDDS, CycloneDDS 및 RTI Connext)의 성능을 비교합니다 [Kronauer et al., 2021].

Consistent with [Maruyama et al., 2016], it is observed that latency exhibits a sharp increase beyond the UDP fragment size of 64kB.
[Maruyama et al., 2016]과 일치하여 UDP 조각 크기인 64kB를 초과하면 지연 시간이 급격히 증가하는 것으로 관찰됩니다.

Furthermore, the authors conclude that DDS is the primary contributor to latency.
또한 저자는 DDS가 지연 시간의 주요 원인이라고 결론지었습니다.

## 3. AUTOMATED DRIVING SYSTEM
For automated driving, Fraunhofer FOKUS uses a hybrid Mercedes E-Class, which is able to plan and drive paths in an automated way.
자율 주행을 위해 프라운호퍼 FOKUS는 자동화된 방식으로 경로를 계획하고 주행할 수 있는 하이브리드 메르세데스 E-Class를 사용합니다.x

The vehicle is used to developed different ADS, such as automated valet parking [Schäufele et al., 2017].
이 차량은 자동 발렛 주차와 같은 다양한 ADS를 개발하는 데 사용됩니다 [Schäufele et al., 2017].

It is equipped with communication hardware for cooperative maneuvers as well [Schaeufele et al., 2017, Eiermann et al., 2020].
또한 협력 기동을 위한 통신 하드웨어도 탑재되어 있습니다 [Schaeufele et al., 2017, Eiermann et al., 2020].

Due to its complexity, the overall system is divided in subsystems.
복잡성으로 인해 전체 시스템은 하위 시스템으로 나뉩니다.

As a result of the modular architecture, ROS2 was selected, because it allows for simple communication between components through a publish and subscribe mechanism.
모듈식 아키텍처의 결과로 ROS2가 선택되었습니다. ROS2는 게시 및 구독 메커니즘을 통해 구성 요소 간의 간단한 통신을 허용하기 때문입니다.

Besides, ROS2 [Macenski et al., 2022] offers many robotic libraries that can be applied for an ADS.
게다가 ROS2 [Macenski et al., 2022]는 ADS에 적용할 수 있는 많은 로봇 라이브러리를 제공합니다.

ROS2 is used to implement the components of the architecture.
ROS2는 아키텍처의 구성 요소를 구현하는 데 사용됩니다.

The design of the system follows the pattern of Sense, Plan, Act [During and Lemmer, 2016].
시스템의 설계는 감지, 계획, 작동 패턴을 따릅니다 [During and Lemmer, 2016].

First, a representation of the environment of the vehicle is created with sensing.
먼저 감지를 통해 차량 환경에 대한 표현이 생성됩니다.

The sensors are evaluated in the Perception Unit (PU), which is an on-board PC with high performance graphics hardware.
센서는 고성능 그래픽 하드웨어가 탑재된 온보드 PC인 인식 장치(PU)에서 평가됩니다.

In the planning stage, the environment model and other constraints, such as vehicle parameters, are used for the calculation of a drivable trajectory for the vehicle.
계획 단계에서 환경 모델과 차량 매개변수와 같은 기타 제약 조건을 사용하여 차량의 주행 가능한 궤적을 계산합니다.

In acting, the planned trajectory is controlled and executed.
작동 시 계획된 궤적을 제어하고 실행합니다.

The perception system of the automated vehicle, called 3D Vision, allows full understanding of the surroundings.
3D 비전이라고 하는 자율 주행 차량의 인식 시스템을 통해 주변 환경을 완전히 이해할 수 있습니다.

The test vehicle can be seen in Figure 1.
테스트 차량은 그림 1에서 볼 수 있습니다.

For the 3D Vision, the car is equipped with a sensor rig that can hold various sensors.
3D 비전을 위해 차량에는 다양한 센서를 장착할 수 있는 센서 리그가 장착되어 있습니다.

A schematic overview of the sensor rig is shown in Figure 2.
센서 리그의 개략적인 개요는 그림 2에 나와 있습니다.

The sensor setup consists of three LIDAR scanners, which create a 3D point cloud of the vehicle surroundings.
센서 설정은 차량 주변의 3D 포인트 클라우드를 생성하는 3개의 LIDAR 스캐너로 구성됩니다.

For a full view in camera images, seven cameras are mounted on the sensor rig, one camera with a 60 degrees aperture to the front, four cameras with 100 degrees aperture on the corners, and additionally two front cameras with 100 degrees aperture, which provide stereo images.
카메라 이미지에서 전체 보기를 위해 7개의 카메라가 센서 리그에 장착됩니다. 전면에 60도 조리개가 있는 카메라 1개, 모서리에 100도 조리개가 있는 카메라 4개, 스테레오 이미지를 제공하는 100도 조리개가 있는 전면 카메라 2개가 추가로 장착됩니다.

The sensor evaluation is performed with neural networks at the PU.
센서 평가는 PU에서 신경망을 사용하여 수행됩니다.

In an early fusion, LIDAR points are projected onto the 2D camera images, which are processed with Convolutional Neural Networks (CNN).
초기 융합에서 LIDAR 포인트는 컨볼루션 신경망(CNN)으로 처리되는 2D 카메라 이미지에 투영됩니다.

Due to the projection, the 3D coordinates of the object detections from the camera images can be determined.
투영으로 인해 카메라 이미지에서 물체 감지의 3D 좌표를 결정할 수 있습니다.

Figure 3 shows the results from the image processing.
그림 3은 이미지 처리 결과를 보여줍니다.

The network detects various traffic objects, such as cars, scooters, and traffic signs.
네트워크는 자동차, 스쿠터 및 교통 표지판과 같은 다양한 교통 물체를 감지합니다.

For lane detection, a novel early fusion approach is implemented [Wulff et al., 2018].
차선 감지를 위해 새로운 초기 융합 접근 방식이 구현됩니다 [Wulff et al., 2018].

For LIDAR perception, the points are grouped in bins and various features are calculated for each bin with neural networks, such as Pointpillars [Lang et al., 2019] and SECOND [Yan et al., 2018].
LIDAR 인식의 경우 포인트가 빈에 그룹화되고 신경망을 사용하여 각 빈에 대해 Pointpillars [Lang et al., 2019] 및 SECOND [Yan et al., 2018]와 같은 다양한 기능이 계산됩니다.

These networks can process data more efficiently compared to raw point clouds.
이러한 네트워크는 원시 포인트 클라우드에 비해 데이터를 더 효율적으로 처리할 수 있습니다.

This efficiency stems from the utilization of an internal representations in the form of bins, enabling faster processing while still yielding valuable outcomes.
이러한 효율성은 빈 형태의 내부 표현을 활용하여 더 빠른 처리를 가능하게 하면서도 가치 있는 결과를 얻을 수 있기 때문에 발생합니다.

The perception results are shown in Figure 4.
인식 결과는 그림 4에 나와 있습니다.

The top left, bottom left, and bottom center show the 2D bounding boxes in camera images.
왼쪽 상단, 왼쪽 하단 및 중앙 하단에는 카메라 이미지의 2D 경계 상자가 표시됩니다.

The top center shows an internal representation of the LIDAR processing, in which each LIDAR point is assigned to a specific bin.
상단 중앙에는 각 LIDAR 포인트가 특정 빈에 할당되는 LIDAR 처리의 내부 표현이 표시됩니다.

The 3D bounding boxes are shown in the top right.
3D 경계 상자는 오른쪽 상단에 표시됩니다.

The hardware setup can be seen in Figure 5 with the PU rack in the center and devices for sensor connection and vehicular communication.
하드웨어 설정은 그림 5에서 중앙에 PU 랙과 센서 연결 및 차량 통신을 위한 장치와 함께 볼 수 있습니다.

The objects derived from camera and LIDAR are collected in an environment model.
카메라와 LIDAR에서 파생된 물체는 환경 모델에 수집됩니다.

It takes care of tracking objects, i.e., assigning a unique identifier over consecutive time frames.
연속적인 시간 프레임에 걸쳐 고유한 식별자를 할당하는 등 물체를 추적합니다.

Thereby, the environment model fuses the object detections from the different sensors to a single internal object representation.
따라서 환경 모델은 서로 다른 센서의 물체 감지를 단일 내부 물체 표현으로 융합합니다.

The output of the environment model is passed to the planning and acting stages.
환경 모델의 출력은 계획 및 작동 단계로 전달됩니다.

The path planning builds upon the environment model and an existing route, which defines the vehicle’s path at the road segment level, determining the segments to traverse and the turns to take at intersections.
경로 계획은 환경 모델과 기존 경로를 기반으로 합니다. 기존 경로는 도로 세그먼트 수준에서 차량의 경로를 정의하여 통과할 세그먼트와 교차로에서 회전할 방향을 결정합니다.

The route is map-based, but during path planning, it is enriched with real-time information from the perception and refined at lane level.
경로는 지도 기반이지만 경로 계획 중에 인식에서 얻은 실시간 정보로 풍부해지고 차선 수준에서 개선됩니다.

The resulting path includes lane changes and avoids obstacles.
결과 경로에는 차선 변경이 포함되고 장애물을 피합니다.

To further refine the path, a drivable trajectory is generated.
경로를 더욱 개선하기 위해 주행 가능한 궤적이 생성됩니다.

This trajectory defines the desired position and time of the vehicle using a 2D spline, which is transmitted to the control system.
이 궤적은 제어 시스템에 전송되는 2D 스플라인을 사용하여 차량의 원하는 위치와 시간을 정의합니다.

The spline is continuous and adheres to vehicle constraints.
스플라인은 연속적이며 차량 제약 조건을 준수합니다.

It considers detected dynamic objects, such as vehicles and pedestrians, to ensure collision avoidance.
충돌을 피하기 위해 차량 및 보행자와 같은 감지된 동적 물체를 고려합니다.

Additionally, the trajectory is optimized to achieve efficient and comfortable driving.
또한 궤적은 효율적이고 편안한 주행을 달성하도록 최적화됩니다.

For control, the vehicle is equipped with the Schaeffler Paravan Drive-by-Wire system [Unseld, 2020], which allows to actuate the steering wheel, and the throttle and brake controls.
제어를 위해 차량에는 스티어링 휠, 스로틀 및 브레이크 컨트롤을 작동할 수 있는 Schaeffler Paravan Drive-by-Wire 시스템이 장착되어 있습니다 [Unseld, 2020].

The control loop to follow the calculated spline is based on the pure pursuit algorithm [Samuel et al., 2016].
계산된 스플라인을 따르는 제어 루프는 순수 추적 알고리즘을 기반으로 합니다 [Samuel et al., 2016].

The required steering wheel angles and forces are applied by the Drive-by-Wire system.
필요한 스티어링 휠 각도와 힘은 Drive-by-Wire 시스템에서 적용합니다.
