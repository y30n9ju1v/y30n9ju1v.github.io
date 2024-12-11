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

4. EVALUATION SYSTEM
To be able to correctly assess the performance of ROS2, the entire software stack must be taken into account, and influencing factors must be considered as isolated as possible.
ROS2의 성능을 정확하게 평가하려면 전체 소프트웨어 스택을 고려해야 하며 영향을 미치는 요소는 가능한 한 분리하여 고려해야 합니다.

The aim of the measurements is to evaluate the performance of ROS2 regarding the requirements in an automated driving system.
측정의 목표는 자율 주행 시스템의 요구 사항과 관련하여 ROS2의 성능을 평가하는 것입니다.

Two aspects are particularly essential for this.
이를 위해 두 가지 측면이 특히 중요합니다.

Is ROS2 fast enough to function as the backbone of interprocess communication in a real-time system? This condition can be measured very well with the latency, which for this work is defined as the elapsed time between sending the message and receiving the message in the user application.
ROS2가 실시간 시스템에서 프로세스 간 통신의 중추 역할을 할 만큼 충분히 빠릅니까? 이 조건은 지연 시간으로 매우 잘 측정할 수 있습니다. 이 작업의 경우 지연 시간은 메시지를 보내는 시간과 사용자 애플리케이션에서 메시지를 받는 시간 사이의 경과 시간으로 정의됩니다.

Another aspect to consider is the error rate in the system.
고려해야 할 또 다른 측면은 시스템의 오류율입니다.

Reliable delivery of messages is essential for an automated system, as the loss of information can lead to potentially dangerous wrong decisions.
정보 손실은 잠재적으로 위험한 잘못된 결정으로 이어질 수 있으므로 자동화된 시스템에는 메시지의 안정적인 전달이 필수적입니다.

To estimate this metric, the occurring packet loss is measured as a percentage.
이 메트릭을 추정하기 위해 발생하는 패킷 손실을 백분율로 측정합니다.

Only local measurements are carried out for the measurement scenario, which corresponds to the current setup of the test vehicles used.
사용된 테스트 차량의 현재 설정에 해당하는 측정 시나리오에 대해서만 로컬 측정이 수행됩니다.

The participants, nodes, and topics are also predefined; fluctuating behavior is not evaluated.
참가자, 노드 및 주제도 미리 정의되어 있습니다. 변동하는 동작은 평가되지 않습니다.

The full functional scope of ROS2 is realized via four different abstraction levels.
ROS2의 전체 기능 범위는 네 가지 추상화 수준을 통해 실현됩니다.

Applications are written with the help of client libraries.
애플리케이션은 클라이언트 라이브러리의 도움으로 작성됩니다.

These map the API in a specific programming language, officially supported here are C++ (rclcpp) and Python (rclpy).
이들은 API를 특정 프로그래밍 언어로 매핑합니다. 여기서 공식적으로 지원되는 언어는 C++(rclcpp) 및 Python(rclpy)입니다.

Most of the functionality is implemented in C and available as ROS client library.
대부분의 기능은 C로 구현되고 ROS 클라이언트 라이브러리로 사용할 수 있습니다.

Communication with the specific DDS implementation, which manages the sending of messages and the discovery of other participants, is handled by the ROS middleware interface (rmw).
메시지 전송 및 다른 참가자 검색을 관리하는 특정 DDS 구현과의 통신은 ROS 미들웨어 인터페이스(rmw)에서 처리합니다.

Figure 6 summarizes the internal structure of ROS2.
그림 6은 ROS2의 내부 구조를 요약한 것입니다.

Each of the layers influences the overall performance.
각 계층은 전체 성능에 영향을 미칩니다.

The first elementary influencing factor that comes into play is the DDS.
작용하는 첫 번째 기본 영향 요소는 DDS입니다.

Each DDS is used in its standard configuration to ensure basic comparability.
각 DDS는 기본적인 비교 가능성을 보장하기 위해 표준 구성으로 사용됩니다.

The ROS2 stack with the rmw and rcl layer then follows, based on the DDS.
그런 다음 DDS를 기반으로 rmw 및 rcl 계층이 있는 ROS2 스택이 이어집니다.

The same version is used for each measurement to rule out possible deviations due to changes to these layers.
이러한 계층에 대한 변경으로 인한 가능한 편차를 배제하기 위해 각 측정에 동일한 버전이 사용됩니다.

rclcpp is used to implement the user applications for the measurement.
rclcpp는 측정을 위한 사용자 애플리케이션을 구현하는 데 사용됩니다.

In addition to the hardware and software stack, the way in which the system is used is also important.
하드웨어 및 소프트웨어 스택 외에도 시스템 사용 방식도 중요합니다.

Parameters that have a major influence on the possible performance here are the data size per message, the number of messages sent per time unit, the number of nodes in the entire network, the number of topics used, and the number of publishers and subscribers per topic.
여기에서 가능한 성능에 큰 영향을 미치는 매개변수는 메시지당 데이터 크기, 시간 단위당 전송되는 메시지 수, 전체 네트워크의 노드 수, 사용된 주제 수, 주제당 게시자 및 구독자 수입니다.

These parameters in particular are of great interest for the measurements, as they demand the central aspects of the DDS and ROS2 implementation with regard to their efficiency.
이러한 매개변수는 특히 DDS 및 ROS2 구현의 중심 측면을 효율성 측면에서 요구하기 때문에 측정에 매우 중요합니다.

Only the publish / subscribe pattern is considered for the measurements, as both services and actions are based on this methodology.
서비스와 작업 모두 이 방법론을 기반으로 하므로 측정에는 게시/구독 패턴만 고려됩니다.

Furthermore, the following assumptions are made for the measurements: A node is either a publisher or a subscriber, never both at the same time; each measurement is performed with one DDS; this is not exchanged during a measurement; and all nodes involved in the measurement are started beforehand; there are no late-joining components.
또한 측정을 위해 다음과 같은 가정을 합니다.
노드는 게시자 또는 구독자이며 동시에 둘 다가 아닙니다.
각 측정은 하나의 DDS로 수행됩니다.
측정 중에는 교환되지 않습니다.
측정에 관련된 모든 노드는 미리 시작됩니다.
나중에 참여하는 구성 요소는 없습니다.

In a first iteration, the different DDS implementations are compared with each other.
첫 번째 반복에서는 서로 다른 DDS 구현을 서로 비교합니다.

For this purpose, the performance test framework from Apex.AI [Pemmaiah et al., 2022] is used.
이를 위해 Apex.AI [Pemmaiah et al., 2022]의 성능 테스트 프레임워크가 사용됩니다.

Each DDS is tested in different scenarios with varying data sizes and numbers of participants.
각 DDS는 다양한 데이터 크기와 참가자 수를 사용하는 다양한 시나리오에서 테스트됩니다.

Based on the results, a selected DDS is then evaluated in detail to gain a deeper understanding of the performance of ROS2.
결과를 바탕으로 ROS2의 성능에 대한 더 깊은 이해를 얻기 위해 선택한 DDS를 자세히 평가합니다.

For this purpose, a much larger number of factors are permuted and analyzed using tracing [Bédard et al., 2022] to track the path of the message through the software stack in order to precisely localize possible performance losses.
이를 위해 더 많은 수의 요소를 순열하고 추적을 사용하여 분석하여 [Bédard et al., 2022] 소프트웨어 스택을 통한 메시지 경로를 추적하여 가능한 성능 손실을 정확하게 찾아냅니다.

ROS2 works on the basis of workspaces [Open Robotics, 2022b].
ROS2는 작업 공간을 기반으로 작동합니다 [Open Robotics, 2022b].

A workspace comprises a collection of ROS2 packages, i.e., ROS2-based software projects, which are built with the help of the  ROS2-specific build tool Colcon [Open Robotics, 2022c].
작업 공간은 ROS2 기반 소프트웨어 프로젝트인 ROS2 패키지 컬렉션으로 구성되며 ROS2 특정 빌드 도구인 Colcon을 사용하여 빌드됩니다 [Open Robotics, 2022c].

These workspaces can in turn build on each other so that the hierarchically higher workspace has access to all packages of the underlying workspace.
이러한 작업 공간은 차례로 서로 빌드하여 계층적으로 더 높은 작업 공간이 기본 작업 공간의 모든 패키지에 액세스할 수 있도록 합니다.

This enables a structured and clean setup of all necessary packages without having to install packages that are not required for the specific measurement.
이를 통해 특정 측정에 필요하지 않은 패키지를 설치할 필요 없이 필요한 모든 패키지를 구조화되고 깔끔하게 설정할 수 있습니다.

As first step of the implementation, the lowest common denominator of required packages is installed in a workspace.
구현의 첫 번째 단계로 필요한 패키지의 최소 공통 분모가 작업 공간에 설치됩니다.

Here, this is the ROS2 library itself with the basic functionalities in the Rolling version.
여기서는 롤링 버전의 기본 기능을 갖춘 ROS2 라이브러리 자체입니다.

This workspace is built using the build tools, and all external dependencies are installed.
이 작업 공간은 빌드 도구를 사용하여 빌드되고 모든 외부 종속성이 설치됩니다.

For the initial comparison testing a new workspace is created, and the Apex.AI Performance Test Project [Pemmaiah et al., 2022] is built.
초기 비교 테스트를 위해 새 작업 공간이 생성되고 Apex.AI 성능 테스트 프로젝트가 빌드됩니다 [Pemmaiah et al., 2022].

With the help of a Python script, a bash script is created from the benchmark configuration for the parameters.
Python 스크립트의 도움으로 매개변수에 대한 벤치마크 구성에서 bash 스크립트가 생성됩니다.

As part of the benchmark, the workspace is rebuilt once for each DDS so that the performance test uses it accordingly.
벤치마크의 일부로 성능 테스트에서 이에 따라 사용할 수 있도록 각 DDS에 대해 작업 공간이 한 번 다시 빌드됩니다.

All subscribers are started for each configuration and each DDS; the executable provided by Apex.AI is called up configured accordingly for this purpose.
각 구성 및 각 DDS에 대해 모든 구독자가 시작됩니다. 이를 위해 Apex.AI에서 제공하는 실행 파일이 이에 따라 구성되어 호출됩니다.

After a short wait to ensure that all subscribers have been initialized, the publishers are also started; the Apex.AI executable is also used here.
모든 구독자가 초기화되었는지 확인하기 위해 잠시 기다린 후 게시자도 시작됩니다. 여기에서도 Apex.AI 실행 파일이 사용됩니다.

After the configured time, the processes end automatically, and all open log files are closed.
구성된 시간이 지나면 프로세스가 자동으로 종료되고 열려 있는 모든 로그 파일이 닫힙니다.

The process is repeated for each of the configurations.
각 구성에 대해 프로세스가 반복됩니다.

Only the layers below the rmw layer are used here.
여기서는 rmw 계층 아래의 계층만 사용됩니다.

This allows to consider only the influence of the specific DDS implementation and to exclude any influences from higher layers.
이를 통해 특정 DDS 구현의 영향만 고려하고 상위 계층의 영향을 배제할 수 있습니다.

Table 1 lists the permutations of the parameters; each of the DDS implementations is tested once for each of the specified configurations over a runtime of 60 seconds.
표 1에는 매개변수의 순열이 나열되어 있습니다. 각 DDS 구현은 60초의 런타임 동안 지정된 각 구성에 대해 한 번씩 테스트됩니다.

The aim of the configurations is to obtain a comprehensive overview of the performance in order to be able to compare the various DDSs as accurately possible.
구성의 목표는 다양한 DDS를 가능한 한 정확하게 비교할 수 있도록 성능에 대한 포괄적인 개요를 얻는 것입니다.

The first three configurations serve as a basis for this.
처음 세 가지 구성이 이를 위한 기초 역할을 합니다.

The simple 1 - 1 communication reduces the possible interference to a minimum.
간단한 1-1 통신은 가능한 간섭을 최소한으로 줄입니다.

The 1 - 31 communication is already more demanding, as the DDS must now distribute the message to 31 subscribers, which leads to a considerably larger required bandwidth, especially with larger data packets.
1-31 통신은 이미 더 까다롭습니다. DDS가 이제 메시지를 31명의 구독자에게 배포해야 하므로 특히 더 큰 데이터 패킷을 사용할 경우 상당히 더 큰 대역폭이 필요하기 때문입니다.

32 nodes in a network comes much closer to an application in the field of automated driving in terms of the number of participants and can therefore provide initial indications of performance under load.
네트워크의 32개 노드는 참가자 수 측면에서 자율 주행 분야의 애플리케이션에 훨씬 더 가깝기 때문에 부하 상태에서 성능에 대한 초기 표시를 제공할 수 있습니다.

The frequency set for all tests is 10Hz, which corresponds to the frequency most commonly used by sensors and computing components in automated driving systems and therefore serves as a sensible clock rate for generating the load.
모든 테스트에 대해 설정된 빈도는 10Hz입니다. 이는 자율 주행 시스템에서 센서 및 컴퓨팅 구성 요소에서 가장 일반적으로 사용되는 빈도에 해당하므로 부하를 생성하기 위한 적절한 클럭 속도 역할을 합니다.

The different data sizes represent different scenarios.
서로 다른 데이터 크기는 서로 다른 시나리오를 나타냅니다.

Struct16 is the smallest message and can, for example, be equated with a message from a simple sensor in the vehicle, such as acceleration.
Struct16은 가장 작은 메시지이며 예를 들어 가속도와 같은 차량의 간단한 센서의 메시지와 동일시할 수 있습니다.

Array64k represents more complex data, such as a trajectory or recognized objects.
Array64k는 궤적 또는 인식된 물체와 같은 더 복잡한 데이터를 나타냅니다.

Pointcloud1m is the largest message and is used to represent LIDAR scans or camera streams.
Pointcloud1m은 가장 큰 메시지이며 LIDAR 스캔 또는 카메라 스트림을 나타내는 데 사용됩니다.

This message size usually forms the upper limit of the messages used in the automated vehicle in terms of size per message.
이 메시지 크기는 일반적으로 메시지당 크기 측면에서 자율 주행 차량에 사용되는 메시지의 상한을 형성합니다.

Overall, the parameters thus cover a good range from the actual application and provide initial indications of the performance of the various DDS implementations.
전반적으로 매개변수는 실제 애플리케이션에서 좋은 범위를 포괄하며 다양한 DDS 구현의 성능에 대한 초기 표시를 제공합니다.

Several ROS2 packages are required for more detailed measurements of a selected DDS.
선택한 DDS를 더 자세히 측정하려면 여러 ROS2 패키지가 필요합니다.

ROS2 tracing is elementary here [Bédard et al., 2022], as well as a special version of the DDS, to provide the necessary insight at this level.
ROS2 추적은 여기서 기본입니다 [Bédard et al., 2022]. 이 수준에서 필요한 통찰력을 제공하기 위해 특별 버전의 DDS도 마찬가지입니다.

With the help of the ROS2 tracing package [Bédard et al., 2022] the message can be traced through the entire stack [Bédard et al., 2023] to understand how the latency arises, and where message losses occur.
ROS2 추적 패키지의 도움으로 [Bédard et al., 2022] 메시지는 전체 스택을 추적하여 [Bédard et al., 2023] 지연 시간이 어떻게 발생하고 메시지 손실이 어디에서 발생하는지 이해할 수 있습니다.

It is demonstrated that the additional overhead caused by tracing is minimal and therefore does not distort the results [Bédard et al., 2022].
추적으로 인한 추가 오버헤드가 최소화되므로 결과가 왜곡되지 않는다는 것이 입증되었습니다 [Bédard et al., 2022].

For measurements, a package is required to generate the load in the system according to the configuration.
측정을 위해 구성에 따라 시스템에서 부하를 생성하는 패키지가 필요합니다.

For this purpose, a simple but fully configurable node is implemented as a publisher and subscriber.
이를 위해 간단하지만 완전히 구성 가능한 노드가 게시자 및 구독자로 구현됩니다.

When the process is started, the publisher receives the frequency in milliseconds at which the message is to be sent, the size of the message in bytes, the length of the measurement in seconds and the topic on which it is to send.
프로세스가 시작되면 게시자는 메시지를 보낼 빈도(밀리초), 메시지 크기(바이트), 측정 길이(초) 및 보낼 주제를 수신합니다.

Based on this information, it then starts a ROS2 timer to periodically publish a packet on the topic according to the frequency.
그런 다음 이 정보를 기반으로 ROS2 타이머를 시작하여 빈도에 따라 주제에 대한 패킷을 주기적으로 게시합니다.

The message sent consists of a header, containing a timestamp and an ID in the form of an integer number, and a byte array of the defined size, which is filled with random values.
전송된 메시지는 타임스탬프와 정수 형식의 ID가 포함된 헤더와 임의 값으로 채워진 정의된 크기의 바이트 배열로 구성됩니다.

The timestamp is entered in the message immediately before the actual transmission.
타임스탬프는 실제 전송 직전에 메시지에 입력됩니다.

The structure of the subscriber nodes is similar.
구독자 노드의 구조는 유사합니다.

They receive a list of topics for which corresponding subscriptions are created.
해당 구독이 생성된 주제 목록을 수신합니다.

A timestamp is also taken directly in the associated callback, which is then compared with the timestamp from the received message, and the difference is saved.
관련 콜백에서도 타임스탬프가 직접 가져와 수신된 메시지의 타임스탬프와 비교하여 그 차이를 저장합니다.

The actual measurement of the latency then follows using the recorded traces.
그런 다음 기록된 추적을 사용하여 지연 시간을 실제로 측정합니다.

The measurement is started with a Python script, which parses the various configurations and instantiates a corresponding number of publishers and subscribers, starts the tracing, and activates the corresponding DDS using an environment variable.
측정은 다양한 구성을 구문 분석하고 해당 수의 게시자와 구독자를 인스턴스화하고 추적을 시작하고 환경 변수를 사용하여 해당 DDS를 활성화하는 Python 스크립트로 시작됩니다.

The aim of these benchmarks is to gain a more detailed, in-depth insight into the performance of ROS2.
이러한 벤치마크의 목표는 ROS2의 성능에 대한 더 자세하고 심층적인 통찰력을 얻는 것입니다.

For this purpose, the parameter space is significantly enlarged to obtain a higher resolution of the test results.
이를 위해 테스트 결과의 해상도를 높이기 위해 매개변수 공간이 크게 확장됩니다.

The aim is also to go beyond the current requirements to be able to assess the performance limits of ROS2.
또한 ROS2의 성능 한계를 평가할 수 있도록 현재 요구 사항을 뛰어넘는 것을 목표로 합니다.

Table 2 shows the possible configurations of the parameters for the detailed benchmark.
표 2는 자세한 벤치마크에 대한 매개변수의 가능한 구성을 보여줍니다.

Based on the number of nodes, three topologies are created for the publisher-subscriber-subscriber-per-node ratio, as shown in Figure 7.
노드 수를 기반으로 그림 7과 같이 노드당 게시자-구독자-구독자 비율에 대해 세 가지 토폴로지가 생성됩니다.

The first topology has exactly as many publishers as subscribers, each with their own topic and one subscriber per node.
첫 번째 토폴로지에는 구독자 수만큼 게시자가 있으며 각 게시자는 고유한 주제와 노드당 하나의 구독자를 갖습니다.

The second topology changes the ratio, with only one publisher serving all the remaining nodes as a subscriber; here, too, there is only one subscriber per node.
두 번째 토폴로지는 비율을 변경하여 하나의 게시자만 나머지 모든 노드를 구독자로 사용합니다. 여기에도 노드당 구독자는 하나뿐입니다.

Finally, this relationship is reversed, and a node with many subscribers is served by the remaining nodes as a publisher.
마지막으로 이 관계가 반전되고 많은 구독자가 있는 노드가 나머지 노드에서 게시자로 사용됩니다.

Topology 1 makes it possible to evaluate the influence of the number of topics on the overall performance and thus to test the scalability of topics.
토폴로지 1을 사용하면 주제 수가 전체 성능에 미치는 영향을 평가하여 주제의 확장성을 테스트할 수 있습니다.

Topology 2 makes it possible to check how efficiently the distribution of messages on a topic works and how great the influence of the number of subscribers per topic is on the overall performance.
토폴로지 2를 사용하면 주제에 대한 메시지 배포가 얼마나 효율적으로 작동하는지 그리고 주제당 구독자 수가 전체 성능에 얼마나 큰 영향을 미치는지 확인할 수 있습니다.

Topology 3 allows to check how well the executor of a single node scales under load with more callbacks and how many subscribers per node can be effectively implemented before the SingleThreadedExecutor is overloaded.
토폴로지 3을 사용하면 단일 노드의 실행기가 더 많은 콜백으로 부하 상태에서 얼마나 잘 확장되는지 그리고 SingleThreadedExecutor가 과부하되기 전에 노드당 구독자를 효과적으로 몇 개나 구현할 수 있는지 확인할 수 있습니다.

In practical applications, a mixture of all three topologies can be found.
실제 애플리케이션에서는 세 가지 토폴로지가 모두 혼합되어 있습니다.

Each of the topologies is measured for each combination of frequency and data size over a runtime of 60 seconds.
각 토폴로지는 60초의 런타임 동안 빈도와 데이터 크기의 각 조합에 대해 측정됩니다.

For the detailed benchmarks, additional data sizes were added to the three data packets from the previous benchmarks.
자세한 벤치마크를 위해 이전 벤치마크의 세 가지 데이터 패킷에 추가 데이터 크기가 추가되었습니다.

First, 512kB was added as the middle value of the previous value range to achieve better coverage in this area.
첫째, 이 영역에서 더 나은 범위를 달성하기 위해 이전 값 범위의 중간값으로 512kB가 추가되었습니다.

Secondly, in order to test the limits, twice the previous maximum was added again with the aim of demanding the maximum bandwidth.
둘째, 한계를 테스트하기 위해 최대 대역폭을 요구하는 것을 목표로 이전 최대값의 두 배가 다시 추가되었습니다.

Measurements at 100Hz were also added to the previously used frequency of 10Hz in order to gain an insight into the extent to which there is still potential for improvement here.
여기에 얼마나 개선의 여지가 있는지에 대한 통찰력을 얻기 위해 이전에 사용된 10Hz 빈도에 100Hz에서의 측정값도 추가되었습니다.

All benchmarks are run on the PU with an Intel® Xeon® E5-2667 v4 CPU and 8x 32GB RDIMM DDR4-2400+ reg ECC.
모든 벤치마크는 Intel® Xeon® E5-2667 v4 CPU 및 8x 32GB RDIMM DDR4-2400+ reg ECC가 탑재된 PU에서 실행됩니다.

An Intel X550-T2 network card handles communication with the connected sensors.
Intel X550-T2 네트워크 카드는 연결된 센서와의 통신을 처리합니다.

The system is running Ubuntu 20.04 LTS.
시스템에서 Ubuntu 20.04 LTS를 실행하고 있습니다.

## 5. EVALUATION RESULTS
A direct comparison of the three officially supported DDS implementations reveals differences but also similarities in performance behavior.
공식적으로 지원되는 세 가지 DDS 구현을 직접 비교하면 성능 동작의 차이점뿐만 아니라 유사점도 드러납니다.

Figure 8 shows the course of the latency in milliseconds over the measurement period of 60 seconds, with Struct16 in blue, Array64k in orange, and PointCloud1m in green as introduced in section 4.
그림 8은 60초의 측정 기간 동안 지연 시간(밀리초)의 경과를 보여줍니다. 4장에서 소개된 것처럼 파란색은 Struct16, 주황색은 Array64k, 녹색은 PointCloud1m입니다.

Each column represents one of the three DDS implementations.
각 열은 세 가지 DDS 구현 중 하나를 나타냅니다.

The top row shows the results for a publisher communicating with a subscriber.
맨 위 행은 게시자가 구독자와 통신하는 결과를 보여줍니다.

The bottom row shows the results for a publisher communicating with 31 subscribers.
맨 아래 행은 게시자가 31명의 구독자와 통신하는 결과를 보여줍니다.

The first scenario here serves as a basic assessment, while the second scenario is more of an application scenario from the field of automated driving regarding the subscribers.
여기서 첫 번째 시나리오는 기본 평가 역할을 하는 반면 두 번째 시나리오는 구독자와 관련하여 자율 주행 분야의 애플리케이션 시나리오에 가깝습니다.

The individual measurement lines represent the different packet sizes.
개별 측정선은 서로 다른 패킷 크기를 나타냅니다.

The measurement results are generally very good.
측정 결과는 일반적으로 매우 좋습니다.

For most scenarios, the latency remains below 2 ms, leaving a clear margin up to a frequency of 10 Hz.
대부분의 시나리오에서 지연 시간은 2ms 미만으로 유지되어 10Hz의 주파수까지 명확한 여유가 있습니다.

Only the latency for PointCloud1m is higher, which is particularly clear for the higher number of subscribers.
PointCloud1m에 대한 지연 시간만 더 높으며, 이는 특히 구독자 수가 더 많은 경우에 분명합니다.

In this case, the latency increases to at least 8ms (FastDDS) and on average to 15-20 ms.
이 경우 지연 시간은 최소 8ms(FastDDS)로 증가하고 평균 15-20ms로 증가합니다.

It is worth noting that the variance for each combination is below 1 ms, apart from ConnextDDS for 31 subscribers / PointCloud1m, where the variance is around 18 ms.
분산이 약 18ms인 31명의 구독자/PointCloud1m에 대한 ConnextDDS를 제외하고 각 조합의 분산은 1ms 미만이라는 점은 주목할 가치가 있습니다.

This low variance indicates stable message transmission behavior, as does the almost constant latency over the course of the measurement.
이 낮은 분산은 측정 과정에서 거의 일정한 지연 시간과 마찬가지로 안정적인 메시지 전송 동작을 나타냅니다.

The packet loss is also fairly limited.
패킷 손실도 상당히 제한적입니다.

In the maximum case it is 0.88%, in most cases it is 0%.
최대의 경우 0.88%이고 대부분의 경우 0%입니다.

It is noticeable, however, that ConnextDDS is the only one with a packet loss of 0.18% in the 1-1 communication for the PointCloud1m messages, while all other 1-1 communication scenarios each have 0%.
그러나 ConnextDDS는 PointCloud1m 메시지에 대한 1-1 통신에서 패킷 손실이 0.18%인 유일한 DDS인 반면 다른 모든 1-1 통신 시나리오는 각각 0%입니다.

ConnextDDS also performs worse in the 1-31 scenario, losing a small number of packets for each message size.
ConnextDDS는 1-31 시나리오에서도 성능이 좋지 않아 각 메시지 크기에 대해 소수의 패킷을 잃습니다.

CycloneDDS and FastDDS predominantly lose packets for Struct16 in this case, which is presumably due to the fact that this packet is not fragmented.
CycloneDDS와 FastDDS는 이 경우 주로 Struct16에 대한 패킷을 잃습니다. 이는 아마도 이 패킷이 조각화되지 않았기 때문일 것입니다.

This means that the loss of a single UDP packet is not noticeable, whereas with fragmented messages there is a higher chance that at least one fragment will arrive and thus trigger a resend [Granados, 2017].
즉, 단일 UDP 패킷의 손실은 눈에 띄지 않는 반면 조각화된 메시지의 경우 하나 이상의 조각이 도착하여 재전송을 트리거할 가능성이 더 큽니다 [Granados, 2017].

Overall, both latency and packet loss are satisfactory, even for more subscribers and larger data packets.
전반적으로 지연 시간과 패킷 손실 모두 더 많은 구독자와 더 큰 데이터 패킷에 대해서도 만족스럽습니다.

This is illustrated again in Figure 9.
이는 그림 9에 다시 설명되어 있습니다.

The boxplot shows the latency per packet size for each of the three DDS.
상자 그림은 세 가지 DDS 각각에 대한 패킷 크기별 지연 시간을 보여줍니다.

The box includes the upper and lower quartiles, the line within the box shows the median latency.
상자에는 상위 및 하위 사분위수가 포함되고 상자 내의 선은 중앙값 지연 시간을 나타냅니다.

The whiskers show the 1.5-fold quartile distance.
수염은 1.5배 사분위수 거리를 나타냅니다.

Neither the quartile distances nor the whiskers are significantly wide in most cases.
대부분의 경우 사분위수 거리나 수염이 크게 넓지 않습니다.

After an initial comparative measurement, CycloneDDS is evaluated again using tracing and an enlarged parameter space as an example, as CycloneDDS appears to be the most reliable, particularly in terms of low packet loss and low latency variance.
초기 비교 측정 후 CycloneDDS는 특히 낮은 패킷 손실 및 낮은 지연 시간 분산 측면에서 가장 안정적인 것으로 보이므로 예로 추적 및 확대된 매개변수 공간을 사용하여 다시 평가됩니다.

Figure 10 again shows the performance of CycloneDDS for 1-N communication.
그림 10은 1-N 통신에 대한 CycloneDDS의 성능을 다시 보여줍니다.

The left-hand plot shows the measurement data for a frequency of 10Hz, while the right-hand plot shows the same measurements with a frequency of 100Hz.
왼쪽 그림은 10Hz 주파수에 대한 측정 데이터를 보여주고 오른쪽 그림은 100Hz 주파수에서 동일한 측정값을 보여줍니다.

The x-axis shows the subscriber distribution (1, 7, 31, and 63), which is further divided according to packet size.
x축은 구독자 분포(1, 7, 31 및 63)를 보여주며 패킷 크기에 따라 더 세분화됩니다.

For the frequency of 10 Hz, the latencies are still below the frequency limit on average, even if the whiskers exceed it, especially for the 2Mb packet.
10Hz 주파수의 경우 수염이 특히 2Mb 패킷에 대해 주파수 제한을 초과하더라도 지연 시간은 여전히 평균적으로 주파수 제한 아래에 있습니다.

For the higher frequency of 100 Hz, the frequency limit of 10 ms is already exceeded for the 1 - 1 communication for the largest packet, and, as the number of subscribers increases, the 512kB and 1Mb packets also exceed this limit.
100Hz의 더 높은 주파수의 경우 가장 큰 패킷에 대한 1-1 통신에 대해 10ms의 주파수 제한이 이미 초과되었으며 구독자 수가 증가함에 따라 512kB 및 1Mb 패킷도 이 제한을 초과합니다.

It is also noticeable in this case that the quartile distance for these measurements is in most cases significantly larger than in the comparative measurements before.
또한 이 경우 이러한 측정에 대한 사분위수 거리가 대부분의 경우 이전의 비교 측정보다 상당히 크다는 점도 눈에 띕니다.

The increased frequency and the larger number of subscribers therefore show a strong influence on this.
따라서 주파수 증가와 구독자 수 증가는 이에 큰 영향을 미칩니다.

It is also worth noting that the latency for 100 Hz, especially for 63 subscribers, shows a lower latency and lower variance.
또한 100Hz, 특히 63명의 구독자에 대한 지연 시간은 더 낮은 지연 시간과 더 낮은 분산을 보여준다는 점도 주목할 가치가 있습니다.

One reason for this behavior is due to the following correlation.
이러한 동작의 한 가지 이유는 다음과 같은 상관 관계 때문입니다.

Figure 11 shows the categorized latency of all received messages per number of subscribers and data size.
그림 11은 구독자 수와 데이터 크기별로 수신된 모든 메시지의 분류된 지연 시간을 보여줍니다.

The lower plot shows the view of the subscribers.
아래쪽 그림은 구독자의 보기를 보여줍니다.

As can be seen in the previous figure, almost every message arrives below the frequency limit of 100ms.
이전 그림에서 볼 수 있듯이 거의 모든 메시지가 100ms의 주파수 제한 아래에 도착합니다.

These are categorized as ”in time” in the plot.
이들은 그림에서 "제시간에"로 분류됩니다.

Only for the more complex configurations messages are occasionally lost or arrive too late.
더 복잡한 구성의 경우에만 메시지가 가끔 손실되거나 너무 늦게 도착합니다.

However, the view of the publishers in the upper plot is conspicuous.
그러나 위쪽 그림에서 게시자의 보기는 눈에 띕니다.

Even for the simplest configuration, the publisher does not manage to send all messages in the given frequency time.
가장 간단한 구성의 경우에도 게시자는 주어진 주파수 시간에 모든 메시지를 보내는 것을 관리하지 못합니다.

This explains why the packet loss on the subscriber side remains so low despite the high load and large packet size.
이는 부하가 높고 패킷 크기가 크더라도 구독자 측의 패킷 손실이 낮게 유지되는 이유를 설명합니다.

The majority of messages are not sent within the measurement window and therefore cannot be received on the subscriber side.
대부분의 메시지는 측정 기간 내에 전송되지 않으므로 구독자 측에서 수신할 수 없습니다.

As this behavior increases for higher frequencies, it is obvious that this is the reason for the better latencies in comparison.
이러한 동작은 더 높은 주파수에 대해 증가하므로 비교에서 더 나은 지연 시간의 이유가 분명합니다.

According to the frequency used and the measurement period, 600 messages should be sent in each configuration, but this is only possible for the two smallest packets in most cases.
사용된 주파수와 측정 기간에 따르면 각 구성에서 600개의 메시지를 보내야 하지만 대부분의 경우 두 개의 가장 작은 패킷에 대해서만 가능합니다.

From 512kB upwards, the messages are increasingly delayed so that the total number of 600 is no longer reached in the measurement period.
512kB 이상부터는 메시지가 점점 더 지연되어 측정 기간에 총 600개에 도달하지 않습니다.

Tracing can be used to determine where these delays occur. Figure 12 shows this broken down by the various layers of the ROS2 architecture on side of the publisher, again categorized by subscriber and data size.
추적을 사용하여 이러한 지연이 발생하는 위치를 확인할 수 있습니다.

It is clear here that by far the most time is required at the DDS level, since the message is serialized and prepared for transmission at this level.
그림 12는 구독자와 데이터 크기별로 다시 분류된 게시자 측의 ROS2 아키텍처의 다양한 계층으로 나누어 이를 보여줍니다. 여기서 메시지가 이 수준에서 직렬화되고 전송을 위해 준비되므로 DDS 수준에서 훨씬 더 많은 시간이 필요하다는 것이 분명합니다.

It is already established that the serialization process takes a significant amount of time [Wang et al., 2018], especially as the message format of ROS2 and DDS is not uniform and therefore each requires its own processing.
직렬화 프로세스에는 상당한 시간이 걸린다는 것이 이미 확립되어 있습니다 [Wang et al., 2018]. 특히 ROS2와 DDS의 메시지 형식이 균일하지 않아 각각 고유한 처리가 필요하기 때문입니다.

The figure also shows that the effect increases primarily with the data size, which also points to the serialization step.
이 그림은 또한 효과가 주로 데이터 크기에 따라 증가한다는 것을 보여줍니다. 이는 직렬화 단계를 나타냅니다.

The comparison between topologies 2 and 3 is also relevant: on the one hand, a publisher serves a larger number of subscribers, and, conversely, a large number of publishers serve a single node with many subscribers.
토폴로지 2와 3의 비교도 관련이 있습니다. 한편으로 게시자는 더 많은 수의 구독자에게 서비스를 제공하고 반대로 많은 수의 게시자는 많은 구독자가 있는 단일 노드에 서비스를 제공합니다.

This evaluates the efficiency of the executor in particular, in this case the SingleThreadedExecutor.
이는 특히 실행기의 효율성을 평가합니다. 이 경우 SingleThreadedExecutor입니다.

Figure 13 shows this comparison: The left-hand side shows the latencies for 1-N communication, while the right-hand side shows the N-1 scenario.
그림 13은 이 비교를 보여줍니다. 왼쪽은 1-N 통신의 지연 시간을 보여주고 오른쪽은 N-1 시나리오를 보여줍니다.

Both scenarios are almost identical, especially for 7 and 31 subscribers.
두 시나리오는 특히 7명과 31명의 구독자에게 거의 동일합니다.

For 63 subscribers, however, the difference is noticeably greater.
그러나 63명의 구독자의 경우 차이가 눈에 띄게 커집니다.

Though, the influence of the ratio of publishers and subscribers on latency does not have a significant impact in general.
그러나 지연 시간에 대한 게시자와 구독자 비율의 영향은 일반적으로 큰 영향을 미치지 않습니다.

The significantly greater variance for the N-1 scenario is due to the executor, as it processes all callbacks sequentially and therefore cannot process the open events quickly enough, especially under high load.
N-1 시나리오에 대한 상당히 큰 분산은 실행기 때문입니다. 모든 콜백을 순차적으로 처리하므로 특히 부하가 높은 경우 열려 있는 이벤트를 충분히 빠르게 처리할 수 없기 때문입니다.

Finally, it is checked whether the subscriber fairness described in [Maruyama et al., 2016], which is one essential change compared to ROS1, can also withstand more complex scenarios.
마지막으로 ROS1에 비해 한 가지 중요한 변경 사항인 [Maruyama et al., 2016]에 설명된 구독자 공정성이 더 복잡한 시나리오에도 견딜 수 있는지 확인합니다.

Figure 14 shows this case as an example for a 1-N scenario, a packet size of 64kB, and a frequency of 10 Hz.
그림 14는 1-N 시나리오, 64kB의 패킷 크기 및 10Hz의 주파수에 대한 예로 이 경우를 보여줍니다.

Even if the latency for each subscriber differs slightly, they are on average max. 2 ms apart.
각 구독자의 지연 시간이 약간씩 다르더라도 평균적으로 최대 2ms 차이가 납니다.

There is also no staircase like increase for the subscribers, all have a latency of around 7 ms.
또한 구독자에 대한 계단식 증가는 없으며 모두 약 7ms의 지연 시간을 갖습니다.

The variance for this scenario is greater than for smaller scenarios, but here too none of the subscribers are significantly further apart than the others.
이 시나리오의 분산은 더 작은 시나리오보다 크지만 여기에서도 구독자가 다른 구독자보다 크게 떨어져 있지 않습니다.

This plot is comparable for all other combinations of parameters evaluated.
이 그림은 평가된 다른 모든 매개변수 조합에 대해 비교할 수 있습니다.

This leads to the assumption that other influencing factors, such as frequency and size, have a stronger negative influence earlier, and therefore the performance collapses before the subscriber fair behavior can no longer be maintained.
이로 인해 주파수 및 크기와 같은 다른 영향 요소가 더 일찍 더 강한 부정적인 영향을 미치므로 구독자 공정 동작을 더 이상 유지할 수 없기 전에 성능이 저하된다는 가정이 생깁니다.

## 6. SUMMARY
As the development of automated vehicles is an ongoing research task, this paper presents an evaluation of a ROS2 based ADS.
자율 주행 차량 개발은 진행 중인 연구 과제이므로 본 논문에서는 ROS2 기반 ADS에 대한 평가를 제시합니다.

The automated Mercedes E-Class of Fraunhofer FOKUS comprises both hardware and software components.
프라운호퍼 FOKUS의 자동화된 메르세데스 E-Class는 하드웨어 및 소프트웨어 구성 요소를 모두 포함합니다.

The hardware setup consists of the sensor installation, the on-board PU for processing and planning, and the actuation hardware to control the vehicle.
하드웨어 설정은 센서 설치, 처리 및 계획을 위한 온보드 PU, 차량을 제어하기 위한 작동 하드웨어로 구성됩니다.

With a sensor rig, several cameras and LIDAR sensors are mounted on the roof of the vehicle.
센서 리그를 사용하여 여러 대의 카메라와 LIDAR 센서가 차량 지붕에 장착됩니다.

For vehicle control, a Drive-by-Wire system by Schaeffler Paravan is installed.
차량 제어를 위해 Schaeffler Paravan의 Drive-by-Wire 시스템이 설치됩니다.

The software components of the architecture are split in three segments: sensing, planning, and acting.
아키텍처의 소프트웨어 구성 요소는 감지, 계획 및 작동의 세 부분으로 나뉩니다.

The complexity of the distributed nature of the ADS leads to the research question, if ROS2 fulfills the performance requirements for automated driving.
ADS의 분산 특성의 복잡성으로 인해 ROS2가 자율 주행에 대한 성능 요구 사항을 충족하는지 여부에 대한 연구 질문이 발생합니다.

Thus, a thorough analysis of ROS2 is performed for this paper.
따라서 본 논문에서는 ROS2에 대한 철저한 분석을 수행합니다.

Two important aspects to consider are the latency, which measures the elapsed time between sending and receiving a message, and the packet loss, which measures the percentage of lost messages.
고려해야 할 두 가지 중요한 측면은 메시지 전송과 수신 사이의 경과 시간을 측정하는 지연 시간과 손실된 메시지의 백분율을 측정하는 패킷 손실입니다.

The data size per message, number of messages sent per time unit, number of nodes, number of topics, and number of publishers and subscribers per topic are parameters of interest for the measurements.
메시지당 데이터 크기, 시간 단위당 전송되는 메시지 수, 노드 수, 주제 수, 주제당 게시자 및 구독자 수는 측정에 중요한 매개변수입니다.

Different DDS implementations are compared using a performance test framework, and one selected DDS is further evaluated using tracing to identify performance losses.
성능 테스트 프레임워크를 사용하여 서로 다른 DDS 구현을 비교하고 성능 손실을 식별하기 위해 추적을 사용하여 선택한 DDS를 추가로 평가합니다.

The subscribers and publishers are started accordingly for of the three official DDS systems, FastDDS, CycloneDDS, and RTI Connext.
구독자와 게시자는 세 가지 공식 DDS 시스템인 FastDDS, CycloneDDS 및 RTI Connext에 따라 시작됩니다.

Only the layers below the rmw layer are inspected for the comparative benchmark to isolate the influence of the DDS implementation.
DDS 구현의 영향을 분리하기 위해 비교 벤치마크에 대해 rmw 계층 아래의 계층만 검사합니다.

In the detailed benchmark tracing is used to track the message progress through the complete stack and to understand latency and message losses.
자세한 벤치마크에서 추적은 전체 스택을 통한 메시지 진행 상황을 추적하고 지연 시간과 메시지 손실을 이해하는 데 사용됩니다.

Three different publish/subscriber topologies are assessed.
세 가지 게시/구독 토폴로지가 평가됩니다.

The first one has a 1-1 relation between publishers and subscribers and topics, respectively.
첫 번째는 게시자와 구독자 및 주제 간에 각각 1-1 관계가 있습니다.

The next one has a 1-N publisher-subscriber relation, while the last one reverses this relation.
다음은 1-N 게시자-구독자 관계가 있는 반면 마지막은 이 관계를 반전시킵니다.

They show comparable results for latency, error rate, and bandwidth.
지연 시간, 오류율 및 대역폭에 대해 비슷한 결과를 보여줍니다.

Latency depends mainly on packet size and number of nodes in the system.
지연 시간은 주로 패킷 크기와 시스템의 노드 수에 따라 달라집니다.

With high load, fragmentation of messages can lead to a lower packet loss.
부하가 높으면 메시지 조각화로 인해 패킷 손실이 줄어들 수 있습니다.

In general, packet loss is very low in the tested configurations.
일반적으로 테스트된 구성에서 패킷 손실은 매우 낮습니다.

A large part of the latency is generated on the publisher side before the actual sending and does not count into the transmission time.
지연 시간의 대부분은 실제 전송 전에 게시자 측에서 생성되며 전송 시간에 포함되지 않습니다.

However, this affects the performance of the system, especially for high frequencies and large packet sizes.
그러나 이는 특히 높은 주파수와 큰 패킷 크기의 경우 시스템 성능에 영향을 미칩니다.

Latency remains very similar, comparing different topologies.
서로 다른 토폴로지를 비교하면 지연 시간이 매우 유사하게 유지됩니다.

Only in the n:1 scenario, the average latency is not changing much, while the variance increases significantly, due to the single threaded execution.
n:1 시나리오에서만 평균 지연 시간은 크게 변경되지 않지만 단일 스레드 실행으로 인해 분산이 크게 증가합니다.

In the 1:n scenario it can be observed that subscribers are served in a fair manner, and all have similar latency results.
1:n 시나리오에서는 구독자에게 공정한 방식으로 서비스가 제공되고 모두 유사한 지연 시간 결과를 갖는다는 것을 알 수 있습니다.

As overall both latency and packet loss are low in all tested setups, ROS2 proves as an efficient and reliable communication framework for an ADS.
테스트된 모든 설정에서 지연 시간과 패킷 손실이 모두 낮기 때문에 ROS2는 ADS에 효율적이고 안정적인 통신 프레임워크임이 입증되었습니다.

It should of course be noted that ROS2 does not support hard real time rigor.
물론 ROS2는 하드 실시간 엄격성을 지원하지 않는다는 점에 유의해야 합니다.

However, for the majority of communication, where low latency but no strict real-time capability is mandatory, it is a flexible communication framework that can be used to connect the components within an ADS.
그러나 낮은 지연 시간이 필요하지만 엄격한 실시간 기능이 필수는 아닌 대부분의 통신의 경우 ADS 내의 구성 요소를 연결하는 데 사용할 수 있는 유연한 통신 프레임워크입니다.

However, hard real time rigor should be implemented at least for the actuators.
그러나 하드 실시간 엄격성은 최소한 액추에이터에 대해 구현되어야 합니다.

Many other interesting measurements could be considered due to the various customization options available for ROS2 and the underlying DDS.
ROS2 및 기본 DDS에 사용할 수 있는 다양한 사용자 지정 옵션으로 인해 다른 많은 흥미로운 측정을 고려할 수 있습니다.

This includes the influence of Quality of Service (QoS) profiles and their reliability with the coverage of various bandwidths, which is relevant for automated driving systems.
여기에는 자율 주행 시스템과 관련된 다양한 대역폭 범위를 갖는 서비스 품질(QoS) 프로필의 영향과 안정성이 포함됩니다.

Additionally, there are different execution models from single thread to multi threaded.
또한 단일 스레드에서 다중 스레드까지 다양한 실행 모델이 있습니다.

This may require higher hardware requirements but can lead to significant performance improvements.
이를 위해 더 높은 하드웨어 요구 사항이 필요할 수 있지만 성능이 크게 향상될 수 있습니다.

Another leverage point for performance is the specific configuration of the DDS used.
성능을 위한 또 다른 레버리지 포인트는 사용되는 DDS의 특정 구성입니다.

All three implementations offer extensive options to adapt behavior for the scenario.
세 가지 구현 모두 시나리오에 대한 동작을 조정할 수 있는 광범위한 옵션을 제공합니다.

For example, using Shared Memory (SHMEM) instead of UDP can avoid fragmentation of large messages and can reduce overall load.
예를 들어 UDP 대신 공유 메모리(SHMEM)를 사용하면 큰 메시지의 조각화를 방지하고 전체 부하를 줄일 수 있습니다.

The Towards Zero Copy (TZC) technique presented in a study [Wang et al., 2018] eventually eliminates the overhead of serializing and copying messages.
연구에서 제시된 제로 카피(TZC) 기술 [Wang et al., 2018]은 결국 메시지 직렬화 및 복사의 오버헤드를 제거합니다.
