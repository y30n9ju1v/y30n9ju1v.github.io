+++
title = 'ROS2 basics'
date = 2024-09-04T08:05:25+09:00
draft = false
+++

### 1. Node

#### 정의

* ROS2 시스템에서 실행되는 독립적인 프로그램

* 일반적으로 각 노드는 특정한 작업을 수행하며, 서로 통신하면서 협력하여 더 큰 로봇 애플리케이션을 형성함

* 예를 들어, 하나의 노드는 카메라 센서로부터 이미지를 수집하고, 다른 노드는 이미지를 처리하며, 또 다른 노드는 이동 명령을 내리는 등 다양한 역할을 할 수 있음

#### 특징

* 독립적 실행

    각 노드는 자체적인 프로세스로 실행되며, 하나의 노드가 종료되거나 충돌하더라도 다른 노드에는 영향을 미치지 않음

* 통신 메커니즘

    노드는 주로 토픽(Topics), 서비스(Services), 액션(Actions)을 통해 통신함

    토픽(Topics): 퍼블리셔-서브스크라이버(pub-sub) 모델을 사용하여 데이터를 비동기적으로 주고받음

    서비스(Services): 요청-응답 모델을 사용하여 동기식 통신을 함

    액션(Actions): 장기적인 작업을 위한 비동기적 요청-응답 모델로, 피드백과 상태 업데이트를 받을 수 있음

* 구성 파일

    노드는 launch 파일을 사용하여 구성할 수 있음

    이 파일은 여러 노드를 한꺼번에 실행하거나, 파라미터를 설정하는 데 사용됨

### 2. Topic

#### 정의

* ROS2 시스템에서 노드가 데이터를 퍼블리시(발행)하고 구독(서브스크라이브)할 수 있는 이름 기반의 통신 채널

* 데이터를 송신하려는 노드는 특정 토픽에 메시지를 퍼블리시하고, 데이터를 수신하려는 노드는 같은 토픽을 서브스크라이브함

* 이러한 토픽 기반 통신은 노드 간 결합도를 낮추어 유연한 시스템 구성이 가능하게 함

#### 특징

* 퍼블리셔-서브스크라이버 모델

    노드는 토픽을 통해 데이터를 주고받기 위해 퍼블리셔 또는 서브스크라이버가 됨

	퍼블리셔(Publisher): 특정 토픽에 데이터를 발행하는 노드

	서브스크라이버(Subscriber): 특정 토픽에서 데이터를 수신하는 노드

* 비동기 통신

    토픽 기반 통신은 비동기적으로 이루어지기 때문에 퍼블리셔와 서브스크라이버 간의 직접적인 연결이 필요하지 않음

* 다대다 통신

    하나의 퍼블리셔가 여러 서브스크라이버와 통신할 수 있고, 반대로 하나의 서브스크라이버가 여러 퍼블리셔로부터 데이터를 받을 수 있음

* 메시지 타입

    토픽은 특정한 메시지 타입을 가지며, 퍼블리셔와 서브스크라이버는 동일한 메시지 타입을 사용해야 함

    예를 들어, std_msgs/String 타입의 메시지를 발행하는 퍼블리셔와 구독하는 서브스크라이버는 모두 std_msgs/String 타입을 사용해야 함

### 3. Service

#### 정의

* 서비스는 하나의 노드가 요청을 보내면 다른 노드가 그 요청을 처리하고 응답하는 형태의 통신 방식

* 서비스는 요청(Request)와 응답(Response)이라는 두 가지 메시지 타입을 사용하여 통신함

* 일반적으로 서비스는 로봇 시스템에서 상태 정보를 요청하거나 명령을 실행하는 데 사용됨. 예를 들어, 로봇 팔의 특정 위치로 이동을 명령하는 서비스, 센서 데이터를 요청하는 서비스 등이 있음

#### 특징

* 동기식 통신

    서비스는 요청을 보낸 노드가 응답을 받을 때까지 기다리기 때문에 동기적으로 작동함

* 요청-응답 모델

    서비스는 클라이언트(Client)와 서버(Server) 모델을 따름

    서비스 서버(Service Server): 요청을 처리하고 응답을 반환하는 노드

    서비스 클라이언트(Service Client): 요청을 보내고 응답을 기다리는 노드

* 명확한 통신 종료

    서비스는 요청-응답을 통해 통신이 완료되므로, 특정 작업의 성공 또는 실패 여부를 알 수 있음

* 사용 사례

    로봇이 특정 동작을 수행해야 하는 상황(예: 문 열기, 특정 좌표로 이동 등)이나 특정 데이터를 요청하고 즉시 응답을 받아야 하는 상황에서 사용됨

#### 서비스와 토픽의 차이점

* 통신 방식

    토픽은 비동기식(pub-sub) 방식으로 다대다 통신을 지원하며, 서비스는 동기식(request-response) 방식으로 일대일 통신을 지원함

* 사용 목적

    토픽은 지속적인 데이터 스트리밍에 적합하고, 서비스는 단발성 요청과 응답이 필요한 작업에 적합함

* 통신 종료

    토픽은 지속적으로 데이터를 발행할 수 있는 반면, 서비스는 요청이 완료되면 통신이 종료됨

### 4. Parameter

#### 정의

* 파라미터는 노드의 동작을 구성하고 제어하기 위해 사용되는 키-값 쌍의 설정

* 파라미터는 특정 노드에 속해 있으며, 각 노드는 자신만의 파라미터 집합을 가질 수 있음

* 예를 들어, 로봇의 속도, 센서의 업데이트 주기, 로봇 팔의 최대 속도와 같은 값들을 파라미터로 설정할 수 있음

#### 특징

* 동적 구성 가능

    노드는 실행 중에 파라미터를 설정하거나 변경할 수 있어, 로봇 시스템의 동적 제어와 구성이 가능함

* 타입이 있는 키-값 쌍

    파라미터는 특정 데이터 타입을 가짐. 지원되는 타입은 bool, int, double, string, byte[], bool[], int[], double[], string[] 등 임

* 퍼시스턴스

    파라미터는 노드가 실행되는 동안 유지되며, 노드가 종료되면 사라짐. 필요할 경우, 파라미터 파일을 통해 영구적으로 저장하고 사용할 수 있음

#### 파라미터와 노드 간의 통합

* ROS2의 파라미터는 노드의 구성 요소로서, 개발자가 로봇의 동작을 유연하게 제어하고 필요에 따라 설정을 변경할 수 있도록 함. 이는 로봇 시스템의 확장성과 유지보수성을 높이는 데 중요한 역할을 함

* ROS2의 파라미터 기능은 로봇 애플리케이션의 동작을 조정하고 실시간으로 구성할 수 있는 강력한 도구로, 이를 통해 복잡한 로봇 시스템을 효과적으로 제어할 수 있음

### 5. Action

#### 정의

* 액션은 노드가 요청한 작업이 완료될 때까지 시간이 걸릴 수 있는 경우에 사용되는 통신 메커니즘

* 요청(Request), 피드백(Feedback), **결과(Result)**의 3가지 타입의 메시지를 사용하여 작업의 시작, 진행 상황, 완료를 관리함

* 예를 들어, 로봇의 특정 위치로의 이동, 매핑, 물체 인식 등의 장기 작업을 수행할 때 사용됨

#### 특징

* 비동기적 통신

    액션은 비동기적으로 작동하며, 요청을 보내고 응답을 기다리는 동안 클라이언트는 다른 작업을 수행할 수 있음

* 상태 관리

    액션은 목표의 상태를 지속적으로 업데이트하며, 작업의 진행 상황을 피드백으로 제공할 수 있음

* 취소와 재시도

    클라이언트는 요청한 작업을 취소하거나 재시도할 수 있음

* 구성 요소:

    액션 서버(Action Server): 작업을 수행하고 결과를 반환하는 노드

	액션 클라이언트(Action Client): 작업을 요청하고 결과를 기다리는 노드

#### 활용 사례

* 로봇의 특정 위치로 이동

    로봇에게 목표 위치로 이동하라는 명령을 내리고 이동하는 동안 경로와 상태를 피드백으로 받을 수 있음

* 물체 잡기 작업

    로봇 팔이 물체를 잡을 때 그 작업이 완료될 때까지 진행 상황을 지속적으로 피드백 받을 수 있음

* 장기적인 데이터 처리

    예를 들어, 이미지 처리와 같은 작업은 시간이 걸릴 수 있으며, 처리 진행 상황을 알 수 있는 액션을 통해 구현할 수 있음