+++
title = 'ISTQB CT-AI 1장'
date = 2024-10-01T08:16:04+09:00
draft = false
+++

이 글은 ISTQB의 CT-AI(Certified Tester AI Testing) 자격증을 준비하면서 한국어 실라버스를 참고하여 정리 요약한 글입니다.
실라버스는 아래 링크에서 다운 받을 수 있습니다.

https://www.kstqb.org/board_skin/board_view.asp?idx=718&page=1&bbs_code=4&key=0&word=&etc=

## 1. 인공지능 소개
### 1.1 인공지능의 정의와 인공지능 효과
인공지능이라는 용어는 1950년대 처음 사용되었으며 인간을 모방할 수 있는 "지능형" 기계를 만들고 프로그래밍 하는 것을 목적으로 했다.

많이 변화된 현재의 정의는 다음과 같다.

> 엔지니어링 된 시스템이 지식과 기술을 습득, 처리, 적용할 수 있는 능력

어떤 요소를 가져야 인공지능인가에 대한 인식의 변화를 "인공지능 효과"라고 한다.

### 1.2 약인공지능, 강인공지능, 초인공지능
* 약인공지능 시스템

제한된 환경에서 특정 작업을 수행하도록 프로그래밍 되어 있다. 예) 게임 플레이 시스템, 스팸 필터, 테스트 케이스 생성기, 음성 도우미 등

* 강인공지능 시스템

인간과 유사한 일반적인 인지 능력을 갖추고 있다. 이런 인공지능기반 시스템은 인간처럼 환경을 추론하고 이해할 수 잇으며 그에 따라 행동할 수 있다.

* 초인공지능 시스템

인간의 인지를 복제할 수 있고 방대한 처리 능력 사실상 무제한의 기억 능력 그리고 모든 인간의 지식을 사용할 수 있다. 인공지능기반 시스템이 강인공지능에서 초인공지능으로 전환되는 시점을 일반적으로 기술적 특이점(technological singularity)이라고 한다.

### 1.3 인공지능기반 시스템과 전통적 시스템
* 일반적인 전통 컴퓨터 시스템

소프트웨어는 사람이 if-then-else 및 루프와 같은 구문을 포함한 명령형 언어를 사용해 프로그래밍하게 된다. 시스템이 입력을 출력으로 변환하는 방법을 사람이 비교적 쉽게 이해할 수 있다.

* 기계학습을 사용하는 인공지능 기반 시스템

데이터의 패턴을 사용해 미래의 새로운 데이터에 어떻게 반응할지 결정하게 된다. 대부분 인공지능기반 시스템의 이런 예측 절차는 인간이 이해하기 쉽지 않다.

### 1.4 인공지능 기술
인공지능은 다음과 같은 다양한 기술을 사용해 구현할 수 있다.
* 퍼지 논리
* 검색 알고리즘
* 추론 기법
  * 규칙 엔진
  * 연역적 분류기
  * 사례 기반 추론
  * 절차적 추론
* 기계학습
  * 신경망
  * 베이지안 모델
  * 결정 트리
  * 결정 트리
  * 랜덤 포레스트
  * 선형 회귀
  * 로지스틱 회귀
  * 클러스터링 알고리즘
  * 유전 알고리즘
  * 서포트 벡터 머신

### 1.5 인공지능 개발 프레임워크
### 1.6 인공지능기반 시스템을 위한 하드웨어
기계학습에는 일반적으로 다음과 같은 특성을 지원하는 하드웨어가 필요하다.
* 낮은 정밀도 산술이 가능한 하드웨어(예: 32비트가 아닌 8비트 계산)
* 대규모 데이터 구조(예: 행렬 곱셈 지원) 처리 능력이 있는 하드웨어
* 대규모 병렬(동시) 처리가 가능한 하드웨어

**CPU** 일반적으로 기계학습 애플리케이션에 불필요한 복잡한 작업을 지원하며 제한된 몇 개의 코어만 가지고 있다. GPU에 비해 기계핛브 모델을 훈련하고 실행하는데 효율적이지 못하다.

**GPU** 클럭은 CPU보다 느리지만 보통 수천 개의 코어를 가지며 이미지 처리와 같은 대규모 병렬 처리가 필요하지만 상대적으로 간단한 작업을 하도록 설계된다.

**ASIC** 인공지능 전용 솔루션으로 설계되며 다수의 코어, 특별한 데이터 관리, 인-메모리 처리 등의 기능을 갖추고 있다.

### 1.7 서비스형 인공지능
#### 1.7.1 서비스형 인공지능 계약
이런 인공지능 서비스는 일반적으로 비 인공지능 클라우드 기반 서비스형 소프트웨어(SaaS)와 유사한 계약을 통해 제공된다.
#### 1.7.1 서비스형 인공지능 사례
### 1.8 사전 훈련 모델
#### 1.8.1 사전 훈련 모델
기계학습 모델을 훈련하느 데 많은 비용이 소모된다. 데이터를 준비하고 모델을 훈련해야 하는데 전자는 많은 인적 자원을 소비하고 후자는 많은 컴퓨팅 자원을 소비한다.

더 저렴하고 때에 따라 더 효과적인 대안은 사전 훈련된 모델을 사용하는 것이다. 이런 모델은 신경망과 랜덤 포레스트 등 몇 가지 기법에 대해서만 존재한다. 

#### 1.8.2 전이 학습
사전 훈련 모델을 가져와 다른 요구사항을 수행하도록 수정할 수 있는데 이것을전이학습이라 한다.

딥러닝 신경망은 초기층이 일반적으로 매우 기본적인 작업(예: 이미지 분류기에서 직선과 곡선의 차이 식별)을 수행하는 반면 후기 층은 보다 구체적인 작업(예: 건축물 유형 구분)을 수행한다. 이런 경우 이미지 분류기의 후기층을 제외한 모든 층을 재사용할 수 있으므로 초기 층을 다시 훈련할 필요가 없으며 후기 층은 새로운 분류기의 고유한 요구사항을 처리하도록 다시 훈련시킨다.

#### 1.8.3 사전 훈련 모델과 전이 학습 사용 리스크
사전 훈련 모델과 전이 학습을 사용하는 것은 모두 인공지능기반 시스템을 구축하는 일반적인 접근법이지만 여기에는 다음과 같은 몇 가지 리스크가 있다.
* 사전 훈련 모델은 내부적으로 제작한 모델에 비해 투명성이 부족할 수 있다.
* 사전 훈련 모델이 수행하는 기능과 필요한 기능간의 유사도가 충분하지 않을 수도 있다.
* 사전 훈련 모델의 초기 개발시 사용한 데이터 준비절차와 ㅁ이 모델을 새로운 시스템에 적용할 때 사용한 데이터 준비 절차의 차이가 기능적 성능에 영향을 미칠 수 있다.
* 사전 훈련 모델의 단점은 그것을 사용하는 사람들에게 승계될 가능성이 높으며 문서화되지 않았을 수 있다.
* 전이 학습을 통해 만들어진 모델은 그 기반이 되는 사전 훈련 모델과 동일한 취약성에 민감할 가능성이 높다.
* 위의 리스크 중 다수는 사전 훈련 모델에 대한 상세 문서의 제공을 통해 쉽게 완화할 수 있다는 점을 인지할 필요가 있다.

### 1.9 표준, 규정, 인공지능
인공지능 소위원회(ISO/IEC JTC 1/SC42)가 2017년에 설치되어 소프트웨어 및 시스템 엔지니어링을 다루는 ISO/IEC JTC 1/SC7에서는 "인공지능기반 시스템 테스트"에 대한 기술 보고서를 발간했다.

인공지능이 안전 관련 시스템에 사용되는 경우 고나련 규제 표준이 적용된다. 자동차 시스템에 대한 ISO 26262, ISO/PAS 21448(SOTIF) 등이 이런 예가 될 수 있다. 이런 규제 표준은 일반적으로 정부 기관에 의해 의무화되며 포함된 소프트웨어가 ISO 26262를 준수하지 않을 경우 일부 국가에서 해당 소프트웨어가 포함된 자동차를 판매하는 것은 불법으로 간주된다.