+++
title = 'DeepGauge 1'
date = 2024-08-07T19:51:13+09:00
draft = false
+++

이 글은 DeepGauge: Multi-Granularity Testing Criteria for Deep Learning Systems (https://arxiv.org/abs/1803.07519)을 번역 및 요약한 글입니다.

## Abstract
Deep learning (DL) defines a new data-driven programming paradigm that constructs the internal system logic of a crafted neuron network through a set of training data.
We have seen wide adoption of DL in many safety-critical scenarios.
However, a plethora of studies have shown that the state-of-the-art DL systems suffer from various vulnerabilities which can lead to severe consequences when applied to real-world applications.

딥러닝(DL)은 훈련 데이터 세트를 통해 설계된 뉴런 네트워크의 내부 시스템 논리를 구성하는 새로운 데이터 중심 프로그래밍 패러다임을 정의합니다.
우리는 많은 안전이 중요한 시나리오에서 DL의 광범위한 채택을 보았습니다.
그러나 수많은 연구들은 최신 DL 시스템이 다양한 취약점을 가지고 있으며, 이것이 실제 응용 프로그램에 적용될 때 심각한 결과를 초래할 수 있음을 보여주었습니다.

Currently, the testing adequacy of a DL system is usually measured by the accuracy of test data.
Considering the limitation of accessible high quality test data, good accuracy performance on test data can hardly provide confidence to the testing adequacy and generality of DL systems.
Unlike traditional software systems that have clear and controllable logic and functionality, the lack of interpretability in a DL system makes system analysis and defect detection difficult, which could potentially hinder its real-world deployment.

현재 DL 시스템의 테스트 적절성은 일반적으로 테스트 데이터의 정확도로 측정됩니다.
접근 가능한 고품질 테스트 데이터의 한계를 고려할 때, 테스트 데이터에서의 우수한 정확도 성능은 DL 시스템의 테스트 적절성과 일반성에 대한 신뢰를 제공하기 어렵습니다.
명확하고 제어 가능한 논리와 기능을 가진 전통적인 소프트웨어 시스템과 달리, DL 시스템의 해석 가능성 부족은 시스템 분석 및 결함 탐지를 어렵게 하여 실제 배포를 저해할 수 있습니다.

In this paper, we propose DeepGauge, a set of multi-granularity testing criteria for DL systems.
The in-depth evaluation of our proposed testing criteria is demonstrated on two well-known datasets, five DL systems, and with four state-of-the-art adversarial attack techniques against DL.

본 논문에서는 DL 시스템을 위한 다중-그레인 테스트 기준 세트인 DeepGauge를 제안합니다.
제안된 테스트 기준의 심층 평가는 두 개의 잘 알려진 데이터셋, 다섯 개의 DL 시스템, 그리고 DL에 대한 네 가지 최신 적대적 공격 기법을 통해 입증됩니다.

## 1. Introduction
DNN-based software systems, such as autonomous driving, often exhibit erroneous behaviors that lead to fatal consequences.
For example, several accidents [21] have been reported due to autonomous vehicle’s failure to handle unexpected/corner-case driving conditions.

자율 주행과 같은 DNN 기반 소프트웨어 시스템은 종종 치명적인 결과를 초래하는 오류 동작을 나타냅니다.
예를 들어, 자율 주행 차량이 예상치 못한 상황이나 예외적인 주행 조건을 처리하지 못해 여러 사고가 보고되었습니다 [21].

One of the trending research areas is to investigate the cause of vulnerability in DL systems by means of generating adversarial test examples for image- and video-based DL systems.
Such carefully learned pixel-level perturbations, imperceptible to human eyes, can cause the DL-based classification system to output completely wrong decisions with high confidence [20].
Ever since the inception of adversarial attacks on the DL systems, more and more research has been dedicated to building up strong attackers [6, 25, 55, 60].
As a consequence, better defense mechanisms in DL systems against adversarial attacks are in dire need.
Various techniques to nullify adversarial attacks and to train a more robust DL system are emerging in recent studies [18, 23, 41, 43, 45, 51, 56].
Together, research in both realms forms a virtuous circle and blazes a trail for better understanding of how to build more generic and robust DL systems.

딥러닝(DL) 시스템의 취약성을 조사하는 유행하는 연구 분야 중 하나는 이미지 및 비디오 기반 DL 시스템을 위한 적대적 테스트 예제를 생성하는 것입니다.
인간의 눈에는 감지할 수 없는 픽셀 수준의 정교하게 학습된 작은 변동은 DL 기반 분류 시스템이 높은 확신을 가지고 완전히 잘못된 결정을 내리게 할 수 있습니다 [20].
DL 시스템에 대한 적대적 공격이 시작된 이후, 더 강력한 공격자를 구축하기 위한 연구가 점점 더 많이 진행되고 있습니다 [6, 25, 55, 60].
그 결과, 적대적 공격에 대한 DL 시스템의 더 나은 방어 메커니즘이 절실히 필요하게 되었습니다.
최근 연구에서는 적대적 공격을 무력화하고 더 견고한 DL 시스템을 훈련시키기 위한 다양한 기술이 등장하고 있습니다 [18, 23, 41, 43, 45, 51, 56].
이 두 분야의 연구가 함께 선순환을 이루며, 더 일반적이고 견고한 DL 시스템을 구축하는 방법에 대한 이해를 높이는 길을 열어줍니다.

However, what is still lacking is a systematic way of gauging the testing adequacy of given DL systems.
Current studies focus only on pursuing high accuracy of DL systems as a testing criterion, for which we show several caveats as follows.

그러나 여전히 부족한 것은 주어진 DL 시스템의 테스트 적절성을 체계적으로 측정하는 방법입니다.
현재 연구들은 DL 시스템의 높은 정확도를 테스트 기준으로 삼는 데 집중하고 있으며, 이에 따른 몇 가지 경고를 다음과 같이 제시합니다.

First, measuring the software quality from DL output alone is superficial in the sense that fundamental understanding of the DL internal neuron activities and network behaviors is not touched upon.
We agree that it could be an indicator of DL system quality and generality, but it is far from complete, and oftentimes unreliable.

첫째, DL 출력만으로 소프트웨어 품질을 측정하는 것은 피상적이며, DL 내부의 뉴런 활동과 네트워크 동작에 대한 근본적인 이해를 다루지 않습니다.
우리는 DL 시스템의 품질과 일반성을 나타낼 수 있는 지표라고 동의하지만, 이는 완전하지 않으며 종종 신뢰할 수 없습니다.

Second, a criterion solely based on DL output will rely heavily on how representative the test data are.
Having achieved high-performance DL output does not necessarily mean that the system is utmost generic, and achieving low-performance does not indicate the opposite either.
A DL model can be immune to many known types of adversarial attacks, but may fail from unseen attacks.
This is because such a criterion based only on DL outputs is far from being comprehensive, and it leaves high risks for currently cocooned DL systems to be deployed in the real-world environment where newly evolved adversarial attacks are inescapable.

둘째, DL 출력에만 기반한 기준은 테스트 데이터의 대표성에 크게 의존합니다.
높은 성능의 DL 출력을 달성했다고 해서 시스템이 가장 일반적이라는 것을 의미하지 않으며, 낮은 성능을 달성했다고 해서 반대라는 것도 아닙니다.
DL 모델은 많은 알려진 유형의 적대적 공격에 면역일 수 있지만, 보이지 않는 공격에는 실패할 수 있습니다.
이는 DL 출력에만 기반한 기준이 포괄적이지 않기 때문이며, 새로운 적대적 공격이 불가피한 현실 세계에 배포될 때 현재 보호된 DL 시스템에 높은 위험을 남깁니다.

Third, any DL system that passes systematic testing should be able to withstand all types of adversarial attacks to some extent.
Such generality upon various attacks is of vital importance for DL systems to be deployed.
But apparently this is not the case, unless we stick to a set of more comprehensive gauging criteria.

셋째, 체계적인 테스트를 통과하는 모든 DL 시스템은 어느 정도 모든 유형의 적대적 공격을 견뎌낼 수 있어야 합니다.
다양한 공격에 대한 일반성은 DL 시스템이 배포되는 데 있어 매우 중요합니다.
그러나 이는 명백히 그렇지 않으며, 더 포괄적인 측정 기준을 따르지 않는 한 가능합니다.

Towards addressing the aforementioned limitations, a set of testing criteria is needed, as opposed to the sole criterion based on DL decision output.
In addition to being scalable, the proposed criteria will have to monitor and gauge the neuron activities and intrinsic network connectivity at various granularity levels, so that a multi-faceted in-depth portrayal of the DL system and testing quality measures become desirable.

앞서 언급한 한계점을 해결하기 위해, DL 결정 출력에만 기반한 단일 기준이 아닌 일련의 테스트 기준이 필요합니다.
제안된 기준은 확장 가능할 뿐만 아니라, 다양한 세분화 수준에서 뉴런 활동과 내재된 네트워크 연결성을 모니터링하고 측정해야 합니다.
이를 통해 DL 시스템의 다면적이고 심층적인 묘사와 테스트 품질 측정이 가능해집니다.

In this work, we are probing this problem from a software engineering and software testing perspective.
At a high level, erroneous behaviors appeared in DNNs are analogous to logic bugs in traditional software.
However, these two types of software are fundamentally different in their designs.
Traditional software represents its logic as control flows crafted by human knowledge, while a DNN characterizes its behaviors by the weights of neuron edges and the nonlinear activation functions (determined by the training data).
Therefore, detecting erroneous behaviors in DNNs is different from detecting those in traditional software in nature, which necessitates novel test generation approaches.

이 연구에서는 소프트웨어 공학 및 소프트웨어 테스트 관점에서 이 문제를 탐구합니다.
높은 수준에서 보면, DNN에서 나타나는 오류 동작은 전통적인 소프트웨어의 논리 버그와 유사합니다.
그러나 이 두 가지 유형의 소프트웨어는 근본적으로 설계 방식에서 다릅니다.
전통적인 소프트웨어는 인간의 지식에 의해 만들어진 제어 흐름으로 논리를 표현하는 반면, DNN은 뉴런 엣지의 가중치와 비선형 활성화 함수(훈련 데이터에 의해 결정됨)로 그 동작을 특징짓습니다.
따라서 DNN에서 오류 동작을 감지하는 것은 전통적인 소프트웨어에서 오류를 감지하는 것과 본질적으로 다르며, 이는 새로운 테스트 생성 접근 방식이 필요함을 의미합니다.

To achieve this goal, the very first step is to precisely define a set of suitable coverage criteria, which can guide test design and evaluate test quality.
Despite a number of criteria existing for traditional software, e.g., statement, branch, data-flow coverage, they completely lose effect in testing DNNs.
To the best of our knowledge, the design of testing coverage criteria for DNNs is still at the early stage [38, 47].

이 목표를 달성하기 위해 첫 번째 단계는 테스트 설계를 안내하고 테스트 품질을 평가할 수 있는 적절한 커버리지 기준을 정확하게 정의하는 것입니다.
전통적인 소프트웨어에 대한 여러 기준(예: 구문, 분기, 데이터 흐름 커버리지)이 존재하지만, DNN을 테스트할 때는 전혀 효과가 없습니다.
우리가 아는 한, DNN을 위한 테스트 커버리지 기준의 설계는 여전히 초기 단계에 있습니다 [38, 47].

Without a comprehensive set of criteria, (1) designing tests to cover different learned logics and rules of DNNs is difficult to achieve.
Consequently, erroneous behaviors may be missed; (2) evaluating test quality is biased, and the confidence of obtained testing results may be overestimated.
In this paper, we propose DeepGauge—a set of testing criteria based on multi-level and -granularity coverage for testing DNNs and measure the testing quality.
Our contributions are summarized as follows:

포괄적인 기준 세트가 없으면, (1) DNN의 다양한 학습된 논리와 규칙을 커버하기 위한 테스트를 설계하기가 어렵습니다.
그 결과, 오류 동작이 놓칠 수 있습니다; (2) 테스트 품질 평가가 편향되며, 얻어진 테스트 결과에 대한 신뢰도가 과대평가될 수 있습니다.
본 논문에서는 DNN을 테스트하고 테스트 품질을 측정하기 위한 다중 수준 및 다중 세분화 커버리지 기준에 기반한 DeepGauge를 제안합니다.
우리의 기여는 다음과 같이 요약됩니다:

* Our proposed criteria facilitate the understanding of DNNs as well as the test data quality from different levels and angles.
In general, we find defects could potentially distribute on both major function regions as well as the corner-case regions of DNNs.
Given a set of inputs, our criteria could measure to what extent it covers the main functionality and the corner cases of the neurons, where DL defects could incur.
Our evaluation results reveal that the existing test data of a given DL in general skew more towards testing the major function region, with relatively few cases covering the corner-case region.

우리의 제안된 기준은 다양한 수준과 관점에서 DNN과 테스트 데이터 품질을 이해하는 데 도움을 줍니다.
일반적으로 결함은 DNN의 주요 기능 영역뿐만 아니라 코너 케이스 영역에도 분포할 수 있습니다.
주어진 입력 세트를 고려할 때, 우리의 기준은 주 기능과 뉴런의 코너 케이스를 어느 정도까지 커버하는지 측정할 수 있습니다.
DL 결함이 발생할 수 있는 부분입니다.
평가 결과, 주어진 DL의 기존 테스트 데이터는 일반적으로 주요 기능 영역 테스트에 치우쳐 있으며, 상대적으로 적은 경우만이 코너 케이스 영역을 커버함을 보여줍니다.

* In line with existing test data of DNNs, we evaluate the usefulness of our coverage criteria as indicators to quantify defect detection ability of test data on DNNs, through generating new adversarial test data using 4 well-known adversarial data generation algorithms (i.e., Fast Gradient Sign Method (FGSM) [20], Basic Iterative Method (BIM) [31], Jacobian-based Saliency Map Attack (JSMA) [37] and Carlini/Wagner attack (CW) [8]).
The extensive evaluation shows that our criteria can effectively capture the difference between the original test data and adversarial examples, where DNNs could and could not correctly recognize, respectively, demonstrating that a higher coverage of our criteria potentially indicates a higher chance to detect the DNN’s defects.

기존 DNN 테스트 데이터를 바탕으로, 우리는 네 가지 잘 알려진 적대적 데이터 생성 알고리즘(Fast Gradient Sign Method (FGSM) [20], Basic Iterative Method (BIM) [31], Jacobian-based Saliency Map Attack (JSMA) [37], Carlini/Wagner attack (CW) [8])을 사용하여 새로운 적대적 테스트 데이터를 생성함으로써, DNN 테스트 데이터의 결함 탐지 능력을 정량화하는 지표로서 우리의 커버리지 기준의 유용성을 평가합니다.
광범위한 평가 결과, 우리의 기준이 원래 테스트 데이터와 DNN이 각각 올바르게 인식할 수 있거나 인식할 수 없는 적대적 예제 간의 차이를 효과적으로 포착할 수 있음을 보여줍니다.
이는 우리의 기준을 높게 커버할수록 DNN의 결함을 탐지할 가능성이 더 높음을 시사합니다.

The various criteria proposed behave differently on DNNs w.r.t. network complexity and dataset under analysis.
Altogether, these criteria can potentially help us gain insights of testing DNNs.
By providing these insights, we hope that both software engineering and machine learning communities can benefit from applying new criteria for gauging the testing quality of the DNNs to gain confidence towards constructing generic and robust DL systems.

제안된 다양한 기준은 네트워크 복잡성과 분석 중인 데이터셋에 따라 DNN에서 다르게 작동합니다.
이러한 기준들은 DNN 테스트에 대한 통찰을 제공하는 데 도움이 될 수 있습니다.
이러한 통찰을 제공함으로써, 소프트웨어 공학 및 기계 학습 커뮤니티가 DNN의 테스트 품질을 측정하기 위한 새로운 기준을 적용하여 일반적이고 견고한 DL 시스템을 구축하는 데 자신감을 가질 수 있기를 바랍니다.

## 2. Preliminaries
### 2.1 Coverage Criteria in Traditional Software Testing
We regard traditional software as any program written in high-level programming languages (e.g., C/C++, Java, Python).
Specially, each statement in traditional program performs some certain operation that either transforms the outputs from the previous statement to the next one or changes the program states (e.g., assign new values to variables).
Software defects (bugs) can be introduced by developers due to incorrect implementation, which may cause unexpected outputs or even fail-stop errors (e.g., program crashes).

전통적인 소프트웨어는 C/C++, Java, Python과 같은 고급 프로그래밍 언어로 작성된 모든 프로그램으로 간주됩니다.
특히, 전통적인 프로그램의 각 문장은 이전 문장의 출력을 다음 문장으로 변환하거나 프로그램 상태를 변경하는(예: 변수에 새로운 값을 할당하는) 특정 작업을 수행합니다.
소프트웨어 결함(버그)은 개발자가 잘못된 구현으로 인해 도입될 수 있으며, 이는 예상치 못한 출력이나 심지어 실패 정지 오류(예: 프로그램 충돌)를 유발할 수 있습니다.

it feeds these test data as inputs to program and validates the correctness of the program’s run-time behavior by comparing the actual outputs with expected ones (test oracles).
The program with higher test coverage often suggests that it has a lower chance of containing defects.
Many software testing standards require a software product to be thoroughly tested with high test coverage before shipment, which is used as an indicator and confidence of the software quality.
For example, ECSS-E-ST-40C [15] standards demand 100% statement coverage of the software under test for two critical levels.

주어진 테스트 데이터를 사용하여 프로그램에 입력으로 제공하고, 실제 출력과 예상 출력(테스트 오라클)을 비교하여 프로그램의 런타임 동작의 정확성을 검증합니다.
더 높은 테스트 커버리지를 가진 프로그램은 결함을 포함할 가능성이 낮다는 것을 시사합니다.
많은 소프트웨어 테스트 표준은 소프트웨어 제품이 출하되기 전에 높은 테스트 커버리지로 철저히 테스트될 것을 요구하며, 이는 소프트웨어 품질의 지표이자 신뢰도로 사용됩니다.
예를 들어, ECSS-E-ST-40C [15] 표준은 두 가지 중요한 수준에서 테스트 중인 소프트웨어의 100% 문장 커버리지를 요구합니다.

For traditional software, a number of coverage criteria have already been defined at different levels, to analyze the software runtime behavior from different perspectives, i.e., code level (e.g., statement, branch, data-flow coverage and mutation testing [27, 46, 62]) or model-level (e.g., state and transition coverage [2, 14]) to cater for different testing methods and granularities.
Some commonly used test coverage criteria are listed as follows:

전통적인 소프트웨어의 경우, 소프트웨어의 런타임 동작을 다양한 관점에서 분석하기 위해 여러 수준에서 다양한 커버리지 기준이 이미 정의되어 있습니다.
즉, 코드 수준(예: 문장, 분기, 데이터 흐름 커버리지 및 변이 테스트 [27, 46, 62]) 또는 모델 수준(예: 상태 및 전이 커버리지 [2, 14])에서 서로 다른 테스트 방법과 세분성을 충족시키기 위해 정의되었습니다.
일반적으로 사용되는 테스트 커버리지 기준은 다음과 같습니다:

* Statement coverage measures whether each instruction has been executed, and branch coverage focuses on whether each branch of control structure (e.g., in if or switch-case statements) has been covered, both of which are control-flow-based criteria.

문장 커버리지는 각 명령어가 실행되었는지를 측정하며, 분기 커버리지는 제어 구조의 각 분기(예: if 또는 switch-case 문)가 커버되었는지를 중점으로 둡니다.
이 두 가지는 모두 제어 흐름 기반 기준입니다.

* Data-flow coverage [46] enforces the coverage of each variable definition and its uses to detect data-flow anomalies.

데이터 흐름 커버리지 [46]는 각 변수 정의와 그 사용을 커버하도록 하여 데이터 흐름 이상을 탐지합니다.

* Model-based coverage criteria [3, 52] aim to cover more program behaviors via abstracted behavior models.
Other comprehensive variants of test coverage could be referred to [2].

모델 기반 커버리지 기준 [3, 52]는 추상화된 동작 모델을 통해 더 많은 프로그램 동작을 커버하는 것을 목표로 합니다.
테스트 커버리지의 다른 포괄적인 변형에 대해서는 [2]를 참조할 수 있습니다.

## 3. Coverage Criteria For Testing DL Systems
Testing coverage criteria is proposed to shatter and approximate the software internal states.
It partitions the input space and establishes the relation of an input subspace and an approximated software internal state.
In this way, compared with the test data from a single input subspace, the same number of test data from different input sub-spaces would have a higher chance to cover more diverse software states, resulting in a higher possibility to detect more diverse software defects.

테스트 커버리지 기준은 소프트웨어의 내부 상태를 분할하고 근사화하기 위해 제안되었습니다.
이는 입력 공간을 분할하고, 입력 하위 공간과 근사화된 소프트웨어 내부 상태 간의 관계를 설정합니다.
이렇게 함으로써, 단일 입력 하위 공간에서 나온 테스트 데이터와 비교하여, 서로 다른 입력 하위 공간에서 나온 동일한 수의 테스트 데이터가 더 다양한 소프트웨어 상태를 커버할 가능성이 높아져, 더 다양한 소프트웨어 결함을 감지할 가능성이 높아집니다.

Over the past decades, a set of well-designed coverage criteria [2] (e.g., statement coverage, branch coverage) have demonstrated their practical value and are widely adopted in software industry to systematically guide the testing process to unveil the software defects at different levels, e.g., (1) Unit level: testing small snippets of functions.
(2) Integration level: testing multiple sub-modules or functions to check their interactions.
(3) System level: testing the software system as a whole.

지난 수십 년 동안, 잘 설계된 일련의 커버리지 기준(예: 구문 커버리지, 분기 커버리지)은 그 실용적 가치를 입증했으며, 소프트웨어 결함을 다양한 수준에서 체계적으로 발견하는 테스트 과정을 안내하기 위해 소프트웨어 업계에서 널리 채택되었습니다.
예를 들어:

1. **단위 수준(Unit level)**: 함수의 작은 코드 조각을 테스트.
2. **통합 수준(Integration level)**: 여러 하위 모듈 또는 함수의 상호작용을 테스트.
3. **시스템 수준(System level)**: 소프트웨어 시스템 전체를 테스트.

이러한 다양한 수준에서의 테스트는 소프트웨어 결함을 효과적으로 드러내기 위한 체계적인 접근 방식을 제공합니다.

The current state-of-the-practice DNN testing, however, is still at its early stage and mainly relies on the prediction accuracy (similar to black-box system level testing that only observes inputs and its corresponding outputs), lacking systematic testing coverage criteria for defect detection.
Furthermore, traditional software and DNNs have obvious differences, so existing coverage criteria for traditional software could not be directly applied to DNNs.

현재의 DNN 테스트 실무는 여전히 초기 단계에 있으며 주로 예측 정확도에 의존하고 있습니다(입력과 그에 상응하는 출력만을 관찰하는 블랙박스 시스템 수준 테스트와 유사함).
이는 결함 감지를 위한 체계적인 테스트 커버리지 기준이 부족하다는 것을 의미합니다.
또한, 전통적인 소프트웨어와 DNN은 명확한 차이점이 있기 때문에 기존의 전통적인 소프트웨어 커버리지 기준을 DNN에 직접 적용할 수 없습니다.

In this section, we design a set of DNN testing coverage criteria from multiple levels, aiming to gauge the testing adequacy of DNNs and facilitate the detection of those erroneous behaviors from multiple portrayals.
To be useful towards industry level applications, we believe that the test criteria should be simple, scalable as well general enough to be applied to a large range of DNNs without confining on specific DNN structure or activation functions.
Conceptually, similar to traditional software, the behaviors of DNNs can be divided into two categories, i.e., major function behaviors and corner-case behaviors, both of which may contain erroneous behaviors (see Figure 1(b) and our evaluation results in Section 4).

이 섹션에서는 DNN의 테스트 적합성을 평가하고 여러 가지 측면에서 잘못된 동작을 감지하기 위해 여러 수준에서 DNN 테스트 커버리지 기준을 설계합니다.
산업 수준의 응용 프로그램에 유용하려면, 테스트 기준은 간단하고 확장 가능하며 특정 DNN 구조나 활성화 함수에 국한되지 않고 다양한 DNN에 일반적으로 적용할 수 있어야 한다고 믿습니다.
개념적으로, 전통적인 소프트웨어와 유사하게 DNN의 동작은 주요 기능 동작과 코너 케이스 동작의 두 가지 범주로 나눌 수 있으며, 두 범주 모두 오류 동작을 포함할 수 있습니다(그림 1(b) 및 섹션 4의 평가 결과 참조).

### 3.1 Neuron-Level Coverage Criteria
At the neuron-level, we use the output values of neuron n determined from the training to characterize its behaviors.
Since the internal logic of a DNN is mostly programmed by training data, intuitively, the functionality (i.e., neuron output) for each neuron of a DNN should follow some statistical distribution that is largely determined by the training data.
The output distribution of a neuron obtained from training data analysis would allow to approximately characterize the major function regions whose output values are often triggered by input data with a similar statistical distribution to the training data, and the corner cases whose output values rarely occur.
we leverage the neuron output value boundaries obtained from training data to approximate the major function region and corner-case region.

뉴런 수준에서, 우리는 뉴런 \(n\)의 동작을 특징짓기 위해 훈련에서 결정된 뉴런의 출력 값을 사용합니다.
DNN의 내부 논리는 대부분 훈련 데이터에 의해 프로그래밍되기 때문에, 직관적으로 각 DNN 뉴런의 기능(즉, 뉴런 출력)은 주로 훈련 데이터에 의해 결정되는 통계적 분포를 따라야 합니다.
훈련 데이터 분석을 통해 얻은 뉴런의 출력 분포는 훈련 데이터와 유사한 통계적 분포를 가진 입력 데이터에 의해 자주 발생하는 주요 기능 영역과 드물게 발생하는 코너 케이스를 대략적으로 특징짓는 데 도움이 됩니다.
우리는 훈련 데이터에서 얻은 뉴런 출력 값의 경계를 활용하여 주요 기능 영역과 코너 케이스 영역을 대략적으로 추정합니다.

Specially, for a neuron \(n\), let \(\text{high}_n\) and \(\text{low}_n\) be its upper and lower boundary output values, respectively, on the value range of its activation function, where \(\text{high}_n\) and \(\text{low}_n\) are derived from the training dataset analysis.
We refer to [\(\text{low}_n\), \(\text{high}_n\)] as the major function region of a neuron \(n\).

특히, 뉴런 \(n\)에 대해, \(\text{high}_n\) 과 \(\text{low}_n\)을 해당 활성화 함수의 값 범위에서의 상한과 하한 출력 값으로 정의합니다.
이때 \(\text{high}_n\)과 \(\text{low}_n\)은 훈련 데이터셋 분석에서 도출됩니다.
우리는 \([\text{low}_n, \text{high}_n]\)을 뉴런 \(n\)의 주요 기능 영역이라고 부릅니다.

Definition 3.1. For a test input x ∈ T, we say that a DNN is located in its **major function regiong** given x iff ∀n ∈ N :φ(x,n) ∈ [lown , highn ].

정의 3.1. 테스트 입력 \(x \in T\)에 대해, \(∀n ∈ N\)에 대해 \(\phi(x,n) ∈ [\text{low}_n , \text{high}_n]\)이면 DNN이 \(x\)에 대해 주요 기능 영역에 위치한다고 말합니다.

To exhaustively cover the major function regions, we partition [lown , highn ] into k sections, and require each of them to be covered by the test inputs.
We name this coverage as k-multisection neuron coverage.

주요 기능 영역을 철저하게 커버하기 위해, \([\text{low}_n , \text{high}_n]\)을 \(k\)개의 구간으로 나누고, 각 구간이 테스트 입력에 의해 커버되도록 요구합니다.
우리는 이 커버리지를 \(k\)-멀티섹션 뉴런 커버리지라고 명명합니다.

(i) k-multisection Neuron Coverage. Given a neuron n, the k-multisection neuron coverage measures how thoroughly the given set of test inputs T covers the range [lown , highn ]. To quantify this, we divide the range [lown , highn ] into k equal sections (i.e., k-multisections), for k > 0. We write Sin to denote the set of values inthei-thsectionfor1≤i ≤k.

(i) \(k\)-멀티섹션 뉴런 커버리지. 주어진 뉴런 \(n\)에 대해, \(k\)-멀티섹션 뉴런 커버리지는 주어진 테스트 입력 집합 \(T\)가 \([\text{low}_n , \text{high}_n]\) 범위를 얼마나 철저하게 커버하는지를 측정합니다. 이를 정량화하기 위해, \([\text{low}_n , \text{high}_n]\) 범위를 \(k > 0\)인 경우 \(k\)개의 동일한 구간(즉, \(k\)-멀티섹션)으로 나눕니다. 우리는 \(S_i^n\)을 \(1 \leq i \leq k\)일 때, \(i\)번째 구간 내의 값들의 집합을 나타내는 것으로 표기합니다.

If φ(x,n) ∈ Sin, we say the i-th section is covered by the test input x. Therefore, for a given set of test inputs T and the neuron n, its k-multisection neuron coverage is defined as the ratio of the number of sections covered by T and the total number of sections, i.e., k in our definition. We define the k-multisection coverage of a neuron n as:

\(\phi(x,n) \in S_i^n\)이면, i번째 구간이 테스트 입력 x에 의해 커버되었다고 합니다. 따라서, 주어진 테스트 입력 집합 T와 뉴런 n에 대해, k-멀티섹션 뉴런 커버리지는 T에 의해 커버된 구간의 수와 전체 구간 수(즉, 우리의 정의에서는 k)의 비율로 정의됩니다. 뉴런 n의 k-멀티섹션 커버리지는 다음과 같이 정의됩니다:

\[\frac{|\{S_i^n \mid \exists x \in T : \phi(x, n) \in S_i^n \}|}{k} \]

We further define the k-multisection neuron coverage of a DNN as

우리는 DNN의 \(k\)-multisection 뉴런 커버리지를 다음과 같이 정의합니다.

\[\text{KMNCov}(T, k) = \frac{\sum_{n \in N} | \{ S^n_i \mid \exists \mathbf{x} \in T : \phi(\mathbf{x}, n) \in S^n_i \} |}{k \times |N|}\]

However, for a neuron n, there are also cases where φ(x, n) may locate out of [low , high ], i.e., φ(x,n) ∈ (−∞,low ) or φ(x,n) ∈ nnn (highn , +∞). We refer to (−∞, lown ) ∪ (highn , +∞) as the corner-case region of a neuron n.

그러나 뉴런 \(n\)의 경우, \(\phi(x, n)\)이 \([\text{low}_n, \text{high}_n]\) 범위를 벗어나는 경우도 있습니다. 즉, \(\phi(x, n)\) \in \((-\infty, \text{low}_n)\) 또는 \(\phi(x, n) \in (\text{high}_n, +\infty)\)일 수 있습니다.
우리는 \((-\infty, \text{low}_n) \cup (\text{high}_n, +\infty)\)를 뉴런 \(n\)의 코너 케이스 영역이라고 부릅니다.

Definition 3.2. For a test input x ∈ T, we say that a DNN is located in its corner-case region given x iff ∃n ∈ N : φ(x,n) ∈ (−∞, lown ) ∪ (highn , +∞).

정의 3.2. 테스트 입력 \(x \in T\)에 대해, \(x\)가 주어졌을 때 DNN이 코너 케이스 영역에 위치한다고 말합니다.
이는 \(\exists n \in N : \phi(x,n) \in (-\infty, \text{low}_n) \cup (\text{high}_n, +\infty)\)인 경우입니다.

Note that the profiled outputs of a neuron obtained from the training data would not locate into the corner-case region.
n other words, if test inputs follow a similar statistical distribution with the training data, a neuron output would rarely locate in corner-case region as well.
Nevertheless, it does not mean that testing the corner cases of a neuron is not important because defects of DNNs could also locate in the corner-case regions (see Section 4.3).

훈련 데이터로부터 얻은 뉴런의 프로파일링된 출력은 코너 케이스 영역에 위치하지 않는다는 점에 유의하세요.
다시 말해, 테스트 입력이 훈련 데이터와 유사한 통계적 분포를 따른다면 뉴런 출력이 코너 케이스 영역에 위치하는 일은 드물 것입니다.
그럼에도 불구하고, 뉴런의 코너 케이스를 테스트하는 것이 중요하지 않다는 의미는 아닙니다.
DNN의 결함이 코너 케이스 영역에 위치할 수 있기 때문입니다(4.3절 참조).

To cover these corner-case regions of DNNs, we define two coverage criteria, i.e., neuron boundary coverage and strong neuron activation coverage.
Given a test input x, if φ(x, n) belongs to (−∞, lown ) or (highn , +∞), we say the corresponding corner-case region is covered.
To quantify this, we first define the number of covered corner-case regions as follows:

DNN의 이러한 코너 케이스 영역을 포함하기 위해 우리는 뉴런 경계 커버리지와 강한 뉴런 활성화 커버리지라는 두 가지 커버리지 기준을 정의합니다.
테스트 입력 \(x\)가 주어졌을 때, 만약 \(\phi(x, n)\)이 \((-\infty, \text{low}_n)\) 또는 \((\text{high}_n, +\infty)\)에 속하면 해당 코너 케이스 영역이 커버되었다고 말합니다.
이를 정량화하기 위해, 먼저 커버된 코너 케이스 영역의 수를 다음과 같이 정의합니다:

UpperCornerNeuron = \(\{n \in N \mid \exists x \in T : \phi(x, n) \in (\text{high}_n, +\infty)\};\)  
LowerCornerNeuron = \(\{n \in N \mid \exists x \in T : \phi(x, n) \in (-\infty, \text{low}_n)\}.\)

(ii) Neuron Boundary Coverage.
Neuron boundary coverage measures how many corner-case regions (w.r.t. both of the upper boundary and the lower boundary values) have been covered by the given test input set T .
It is defined as the ratio of the number of covered corner cases and the total number of corner cases (2 × |N |):

(ii) 뉴런 경계 커버리지.
뉴런 경계 커버리지는 주어진 테스트 입력 집합 \(T\)에 의해 얼마나 많은 코너 케이스 영역(상한값과 하한값 모두에 대해)이 커버되었는지를 측정합니다.
이는 커버된 코너 케이스의 수와 총 코너 케이스 수(2 × |N|)의 비율로 정의됩니다.

\[\text{NBCov}(T) = \frac{|\text{UpperCornerNeuron}| + |\text{LowerCornerNeuron}|}{2 \times |N|}\]

Some recent research on DNNs interpretability empirically shows that the hyperactive neurons might potentially deliver useful learning patterns within DNNs [30, 61].
Based on this intuition, the proposed coverage criteria in the rest of this section focus more on the hyperactive neuron cases (e.g., top-k neuron coverage in the next subsection).
Similar to neuron boundary coverage, we further define strong neuron activation coverage to measure the coverage status of upper-corner cases.

일부 최근 DNN 해석 가능성에 관한 연구는 과활성 뉴런이 DNN 내에서 유용한 학습 패턴을 제공할 수 있음을 실증적으로 보여줍니다 [30, 61].
이러한 직관에 기반하여, 이 절의 나머지 부분에서 제안된 커버리지 기준은 과활성 뉴런 사례(예: 다음 절에서의 top-k 뉴런 커버리지)에 더 중점을 둡니다.
뉴런 경계 커버리지와 유사하게, 우리는 상한 코너 케이스의 커버리지 상태를 측정하기 위해 강한 뉴런 활성화 커버리지를 추가로 정의합니다.

(iii) Strong Neuron Activation Coverage.
Strong neuron activation coverage measures how many corner cases (w.r.t. the upper boundary value highn ) have been covered by the given test inputs T.
It is defined as the ratio of the number of covered upper-corner cases and the total number of corner cases (|N |):

(iii) 강한 뉴런 활성화 커버리지.
강한 뉴런 활성화 커버리지는 주어진 테스트 입력 \(T\)에 의해 얼마나 많은 코너 케이스(상한값 \(\text{high}_n\)에 대해)가 커버되었는지를 측정합니다.
이는 커버된 상한 코너 케이스의 수와 총 코너 케이스 수(\(|N|\))의 비율로 정의됩니다.

\[\text{SNACov}(T) = \frac{|\text{UpperCornerNeuron}|}{|N|}\]

### 3.2 Layer-Level Coverage Criteria
At layer-level, we use the top hyperactive neurons and their combi- nations (or the sequences) to characterize the behaviors of a DNN.

레이어 수준에서는 최상위 과활성 뉴런과 그 조합(또는 순서)을 사용하여 DNN의 동작을 특성화합니다.

For a given test input x and neurons n1 and n2 on the same layer, we say n1 is more active than n2 given x if φ(x,n1) > φ(x,n2).

주어진 테스트 입력 \(x\)와 동일한 레이어에 있는 뉴런 \(n_1\)과 \(n_2\)에 대해, 만약 \(\phi(x, n_1)\) > \(\phi(x, n_2)\)라면 \(n_1\)이 \(n_2\)보다 더 활성화되었다고 말합니다.

For the i-th layer, we use topk(x,i) to denote the neurons that have the largest k outputs on that layer given x.
For example, in Figure 1(a), assume φ(x,n1) and φ(x,n3) are larger than φ(x,n2), the top-2 neurons on layer 1 are n1 and n3 (depicted in green).

\(i\)번째 레이어에서, 우리는 \( \text{topk}(x, i) \)를 주어진 \(x\)에 대해 해당 레이어에서 출력값이 가장 큰 \(k\)개의 뉴런을 나타내는 데 사용합니다.
예를 들어, 그림 1(a)에서 \( \phi(x, n_1) \)과 \( \phi(x, n_3) \)가 \( \phi(x, n_2) \)보다 크다고 가정하면, 레이어 1의 상위 2개의 뉴런은 \( n_1 \)과 \( n_3 \)이며, 이는 녹색으로 표시됩니다.

(i) Top-k Neuron Coverage.
The top-k neuron coverage measures how many neurons have once been the most active k neurons on each layer.
It is defined as the ratio of the total number of top-k neurons on each layer and the total number of neurons in a DNN:

(i) Top-k 뉴런 커버리지.
Top-k 뉴런 커버리지는 각 레이어에서 한 번이라도 가장 활성화된 k개의 뉴런이 된 뉴런의 수를 측정합니다.
이는 각 레이어에서의 Top-k 뉴런의 총 수와 DNN 내의 뉴런 총 수의 비율로 정의됩니다.

\[\text{TKNCov}(T, k) = \frac{|\bigcup_{x \in T} (\bigcup_{1 \leq i \leq l} \text{top}_k(x, i))|}{|N|}\]

The neurons from the same layer of a DNN often play similar roles and the top active neurons from different layers are impor- tant indicators to characterize the major functionality of a DNN.
Intuitively, to more thoroughly test a DNN, a test dataset should uncover more top active neurons.

DNN의 동일한 레이어에 있는 뉴런들은 종종 유사한 역할을 수행하며, 서로 다른 레이어에서의 최상위 활성 뉴런들은 DNN의 주요 기능을 특성화하는 중요한 지표입니다.
직관적으로, DNN을 보다 철저히 테스트하기 위해서는 테스트 데이터셋이 더 많은 최상위 활성 뉴런을 밝혀내야 합니다.

(ii) Top-k Neuron Patterns.
Given a test input x, the sequence of the top-k neurons on each layer also forms a pattern.
In Figure 1(a), assume the neurons in green are the top-2 neurons on each layer, thepatterncanberepresentedas({n1,n3},{n5,n6},{n8,n9}).
More formally, a pattern is an element of 2L1 × 2L2 × ··· × 2Ll , where 2Li isthesetofsubsetsoftheneuronsoni-thlayer,for1≤i≤l. Given the test input set T , the number of top-k neuron patterns for T is defined as:

(ii) Top-k 뉴런 패턴.
주어진 테스트 입력 \(x\)에 대해, 각 레이어에서의 top-k 뉴런들의 순서도 하나의 패턴을 형성합니다.
그림 1(a)에서 녹색으로 표시된 뉴런들이 각 레이어에서 top-2 뉴런들이라고 가정하면, 그 패턴은 \(\{n_1, n_3\}, \{n_5, n_6\}, \{n_8, n_9\}\)로 표현될 수 있습니다.
더 공식적으로 말하면, 패턴은 \(2^{L_1} \times 2^{L_2} \times \cdots \times 2^{L_l}\)의 요소이며, 여기서 \(2^{L_i}\)는 \(i\)번째 레이어의 뉴런들의 부분 집합들로 구성된 집합입니다.
주어진 테스트 입력 집합 \(T\)에 대해, \(T\)의 top-k 뉴런 패턴의 수는 다음과 같이 정의됩니다:

\[\text{TKNPat}(T, k) = |\{ (\text{top}_k(x, 1), \ldots, \text{top}_k(x, l)) \mid x \in T \}|\]

![Figure 1](/posts/paper/DeepGause/figure1.png)

Intuitively, the top-k neuron patterns denote different kinds of activated scenarios from the top hyperactive neurons of each layer.

직관적으로, top-k 뉴런 패턴은 각 레이어의 최상위 과활성 뉴런들로부터 활성화된 다양한 시나리오를 나타냅니다.
