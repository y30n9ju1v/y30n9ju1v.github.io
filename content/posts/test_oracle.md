+++
title = 'Test Oracle'
date = 2024-09-08T15:01:28+09:00
draft = false
+++

이 글은 https://en.wikipedia.org/wiki/Test_oracle 을 우선 번역하였고 추가적으로 이해한 내용을 추가, 수정하겠습니다.

In software testing, a test oracle (or just oracle) is a provider of information that describes correct output based on the input of a test case.
Testing with an oracle involves comparing actual results of the system under test (SUT) with the expected results as provided by the oracle.

소프트웨어 테스트에서 테스트 오라클(또는 간단히 오라클)은 테스트 케이스의 입력을 기반으로 올바른 출력을 설명하는 정보를 제공하는 도구입니다.
오라클을 사용한 테스트는 테스트 대상 시스템(SUT, System Under Test)의 실제 결과를 오라클이 제공한 예상 결과와 비교하는 작업을 포함합니다.

The term "test oracle" was first introduced in a paper by William E. Howden.
Additional work on different kinds of oracles was explored by Elaine Weyuker.

“테스트 오라클”이라는 용어는 William E. Howden의 논문에서 처음 도입되었습니다.
다양한 종류의 오라클에 대한 추가 연구는 Elaine Weyuker에 의해 탐구되었습니다.

An oracle can operate separately from the SUT; accessed at test runtime, or it can be used before a test is run with expected results encoded into the test logic.

오라클은 SUT와 별도로 작동할 수 있으며, 테스트 실행 시 접근하거나, 테스트 실행 전에 예상 결과를 테스트 논리에 인코딩하여 사용할 수 있습니다.

Determining the correct output for a given input (and a set of program or system states) is known as the oracle problem or test oracle problem, which some consider a relatively hard problem, and involves working with problems related to controllability and observability.

주어진 입력(및 프로그램 또는 시스템 상태 집합)에 대한 올바른 출력을 결정하는 것을 오라클 문제 또는 테스트 오라클 문제라고 합니다.
일부는 이를 비교적 어려운 문제로 간주하며, 이는 제어 가능성과 관찰 가능성과 관련된 문제들을 다루는 작업을 포함합니다.

## Categories

### Specified
A specified oracle is typically associated with formalized approaches to software modeling and software code construction.
It is connected to formal specification, model-based design which may be used to generate test oracles, state transition specification for which oracles can be derived to aid model-based testing and protocol conformance testing, and design by contract for which the equivalent test oracle is an assertion.

명시적 오라클(specified oracle)은 일반적으로 소프트웨어 모델링과 소프트웨어 코드 구축의 형식화된 접근 방식과 관련이 있습니다.
이는 형식 명세(formal specification), 테스트 오라클 생성을 위해 사용될 수 있는 모델 기반 설계(model-based design), 모델 기반 테스트(model-based testing)를 지원하기 위해 오라클을 도출할 수 있는 상태 전이 명세(state transition specification), 그리고 동등한 테스트 오라클이 단언문(assertion)인 계약에 의한 설계(design by contract)와 연결되어 있습니다.

Specified test oracles have a number of challenges.
Formal specification relies on abstraction, which in turn may naturally have an element of imprecision as all models cannot capture all behavior.

명시적 테스트 오라클은 여러 도전 과제를 가지고 있습니다.
형식 명세는 추상화에 의존하는데, 이는 모든 모델이 모든 동작을 포착할 수 없기 때문에 자연스럽게 부정확한 요소를 포함할 수 있습니다.

### Derived
A derived test oracle differentiates correct and incorrect behavior by using information derived from artifacts of the system.
These may include documentation, system execution results and characteristics of versions of the SUT.
Regression test suites (or reports) are an example of a derived test oracle - they are built on the assumption that the result from a previous system version can be used as aid (oracle) for a future system version.
Previously measured performance characteristics may be used as an oracle for future system versions, for example, to trigger a question about observed potential performance degradation.
Textual documentation from previous system versions may be used as a basis to guide expectations in future system versions.

유도된 테스트 오라클(derived test oracle)은 시스템의 아티팩트에서 도출된 정보를 사용하여 올바른 동작과 잘못된 동작을 구분합니다.
이러한 아티팩트에는 문서, 시스템 실행 결과, SUT(System Under Test)의 버전 특성 등이 포함될 수 있습니다.
예를 들어, 회귀 테스트 스위트(또는 보고서)는 유도된 테스트 오라클의 한 예입니다.
이는 이전 시스템 버전의 결과를 향후 시스템 버전의 오라클로 사용할 수 있다는 가정에 기반합니다.
이전에 측정된 성능 특성은 향후 시스템 버전의 오라클로 사용될 수 있으며, 예를 들어 잠재적인 성능 저하를 관찰하고 이에 대한 질문을 제기할 수 있습니다.
이전 시스템 버전의 텍스트 문서는 향후 시스템 버전에서 기대치를 안내하는 기준으로 사용될 수 있습니다.

A pseudo-oracle falls into the category of derived test oracle.
A pseudo-oracle, as defined by Weyuker, is a separately written program which can take the same input as the program or SUT so that their outputs may be compared to understand if there might be a problem to investigate.

유사 오라클(pseudo-oracle)은 유도된 테스트 오라클의 범주에 속합니다.
Weyuker에 의해 정의된 유사 오라클은 프로그램이나 SUT와 동일한 입력을 받아 그 출력을 비교함으로써 조사할 수 있는 문제가 있는지 이해할 수 있는 별도로 작성된 프로그램입니다.

A partial oracle is a hybrid between specified test oracle and derived test oracle. It specifies important (but not complete) properties of the SUT. For example, metamorphic testing exploits such properties, called metamorphic relations, across multiple executions of the system.

부분 오라클(partial oracle)은 명시적 테스트 오라클과 유도된 테스트 오라클의 혼합형입니다.
이는 SUT의 중요한(하지만 완전하지는 않은) 속성을 명시합니다.
예를 들어, 메타모픽 테스트는 시스템의 여러 실행에 걸쳐 메타모픽 관계라고 불리는 이러한 속성을 활용합니다.

### Implicit
An implicit test oracle relies on implied information and assumptions.
For example, there may be some implied conclusion from a program crash, i.e. unwanted behavior - an oracle to determine that there may be a problem.
There are a number of ways to search and test for unwanted behavior, whether some call it negative testing, where there are specialized subsets such as fuzzing.

암묵적 테스트 오라클(implicit test oracle)은 암묵적인 정보와 가정에 의존합니다.
예를 들어, 프로그램이 충돌(crash)했을 때 암묵적으로 문제 발생을 의미하는 결론을 내릴 수 있습니다.
이는 원하지 않는 동작이 있을 수 있다는 오라클 역할을 합니다.
원하지 않는 동작을 검색하고 테스트하는 방법에는 여러 가지가 있으며, 그중 일부는 퍼징(fuzzing)과 같은 특수한 하위 집합을 포함하는 부정 테스트(negative testing)라고 불리기도 합니다.

There are limitations in implicit test oracles - as they rely on implied conclusions and assumptions.
For example, a program or process crash may not be a priority issue if the system is a fault-tolerant system and so operating under a form of self-healing/self-management.
Implicit test oracles may be susceptible to false positives due to environment dependencies.
Property based testing relies on implicit oracles.

암묵적 테스트 오라클에는 제한 사항이 있습니다.
이는 암시된 결론과 가정에 의존하기 때문입니다.
예를 들어, 시스템이 장애 허용(fault-tolerant) 시스템이라서 자가 치유/자가 관리 형태로 운영되는 경우, 프로그램이나 프로세스의 충돌이 우선적인 문제가 아닐 수 있습니다.
암묵적 테스트 오라클은 환경 의존성으로 인해 잘못된 양성(false positive)에 취약할 수 있습니다.
속성 기반 테스트(property-based testing)는 암묵적 오라클에 의존합니다.

### Human
A human can act as a test oracle.
This approach can be categorized as quantitative or qualitative.
A quantitative approach aims to find the right amount of information to gather on a SUT (e.g., test results) for a stakeholder to be able to make decisions on fit-for-purpose or the release of the software.
A qualitative approach aims to find the representativeness and suitability of the input test data and context of the output from the SUT.
An example is using realistic and representative test data and making sense of the results (if they are realistic).
These can be guided by heuristic approaches, such as gut instincts, rules of thumb, checklist aids, and experience to help tailor the specific combination selected for the SUT.

사람이 테스트 오라클로 역할할 수 있습니다.
이러한 접근 방식은 정량적 또는 정성적으로 분류될 수 있습니다.
정량적 접근 방식은 이해관계자가 소프트웨어의 목적 적합성 또는 릴리스 여부에 대한 결정을 내릴 수 있도록 SUT(System Under Test)에서 수집할 적절한 정보(예: 테스트 결과)의 양을 찾는 것을 목표로 합니다.
정성적 접근 방식은 입력 테스트 데이터의 대표성(representativeness)과 SUT 출력의 적합성을 찾는 것을 목표로 합니다.
예를 들어, 현실적이고 대표적인 테스트 데이터를 사용하여 결과가 현실적인지 이해하는 것입니다.
이러한 접근 방식은 직관(gut instincts), 경험 규칙(rules of thumb), 체크리스트 도구, 경험과 같은 휴리스틱 접근법에 의해 안내되어 SUT에 적합한 특정 조합을 맞춤화하는 데 도움이 될 수 있습니다.

## Examples
Test oracles are most commonly based on specifications and documentation.
A formal specification used as input to model-based design and model-based testing would be an example of a specified test oracle.
The model-based oracle uses the same model to generate and verify system behavior.
Documentation that is not a full specification of the product, such as a usage or installation guide, or a record of performance characteristics or minimum machine requirements for the software, would typically be a derived test oracle.

테스트 오라클은 대부분 사양(specifications)과 문서화(documentation)를 기반으로 합니다.
모델 기반 설계와 모델 기반 테스트에 입력으로 사용되는 형식 명세(formal specification)는 명시적 테스트 오라클의 예입니다.
모델 기반 오라클은 동일한 모델을 사용하여 시스템 동작을 생성하고 검증합니다.
제품의 전체 사양이 아닌 사용 설명서나 설치 가이드, 소프트웨어의 성능 특성 기록 또는 최소 시스템 요구 사항과 같은 문서는 일반적으로 유도된 테스트 오라클로 간주됩니다.

A consistency oracle compares the results of one test execution to another for similarity.
This is another example of a derived test oracle.

일관성 오라클(consistency oracle)은 한 번의 테스트 실행 결과를 다른 실행 결과와 비교하여 유사성을 확인합니다.
이것도 유도된 테스트 오라클의 또 다른 예입니다.

An oracle for a software program might be a second program that uses a different algorithm to evaluate the same mathematical expression as the product under test.
This is an example of a pseudo-oracle, which is a derived test oracle.

소프트웨어 프로그램의 오라클은 테스트 대상 제품과 동일한 수학적 표현을 평가하기 위해 다른 알고리즘을 사용하는 두 번째 프로그램이 될 수 있습니다.
이는 유사 오라클(pseudo-oracle)의 예로, 유도된 테스트 오라클에 속합니다.

During Google search, we do not have a complete oracle to verify whether the number of returned results is correct.
We may define a metamorphic relation such that a follow-up narrowed-down search will produce fewer results.
This is an example of a partial oracle, which is a hybrid between specified test oracle and derived test oracle.

구글 검색에서는 반환된 결과의 수가 올바른지 확인하기 위한 완전한 오라클이 없습니다.
좁혀진 후속 검색이 더 적은 결과를 산출하는 메타모픽 관계를 정의할 수 있습니다.
이는 명시적 테스트 오라클과 유도된 테스트 오라클의 혼합형인 부분 오라클(partial oracle)의 예입니다.

A statistical oracle uses probabilistic characteristics, for example with image analysis where a range of certainty and uncertainty is defined for the test oracle to pronounce a match or otherwise.
This would be an example of a quantitative approach in human test oracle.

통계적 오라클(statistical oracle)은 확률적 특성을 사용합니다.
예를 들어, 이미지 분석에서는 일치 여부를 판단하기 위해 테스트 오라클에 대한 확실성과 불확실성의 범위를 정의합니다.
이는 인간 테스트 오라클의 정량적 접근의 한 예입니다.

A heuristic oracle provides representative or approximate results over a class of test inputs.
This would be an example of a qualitative approach in human test oracle.

휴리스틱 오라클(heuristic oracle)은 테스트 입력의 클래스에 대해 대표적이거나 대략적인 결과를 제공합니다.
이는 인간 테스트 오라클의 정성적 접근의 한 예입니다.
