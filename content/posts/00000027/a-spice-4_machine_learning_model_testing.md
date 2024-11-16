+++
title = 'A-SPICE4.0 MLE.4 Machine Learning Model Testing'
date = 2024-10-30T10:38:42+09:00
draft = false
+++

이 글은 draft를 보고 번역한 것입니다.

|||
|---|---|
| Process Purpose | The purpose is to ensure compliance of the trained ML model and the deployed ML model with the ML requirements. 목적은 학습된 ML 모델과 배포된 ML 모델이 ML 요구사항을 준수하도록 보장하는 것입니다. |
| Process outcomes | 1. A ML test approach is defined. ML 테스트 접근 방식을 정의합니다. 2. A ML test data set is created. ML 테스트 데이터셋을 생성합니다. 3. The trained ML model is tested. 학습된 ML 모델을 테스트합니다. 4. The deployed ML model is derived from the trained ML model and tested. 학습된 ML 모델에서 파생된 배포된 ML 모델을 테스트합니다. 5. Consistency and bidirectional traceability are established between the ML test approach and the ML requirements, and the ML test data set and the ML data requirements; and bidirectional traceability is established between the ML test approach and ML test results. ML 테스트 접근 방식과 ML 요구사항, ML 테스트 데이터셋과 ML 데이터 요구사항 간의 일관성 및 양방향 추적 가능성을 확립하고, ML 테스트 접근 방식과 ML 테스트 결과 간에도 양방향 추적 가능성을 확립합니다. 6. Results of the ML model testing are summarized and communicated with the deployed ML model to all affected parties. ML 모델 테스트 결과를 요약하여 배포된 ML 모델과 관련된 모든 이해관계자에게 전달합니다. |

## MLE.4 with 7 Base practices
### MLE.4.BP1: Specify an ML test approach.

Specify an ML test approach suitable to provide evidence for compliance of the trained ML model and the deployed ML model with the ML requirements. The ML test approach includes

훈련된 ML 모델과 배포된 ML 모델이 ML 요구사항을 준수하는지에 대한 증거를 제공할 수 있는 ML 테스트 접근 방식을 명시합니다. ML 테스트 접근 방식에는 다음 사항이 포함됩니다.

* ML test scenarios with distribution of data characteristics (e.g., gender, weather conditions, street conditions within the ODD) defined by ML requirements,

  ML 요구사항에 의해 정의된 데이터 특성 분포에 따른 ML 테스트 시나리오(예: 성별, 날씨 조건, ODD 내 도로 조건),

* distribution and frequency of each ML test scenario inside the ML test data set,

  ML 테스트 데이터 세트 내에서 각 ML 테스트 시나리오의 분포와 빈도,

* expected test result per test datum,

  각 테스트 데이터에 대한 예상 테스트 결과,

* entry and exit criteria of the testing,

  테스트의 시작 및 종료 기준,

* approach for data set creation and modification, and

  데이터 세트 생성 및 수정 접근 방식, 그리고

* the required testing infrastructure and environment setup.

  필요한 테스트 인프라 및 환경 설정.

이와 같은 사항을 포함하여 ML 테스트 접근 방식을 구체화합니다.

Note 1: Expected test result per test datum might require labeling of test data to support comparison of output of the ML model with the expected output.

주석 1: 각 테스트 데이터에 대한 예상 테스트 결과는 ML 모델의 출력과 예상 출력의 비교를 지원하기 위해 테스트 데이터의 라벨링을 필요로 할 수 있습니다.

Note 2: Test datum is the smallest amount of data which is processed by the ML model into only one output. E.g., one image in photo processing or an audio sequence in voice recognition.

주석 2: 테스트 데이터는 ML 모델이 단 하나의 출력을 내는 데 사용되는 최소 데이터 단위입니다. 예를 들어, 사진 처리에서는 한 이미지, 음성 인식에서는 오디오 시퀀스가 해당합니다.

Note 3: Data characteristic is one property of the data that may have different expressions in the ODD. E.g., weather condition may contain expressions like sunny, foggy or rainy.

주석 3: 데이터 특성은 ODD에서 다양한 표현을 가질 수 있는 데이터의 속성 중 하나입니다. 예를 들어, 날씨 조건은 맑음, 안개, 비와 같은 표현을 포함할 수 있습니다.

Note 4: An ML test scenario is a combination of expressions of all defined data characteristics e.g., weather conditions = sunny, street conditions = gravel road.

주석 4: ML 테스트 시나리오는 모든 정의된 데이터 특성의 표현을 결합한 조합입니다. 예를 들어, 날씨 조건 = 맑음, 도로 조건 = 자갈길 등이 이에 해당합니다.

### MLE.4.BP2: Create ML test data set.

Create the ML test data set needed for testing of the trained ML model and testing of the deployed ML model from the ML data collection provided by SUP.11 considering the ML test approach.
The ML test data set shall not be used for training. 

훈련된 ML 모델의 테스트와 배포된 ML 모델의 테스트에 필요한 ML 테스트 데이터 세트를 SUP.11에서 제공된 ML 데이터 컬렉션에서 ML 테스트 접근 방식을 고려하여 생성합니다.
이 ML 테스트 데이터 세트는 훈련에 사용되지 않아야 합니다.

Note 5: The ML test data set for the trained ML model might differ from the test data set of the deployed ML model.

주석 5: 훈련된 ML 모델을 위한 ML 테스트 데이터 세트는 배포된 ML 모델의 테스트 데이터 세트와 다를 수 있습니다.

Note 6: Additional data sets might be used for special purposes like assurance of safety, fairness, robustness.

주석 6: 안전성, 공정성, 견고성 보장과 같은 특별한 목적을 위해 추가 데이터 세트를 사용할 수 있습니다.

### MLE.4.BP3: Test trained ML model.

Test the trained ML model according to the ML test approach using the created ML test data set.
Record and evaluate the ML test results.

생성된 ML 테스트 데이터 세트를 사용하여 ML 테스트 접근 방식에 따라 훈련된 ML 모델을 테스트합니다.
ML 테스트 결과를 기록하고 평가합니다.

Note 7: Evaluation of test logs might include pattern analysis of failed test data to support e.g., trustworthiness.

주석 7: 테스트 로그 평가에는 신뢰성 지원을 위한 실패한 테스트 데이터의 패턴 분석이 포함될 수 있습니다.

### MLE.4.BP4: Derive deployed ML model.

Derive the deployed ML model from the trained ML model according to the ML architecture.
The deployed ML model shall be used for testing and delivery to software integration.

훈련된 ML 모델로부터 ML 아키텍처에 따라 배포된 ML 모델을 도출합니다.
배포된 ML 모델은 테스트 및 소프트웨어 통합을 위한 배포에 사용됩니다.

Note 8: The deployed ML model will be integrated into the target system and may differ from the trained ML model which often requires powerful hardware and uses interpretative languages.

주석 8: 배포된 ML 모델은 최종 시스템에 통합될 것이며, 종종 강력한 하드웨어와 해석형 언어를 사용하는 훈련된 ML 모델과 다를 수 있습니다.

### MLE4.BP5: Test deployed ML model.

Test the deployed ML model according to the ML test approach using the created ML test data set.
Record and evaluate the ML test results.

생성된 ML 테스트 데이터 세트를 사용하여 ML 테스트 접근 방식에 따라 배포된 ML 모델을 테스트합니다. 
ML 테스트 결과를 기록하고 평가합니다.

### MLE.4.BP6: Ensure consistency and establish bidirectional traceability.

Ensure consistency and establish bidirectional traceability between the ML test approach and the ML requirements, and the ML test data set and the ML data requirements; and bidirectional traceability is established between the ML test approach and ML test results.

ML 테스트 접근 방식과 ML 요구사항 간의, 그리고 ML 테스트 데이터 세트와 ML 데이터 요구사항 간의 일관성을 보장하고 양방향 추적성을 수립합니다.
또한 ML 테스트 접근 방식과 ML 테스트 결과 간에도 양방향 추적성을 수립합니다.

Note 9: Bidirectional traceability supports consistency, and facilitates impact analyses of change requests, and verification coverage demonstration.
Traceability alone, e.g., the existence of links, does not necessarily mean that the information is consistent with each other.

주석 9: 양방향 추적성은 일관성을 지원하고 변경 요청에 대한 영향 분석 및 검증 범위 입증을 용이하게 합니다.
단순한 추적성, 예를 들어 링크의 존재만으로는 정보가 서로 일관성이 있음을 의미하지 않습니다.

### MLE.4.BP7: Summarize and communicate results.

Summarize the ML test results of the ML model.
Inform all affected parties about the agreed results and the deployed ML model.

ML 모델의 ML 테스트 결과를 요약합니다.
합의된 결과와 배포된 ML 모델에 대해 관련된 모든 당사자에게 알립니다.

