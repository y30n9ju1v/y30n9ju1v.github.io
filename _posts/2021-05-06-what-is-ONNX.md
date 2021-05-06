---
layout: post                          # (require) default post layout
title: "What is ONNX"                 # (require) a string title
date: 2021-05-06 18:00:00 +0900       # (require) a post date
---

## What is ONNX? 
---
<p align="center"><img width="70%" src="https://raw.githubusercontent.com/y30n9ju1v/y30n9ju1v.github.io/master/static/img/_posts/onnx_logo.png" /><br>(이미지: https://github.com/onnx/onnx></p>

ONNX는 추론 엔진에서 널리 사용되는 기계학습 모델 형식인 Open Neural Network Exchange의 약자입니다.
2017 년 9 월 Facebook과 Microsoft는 오픈 소스 인공 지능 생태계 기업들 간 협력과 PyTorch 및 Caffe2와 같은 기계 학습 프레임 워크 간 손쉬운 전환을 위해 기계 학습 알고리즘과 소프트웨어 도구를 표현하는 개방형 표준을 도입했습니다.
<br>
<br>
<br>

## Common IR(Intermediate Representation)
---
<p align="center"><img width="70%" src="https://raw.githubusercontent.com/y30n9ju1v/y30n9ju1v.github.io/master/static/img/_posts/onnx_common_ir.png" /><br>(이미지: https://docs.microsoft.com/ko-kr/azure/machine-learning/concept-onnx)</p>

인공지능 프레임 워크는 일반적으로 빠른 학습, 복잡한 네트워크 아키텍처 지원, 모바일 장치에서의 추론 등과 같은 일부 특성에 맞게 최적화됩니다. 
ONNX는 그래프의 common IR을 제공함으로써 개발자가 자신의 작업에 적합한 프레임 워크를 선택하도록 돕고 하드웨어 공급 업체가 플랫폼에 대한 최적화를 간소화할 수 있도록 합니다. 
그리고 ONNX는 강력한 에코 시스템을 구축하는 데 도움을 줍니다.
<br>
<br>
<br>

## Components
---
<p align="center"><img width="70%" src="https://raw.githubusercontent.com/y30n9ju1v/y30n9ju1v.github.io/master/static/img/_posts/onnx_graph.png" /><br>(이미지: https://github.com/lutzroeder/netron)</p>


ONNX는 다음 구성 요소로 구성되었습니다.

    1. 확장 가능한 계산 그래프 모델의 정의
    2. 표준 데이터 유형의 정의
    3. 내장 연산자의 정의

1과 2는 함께 여기에서 다루는 ONNX Intermediate Representation 또는 'IR'사양을 구성합니다.
<br>
<br>

### 1. 확장 가능한 계산 그래프 모델
최상위 ONNX 구성은 '모델'이며 프로토콜 버퍼에 유형으로 표시됩니다. 
모델 구조의 주요 목적은 모든 실행 가능한 요소를 포함하는 그래프와 메타 데이터를 연결하는 것입니다. 
메타 데이터는 모델 파일을 처음 읽을 때 사용되며, 모델을 실행할 수 있는지, 로깅 메시지, 오류 보고서 등을 생성 할 수 있는지 여부를 결정하는 데 필요한 정보를 구현에 제공합니다.
<br>
<br>

### 2. 표준 데이터 유형
Tensor, 입력/출력 데이터와 속성 유형이 있으며 속성 값은 조밀 텐서, 희소 텐서, 스칼라 숫자 값, 문자열, 그래프 또는 위에서 언급 한 유형 중 하나의 반복값일 수 있습니다.
<br>
<br>
<br>

## File Format
---
ONNX를 파일 형태로 저장하기 위해 직렬화/역직렬화 방법으로 Google에서 만든 protobuf를 사용합니다.
프로토콜 버퍼 (a.k.a., protobuf)는 구조화 된 데이터를 직렬화하기위한 Google의 컴퓨팅 랭귀지 중립적, 플랫폼 중립적 그리고 확장 가능한 메커니즘입니다.
데이터를 한 번 구조화하는 방법을 정의한 다음 특수 생성 된 소스 코드를 사용하여 다양한 데이터 스트림과 다양한 언어를 사용하여 구조화된 데이터를 쉽게 쓰고 읽을 수 있습니다.
<br>
<br>
<br>

## Similarities and differences to NNEF
---
<p align="center"><img width="70%" src="https://raw.githubusercontent.com/y30n9ju1v/y30n9ju1v.github.io/master/static/img/_posts/nnef_logo.png" /><br>(이미지: https://www.khronos.org/nnef)</p>

NNEF는 Khronos 그룹에서는 서로 다른 신경망 프레임 워크간에 신경망 정보를 주고받기 위한 상호 호환성 제공을 위해 표준화하였습니다.
NNEF는 그래프 구조와 데이터파일을 분리합니다. 
특히, NNEF는 텐서 개념을 확장하여 프레그먼트와 신경망 연산에 필요한 함수 및 구문을 정의하고 있습니다. 
이는 네트워크 구조 또는 개별 매개변수 데이터에 대한 독립적인 액세스를 지원하며, 파일은 tar 또는 zip과 같은 선택적 압축 및 암호화를 제공합니다.
<br>
<br>
### NNEF와 ONNX의 비교 

||NNEF|ONNX|
| --- | --- | --- |
|주도|하드웨어 업체|소프트웨어 업체|
|개방|비영리 단체의 오픈소스 구현|오픈소스 프로젝트|
|형식|텍스트 기반의 절차형식|데이터구조 지향적인 Protobuf|
|오퍼레이션|Flat 형식으로 네트워크 표현. <br>복합 오퍼레이션 표현 가능|Flat 형식으로 네트워크 표현|
|흐름 구조|절차구문으로 동적그래프 표현|컨트롤플로우를 통한 동적그래프 표현|
|데이터|개념수준에서 양자화에 접근하여 추론 가속화를 위한 최적화 지원|Tensor의 구체적인 데이터 유형 사용|

<br>
<br>

### 참고

1. https://github.com/onnx/onnx/blob/master/docs/Overview.md
2. https://github.com/onnx/onnx/blob/master/docs/IR.md
3. https://en.wikipedia.org/wiki/Open_Neural_Network_Exchange
4. https://github.com/protocolbuffers/protobuf
5. https://www.khronos.org/blog/nnef-and-onnx-similarities-and-differences
6. 임베디드 시스템용 딥러닝 추론엔진 기술 동향
