+++
title = 'Model vs system level testing of autonomous driving systems'
date = 2024-08-17T10:51:06+09:00
draft = true
+++

이 글은 Model vs system level testing of autonomous driving systems:a replication and extension study 을 번역 및 요약한 글입니다.

## Abstract
Offline model-level testing of autonomous driving software is much cheaper, faster, and diversified than in-field, online system-level testing.
Hence, researchers have compared empirically model-level vs system-level testing using driving simulators.
They reported the general usefulness of simulators at reproducing the same conditions experienced in-field, but also some inadequacy of model-level testing at exposing failures that are observable only in online mode.
In this work, we replicate the reference study on model vs system- level testing of autonomous vehicles while acknowledging several assumptions that we had reconsidered.

오프라인 모델 수준의 자율주행 소프트웨어 테스트는 현장 온라인 시스템 수준 테스트보다 훨씬 저렴하고, 빠르며, 다양한 상황을 다룰 수 있습니다.
따라서 연구자들은 주행 시뮬레이터를 사용하여 모델 수준과 시스템 수준 테스트를 경험적으로 비교했습니다.
그들은 시뮬레이터가 현장에서 경험한 동일한 조건을 재현하는 데 일반적으로 유용하다는 점을 보고했지만, 모델 수준의 테스트가 온라인 모드에서만 관찰할 수 있는 실패를 발견하는 데 있어서는 부적합하다는 점도 지적했습니다.
이 연구에서 우리는 자율주행 차량의 모델 대 시스템 수준 테스트에 대한 참조 연구를 복제하며, 원래 연구에 영향을 미치는 몇 가지 타당성 위협과 관련된 가정들을 재고했습니다.

Our results show that simulator-based testing of autonomous driving systems yields predictions that are close to the ones of real-world datasets when using neural-based translation to mitigate the real- ity gap induced by the simulation platform.
On the other hand, model-level testing failures are in line with those experienced at the system level, both in simulated and physical environments, when considering the pre-failure site, similar-looking images, and accurate labels.

우리의 결과는 시뮬레이터 기반 테스트가 시뮬레이션 플랫폼에 의해 유도된 현실 차이를 완화하기 위해 신경망 기반 번역을 사용할 때 실제 데이터셋과 유사한 예측을 생성함을 보여줍니다.
한편, 모델 수준의 테스트 실패는 실패 직전의 지점, 유사한 이미지, 정확한 라벨을 고려할 때 시뮬레이션 및 물리적 환경 모두에서 시스템 수준에서 경험한 실패와 일치합니다.

## 1. Introduction

