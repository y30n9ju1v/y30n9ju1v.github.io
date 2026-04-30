---
title: "자율주행 시뮬레이션을 위한 HD 맵 포맷: OpenDRIVE(XODR)와 Lanelet2"
date: 2026-04-30T14:00:00+09:00
draft: false
tags: ["자율주행", "HD맵", "OpenDRIVE", "XODR", "Lanelet2", "시뮬레이션", "HERE", "ROS"]
categories: ["자율주행"]
description: "자율주행 시뮬레이션에서 핵심 인프라인 HD 맵 포맷 OpenDRIVE(XODR)와 Lanelet2를 비교 분석합니다. 각 포맷의 구조, 장단점, 대표 툴체인을 정리합니다."
---

## 들어가며

자율주행 시뮬레이션을 구축하려면 **도로 환경을 정밀하게 표현하는 HD 맵**이 반드시 필요합니다. 센서 퓨전이나 인지 알고리즘만큼이나, 차량이 달리는 도로의 기하 구조와 의미 정보를 얼마나 정확하게 모델링하느냐가 시뮬레이션 품질을 결정합니다.

현재 업계에서 가장 널리 쓰이는 두 가지 HD 맵 포맷은 다음과 같습니다:

- **OpenDRIVE (XODR)** — ASAM이 관리하는 국제 표준. Carla, SUMO, CarMaker 등 대부분의 상용 시뮬레이터가 사용.
- **Lanelet2** — ROS 생태계 중심의 오픈소스 포맷. Autoware, Apollo 등 오픈소스 AV 스택과 연동.

이 글에서는 두 포맷의 구조와 개념을 정리하고, 실무에서 어떤 상황에 무엇을 선택할지 안내합니다.

---

## 1. OpenDRIVE (XODR)

### 1.1 개요

OpenDRIVE는 [ASAM(Association for Standardization of Automation and Measuring Systems)](https://www.asam.net/)이 관리하는 HD 맵 표준입니다. 파일 확장자는 `.xodr`이며 XML 기반입니다. 2002년 처음 발표된 이후 자동차 OEM, 시뮬레이터 업체 사이에서 사실상의 표준이 되었습니다.

**주요 사용처:**
- CARLA Simulator
- SUMO (Simulation of Urban Mobility)
- IPG CarMaker
- dSPACE ASM
- ASAM OpenSCENARIO와 함께 사용

### 1.2 핵심 개념

#### Road (도로)

OpenDRIVE에서 모든 도로는 `<road>` 요소로 표현됩니다. 각 도로는 고유한 `id`를 가지며, 도로들이 `<junction>`을 통해 연결됩니다.

```xml
<road name="Highway_A" length="500.0" id="1" junction="-1">
  ...
</road>
```

#### Reference Line (기준선)

각 도로의 기하 정보는 **기준선(reference line)** 을 중심으로 정의됩니다. 기준선은 도로의 중앙을 따라가는 가상의 선이며, 이 선을 기준으로 차선의 좌우 위치(offset)를 표현합니다.

기준선 기하는 다음 요소들의 조합으로 표현됩니다:

| 기하 요소 | 설명 |
| :--- | :--- |
| `<line>` | 직선 구간 |
| `<arc>` | 일정 곡률의 호 |
| `<spiral>` | 클로소이드(Euler spiral) — 직선↔호 전환 완화 구간 |
| `<poly3>` | 3차 다항식 |
| `<paramPoly3>` | 매개변수 3차 다항식 |

```xml
<geometry s="0.0" x="100.0" y="200.0" hdg="0.523" length="50.0">
  <arc curvature="0.01"/>
</geometry>
```

여기서 `s`는 도로 시작점으로부터의 호장(arc length) 거리입니다. OpenDRIVE에서 위치를 표현하는 핵심 좌표계가 바로 이 **s-t 좌표계**입니다.

- `s`: 기준선을 따라 측정한 도로 진행 방향 거리
- `t`: 기준선에 수직인 횡방향 거리 (왼쪽 양수, 오른쪽 음수)

#### Lane Section & Lane

차선 정보는 `<laneSection>` 안에 정의됩니다. 각 `<laneSection>`은 `s` 값의 구간 내에서 유효하며, 하나의 도로에 여러 laneSection이 있을 수 있습니다(차선 수가 바뀌는 구간마다).

```xml
<laneSection s="0.0">
  <left>
    <lane id="2" type="driving" level="false">
      <width sOffset="0.0" a="3.5" b="0.0" c="0.0" d="0.0"/>
    </lane>
    <lane id="1" type="driving" level="false">
      <width sOffset="0.0" a="3.5" b="0.0" c="0.0" d="0.0"/>
    </lane>
  </left>
  <center>
    <lane id="0" type="none"/>
  </center>
  <right>
    <lane id="-1" type="driving" level="false">
      <width sOffset="0.0" a="3.5" b="0.0" c="0.0" d="0.0"/>
    </lane>
  </right>
</laneSection>
```

- 기준선(id=0) 기준 왼쪽은 양수 id, 오른쪽은 음수 id
- `type` 속성: `driving`, `shoulder`, `border`, `sidewalk`, `parking`, `biking` 등

#### Junction (교차로)

교차로는 `<junction>` 요소로 정의하며, 진입로(incoming road)와 연결로(connecting road) 간의 관계를 `<connection>`으로 매핑합니다.

```xml
<junction id="5" name="Intersection_A">
  <connection id="0" incomingRoad="1" connectingRoad="10" contactPoint="start">
    <laneLink from="-1" to="-1"/>
  </connection>
</junction>
```

#### Object & Signal

도로 위의 정적 객체(가드레일, 신호등, 표지판 등)는 `<objects>`와 `<signals>` 섹션에 정의합니다.

```xml
<signals>
  <signal s="120.0" t="-1.5" id="1001" name="StopSign"
          dynamic="no" orientation="+" zOffset="0.0"
          country="OpenDRIVE" type="206" subtype="-1"/>
</signals>
```

### 1.3 전체 파일 구조 예시

```xml
<?xml version="1.0" encoding="UTF-8"?>
<OpenDRIVE>
  <header revMajor="1" revMinor="6" name="MyMap" version="1.00"
          date="2026-04-30" north="0.0" south="0.0" east="0.0" west="0.0"/>

  <road name="MainRoad" length="200.0" id="1" junction="-1">
    <link>
      <successor elementType="junction" elementId="5"/>
    </link>
    <planView>
      <geometry s="0.0" x="0.0" y="0.0" hdg="0.0" length="200.0">
        <line/>
      </geometry>
    </planView>
    <lanes>
      <laneSection s="0.0">
        <left>
          <lane id="1" type="driving" level="false">
            <width sOffset="0.0" a="3.5" b="0.0" c="0.0" d="0.0"/>
            <roadMark sOffset="0.0" type="solid" weight="standard"
                      color="white" width="0.12"/>
          </lane>
        </left>
        <center>
          <lane id="0" type="none"/>
        </center>
        <right>
          <lane id="-1" type="driving" level="false">
            <width sOffset="0.0" a="3.5" b="0.0" c="0.0" d="0.0"/>
            <roadMark sOffset="0.0" type="broken" weight="standard"
                      color="white" width="0.12"/>
          </lane>
        </right>
      </laneSection>
    </lanes>
  </road>

  <junction id="5" name="Intersection_A">
    <connection id="0" incomingRoad="1" connectingRoad="10" contactPoint="start">
      <laneLink from="-1" to="-1"/>
    </connection>
  </junction>
</OpenDRIVE>
```

### 1.4 장단점

| | 내용 |
| :--- | :--- |
| **장점** | 국제 표준, 상용 시뮬레이터 호환성 최고, 기하 표현력 우수 |
| **장점** | 교통 신호·표지판·객체 모두 하나의 파일에 통합 표현 가능 |
| **단점** | XML 특성상 파일 크기 큼, 파싱 비용 높음 |
| **단점** | 의미 정보(semantic) 표현이 Lanelet2보다 약함 |
| **단점** | 복잡한 교차로 모델링이 번거로움 |

---

## 2. Lanelet2

### 2.1 개요

Lanelet2는 FZI 연구소에서 개발한 오픈소스 HD 맵 라이브러리 및 포맷입니다. [GitHub](https://github.com/fzi-forschungszentrum-informatik/Lanelet2)에서 관리되며 ROS/ROS2와 긴밀하게 통합됩니다. 파일은 **OSM(OpenStreetMap) XML** 형식(`.osm`)을 확장한 구조입니다.

**주요 사용처:**
- Autoware (Universe/Auto)
- Apollo (변환 레이어 통해)
- ROS 기반 연구 플랫폼
- CommonRoad

### 2.2 핵심 개념

Lanelet2는 세 가지 기본 요소로 구성됩니다: **Point → Linestring → Lanelet/Area**.

#### Point

모든 기하의 기본 단위입니다. WGS84 위경도 또는 지역 좌표계(UTM 등)로 표현합니다.

```xml
<node id="1" lat="49.0050" lon="8.4040">
  <tag k="ele" v="112.5"/>
  <tag k="local_x" v="100.0"/>
  <tag k="local_y" v="200.0"/>
</node>
```

#### Linestring

Point들을 연결한 선입니다. 차선 경계, 정지선, 울타리 등 모든 선 요소를 표현합니다.

```xml
<way id="100">
  <nd ref="1"/>
  <nd ref="2"/>
  <nd ref="3"/>
  <tag k="type" v="line_thin"/>
  <tag k="subtype" v="dashed"/>
</way>
```

주요 `type` 값:

| type | 의미 |
| :--- | :--- |
| `line_thin` / `line_thick` | 차선 경계선 (얇음/두꺼움) |
| `curbstone` | 연석 |
| `guard_rail` | 가드레일 |
| `stop_line` | 정지선 |
| `virtual` | 물리적 경계 없는 가상 경계 (교차로 내부 등) |

#### Lanelet

Lanelet은 Lanelet2의 핵심 요소입니다. **좌측 경계(left bound)와 우측 경계(right bound)** 두 Linestring으로 정의되는 방향성 있는 차선 구간입니다.

```xml
<relation id="1000">
  <member type="way" ref="100" role="left"/>
  <member type="way" ref="101" role="right"/>
  <tag k="type" v="lanelet"/>
  <tag k="subtype" v="road"/>
  <tag k="speed_limit" v="50"/>
  <tag k="one_way" v="yes"/>
  <tag k="location" v="urban"/>
</relation>
```

- 진행 방향은 left/right bound 선분의 방향으로 결정됩니다
- 인접한 Lanelet들은 경계 Linestring을 공유함으로써 토폴로지를 표현합니다

`subtype` 주요 값:

| subtype | 의미 |
| :--- | :--- |
| `road` | 일반 도로 차선 |
| `highway` | 고속도로 차선 |
| `play_street` | 어린이 보호구역 등 |
| `bicycle_lane` | 자전거 전용 차선 |
| `exit` / `entry` | 진출입로 |
| `walkway` | 보행로 |

#### Area

Lanelet이 방향성 있는 선형 구간을 표현한다면, Area는 **방향성 없는 다각형 공간**을 표현합니다. 주차장, 교차로 내부, 보행자 광장 등에 사용합니다.

```xml
<relation id="2000">
  <member type="way" ref="200" role="outer"/>
  <tag k="type" v="multipolygon"/>
  <tag k="subtype" v="parking"/>
</relation>
```

#### Regulatory Element (규제 요소)

신호등, 정지 표지, 속도 제한 등 **교통 규제**를 Lanelet에 연결하는 요소입니다. Lanelet2의 의미 정보 표현 능력이 OpenDRIVE보다 강한 부분입니다.

```xml
<!-- 신호등 규제 -->
<relation id="3000">
  <member type="relation" ref="1000" role="refers"/>   <!-- 적용 대상 Lanelet -->
  <member type="node"     ref="500"  role="refers"/>   <!-- 신호등 위치 -->
  <member type="way"      ref="600"  role="stop_line"/>
  <tag k="type"    v="regulatory_element"/>
  <tag k="subtype" v="traffic_light"/>
</relation>

<!-- 정지 표지 규제 -->
<relation id="3001">
  <member type="relation" ref="1001" role="refers"/>
  <member type="node"     ref="501"  role="refers"/>
  <member type="way"      ref="601"  role="stop_line"/>
  <tag k="type"    v="regulatory_element"/>
  <tag k="subtype" v="traffic_sign"/>
</relation>

<!-- 우선권 규칙 -->
<relation id="3002">
  <member type="relation" ref="1002" role="refers"/>
  <member type="relation" ref="1003" role="yield"/>
  <tag k="type"    v="regulatory_element"/>
  <tag k="subtype" v="right_of_way"/>
</relation>
```

### 2.3 토폴로지 표현 방식

Lanelet2는 Linestring 공유를 통해 토폴로지를 표현합니다. 인접한 두 Lanelet이 같은 Linestring을 경계로 공유하면, 라이브러리가 자동으로 이웃 관계를 파악합니다.

```
Lanelet A (id=1000)         Lanelet B (id=1001)
┌─────────────────┐         ┌─────────────────┐
│  left:  way 100 │         │  left:  way 102 │
│  right: way 101 │◄ 공유 ►│  right: way 101 │
└─────────────────┘         └─────────────────┘
```

`way 101`을 A의 right, B의 left로 공유하면 → A와 B가 인접 차선임을 자동 인식합니다.

### 2.4 C++ API 사용 예시

```cpp
#include <lanelet2_io/Io.h>
#include <lanelet2_projection/UTM.h>
#include <lanelet2_routing/RoutingGraph.h>
#include <lanelet2_traffic_rules/TrafficRulesFactory.h>

// 맵 로드
lanelet::projection::UtmProjector projector(lanelet::Origin({49.005, 8.404}));
lanelet::LaneletMapPtr map = lanelet::load("map.osm", projector);

// 경로 계획
auto traffic_rules = lanelet::traffic_rules::TrafficRulesFactory::create(
    lanelet::Locations::Germany, lanelet::Participants::Vehicle);
auto routing_graph = lanelet::routing::RoutingGraph::build(*map, *traffic_rules);

// 특정 Lanelet에서 목적지까지 경로
lanelet::ConstLanelet start = map->laneletLayer.get(1000);
lanelet::ConstLanelet goal  = map->laneletLayer.get(2000);
auto route = routing_graph->getRoute(start, goal);
```

### 2.5 장단점

| | 내용 |
| :--- | :--- |
| **장점** | ROS/ROS2 완벽 통합, Autoware와 바로 연동 가능 |
| **장점** | 의미 정보(신호등·표지판·우선권) 표현 구조가 명확 |
| **장점** | OSM 기반이므로 기존 OSM 데이터 활용 가능 |
| **장점** | C++ 라이브러리가 경로 계획, 공간 질의, 매칭 API 제공 |
| **단점** | 기하 표현력이 OpenDRIVE보다 떨어짐 (클로소이드 없음) |
| **단점** | 상용 시뮬레이터 지원 미흡 (변환 도구 필요) |
| **단점** | 도로 네트워크 규모가 커질수록 파일 관리 복잡 |

---

## 3. 두 포맷 비교

| 항목 | OpenDRIVE (XODR) | Lanelet2 |
| :--- | :--- | :--- |
| **관리 주체** | ASAM (국제 표준) | FZI (오픈소스) |
| **파일 형식** | XML (`.xodr`) | OSM XML (`.osm`) |
| **좌표 기준** | 로컬 좌표계 (x, y, z) | WGS84 or UTM |
| **기하 표현** | 클로소이드, 3차 다항식 등 정밀 커브 | Point 연결 Polyline (근사) |
| **의미 정보** | 신호·표지판을 도로에 직접 첨부 | Regulatory Element로 분리 관리 |
| **차선 연결** | 명시적 laneLink | Linestring 공유로 묵시적 표현 |
| **교차로** | Junction + connecting road | 다수 Lanelet + 가상 경계 |
| **에코시스템** | CARLA, SUMO, CarMaker, dSPACE | Autoware, ROS, CommonRoad |
| **API/툴** | esmini, RoadRunner, VTD | Lanelet2 C++ lib, JOSM 편집기 |
| **학습 난이도** | 중상 (s-t 좌표계 이해 필요) | 중 (OSM 개념과 유사) |

---

## 4. 변환 도구

두 포맷 간 변환이 필요한 경우 다음 도구들을 사용합니다.

### 4.1 XODR → Lanelet2

```bash
# lanelet2_extension 패키지의 변환 노드 (Autoware)
ros2 run lanelet2_extension xodr_converter \
  --input map.xodr \
  --output map.osm
```

### 4.2 Lanelet2 → XODR

직접 변환 도구는 드물며, 보통 중간 포맷(JOSM 편집 후 수동 조정)을 거칩니다.

### 4.3 RoadRunner (MathWorks)

상용 도구인 RoadRunner는 GUI 환경에서 XODR, Lanelet2 등 다양한 포맷으로 내보내기를 지원하므로, 새 맵을 처음부터 만들 때 가장 편리합니다.

---

## 5. 시뮬레이션 스택별 맵 포맷 선택 가이드

| 시뮬레이션 스택 | 권장 포맷 | 비고 |
| :--- | :--- | :--- |
| **CARLA + Scenario Runner** | XODR | CARLA가 XODR 직접 파싱 |
| **SUMO** | XODR 또는 자체 `.net.xml` | netconvert로 XODR → SUMO 변환 |
| **Autoware Universe** | Lanelet2 | `map_loader` 패키지가 직접 로드 |
| **Apollo** | Lanelet2 (또는 자체 포맷) | 변환 레이어 존재 |
| **CommonRoad** | Lanelet2 | 직접 지원 |
| **IPG CarMaker** | XODR | 공식 지원 |

---

## 6. 실무 팁

### CARLA에서 XODR 사용하기

CARLA는 `OpenDriveMap` Actor를 통해 런타임에 XODR 파일을 로드할 수 있습니다.

```python
import carla

client = carla.Client("localhost", 2000)
world = client.get_world()

# XODR 파일로 맵 직접 로드
with open("my_map.xodr", "r") as f:
    xodr_content = f.read()

vertex_distance = 2.0   # 메쉬 버텍스 간격 (m)
max_road_length = 50.0  # 메쉬 분할 최대 길이 (m)
wall_height      = 1.0  # 도로 경계 벽 높이 (m)
extra_width      = 0.6  # 도로 폭 여유 (m)

world = client.generate_opendrive_world(
    xodr_content,
    carla.OpendriveGenerationParameters(
        vertex_distance=vertex_distance,
        max_road_length=max_road_length,
        wall_height=wall_height,
        additional_width=extra_width,
        smooth_junctions=True,
        enable_mesh_visibility=True,
    )
)
```

### Autoware에서 Lanelet2 맵 로드하기

```python
# map_loader 파라미터 설정 (launch 파일)
lanelet2_map_loader_param = {
    "lanelet2_map_path": "/path/to/map.osm",
    "center_line_resolution": 5.0,
}
```

---

## 마치며

| 상황 | 선택 |
| :--- | :--- |
| CARLA/SUMO 기반 시뮬레이션 | **OpenDRIVE** |
| Autoware/ROS 기반 AV 스택 연구 | **Lanelet2** |
| 고정밀 기하가 필요한 고속도로 시나리오 | **OpenDRIVE** |
| 복잡한 교통 규제 의미 표현이 중요할 때 | **Lanelet2** |
| 맵을 새로 제작할 때 | **RoadRunner** (양쪽 내보내기 가능) |

두 포맷은 상호 배타적이지 않습니다. 실제 자율주행 개발에서는 시뮬레이터용 XODR 맵과 온보드 스택용 Lanelet2 맵을 병행하여 관리하는 경우가 많습니다. 변환 파이프라인을 잘 구축해두면 하나의 원본 데이터에서 두 포맷을 동시에 유지할 수 있습니다.
