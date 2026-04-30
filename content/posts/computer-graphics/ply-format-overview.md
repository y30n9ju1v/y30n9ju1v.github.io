---
title: "PLY 포맷 총정리: Point Cloud와 3DGS의 사실상 표준"
date: 2026-04-30T10:45:00+09:00
draft: false
tags: ["컴퓨터 그래픽스", "3D", "PLY 포맷", "포인트 클라우드", "3DGS", "데이터 포맷"]
categories: ["컴퓨터 그래픽스"]
description: "Point Cloud와 3D Gaussian Splatting에서 핵심 데이터 저장 포맷으로 널리 사용되는 PLY 포맷의 구조, 특징, 그리고 확장성에 대해 알아봅니다."
---

**PLY(Polygon File Format)**는 1990년대 중반 스탠포드 대학에서 3D 스캐너로 얻은 데이터를 저장하기 위해 개발한 3D 모델 파일 포맷입니다. 스탠포드 그래픽스 연구소에서 만들었기 때문에 **Stanford Triangle Format**이라고도 불립니다.

처음에는 다각형(Polygon) 데이터를 저장하기 위해 만들어졌지만, 현재는 뛰어난 확장성 덕분에 **Point Cloud**와 **3D Gaussian Splatting(3DGS)** 파라미터를 저장하는 핵심 포맷으로 널리 쓰이고 있습니다.

## 1. PLY 포맷의 핵심 구조

PLY 포맷의 가장 큰 특징은 **데이터의 구조를 사용자가 자유롭게 정의할 수 있는 유연성**에 있습니다. 파일은 크게 인간이 읽을 수 있는 **헤더(Header)** 부분과 실제 데이터가 담긴 **바디(Body)** 부분으로 나뉩니다.

### 지원 데이터 타입

PLY 헤더의 `property` 선언에서 사용할 수 있는 타입은 다음과 같습니다.

| 타입 키워드 | 크기 | 범위 / 용도 |
| :--- | :--- | :--- |
| `char` / `int8` | 1 byte | -128 ~ 127 |
| `uchar` / `uint8` | 1 byte | 0 ~ 255. RGB 색상에 주로 사용 |
| `short` / `int16` | 2 bytes | -32768 ~ 32767 |
| `ushort` / `uint16` | 2 bytes | 0 ~ 65535 |
| `int` / `int32` | 4 bytes | 정점 인덱스 등 정수값에 주로 사용 |
| `uint` / `uint32` | 4 bytes | 0 ~ 4,294,967,295 |
| `float` / `float32` | 4 bytes | 좌표(x, y, z), 법선, SH 계수 등 실수값에 주로 사용 |
| `double` / `float64` | 8 bytes | 고정밀 좌표가 필요할 때 (GPS 좌표 등) |

### 헤더 (Header)
헤더는 항상 ASCII 텍스트로 작성되며, 파일에 어떤 데이터가 들어있는지(Element)와 각 데이터가 어떤 속성(Property)을 가지는지 정의합니다.

```text
ply
format ascii 1.0           // 포맷 형식 (ascii, binary_little_endian, binary_big_endian)
comment 작성자: 홍길동        // 주석
element vertex 3           // 'vertex'라는 요소가 3개 있음을 선언
property float x           // vertex의 속성 1: x 좌표
property float y           // vertex의 속성 2: y 좌표
property float z           // vertex의 속성 3: z 좌표
property uchar red         // vertex의 속성 4: 빨간색 (0-255)
property uchar green       // vertex의 속성 5: 초록색 (0-255)
property uchar blue        // vertex의 속성 6: 파란색 (0-255)
element face 1             // 'face(면)' 요소가 1개 있음을 선언
property list uchar int vertex_indices // face의 속성: 리스트 형태의 정점 인덱스
end_header                 // 헤더 종료

```

헤더에서 `property list uchar int vertex_indices`가 특이해 보일 수 있습니다. 이는 **면(Face)마다 꼭짓점 수가 다를 수 있기 때문**입니다. 삼각형은 3개, 사각형은 4개처럼 길이가 가변적인 데이터를 표현하기 위해 `list` 타입을 씁니다. `uchar`은 "뒤따르는 원소 개수를 저장하는 타입", `int`는 "각 원소(정점 인덱스)의 타입"입니다.

### 바디 (Body)
`end_header` 이후부터는 헤더에서 선언한 순서와 타입에 맞춰 실제 데이터가 나열됩니다. 바디에는 주석을 쓸 수 없습니다.

```text
0.0 0.0 0.0 255 0 0
0.0 1.0 0.0 0 255 0
1.0 0.0 0.0 0 0 255
3 0 1 2
```

순서대로 정점(vertex) 3개 (x, y, z, r, g, b), 그 뒤 면(face) 1개 (정점 개수, 정점 인덱스들)입니다.

## 2. ASCII vs Binary 포맷

PLY는 세 가지 저장 방식을 지원합니다.
1. **ASCII**: 텍스트 편집기로 열어서 직접 읽고 수정할 수 있습니다. 디버깅에 유리하지만 용량이 매우 큽니다.
2. **Binary Little Endian**: 데이터를 이진(Binary) 형태로 저장하여 파일 크기를 대폭 줄이고 읽기/쓰기 속도를 높입니다. (현대 PC/GPU 환경에서 주로 사용)
3. **Binary Big Endian**: 다른 아키텍처를 위한 이진 포맷입니다.

용량 차이는 상당합니다. x, y, z, r, g, b 6개 속성을 가진 점 100만 개 기준으로, ASCII는 각 숫자를 텍스트로 저장하므로 약 **50~80MB**가 되지만, Binary는 float32(4 bytes) × 3 + uint8(1 byte) × 3 = 15 bytes/점으로 **약 15MB**입니다. 즉 파일 크기가 3~5배 차이납니다.

대용량의 Point Cloud나 3DGS 데이터를 다룰 때는 필수적으로 **Binary** 포맷을 사용합니다.

## 3. 왜 현대 3D 비전/그래픽스에서 PLY가 각광받는가?

전통적인 게임이나 CG 산업에서는 OBJ나 FBX 포맷을 많이 사용하지만, 3D 비전 연구(특히 최근의 뉴럴 렌더링)에서는 PLY가 압도적으로 많이 쓰입니다. 그 이유는 **"Custom Property(사용자 정의 속성)"의 무한한 확장성** 때문입니다.

### Point Cloud의 표준
OBJ 파일은 기본적으로 정점(x, y, z)과 면을 정의하는 데 최적화되어 있어, 각 정점마다 색상(RGB), 레이저 반사율(Intensity), 법선 벡터(Normal) 등의 추가 정보를 유연하게 담기 어렵습니다. 반면 PLY는 헤더에 `property float intensity` 한 줄만 추가하면 어떤 속성이든 쉽게 확장할 수 있습니다.

### 3D Gaussian Splatting (3DGS)과의 찰떡궁합
최근 3DGS가 등장하면서 PLY 포맷의 진가가 다시 한번 발휘되었습니다. 3DGS는 수백만 개의 3D 가우시안 파라미터를 저장해야 하는데, 3DGS의 공식 구현체는 새로운 파일 포맷을 고안하는 대신 **PLY 포맷을 그대로 활용**했습니다.

3DGS `.ply` 파일의 헤더를 실제로 열어보면 아래와 같이 정의되어 있습니다:

```text
ply
format binary_little_endian 1.0
element vertex 3500000
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float f_rest_0
property float f_rest_1
...
property float f_rest_44
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
```

각 속성의 의미는 다음과 같습니다:
- `x, y, z`: 가우시안의 중심 위치 (Position)
- `nx, ny, nz`: 법선 벡터 (학습 시 사용, 렌더링에는 미사용)
- `f_dc_0, f_dc_1, f_dc_2`: 기본 색상 (Spherical Harmonics의 DC 성분, l=0)
- `f_rest_0 ~ f_rest_44`: 시점 의존적 색상을 위한 SH 나머지 계수 (degree 3까지 = 45개)
- `opacity`: 불투명도 (sigmoid를 통과하기 전의 raw 값)
- `scale_0, scale_1, scale_2`: 3D 스케일 (log scale로 저장)
- `rot_0, rot_1, rot_2, rot_3`: 회전을 나타내는 쿼터니언 (w, x, y, z 순서)

이처럼 PLY는 단순히 다각형 모델을 넘어서 **"3D 공간상의 데이터 배열(Array-of-Structs)을 직렬화하는 범용 컨테이너"** 역할을 완벽히 수행하고 있습니다.

## 4. 다른 포맷과의 비교

| 특성 | PLY | OBJ | STL | LAS / LAZ | glTF |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **주 목적** | Point Cloud, 스캔 데이터, 3DGS | 폴리곤 메시 (텍스처·머티리얼) | 3D 프린팅 (순수 기하학) | LiDAR Point Cloud (지형·측량) | 범용 3D 씬 (웹·게임) |
| **데이터 구조** | 유연 (사용자 정의 속성) | 고정 (v, vn, vt, f) | 단순 (삼각형·법선만) | 고정 + 표준 확장 필드 | 계층적 씬 그래프 |
| **바이너리 지원** | 공식 지원 | 미지원 (텍스트 전용) | 공식 지원 | 공식 지원 (LAZ는 압축까지) | 공식 지원 (GLB) |
| **Point Cloud 처리** | **최우수** (임의 속성 추가 가능) | 부적합 | 불가능 | **최우수** (GPS, Intensity 표준화) | 불가능 |
| **씬 그래프·애니메이션** | 불가능 | 부분 지원 (MTL) | 불가능 | 불가능 | **최우수** |
| **압축** | 미지원 | 미지원 | 미지원 | LAZ로 ~10:1 압축 | Draco 압축 지원 |
| **주요 사용처** | 3D 비전 연구, 3DGS | 3D 모델링, CG | 3D 프린팅 | 자율주행 LiDAR, GIS 측량 | 웹 3D, AR/VR |

> **LAS / LAZ**: LiDAR 데이터 전용 표준 포맷입니다. GPS 시간, 반사율(Intensity), 스캔 각도 등 LiDAR 특화 필드가 표준으로 정의되어 있습니다. LAZ는 LAS의 무손실 압축 버전으로, 자율주행 데이터셋(nuScenes, Waymo Open Dataset)에서 널리 사용됩니다. PLY가 "자유롭게 정의"하는 방식이라면, LAS는 "LiDAR에 최적화된 사전 정의 스키마"라고 볼 수 있습니다.

## 5. Python에서 PLY 읽고 쓰기

실제로 PLY 파일을 다루는 방법은 라이브러리마다 조금씩 다릅니다. 대표적인 두 가지 방법을 소개합니다.

### Open3D (Point Cloud 중심)

```python
import open3d as o3d
import numpy as np

# PLY 읽기
pcd = o3d.io.read_point_cloud("cloud.ply")
points = np.asarray(pcd.points)   # (N, 3) ndarray
colors = np.asarray(pcd.colors)   # (N, 3) ndarray, 0~1 범위

# PLY 쓰기
pcd_out = o3d.geometry.PointCloud()
pcd_out.points = o3d.utility.Vector3dVector(points)
pcd_out.colors = o3d.utility.Vector3dVector(colors)
o3d.io.write_point_cloud("out.ply", pcd_out, write_ascii=False)
```

Open3D는 Point Cloud와 Mesh를 쉽게 다룰 수 있지만, **커스텀 속성(SH 계수, opacity 등)은 직접 접근이 어렵습니다**.

### plyfile (커스텀 속성 중심)

3DGS처럼 커스텀 속성이 많은 PLY를 다룰 때는 `plyfile` 라이브러리가 더 적합합니다.

```python
from plyfile import PlyData, PlyElement
import numpy as np

# PLY 읽기
plydata = PlyData.read("gaussian.ply")
vertex = plydata["vertex"]

xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1)  # (N, 3)
opacity = vertex["opacity"]                                        # (N,)
scales = np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=-1)

# PLY 쓰기 (커스텀 속성 포함)
dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("opacity", "f4")]
arr = np.zeros(len(xyz), dtype=dtype)
arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
arr["opacity"] = opacity

el = PlyElement.describe(arr, "vertex")
PlyData([el], byte_order="<").write("out.ply")  # "<" = little endian
```

| | Open3D | plyfile |
| :--- | :--- | :--- |
| **설치** | `pip install open3d` | `pip install plyfile` |
| **커스텀 속성 접근** | 제한적 | 완전한 딕셔너리 방식 접근 |
| **시각화** | 내장 (`o3d.visualization`) | 없음 |
| **속도** | 빠름 (C++ 백엔드) | 순수 Python, 대용량에 다소 느림 |
| **추천 상황** | Point Cloud 처리·시각화 | 3DGS 등 커스텀 속성 다수인 경우 |

## 6. PLY 포맷의 한계

"총정리"를 위해서는 단점도 알아야 합니다. PLY는 1994년에 설계된 포맷으로, 현대적인 관점에서 몇 가지 한계가 있습니다.

- **스키마 버전 관리 없음**: 헤더에 커스텀 속성을 자유롭게 추가할 수 있지만, 공식 버전 관리 메커니즘이 없습니다. 예를 들어 3DGS 구현체마다 SH 계수 속성 이름이 조금씩 다를 수 있어, 파일을 읽는 쪽에서 스키마를 직접 파악해야 합니다.
- **계층 구조 불가**: Element 간의 계층 관계(예: 씬 → 오브젝트 → 메시)를 표현할 수 없습니다. 복잡한 씬 그래프가 필요하다면 glTF나 USD 같은 포맷이 더 적합합니다.
- **압축 미지원**: Binary 포맷이라도 데이터 압축은 지원하지 않습니다. 수백만 개의 가우시안을 다루는 3DGS 파일은 수백 MB~수 GB에 달할 수 있어, 실제 서비스에서는 별도 압축이 필요합니다.
- **메타데이터 표준 없음**: `comment` 키워드로 주석을 달 수 있지만, 좌표계·단위·타임스탬프 같은 메타데이터를 위한 표준 규격이 없습니다.

이러한 한계에도 불구하고, **단순함과 확장성의 균형**이 뛰어나기 때문에 3D 비전 연구 커뮤니티에서의 인기는 여전합니다.
