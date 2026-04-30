---
title: "WGSL이란? GPU에게 말을 거는 언어, 입문자도 이해하는 셰이더 언어"
date: 2026-04-30T15:00:00+09:00
draft: false
tags: ["WGSL", "WebGPU", "셰이더", "GPU", "그래픽스", "wgpu"]
categories: ["컴퓨터 그래픽스"]
description: "WGSL(WebGPU Shading Language)이 무엇인지, 왜 필요한지, 어떻게 생겼는지 입문자도 이해할 수 있도록 설명합니다. GPU와 CPU의 차이부터 시작해 셰이더의 역할, WGSL 문법까지 단계별로 알아봅니다."
---

## 들어가며: GPU에게 명령을 내리려면?

컴퓨터에는 두 종류의 프로세서가 있습니다.

- **CPU**: 복잡한 일을 순서대로 잘 처리하는 "만능 일꾼"
- **GPU**: 단순한 일을 동시에 수천 개 처리하는 "병렬 처리 공장"

화면에 삼각형 하나를 그리는 것도 사실 수천 개의 픽셀을 계산해야 합니다. 이 계산을 GPU에게 맡기면 훨씬 빠릅니다. 그런데 GPU는 CPU와 언어가 다릅니다. **WGSL은 GPU에게 "이렇게 계산해라"고 말하는 언어**입니다.

---

## 1. 셰이더(Shader)란?

GPU에서 실행되는 프로그램을 **셰이더(Shader)**라고 부릅니다. "셰이더"라는 이름은 원래 빛의 명암(shading)을 계산하던 프로그램에서 유래했지만, 지금은 GPU에서 실행되는 모든 프로그램을 통칭합니다.

셰이더는 크게 두 종류입니다.

| 종류 | 역할 | 비유 |
| :--- | :--- | :--- |
| **버텍스 셰이더 (Vertex Shader)** | 3D 좌표 → 화면 좌표 변환 | 건물의 설계도를 화면 위치로 옮김 |
| **프래그먼트 셰이더 (Fragment Shader)** | 각 픽셀의 색상 계산 | 건물의 각 벽면에 페인트 칠하기 |

```
3D 모델 데이터
    ↓
[버텍스 셰이더] ← "각 꼭짓점을 어디에 놓을까?"
    ↓
삼각형 조립 (GPU 자동 처리)
    ↓
[프래그먼트 셰이더] ← "각 픽셀을 무슨 색으로 칠할까?"
    ↓
최종 화면
```

---

## 2. WGSL이란?

**WGSL = WebGPU Shading Language**

W3C(웹 표준 기구)가 만든 셰이더 언어로, WebGPU API와 함께 사용합니다. 웹(브라우저)뿐 아니라 Rust의 **wgpu** 라이브러리를 통해 데스크톱 앱에서도 쓸 수 있습니다.

### 기존 셰이더 언어들과의 비교

| 언어 | 사용 API | 주요 사용처 |
| :--- | :--- | :--- |
| **GLSL** | OpenGL / WebGL | 오래된 웹·데스크톱 그래픽스 |
| **HLSL** | DirectX | Windows 게임 |
| **MSL** | Metal | Apple 기기 |
| **WGSL** | WebGPU / wgpu | 웹, 크로스플랫폼 |

WGSL의 장점은 **하나의 코드로 모든 플랫폼에서 동작**한다는 점입니다. wgpu가 내부적으로 WGSL을 Metal/Vulkan/DX12용 코드로 번역해줍니다.

---

## 3. WGSL 문법 맛보기

### 3.1 가장 단순한 셰이더: 빨간 삼각형

```wgsl
// 버텍스 셰이더: 삼각형의 세 꼭짓점 위치를 반환
@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> @builtin(position) vec4<f32> {
    // 미리 정해둔 세 꼭짓점 좌표 (x, y)
    var positions = array<vec2<f32>, 3>(
        vec2<f32>( 0.0,  0.5),  // 위
        vec2<f32>(-0.5, -0.5),  // 왼쪽 아래
        vec2<f32>( 0.5, -0.5),  // 오른쪽 아래
    );

    let pos = positions[in_vertex_index];
    // vec4: x, y, z(깊이), w(원근) — 화면 좌표는 z=0, w=1
    return vec4<f32>(pos.x, pos.y, 0.0, 1.0);
}

// 프래그먼트 셰이더: 모든 픽셀을 빨간색으로
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    // RGBA: 빨강=1, 초록=0, 파랑=0, 불투명도=1
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
```

이 코드가 하는 일:
1. `vs_main`: 꼭짓점 0, 1, 2의 화면 위치를 지정
2. `fs_main`: 삼각형 안의 모든 픽셀을 빨간색으로 칠함

### 3.2 WGSL 기본 문법 요소

**타입 (Types)**

```wgsl
// 스칼라
let a: f32 = 1.0;    // 32비트 부동소수점
let b: u32 = 42u;    // 32비트 부호 없는 정수
let c: i32 = -7;     // 32비트 정수
let d: bool = true;

// 벡터 (그래픽스에서 자주 씀)
let v2: vec2<f32> = vec2<f32>(1.0, 2.0);       // 2D 좌표
let v3: vec3<f32> = vec3<f32>(1.0, 0.0, 0.0);  // RGB 색상 또는 3D 좌표
let v4: vec4<f32> = vec4<f32>(1.0, 0.0, 0.0, 1.0); // RGBA 또는 동차 좌표

// 행렬 (카메라 변환 등에 사용)
let m: mat4x4<f32> = mat4x4<f32>(...);
```

**벡터 성분 접근**

```wgsl
let color = vec4<f32>(0.2, 0.5, 0.8, 1.0);

// .xyzw 또는 .rgba로 접근 가능 (같은 의미)
let r = color.x;   // 0.2
let g = color.y;   // 0.5
let b = color.z;   // 0.8
let a = color.w;   // 1.0

// 스위즐링: 성분을 자유롭게 조합
let rgb = color.xyz;          // vec3
let inverted = color.zyx;     // 성분 순서 바꾸기
```

**함수**

```wgsl
fn add(a: f32, b: f32) -> f32 {
    return a + b;
}

// 내장 함수들
let s = sin(3.14);         // 사인
let c = cos(0.0);          // 코사인
let l = length(vec3<f32>(1.0, 2.0, 3.0));  // 벡터 길이
let n = normalize(v3);     // 단위 벡터로 정규화
let d = dot(v3a, v3b);     // 내적
let x = mix(0.0, 1.0, 0.5); // 선형 보간 → 0.5
```

---

## 4. GPU와 CPU의 소통 방법

WGSL 셰이더만으로는 아무것도 할 수 없습니다. CPU(Rust/JavaScript 코드)가 데이터를 GPU로 보내줘야 합니다.

### 4.1 바인딩(Binding): 데이터 전달 통로

```wgsl
// GPU 측 (WGSL)
struct CameraUniform {
    view_proj: mat4x4<f32>,  // 카메라 변환 행렬
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;  // CPU에서 보낸 데이터를 여기서 받음

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    // 카메라 행렬로 3D 좌표 변환
    return camera.view_proj * vec4<f32>(position, 1.0);
}
```

```rust
// CPU 측 (Rust)
// 카메라 데이터를 GPU 버퍼에 업로드
queue.write_buffer(&camera_buffer, 0, bytemuck::cast_slice(&[camera_uniform]));
```

`@group(0) @binding(0)`은 "0번 그룹의 0번 슬롯으로 보낸 데이터를 받겠다"는 주소입니다.

### 4.2 버텍스 어트리뷰트: 꼭짓점 데이터

```wgsl
struct VertexInput {
    @location(0) position: vec3<f32>,  // 위치
    @location(1) color:    vec3<f32>,  // 색상
    @location(2) uv:       vec2<f32>,  // 텍스처 좌표
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(in.position, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
```

---

## 5. 실용 예제: 그라디언트 삼각형

꼭짓점마다 색상을 다르게 주고, 픽셀은 그 사이를 자동으로 보간합니다.

```wgsl
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>( 0.0,  0.5),
        vec2<f32>(-0.5, -0.5),
        vec2<f32>( 0.5, -0.5),
    );
    // 꼭짓점마다 다른 색상: 빨강, 초록, 파랑
    var colors = array<vec3<f32>, 3>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
    );

    var out: VertexOutput;
    out.position = vec4<f32>(positions[idx], 0.0, 1.0);
    out.color = colors[idx];
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // GPU가 세 꼭짓점 색상을 자동으로 보간해서 in.color에 넣어줌
    return vec4<f32>(in.color, 1.0);
}
```

결과: 꼭짓점은 빨강·초록·파랑이고, 중간은 자동으로 섞인 색이 됩니다.

---

## 6. WGSL과 다른 셰이더 언어의 차이

GLSL(WebGL)을 써본 적 있다면 익숙하지만 차이가 있습니다.

| 항목 | GLSL | WGSL |
| :--- | :--- | :--- |
| 진입점 선언 | `void main()` | `@vertex fn vs_main()` / `@fragment fn fs_main()` |
| 전역 변수 | `uniform`, `varying`, `attribute` | `@group` `@binding`, `@location` 어트리뷰트 |
| 벡터 타입 | `vec2`, `vec3`, `vec4` | `vec2<f32>`, `vec3<f32>`, `vec4<f32>` |
| 포인터 | 없음 | `ptr<storage, T>` 등 지원 |
| 안전성 | 런타임 오류 가능 | 정적 분석으로 많은 오류 컴파일 타임에 차단 |

---

## 7. WGSL이 쓰이는 곳

- **웹 브라우저**: WebGPU API를 통해 Chrome/Firefox에서 고성능 그래픽스
- **Rust + wgpu**: 데스크톱 앱, 게임, 과학 시각화
- **3D Gaussian Splatting 뷰어**: 가우시안 스플랫을 화면에 렌더링
- **컴퓨트 셰이더**: 머신러닝 추론, 물리 시뮬레이션, 이미지 처리

특히 **컴퓨트 셰이더**를 쓰면 GPU를 렌더링이 아닌 범용 계산에도 활용할 수 있습니다.

```wgsl
// 컴퓨트 셰이더 예시: 배열의 각 원소를 2배로
@group(0) @binding(0) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    data[i] = data[i] * 2.0;  // 64개씩 병렬 처리
}
```

---

## 마치며

WGSL을 한 문장으로 요약하면:

> **"수천 개의 연산을 동시에 처리하는 GPU에게 명령을 내리는 언어"**

CPU 코드(Rust, JavaScript 등)가 데이터를 준비하면, WGSL 셰이더가 GPU에서 병렬로 그 데이터를 처리합니다. 처음에는 `@vertex`, `@fragment`, `vec4` 같은 낯선 키워드가 어색하지만, 결국 하는 일은 단순합니다.

- 버텍스 셰이더: 꼭짓점을 화면 어디에 놓을지
- 프래그먼트 셰이더: 각 픽셀을 무슨 색으로 칠할지

이 두 가지만 기억하면 WGSL의 절반은 이해한 겁니다.

---

## 더 읽을거리

- [WebGPU Shading Language 공식 스펙](https://www.w3.org/TR/WGSL/)
- [Learn WGPU (Rust 튜토리얼)](https://sotrh.github.io/learn-wgpu/)
- [WebGPU Fundamentals](https://webgpufundamentals.org/)
