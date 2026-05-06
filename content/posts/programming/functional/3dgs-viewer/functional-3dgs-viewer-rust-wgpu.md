---
title: "함수형 Rust로 크로스플랫폼 3D Gaussian Splatting 뷰어 개발"
date: 2026-05-06T13:00:00+09:00
draft: false
tags: ["3DGS", "Gaussian Splatting", "Rust", "wgpu", "함수형 프로그래밍", "렌더링"]
categories: ["컴퓨터 그래픽스"]
description: "Rust + wgpu를 사용해 크로스플랫폼 3D Gaussian Splatting 뷰어를 개발합니다. 에릭 노먼드의 계산/액션/데이터 원칙에 따라 함수형 프로그래밍으로 구성하며, 모든 데스크톱 플랫폼에서 동일하게 동작합니다."
---

## 들어가며

3D Gaussian Splatting(3DGS)은 실시간 신경 렌더링 기술로 자율주행, XR, 3D 재구성 분야에서 빠르게 확산되고 있습니다. 대부분의 공개 구현체는 CUDA 기반이지만, **Rust + wgpu 조합**을 사용하면 모든 데스크톱 플랫폼(macOS, Linux, Windows)에서 동일하게 동작하는 크로스플랫폼 뷰어를 만들 수 있습니다.

이 글은 함수형 프로그래밍 원칙(에릭 노먼드의 계산/액션/데이터 분리)을 따르면서 3DGS 뷰어를 구축하는 방법론을 제시합니다.

---

## 1. 기술 선택: Rust + wgpu

| 특성 | 장점 |
| :--- | :--- |
| **크로스플랫폼** | 모든 데스크톱(macOS, Linux, Windows)에서 동일 코드 실행 |
| **성능** | 네이티브 GPU 드라이버 접근, 메모리 안전성과 속도 동시 달성 |
| **GPU API 추상화** | wgpu가 Metal/Vulkan/DX12 자동 선택 |
| **테스트 가능성** | 함수형 구조로 순수 함수 단위 테스트 가능 |

wgpu는 `Backends::all()`을 지정하면 플랫폼에 따라 Metal/Vulkan/DX12를 자동으로 선택합니다. 별도의 플랫폼 종속 코드가 필요 없습니다.

> `tokio` 대신 `pollster`를 사용합니다. winit의 이벤트 루프는 메인 스레드를 블로킹 점유하므로 `tokio::main`과 충돌합니다. `pollster::block_on`으로 async 초기화를 처리하고 이벤트 루프는 동기로 실행합니다.

---

## 2. 아키텍처: 데이터 파이프라인

```
PLY 파일 (3DGS 가우시안)
    ↓
Rust에서 Binary Parser (bytemuck 사용)
    ↓
GPU 메모리 업로드 (wgpu Buffer)
    ↓
Compute Shader에서 가우시안 정렬 및 투영
    ↓
Render Shader에서 Splatting 렌더링
    ↓
Metal/Vulkan → 윈도우 출력
```

---

## 3. 함수형 프로그래밍으로 설계

### 3.1 계산 / 액션 / 데이터 분리

**에릭 노먼드의 핵심 원칙:**

| 카테고리 | 특징 | 예시 |
| :--- | :--- | :--- |
| **데이터** | 불변, 순수 값 | `Gaussian`, `CameraState`, `AppState` |
| **계산** | 입력→출력, 부작용 없음 | `update_camera_angles()`, `sort_gaussians_by_depth()` |
| **액션** | 부작용, 외부 세계 상호작용 | `load_ply_file()`, `render_frame()` |

이 구분이 중요한 이유는 **테스트 가능 범위**가 명확해지기 때문입니다. `data/`와 `compute/`는 GPU 없이 100% 단위 테스트할 수 있고, `action/`만 통합 테스트 대상이 됩니다.

### 3.2 프로젝트 구조

```
src/
├── main.rs                    # 액션: 진입점, 메인 루프
├── data/
│   ├── gaussian.rs           # 데이터: Gaussian (CPU) + GaussianGpu (GPU 레이아웃)
│   ├── camera.rs             # 데이터: CameraState 구조체
│   └── app_state.rs          # 데이터: AppState
├── compute/                  # 순수 계산 함수들
│   ├── ply_parse.rs          # 계산: 바이너리 파싱
│   ├── camera_ops.rs         # 계산: 카메라 변환
│   └── gaussian_ops.rs       # 계산: 깊이 정렬, GPU 변환
├── action/                   # 부작용이 있는 함수들
│   ├── io.rs                 # 액션: 파일 읽기
│   ├── gpu.rs                # 액션: GPU 버퍼 생성/갱신
│   └── render.rs             # 액션: 렌더링
└── shaders/
    └── render.wgsl
```

---

## 4. 데이터 계층 (src/data/)

데이터 구조체는 불변 값을 표현합니다. 메서드를 가질 수 있지만 self를 변경하지 않습니다.

**`CameraState`** — 카메라의 구면 좌표계 상태를 담는 순수 값입니다. 메서드 없이 필드만 노출하며, 변환 로직은 `compute/camera_ops.rs`에 위임합니다.

**`AppState`** — 앱 전체 상태를 담습니다. `gaussians`를 `Arc<Vec<Gaussian>>`으로 감싸서, 카메라만 바뀔 때 가우시안 데이터를 복사하지 않고 포인터만 공유합니다.

```rust
pub fn with_camera(&self, camera: CameraState) -> Self {
    AppState {
        camera,
        gaussians: Arc::clone(&self.gaussians),  // O(1) clone
    }
}
```

**`GaussianGpu`** — GPU 업로드 전용 구조체입니다. WGSL에서 `array<f32, 45>` 같은 배열 멤버는 16바이트 정렬을 요구하므로, CPU 측 `Gaussian`과 레이아웃이 다릅니다. `From<&Gaussian>` 변환으로 명시적으로 분리합니다.

```rust
pub struct GaussianGpu {
    pub pos: [f32; 3],
    pub opacity: f32,       // 16바이트 정렬
    pub color_dc: [f32; 3],
    pub _pad0: f32,         // 패딩
    pub scale: [f32; 3],
    pub _pad1: f32,
    pub rot: [f32; 4],
    pub f_rest: [f32; 45],
    pub _pad2: [f32; 3],    // 45 → 48로 맞춤
}
```

---

## 5. 계산 계층 (src/compute/)

### 5.1 카메라 연산

카메라 조작의 핵심은 **상태를 변경하지 않고 새 상태를 반환**하는 것입니다.

```rust
// ❌ 나쁜 예: 계산과 상태 변경이 섞임
impl Camera {
    fn update_from_input(&mut self, input: Input) {
        self.theta += input.delta_x * 0.01;
    }
}

// ✅ 좋은 예: 입력 → 새 상태를 반환하는 순수 함수
pub fn update_camera_angles(state: CameraState, dx: f32, dy: f32) -> CameraState {
    CameraState {
        theta: state.theta + dx * 0.01,
        phi: (state.phi + dy * 0.01).clamp(0.1, PI - 0.1),
        ..state
    }
}
```

카메라 위치 계산, 뷰 행렬 생성, 줌 모두 같은 방식으로 순수 함수로 정의합니다. 이 함수들은 `&mut self` 없이 `CameraState`를 받아 새 값을 반환합니다.

### 5.2 가우시안 깊이 정렬

Alpha Blending은 back-to-front 순서를 요구합니다. `rayon`의 `par_sort_by`로 멀티코어 정렬을 수행하며, `length_squared`로 `sqrt` 연산을 생략합니다.

```rust
pub fn sort_gaussians_by_depth(gaussians: &[Gaussian], camera_pos: Vec3) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..gaussians.len()).collect();
    indices.par_sort_by(|&a, &b| {
        let dist_a = (gaussians[a].position() - camera_pos).length_squared();
        let dist_b = (gaussians[b].position() - camera_pos).length_squared();
        dist_b.partial_cmp(&dist_a).unwrap()
    });
    indices
}
```

### 5.3 PLY 파싱

바이너리 PLY 파싱도 순수 함수입니다. 파일 읽기(액션)와 파싱(계산)을 분리하면, `&[u8]`을 직접 넘겨 파싱 로직만 단독 테스트할 수 있습니다.

```rust
// 계산: &[u8] → Gaussian (부작용 없음, 테스트 가능)
pub fn parse_gaussians(data: &[u8], stride: usize, count: usize) -> Result<Vec<Gaussian>, String>

// 액션: 파일 경로 → Gaussian (디스크 접근, 테스트 어려움)
pub fn load_ply_file(path: &str) -> anyhow::Result<Vec<Gaussian>>
```

3DGS SH degree=3 기준 stride는 `3(pos) + 3(normal) + 3(f_dc) + 45(f_rest) + 1(opacity) + 3(scale) + 4(rot) = 62개 f32 = 248바이트`입니다.

---

## 6. 액션 계층 (src/action/)

액션은 계산 결과를 받아 외부 세계에 적용합니다. GPU 버퍼 생성, 파일 읽기, 렌더링이 여기에 해당합니다.

**중요한 설계 결정**: GPU 버퍼를 만들 때 `CameraUniform` 데이터 조립은 순수 함수(`build_camera_uniform`)로 분리하고, 버퍼 생성만 액션으로 둡니다. 이렇게 하면 유니폼 데이터가 올바른지 GPU 없이도 검증할 수 있습니다.

렌더링에서 가우시안 하나당 빌보드 쿼드 6개 버텍스를 사용합니다. 버텍스 버퍼 없이 `vertex_index`만으로 쿼드를 생성하는 방식입니다.

```rust
// 가우시안 하나당 빌보드 쿼드 6 vertex
rpass.draw(0..gaussian_count * 6, 0..1);
```

---

## 7. 셰이더 (src/shaders/render.wgsl)

WGSL은 Metal/Vulkan/DX12로 자동 변환되는 wgpu 표준 셰이더 언어입니다.

버텍스 셰이더는 `vertex_index`로 어느 가우시안의 어느 코너인지 계산합니다. 별도의 버텍스 버퍼 없이 Storage Buffer에서 직접 가우시안 데이터를 읽습니다.

```wgsl
@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    let gaussian_idx = idx / 6u;
    let corner_idx = idx % 6u;
    // corners 배열에서 UV 읽어 빌보드 오프셋 계산
}
```

프래그먼트 셰이더는 UV 기준으로 가우시안 감쇠(Gaussian falloff)를 계산합니다. 원 밖(`dist2 > 1.0`)은 `discard`로 버리고, 안쪽은 `exp(-0.5 * dist2 * 4.0)`으로 가중치를 줍니다.

---

## 8. 메인 루프: 액션 조율

메인 루프는 계산과 액션을 순서대로 연결하는 역할만 합니다.

```rust
fn update(&mut self, input: InputEvent) {
    // 1. 순수 계산: 새 카메라 상태 생성
    let new_camera = match input {
        InputEvent::MouseMove(dx, dy) => camera_ops::update_camera_angles(self.state.camera, dx, dy),
        InputEvent::Zoom(delta) => camera_ops::zoom_camera(self.state.camera, delta),
        InputEvent::None => self.state.camera,
    };

    // 2. 상태 교체 (O(1) — gaussians는 Arc 포인터만 복사)
    self.state = self.state.with_camera(new_camera);

    // 3. 액션: GPU 버퍼 업데이트
    let view_matrix = camera_ops::camera_to_view_matrix(self.state.camera);
    gpu::update_buffer(&self.queue, &self.camera_buffer, &[build_camera_uniform(...)]);
}
```

이 패턴의 핵심: **계산이 먼저, 액션은 마지막**. 상태 변경 의도를 순수 함수로 표현한 뒤, 그 결과를 액션에 넘깁니다.

---

## 9. 함수형 설계의 실제 이점

### 테스트 가능성

순수 함수는 GPU 없이 테스트할 수 있습니다. 렌더링 파이프라인 전체를 띄우지 않아도 카메라 로직, 정렬, 파싱을 독립적으로 검증합니다.

```rust
#[test]
fn test_zoom_preserves_angles() {
    let state = CameraState::new();
    let zoomed = camera_ops::zoom_camera(state, 1.0);
    assert!(zoomed.radius < state.radius);   // 줌인
    assert_eq!(zoomed.theta, state.theta);   // 각도는 불변
}

#[test]
fn test_depth_sort_order() {
    // 가까운 가우시안이 정렬 결과 앞에 와야 함
    let sorted = gaussian_ops::sort_gaussians_by_depth(&gaussians, Vec3::ZERO);
    assert_eq!(sorted[0], closer_gaussian_index);
}
```

### 상태 관리의 명확성

`with_camera` 같은 패턴은 어느 필드가 바뀌었는지 타입 수준에서 드러냅니다. `&mut App`을 넘기면 내부에서 무엇이 변경되는지 함수 시그니처만으로는 알 수 없습니다.

### 병렬화

순수 함수는 공유 상태가 없으므로 `rayon`으로 데이터 병렬화를 바로 적용할 수 있습니다. 락 없이 멀티코어를 활용합니다.

---

## 10. 성능 최적화

### 매 프레임 할당 방지

정렬 인덱스 버퍼를 매 프레임 새로 할당하는 대신, `out: &mut Vec<u32>`로 받아 `clear()` 후 재사용합니다.

### 적응형 LOD

거리에 따라 렌더링할 가우시안을 필터링하면 GPU 부하를 줄일 수 있습니다. 필터링 자체는 순수 함수이므로 결과를 캐시하거나 병렬 처리하기 쉽습니다.

### GPU 기반 깊이 정렬

CPU `rayon` 정렬은 시작점으로 충분하지만, 대용량 장면에서는 Compute Shader 기반 Bitonic Sort로 전환해야 합니다. `action/` 계층의 렌더 함수 하나만 교체하면 되고, 계산/데이터 계층은 그대로 유지됩니다.

---

## 11. 문제 해결

### 가우시안이 보이지 않음

계산 계층부터 단계적으로 검증합니다. 카메라 위치가 올바른지 (`compute_camera_position` 테스트), 가우시안 데이터가 유효한지 (`validate_gaussian_data` 순수 함수), 마지막으로 GPU 업로드 결과를 확인하는 순서입니다.

### 성능이 너무 느림

`cargo bench`로 계산 함수별 병목을 측정합니다. 순수 함수는 입력을 고정할 수 있어 벤치마크가 안정적입니다. LOD 필터링 → Rayon 병렬화 → GPU Compute Sort 순으로 단계적으로 적용합니다.

### Metal 벡터 인코딩 오류

wgpu 버전과 macOS 버전 불일치가 원인인 경우가 많습니다. `cargo update wgpu`로 해결되며, macOS 12+ 가 필요합니다.

---

## 다음 단계

- **Spherical Harmonics 렌더링**: 시점 의존적 색상 계산. `f_rest` 45개 계수를 사용하는 순수 함수로 추가합니다.
- **GPU 기반 깊이 정렬**: `action/render.rs`에 Compute Shader 패스를 추가합니다.
- **egui UI**: 상태 → 새 상태 패턴을 그대로 따르므로 기존 구조와 자연스럽게 통합됩니다.

---

## 참고자료

- **3DGS 공식 논문**: Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (SIGGRAPH 2023)
- **3DGS 공식 구현**: https://github.com/graphdeco-inria/gaussian-splatting
- **WGPU 문서**: https://docs.rs/wgpu/
- **PLY 포맷**: [PLY 포맷 총정리]({{< ref "ply-format-overview.md" >}})
- **3D 데이터 표현**: [3D 데이터 표현 방식 총정리]({{< ref "3d-data-representations.md" >}})
