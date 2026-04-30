---
title: "함수형 Rust로 크로스플랫폼 3D Gaussian Splatting 뷰어 개발"
date: 2026-04-30T13:00:00+09:00
draft: false
tags: ["3DGS", "Gaussian Splatting", "Rust", "wgpu", "함수형 프로그래밍", "렌더링"]
categories: ["컴퓨터 그래픽스"]
description: "Rust + wgpu를 사용해 크로스플랫폼 3D Gaussian Splatting 뷰어를 개발합니다. 에릭 노먼드의 계산/액션/데이터 원칙에 따라 함수형 프로그래밍으로 구성하며, 모든 데스크톱 플랫폼에서 동일하게 동작합니다."
---

## 들어가며

3D Gaussian Splatting(3DGS)은 실시간 신경 렌더링 기술로 자율주행, XR, 3D 재구성 분야에서 빠르게 확산되고 있습니다. 대부분의 공개 구현체는 CUDA 기반이지만, **Rust + wgpu 조합**을 사용하면 모든 데스크톱 플랫폼(macOS, Linux, Windows)에서 동일하게 동작하는 크로스플랫폼 뷰어를 만들 수 있습니다.

이 글은 함수형 프로그래밍 원칙(에릭 노먼드의 계산/액션/데이터 분리)을 따르면서 3DGS 뷰어를 구축하는 방법론을 제시합니다. 플랫폼 독립적이며 테스트 가능하고 유지보수하기 쉬운 구조를 제공합니다.

---

## 1. 기술 선택: Rust + wgpu

### 1.1 왜 Rust + wgpu인가?

| 특성 | 장점 |
| :--- | :--- |
| **크로스플랫폼** | 모든 데스크톱(macOS, Linux, Windows)에서 동일 코드 실행 |
| **성능** | 네이티브 GPU 드라이버 접근, 메모리 안전성과 속도 동시 달성 |
| **GPU API 추상화** | wgpu가 Metal/Vulkan/DX12 자동 선택 (플랫폼 종속성 제거) |
| **메모리 안전성** | Rust의 소유권 시스템으로 버퍼 오버플로우 방지 |
| **테스트 가능성** | 함수형 구조로 순수 함수 테스트 가능 |

### 1.2 wgpu의 자동 플랫폼 선택

```rust
// 자동으로 적절한 GPU API 선택 (코드는 동일)
let instance = Instance::new(InstanceDescriptor {
    backends: Backends::all(),  // wgpu가 자동으로 최적 선택
    ..Default::default()
});
```

- **Linux**: Vulkan
- **Windows**: DX12 (또는 Vulkan)
- **웹(WASM)**: WebGPU

별도의 플랫폼 종속 코드가 필요 없습니다.

---

## 2. 프로젝트 설정

### 2.1 Cargo.toml

```toml
[package]
name = "3dgs-viewer"
version = "0.1.0"
edition = "2021"

[dependencies]
wgpu = "0.20"
winit = "0.29"
glam = "0.28"
bytemuck = { version = "1.14", features = ["derive"] }
anyhow = "1.0"
log = "0.4"
env_logger = "0.11"
pollster = "0.3"
rayon = "1.8"

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
strip = true
```

> `tokio` 대신 `pollster`를 사용합니다. winit의 이벤트 루프는 메인 스레드를 블로킹 점유하므로 `tokio::main`과 충돌합니다. `pollster::block_on`으로 async 초기화를 처리하고 이벤트 루프는 동기로 실행합니다.

---

## 3. 아키텍처 설계

### 3.1 데이터 파이프라인

```
PLY 파일 (3DGS 가우시안)
    ↓
Rust에서 Binary Parser (bytemuck, nom 사용)
    ↓
GPU 메모리 업로드 (wgpu Buffer)
    ↓
Compute Shader에서 가우시안 정렬 및 투영
    ↓
Render Shader에서 Splatting 렌더링
    ↓
Metal/Vulkan → 윈도우 출력
```

### 3.2 핵심 컴포넌트

1. **PLY 로더**: `ply-rs` 또는 직접 파싱으로 바이너리 PLY 로드
2. **버퍼 관리**: wgpu로 GPU 메모리 업로드 및 생명주기 관리
3. **Compute Shader**: 가우시안 깊이 정렬, 2D 투영, 코바리언스 계산
4. **Render Shader**: Alpha Blending으로 최종 이미지 합성
5. **윈도우/입력**: winit로 창 관리, 카메라 제어

---

## 4. 함수형 프로그래밍으로 설계

### 4.1 계산(Pure Functions) vs 액션(Side Effects) vs 데이터 구분

**핵심 원칙** (에릭 노먼드):
- **데이터**: 값 그 자체 (불변)
- **계산**: 입력 → 출력, 부작용 없음 (순수함수)
- **액션**: 부작용을 일으키는 작업 (IO, 렌더링)

```rust
// ❌ 나쁜 예: 계산과 액션이 섞여있음
impl Camera {
    fn update_from_input(&mut self, input: Input) {
        self.theta += input.delta_x * 0.01;  // 액션(상태 변경)
        self.position = self.compute_position(); // 계산(함께 섞여있음)
    }
}

// ✅ 좋은 예: 계산과 액션을 분리
fn update_camera_angles(theta: f32, phi: f32, input: &InputDelta) -> (f32, f32) {
    let new_theta = theta + input.delta_x * 0.01;
    let new_phi = (phi + input.delta_y * 0.01).clamp(0.1, PI - 0.1);
    (new_theta, new_phi)
}

fn compute_camera_position(theta: f32, phi: f32, radius: f32) -> Vec3 {
    Vec3::new(
        radius * phi.sin() * theta.cos(),
        radius * phi.cos(),
        radius * phi.sin() * theta.sin(),
    )
}
```

### 4.2 프로젝트 구조 (FP 스타일)

```
3dgs-viewer/
├── src/
│   ├── main.rs                    # 액션: 진입점, 메인 루프
│   ├── data/
│   │   ├── mod.rs
│   │   ├── gaussian.rs           # 데이터: Gaussian 구조체
│   │   ├── camera.rs             # 데이터: CameraState 구조체
│   │   └── app_state.rs          # 데이터: AppState 구조체
│   ├── compute/                  # 순수 계산 함수들
│   │   ├── mod.rs
│   │   ├── ply_parse.rs          # 계산: 바이너리 파싱
│   │   ├── camera_ops.rs         # 계산: 카메라 변환
│   │   ├── gaussian_ops.rs       # 계산: 가우시안 연산
│   │   └── shader_compile.rs     # 계산: 셰이더 생성
│   ├── action/                   # 부작용이 있는 함수들
│   │   ├── mod.rs
│   │   ├── io.rs                 # 액션: 파일 읽기
│   │   ├── gpu.rs                # 액션: GPU 버퍼 생성
│   │   ├── render.rs             # 액션: 렌더링
│   │   └── window.rs             # 액션: 윈도우 관리
│   └── shaders/
│       ├── render.wgsl
│       └── compute.wgsl
├── Cargo.toml
└── assets/
    └── example.ply
```

### 4.3 Step 1: 데이터 정의 (src/data/)

**데이터 구조체들 - 불변, 순수 값 표현:**

```rust
// src/data/gaussian.rs
use glam::Vec3;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Pod, Zeroable, Clone, Copy, Debug)]
pub struct Gaussian {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub nx: f32,
    pub ny: f32,
    pub nz: f32,
    pub f_dc_0: f32,
    pub f_dc_1: f32,
    pub f_dc_2: f32,
    pub f_rest: [f32; 45],
    pub opacity: f32,
    pub scale_0: f32,
    pub scale_1: f32,
    pub scale_2: f32,
    pub rot_0: f32,
    pub rot_1: f32,
    pub rot_2: f32,
    pub rot_3: f32,
}

impl Gaussian {
    pub fn position(&self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }
    
    pub fn color(&self) -> Vec3 {
        Vec3::new(self.f_dc_0, self.f_dc_1, self.f_dc_2)
    }
    
    pub fn scale(&self) -> Vec3 {
        Vec3::new(self.scale_0, self.scale_1, self.scale_2)
    }
    
    pub fn rotation(&self) -> [f32; 4] {
        [self.rot_0, self.rot_1, self.rot_2, self.rot_3]
    }
}

// src/data/camera.rs
#[derive(Clone, Copy, Debug)]
pub struct CameraState {
    pub theta: f32,
    pub phi: f32,
    pub radius: f32,
}

impl CameraState {
    pub fn new() -> Self {
        use std::f32::consts::PI;
        Self {
            theta: 0.0,
            phi: PI / 4.0,
            radius: 2.0,
        }
    }
}

// src/data/app_state.rs
pub struct AppState {
    pub camera: CameraState,
    pub gaussians: Vec<Gaussian>,
}

impl AppState {
    pub fn new(gaussians: Vec<Gaussian>) -> Self {
        Self {
            camera: CameraState::new(),
            gaussians,
        }
    }
    
    /// 이전 상태를 변경하지 않고 새 상태 반환
    pub fn with_camera(self, camera: CameraState) -> Self {
        AppState {
            camera,
            gaussians: self.gaussians,
        }
    }
}
```

### 4.4 Step 2: 순수 계산 함수 (src/compute/)

**계산(Pure Functions):**

```rust
// src/compute/ply_parse.rs - 순수 파싱 로직
use crate::data::gaussian::Gaussian;

/// 순수 함수: 바이너리 데이터 → Gaussian
/// stride는 SH degree=3 기준 248바이트 (62개 f32).
/// degree가 다르면 f_rest 크기가 달라지므로 stride도 변경 필요.
pub fn parse_gaussian_from_bytes(data: &[u8]) -> Result<Gaussian, String> {
    let required = std::mem::size_of::<Gaussian>();
    if data.len() < required {
        return Err(format!("insufficient data: got {} bytes, need {}", data.len(), required));
    }
    
    let read_f32 = |offset: usize| -> f32 {
        f32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ])
    };
    
    let mut f_rest = [0.0f32; 45];
    for j in 0..45 {
        f_rest[j] = read_f32(36 + j * 4);
    }
    
    Ok(Gaussian {
        x: read_f32(0),
        y: read_f32(4),
        z: read_f32(8),
        nx: read_f32(12),
        ny: read_f32(16),
        nz: read_f32(20),
        f_dc_0: read_f32(24),
        f_dc_1: read_f32(28),
        f_dc_2: read_f32(32),
        f_rest,
        opacity: read_f32(216),
        scale_0: read_f32(220),
        scale_1: read_f32(224),
        scale_2: read_f32(228),
        rot_0: read_f32(232),
        rot_1: read_f32(236),
        rot_2: read_f32(240),
        rot_3: read_f32(244),
    })
}

/// 순수 함수: 바이너리 집합 → Gaussian 벡터
pub fn parse_gaussians(data: &[u8], stride: usize, count: usize) -> Result<Vec<Gaussian>, String> {
    (0..count)
        .map(|i| {
            let offset = i * stride;
            parse_gaussian_from_bytes(&data[offset..offset + stride])
        })
        .collect()
}

// src/compute/camera_ops.rs - 순수 카메라 연산
use crate::data::camera::CameraState;
use glam::Vec3;
use std::f32::consts::PI;

/// 순수 함수: 입력 델타 + 현재 카메라 상태 → 새 카메라 상태
pub fn update_camera_angles(
    state: CameraState,
    delta_x: f32,
    delta_y: f32,
) -> CameraState {
    let new_theta = state.theta + delta_x * 0.01;
    let new_phi = (state.phi + delta_y * 0.01).clamp(0.1, PI - 0.1);
    
    CameraState {
        theta: new_theta,
        phi: new_phi,
        radius: state.radius,
    }
}

/// 순수 함수: 카메라 각도 → 월드 위치
pub fn compute_camera_position(state: CameraState) -> Vec3 {
    let x = state.radius * state.phi.sin() * state.theta.cos();
    let y = state.radius * state.phi.cos();
    let z = state.radius * state.phi.sin() * state.theta.sin();
    
    Vec3::new(x, y, z)
}

/// 순수 함수: 카메라 상태 → 뷰 행렬
pub fn camera_to_view_matrix(state: CameraState) -> glam::Mat4 {
    let position = compute_camera_position(state);
    let target = Vec3::ZERO;
    let up = Vec3::Y;
    
    glam::Mat4::look_at_rh(position, target, up)
}

/// 순수 함수: 줌
/// delta는 winit LineDelta 값 (보통 -3~+3 범위)
pub fn zoom_camera(state: CameraState, delta: f32) -> CameraState {
    let new_radius = (state.radius * (-delta * 0.1).exp()).clamp(0.5, 10.0);
    
    CameraState {
        radius: new_radius,
        ..state
    }
}

// src/compute/gaussian_ops.rs - 가우시안 연산
use crate::data::gaussian::Gaussian;
use glam::{Vec3, Quat};

/// 순수 함수: 쿼터니언 → 회전 행렬
pub fn quat_to_mat3(q: &[f32; 4]) -> glam::Mat3 {
    let quat = Quat::from_array(*q);
    glam::Mat3::from_quat(quat)
}

/// 순수 함수: 가우시안 공분산 행렬 계산
pub fn compute_covariance(gaussian: &Gaussian) -> glam::Mat3 {
    let scale = Vec3::new(gaussian.scale_0, gaussian.scale_1, gaussian.scale_2);
    let rot = quat_to_mat3(&gaussian.rotation());
    
    let scale_mat = glam::Mat3::from_diagonal(scale);
    rot * scale_mat * rot.transpose()
}

/// 순수 함수: 깊이 기준으로 가우시안 정렬
pub fn sort_gaussians_by_depth(
    gaussians: &[Gaussian],
    camera_pos: Vec3,
) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..gaussians.len()).collect();
    
    indices.sort_by(|&a, &b| {
        let dist_a = (gaussians[a].position() - camera_pos).length_squared();
        let dist_b = (gaussians[b].position() - camera_pos).length_squared();
        dist_b.partial_cmp(&dist_a).unwrap()
    });
    
    indices
}

/// 순수 함수: LOD 기반 가우시안 필터링
pub fn filter_gaussians_by_lod(
    gaussians: &[Gaussian],
    camera_pos: Vec3,
    lod_level: u32,
) -> Vec<usize> {
    let skip_rate = 1 << lod_level;
    
    gaussians
        .iter()
        .enumerate()
        .filter_map(|(i, g)| {
            if i % skip_rate as usize == 0 {
                let dist = (g.position() - camera_pos).length();
                if dist < 50.0 {
                    return Some(i);
                }
            }
            None
        })
        .collect()
}
```

### 4.5 Step 3: 액션 함수 (src/action/)

**액션(Side Effects) - 외부 세계와 상호작용:**

```rust
// src/action/io.rs - 파일 I/O 액션
use crate::data::gaussian::Gaussian;
use crate::compute::ply_parse;
use std::fs::File;
use std::io::Read;

/// 액션: 파일 읽기 (부작용: 디스크 접근)
pub fn load_ply_file(path: &str) -> anyhow::Result<Vec<Gaussian>> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    
    // 헤더 끝(end_header\n) 위치를 찾아 바이너리 데이터 시작점 결정
    let (header_str, data_start) = parse_ply_header(&buffer)?;
    let vertex_count = extract_vertex_count(&header_str)?;
    
    // 3DGS SH degree=3 기준 stride: 3(pos) + 3(normal) + 3(f_dc) + 45(f_rest)
    // + 1(opacity) + 3(scale) + 4(rot) = 62개 f32 = 248바이트
    // 단, 실제 파일의 property 목록에 따라 달라질 수 있음
    let stride = std::mem::size_of::<Gaussian>();
    
    ply_parse::parse_gaussians(&buffer[data_start..], stride, vertex_count)
        .map_err(|e| anyhow::anyhow!(e))
}

/// 순수 함수: PLY 헤더 파싱 → (헤더 문자열, 바이너리 데이터 시작 오프셋)
fn parse_ply_header(buffer: &[u8]) -> anyhow::Result<(String, usize)> {
    const END_HEADER: &[u8] = b"end_header\n";
    
    let end_pos = buffer
        .windows(END_HEADER.len())
        .position(|w| w == END_HEADER)
        .ok_or_else(|| anyhow::anyhow!("end_header not found in PLY file"))?;
    
    let header_str = std::str::from_utf8(&buffer[..end_pos])
        .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in PLY header: {}", e))?
        .to_string();
    
    Ok((header_str, end_pos + END_HEADER.len()))
}

/// 순수 함수: 헤더 문자열 → vertex 개수
fn extract_vertex_count(header: &str) -> anyhow::Result<usize> {
    header
        .lines()
        .find_map(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 3 && parts[0] == "element" && parts[1] == "vertex" {
                parts[2].parse().ok()
            } else {
                None
            }
        })
        .ok_or_else(|| anyhow::anyhow!("element vertex count not found in PLY header"))
}

// src/action/gpu.rs - GPU 상호작용 액션
use wgpu::*;
use crate::data::gaussian::Gaussian;

/// 액션: GPU 버퍼 생성 (부작용: GPU 메모리 할당)
pub fn create_gaussian_buffer(
    device: &Device,
    gaussians: &[Gaussian],
) -> Buffer {
    device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Gaussian Buffer"),
        contents: bytemuck::cast_slice(gaussians),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    })
}

/// 액션: 카메라 유니폼 버퍼 생성 (부작용: GPU 메모리 할당)
pub fn create_camera_buffer(
    device: &Device,
    view_matrix: glam::Mat4,
    proj_matrix: glam::Mat4,
) -> Buffer {
    #[repr(C)]
    #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
    struct CameraUniform {
        view: [f32; 16],
        projection: [f32; 16],
    }
    
    let camera = CameraUniform {
        view: view_matrix.to_cols_array(),
        projection: proj_matrix.to_cols_array(),
    };
    
    device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Camera Buffer"),
        contents: bytemuck::cast_slice(&[camera]),
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    })
}

/// 액션: 셰이더 모듈 생성 (부작용: GPU 컴파일)
pub fn create_shader_module(
    device: &Device,
    source: &str,
) -> ShaderModule {
    device.create_shader_module(ShaderModuleDescriptor {
        label: Some("Shader"),
        source: ShaderSource::Wgsl(source.into()),
    })
}

// src/action/render.rs - 렌더링 액션
use wgpu::*;

/// 액션: 렌더 패스 실행 (부작용: GPU 렌더링)
pub fn render_frame(
    device: &Device,
    queue: &Queue,
    view: &TextureView,
    pipeline: &RenderPipeline,
    bind_group: &BindGroup,
    gaussian_count: u32,
) {
    let mut encoder = device.create_command_encoder(&Default::default());
    
    {
        let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        
        rpass.set_pipeline(pipeline);
        rpass.set_bind_group(0, bind_group, &[]);
        // 가우시안 하나당 빌보드 쿼드 6 vertex
        rpass.draw(0..gaussian_count * 6, 0..1);
    }
    
    queue.submit(std::iter::once(encoder.finish()));
}

/// 액션: 버퍼 업데이트 (부작용: GPU 메모리 쓰기)
pub fn update_buffer<T: bytemuck::Pod>(
    queue: &Queue,
    buffer: &Buffer,
    data: &[T],
) {
    queue.write_buffer(buffer, 0, bytemuck::cast_slice(data));
}
```

### 4.6 Step 4: 메인 루프 (src/main.rs)

**액션을 조율하는 메인:**

```rust
// src/main.rs
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
use wgpu::*;

mod data;
mod compute;
mod action;

use data::*;
use compute::*;
use action::*;

struct AppContext {
    state: AppState,
    device: Device,
    queue: Queue,
    surface: Surface<'static>,
    config: SurfaceConfiguration,
    pipeline: RenderPipeline,
    bind_group: BindGroup,
    gaussian_buffer: Buffer,
    camera_buffer: Buffer,
}

impl AppContext {
    async fn new(window: Arc<winit::window::Window>, gaussians: Vec<Gaussian>) -> Self {
        // 1. GPU 초기화 (액션)
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });
        
        let surface = instance
            .create_surface(window)
            .expect("Failed to create surface");
        
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback: false,
            })
            .await
            .expect("Failed to request adapter");
        
        let (device, queue) = adapter
            .request_device(&DeviceDescriptor::default(), None)
            .await
            .expect("Failed to request device");
        
        let config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_capabilities(&adapter).formats[0],
            width: window.inner_size().width,
            height: window.inner_size().height,
            present_mode: PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: CompositeAlphaMode::Auto,
        };
        
        surface.configure(&device, &config);
        
        // 2. 초기 상태 생성
        let state = AppState::new(gaussians);
        
        // 3. 카메라 행렬 계산 (순수 계산)
        let view_matrix = camera_ops::camera_to_view_matrix(state.camera);
        let proj_matrix = glam::Mat4::perspective_rh(
            1.0,
            config.width as f32 / config.height as f32,
            0.1,
            100.0,
        );
        
        // 4. GPU 리소스 생성 (액션)
        let gaussian_buffer = gpu::create_gaussian_buffer(&device, &state.gaussians);
        let camera_buffer = gpu::create_camera_buffer(&device, view_matrix, proj_matrix);
        let shader = gpu::create_shader_module(&device, include_str!("../shaders/render.wgsl"));
        
        // ... 파이프라인 설정
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Main"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Main"),
            layout: &bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: gaussian_buffer.as_entire_binding(),
                },
            ],
        });
        
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Render"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList, // 빌보드 쿼드용
                ..Default::default()
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(ColorTargetState {
                    format: config.format,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            multiview: None,
        });
        
        Self {
            state,
            device,
            queue,
            surface,
            config,
            pipeline,
            bind_group,
            gaussian_buffer,
            camera_buffer,
        }
    }
    
    /// 순수 상태 변환 + 액션 조율
    fn update(&mut self, input: InputEvent) {
        // 1. 순수 계산으로 새 상태 생성
        let new_camera = match input {
            InputEvent::MouseMove(dx, dy) => {
                camera_ops::update_camera_angles(self.state.camera, dx, dy)
            }
            InputEvent::Zoom(delta) => {
                camera_ops::zoom_camera(self.state.camera, delta)
            }
            InputEvent::None => self.state.camera,
        };
        
        // 2. 상태 업데이트
        self.state = self.state.clone().with_camera(new_camera);
        
        // 3. 새로운 행렬 계산 (순수 계산)
        let view_matrix = camera_ops::camera_to_view_matrix(self.state.camera);
        let proj_matrix = glam::Mat4::perspective_rh(
            1.0,
            self.config.width as f32 / self.config.height as f32,
            0.1,
            100.0,
        );
        
        // 4. GPU 버퍼 업데이트 (액션)
        #[repr(C)]
        #[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
        struct CameraUniform {
            view: [f32; 16],
            projection: [f32; 16],
        }
        
        let camera_data = CameraUniform {
            view: view_matrix.to_cols_array(),
            projection: proj_matrix.to_cols_array(),
        };
        
        gpu::update_buffer(&self.queue, &self.camera_buffer, &[camera_data]);
    }
    
    fn render(&self) {
        // 액션: 렌더링
        let output = self.surface
            .get_current_texture()
            .expect("Failed to get texture");
        
        let view = output.texture.create_view(&TextureViewDescriptor::default());
        
        render::render_frame(
            &self.device,
            &self.queue,
            &view,
            &self.pipeline,
            &self.bind_group,
            self.state.gaussians.len() as u32,
        );
        
        output.present();
    }
}

enum InputEvent {
    MouseMove(f32, f32),
    Zoom(f32),
    None,
}

fn main() {
    env_logger::init();
    
    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("3DGS Viewer")
            .with_inner_size(winit::dpi::LogicalSize::new(1024.0, 768.0))
            .build(&event_loop)
            .unwrap()
    );
    
    // 1. 액션: 파일 로드 (동기)
    let gaussians = io::load_ply_file("assets/example.ply")
        .expect("Failed to load model");
    
    println!("Loaded {} gaussians", gaussians.len());
    
    // 2. 액션: GPU 초기화 (pollster로 async 블로킹 실행)
    // tokio::main 대신 pollster를 사용하는 이유:
    // winit 이벤트 루프가 메인 스레드를 점유하므로 tokio 런타임과 충돌함
    let mut app = pollster::block_on(AppContext::new(Arc::clone(&window), gaussians));
    
    // 3. 메인 루프
    let _ = event_loop.run(move |event, elwt| {
        match event {
            Event::AboutToWait => {
                window.request_redraw();
            }
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::RedrawRequested => {
                    // 액션: 렌더링
                    app.render();
                }
                WindowEvent::CloseRequested => {
                    elwt.exit();
                }
                WindowEvent::MouseInput { state, .. } => {
                    // TODO: 마우스 입력 처리
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    match delta {
                        winit::event::MouseScrollDelta::LineDelta(_, y) => {
                            app.update(InputEvent::Zoom(y));
                        }
                        _ => {}
                    }
                }
                _ => {}
            },
            _ => {}
        }
        elwt.set_control_flow(ControlFlow::Poll);
    });
}
```

### 4.7 Step 5: 셰이더 (src/shaders/render.wgsl)

WGSL은 WebGPU/wgpu의 표준 셰이더 언어로, Metal/Vulkan/DX12로 자동 변환됩니다:

```wgsl
WGSL에서 `array<f32, 45>` 같은 배열 멤버는 **16바이트 정렬**을 요구합니다. Rust의 `#[repr(C)]` 구조체와 레이아웃이 달라질 수 있으므로, GPU 업로드용 구조체를 별도로 정의하고 정렬 패딩을 명시합니다.

**Rust 쪽 GPU 전용 구조체 (src/action/gpu.rs):**

```rust
/// GPU 업로드 전용 구조체. WGSL struct와 레이아웃이 1:1로 일치해야 함.
/// f_rest 이전에 3바이트 패딩을 추가해 16바이트 경계를 맞춤.
#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Clone, Copy)]
pub struct GaussianGpu {
    pub pos: [f32; 3],
    pub opacity: f32,         // 16바이트 정렬
    pub color_dc: [f32; 3],
    pub _pad0: f32,           // 16바이트 정렬 패딩
    pub scale: [f32; 3],
    pub _pad1: f32,
    pub rot: [f32; 4],        // 이미 16바이트
    pub f_rest: [f32; 45],
    pub _pad2: [f32; 3],      // 45 -> 48 (16바이트 단위)
}

impl From<&Gaussian> for GaussianGpu {
    fn from(g: &Gaussian) -> Self {
        let mut f_rest = [0.0f32; 45];
        f_rest.copy_from_slice(&g.f_rest);
        Self {
            pos: [g.x, g.y, g.z],
            opacity: g.opacity,
            color_dc: [g.f_dc_0, g.f_dc_1, g.f_dc_2],
            _pad0: 0.0,
            scale: [g.scale_0, g.scale_1, g.scale_2],
            _pad1: 0.0,
            rot: [g.rot_0, g.rot_1, g.rot_2, g.rot_3],
            f_rest,
            _pad2: [0.0; 3],
        }
    }
}
```

**셰이더 (src/shaders/render.wgsl):**

```wgsl
// src/shaders/render.wgsl
struct CameraUniform {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
}

// GaussianGpu와 레이아웃 일치: 각 필드는 16바이트 경계에 배치
struct Gaussian {
    pos: vec3<f32>,
    opacity: f32,
    color_dc: vec3<f32>,
    _pad0: f32,
    scale: vec3<f32>,
    _pad1: f32,
    rot: vec4<f32>,
    f_rest: array<vec4<f32>, 12>,  // 45 f32 → 48 f32 (12 vec4)로 패딩 포함
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian>;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) alpha: f32,
    @location(2) uv: vec2<f32>,  // 가우시안 로컬 UV (-1~1)
}

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

// 하나의 가우시안을 빌보드 쿼드(6 vertex)로 확장
// draw call: draw(0..gaussian_count * 6, 0..1)
@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    let gaussian_idx = idx / 6u;
    let corner_idx = idx % 6u;
    
    // 쿼드의 6개 꼭짓점 UV (-1~1 범위)
    let corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0),
    );
    let uv = corners[corner_idx];
    
    let g = gaussians[gaussian_idx];
    
    let pos_view = camera.view * vec4<f32>(g.pos, 1.0);
    // scale을 화면 공간 오프셋으로 사용 (간략화된 2D 투영)
    let screen_offset = vec4<f32>(uv * max(g.scale.x, g.scale.y) * 0.5, 0.0, 0.0);
    let pos_proj = camera.projection * (pos_view + screen_offset);
    
    var output: VertexOutput;
    output.clip_pos = pos_proj;
    output.color = g.color_dc;
    output.alpha = sigmoid(g.opacity);
    output.uv = uv;
    
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // 로컬 UV 기준으로 가우시안 감쇠 계산
    let dist2 = dot(input.uv, input.uv);
    if dist2 > 1.0 {
        discard;  // 원 밖은 버림
    }
    let weight = exp(-0.5 * dist2 * 4.0);
    
    return vec4<f32>(input.color, input.alpha * weight);
}
```

---

## 5. 함수형 설계의 장점

### 5.1 테스트 가능성

순수 함수는 입력만 주면 항상 동일한 결과를 반환하므로 테스트가 간단합니다:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_camera_zoom() {
        let state = CameraState::new();
        let zoomed = camera_ops::zoom_camera(state, 1.0);
        
        assert!(zoomed.radius > state.radius);
        assert_eq!(zoomed.theta, state.theta); // 다른 속성은 변하지 않음
    }
    
    #[test]
    fn test_gaussian_sorting() {
        let gaussians = vec![
            Gaussian { x: 0.0, y: 0.0, z: 10.0, .. },
            Gaussian { x: 0.0, y: 0.0, z: 5.0, .. },
        ];
        let camera_pos = Vec3::ZERO;
        
        let sorted_indices = gaussian_ops::sort_gaussians_by_depth(&gaussians, camera_pos);
        
        // 더 가까운 가우시안이 먼저 와야 함
        assert_eq!(sorted_indices[0], 1);
    }
}
```

### 5.2 상태 관리의 명확성

```rust
// ❌ 나쁜 예: 상태가 암묵적으로 변경됨
fn update_and_render(app: &mut App) {
    app.camera.on_mouse_move(1.0, 2.0);  // app이 변경됨
    // 어느 부분이 변경되었는지 모름
    app.render();
}

// ✅ 좋은 예: 명시적 상태 흐름
fn update_and_render(mut app: AppState) -> AppState {
    // 1. 순수 계산: 새로운 상태 생성
    let new_camera = camera_ops::update_camera_angles(app.camera, 1.0, 2.0);
    
    // 2. 상태 업데이트: 명시적
    let new_app = app.with_camera(new_camera);
    
    // 3. 렌더링 (액션)
    // render::render_frame(...)
    
    new_app
}
```

### 5.3 버그 감소

- **불변성**: 상태를 변경할 수 없으므로 원치 않은 사이드 이펙트 불가능
- **순수성**: 함수 간 숨겨진 의존성 없음
- **구성성**: 작은 함수들을 조합해 복잡한 로직 구성

---

## 6. 주요 최적화 기법 (순수 계산)

### 6.1 메모리 효율화

3DGS의 가장 큰 문제는 메모리입니다. 100만 개 가우시안 = 약 150-200MB.

**기법 1: 적응형 LOD (Level of Detail)**

```rust
// src/compute/lod.rs
/// 순수 함수: 거리 → LOD 레벨
pub fn compute_lod_level(distance: f32, base_distance: f32) -> u32 {
    let level = (distance / base_distance).log2().ceil();
    level.max(0.0) as u32
}

/// 순수 함수: 가우시안 필터링 (이미 구현됨)
pub fn filter_gaussians_by_lod(
    gaussians: &[Gaussian],
    camera_pos: Vec3,
    lod_level: u32,
) -> Vec<usize> {
    // src/compute/gaussian_ops.rs 참조
}

/// 액션: 필터링된 가우시안만 GPU에 업로드
pub fn upload_lod_gaussians(
    queue: &Queue,
    buffer: &Buffer,
    gaussians: &[Gaussian],
    indices: &[usize],
) {
    let filtered: Vec<_> = indices.iter().map(|&i| gaussians[i]).collect();
    gpu::update_buffer(queue, buffer, &filtered);
}
```

**기법 2: 깊이 정렬 (Compute Shader)**

```wgsl
// src/shaders/sort.wgsl
@group(0) @binding(0) var<storage, read_write> indices: array<u32>;
@group(0) @binding(1) var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(2) var<uniform> camera_pos: vec3<f32>;

@compute @workgroup_size(256, 1, 1)
fn sort_step(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let idx = global_id.x;
    let size = arrayLength(&indices);
    
    if (idx >= size / 2u) {
        return;
    }
    
    // Bitonic sort step
    let g_idx_a = indices[idx * 2u];
    let g_idx_b = indices[idx * 2u + 1u];
    
    let dist_a = distance(vec3<f32>(gaussians[g_idx_a].x, gaussians[g_idx_a].y, gaussians[g_idx_a].z), camera_pos);
    let dist_b = distance(vec3<f32>(gaussians[g_idx_b].x, gaussians[g_idx_b].y, gaussians[g_idx_b].z), camera_pos);
    
    if (dist_a < dist_b) {
        // Swap
        indices[idx * 2u] = g_idx_b;
        indices[idx * 2u + 1u] = g_idx_a;
    }
}
```

---

## 7. 성능 최적화 (플랫폼 독립적)

### 7.1 배치 처리를 통한 자동 벡터화

순수 함수로 가우시안 연산을 배치 처리하면, Rust 컴파일러가 자동으로 SIMD 명령어를 생성합니다:

```rust
// src/compute/gaussian_ops.rs
/// 순수 함수: 배치 카메라 위치 변환
pub fn transform_gaussians_batch(
    gaussians: &[Gaussian],
    view_matrix: glam::Mat4,
) -> Vec<Vec3> {
    // 컴파일러가 SIMD로 자동 벡터화
    gaussians
        .iter()
        .map(|g| {
            let pos = Vec3::new(g.x, g.y, g.z);
            (view_matrix * pos.extend(1.0)).xyz()
        })
        .collect()
}

/// 순수 함수: 배치 거리 계산
pub fn compute_distances_batch(
    gaussians: &[Gaussian],
    camera_pos: Vec3,
) -> Vec<f32> {
    gaussians
        .iter()
        .map(|g| (g.position() - camera_pos).length_squared())
        .collect()
}
```

### 7.2 병렬 처리 (Rayon)

CPU 멀티코어를 활용한 병렬 정렬:

```rust
// Cargo.toml에 추가
[dependencies]
rayon = "1.8"

// src/compute/gaussian_ops.rs
use rayon::prelude::*;

/// 병렬 거리 정렬
pub fn sort_gaussians_by_depth_parallel(
    gaussians: &[Gaussian],
    camera_pos: Vec3,
) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..gaussians.len()).collect();
    
    indices.par_sort_by(|&a, &b| {
        let dist_a = (gaussians[a].position() - camera_pos).length_squared();
        let dist_b = (gaussians[b].position() - camera_pos).length_squared();
        dist_b.partial_cmp(&dist_a).unwrap()
    });
    
    indices
}
```

---

## 8. 개발 환경 설정

### 8.1 프로젝트 초기화

```bash
cargo new 3dgs-viewer
cd 3dgs-viewer
mkdir -p src/{data,compute,action}
mkdir -p shaders
mkdir -p assets
```

Cargo.toml은 **섹션 2.1**을 참고합니다.

### 8.2 빌드 및 테스트

```bash
# 개발 빌드
cargo build

# Release 빌드 (최적화)
cargo build --release

# 테스트 실행
cargo test

# 성능 벤치마크
cargo bench
```

### 8.3 모듈 선언 (src/lib.rs)

```rust
pub mod data {
    pub mod gaussian;
    pub mod camera;
    pub mod app_state;
}

pub mod compute {
    pub mod ply_parse;
    pub mod camera_ops;
    pub mod gaussian_ops;
}

pub mod action {
    pub mod io;
    pub mod gpu;
    pub mod render;
}
```

---

## 9. Rust 기반의 장점

### 9.1 메모리 안전성 (소유권 시스템)

```rust
// Rust의 소유권 시스템이 자동으로 메모리 누수 방지
pub fn create_gaussian_buffer(gaussians: &[Gaussian]) -> Buffer {
    let buffer = device.create_buffer_init(&BufferInitDescriptor {
        contents: bytemuck::cast_slice(gaussians),
        usage: BufferUsages::STORAGE,
        label: Some("Gaussian Buffer"),
    });
    // buffer는 스코프 벗어날 때 자동으로 해제됨
    buffer
}
```

### 9.2 성능 예상

| 작업 | 성능 | 비고 |
| :--- | :--- | :--- |
| **100만 가우시안 로드** | <100ms | 바이너리 파싱 포함 |
| **프레임 렌더링 (1920x1080)** | 16ms (60fps) | 가우시안 정렬 포함 |
| **메모리 사용** | 150-200MB | GPU 메모리 |
| **바이너리 크기** | ~3MB | Release 빌드 |

---

## 10. 함수형 프로그래밍 원칙 정리

### 10.1 핵심 원칙 (에릭 노먼드)

이 프로젝트의 구조는 세 가지 기본 원칙을 따릅니다:

| 카테고리 | 특징 | 예시 |
| :--- | :--- | :--- |
| **데이터** | 불변, 순수 값 | `Gaussian`, `CameraState`, `AppState` |
| **계산** | 입력→출력, 부작용 없음 | `camera_ops::update_camera_angles()` |
| **액션** | 부작용, 외부 세계 상호작용 | `io::load_ply_file()`, `render::render_frame()` |

### 10.2 코드 조직화 전략

```
src/
├── data/          # 순수 데이터 구조
├── compute/       # 순수 계산 함수 (테스트 가능)
├── action/        # 액션 함수 (I/O, GPU, 렌더링)
└── main.rs        # 액션 조율 (가장 높은 수준에서 부작용 관리)
```

**이점:**
- **테스트**: `data/`와 `compute/`는 100% 테스트 가능
- **재사용성**: 계산 함수를 다양한 맥락에서 사용 가능
- **버그 감소**: 액션과 계산이 분리되어 부작용 추적 용이
- **병렬화**: 순수 함수들은 안전하게 병렬 실행 가능
- **플랫폼 독립성**: 데이터와 계산 로직이 GPU API에 종속되지 않음

### 10.3 실제 적용 예제

```rust
// ❌ 절차형 (테스트 불가, 부작용 섞임)
fn process_input(mut app: &mut App, input: Input) {
    app.camera.theta += input.delta_x * 0.01;  // 암묵적 변경
    let view = app.camera.get_view_matrix();   // 부작용에 의존
    app.render();                               // 렌더링 강제
}

// ✅ 함수형 (테스트 가능, 명시적)
fn process_input(app: AppState, input: InputDelta) -> AppState {
    // 1. 순수 계산
    let new_camera = camera_ops::update_camera_angles(app.camera, input.delta_x, input.delta_y);
    let new_app = app.with_camera(new_camera);
    
    // 2. 반환된 상태는 호출자가 사용 결정
    new_app
}

// 호출자가 액션 선택
let new_app = process_input(app, input);
if should_render {
    app_context.render();  // 액션은 app_context에서만 발생
}
```

### 10.4 다음 단계 및 확장

**즉시 구현 가능:**

1. **Spherical Harmonics 렌더링** (순수 계산)
   ```rust
   // src/compute/sh.rs
   pub fn evaluate_sh_color(coeffs: &[f32; 45], direction: Vec3) -> Vec3 {
       // SH 기저 함수로 시점 의존적 색상 계산
       let dc = Vec3::new(coeffs[0], coeffs[1], coeffs[2]);
       // ... 나머지 SH 계수 추가
       dc
   }
   
   #[test]
   fn test_sh_evaluation() {
       // 완전히 테스트 가능한 순수 함수
   }
   ```

2. **깊이 정렬 Compute Shader** (액션)
   - 이미 `src/shaders/sort.wgsl`에 구조 제공

3. **UI 개발** (egui 라이브러리)
   ```rust
   // UI도 상태 → 새 상태 패턴 따름
   fn render_ui(state: &AppState) -> UIState { ... }
   ```

**중기 계획:**
- **4D Gaussian Splatting**: 시간축 포함 동적 3DGS
- **실시간 편집**: 가우시안 선택, 이동 (순수 계산)
- **다중 뷰 지원**: 360도 애니메이션

**장기 계획:**
- **웹 내보내기** (WASM으로 WebGL 변환)
- **학습 파이프라인**: 이미지로부터 3DGS 생성
- **자율주행 시뮬레이션**: UniSim, MARS 연동

---

## 11. 문제 해결

### Q: "Metal 벡터 인코딩 오류"

**원인:** wgpu 버전과 OS 버전 불일치

```bash
cargo update wgpu
# macOS 12+ 필요
```

### Q: "가우시안이 보이지 않음"

**함수형 디버깅 전략:**

```rust
// 1. 순수 계산 검증 (테스트로 확인)
#[test]
fn test_camera_position_calculation() {
    let state = CameraState::new();
    let pos = camera_ops::compute_camera_position(state);
    
    // 카메라가 원점에서 반지름 2만큼 떨어져 있어야 함
    assert!((pos.length() - 2.0).abs() < 0.01);
}

// 2. 가우시안 데이터 검증 (pure 함수)
pub fn validate_gaussian_data(g: &Gaussian) -> Result<(), String> {
    if g.position().length() > 100.0 {
        return Err("Gaussian too far".to_string());
    }
    if g.opacity < -20.0 || g.opacity > 20.0 {
        return Err("Opacity out of range".to_string());
    }
    Ok(())
}

// 3. 렌더링 파이프라인 검증 (액션)
pub fn validate_render_pipeline(
    device: &Device,
    pipeline: &RenderPipeline,
) -> bool {
    // GPU 파이프라인 상태 확인
    true
}
```

### Q: "성능이 너무 느림"

**함수형 최적화 프로세스:**

```rust
// 1. 순수 함수의 성능 측정 (벤치마크)
#[bench]
fn bench_sort_gaussians(b: &mut Bencher) {
    let gaussians = create_test_gaussians(1_000_000);
    let camera_pos = Vec3::ZERO;
    
    b.iter(|| {
        gaussian_ops::sort_gaussians_by_depth(&gaussians, camera_pos)
    });
}

// 2. LOD 필터링 추가 (계산 단계 최적화)
pub fn filtered_render(
    state: &AppState,
    camera_pos: Vec3,
) -> (Vec<usize>, AppState) {
    let lod_level = compute_lod_level(
        (compute_camera_position(state.camera) - camera_pos).length(),
        1.0
    );
    
    let filtered = gaussian_ops::filter_gaussians_by_lod(
        &state.gaussians,
        camera_pos,
        lod_level
    );
    
    (filtered, state.clone())
}

// 3. GPU 구현으로 마이그레이션 (액션 단계 최적화)
pub fn render_with_compute_sort(
    device: &Device,
    queue: &Queue,
    // ... 기존 파라미터
) {
    // Compute shader 기반 정렬 사용
    render::render_frame_with_sort(device, queue, /* ... */)
}
```

### Q: "상태 관리가 복잡해짐"

**함수형 해결책:**

```rust
// 1. 상태 렌즈(Lenses) 패턴으로 중첩 상태 관리
pub struct AppLens<T> {
    getter: fn(&AppState) -> T,
    setter: fn(AppState, T) -> AppState,
}

impl AppLens<CameraState> {
    pub fn camera() -> Self {
        AppLens {
            getter: |app| app.camera,
            setter: |app, cam| app.with_camera(cam),
        }
    }
}

// 2. 조합 함수로 다중 상태 변환
fn apply_multiple_inputs(
    mut state: AppState,
    inputs: &[InputDelta],
) -> AppState {
    for input in inputs {
        state = match input {
            InputDelta::MouseMove(dx, dy) => {
                let new_cam = camera_ops::update_camera_angles(state.camera, *dx, *dy);
                state.with_camera(new_cam)
            }
            InputDelta::Zoom(delta) => {
                let new_cam = camera_ops::zoom_camera(state.camera, *delta);
                state.with_camera(new_cam)
            }
        };
    }
    state
}
```

---

## 참고자료

- **3DGS 공식 논문**: Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (SIGGRAPH 2023)
- **3DGS 공식 구현**: https://github.com/graphdeco-inria/gaussian-splatting
- **WGPU 문서**: https://docs.rs/wgpu/
- **PLY 포맷**: [PLY 포맷 총정리]({{< ref "ply-format-overview.md" >}})
- **3D 데이터 표현**: [3D 데이터 표현 방식 총정리]({{< ref "3d-data-representations.md" >}})
- **ARM SIMD 최적화**: https://developer.arm.com/documentation/101458/latest/
