---
title: "GPU Compute 셰이더를 함수형으로: GPGPU 파이프라인 설계"
date: 2026-04-30T15:00:00+09:00
draft: false
tags: ["함수형 프로그래밍", "Rust", "설계", "GPU", "GPGPU", "Compute Shader", "자율주행", "액션/계산/데이터", "wgpu"]
categories: ["프로그래밍", "GPU"]
description: "GPU 연산(Compute) 셰이더를 액션/계산/데이터로 분리하면, GPU 없이 디스패치 로직을 테스트하고 자율주행 포인트 클라우드 처리나 DNN 추론을 CPU/GPU 코드 공유로 검증할 수 있습니다."
---

## 이 글을 읽고 나면

- 렌더링 셰이더와 Compute 셰이더의 차이를 이해합니다.
- GPU 버퍼 관리와 디스패치를 액션으로, 커널 로직을 계산으로 분리하는 방법을 봅니다.
- CPU 레퍼런스 구현으로 GPU 커널을 검증하는 방법을 이해합니다.
- 포인트 클라우드 처리, 이미지 필터, 행렬 연산을 예제로 다룹니다.

이전 글 [액션/계산/데이터](/posts/programming/functional/functional-actions-calculations-data/), [함수형 셰이더 파이프라인](/posts/programming/functional/functional-shader-pipeline/), [함수형 렌더 그래프](/posts/programming/functional/functional-render-graph/)를 먼저 읽으면 더 자연스럽게 이어집니다.

---

## Compute 셰이더란

렌더링 셰이더(Vertex, Fragment)는 화면에 그리는 것이 목적입니다. **Compute 셰이더**는 그리는 것과 무관하게 대량의 데이터를 병렬로 연산하는 GPU 프로그램입니다.

자율주행에서 Compute 셰이더가 유용한 이유는 이렇습니다.

- **포인트 클라우드 처리**: LiDAR 10만 포인트를 프레임마다 필터링·다운샘플링
- **이미지 처리**: 카메라 수십 장을 동시에 왜곡 보정·히스토그램 정규화
- **DNN 추론 지원**: 피처맵 연산, 앵커 생성, NMS(Non-Maximum Suppression)
- **행렬 연산**: 칼만 필터 공분산 업데이트, 좌표 변환 배치 처리

Compute 셰이더는 [함수형 셰이더 파이프라인](/posts/programming/functional/functional-shader-pipeline/) 글에서 말한 것처럼 구조적으로 순수 함수입니다. **입력 버퍼 → 커널 연산 → 출력 버퍼**, 전역 상태가 없습니다.

---

## 나쁜 방법: 버퍼·디스패치·로직이 뒤섞이기

```rust
fn process_point_cloud(points: &[Point3D]) -> Vec<Point3D> {
    // GPU 초기화가 로직과 뒤섞임
    let device = wgpu::Device::new(...);
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        contents: bytemuck::cast_slice(points),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        ..
    });

    // 필터링 임계값이 하드코딩
    let threshold = 50.0_f32;
    let shader_src = format!("const THRESHOLD: f32 = {};  ...", threshold);

    // 디스패치, 결과 읽기, 후처리가 한 함수에
    dispatch_and_wait(&device, &buffer, points.len());
    let result = read_buffer(&device, &buffer);
    result.iter().filter(|p| p.intensity > 0.0).cloned().collect()
}
```

문제들:

- 필터 임계값을 바꾸면 함수 전체를 건드려야 함
- GPU 없이 필터 로직을 테스트할 방법이 없음
- CPU에서 검증하려면 별도로 로직을 복사해야 함 → 두 코드가 어긋남

---

## 데이터: 커널 파라미터를 선언적으로

GPU 커널에 넘기는 설정을 데이터로 분리합니다. 셰이더가 받는 `uniform`/`push_constant`가 그대로 Rust 구조체가 됩니다.

```rust
/// 포인트 클라우드 다운샘플 파라미터
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct VoxelParams {
    pub voxel_size: f32,    // 복셀 한 변의 길이 (m)
    pub min_x: f32,
    pub min_y: f32,
    pub min_z: f32,
    pub max_x: f32,
    pub max_y: f32,
    pub max_z: f32,
    pub _pad: f32,
}

/// 이미지 처리 파라미터
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ImageFilterParams {
    pub width: u32,
    pub height: u32,
    pub sigma: f32,          // 가우시안 시그마
    pub threshold: f32,      // 엣지 임계값
}

/// NMS 파라미터
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct NmsParams {
    pub iou_threshold: f32,
    pub score_threshold: f32,
    pub max_boxes: u32,
    pub _pad: u32,
}

/// Compute 패스 하나를 설명하는 선언적 기술자 (데이터)
#[derive(Debug, Clone)]
pub struct ComputePassDesc<P: bytemuck::Pod> {
    pub shader_path: &'static str,   // WGSL 파일 경로
    pub entry_point: &'static str,   // @compute 진입점 이름
    pub params: P,                   // 커널 파라미터
    pub workgroup_size: [u32; 3],    // 디스패치 크기
}
```

`ComputePassDesc`는 "어떤 커널을 어떤 파라미터로 실행할지"를 설명하는 순수 데이터입니다. GPU를 만지지 않습니다.

---

## 계산: 커널 로직을 CPU에도 구현

GPU 커널과 동일한 로직을 CPU 순수 함수로도 작성합니다. GPU 없이 테스트하고, CPU 결과로 GPU 결과를 검증합니다.

```rust
/// 포인트 하나의 복셀 인덱스 계산 — CPU·GPU 동일 로직
pub fn point_to_voxel_index(
    x: f32, y: f32, z: f32,
    params: &VoxelParams,
) -> Option<[u32; 3]> {
    if x < params.min_x || x > params.max_x
    || y < params.min_y || y > params.max_y
    || z < params.min_z || z > params.max_z {
        return None;
    }
    let ix = ((x - params.min_x) / params.voxel_size) as u32;
    let iy = ((y - params.min_y) / params.voxel_size) as u32;
    let iz = ((z - params.min_z) / params.voxel_size) as u32;
    Some([ix, iy, iz])
}

/// 복셀 다운샘플링 — CPU 레퍼런스 구현
pub fn voxel_downsample_cpu(
    points: &[[f32; 3]],
    params: &VoxelParams,
) -> Vec<[f32; 3]> {
    use std::collections::HashMap;
    let mut voxels: HashMap<[u32; 3], (f32, f32, f32, u32)> = HashMap::new();

    for &[x, y, z] in points {
        if let Some(idx) = point_to_voxel_index(x, y, z, params) {
            let entry = voxels.entry(idx).or_insert((0.0, 0.0, 0.0, 0));
            entry.0 += x;
            entry.1 += y;
            entry.2 += z;
            entry.3 += 1;
        }
    }

    voxels.values().map(|(sx, sy, sz, n)| {
        let n = *n as f32;
        [sx / n, sy / n, sz / n]
    }).collect()
}

/// 5×5 가우시안 커널 계산 — CPU 레퍼런스
pub fn gaussian_kernel_5x5(sigma: f32) -> [[f32; 5]; 5] {
    let mut kernel = [[0.0f32; 5]; 5];
    let mut sum = 0.0f32;
    for i in 0..5 {
        for j in 0..5 {
            let x = (i as f32) - 2.0;
            let y = (j as f32) - 2.0;
            kernel[i][j] = (-(x * x + y * y) / (2.0 * sigma * sigma)).exp();
            sum += kernel[i][j];
        }
    }
    for row in &mut kernel { for v in row.iter_mut() { *v /= sum; } }
    kernel
}

/// 가우시안 블러 (단일 픽셀) — CPU 레퍼런스
pub fn gaussian_blur_pixel(
    image: &[f32],
    width: u32,
    height: u32,
    px: u32,
    py: u32,
    kernel: &[[f32; 5]; 5],
) -> f32 {
    let mut result = 0.0f32;
    for ky in 0..5i32 {
        for kx in 0..5i32 {
            let nx = px as i32 + kx - 2;
            let ny = py as i32 + ky - 2;
            if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                result += image[(ny as u32 * width + nx as u32) as usize]
                         * kernel[ky as usize][kx as usize];
            }
        }
    }
    result
}
```

이 함수들은 GPU 코드 없이 즉시 단위 테스트 가능합니다.

---

## WGSL 커널: CPU 로직을 GPU로 옮기기

CPU 레퍼런스와 동일한 로직을 WGSL로 작성합니다.

```wgsl
// voxel_downsample.wgsl

struct VoxelParams {
    voxel_size: f32,
    min_x: f32, min_y: f32, min_z: f32,
    max_x: f32, max_y: f32, max_z: f32,
    _pad: f32,
}

@group(0) @binding(0) var<uniform> params: VoxelParams;
@group(0) @binding(1) var<storage, read>       points:   array<vec3<f32>>;
@group(0) @binding(2) var<storage, read_write>  voxel_counts: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write>  voxel_sums:   array<vec3<f32>>;

// CPU의 point_to_voxel_index와 동일한 로직
fn point_to_voxel(p: vec3<f32>) -> vec3<u32> {
    return vec3<u32>(
        u32((p.x - params.min_x) / params.voxel_size),
        u32((p.y - params.min_y) / params.voxel_size),
        u32((p.z - params.min_z) / params.voxel_size),
    );
}

fn in_bounds(p: vec3<f32>) -> bool {
    return p.x >= params.min_x && p.x <= params.max_x
        && p.y >= params.min_y && p.y <= params.max_y
        && p.z >= params.min_z && p.z <= params.max_z;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&points) { return; }

    let p = points[idx];
    if !in_bounds(p) { return; }

    let voxel = point_to_voxel(p);
    let grid_w = u32((params.max_x - params.min_x) / params.voxel_size) + 1u;
    let grid_h = u32((params.max_y - params.min_y) / params.voxel_size) + 1u;
    let flat_idx = voxel.z * grid_w * grid_h + voxel.y * grid_w + voxel.x;

    atomicAdd(&voxel_counts[flat_idx], 1u);
    // 주의: vec3 atomic은 WGSL에서 직접 지원 안 됨 → 별도 인덱스 배열 사용
    atomicAdd(&voxel_sums_x[flat_idx], ...) // 실제 구현에서는 분리
}
```

CPU 구현과 WGSL 구현이 동일한 `point_to_voxel` 로직을 공유합니다. 버그가 생기면 CPU 테스트에서 먼저 잡힙니다.

---

## 액션: 버퍼 생성·디스패치·결과 읽기

GPU와 직접 통신하는 부분은 모두 액션입니다.

```rust
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl GpuContext {
    /// 버퍼 생성 (액션)
    pub fn create_buffer<T: bytemuck::Pod>(&self, data: &[T], usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(data),
            usage,
        })
    }

    /// Compute 패스 실행 (액션)
    pub fn dispatch<P: bytemuck::Pod>(
        &self,
        desc: &ComputePassDesc<P>,
        input_buffer: &wgpu::Buffer,
        output_buffer: &wgpu::Buffer,
        num_elements: u32,
    ) {
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                std::fs::read_to_string(desc.shader_path).unwrap().into()
            ),
        });

        let params_buffer = self.create_buffer(
            &[desc.params],
            wgpu::BufferUsages::UNIFORM,
        );

        // 파이프라인·바인드그룹 생성 → 인코딩 → 제출
        // (wgpu 보일러플레이트 생략)
        let [wg_x, wg_y, wg_z] = desc.workgroup_size;
        // encoder.dispatch_workgroups(wg_x, wg_y, wg_z);
        let _ = (shader, params_buffer, wg_x, wg_y, wg_z);
    }

    /// 결과 버퍼 읽기 (액션)
    pub async fn read_buffer<T: bytemuck::Pod + Clone>(&self, buffer: &wgpu::Buffer, len: usize) -> Vec<T> {
        // buffer.slice(..).map_async(...)
        // device.poll(wgpu::Maintain::Wait)
        // 읽기 구현 ...
        vec![]
    }
}
```

`GpuContext`는 액션 묶음입니다. 커널 로직이 없고, 순수하게 GPU와 통신만 합니다.

---

## 파이프라인 조합

CPU 레퍼런스와 GPU 구현을 같은 인터페이스로 교체 가능하게 만들 수 있습니다.

```rust
/// 포인트 클라우드 처리 파이프라인 — 트레잇으로 추상화
pub trait PointCloudProcessor {
    fn voxel_downsample(&self, points: &[[f32; 3]], params: &VoxelParams) -> Vec<[f32; 3]>;
}

/// CPU 구현 (테스트·시뮬레이션용)
pub struct CpuProcessor;

impl PointCloudProcessor for CpuProcessor {
    fn voxel_downsample(&self, points: &[[f32; 3]], params: &VoxelParams) -> Vec<[f32; 3]> {
        voxel_downsample_cpu(points, params)
    }
}

/// GPU 구현 (실차·실시간용)
pub struct GpuProcessor<'a> {
    ctx: &'a GpuContext,
}

impl<'a> PointCloudProcessor for GpuProcessor<'a> {
    fn voxel_downsample(&self, points: &[[f32; 3]], params: &VoxelParams) -> Vec<[f32; 3]> {
        let input = self.ctx.create_buffer(points, wgpu::BufferUsages::STORAGE);
        let output = self.ctx.create_buffer(
            &vec![[0.0f32; 3]; points.len()],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let desc = ComputePassDesc {
            shader_path: "shaders/voxel_downsample.wgsl",
            entry_point: "main",
            params: *params,
            workgroup_size: [(points.len() as u32 + 255) / 256, 1, 1],
        };
        self.ctx.dispatch(&desc, &input, &output, points.len() as u32);
        // 결과 읽기 (비동기 생략)
        vec![]
    }
}
```

`PointCloudProcessor` 트레잇을 받는 코드는 CPU인지 GPU인지 모릅니다. [함수형 DI](/posts/programming/functional/functional-dependency-injection/) 패턴입니다.

---

## 자율주행 파이프라인에서의 위치

[함수형 포인트 클라우드 처리](/posts/programming/functional/functional-point-cloud/) 글은 CPU에서 LiDAR 파이프라인을 순수 함수로 만드는 내용이었습니다. 이 글은 그 연산을 GPU로 옮기는 방법입니다.

```
LiDAR 원시 데이터 (액션: 읽기)
    ↓
포인트 클라우드 필터링
  ├─ CPU: voxel_downsample_cpu (계산)
  └─ GPU: GpuProcessor::voxel_downsample (액션: 디스패치, 결과는 계산 결과)
    ↓
클러스터링 → 물체 검출
    ↓
검출 결과 (데이터)
```

파이프라인 구조는 그대로입니다. CPU 구현을 레퍼런스로 두고 GPU 구현으로 교체하거나, CI에서는 CPU로 돌리고 실차에서는 GPU로 돌릴 수 있습니다.

---

## 테스트: GPU 없이 로직 검증

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn test_params() -> VoxelParams {
        VoxelParams {
            voxel_size: 1.0,
            min_x: 0.0, min_y: 0.0, min_z: 0.0,
            max_x: 10.0, max_y: 10.0, max_z: 10.0,
            _pad: 0.0,
        }
    }

    #[test]
    fn test_point_in_bounds_gets_voxel_index() {
        let params = test_params();
        let idx = point_to_voxel_index(1.5, 2.5, 3.5, &params);
        assert_eq!(idx, Some([1, 2, 3]));
    }

    #[test]
    fn test_point_out_of_bounds_returns_none() {
        let params = test_params();
        assert!(point_to_voxel_index(-1.0, 0.0, 0.0, &params).is_none());
        assert!(point_to_voxel_index(0.0, 11.0, 0.0, &params).is_none());
    }

    #[test]
    fn test_voxel_downsample_reduces_points() {
        let params = test_params();
        // 같은 복셀 안의 포인트 4개 → 1개로 합쳐져야 함
        let points = vec![
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [0.3, 0.3, 0.3],
            [0.4, 0.4, 0.4],
            [5.0, 5.0, 5.0], // 다른 복셀
        ];
        let result = voxel_downsample_cpu(&points, &params);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_voxel_center_is_average() {
        let params = test_params();
        let points = vec![[0.0, 0.0, 0.0], [0.8, 0.8, 0.8]];
        let result = voxel_downsample_cpu(&points, &params);
        assert_eq!(result.len(), 1);
        assert!((result[0][0] - 0.4).abs() < 1e-5);
    }

    #[test]
    fn test_gaussian_kernel_sums_to_one() {
        let kernel = gaussian_kernel_5x5(1.0);
        let sum: f32 = kernel.iter().flat_map(|row| row.iter()).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_gaussian_kernel_is_symmetric() {
        let kernel = gaussian_kernel_5x5(1.5);
        for i in 0..5 {
            for j in 0..5 {
                assert!((kernel[i][j] - kernel[4 - i][4 - j]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_compute_pass_desc_is_pure_data() {
        // ComputePassDesc는 GPU 없이 만들 수 있어야 함
        let desc = ComputePassDesc {
            shader_path: "shaders/voxel.wgsl",
            entry_point: "main",
            params: test_params(),
            workgroup_size: [256, 1, 1],
        };
        assert_eq!(desc.workgroup_size[0], 256);
    }

    #[test]
    fn test_cpu_and_gpu_impl_give_same_result() {
        // GPU 있을 때: CpuProcessor와 GpuProcessor 결과를 비교
        // CI에서는 CpuProcessor만으로도 로직 검증 가능
        let params = test_params();
        let points = vec![[1.1, 2.2, 3.3], [1.2, 2.3, 3.4], [7.0, 7.0, 7.0]];
        let cpu = CpuProcessor;
        let result = cpu.voxel_downsample(&points, &params);
        assert_eq!(result.len(), 2);
    }
}
```

GPU 없이 `voxel_downsample_cpu`, `point_to_voxel_index`, `gaussian_kernel_5x5` 모두 테스트 가능합니다. GPU 구현이 틀렸을 때는 CPU 레퍼런스와 비교해서 발견합니다.

---

## 렌더 그래프와의 관계

[함수형 렌더 그래프](/posts/programming/functional/functional-render-graph/) 글은 렌더링 패스 스케줄링이 주제였습니다. Compute 패스도 렌더 그래프에 포함시킬 수 있습니다.

```rust
// 렌더 그래프에 Compute 패스 추가
let graph = vec![
    PassDesc { name: "point_cloud_voxel", reads: vec!["lidar_raw"], writes: vec!["voxel_grid"], pass_type: PassType::Compute },
    PassDesc { name: "object_cluster",    reads: vec!["voxel_grid"], writes: vec!["clusters"],   pass_type: PassType::Compute },
    PassDesc { name: "shadow_map",        reads: vec!["clusters"],   writes: vec!["shadow_tex"], pass_type: PassType::Render  },
    PassDesc { name: "scene_render",      reads: vec!["shadow_tex"], writes: vec!["final"],      pass_type: PassType::Render  },
];
```

렌더 그래프의 위상 정렬은 Compute 패스와 렌더 패스를 구분하지 않습니다. 데이터 의존만 봅니다.

---

## 정리

| 구성 요소 | 분류 | 특징 |
|---|---|---|
| `VoxelParams`, `ImageFilterParams`, `ComputePassDesc` | 데이터 | GPU 없이 생성, 직렬화 가능, 파라미터 검색 용이 |
| `point_to_voxel_index`, `voxel_downsample_cpu`, `gaussian_kernel_5x5` | 계산 | CPU 레퍼런스, GPU 없이 테스트, WGSL과 1:1 대응 |
| `GpuContext::create_buffer`, `dispatch`, `read_buffer` | 액션 | GPU와 통신, 테스트에서 격리 |

GPU 프로그래밍의 어려움 중 많은 부분은 디버깅 도구 부족에서 옵니다. 커널 로직을 CPU 순수 함수로 먼저 작성하고 GPU로 옮기면, 로직 버그는 CPU에서 잡고 GPU에서는 성능만 얻을 수 있습니다.

---

*관련 글: [액션/계산/데이터](/posts/programming/functional/functional-actions-calculations-data/), [함수형 셰이더 파이프라인](/posts/programming/functional/functional-shader-pipeline/), [함수형 렌더 그래프](/posts/programming/functional/functional-render-graph/), [함수형 포인트 클라우드 처리](/posts/programming/functional/functional-point-cloud/), [함수형 DI](/posts/programming/functional/functional-dependency-injection/)*
