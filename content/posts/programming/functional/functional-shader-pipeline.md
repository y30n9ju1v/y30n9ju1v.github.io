---
title: "함수형 셰이더 파이프라인: GPU가 이미 순수 함수로 동작한다"
date: 2026-04-30T08:00:00+09:00
draft: false
tags: ["함수형 프로그래밍", "Rust", "설계", "GPU", "셰이더", "렌더링", "액션/계산/데이터"]
categories: ["프로그래밍", "GPU"]
description: "GPU 셰이더 파이프라인은 이미 액션/계산/데이터 구조를 따릅니다. CPU 쪽 파이프라인 설정 코드도 같은 방식으로 나누면 테스트와 재사용이 쉬워집니다."
---

## 이 글을 읽고 나면

- GPU 셰이더가 왜 자연스럽게 순수 함수인지 이해합니다.
- CPU 쪽 파이프라인 설정 코드를 액션/계산/데이터로 나누는 방법을 봅니다.
- 셰이더 파라미터 계산을 GPU 없이 테스트하는 방법을 이해합니다.

이전 글 [액션, 계산, 데이터](/posts/programming/functional-actions-calculations-data/)를 먼저 읽으면 더 자연스럽게 이어집니다.

---

## GPU는 이미 함수형이다

GPU 셰이더를 처음 보면 낯설지만, 구조를 보면 함수형 프로그래밍의 이상적인 형태입니다.

```wgsl
// 버텍스 셰이더: 3D 위치 → 화면 좌표
@vertex
fn vs_main(vertex: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = uniforms.mvp_matrix * vec4f(vertex.position, 1.0);
    out.color = vertex.color;
    return out;
}

// 프래그먼트 셰이더: 화면 좌표 + 색상 → 픽셀 색
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let light = dot(in.normal, uniforms.light_dir);
    return in.color * max(light, 0.1);
}
```

- 전역 상태 없음
- 다른 버텍스/픽셀의 결과에 영향받지 않음
- 같은 입력이면 항상 같은 출력

GPU 하드웨어가 강제하는 제약이 순수 함수를 만듭니다. 병렬 실행이 가능한 이유도 이것입니다.

문제는 **CPU 쪽 설정 코드**입니다. 버퍼 생성, 파이프라인 설정, 유니폼 업데이트가 뒤섞이면 복잡해집니다.

---

## 문제: CPU 쪽 코드가 뒤섞이면

```rust
fn render_frame(device: &Device, queue: &Queue, scene: &Scene) {
    // 행렬 계산 (계산)
    let mvp = scene.camera.projection * scene.camera.view * scene.object.transform;

    // 버퍼 업데이트 (액션)
    queue.write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&mvp));

    // 커맨드 인코딩 (액션)
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&render_pass_desc);
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    // GPU 제출 (액션)
    queue.submit([encoder.finish()]);
}
```

행렬 계산(`mvp`)이 GPU 제출 코드 안에 묻혀 있습니다. 행렬이 올바른지 테스트하려면 GPU가 필요합니다. 카메라 파라미터가 바뀌면 어디를 건드려야 할지 추적하기 어렵습니다.

---

## 액션/계산/데이터로 나누기

```
데이터(Data)           계산(Calculation)              액션(Action)
────────────           ─────────────────              ─────────────
CameraParams      →   compute_view_matrix()      →   queue.write_buffer()
LightParams       →   compute_projection()        →   encoder.begin_render_pass()
ObjectTransform   →   compute_mvp()               →   queue.submit()
                  →   compute_light_uniforms()
                  →   compute_draw_params()
```

계산은 GPU 없이 실행됩니다. 액션은 GPU 경계에서만 발생합니다.

---

## 데이터: 렌더링 파라미터

```rust
#[derive(Debug, Clone)]
struct CameraParams {
    position: [f32; 3],
    target: [f32; 3],
    up: [f32; 3],
    fov_rad: f32,
    aspect: f32,
    near: f32,
    far: f32,
}

#[derive(Debug, Clone)]
struct LightParams {
    direction: [f32; 3], // 정규화된 방향벡터
    color: [f32; 3],
    intensity: f32,
}

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Uniforms {
    mvp: [[f32; 4]; 4],
    light_dir: [f32; 4],   // w는 패딩
    light_color: [f32; 4], // w는 intensity
}

#[derive(Debug, Clone)]
struct DrawParams {
    vertex_count: u32,
    instance_count: u32,
}
```

`Uniforms`는 GPU에 직접 올라가는 바이트 구조체입니다. 계산 결과를 담는 컨테이너일 뿐, 하드웨어 의존이 없습니다.

---

## 계산: 행렬과 파라미터 계산

```rust
// 뷰 행렬: 카메라 위치/방향 → 4x4 행렬
fn compute_view_matrix(camera: &CameraParams) -> [[f32; 4]; 4] {
    let eye = camera.position;
    let center = camera.target;
    let up = camera.up;

    let f = normalize(sub3(center, eye));
    let s = normalize(cross3(f, up));
    let u = cross3(s, f);

    [
        [s[0],  u[0], -f[0], 0.0],
        [s[1],  u[1], -f[1], 0.0],
        [s[2],  u[2], -f[2], 0.0],
        [-dot3(s, eye), -dot3(u, eye), dot3(f, eye), 1.0],
    ]
}

// 투영 행렬: FOV/종횡비/클리핑 → 4x4 행렬
fn compute_projection_matrix(camera: &CameraParams) -> [[f32; 4]; 4] {
    let f = 1.0 / (camera.fov_rad / 2.0).tan();
    let range = camera.near - camera.far;
    [
        [f / camera.aspect, 0.0, 0.0,                                    0.0],
        [0.0,               f,   0.0,                                    0.0],
        [0.0,               0.0, (camera.far + camera.near) / range,    -1.0],
        [0.0,               0.0, (2.0 * camera.far * camera.near) / range, 0.0],
    ]
}

// MVP 행렬 합성
fn compute_mvp(
    model: &[[f32; 4]; 4],
    camera: &CameraParams,
) -> [[f32; 4]; 4] {
    let view = compute_view_matrix(camera);
    let proj = compute_projection_matrix(camera);
    mat4_mul(&proj, &mat4_mul(&view, model))
}

// 유니폼 구조체 계산
fn compute_uniforms(
    model: &[[f32; 4]; 4],
    camera: &CameraParams,
    light: &LightParams,
) -> Uniforms {
    Uniforms {
        mvp: compute_mvp(model, camera),
        light_dir: [light.direction[0], light.direction[1], light.direction[2], 0.0],
        light_color: [light.color[0], light.color[1], light.color[2], light.intensity],
    }
}

// 드로우 파라미터 계산 (메쉬 데이터 기반)
fn compute_draw_params(vertex_count: u32) -> DrawParams {
    DrawParams { vertex_count, instance_count: 1 }
}
```

`compute_uniforms`는 카메라, 조명, 모델 데이터만 있으면 됩니다. GPU 컨텍스트가 없어도 실행됩니다.

---

## 액션: GPU 경계에서만

```rust
// 액션: 유니폼 데이터를 GPU 버퍼에 업로드
fn upload_uniforms(uniforms: &Uniforms, buffer: &Buffer, queue: &Queue) {
    queue.write_buffer(buffer, 0, bytemuck::bytes_of(uniforms));
}

// 액션: 렌더 패스 실행
fn record_render_pass(
    encoder: &mut CommandEncoder,
    view: &TextureView,
    pipeline: &RenderPipeline,
    bind_group: &BindGroup,
    draw: &DrawParams,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })],
        ..Default::default()
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.draw(0..draw.vertex_count, 0..draw.instance_count);
}

// 액션: GPU에 제출
fn submit_frame(device: &Device, queue: &Queue, encoder: CommandEncoder) {
    queue.submit([encoder.finish()]);
    device.poll(wgpu::Maintain::Wait);
}
```

---

## 파이프라인 조합

계산을 먼저 모두 끝내고, 액션을 마지막에 몰아서 실행합니다.

```rust
fn render_frame(
    device: &Device,
    queue: &Queue,
    surface_view: &TextureView,
    pipeline: &RenderPipeline,
    bind_group: &BindGroup,
    uniform_buffer: &Buffer,
    scene: &Scene,
) {
    // ── 계산 (GPU 없이) ──────────────────────────────
    let uniforms = compute_uniforms(&scene.model_matrix, &scene.camera, &scene.light);
    let draw = compute_draw_params(scene.vertex_count);

    // ── 액션 (GPU 경계) ──────────────────────────────
    upload_uniforms(&uniforms, uniform_buffer, queue);

    let mut encoder = device.create_command_encoder(&Default::default());
    record_render_pass(&mut encoder, surface_view, pipeline, bind_group, &draw);
    submit_frame(device, queue, encoder);
}
```

데이터 흐름이 명확합니다.

```
scene.camera          ─┐
scene.light           ─┼─→ compute_uniforms() → Uniforms ─→ upload_uniforms() (액션)
scene.model_matrix    ─┘
scene.vertex_count     ──→ compute_draw_params() → DrawParams ─→ record_render_pass() (액션)
                                                                → submit_frame() (액션)
```

---

## 테스트: GPU 없이 행렬 검증

계산 함수들이 순수하므로 GPU 없이 수학적으로 검증할 수 있습니다.

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn default_camera() -> CameraParams {
        CameraParams {
            position: [0.0, 0.0, 3.0],
            target: [0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fov_rad: PI / 4.0,
            aspect: 16.0 / 9.0,
            near: 0.1,
            far: 100.0,
        }
    }

    #[test]
    fn test_view_matrix_origin_at_identity() {
        let camera = CameraParams {
            position: [0.0, 0.0, 0.0],
            target: [0.0, 0.0, -1.0], // -Z 방향
            up: [0.0, 1.0, 0.0],
            ..default_camera()
        };
        let view = compute_view_matrix(&camera);
        // 원점에서 -Z를 바라보는 뷰 행렬은 대각이 1에 가까워야 함
        assert!((view[0][0] - 1.0).abs() < 1e-5);
        assert!((view[1][1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_projection_clips_near_far() {
        let camera = default_camera();
        let proj = compute_projection_matrix(&camera);
        // near 평면의 점은 NDC z ≈ -1, far 평면은 z ≈ 1
        // proj[2][2]와 proj[3][2]로 검증
        assert!(proj[2][2] < 0.0); // 부호 확인
        assert!(proj[3][2] < 0.0);
    }

    #[test]
    fn test_uniforms_light_direction_preserved() {
        let model = identity_matrix();
        let camera = default_camera();
        let light = LightParams {
            direction: [0.0, 1.0, 0.0],
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
        };
        let uniforms = compute_uniforms(&model, &camera, &light);
        assert!((uniforms.light_dir[1] - 1.0).abs() < 1e-5);
        assert!((uniforms.light_color[3] - 1.0).abs() < 1e-5); // intensity
    }

    #[test]
    fn test_draw_params() {
        let params = compute_draw_params(36);
        assert_eq!(params.vertex_count, 36);
        assert_eq!(params.instance_count, 1);
    }
}

fn identity_matrix() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}
```

행렬 계산 버그를 GPU 없이 잡을 수 있습니다. CI에서도 실행됩니다.

---

## 셰이더 자체는 이미 순수 함수

셰이더 언어(WGSL, GLSL, HLSL)는 구조적으로 순수 함수를 강제합니다.

- 전역 변수 쓰기 불가 (uniforms는 읽기 전용)
- 다른 스레드(버텍스/픽셀)의 상태 접근 불가
- 같은 입력 → 항상 같은 출력

CPU에서 계산 코드를 순수 함수로 관리하면, 전체 렌더링 파이프라인이 일관된 철학을 갖습니다.

```
CPU 계산 (순수 함수)          CPU 액션              GPU 셰이더 (순수 함수)
────────────────────         ─────────────         ─────────────────────
compute_mvp()           →   upload_uniforms()  →   vs_main(vertex) → clip_pos
compute_uniforms()      →   record_pass()      →   fs_main(frag)   → pixel_color
compute_draw_params()   →   submit_frame()
```

계산 레이어가 CPU와 GPU 양쪽에서 일관되게 유지됩니다.

---

## 정리

| 분류 | 내용 | 특징 |
|------|------|------|
| 데이터 | `CameraParams`, `LightParams`, `Uniforms`, `DrawParams` | GPU 무관, 직렬화 가능 |
| 계산 | `compute_view_matrix`, `compute_mvp`, `compute_uniforms` | 순수 함수, GPU 없이 테스트 |
| 액션 | `upload_uniforms`, `record_render_pass`, `submit_frame` | GPU 경계에만 존재 |
| 셰이더 | `vs_main`, `fs_main` | 하드웨어가 강제하는 순수 함수 |

GPU 셰이더가 이미 순수 함수인 것처럼, CPU 쪽 설정 코드도 같은 방식으로 나눌 수 있습니다. 계산이 두껍고 액션이 얇을수록 테스트하기 쉽고 재사용하기 쉽습니다.

---

*관련 글: [액션, 계산, 데이터](/posts/programming/functional-actions-calculations-data/), [함수 컴포지션](/posts/programming/functional-composition/), [함수형 역기구학](/posts/programming/functional-inverse-kinematics/)*
