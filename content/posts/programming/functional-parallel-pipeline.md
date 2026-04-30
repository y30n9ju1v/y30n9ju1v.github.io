---
title: "함수형 병렬 파이프라인: Rayon과 GPU Compute의 역할 분담"
date: 2026-04-30T21:00:00+09:00
draft: false
tags: ["함수형 프로그래밍", "Rust", "설계", "GPU", "병렬 처리", "Rayon", "액션/계산/데이터", "자율주행"]
categories: ["프로그래밍", "GPU"]
description: "Rayon의 par_iter()는 함수형 패턴과 자연스럽게 맞습니다. 언제 CPU 병렬(Rayon)을 쓰고 언제 GPU Compute를 써야 하는지, 그리고 둘을 함께 쓸 때 액션/계산/데이터가 어떻게 분리되는지 설명합니다."
---

## 이 글을 읽고 나면

- Rayon의 `par_iter()`가 함수형 패턴과 왜 자연스럽게 맞는지 이해합니다.
- CPU 병렬(Rayon)과 GPU Compute 중 어떤 상황에서 무엇을 선택해야 하는지 기준을 가집니다.
- 둘을 혼합한 파이프라인에서도 액션/계산/데이터 분리가 유지되는 구조를 봅니다.
- 자율주행 인식 파이프라인을 예제로 구체적인 분기점을 확인합니다.

이전 글 [액션/계산/데이터](/posts/programming/functional-actions-calculations-data/), [GPU Compute 셰이더를 함수형으로](/posts/programming/functional-gpu-compute/)를 먼저 읽으면 더 자연스럽게 이어집니다.

---

## Rayon이 함수형과 잘 맞는 이유

Rayon은 Rust의 데이터 병렬 라이브러리입니다. `iter()`를 `par_iter()`로 바꾸면 CPU 코어를 나눠 쓰는 병렬 처리가 됩니다.

```rust
// 순차 처리
let results: Vec<_> = points.iter().map(|p| process(p)).collect();

// Rayon 병렬 처리
use rayon::prelude::*;
let results: Vec<_> = points.par_iter().map(|p| process(p)).collect();
```

`par_iter()`가 안전하게 병렬 실행되는 조건은 딱 하나입니다. **`process`가 순수 함수여야 합니다.** 공유 가변 상태가 없어야 하고, 입력만 보고 출력을 결정해야 합니다.

이것이 [액션/계산/데이터](/posts/programming/functional-actions-calculations-data/)와 직결되는 이유입니다. 계산(순수 함수)은 병렬 실행해도 안전합니다. 액션(부수효과)은 순서가 중요해서 병렬 실행 시 버그가 생깁니다.

```rust
// 계산 → par_iter() 안전
let filtered: Vec<_> = points.par_iter()
    .filter(|p| p.z > 0.5)            // 순수: 입력만 봄
    .map(|p| normalize(*p))            // 순수: 새 값 반환
    .collect();

// 액션 → par_iter() 위험
let mut count = 0;
points.par_iter().for_each(|p| {
    count += 1; // 공유 가변 상태 → 데이터 레이스
});
```

Rayon은 순수 함수를 강제하는 방향으로 설계되어 있습니다. `par_iter().map()`은 소유권 규칙상 공유 가변 상태를 쓸 수 없어서, 컴파일러가 액션을 병렬 실행하려는 시도를 거부합니다.

---

## CPU 병렬 vs GPU Compute: 언제 무엇을?

GPU Compute가 항상 빠르지는 않습니다. GPU로 보내는 것 자체에 비용이 있습니다.

```
CPU → GPU 데이터 전송 (PCIe, 수 ms)
GPU 커널 실행
GPU → CPU 결과 읽기 (PCIe, 수 ms)
```

데이터가 작거나 연산이 단순하면 전송 비용이 연산 비용을 압도합니다.

판단 기준을 정리하면 이렇습니다.

| 상황 | 선택 | 이유 |
|---|---|---|
| 데이터 이미 GPU에 있음 | GPU | 전송 없음 |
| 데이터 10만 포인트 이상, 단순 연산 | GPU | 대규모 병렬이 이득 |
| 데이터 1만 포인트 이하 | Rayon | 전송 비용이 더 큼 |
| 복잡한 조건 분기 | Rayon | GPU는 분기에 취약 |
| 동적 크기 (프레임마다 다름) | Rayon | GPU 버퍼 재할당 비용 |
| 추론 결과 후처리 (NMS 등) | Rayon | 검출 박스 수십~수백 개 |
| LiDAR 10만 포인트 다운샘플링 | GPU | 규칙적 단순 연산 |
| 이미지 배치 처리 (카메라 8대) | GPU | 동일 연산 대량 반복 |

자율주행 파이프라인에서 실제로 어떻게 분기되는지 보겠습니다.

---

## 나쁜 방법: 모든 것을 GPU로

```rust
fn process_frame(image: &[u8], points: &[Point3D]) -> FrameResult {
    // 이미지 전처리 → GPU
    let gpu_image = upload_to_gpu(image);
    let normalized = gpu_normalize(gpu_image);   // 10ms 전송 + 0.1ms 연산

    // 검출 결과 NMS → GPU
    let raw_boxes = run_inference(normalized);
    let gpu_boxes = upload_to_gpu(&raw_boxes);
    let nms_result = gpu_nms(gpu_boxes);         // 1ms 전송 + 0.01ms 연산

    // 포인트 클라우드 필터링 → GPU
    let gpu_points = upload_to_gpu(points);
    let filtered = gpu_voxel(gpu_points);         // 5ms 전송 + 2ms 연산

    FrameResult { /* ... */ }
}
```

박스 수십 개에 NMS를 GPU로 하는 건 전송 비용 낭비입니다. GPU가 항상 빠르다는 가정이 오히려 느리게 만듭니다.

---

## 데이터: 파이프라인 단계를 선언적으로

```rust
/// 처리 백엔드를 선택하는 설정 (데이터)
#[derive(Debug, Clone, PartialEq)]
pub enum Backend {
    Sequential,      // 단일 스레드 (디버그·기준선)
    CpuParallel,     // Rayon
    Gpu,             // GPU Compute
}

/// 파이프라인 단계 설정 (데이터)
#[derive(Debug, Clone)]
pub struct StageConfig {
    pub name: &'static str,
    pub backend: Backend,
    pub batch_size: usize,
}

/// 파이프라인 전체 설정 (데이터)
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub point_cloud_stage: StageConfig,
    pub preprocess_stage: StageConfig,
    pub postprocess_stage: StageConfig,
}

impl PipelineConfig {
    /// 실차용: 데이터 규모에 맞는 기본값
    pub fn production() -> Self {
        PipelineConfig {
            point_cloud_stage: StageConfig {
                name: "point_cloud",
                backend: Backend::Gpu,          // 10만 포인트 → GPU
                batch_size: 131072,
            },
            preprocess_stage: StageConfig {
                name: "preprocess",
                backend: Backend::CpuParallel,  // 카메라 8대 동시 → Rayon
                batch_size: 8,
            },
            postprocess_stage: StageConfig {
                name: "postprocess",
                backend: Backend::CpuParallel,  // 박스 수백 개 → Rayon
                batch_size: 1,
            },
        }
    }

    /// CI용: GPU 없이 로직만 검증
    pub fn testing() -> Self {
        PipelineConfig {
            point_cloud_stage: StageConfig {
                name: "point_cloud",
                backend: Backend::Sequential,
                batch_size: 100,
            },
            preprocess_stage: StageConfig {
                name: "preprocess",
                backend: Backend::Sequential,
                batch_size: 1,
            },
            postprocess_stage: StageConfig {
                name: "postprocess",
                backend: Backend::Sequential,
                batch_size: 1,
            },
        }
    }
}
```

백엔드 선택이 데이터입니다. 파이프라인 로직을 건드리지 않고 설정만 바꿔 CPU·GPU를 전환합니다.

---

## 계산: 순수 함수로 작성된 처리 로직

```rust
use rayon::prelude::*;

/// 포인트 하나 필터링 — 순수 계산 (CPU·GPU 공용)
pub fn filter_point(p: &[f32; 3], min_z: f32, max_range: f32) -> bool {
    let range = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    p[2] > min_z && range < max_range
}

/// 포인트 정규화 — 순수 계산
pub fn normalize_point(p: &[f32; 3], range: f32) -> [f32; 3] {
    [p[0] / range, p[1] / range, p[2] / range]
}

/// 이미지 전처리 — 순수 계산
pub fn preprocess_image(
    image: &[u8],
    width: usize, height: usize,
    target_w: usize, target_h: usize,
) -> Vec<f32> {
    // 리사이즈 + 정규화
    let scale_x = width as f32 / target_w as f32;
    let scale_y = height as f32 / target_h as f32;

    (0..target_h * target_w).map(|idx| {
        let dy = idx / target_w;
        let dx = idx % target_w;
        let sx = (dx as f32 * scale_x) as usize;
        let sy = (dy as f32 * scale_y) as usize;
        image[(sy * width + sx).min(image.len() - 1)] as f32 / 255.0
    }).collect()
}
```

이 함수들은 `&self`도, `mut`도 없습니다. Rayon의 `par_iter()`에 그대로 넘길 수 있습니다.

---

## 계산: Rayon 병렬 처리

```rust
/// CPU 병렬 포인트 클라우드 필터링
pub fn filter_points_parallel(
    points: &[[f32; 3]],
    min_z: f32,
    max_range: f32,
) -> Vec<[f32; 3]> {
    points.par_iter()
        .filter(|p| filter_point(p, min_z, max_range))
        .copied()
        .collect()
}

/// CPU 병렬 카메라 전처리 (카메라 N대 동시 처리)
pub fn preprocess_cameras_parallel(
    images: &[&[u8]],
    widths: &[usize],
    heights: &[usize],
    target_w: usize,
    target_h: usize,
) -> Vec<Vec<f32>> {
    images.par_iter()
        .zip(widths.par_iter())
        .zip(heights.par_iter())
        .map(|((img, &w), &h)| preprocess_image(img, w, h, target_w, target_h))
        .collect()
}

/// CPU 병렬 NMS (카테고리별 독립 처리)
pub fn nms_parallel(
    detections_by_class: &[Vec<Detection>],
    iou_threshold: f32,
) -> Vec<Detection> {
    detections_by_class.par_iter()
        .flat_map(|class_dets| nms_single_class(class_dets, iou_threshold))
        .collect()
}

fn nms_single_class(dets: &[Detection], iou_threshold: f32) -> Vec<Detection> {
    let mut sorted = dets.to_vec();
    sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    let mut kept = vec![];
    let mut suppressed = vec![false; sorted.len()];

    for i in 0..sorted.len() {
        if suppressed[i] { continue; }
        kept.push(sorted[i].clone());
        for j in (i + 1)..sorted.len() {
            if iou(&sorted[i], &sorted[j]) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }
    kept
}

#[derive(Debug, Clone)]
pub struct Detection {
    pub x1: f32, pub y1: f32,
    pub x2: f32, pub y2: f32,
    pub score: f32,
    pub class_id: usize,
}

pub fn iou(a: &Detection, b: &Detection) -> f32 {
    let ix1 = a.x1.max(b.x1);
    let iy1 = a.y1.max(b.y1);
    let ix2 = a.x2.min(b.x2);
    let iy2 = a.y2.min(b.y2);
    let inter = ((ix2 - ix1).max(0.0)) * ((iy2 - iy1).max(0.0));
    let a_area = (a.x2 - a.x1) * (a.y2 - a.y1);
    let b_area = (b.x2 - b.x1) * (b.y2 - b.y1);
    let union = a_area + b_area - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}
```

`par_iter()`에 들어가는 클로저가 모두 순수 함수를 호출합니다. Rayon이 내부적으로 스레드풀을 관리하지만, 호출 코드는 순차 처리와 거의 동일합니다.

---

## 액션: GPU Compute와 카메라 읽기

```rust
pub struct GpuPointCloudProcessor {
    // wgpu Device, Queue, 버퍼 캐시
}

impl GpuPointCloudProcessor {
    /// 포인트 클라우드 필터링 (액션: GPU 통신)
    pub fn filter_points(
        &self,
        points: &[[f32; 3]],
        min_z: f32,
        max_range: f32,
    ) -> Vec<[f32; 3]> {
        // CPU → GPU 업로드
        // Compute 셰이더 디스패치 (filter_point와 동일 로직)
        // GPU → CPU 다운로드
        // (GPU Compute 셰이더 글 참고)
        vec![]
    }
}

/// 처리 백엔드 트레잇
pub trait PointCloudBackend {
    fn filter_points(&self, points: &[[f32; 3]], min_z: f32, max_range: f32) -> Vec<[f32; 3]>;
}

pub struct CpuBackend;
impl PointCloudBackend for CpuBackend {
    fn filter_points(&self, points: &[[f32; 3]], min_z: f32, max_range: f32) -> Vec<[f32; 3]> {
        filter_points_parallel(points, min_z, max_range)
    }
}

impl PointCloudBackend for GpuPointCloudProcessor {
    fn filter_points(&self, points: &[[f32; 3]], min_z: f32, max_range: f32) -> Vec<[f32; 3]> {
        self.filter_points(points, min_z, max_range)
    }
}
```

`PointCloudBackend` 트레잇을 받는 코드는 Rayon인지 GPU인지 모릅니다. 설정(`PipelineConfig`)을 바꾸면 전환됩니다.

---

## 파이프라인 조합

```rust
/// 프레임 처리 파이프라인 — 백엔드 선택에 무관한 구조
pub fn process_frame<P: PointCloudBackend>(
    lidar_points: &[[f32; 3]],
    camera_images: &[&[u8]],
    camera_widths: &[usize],
    camera_heights: &[usize],
    point_backend: &P,
    config: &PipelineConfig,
) -> FrameResult {
    // ── 계산 (CPU 병렬: Rayon) ──────────────────────────
    let preprocessed_images = preprocess_cameras_parallel(
        camera_images, camera_widths, camera_heights, 640, 640,
    );

    // ── 계산 또는 액션 (설정에 따라 CPU/GPU) ────────────
    let filtered_points = point_backend.filter_points(
        lidar_points, 0.5, 50.0,
    );

    // ── 계산 (CPU 병렬: Rayon) ──────────────────────────
    // 추론 결과를 카테고리별로 분리 후 병렬 NMS
    let raw_detections = run_inference_batch(&preprocessed_images); // 액션
    let detections_by_class = group_by_class(&raw_detections);      // 계산
    let final_detections = nms_parallel(&detections_by_class, 0.5); // 계산

    FrameResult {
        detections: final_detections,
        filtered_points,
    }
}

pub struct FrameResult {
    pub detections: Vec<Detection>,
    pub filtered_points: Vec<[f32; 3]>,
}

fn group_by_class(dets: &[Detection]) -> Vec<Vec<Detection>> {
    if dets.is_empty() { return vec![]; }
    let max_class = dets.iter().map(|d| d.class_id).max().unwrap();
    let mut groups = vec![vec![]; max_class + 1];
    for d in dets { groups[d.class_id].push(d.clone()); }
    groups
}

fn run_inference_batch(_images: &[Vec<f32>]) -> Vec<Detection> {
    // InferenceSession::run() 호출 → 액션
    vec![]
}
```

파이프라인 안에서 계산과 액션이 명확히 구분됩니다.

```
lidar_points, camera_images
    │
    ├─ preprocess_cameras_parallel()  ← 계산 (Rayon)
    │
    ├─ point_backend.filter_points()  ← CPU: 계산(Rayon) / GPU: 액션
    │
    ├─ run_inference_batch()          ← 액션 (GPU 추론)
    │
    ├─ group_by_class()               ← 계산
    │
    └─ nms_parallel()                 ← 계산 (Rayon)
                │
                ▼
           FrameResult (데이터)
```

---

## 병렬 처리에서 흔한 실수

### 공유 가변 상태를 액션 안으로 끌어들이기

```rust
// 위험: 카운터를 병렬 클로저 안에서 수정
let mut total = 0usize;
points.par_iter().for_each(|p| {
    if filter_point(p, 0.5, 50.0) {
        total += 1; // 컴파일 에러 — Rust가 막아줌
    }
});

// 안전: reduce로 집계
let total = points.par_iter()
    .filter(|p| filter_point(p, 0.5, 50.0))
    .count();
```

Rust의 소유권 규칙이 공유 가변 상태를 차단합니다. `par_iter()`에서 컴파일 에러가 나면, 그 클로저에 액션이 섞여 있다는 신호입니다.

### 순서 의존 처리를 par_iter()에 넣기

```rust
// 위험: 앞 단계 결과가 뒤 단계 입력인데 병렬화
let results: Vec<_> = pipeline_stages.par_iter()
    .map(|stage| stage.run(&previous_result)) // previous_result가 공유됨 → 버그
    .collect();

// 안전: 독립적인 데이터에만 par_iter()
let results: Vec<_> = independent_inputs.par_iter()
    .map(|input| run_stage(input))
    .collect();
```

`par_iter()`는 각 원소가 독립적일 때만 안전합니다. 카메라 8대의 전처리는 독립적이므로 안전합니다. 파이프라인 단계들은 순서 의존이 있으므로 `par_iter()` 대상이 아닙니다.

---

## 자율주행에서의 실제 분기점

프레임당 처리해야 하는 데이터를 기준으로 정리하면 이렇습니다.

```
LiDAR 원시 포인트 (10만~50만 개)
  └─ Voxel 다운샘플링           → GPU (규칙적 연산, 대량)
  └─ 범위 필터링                → GPU (단순 조건, 대량)
  └─ 클러스터링 결과 정제       → Rayon (수백 클러스터, 복잡한 분기)

카메라 (8대 × 1920×1080)
  └─ 리사이즈·정규화           → Rayon (카메라 단위로 독립)
  └─ DNN 추론 (배치)           → GPU (동일 연산 대량 반복)
  └─ NMS (박스 수백 개)        → Rayon (작은 데이터, GPU 전송 낭비)

레이더
  └─ 도플러 필터               → Rayon (수십~수백 타겟, 간단한 연산)

센서 퓨전 (GPS·IMU·카메라·LiDAR)
  └─ 칼만 필터 업데이트        → Sequential (의존 관계 있음, 순서 중요)
  └─ 물체 매칭 (헝가리안)      → Sequential (전역 최적화, 분해 불가)
```

판단 기준은 세 가지입니다.

1. **데이터 규모**: 1만 이상이면 GPU 고려, 수백이면 Rayon이 전송 비용 이득
2. **연산 구조**: 조건 분기 많으면 Rayon, 규칙적 반복이면 GPU
3. **순서 의존**: 독립적이면 `par_iter()`, 앞 결과가 뒤 입력이면 순차

---

## 테스트: 백엔드를 바꿔서 결과 비교

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn sample_points() -> Vec<[f32; 3]> {
        vec![
            [1.0, 0.0, 1.0],   // 통과: z > 0.5, range < 50
            [0.0, 0.0, 0.3],   // 제거: z <= 0.5
            [40.0, 30.0, 1.0], // 제거: range > 50
            [-5.0, 3.0, 2.0],  // 통과
        ]
    }

    #[test]
    fn test_filter_sequential() {
        let points = sample_points();
        let result: Vec<_> = points.iter()
            .filter(|p| filter_point(p, 0.5, 50.0))
            .copied()
            .collect();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_filter_parallel_same_result() {
        let points = sample_points();

        let seq: Vec<_> = points.iter()
            .filter(|p| filter_point(p, 0.5, 50.0))
            .copied()
            .collect();

        let par = filter_points_parallel(&points, 0.5, 50.0);

        // 순서는 다를 수 있으나 원소 집합은 동일
        assert_eq!(seq.len(), par.len());
        for p in &seq {
            assert!(par.contains(p));
        }
    }

    #[test]
    fn test_preprocess_cameras_parallel() {
        let img1 = vec![128u8; 4 * 4 * 3];
        let img2 = vec![64u8; 8 * 8 * 3];
        let images: Vec<&[u8]> = vec![&img1, &img2];
        let widths = [4, 8];
        let heights = [4, 8];

        let result = preprocess_cameras_parallel(&images, &widths, &heights, 2, 2);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), 2 * 2);
        assert_eq!(result[1].len(), 2 * 2);
    }

    #[test]
    fn test_nms_parallel_by_class() {
        let by_class = vec![
            // 클래스 0: 겹치는 두 박스
            vec![
                Detection { x1: 0.0, y1: 0.0, x2: 10.0, y2: 10.0, score: 0.9, class_id: 0 },
                Detection { x1: 1.0, y1: 1.0, x2: 11.0, y2: 11.0, score: 0.8, class_id: 0 },
            ],
            // 클래스 1: 독립적인 두 박스
            vec![
                Detection { x1: 50.0, y1: 50.0, x2: 60.0, y2: 60.0, score: 0.7, class_id: 1 },
                Detection { x1: 70.0, y1: 70.0, x2: 80.0, y2: 80.0, score: 0.6, class_id: 1 },
            ],
        ];

        let result = nms_parallel(&by_class, 0.5);
        // 클래스 0에서 1개, 클래스 1에서 2개
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_sequential_and_parallel_nms_same() {
        let dets = vec![
            Detection { x1: 0.0, y1: 0.0, x2: 10.0, y2: 10.0, score: 0.9, class_id: 0 },
            Detection { x1: 1.0, y1: 1.0, x2: 11.0, y2: 11.0, score: 0.8, class_id: 0 },
            Detection { x1: 50.0, y1: 50.0, x2: 60.0, y2: 60.0, score: 0.7, class_id: 0 },
        ];
        let seq = nms_single_class(&dets, 0.5);

        let by_class = vec![dets];
        let par = nms_parallel(&by_class, 0.5);

        assert_eq!(seq.len(), par.len());
    }

    #[test]
    fn test_pipeline_config_selects_backend() {
        let prod = PipelineConfig::production();
        assert_eq!(prod.point_cloud_stage.backend, Backend::Gpu);
        assert_eq!(prod.postprocess_stage.backend, Backend::CpuParallel);

        let test = PipelineConfig::testing();
        assert_eq!(test.point_cloud_stage.backend, Backend::Sequential);
    }
}
```

순차·병렬 처리가 동일한 순수 함수를 공유하므로, 두 결과를 비교하는 테스트가 자연스럽게 작성됩니다.

---

## Rayon과 GPU Compute의 공통점

둘 다 같은 원칙 위에 있습니다.

| 원칙 | Rayon | GPU Compute |
|---|---|---|
| 병렬 안전 조건 | 순수 함수 (`Sync + Send`) | 전역 상태 없는 커널 |
| 입출력 방식 | `par_iter()` → 컬렉션 | 입력 버퍼 → 출력 버퍼 |
| 디버그 전환 | `par_iter()` → `iter()` | GPU 구현 → CPU 레퍼런스 |
| 함수형 분류 | 계산 (순수 함수 보장됨) | 계산 (커널) + 액션 (디스패치) |

GPU Compute에서 CPU 레퍼런스를 먼저 작성하는 이유가 여기에 있습니다. 그 레퍼런스 함수를 Rayon으로 병렬 실행하면 중간 단계 역할도 합니다. 데이터가 작아지면 GPU 대신 Rayon 레퍼런스가 그대로 쓰입니다.

---

## 정리

| 구성 요소 | 분류 | 특징 |
|---|---|---|
| `PipelineConfig`, `StageConfig`, `Backend` | 데이터 | 백엔드 선택을 코드 밖으로 분리 |
| `filter_point`, `preprocess_image`, `iou`, `nms_single_class` | 계산 | 순수 함수, Rayon·GPU·Sequential 모두 사용 가능 |
| `filter_points_parallel`, `preprocess_cameras_parallel`, `nms_parallel` | 계산 (Rayon) | `par_iter()` 래퍼, 부수효과 없음 |
| `GpuPointCloudProcessor::filter_points`, `run_inference_batch` | 액션 | GPU 경계, 트레잇으로 교체 가능 |

Rayon은 함수형 패턴의 자연스러운 연장입니다. 순수 함수를 병렬로 실행하는 기계입니다. GPU Compute는 더 큰 데이터에 같은 원칙을 적용합니다. 둘 중 무엇을 쓰든, 처리 로직이 순수 함수이면 테스트·전환·조합이 쉽습니다.

---

*관련 글: [액션, 계산, 데이터](/posts/programming/functional-actions-calculations-data/), [GPU Compute 셰이더를 함수형으로](/posts/programming/functional-gpu-compute/), [함수형 포인트 클라우드 처리](/posts/programming/functional-point-cloud/), [함수형 DI](/posts/programming/functional-dependency-injection/)*
