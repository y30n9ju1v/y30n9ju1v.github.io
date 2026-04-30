---
title: "함수형 추론 그래프: DNN 추론 파이프라인을 액션/계산/데이터로"
date: 2026-04-30T18:00:00+09:00
draft: false
tags: ["함수형 프로그래밍", "Rust", "설계", "GPU", "딥러닝", "추론", "ONNX", "액션/계산/데이터", "자율주행"]
categories: ["프로그래밍", "GPU"]
description: "ONNX/TensorRT 추론 파이프라인을 액션/계산/데이터로 분리하면, GPU 없이 레이어 의존 관계를 검증하고 CPU 레퍼런스로 추론 결과를 재현할 수 있습니다."
---

## 이 글을 읽고 나면

- DNN 추론 파이프라인에서 무엇이 액션이고 무엇이 계산인지 구분합니다.
- `LayerDesc`를 선언적 데이터로, 레이어 실행 순서 결정을 순수 계산으로 분리하는 구조를 이해합니다.
- GPU 없이 파이프라인 구성 로직과 전처리·후처리를 테스트하는 방법을 봅니다.
- 자율주행 카메라·LiDAR 추론에 이 구조가 어떻게 적용되는지 확인합니다.

이전 글 [액션/계산/데이터](/posts/programming/functional-actions-calculations-data/), [함수형 렌더 그래프](/posts/programming/functional-render-graph/), [GPU Compute 셰이더를 함수형으로](/posts/programming/functional-gpu-compute/)를 먼저 읽으면 더 자연스럽게 이어집니다.

---

## DNN 추론이 렌더 그래프와 닮은 이유

[함수형 렌더 그래프](/posts/programming/functional-render-graph/) 글에서 렌더 패스를 **선언(데이터) → 의존 분석(계산) → GPU 실행(액션)** 세 단계로 나눴습니다. DNN 추론 파이프라인도 구조가 같습니다.

렌더 그래프에서 "패스가 어떤 텍스처를 읽고 쓰는지"를 선언했다면, 추론 그래프에서는 "레이어가 어떤 텐서를 읽고 쓰는지"를 선언합니다.

```
렌더 그래프                   추론 그래프
─────────────                 ──────────────
PassDesc                  ↔   LayerDesc
텍스처 (reads/writes)     ↔   텐서 (inputs/outputs)
topological_sort          ↔   동일
cull_unused_passes        ↔   cull_unused_layers
execute_plan              ↔   run_inference
```

차이는 자율주행에서 DNN 추론이 더 자주 바뀐다는 점입니다. 백본 교체, 헤드 추가, 양자화 적용, TensorRT 버전 변경 — 이 변경들이 파이프라인 선언(데이터)만 건드리고 나머지 코드를 그대로 두려면 분리가 필요합니다.

---

## 문제: 추론 코드가 한 함수에 뭉쳐있을 때

```rust
fn detect_objects(image: &[u8], width: u32, height: u32) -> Vec<Detection> {
    // 전처리가 추론 코드 안에
    let normalized = image.iter().map(|&p| p as f32 / 255.0).collect::<Vec<_>>();
    let resized = bilinear_resize(&normalized, width, height, 640, 640);

    // 모델 로드와 추론이 뒤섞임
    let session = ort::Session::builder().with_model_from_file("yolo.onnx").unwrap();
    let input = Tensor::from_array(([1, 3, 640, 640], resized.as_slice().unwrap())).unwrap();
    let outputs = session.run(ort::inputs![input]).unwrap();

    // 후처리가 추론 결과와 뒤섞임
    let raw_boxes = outputs[0].try_extract_tensor::<f32>().unwrap();
    let boxes = nms(raw_boxes.view(), 0.5, 0.4);
    boxes.iter().map(|b| Detection { ...*b }).collect()
}
```

문제들:

- 전처리·추론·후처리가 한 함수에 있어 단계별 테스트 불가
- 모델 파일이 없으면 함수 전체가 실행 불가 — CI에서 로직 검증 어려움
- NMS 임계값, 입력 해상도가 하드코딩 — 변경 시 함수 전체를 건드림
- 다른 모델(LiDAR 물체 검출, 차선 검출)을 추가하면 함수를 복사·붙여넣기

---

## 액션/계산/데이터로 나누기

```
데이터(Data)               계산(Calculation)              액션(Action)
────────────               ─────────────────              ─────────────
LayerDesc             →   build_graph(layers)        →   load_model(path)
TensorDesc            →   topological_sort()         →   allocate_tensor(desc)
InferenceGraph        →   cull_unused_layers()       →   run_layer(layer, tensors)
ExecutionPlan         →   compute_input_shapes()     →   read_camera()
PreprocessConfig      →   preprocess_cpu(image)      →   write_result()
PostprocessConfig     →   postprocess_cpu(output)
```

전처리와 후처리는 계산입니다. GPU 추론만 액션입니다.

---

## 데이터: 레이어와 텐서를 선언적으로

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorId(pub String);

/// 텐서 형상과 타입 기술자 (데이터)
#[derive(Debug, Clone)]
pub struct TensorDesc {
    pub id: TensorId,
    pub shape: Vec<usize>,   // [batch, channel, height, width] 순
    pub dtype: DType,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DType { F32, F16, U8, I64 }

/// 레이어 하나를 설명하는 선언적 기술자 (데이터)
#[derive(Debug, Clone)]
pub struct LayerDesc {
    pub name: String,
    pub op: LayerOp,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
}

#[derive(Debug, Clone)]
pub enum LayerOp {
    Conv2d { out_channels: usize, kernel: usize, stride: usize, padding: usize },
    BatchNorm,
    Relu,
    MaxPool { kernel: usize, stride: usize },
    Concat { axis: usize },
    Upsample { scale: usize },
    OnnxNode { op_type: String },  // ONNX에서 불러온 레이어
}

/// 전체 추론 그래프 (데이터)
#[derive(Debug, Clone)]
pub struct InferenceGraph {
    pub layers: Vec<LayerDesc>,
    pub tensors: Vec<TensorDesc>,
    pub input_tensors: Vec<TensorId>,   // 외부 입력 (카메라, LiDAR)
    pub output_tensors: Vec<TensorId>,  // 최종 출력 (검출 결과)
}

/// 실행 순서가 결정된 계획 (데이터)
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub ordered_layers: Vec<String>,
    pub active_tensors: Vec<TensorId>,
    pub tensor_shapes: std::collections::HashMap<TensorId, Vec<usize>>,
}
```

`InferenceGraph`는 GPU를 만지지 않습니다. ONNX 파일을 파싱해서 만들 수도 있고, 코드로 직접 조립할 수도 있습니다.

---

## 계산: 그래프 분석

### 의존 관계 구성

```rust
use std::collections::HashMap;

pub fn build_dependency_graph(
    graph: &InferenceGraph,
) -> HashMap<String, Vec<String>> {
    // 각 텐서를 출력으로 갖는 레이어 인덱스
    let mut producer_of: HashMap<&TensorId, &str> = HashMap::new();
    for layer in &graph.layers {
        for tensor in &layer.outputs {
            producer_of.insert(tensor, &layer.name);
        }
    }

    // 각 레이어가 의존하는 레이어 목록
    graph.layers.iter().map(|layer| {
        let deps = layer.inputs.iter()
            .filter_map(|t| producer_of.get(t))
            .map(|&name| name.to_string())
            .collect();
        (layer.name.clone(), deps)
    }).collect()
}
```

### 위상 정렬

```rust
pub fn topological_sort(
    layers: &[LayerDesc],
    deps: &HashMap<String, Vec<String>>,
) -> Result<Vec<String>, String> {
    let mut in_degree: HashMap<String, usize> = layers.iter()
        .map(|l| (l.name.clone(), deps.get(&l.name).map(|d| d.len()).unwrap_or(0)))
        .collect();

    let mut queue: Vec<String> = in_degree.iter()
        .filter(|(_, &deg)| deg == 0)
        .map(|(name, _)| name.clone())
        .collect();
    queue.sort();

    let mut result = vec![];

    while let Some(name) = queue.first().cloned() {
        queue.remove(0);
        result.push(name.clone());

        for layer in layers {
            if deps.get(&layer.name).map(|d| d.contains(&name)).unwrap_or(false) {
                let deg = in_degree.get_mut(&layer.name).unwrap();
                *deg -= 1;
                if *deg == 0 {
                    queue.push(layer.name.clone());
                    queue.sort();
                }
            }
        }
    }

    if result.len() != layers.len() {
        Err("순환 의존이 존재합니다".to_string())
    } else {
        Ok(result)
    }
}
```

### 불필요한 레이어 제거

자율주행에서는 동일한 백본 위에 여러 헤드(2D 검출, 깊이 추정, 차선 검출)가 붙는 경우가 많습니다. 현재 필요한 헤드만 활성화하면 나머지 레이어는 실행하지 않아도 됩니다.

```rust
pub fn cull_unused_layers(
    graph: &InferenceGraph,
    required_outputs: &[TensorId],
) -> Vec<String> {
    let producer_of: HashMap<&TensorId, &str> = graph.layers.iter()
        .flat_map(|l| l.outputs.iter().map(move |t| (t, l.name.as_str())))
        .collect();

    let mut needed: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut queue: Vec<&TensorId> = required_outputs.iter().collect();

    while let Some(tensor) = queue.pop() {
        if let Some(&layer_name) = producer_of.get(tensor) {
            if needed.insert(layer_name.to_string()) {
                if let Some(layer) = graph.layers.iter().find(|l| l.name == layer_name) {
                    queue.extend(&layer.inputs);
                }
            }
        }
    }

    needed.into_iter().collect()
}
```

### 전체 컴파일

```rust
pub fn compile(
    graph: &InferenceGraph,
    required_outputs: &[TensorId],
) -> Result<ExecutionPlan, String> {
    let active_names = cull_unused_layers(graph, required_outputs);
    let active_layers: Vec<LayerDesc> = graph.layers.iter()
        .filter(|l| active_names.contains(&l.name))
        .cloned()
        .collect();

    let active_graph = InferenceGraph {
        layers: active_layers.clone(),
        ..graph.clone()
    };
    let deps = build_dependency_graph(&active_graph);
    let ordered = topological_sort(&active_layers, &deps)?;

    let active_tensors: Vec<TensorId> = active_layers.iter()
        .flat_map(|l| l.inputs.iter().chain(l.outputs.iter()))
        .cloned()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    let tensor_shapes: HashMap<TensorId, Vec<usize>> = graph.tensors.iter()
        .filter(|t| active_tensors.contains(&t.id))
        .map(|t| (t.id.clone(), t.shape.clone()))
        .collect();

    Ok(ExecutionPlan { ordered_layers: ordered, active_tensors, tensor_shapes })
}
```

`compile`은 순수 함수입니다. ONNX 파일도, GPU도, 실제 이미지도 없이 실행됩니다.

---

## 계산: 전처리와 후처리

전처리와 후처리는 추론과 분리된 순수 계산입니다.

```rust
/// 전처리 파라미터 (데이터)
#[derive(Debug, Clone)]
pub struct PreprocessConfig {
    pub target_width: usize,
    pub target_height: usize,
    pub mean: [f32; 3],   // RGB 채널별 평균
    pub std: [f32; 3],    // RGB 채널별 표준편차
}

/// 이미지 전처리 — 순수 계산
pub fn preprocess(
    image: &[u8],
    src_width: usize,
    src_height: usize,
    config: &PreprocessConfig,
) -> Vec<f32> {
    // 1. [0, 255] → [0.0, 1.0]
    let normalized: Vec<f32> = image.iter().map(|&p| p as f32 / 255.0).collect();

    // 2. 리사이즈 (bilinear)
    let resized = bilinear_resize_chw(
        &normalized,
        src_width, src_height,
        config.target_width, config.target_height,
    );

    // 3. 정규화: (x - mean) / std
    resized.iter().enumerate().map(|(i, &v)| {
        let channel = (i / (config.target_width * config.target_height)) % 3;
        (v - config.mean[channel]) / config.std[channel]
    }).collect()
}

fn bilinear_resize_chw(
    input: &[f32],
    src_w: usize, src_h: usize,
    dst_w: usize, dst_h: usize,
) -> Vec<f32> {
    let channels = input.len() / (src_w * src_h);
    let mut output = vec![0.0f32; channels * dst_w * dst_h];

    for c in 0..channels {
        for dy in 0..dst_h {
            for dx in 0..dst_w {
                let sx = dx as f32 * (src_w as f32 / dst_w as f32);
                let sy = dy as f32 * (src_h as f32 / dst_h as f32);
                let x0 = sx.floor() as usize;
                let y0 = sy.floor() as usize;
                let x1 = (x0 + 1).min(src_w - 1);
                let y1 = (y0 + 1).min(src_h - 1);
                let wx = sx - sx.floor();
                let wy = sy - sy.floor();

                let base = c * src_w * src_h;
                let v = input[base + y0 * src_w + x0] * (1.0 - wx) * (1.0 - wy)
                      + input[base + y0 * src_w + x1] * wx * (1.0 - wy)
                      + input[base + y1 * src_w + x0] * (1.0 - wx) * wy
                      + input[base + y1 * src_w + x1] * wx * wy;

                output[c * dst_w * dst_h + dy * dst_w + dx] = v;
            }
        }
    }
    output
}

/// 검출 결과 (데이터)
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub x1: f32, pub y1: f32,
    pub x2: f32, pub y2: f32,
    pub score: f32,
    pub class_id: usize,
}

/// 후처리 파라미터 (데이터)
#[derive(Debug, Clone)]
pub struct PostprocessConfig {
    pub score_threshold: f32,
    pub iou_threshold: f32,
    pub input_width: usize,
    pub input_height: usize,
    pub orig_width: usize,
    pub orig_height: usize,
}

/// IoU 계산 — 순수 계산
pub fn iou(a: &Detection, b: &Detection) -> f32 {
    let inter_x1 = a.x1.max(b.x1);
    let inter_y1 = a.y1.max(b.y1);
    let inter_x2 = a.x2.min(b.x2);
    let inter_y2 = a.y2.min(b.y2);

    let inter_area = ((inter_x2 - inter_x1).max(0.0)) * ((inter_y2 - inter_y1).max(0.0));
    let a_area = (a.x2 - a.x1) * (a.y2 - a.y1);
    let b_area = (b.x2 - b.x1) * (b.y2 - b.y1);
    let union_area = a_area + b_area - inter_area;

    if union_area <= 0.0 { 0.0 } else { inter_area / union_area }
}

/// NMS (Non-Maximum Suppression) — 순수 계산
pub fn nms(detections: &[Detection], config: &PostprocessConfig) -> Vec<Detection> {
    let mut boxes: Vec<Detection> = detections.iter()
        .filter(|d| d.score >= config.score_threshold)
        .cloned()
        .collect();

    boxes.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    let mut kept: Vec<Detection> = vec![];
    let mut suppressed = vec![false; boxes.len()];

    for i in 0..boxes.len() {
        if suppressed[i] { continue; }
        kept.push(boxes[i].clone());
        for j in (i + 1)..boxes.len() {
            if boxes[i].class_id == boxes[j].class_id
                && iou(&boxes[i], &boxes[j]) > config.iou_threshold
            {
                suppressed[j] = true;
            }
        }
    }

    // 원본 이미지 좌표로 역변환
    let scale_x = config.orig_width as f32 / config.input_width as f32;
    let scale_y = config.orig_height as f32 / config.input_height as f32;

    kept.into_iter().map(|mut d| {
        d.x1 *= scale_x; d.x2 *= scale_x;
        d.y1 *= scale_y; d.y2 *= scale_y;
        d
    }).collect()
}
```

`preprocess`, `iou`, `nms` 모두 순수 함수입니다. 모델 파일 없이, GPU 없이 테스트 가능합니다.

---

## 액션: 모델 로드와 GPU 추론

GPU와 직접 통신하는 부분만 액션입니다.

```rust
pub struct InferenceSession {
    // ort::Session 또는 TensorRT engine 등
    // GPU 리소스를 보유하는 유일한 구조체
}

impl InferenceSession {
    /// 모델 로드 (액션)
    pub fn load(model_path: &str) -> Result<Self, String> {
        // ort::Session::builder().with_model_from_file(model_path)
        // 파일 I/O + GPU 메모리 할당 → 액션
        Ok(InferenceSession { /* ... */ })
    }

    /// 텐서 추론 (액션)
    pub fn run(&self, input: &[f32], shape: &[usize]) -> Result<Vec<f32>, String> {
        // GPU 메모리 업로드 → 커널 실행 → 결과 다운로드
        // 모두 부수효과 → 액션
        Ok(vec![])
    }
}
```

`InferenceSession`이 액션의 경계입니다. 이 구조체를 모킹하거나 CPU 구현체로 교체하면 나머지 코드는 그대로 테스트됩니다.

---

## 파이프라인 조합

```rust
/// 추론 백엔드 트레잇
pub trait InferenceBackend {
    fn run(&self, input: &[f32], shape: &[usize]) -> Result<Vec<f32>, String>;
}

/// GPU 백엔드 (실차)
pub struct OrtBackend(InferenceSession);

impl InferenceBackend for OrtBackend {
    fn run(&self, input: &[f32], shape: &[usize]) -> Result<Vec<f32>, String> {
        self.0.run(input, shape)
    }
}

/// CPU 레퍼런스 백엔드 (테스트·CI)
pub struct CpuRefBackend;

impl InferenceBackend for CpuRefBackend {
    fn run(&self, input: &[f32], _shape: &[usize]) -> Result<Vec<f32>, String> {
        // 단순 패스스루 또는 미리 저장된 결과 반환
        // 전처리·후처리 로직만 검증할 때 사용
        Ok(vec![0.0; 100 * 85]) // 더미 출력
    }
}

/// 전체 파이프라인
pub fn run_detection<B: InferenceBackend>(
    image: &[u8],
    src_width: usize,
    src_height: usize,
    pre_config: &PreprocessConfig,
    post_config: &PostprocessConfig,
    backend: &B,
) -> Result<Vec<Detection>, String> {
    // ── 계산 ────────────────────────────────────────────
    let input_tensor = preprocess(image, src_width, src_height, pre_config);

    // ── 액션 ────────────────────────────────────────────
    let shape = [1, 3, pre_config.target_height, pre_config.target_width];
    let raw_output = backend.run(&input_tensor, &shape)?;

    // ── 계산 ────────────────────────────────────────────
    let raw_detections = parse_yolo_output(&raw_output, 80);
    let detections = nms(&raw_detections, post_config);

    Ok(detections)
}

fn parse_yolo_output(output: &[f32], num_classes: usize) -> Vec<Detection> {
    // cx, cy, w, h, obj_score, class_scores... 형식 파싱 → 순수 계산
    let stride = 5 + num_classes;
    output.chunks(stride)
        .filter_map(|anchor| {
            if anchor.len() < stride { return None; }
            let obj_score = anchor[4];
            let class_scores = &anchor[5..];
            let (class_id, &class_score) = class_scores.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())?;
            let score = obj_score * class_score;
            let (cx, cy, w, h) = (anchor[0], anchor[1], anchor[2], anchor[3]);
            Some(Detection {
                x1: cx - w / 2.0, y1: cy - h / 2.0,
                x2: cx + w / 2.0, y2: cy + h / 2.0,
                score, class_id,
            })
        })
        .collect()
}
```

`run_detection`은 백엔드 트레잇만 받습니다. `OrtBackend`인지 `CpuRefBackend`인지 모릅니다. [함수형 DI](/posts/programming/functional-dependency-injection/) 패턴입니다.

---

## 자율주행 파이프라인에서의 위치

자율주행 인식 스택에서 추론 그래프가 어디에 놓이는지 보면 이렇습니다.

```
카메라 이미지 (액션: 읽기)
    │
    ▼
preprocess()                  ← 계산
    │
    ▼
InferenceBackend::run()       ← 액션 (GPU 경계)
    │
    ▼
parse_yolo_output()           ← 계산
nms()                         ← 계산
    │
    ▼
Vec<Detection>                ← 데이터
    │
    ▼
칼만 필터 (함수형 칼만 필터 참고)  ← 계산
    │
    ▼
TrackedObject 목록             ← 데이터
```

GPU 경계는 `InferenceBackend::run()` 한 줄입니다. 그 앞뒤는 전부 순수 계산이므로, 실차 없이 이미지 파일 하나로 파이프라인 전체를 검증할 수 있습니다.

---

## 멀티헤드 모델: 컬링의 실용적 가치

자율주행에서 자주 쓰는 구조입니다. 백본 하나에 헤드 여러 개가 붙습니다.

```rust
let graph = InferenceGraph {
    layers: vec![
        LayerDesc { name: "backbone".to_string(), op: LayerOp::OnnxNode { op_type: "...".to_string() },
            inputs: vec![TensorId("image".to_string())],
            outputs: vec![TensorId("features".to_string())] },
        LayerDesc { name: "det_head".to_string(), op: LayerOp::OnnxNode { op_type: "...".to_string() },
            inputs: vec![TensorId("features".to_string())],
            outputs: vec![TensorId("detections".to_string())] },
        LayerDesc { name: "depth_head".to_string(), op: LayerOp::OnnxNode { op_type: "...".to_string() },
            inputs: vec![TensorId("features".to_string())],
            outputs: vec![TensorId("depth_map".to_string())] },
        LayerDesc { name: "lane_head".to_string(), op: LayerOp::OnnxNode { op_type: "...".to_string() },
            inputs: vec![TensorId("features".to_string())],
            outputs: vec![TensorId("lanes".to_string())] },
    ],
    tensors: vec![/* 형상 정보 */],
    input_tensors: vec![TensorId("image".to_string())],
    output_tensors: vec![TensorId("detections".to_string())],
};

// 물체 검출만 필요한 상황
let plan = compile(&graph, &[TensorId("detections".to_string())]).unwrap();
// → backbone + det_head만 실행, depth_head와 lane_head는 제외
```

헤드를 추가·제거해도 `compile()`이 자동으로 필요한 레이어만 선택합니다. 실행 코드를 건드리지 않습니다.

---

## 테스트: 모델 파일 없이 로직 검증

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(name: &str) -> TensorId { TensorId(name.to_string()) }

    fn make_graph() -> InferenceGraph {
        InferenceGraph {
            layers: vec![
                LayerDesc {
                    name: "backbone".to_string(),
                    op: LayerOp::OnnxNode { op_type: "Backbone".to_string() },
                    inputs: vec![tensor("image")],
                    outputs: vec![tensor("features")],
                },
                LayerDesc {
                    name: "det_head".to_string(),
                    op: LayerOp::OnnxNode { op_type: "DetHead".to_string() },
                    inputs: vec![tensor("features")],
                    outputs: vec![tensor("detections")],
                },
                LayerDesc {
                    name: "depth_head".to_string(),
                    op: LayerOp::OnnxNode { op_type: "DepthHead".to_string() },
                    inputs: vec![tensor("features")],
                    outputs: vec![tensor("depth_map")],
                },
            ],
            tensors: vec![],
            input_tensors: vec![tensor("image")],
            output_tensors: vec![tensor("detections")],
        }
    }

    #[test]
    fn test_backbone_runs_before_head() {
        let graph = make_graph();
        let plan = compile(&graph, &[tensor("detections")]).unwrap();
        let pos = |name: &str| plan.ordered_layers.iter().position(|n| n == name).unwrap();

        assert!(pos("backbone") < pos("det_head"));
    }

    #[test]
    fn test_unused_head_culled() {
        let graph = make_graph();
        let plan = compile(&graph, &[tensor("detections")]).unwrap();

        assert!(!plan.ordered_layers.contains(&"depth_head".to_string()));
        assert!(plan.ordered_layers.contains(&"det_head".to_string()));
        assert!(plan.ordered_layers.contains(&"backbone".to_string()));
    }

    #[test]
    fn test_both_heads_included_when_needed() {
        let graph = make_graph();
        let plan = compile(
            &graph,
            &[tensor("detections"), tensor("depth_map")],
        ).unwrap();

        assert!(plan.ordered_layers.contains(&"det_head".to_string()));
        assert!(plan.ordered_layers.contains(&"depth_head".to_string()));
    }

    #[test]
    fn test_iou_overlap() {
        let a = Detection { x1: 0.0, y1: 0.0, x2: 2.0, y2: 2.0, score: 0.9, class_id: 0 };
        let b = Detection { x1: 1.0, y1: 1.0, x2: 3.0, y2: 3.0, score: 0.8, class_id: 0 };
        let result = iou(&a, &b);
        // 교집합 1x1=1, 합집합 4+4-1=7
        assert!((result - 1.0 / 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_iou_no_overlap() {
        let a = Detection { x1: 0.0, y1: 0.0, x2: 1.0, y2: 1.0, score: 0.9, class_id: 0 };
        let b = Detection { x1: 2.0, y1: 2.0, x2: 3.0, y2: 3.0, score: 0.8, class_id: 0 };
        assert_eq!(iou(&a, &b), 0.0);
    }

    #[test]
    fn test_nms_removes_duplicate() {
        let config = PostprocessConfig {
            score_threshold: 0.3,
            iou_threshold: 0.5,
            input_width: 640, input_height: 640,
            orig_width: 1280, orig_height: 720,
        };
        let detections = vec![
            Detection { x1: 0.0, y1: 0.0, x2: 100.0, y2: 100.0, score: 0.9, class_id: 0 },
            Detection { x1: 5.0, y1: 5.0, x2: 105.0, y2: 105.0, score: 0.8, class_id: 0 },
            Detection { x1: 300.0, y1: 300.0, x2: 400.0, y2: 400.0, score: 0.7, class_id: 0 },
        ];
        let result = nms(&detections, &config);
        // 첫 두 박스는 겹치므로 하나 제거, 세 번째는 독립
        assert_eq!(result.len(), 2);
        assert!(result[0].score >= result[1].score);
    }

    #[test]
    fn test_nms_different_classes_not_suppressed() {
        let config = PostprocessConfig {
            score_threshold: 0.3,
            iou_threshold: 0.5,
            input_width: 640, input_height: 640,
            orig_width: 640, orig_height: 640,
        };
        let detections = vec![
            Detection { x1: 0.0, y1: 0.0, x2: 100.0, y2: 100.0, score: 0.9, class_id: 0 },
            Detection { x1: 0.0, y1: 0.0, x2: 100.0, y2: 100.0, score: 0.8, class_id: 1 },
        ];
        let result = nms(&detections, &config);
        // 클래스가 다르면 겹쳐도 제거하지 않음
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_preprocess_output_size() {
        let config = PreprocessConfig {
            target_width: 4,
            target_height: 4,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
        };
        let image = vec![128u8; 8 * 8 * 3]; // 8x8 RGB
        let result = preprocess(&image, 8, 8, &config);
        assert_eq!(result.len(), 3 * 4 * 4); // CHW 순서
    }

    #[test]
    fn test_cyclic_dependency_detected() {
        let graph = InferenceGraph {
            layers: vec![
                LayerDesc {
                    name: "a".to_string(),
                    op: LayerOp::Relu,
                    inputs: vec![tensor("b_out")],
                    outputs: vec![tensor("a_out")],
                },
                LayerDesc {
                    name: "b".to_string(),
                    op: LayerOp::Relu,
                    inputs: vec![tensor("a_out")],
                    outputs: vec![tensor("b_out")],
                },
            ],
            tensors: vec![],
            input_tensors: vec![],
            output_tensors: vec![tensor("b_out")],
        };
        let deps = build_dependency_graph(&graph);
        let result = topological_sort(&graph.layers, &deps);
        assert!(result.is_err());
    }
}
```

모델 파일, GPU, 카메라 없이 `cargo test`로 전부 실행됩니다.

---

## 렌더 그래프와의 비교

| | 렌더 그래프 | 추론 그래프 |
|---|---|---|
| 선언 단위 | `PassDesc` (렌더 패스) | `LayerDesc` (레이어) |
| 리소스 | 텍스처 | 텐서 |
| 컴파일 | `compile(graph, outputs)` | 동일 |
| 실행 | `execute_plan(device)` | `InferenceBackend::run()` |
| 컬링 동기 | 디버그 패스 비활성화 | 멀티헤드 선택적 실행 |
| 주요 차이 | 프레임마다 동일 그래프 | 배포 시 그래프 고정, 파라미터 변경 잦음 |

---

## 정리

| 구성 요소 | 분류 | 특징 |
|---|---|---|
| `LayerDesc`, `TensorDesc`, `InferenceGraph`, `ExecutionPlan` | 데이터 | GPU 무관, 선언적, ONNX 파싱 결과로 만들 수 있음 |
| `build_dependency_graph`, `topological_sort`, `cull_unused_layers`, `compile`, `preprocess`, `nms`, `iou`, `parse_yolo_output` | 계산 | 순수 함수, 모델 없이 테스트 |
| `InferenceSession::load`, `InferenceBackend::run` | 액션 | GPU 경계, 테스트에서 교체 가능 |

DNN 추론에서 어려운 부분의 대부분은 전처리·후처리 버그와 그래프 구성 실수입니다. 이 부분을 순수 함수로 만들면 모델 없이 버그를 잡을 수 있습니다.

---

*관련 글: [액션, 계산, 데이터](/posts/programming/functional-actions-calculations-data/), [함수형 렌더 그래프](/posts/programming/functional-render-graph/), [GPU Compute 셰이더를 함수형으로](/posts/programming/functional-gpu-compute/), [함수형 DI](/posts/programming/functional-dependency-injection/), [함수형 포인트 클라우드 처리](/posts/programming/functional-point-cloud/)*
