---
title: "함수형 렌더 그래프: GPU 패스 스케줄링을 순수 계산으로"
date: 2026-04-30T12:00:00+09:00
draft: false
tags: ["함수형 프로그래밍", "Rust", "설계", "GPU", "렌더 그래프", "렌더링", "액션/계산/데이터"]
categories: ["프로그래밍", "GPU"]
description: "어떤 렌더 패스를 어떤 순서로 실행할지 결정하는 렌더 그래프를 액션/계산/데이터로 나누면, GPU 없이 패스 의존 관계를 검증하고 최적화 로직을 테스트할 수 있습니다."
---

## 이 글을 읽고 나면

- 렌더 그래프가 무엇이고 왜 필요한지 이해합니다.
- 패스 의존 관계 분석과 실행 순서 결정이 왜 순수 계산인지 봅니다.
- GPU 없이 렌더 그래프 로직을 테스트하는 방법을 이해합니다.

이전 글 [함수형 셰이더 파이프라인](/posts/programming/functional/functional-shader-pipeline/)을 먼저 읽으면 더 자연스럽게 이어집니다.

---

## 문제: 렌더 패스가 늘어나면

현대 렌더링은 단일 패스가 아닙니다.

```
Shadow Map Pass    → 그림자 깊이 맵 생성
G-Buffer Pass      → 위치·법선·색상 버퍼 생성
SSAO Pass          → 주변광 차폐 계산 (G-Buffer 필요)
Lighting Pass      → 조명 계산 (G-Buffer + Shadow Map 필요)
Bloom Pass         → 빛 번짐 (Lighting 결과 필요)
Tone Mapping Pass  → HDR → LDR 변환 (Bloom 결과 필요)
```

패스마다 다른 패스의 결과를 입력으로 씁니다. 이걸 수동으로 관리하면 이렇게 됩니다.

```rust
fn render_frame(device: &Device, queue: &Queue, scene: &Scene) {
    render_shadow_map(device, queue, scene);  // 액션
    render_gbuffer(device, queue, scene);      // 액션
    render_ssao(device, queue, scene);         // 액션 — G-Buffer가 먼저여야 함
    render_lighting(device, queue, scene);     // 액션 — Shadow + G-Buffer 필요
    render_bloom(device, queue, scene);        // 액션 — Lighting 먼저
    render_tone_mapping(device, queue, scene); // 액션 — Bloom 먼저
}
```

패스를 하나 추가할 때마다 순서를 직접 조정해야 합니다. 패스를 비활성화하면 의존하는 패스들도 손으로 찾아야 합니다. 순서가 틀리면 GPU에서 쓰레기 값이 나옵니다.

렌더 그래프는 이 문제를 해결합니다. **어떤 패스가 어떤 텍스처를 읽고 쓰는지 선언하면, 실행 순서를 자동으로 결정합니다.**

---

## 액션/계산/데이터로 나누기

```
데이터(Data)               계산(Calculation)                  액션(Action)
────────────               ─────────────────                  ─────────────
PassDesc              →   build_graph(passes)            →   execute_pass(pass, device)
ResourceDesc          →   topological_sort(graph)        →   create_texture(desc, device)
RenderGraph           →   cull_unused_passes(graph)      →   queue.submit()
ExecutionPlan         →   compute_resource_lifetimes()
```

그래프 구성과 최적화는 전부 계산입니다. GPU는 실행 단계에서만 등장합니다.

---

## 데이터: 패스와 리소스 선언

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ResourceId(String);

#[derive(Debug, Clone)]
struct ResourceDesc {
    id: ResourceId,
    width: u32,
    height: u32,
    format: TextureFormat,
}

#[derive(Debug, Clone)]
enum TextureFormat { Rgba8, Depth32, Rgba16Float, R8 }

#[derive(Debug, Clone)]
struct PassDesc {
    name: String,
    reads: Vec<ResourceId>,  // 이 패스가 읽는 텍스처
    writes: Vec<ResourceId>, // 이 패스가 쓰는 텍스처
}

#[derive(Debug, Clone)]
struct RenderGraph {
    passes: Vec<PassDesc>,
    resources: Vec<ResourceDesc>,
}

#[derive(Debug, Clone)]
struct ExecutionPlan {
    ordered_passes: Vec<String>, // 실행 순서가 결정된 패스 이름 목록
    active_resources: Vec<ResourceId>, // 실제로 필요한 리소스
}
```

`PassDesc`는 패스가 무엇을 읽고 쓰는지를 **선언**합니다. 실제 GPU 명령이 없습니다.

---

## 계산: 그래프 분석과 최적화

### 의존 그래프 구성

```rust
use std::collections::HashMap;

fn build_dependency_graph(graph: &RenderGraph) -> HashMap<String, Vec<String>> {
    // 각 리소스를 쓰는 패스를 역방향으로 인덱싱
    let mut writer_of: HashMap<&ResourceId, &str> = HashMap::new();
    for pass in &graph.passes {
        for resource in &pass.writes {
            writer_of.insert(resource, &pass.name);
        }
    }

    // 각 패스가 의존하는 패스 목록
    let mut deps: HashMap<String, Vec<String>> = HashMap::new();
    for pass in &graph.passes {
        let pass_deps: Vec<String> = pass.reads.iter()
            .filter_map(|r| writer_of.get(r))
            .map(|&name| name.to_string())
            .collect();
        deps.insert(pass.name.clone(), pass_deps);
    }
    deps
}
```

### 위상 정렬: 실행 순서 결정

```rust
fn topological_sort(
    passes: &[PassDesc],
    deps: &HashMap<String, Vec<String>>,
) -> Result<Vec<String>, String> {
    let mut in_degree: HashMap<&str, usize> = passes.iter()
        .map(|p| (p.name.as_str(), 0))
        .collect();

    for pass_deps in deps.values() {
        for dep in pass_deps {
            // dep이 완료되어야 하는 패스의 in_degree 증가
        }
    }

    // 각 패스에 대해 자신을 필요로 하는 패스들의 in_degree 계산
    for pass in passes {
        for dep in deps.get(&pass.name).unwrap_or(&vec![]) {
            let _ = dep; // in_degree는 아래서 계산
        }
    }

    // Kahn's algorithm
    let mut result = vec![];
    let mut in_degree: HashMap<String, usize> = passes.iter()
        .map(|p| (p.name.clone(), 0))
        .collect();

    // 각 패스가 몇 개의 패스에 의존하는지 계산
    for pass in passes {
        let count = deps.get(&pass.name).map(|d| d.len()).unwrap_or(0);
        *in_degree.get_mut(&pass.name).unwrap() = count;
    }

    let mut queue: Vec<String> = in_degree.iter()
        .filter(|(_, &deg)| deg == 0)
        .map(|(name, _)| name.clone())
        .collect();
    queue.sort(); // 결정론적 순서 보장

    while let Some(pass_name) = queue.first().cloned() {
        queue.remove(0);
        result.push(pass_name.clone());

        // 이 패스에 의존하던 패스들의 in_degree 감소
        for pass in passes {
            if deps.get(&pass.name)
                .map(|d| d.contains(&pass_name))
                .unwrap_or(false)
            {
                let deg = in_degree.get_mut(&pass.name).unwrap();
                *deg -= 1;
                if *deg == 0 {
                    queue.push(pass.name.clone());
                    queue.sort();
                }
            }
        }
    }

    if result.len() != passes.len() {
        Err("순환 의존이 존재합니다".to_string())
    } else {
        Ok(result)
    }
}
```

### 불필요한 패스 제거 (Culling)

최종 출력에 기여하지 않는 패스는 실행하지 않아도 됩니다.

```rust
fn cull_unused_passes(
    graph: &RenderGraph,
    output_resources: &[ResourceId],
) -> Vec<String> {
    // 출력 리소스를 쓰는 패스부터 역방향으로 필요한 패스 수집
    let writer_of: HashMap<&ResourceId, &str> = graph.passes.iter()
        .flat_map(|p| p.writes.iter().map(move |r| (r, p.name.as_str())))
        .collect();

    let mut needed: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut queue: Vec<&ResourceId> = output_resources.iter().collect();

    while let Some(resource) = queue.pop() {
        if let Some(&pass_name) = writer_of.get(resource) {
            if needed.insert(pass_name.to_string()) {
                // 이 패스가 읽는 리소스도 추적
                if let Some(pass) = graph.passes.iter().find(|p| p.name == pass_name) {
                    queue.extend(&pass.reads);
                }
            }
        }
    }

    needed.into_iter().collect()
}

// 전체 계획 수립
fn compile(
    graph: &RenderGraph,
    output_resources: &[ResourceId],
) -> Result<ExecutionPlan, String> {
    let active_pass_names = cull_unused_passes(graph, output_resources);
    let active_passes: Vec<PassDesc> = graph.passes.iter()
        .filter(|p| active_pass_names.contains(&p.name))
        .cloned()
        .collect();

    let deps = build_dependency_graph(&RenderGraph {
        passes: active_passes.clone(),
        resources: graph.resources.clone(),
    });
    let ordered = topological_sort(&active_passes, &deps)?;

    let active_resources: Vec<ResourceId> = active_passes.iter()
        .flat_map(|p| p.reads.iter().chain(p.writes.iter()))
        .cloned()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    Ok(ExecutionPlan { ordered_passes: ordered, active_resources })
}
```

`compile`은 `RenderGraph`를 받아 `ExecutionPlan`을 반환합니다. GPU 없이 실행됩니다.

---

## 액션: GPU 실행

계획이 결정된 뒤에야 GPU가 등장합니다.

```rust
// 액션: 리소스(텍스처) 생성
fn create_gpu_resources(
    plan: &ExecutionPlan,
    graph: &RenderGraph,
    device: &Device,
) -> HashMap<ResourceId, Texture> {
    graph.resources.iter()
        .filter(|r| plan.active_resources.contains(&r.id))
        .map(|desc| {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                size: wgpu::Extent3d { width: desc.width, height: desc.height, depth_or_array_layers: 1 },
                // ...
                label: Some(&desc.id.0),
                ..Default::default()
            });
            (desc.id.clone(), texture)
        })
        .collect()
}

// 액션: 계획대로 패스 실행
fn execute_plan(
    plan: &ExecutionPlan,
    pass_executors: &HashMap<String, Box<dyn PassExecutor>>,
    resources: &HashMap<ResourceId, Texture>,
    device: &Device,
    queue: &Queue,
) {
    let mut encoder = device.create_command_encoder(&Default::default());
    for pass_name in &plan.ordered_passes {
        if let Some(executor) = pass_executors.get(pass_name) {
            executor.execute(&mut encoder, resources);
        }
    }
    queue.submit([encoder.finish()]);
}

trait PassExecutor {
    fn execute(&self, encoder: &mut CommandEncoder, resources: &HashMap<ResourceId, Texture>);
}
```

---

## 파이프라인 조합

```rust
fn render_frame(
    graph: &RenderGraph,
    output: &[ResourceId],
    pass_executors: &HashMap<String, Box<dyn PassExecutor>>,
    device: &Device,
    queue: &Queue,
) -> Result<(), String> {
    // ── 계산 (GPU 없이) ──────────────────────────────────
    let plan = compile(graph, output)?;

    // ── 액션 (GPU 경계) ──────────────────────────────────
    let resources = create_gpu_resources(&plan, graph, device);
    execute_plan(&plan, pass_executors, &resources, device, queue);

    Ok(())
}
```

데이터 흐름이 명확합니다.

```
RenderGraph
    │
    ├─ cull_unused_passes()   → 필요한 패스 목록    [계산]
    ├─ build_dependency_graph() → 의존 관계 맵      [계산]
    ├─ topological_sort()     → 실행 순서           [계산]
    └─ compile()              → ExecutionPlan       [계산]
                                      │
                              create_gpu_resources() [액션]
                              execute_plan()         [액션]
```

---

## 테스트: GPU 없이 그래프 로직 검증

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn res(name: &str) -> ResourceId { ResourceId(name.to_string()) }

    fn make_graph() -> RenderGraph {
        RenderGraph {
            passes: vec![
                PassDesc {
                    name: "shadow".to_string(),
                    reads: vec![],
                    writes: vec![res("shadow_map")],
                },
                PassDesc {
                    name: "gbuffer".to_string(),
                    reads: vec![],
                    writes: vec![res("gbuffer_color"), res("gbuffer_normal")],
                },
                PassDesc {
                    name: "lighting".to_string(),
                    reads: vec![res("shadow_map"), res("gbuffer_color"), res("gbuffer_normal")],
                    writes: vec![res("hdr_color")],
                },
                PassDesc {
                    name: "bloom".to_string(),
                    reads: vec![res("hdr_color")],
                    writes: vec![res("bloom_color")],
                },
                PassDesc {
                    name: "unused_pass".to_string(), // 최종 출력에 기여 안 함
                    reads: vec![],
                    writes: vec![res("unused_texture")],
                },
            ],
            resources: vec![],
        }
    }

    #[test]
    fn test_topological_order_respected() {
        let graph = make_graph();
        let deps = build_dependency_graph(&graph);
        let order = topological_sort(&graph.passes, &deps).unwrap();

        let pos = |name: &str| order.iter().position(|n| n == name).unwrap();

        // lighting은 shadow와 gbuffer 이후여야 함
        assert!(pos("shadow") < pos("lighting"));
        assert!(pos("gbuffer") < pos("lighting"));
        // bloom은 lighting 이후여야 함
        assert!(pos("lighting") < pos("bloom"));
    }

    #[test]
    fn test_unused_pass_culled() {
        let graph = make_graph();
        let output = vec![res("bloom_color")];
        let plan = compile(&graph, &output).unwrap();

        assert!(!plan.ordered_passes.contains(&"unused_pass".to_string()));
        assert!(plan.ordered_passes.contains(&"shadow".to_string()));
        assert!(plan.ordered_passes.contains(&"bloom".to_string()));
    }

    #[test]
    fn test_cyclic_dependency_detected() {
        let graph = RenderGraph {
            passes: vec![
                PassDesc {
                    name: "a".to_string(),
                    reads: vec![res("b_out")],
                    writes: vec![res("a_out")],
                },
                PassDesc {
                    name: "b".to_string(),
                    reads: vec![res("a_out")],
                    writes: vec![res("b_out")],
                },
            ],
            resources: vec![],
        };
        let deps = build_dependency_graph(&graph);
        let result = topological_sort(&graph.passes, &deps);
        assert!(result.is_err());
    }

    #[test]
    fn test_compile_returns_correct_resource_list() {
        let graph = make_graph();
        let output = vec![res("bloom_color")];
        let plan = compile(&graph, &output).unwrap();

        assert!(plan.active_resources.contains(&res("shadow_map")));
        assert!(plan.active_resources.contains(&res("hdr_color")));
        assert!(!plan.active_resources.contains(&res("unused_texture")));
    }
}
```

순환 의존 탐지, 패스 컬링, 실행 순서 검증을 GPU 없이 전부 확인합니다.

---

## 패스 추가: 그래프만 선언

TAA(Temporal Anti-Aliasing) 패스를 추가한다고 합시다.

```rust
// 선언만 추가
PassDesc {
    name: "taa".to_string(),
    reads: vec![res("hdr_color"), res("motion_vector")],
    writes: vec![res("taa_color")],
}
```

`compile()`이 자동으로 의존 관계를 분석해 실행 순서에 끼워 넣습니다. 기존 패스 코드를 건드리지 않습니다.

---

## Bevy·Unreal과의 비교

실제 엔진들도 같은 구조를 씁니다.

| 엔진 | 렌더 그래프 | 핵심 원칙 |
|------|-------------|-----------|
| Bevy | `RenderGraph` | 노드(패스) 선언 → 자동 순서 결정 |
| Unreal | `FRDGBuilder` | 리소스 선언 → 컴파일 → 실행 분리 |
| wgpu | 수동 | 순서를 직접 관리 |

액션/계산/데이터 구분이 엔진 설계 수준에서 적용된 사례입니다.

---

## 정리

| 분류 | 내용 | 특징 |
|------|------|------|
| 데이터 | `PassDesc`, `ResourceDesc`, `RenderGraph`, `ExecutionPlan` | GPU 무관, 선언적 |
| 계산 | `build_dependency_graph`, `topological_sort`, `cull_unused_passes`, `compile` | 순수 함수, GPU 없이 테스트 |
| 액션 | `create_gpu_resources`, `execute_plan` | GPU 경계에만 존재 |

패스 선언(데이터)과 실행 순서 결정(계산)을 분리하면, 패스를 추가·제거해도 나머지 코드를 건드리지 않습니다.

---

*관련 글: [액션, 계산, 데이터](/posts/programming/functional/functional-actions-calculations-data/), [함수형 셰이더 파이프라인](/posts/programming/functional/functional-shader-pipeline/), [함수형 포인트 클라우드 처리](/posts/programming/functional/functional-point-cloud/)*
