---
title: "함수형 포인트 클라우드 처리: LiDAR 파이프라인을 순수 계산으로"
date: 2026-04-30T11:00:00+09:00
draft: false
tags: ["함수형 프로그래밍", "Rust", "설계", "로봇", "자율주행", "LiDAR", "포인트 클라우드", "액션/계산/데이터"]
categories: ["프로그래밍", "자율주행"]
description: "LiDAR 원시 데이터를 다운샘플링·노이즈 제거·클러스터링으로 처리하는 파이프라인을 액션/계산/데이터로 나누면, 하드웨어 없이 각 단계를 독립적으로 테스트할 수 있습니다."
---

## 이 글을 읽고 나면

- LiDAR 포인트 클라우드 처리 파이프라인을 액션/계산/데이터로 나누는 방법을 이해합니다.
- 각 처리 단계(다운샘플링, 노이즈 제거, 클러스터링)가 왜 순수 계산인지 봅니다.
- 하드웨어 없이 파이프라인 전체를 테스트하는 방법을 이해합니다.

이전 글 [액션, 계산, 데이터](/posts/programming/functional-actions-calculations-data/)와 [자율주행 센서 파이프라인](/posts/programming/autonomous-sensor-pipeline/)을 먼저 읽으면 더 자연스럽게 이어집니다.

---

## 문제: LiDAR 처리 코드가 뒤섞이면

LiDAR는 초당 수십만 개의 3D 포인트를 쏟아냅니다. 이 데이터를 그대로 쓰면 너무 많고, 노이즈가 섞여 있고, 개별 물체가 구분되지 않습니다. 보통 이런 흐름으로 처리합니다.

```
원시 포인트 클라우드
  → 다운샘플링 (voxel grid)
  → 노이즈 제거 (statistical outlier removal)
  → 클러스터링 (DBSCAN 등)
  → 물체 후보 목록
```

이걸 하나의 함수에 몰아넣으면 이렇게 됩니다.

```rust
fn process(lidar: &LidarDevice, obstacles: &mut Vec<Obstacle>) {
    let raw = lidar.read_scan();          // 하드웨어 I/O
    let downsampled = voxel_grid(&raw);   // 계산
    let filtered = remove_outliers(&downsampled); // 계산
    let clusters = dbscan(&filtered);     // 계산
    *obstacles = clusters.into_iter().map(Obstacle::from).collect(); // 계산 + 상태 변경
}
```

알고리즘 파라미터(voxel 크기, DBSCAN epsilon)를 바꾸면 전체 함수를 건드려야 합니다. 하드웨어 없이 클러스터링만 테스트하려면 `LidarDevice`를 mock해야 합니다. 각 단계의 중간 결과를 검사할 방법이 없습니다.

---

## 액션/계산/데이터로 나누기

```
데이터(Data)              계산(Calculation)                액션(Action)
────────────              ─────────────────                ─────────────
PointCloud           ←─  voxel_grid(cloud, config)   ←─  lidar.read_scan()
VoxelConfig              remove_outliers(cloud, config)
DbscanConfig         →   dbscan(cloud, config)        →   publish_obstacles(obstacles)
Cluster                  extract_obstacles(clusters)
Obstacle
```

계산은 `PointCloud`와 설정값만 있으면 실행됩니다. 하드웨어가 없어도 됩니다.

---

## 데이터: 포인트 클라우드와 설정

```rust
#[derive(Debug, Clone)]
struct Point3D {
    x: f32,
    y: f32,
    z: f32,
    intensity: f32, // 반사 강도
}

#[derive(Debug, Clone)]
struct PointCloud {
    points: Vec<Point3D>,
    timestamp_ms: u64,
}

#[derive(Debug, Clone)]
struct VoxelConfig {
    voxel_size: f32, // 미터 단위 격자 크기
}

#[derive(Debug, Clone)]
struct OutlierConfig {
    k_neighbors: usize, // 통계 기준 이웃 수
    std_dev_threshold: f32, // 표준편차 배수
}

#[derive(Debug, Clone)]
struct DbscanConfig {
    epsilon: f32,    // 이웃 반경 (미터)
    min_points: usize, // 클러스터 최소 포인트 수
}

#[derive(Debug, Clone)]
struct Cluster {
    points: Vec<Point3D>,
}

#[derive(Debug, Clone)]
struct Obstacle {
    center: Point3D,
    size: [f32; 3], // x, y, z 크기
    point_count: usize,
}
```

모든 구조체는 하드웨어와 무관합니다. 파일로 저장하고 테스트 데이터로 재사용할 수 있습니다.

---

## 계산: 각 처리 단계

### 다운샘플링: Voxel Grid

공간을 격자로 나누고 각 격자 안의 포인트를 하나로 합칩니다.

```rust
fn voxel_grid(cloud: &PointCloud, config: &VoxelConfig) -> PointCloud {
    use std::collections::HashMap;

    let inv = 1.0 / config.voxel_size;
    let mut voxels: HashMap<(i32, i32, i32), Vec<&Point3D>> = HashMap::new();

    for p in &cloud.points {
        let key = (
            (p.x * inv).floor() as i32,
            (p.y * inv).floor() as i32,
            (p.z * inv).floor() as i32,
        );
        voxels.entry(key).or_default().push(p);
    }

    let points = voxels.values().map(|pts| {
        let n = pts.len() as f32;
        Point3D {
            x: pts.iter().map(|p| p.x).sum::<f32>() / n,
            y: pts.iter().map(|p| p.y).sum::<f32>() / n,
            z: pts.iter().map(|p| p.z).sum::<f32>() / n,
            intensity: pts.iter().map(|p| p.intensity).sum::<f32>() / n,
        }
    }).collect();

    PointCloud { points, timestamp_ms: cloud.timestamp_ms }
}
```

### 노이즈 제거: Statistical Outlier Removal

각 포인트의 이웃 거리 분포를 계산해, 평균에서 멀리 떨어진 포인트를 제거합니다.

```rust
fn remove_outliers(cloud: &PointCloud, config: &OutlierConfig) -> PointCloud {
    let points = &cloud.points;
    if points.len() <= config.k_neighbors {
        return cloud.clone();
    }

    // 각 포인트의 k-최근접 이웃 평균 거리 계산
    let mean_dists: Vec<f32> = points.iter().map(|p| {
        let mut dists: Vec<f32> = points.iter()
            .filter(|q| !std::ptr::eq(*q, p))
            .map(|q| dist(p, q))
            .collect();
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
        dists[..config.k_neighbors].iter().sum::<f32>() / config.k_neighbors as f32
    }).collect();

    let global_mean = mean_dists.iter().sum::<f32>() / mean_dists.len() as f32;
    let variance = mean_dists.iter().map(|d| (d - global_mean).powi(2)).sum::<f32>()
        / mean_dists.len() as f32;
    let std_dev = variance.sqrt();
    let threshold = global_mean + config.std_dev_threshold * std_dev;

    let filtered = points.iter().zip(mean_dists.iter())
        .filter(|(_, &d)| d <= threshold)
        .map(|(p, _)| p.clone())
        .collect();

    PointCloud { points: filtered, timestamp_ms: cloud.timestamp_ms }
}

fn dist(a: &Point3D, b: &Point3D) -> f32 {
    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.z - b.z).powi(2)).sqrt()
}
```

### 클러스터링: DBSCAN

밀도 기반으로 가까운 포인트들을 같은 클러스터로 묶습니다.

```rust
fn dbscan(cloud: &PointCloud, config: &DbscanConfig) -> Vec<Cluster> {
    let points = &cloud.points;
    let n = points.len();
    let mut labels = vec![-1i32; n]; // -1: 미분류, -2: 노이즈
    let mut cluster_id = 0i32;

    for i in 0..n {
        if labels[i] != -1 { continue; }

        let neighbors = range_query(points, i, config.epsilon);
        if neighbors.len() < config.min_points {
            labels[i] = -2; // 노이즈
            continue;
        }

        labels[i] = cluster_id;
        let mut seeds: Vec<usize> = neighbors;

        let mut j = 0;
        while j < seeds.len() {
            let q = seeds[j];
            if labels[q] == -2 { labels[q] = cluster_id; }
            if labels[q] != -1 { j += 1; continue; }
            labels[q] = cluster_id;
            let q_neighbors = range_query(points, q, config.epsilon);
            if q_neighbors.len() >= config.min_points {
                seeds.extend(q_neighbors);
            }
            j += 1;
        }
        cluster_id += 1;
    }

    (0..cluster_id).map(|id| {
        let pts = points.iter().enumerate()
            .filter(|(i, _)| labels[*i] == id)
            .map(|(_, p)| p.clone())
            .collect();
        Cluster { points: pts }
    }).collect()
}

fn range_query(points: &[Point3D], idx: usize, epsilon: f32) -> Vec<usize> {
    points.iter().enumerate()
        .filter(|(i, q)| *i != idx && dist(&points[idx], q) <= epsilon)
        .map(|(i, _)| i)
        .collect()
}

// 클러스터 → 장애물 변환
fn extract_obstacles(clusters: &[Cluster]) -> Vec<Obstacle> {
    clusters.iter().map(|c| {
        let n = c.points.len() as f32;
        let cx = c.points.iter().map(|p| p.x).sum::<f32>() / n;
        let cy = c.points.iter().map(|p| p.y).sum::<f32>() / n;
        let cz = c.points.iter().map(|p| p.z).sum::<f32>() / n;

        let size_x = c.points.iter().map(|p| p.x).fold(f32::NEG_INFINITY, f32::max)
                   - c.points.iter().map(|p| p.x).fold(f32::INFINITY, f32::min);
        let size_y = c.points.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max)
                   - c.points.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
        let size_z = c.points.iter().map(|p| p.z).fold(f32::NEG_INFINITY, f32::max)
                   - c.points.iter().map(|p| p.z).fold(f32::INFINITY, f32::min);

        Obstacle {
            center: Point3D { x: cx, y: cy, z: cz, intensity: 0.0 },
            size: [size_x, size_y, size_z],
            point_count: c.points.len(),
        }
    }).collect()
}
```

세 함수 모두 `PointCloud`와 설정값만 받습니다. 하드웨어, 전역 상태, 타이머 없음.

---

## 액션: 하드웨어 경계에서만

```rust
// 액션: LiDAR에서 원시 데이터 읽기
fn read_lidar_scan(device: &LidarDevice) -> PointCloud {
    let raw = device.read_scan();
    PointCloud {
        points: raw.into_iter().map(|p| Point3D {
            x: p.x, y: p.y, z: p.z, intensity: p.intensity,
        }).collect(),
        timestamp_ms: device.timestamp_ms(),
    }
}

// 액션: 장애물 목록을 다운스트림으로 발행
fn publish_obstacles(obstacles: &[Obstacle], publisher: &ObstaclePublisher) {
    publisher.send(obstacles);
}
```

---

## 파이프라인 조합

```rust
struct PipelineConfig {
    voxel: VoxelConfig,
    outlier: OutlierConfig,
    dbscan: DbscanConfig,
}

fn run_pipeline(
    device: &LidarDevice,
    config: &PipelineConfig,
    publisher: &ObstaclePublisher,
) {
    // 액션: 원시 데이터 수집
    let raw = read_lidar_scan(device);

    // 계산: 처리 파이프라인
    let downsampled = voxel_grid(&raw, &config.voxel);
    let filtered    = remove_outliers(&downsampled, &config.outlier);
    let clusters    = dbscan(&filtered, &config.dbscan);
    let obstacles   = extract_obstacles(&clusters);

    // 액션: 결과 발행
    publish_obstacles(&obstacles, publisher);
}
```

데이터 흐름이 선형입니다.

```
read_lidar_scan()   → PointCloud (raw)         [액션]
voxel_grid()        → PointCloud (downsampled) [계산]
remove_outliers()   → PointCloud (filtered)    [계산]
dbscan()            → Vec<Cluster>             [계산]
extract_obstacles() → Vec<Obstacle>            [계산]
publish_obstacles() →                          [액션]
```

---

## 테스트: 하드웨어 없이

각 단계가 순수 함수이므로 데이터를 직접 만들어 검증합니다.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn make_cloud(pts: Vec<(f32, f32, f32)>) -> PointCloud {
        PointCloud {
            points: pts.into_iter().map(|(x, y, z)| Point3D { x, y, z, intensity: 1.0 }).collect(),
            timestamp_ms: 0,
        }
    }

    #[test]
    fn test_voxel_grid_reduces_point_count() {
        // 같은 voxel 안에 여러 포인트 → 하나로 합쳐짐
        let cloud = make_cloud(vec![
            (0.0, 0.0, 0.0), (0.05, 0.05, 0.05), // 같은 voxel (size=0.1)
            (1.0, 0.0, 0.0),                       // 다른 voxel
        ]);
        let config = VoxelConfig { voxel_size: 0.1 };
        let result = voxel_grid(&cloud, &config);
        assert_eq!(result.points.len(), 2);
    }

    #[test]
    fn test_dbscan_finds_two_clusters() {
        // 두 그룹이 멀리 떨어진 포인트 클라우드
        let mut pts = vec![];
        for i in 0..5 { pts.push((i as f32 * 0.1, 0.0, 0.0)); } // 클러스터 1
        for i in 0..5 { pts.push((10.0 + i as f32 * 0.1, 0.0, 0.0)); } // 클러스터 2
        let cloud = make_cloud(pts);
        let config = DbscanConfig { epsilon: 0.2, min_points: 2 };
        let clusters = dbscan(&cloud, &config);
        assert_eq!(clusters.len(), 2);
    }

    #[test]
    fn test_extract_obstacles_center() {
        let cluster = Cluster {
            points: vec![
                Point3D { x: 0.0, y: 0.0, z: 0.0, intensity: 1.0 },
                Point3D { x: 2.0, y: 0.0, z: 0.0, intensity: 1.0 },
            ],
        };
        let obstacles = extract_obstacles(&[cluster]);
        assert_eq!(obstacles.len(), 1);
        assert!((obstacles[0].center.x - 1.0).abs() < 1e-5);
        assert!((obstacles[0].size[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_pipeline_config_independence() {
        // 같은 데이터, 다른 voxel 크기 → 포인트 수가 달라야 함
        let cloud = make_cloud(vec![
            (0.0, 0.0, 0.0), (0.05, 0.0, 0.0), (0.5, 0.0, 0.0),
        ]);
        let coarse = voxel_grid(&cloud, &VoxelConfig { voxel_size: 0.2 });
        let fine   = voxel_grid(&cloud, &VoxelConfig { voxel_size: 0.01 });
        assert!(coarse.points.len() <= fine.points.len());
    }
}
```

LiDAR 장비 없이 CI에서 전부 실행됩니다.

---

## 중간 결과 기록과 재현

계산 함수들이 순수하므로 중간 결과를 파일로 덤프하고 나중에 재현할 수 있습니다.

```rust
fn run_pipeline_with_logging(
    device: &LidarDevice,
    config: &PipelineConfig,
    publisher: &ObstaclePublisher,
    log_dir: Option<&str>,
) {
    let raw = read_lidar_scan(device);

    let downsampled = voxel_grid(&raw, &config.voxel);
    let filtered    = remove_outliers(&downsampled, &config.outlier);
    let clusters    = dbscan(&filtered, &config.dbscan);
    let obstacles   = extract_obstacles(&clusters);

    if let Some(dir) = log_dir {
        save_cloud(&raw, &format!("{dir}/raw.bin"));
        save_cloud(&filtered, &format!("{dir}/filtered.bin"));
        save_obstacles(&obstacles, &format!("{dir}/obstacles.json"));
    }

    publish_obstacles(&obstacles, publisher);
}
```

버그가 생기면 `raw.bin`을 읽어서 계산 함수만 다시 돌리면 재현됩니다. 실차가 없어도 됩니다.

---

## 정리

| 분류 | 내용 | 특징 |
|------|------|------|
| 데이터 | `PointCloud`, `VoxelConfig`, `DbscanConfig`, `Cluster`, `Obstacle` | 하드웨어 무관, 파일 저장 가능 |
| 계산 | `voxel_grid`, `remove_outliers`, `dbscan`, `extract_obstacles` | 순수 함수, 단계별 독립 테스트 |
| 액션 | `read_lidar_scan`, `publish_obstacles` | LiDAR 경계에만 존재 |

계산이 두껍고 액션이 얇을수록, 실차 없이 재현하고 개선하기 쉬워집니다.

---

*관련 글: [액션, 계산, 데이터](/posts/programming/functional-actions-calculations-data/), [자율주행 센서 파이프라인](/posts/programming/autonomous-sensor-pipeline/), [함수형 센서 퓨전](/posts/programming/autonomous-sensor-fusion/), [함수형 렌더 그래프](/posts/programming/functional-render-graph/)*
