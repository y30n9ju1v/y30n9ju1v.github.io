---
title: "ROS2 콜백을 함수형으로: 메시지를 데이터로, 핸들러를 계산으로"
date: 2026-04-29T15:00:00+09:00
draft: false
tags: ["함수형 프로그래밍", "Rust", "설계", "자율주행", "로봇", "ROS2"]
categories: ["프로그래밍", "자율주행"]
description: "ROS2 콜백 안에 로직을 직접 쓰면 테스트하기 어려워집니다. 메시지를 데이터로, 처리 로직을 순수 계산으로 분리하면 ROS2 없이도 노드 로직 전체를 검증할 수 있습니다."
---

## 이 글을 읽고 나면

- ROS2 콜백의 어떤 부분이 액션이고 어떤 부분이 계산인지 구분할 수 있습니다.
- ROS2 없이 노드 로직을 단위 테스트하는 패턴을 이해합니다.
- 여러 토픽을 구독하는 노드를 함수형으로 설계하는 방법을 봅니다.

이전 글 [액션/계산/데이터](/posts/programming/functional/functional-actions-calculations-data/)와 [함수형 DI](/posts/programming/functional/functional-dependency-injection/)를 먼저 읽으면 더 자연스럽게 이어집니다.

---

## 문제: 콜백 안에 모든 것이 섞인다

ROS2 노드를 처음 짜면 콜백 함수 안에 모든 로직을 넣게 됩니다.

```rust
// rclrs 기반 pseudo-code
struct ObstacleDetectorNode {
    publisher: Arc<Publisher<ObstacleArray>>,
    last_lidar: Option<PointCloud2>,
}

impl ObstacleDetectorNode {
    fn on_lidar(&mut self, msg: PointCloud2) {
        // 파싱 — 계산인가 액션인가?
        let points = parse_pointcloud(&msg);

        // 필터링 — 계산
        let filtered: Vec<Point3D> = points.iter()
            .filter(|p| p.distance < 80.0)
            .cloned()
            .collect();

        // 클러스터링 — 계산
        let clusters = dbscan(&filtered);

        // 퍼블리시 — 액션
        let obstacle_msg = clusters_to_msg(&clusters);
        self.publisher.publish(obstacle_msg).unwrap(); // 네트워크 I/O

        // 상태 저장 — 액션
        self.last_lidar = Some(msg);
    }
}
```

이 콜백을 테스트하려면 `rclrs` 런타임과 `Publisher` 목(mock)이 필요합니다. `dbscan`의 파라미터 하나를 바꾸려고 해도 전체 노드를 띄워야 합니다.

---

## 세 가지로 나누기

### 데이터: 메시지를 도메인 타입으로

ROS2 메시지 타입(`PointCloud2`, `ObstacleArray`)은 직렬화 포맷입니다. 처리 로직에는 도메인 구조체를 씁니다.

```rust
// 도메인 데이터 — ROS2 의존 없음
#[derive(Debug, Clone)]
pub struct Point3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub intensity: f32,
    pub distance: f32,
}

#[derive(Debug, Clone)]
pub struct DetectedObstacle {
    pub center: Point3D,
    pub point_count: usize,
    pub distance_to_ego: f32,
}

#[derive(Debug, Clone)]
pub struct ObstacleDetectionResult {
    pub obstacles: Vec<DetectedObstacle>,
    pub timestamp_ns: u64,
}
```

---

### 계산: 순수한 처리 로직

ROS2 메시지를 도메인 타입으로 변환하는 파싱도 계산입니다. 네트워크나 하드웨어에 접근하지 않기 때문입니다.

```rust
// 파싱: ROS 메시지 → 도메인 데이터 (계산)
pub fn parse_pointcloud(raw: &[u8], timestamp_ns: u64) -> Vec<Point3D> {
    // 실제로는 PointCloud2 바이트 파싱
    raw.chunks(16).filter_map(|chunk| {
        if chunk.len() < 16 { return None; }
        let x = f32::from_le_bytes(chunk[0..4].try_into().ok()?);
        let y = f32::from_le_bytes(chunk[4..8].try_into().ok()?);
        let z = f32::from_le_bytes(chunk[8..12].try_into().ok()?);
        let intensity = f32::from_le_bytes(chunk[12..16].try_into().ok()?);
        let distance = (x * x + y * y + z * z).sqrt();
        Some(Point3D { x, y, z, intensity, distance })
    }).collect()
}

// 필터링 (계산)
pub fn filter_points(points: &[Point3D], max_distance: f32, min_intensity: f32) -> Vec<Point3D> {
    points.iter()
        .filter(|p| p.distance < max_distance && p.intensity > min_intensity)
        .cloned()
        .collect()
}

// 클러스터링 (계산)
pub fn cluster_obstacles(points: &[Point3D], cluster_radius: f32) -> Vec<DetectedObstacle> {
    let mut obstacles: Vec<DetectedObstacle> = Vec::new();

    for point in points {
        let merged = obstacles.iter_mut().find(|o| {
            let dx = o.center.x - point.x;
            let dy = o.center.y - point.y;
            (dx * dx + dy * dy).sqrt() < cluster_radius
        });

        if let Some(obs) = merged {
            let n = obs.point_count as f32;
            obs.center.x = (obs.center.x * n + point.x) / (n + 1.0);
            obs.center.y = (obs.center.y * n + point.y) / (n + 1.0);
            obs.center.z = (obs.center.z * n + point.z) / (n + 1.0);
            obs.distance_to_ego = (obs.center.x.powi(2) + obs.center.y.powi(2)).sqrt();
            obs.point_count += 1;
        } else {
            obstacles.push(DetectedObstacle {
                center: point.clone(),
                point_count: 1,
                distance_to_ego: point.distance,
            });
        }
    }
    obstacles
}

// 직렬화: 도메인 데이터 → ROS 메시지 포맷 (계산)
pub fn serialize_obstacles(result: &ObstacleDetectionResult) -> Vec<u8> {
    // 실제로는 ObstacleArray 메시지 직렬화
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&result.timestamp_ns.to_le_bytes());
    bytes.extend_from_slice(&(result.obstacles.len() as u32).to_le_bytes());
    bytes
}

// 파이프라인 전체 (계산들의 조합)
pub fn process_lidar(
    raw: &[u8],
    timestamp_ns: u64,
    max_distance: f32,
    min_intensity: f32,
    cluster_radius: f32,
) -> ObstacleDetectionResult {
    let points   = parse_pointcloud(raw, timestamp_ns);
    let filtered = filter_points(&points, max_distance, min_intensity);
    let obstacles = cluster_obstacles(&filtered, cluster_radius);
    ObstacleDetectionResult { obstacles, timestamp_ns }
}
```

`process_lidar`는 바이트 슬라이스를 받아 도메인 결과를 돌려줍니다. ROS2 타입도, 네트워크도 없습니다.

---

### 액션: 콜백은 얇은 껍데기

콜백은 I/O만 담당합니다. 로직은 없습니다.

```rust
struct ObstacleDetectorNode {
    publisher: Arc<Publisher>,
    // 파라미터
    max_distance:   f32,
    min_intensity:  f32,
    cluster_radius: f32,
}

impl ObstacleDetectorNode {
    fn on_lidar(&self, raw_bytes: &[u8], timestamp_ns: u64) {
        // 계산: ROS2 런타임과 무관
        let result = process_lidar(
            raw_bytes, timestamp_ns,
            self.max_distance,
            self.min_intensity,
            self.cluster_radius,
        );

        // 액션: 직렬화 후 퍼블리시
        let msg_bytes = serialize_obstacles(&result);
        self.publisher.publish(&msg_bytes).unwrap();
    }
}
```

콜백이 하는 일: 바이트를 받아 → 계산 함수에 넘기고 → 결과를 퍼블리시. 세 줄입니다.

---

## 여러 토픽을 구독할 때

자율주행 노드는 보통 여러 토픽을 구독하고 결과를 퓨전합니다. 함수형 패턴은 이 경우에도 자연스럽게 확장됩니다.

```rust
#[derive(Debug, Clone)]
pub struct FusionInput {
    pub lidar_result: Option<ObstacleDetectionResult>,
    pub radar_result: Option<RadarDetectionResult>,
    pub camera_result: Option<CameraDetectionResult>,
}

// 퓨전: 세 소스를 합쳐서 최종 장애물 목록 생성 (계산)
pub fn fuse_detections(input: &FusionInput) -> Vec<DetectedObstacle> {
    let mut all: Vec<DetectedObstacle> = Vec::new();

    if let Some(lidar) = &input.lidar_result {
        all.extend(lidar.obstacles.clone());
    }
    if let Some(radar) = &input.radar_result {
        all.extend(radar.obstacles.iter().map(|o| o.into()));
    }
    if let Some(camera) = &input.camera_result {
        all.extend(camera.obstacles.iter().map(|o| o.into()));
    }

    // 중복 제거: 가까운 장애물은 하나로 합침
    deduplicate_obstacles(all, 2.0)
}

pub fn deduplicate_obstacles(
    mut obstacles: Vec<DetectedObstacle>,
    merge_radius: f32,
) -> Vec<DetectedObstacle> {
    obstacles.sort_by(|a, b| b.point_count.cmp(&a.point_count));
    let mut result: Vec<DetectedObstacle> = Vec::new();

    for obs in obstacles {
        let duplicate = result.iter().any(|r| {
            let dx = r.center.x - obs.center.x;
            let dy = r.center.y - obs.center.y;
            (dx * dx + dy * dy).sqrt() < merge_radius
        });
        if !duplicate {
            result.push(obs);
        }
    }
    result
}

// 퓨전 노드: 각 콜백은 내부 상태를 업데이트하고 퓨전 계산을 트리거
struct FusionNode {
    publisher:      Arc<Publisher>,
    lidar_result:   Option<ObstacleDetectionResult>,
    radar_result:   Option<RadarDetectionResult>,
    camera_result:  Option<CameraDetectionResult>,
}

impl FusionNode {
    fn on_lidar(&mut self, raw: &[u8], ts: u64) {
        self.lidar_result = Some(process_lidar(raw, ts, 80.0, 10.0, 1.5)); // 계산
        self.publish_fused(); // 액션
    }

    fn on_radar(&mut self, raw: &[u8]) {
        self.radar_result = Some(process_radar(raw)); // 계산
        self.publish_fused(); // 액션
    }

    fn publish_fused(&self) {
        let input = FusionInput {
            lidar_result:  self.lidar_result.clone(),
            radar_result:  self.radar_result.clone(),
            camera_result: self.camera_result.clone(),
        };
        let fused = fuse_detections(&input); // 계산
        let bytes = serialize_fused(&fused); // 계산
        self.publisher.publish(&bytes).unwrap(); // 액션
    }
}
```

`fuse_detections`와 `deduplicate_obstacles`는 `FusionInput`만 있으면 ROS2 없이 테스트됩니다.

---

## 테스트: ROS2 없이

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn make_obstacle(x: f32, y: f32, count: usize) -> DetectedObstacle {
        DetectedObstacle {
            center: Point3D { x, y, z: 0.0, intensity: 50.0,
                              distance: (x*x + y*y).sqrt() },
            point_count: count,
            distance_to_ego: (x*x + y*y).sqrt(),
        }
    }

    #[test]
    fn test_filter_removes_far_points() {
        let points = vec![
            Point3D { x: 1.0, y: 0.0, z: 0.0, intensity: 50.0, distance: 1.0 },
            Point3D { x: 0.0, y: 0.0, z: 0.0, intensity: 50.0, distance: 90.0 },
        ];
        let filtered = filter_points(&points, 80.0, 10.0);
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn test_deduplication_merges_nearby() {
        let obstacles = vec![
            make_obstacle(0.0, 0.0, 10),
            make_obstacle(0.5, 0.5, 5), // 2m 이내 → 중복
            make_obstacle(10.0, 0.0, 8), // 멀리 있음 → 별도
        ];
        let result = deduplicate_obstacles(obstacles, 2.0);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_fusion_with_only_lidar() {
        let input = FusionInput {
            lidar_result: Some(ObstacleDetectionResult {
                obstacles: vec![make_obstacle(5.0, 0.0, 20)],
                timestamp_ns: 0,
            }),
            radar_result: None,
            camera_result: None,
        };
        let result = fuse_detections(&input);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_full_pipeline_with_raw_bytes() {
        // 포인트 하나짜리 더미 PointCloud2 바이트
        let mut raw = Vec::new();
        raw.extend_from_slice(&5.0_f32.to_le_bytes()); // x
        raw.extend_from_slice(&0.0_f32.to_le_bytes()); // y
        raw.extend_from_slice(&0.0_f32.to_le_bytes()); // z
        raw.extend_from_slice(&50.0_f32.to_le_bytes()); // intensity

        let result = process_lidar(&raw, 0, 80.0, 10.0, 1.5);
        assert_eq!(result.obstacles.len(), 1);
        assert!((result.obstacles[0].distance_to_ego - 5.0).abs() < 0.01);
    }
}
```

`rclrs`, `ros2_launch`, 토픽 연결 — 아무것도 없이 파이프라인 전체를 검증합니다.

---

## 계층 정리

```
┌────────────────────────────────────────────┐
│            ROS2 콜백 레이어                 │  ← on_lidar, on_radar
│  (메시지 수신, 퍼블리시 — 액션만)           │
├────────────────────────────────────────────┤
│            파이프라인 레이어                │  ← process_lidar, fuse_detections
│  (parse → filter → cluster → fuse — 계산) │
├────────────────────────────────────────────┤
│            도메인 데이터 레이어             │  ← Point3D, DetectedObstacle
│  (ROS2 메시지와 분리된 구조체)             │
└────────────────────────────────────────────┘
```

ROS2가 교체되거나 버전이 바뀌어도 파이프라인 레이어와 도메인 데이터 레이어는 건드리지 않습니다. 콜백만 수정합니다.

---

## 정리

ROS2 노드를 함수형으로 설계하는 핵심 원칙:

1. **메시지는 바이트 → 도메인 구조체로 즉시 변환** — 콜백 안에서 ROS 타입으로 로직을 짜지 않음
2. **처리 로직은 순수 계산으로** — `Publisher`도, `Node`도 인자로 받지 않음
3. **콜백은 세 줄** — 수신, 계산 호출, 퍼블리시
4. **퓨전도 계산** — 여러 토픽의 결과를 합치는 로직도 `FusionInput` 구조체만 받는 순수 함수

이 패턴을 지키면 ROS2 빌드 없이 `cargo test`만으로 노드 로직 전체를 검증할 수 있습니다. CI 파이프라인에서 도커 이미지 없이 수백 개의 시나리오를 돌릴 수 있습니다.

---

*관련 글: [액션/계산/데이터](/posts/programming/functional/functional-actions-calculations-data/), [함수형 DI](/posts/programming/functional/functional-dependency-injection/), [자율주행 센서 파이프라인](/posts/programming/functional/autonomous-sensor-pipeline/), [함수형 센서 퓨전](/posts/programming/functional/autonomous-sensor-fusion/), [함수형 PID 제어기](/posts/programming/functional/autonomous-pid-controller/), [시뮬레이션 회귀 테스트 설계](/posts/programming/functional/autonomous-simulation-regression/)*
