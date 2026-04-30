---
title: "로봇 경로 계획과 불변 데이터: A*를 함수형으로 구현하기"
date: 2026-04-29T12:00:00+09:00
draft: false
tags: ["함수형 프로그래밍", "Rust", "설계", "자율주행", "로봇", "경로 계획", "불변 데이터", "A*"]
categories: ["프로그래밍", "자율주행"]
description: "A* 경로 탐색을 불변 데이터 구조로 구현하면, 탐색 트리를 공유하면서 여러 경로를 동시에 평가하고 이전 탐색 결과를 재활용할 수 있습니다."
---

## 이 글을 읽고 나면

- A* 탐색에서 불변 데이터가 왜 유리한지 이해합니다.
- 경로 트리를 구조적 공유로 표현하는 방법을 봅니다.
- 여러 목표 지점을 탐색할 때 공통 부분 트리를 재활용하는 설계를 이해합니다.

이전 글 [불변 데이터와 구조적 공유](/posts/programming/functional/functional-immutable-data/)를 먼저 읽으면 더 자연스럽게 이어집니다.

---

## A*와 경로 계획

A*는 출발지에서 목적지까지 최단 경로를 찾는 알고리즘입니다. 자율주행과 로봇공학에서 가장 널리 쓰이는 경로 계획 알고리즘 중 하나입니다.

전형적인 구현은 이렇게 생겼습니다.

```
open_set = {start}
came_from = {}  // 어떤 노드에서 왔는가

while open_set이 비어있지 않으면:
    current = open_set에서 f값이 가장 낮은 노드
    if current == goal: 경로 복원하고 반환

    for neighbor in current의 이웃:
        tentative_g = g[current] + cost(current, neighbor)
        if tentative_g < g[neighbor]:
            came_from[neighbor] = current  // 상태를 변경!
            g[neighbor] = tentative_g
            open_set에 neighbor 추가
```

`came_from`과 `g` 딕셔너리를 계속 변경합니다. 변경 가능한 상태가 알고리즘 중심에 있습니다.

이 방식의 문제는 탐색 도중의 중간 상태를 보존하기 어렵다는 것입니다. 목표 지점이 바뀌면 처음부터 다시 탐색해야 합니다. 여러 경로 후보를 동시에 유지하려면 구조를 통째로 복사해야 합니다.

---

## 데이터 정의

먼저 지도와 경로를 표현하는 구조체를 정의합니다.

```rust
use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;
use std::rc::Rc;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GridPos {
    row: i32,
    col: i32,
}

impl GridPos {
    fn new(row: i32, col: i32) -> Self { Self { row, col } }

    fn neighbors(&self) -> Vec<GridPos> {
        vec![
            GridPos::new(self.row - 1, self.col),
            GridPos::new(self.row + 1, self.col),
            GridPos::new(self.row, self.col - 1),
            GridPos::new(self.row, self.col + 1),
        ]
    }

    fn manhattan_distance(&self, other: &GridPos) -> u32 {
        ((self.row - other.row).abs() + (self.col - other.col).abs()) as u32
    }
}

struct Grid {
    width: usize,
    height: usize,
    obstacles: Vec<Vec<bool>>, // true면 장애물
}

impl Grid {
    fn is_passable(&self, pos: &GridPos) -> bool {
        pos.row >= 0 && pos.col >= 0
            && (pos.row as usize) < self.height
            && (pos.col as usize) < self.width
            && !self.obstacles[pos.row as usize][pos.col as usize]
    }
}
```

---

## 경로를 불변 연결 리스트로

경로 추적의 핵심 아이디어: 각 노드가 "나는 어디서 왔는가"를 `Rc`(참조 카운트 포인터)로 가리키게 합니다.

```rust
#[derive(Debug, Clone)]
struct PathNode {
    pos: GridPos,
    parent: Option<Rc<PathNode>>, // 불변 참조
    g_cost: u32,
    f_cost: u32,
}

impl PathNode {
    fn root(pos: GridPos) -> Rc<Self> {
        Rc::new(Self { pos, parent: None, g_cost: 0, f_cost: 0 })
    }

    fn child(parent: &Rc<PathNode>, pos: GridPos, g_cost: u32, h_cost: u32) -> Rc<Self> {
        Rc::new(Self {
            pos,
            parent: Some(Rc::clone(parent)), // 부모를 복사하지 않고 참조
            g_cost,
            f_cost: g_cost + h_cost,
        })
    }

    fn reconstruct_path(&self) -> Vec<GridPos> {
        let mut path = vec![self.pos.clone()];
        let mut current = self.parent.clone();
        while let Some(node) = current {
            path.push(node.pos.clone());
            current = node.parent.clone();
        }
        path.reverse();
        path
    }
}
```

`Rc::clone`은 포인터만 복사합니다. 부모 노드들의 데이터는 복사하지 않습니다. 여러 경로 후보가 공통 조상을 **공유**합니다.

```
start ─→ A ─→ B ─→ C (후보 1)
                └─→ D (후보 2)
```

`B`를 복사하지 않고, `C`와 `D` 모두 `B`를 가리킵니다. 이것이 구조적 공유입니다.

---

## A* 탐색: 순수 계산으로

```rust
// BinaryHeap을 위한 역순 비교 (최솟값 우선)
impl PartialEq for PathNode {
    fn eq(&self, other: &Self) -> bool { self.f_cost == other.f_cost }
}
impl Eq for PathNode {}
impl PartialOrd for PathNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for PathNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other.f_cost.cmp(&self.f_cost) // 역순: 낮은 f_cost가 우선
    }
}

fn astar(grid: &Grid, start: GridPos, goal: &GridPos) -> Option<Vec<GridPos>> {
    let mut open_set: BinaryHeap<Rc<PathNode>> = BinaryHeap::new();
    let mut best_g: HashMap<GridPos, u32> = HashMap::new();

    let start_node = PathNode::root(start.clone());
    best_g.insert(start, 0);
    open_set.push(start_node);

    while let Some(current) = open_set.pop() {
        if &current.pos == goal {
            return Some(current.reconstruct_path());
        }

        // 이미 더 좋은 경로를 찾았으면 스킵
        if best_g.get(&current.pos).copied().unwrap_or(u32::MAX) < current.g_cost {
            continue;
        }

        for neighbor_pos in current.pos.neighbors() {
            if !grid.is_passable(&neighbor_pos) {
                continue;
            }

            let new_g = current.g_cost + 1;
            if new_g < best_g.get(&neighbor_pos).copied().unwrap_or(u32::MAX) {
                best_g.insert(neighbor_pos.clone(), new_g);
                let h = neighbor_pos.manhattan_distance(goal);
                let child = PathNode::child(&current, neighbor_pos, new_g, h);
                open_set.push(child);
            }
        }
    }

    None // 경로 없음
}
```

`astar`는 `grid`, `start`, `goal`을 받아 경로를 돌려줍니다. 전역 상태도, 외부 의존도 없습니다.

---

## 여러 목표를 동시에 탐색하기

불변 구조의 진가가 드러나는 시나리오입니다. 주차 공간 탐색처럼 여러 목표 지점 중 가장 좋은 경로를 찾아야 할 때, 공통 탐색 구간을 재활용할 수 있습니다.

```rust
fn find_best_path(
    grid: &Grid,
    start: GridPos,
    goals: &[GridPos],
) -> Option<(GridPos, Vec<GridPos>)> {
    // 각 목표에 대해 독립적으로 탐색
    // 탐색 트리는 서로 공유하지 않지만,
    // 각 astar 호출 내부에서는 PathNode들이 Rc로 공유됨
    goals.iter()
        .filter_map(|goal| {
            astar(grid, start.clone(), goal)
                .map(|path| (goal.clone(), path))
        })
        .min_by_key(|(_, path)| path.len())
}
```

더 나아가면 시작점에서의 탐색 결과를 캐싱해서 여러 목표에 재활용하는 **역방향 다익스트라** 방식도 쓸 수 있습니다. 모든 노드가 불변이므로 안전하게 공유됩니다.

---

## 동적 장애물 처리: 맵을 교체하기

자율주행 환경에서 장애물은 움직입니다. 사람이 지나가거나, 다른 차량이 끼어들면 경로를 재계획해야 합니다.

가변 방식에서는 맵을 직접 수정하면 진행 중인 탐색과 충돌합니다. 불변 방식에서는 새 맵을 만들고 `astar`를 다시 호출합니다.

```rust
#[derive(Clone)]
struct ImmutableGrid {
    width: usize,
    height: usize,
    obstacles: Rc<Vec<Vec<bool>>>, // 구조적 공유
}

impl ImmutableGrid {
    fn with_obstacle(&self, pos: &GridPos) -> ImmutableGrid {
        // obstacles를 복사하고 해당 위치만 변경
        let mut new_obstacles = (*self.obstacles).clone();
        new_obstacles[pos.row as usize][pos.col as usize] = true;
        ImmutableGrid {
            width: self.width,
            height: self.height,
            obstacles: Rc::new(new_obstacles),
        }
    }
}
```

`with_obstacle`은 원래 맵을 변경하지 않습니다. 이전 탐색 결과(경로 노드들)는 여전히 유효합니다. 새 맵으로 새 탐색을 시작하면서 이전 결과를 비교에 활용할 수 있습니다.

---

## 테스트

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn make_grid(rows: usize, cols: usize, obstacles: Vec<(usize, usize)>) -> Grid {
        let mut obs = vec![vec![false; cols]; rows];
        for (r, c) in obstacles { obs[r][c] = true; }
        Grid { width: cols, height: rows, obstacles: obs }
    }

    #[test]
    fn test_simple_path() {
        let grid = make_grid(5, 5, vec![]);
        let path = astar(&grid, GridPos::new(0, 0), &GridPos::new(4, 4)).unwrap();
        assert_eq!(path.first().unwrap(), &GridPos::new(0, 0));
        assert_eq!(path.last().unwrap(), &GridPos::new(4, 4));
        assert_eq!(path.len(), 9); // 맨해튼 거리 8 + 시작점 1
    }

    #[test]
    fn test_path_around_obstacle() {
        // 세로 벽으로 막기
        let obstacles: Vec<(usize, usize)> = (0..4).map(|r| (r, 2)).collect();
        let grid = make_grid(5, 5, obstacles);
        let path = astar(&grid, GridPos::new(0, 0), &GridPos::new(0, 4)).unwrap();
        // 4행을 통해 우회해야 함
        assert!(path.iter().any(|p| p.row == 4));
    }

    #[test]
    fn test_no_path_when_blocked() {
        let obstacles: Vec<(usize, usize)> = (0..5).map(|r| (r, 2)).collect();
        let grid = make_grid(5, 5, obstacles);
        let path = astar(&grid, GridPos::new(0, 0), &GridPos::new(0, 4));
        assert!(path.is_none());
    }

    #[test]
    fn test_path_reconstruction_is_correct() {
        let grid = make_grid(3, 3, vec![]);
        let path = astar(&grid, GridPos::new(0, 0), &GridPos::new(2, 2)).unwrap();
        // 각 스텝이 인접해야 함
        for window in path.windows(2) {
            let diff_r = (window[0].row - window[1].row).abs();
            let diff_c = (window[0].col - window[1].col).abs();
            assert_eq!(diff_r + diff_c, 1, "경로가 연속적이지 않음");
        }
    }
}
```

맵 데이터만 만들면 됩니다. 로봇도, 모터도, ROS도 없습니다.

---

## 가변 vs 불변 A* 비교

| | 가변 방식 | 불변 방식 |
|--|-----------|-----------|
| 탐색 도중 상태 스냅샷 | 전체 복사 필요 | 현재 노드 하나만 보존 |
| 여러 경로 후보 유지 | 자료구조 복제 | `Rc`로 공유 |
| 장애물 업데이트 | 진행 중인 탐색에 영향 가능 | 새 맵, 새 탐색 — 격리됨 |
| 병렬 탐색 | 락 필요 | 불변이면 락 불필요 |
| 단일 탐색 속도 | 약간 빠름 | 약간 느림 (Rc 오버헤드) |

단일 목표를 한 번만 탐색하면 가변 방식이 약간 빠릅니다. 하지만 자율주행처럼 환경이 바뀌고 여러 후보를 비교해야 하는 상황에서는 불변 방식이 훨씬 다루기 쉽습니다.

---

## 정리

1. **경로 노드를 `Rc`로 연결** — 부모 노드를 복사하지 않고 공유
2. **탐색 함수는 순수 계산** — `astar(grid, start, goal)` 형태, 전역 상태 없음
3. **맵 업데이트는 새 맵 생성** — 이전 탐색 결과와 격리
4. **여러 목표 탐색은 독립 호출** — 공통 조상을 자연스럽게 공유

불변 데이터는 경로 계획을 테스트 가능하고 재현 가능하게 만듭니다. 같은 맵과 같은 시작·목표라면 언제나 같은 경로가 나옵니다. 로그에 맵과 시작·목표를 기록해두면 어떤 경로 결정도 사후에 재현할 수 있습니다.

---

*관련 글: [불변 데이터와 구조적 공유](/posts/programming/functional/functional-immutable-data/), [함수 컴포지션](/posts/programming/functional/functional-composition/), [자율주행 센서 퓨전](/posts/programming/functional/autonomous-sensor-fusion/), [자율주행 센서 파이프라인](/posts/programming/functional/autonomous-sensor-pipeline/), [함수형 PID 제어기](/posts/programming/functional/autonomous-pid-controller/), [시뮬레이션 회귀 테스트 설계](/posts/programming/functional/autonomous-simulation-regression/)*
