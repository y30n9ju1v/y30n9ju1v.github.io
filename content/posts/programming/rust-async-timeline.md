---
title: "async/await와 타임라인: 비동기 액션의 순서 문제"
date: 2026-04-28T20:00:00+09:00
draft: false
tags: ["함수형 프로그래밍", "Rust", "설계", "비동기", "Grokking Simplicity"]
categories: ["프로그래밍"]
description: "동기 코드에서 다룬 타임라인 문제가 async/await에서 어떻게 달라지는지, 그리고 어떻게 다뤄야 하는지 설명합니다."
---

## 이 글을 읽고 나면

- 동기 타임라인과 비동기 타임라인의 차이를 이해합니다.
- `async/await`가 액션의 실행 순서를 어떻게 바꾸는지 압니다.
- 비동기 코드에서 계산/액션 구분을 어떻게 유지하는지 봅니다.

이전 글 [타임라인 다이어그램](/posts/programming/functional-timeline-diagram/)을 먼저 읽으면 더 자연스럽게 이어집니다.

---

## 동기 타임라인과 비동기 타임라인

타임라인 다이어그램 글에서 이런 문제를 봤습니다.

```
스레드 1          스레드 2
    │                 │
  읽기             읽기     ← 둘 다 같은 값을 읽음
    │                 │
  쓰기             쓰기     ← 경쟁 조건 발생
```

이것은 **두 스레드가 동시에 실행**될 때 생기는 문제였습니다.

비동기 코드에서는 스레드가 하나여도 비슷한 문제가 생깁니다. `await` 지점마다 실행이 중단되고 다른 작업이 끼어들 수 있기 때문입니다.

```
태스크 1              태스크 2
    │                     │
  await 전 읽기            │
    │ (await — 중단)       │
    │                   읽기
    │                   쓰기
    │ (재개)              │
  await 후 쓰기            │
```

`await` 사이에 다른 태스크가 공유 상태를 바꿀 수 있습니다. 스레드 문제와 구조가 같습니다.

---

## `async`는 액션을 값으로 만든다

`async fn`은 함수를 즉시 실행하지 않습니다. **나중에 실행할 수 있는 값(Future)**을 반환합니다.

```rust
async fn fetch_price(item: &str) -> f64 {
    // 실제로는 네트워크 요청
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    match item {
        "notebook" => 120_000.0,
        _          => 0.0,
    }
}

#[tokio::main]
async fn main() {
    // fetch_price를 호출해도 아직 실행되지 않음
    let future = fetch_price("notebook");

    // .await 해야 실제로 실행됨
    let price = future.await;
    println!("{}", price); // 120000
}
```

함수형 관점에서 `async fn`은 **액션을 데이터로 표현한 것**입니다. `Future`는 "언젠가 실행될 액션"을 담은 값입니다. `.await`가 그 액션을 실제로 실행합니다.

---

## 순서 제어: 순차 vs 병렬

`await`를 쓰는 방식에 따라 실행 순서가 달라집니다.

### 순차 실행: 하나씩 기다리기

```rust
async fn sequential() {
    let price_a = fetch_price("notebook").await; // 완료 후
    let price_b = fetch_price("pen").await;      // 시작
    println!("합계: {}", price_a + price_b);
}
```

타임라인:
```
      │
  fetch_price("notebook") ── await ──▶ 완료
      │
  fetch_price("pen")      ── await ──▶ 완료
      │
  합산
```

두 요청이 순서대로 실행됩니다. 총 시간 = 요청 1 시간 + 요청 2 시간.

### 병렬 실행: 동시에 시작하기

```rust
use tokio::join;

async fn parallel() {
    let (price_a, price_b) = join!(
        fetch_price("notebook"),
        fetch_price("pen"),
    );
    println!("합계: {}", price_a + price_b);
}
```

타임라인:
```
      │
  fetch_price("notebook") ──────────▶ 완료 ─┐
  fetch_price("pen")      ──────────▶ 완료 ─┘
                                             │
                                           합산
```

두 요청이 동시에 실행됩니다. 총 시간 = max(요청 1 시간, 요청 2 시간).

`join!`은 두 Future를 동시에 진행시키고, 둘 다 완료되면 결과를 묶어 반환합니다. 서로 독립적인 액션이라면 병렬로 실행해도 안전합니다.

---

## `await` 사이에서 생기는 경쟁 조건

두 태스크가 같은 상태를 공유할 때 문제가 생깁니다.

```rust
use std::sync::Arc;
use tokio::sync::Mutex;

async fn process_order(
    stock: Arc<Mutex<u32>>,
    order_id: u32,
) {
    let current = *stock.lock().await; // 락 획득 후 즉시 해제됨
                                       // ← 여기서 다른 태스크가 끼어들 수 있음
    if current > 0 {
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        // ← await 동안 다른 태스크가 실행됨
        let mut s = stock.lock().await;
        *s -= 1;
        println!("주문 {} 처리 완료, 재고: {}", order_id, *s);
    } else {
        println!("주문 {} 재고 없음", order_id);
    }
}
```

`lock().await`로 락을 얻고 즉시 해제한 뒤, `await` 지점에서 다른 태스크가 실행되면 읽은 값이 이미 낡은 값이 됩니다.

타임라인:
```
태스크 1                    태스크 2
    │
  lock → current = 1
  unlock
    │ (await — 중단)
    │                     lock → current = 1
    │                     unlock
    │                     (await — 중단)
    │ (재개)
  lock → *s -= 1 → 재고 0
  unlock
    │                     (재개)
    │                     lock → *s -= 1 → 재고 -1 !!
```

재고가 음수가 됩니다.

---

## 해결: 임계 구간을 하나의 락 안에 묶기

읽기와 쓰기를 락 하나로 묶으면 `await` 사이에 끼어들 수 없습니다.

```rust
async fn process_order_safe(
    stock: Arc<Mutex<u32>>,
    order_id: u32,
) {
    let mut s = stock.lock().await; // 락 획득

    if *s > 0 {
        *s -= 1;                    // 읽기 + 쓰기를 락 안에서
        println!("주문 {} 처리 완료, 재고: {}", order_id, *s);
    } else {
        println!("주문 {} 재고 없음", order_id);
    }
    // 락 해제 — MutexGuard가 drop될 때
}
```

단, 락을 들고 `await`하면 다른 태스크가 오래 기다릴 수 있습니다. 임계 구간 안에 네트워크 요청이나 긴 계산이 있으면 안 됩니다.

---

## 계산/액션 구분으로 `await` 줄이기

`await`는 액션입니다. 타임라인 다이어그램 글의 원칙이 그대로 적용됩니다.

> **액션을 최소화하고, 계산으로 대체하라.**

`await`가 많을수록 끼어들 수 있는 지점이 많아집니다. 필요한 데이터를 먼저 모두 모은 뒤, 계산하고, 마지막에 결과를 내보내는 패턴이 여기서도 유효합니다.

```rust
// 나쁜 패턴: await가 중간 중간에 흩어져 있음
async fn process_bad(order_id: u32) -> f64 {
    let price = fetch_price("notebook").await;     // await 1
    let stock = fetch_stock(order_id).await;       // await 2
    if stock > 0 {
        let discount = fetch_discount(order_id).await; // await 3
        save_order(order_id, price * (1.0 - discount)).await; // await 4
        price * (1.0 - discount)
    } else {
        0.0
    }
}

// 좋은 패턴: 데이터 수집(await)과 계산을 분리
async fn process_good(order_id: u32) -> f64 {
    // [액션] 필요한 데이터를 한 번에 병렬로 수집
    let (price, stock, discount) = join!(
        fetch_price("notebook"),
        fetch_stock(order_id),
        fetch_discount(order_id),
    );

    // [계산] 순수한 계산 — await 없음
    let total = calculate_total(price, stock, discount);

    // [액션] 결과 저장
    if total > 0.0 {
        save_order(order_id, total).await;
    }

    total
}

fn calculate_total(price: f64, stock: u32, discount: f64) -> f64 {
    if stock > 0 { price * (1.0 - discount) } else { 0.0 }
}
```

`calculate_total`은 순수한 계산이라 단독으로 테스트할 수 있습니다. `process_good`의 `await`는 두 곳뿐이고, 계산 구간에는 `await`가 없습니다.

---

## 비동기 타임라인 다이어그램

`process_good`을 다이어그램으로 그리면:

```
      │
  join! 시작
  ├── fetch_price ──────▶ 완료 ─┐
  ├── fetch_stock ──────▶ 완료 ─┤
  └── fetch_discount ───▶ 완료 ─┘
                                │
                          calculate_total  ← await 없음, 끼어들 수 없음
                                │
                          save_order ── await ──▶ 완료
                                │
```

`await` 지점이 두 곳입니다. `calculate_total` 구간에는 `await`가 없으므로 다른 태스크가 절대 끼어들 수 없습니다. 타임라인이 단순합니다.

---

## 정리

비동기 코드에서 타임라인 원칙은 동기 코드와 같습니다.

| 원칙 | 동기 | 비동기 |
|---|---|---|
| 액션 수 줄이기 | 함수 호출 횟수 | `await` 횟수 |
| 공유 자원 보호 | `Mutex` | `tokio::sync::Mutex` |
| 독립 액션 병렬화 | 스레드 | `join!` |
| 계산/액션 분리 | 순수 함수 분리 | `await` 없는 함수 분리 |

`async/await`는 비동기 액션을 동기 코드처럼 쓸 수 있게 해주는 문법입니다. 하지만 `.await`마다 타임라인이 일시 중단된다는 사실은 그대로입니다. 타임라인 다이어그램으로 `await` 지점을 시각화하면, 어디서 경쟁 조건이 생길 수 있는지 코드를 짜기 전에 파악할 수 있습니다.

---

*관련 글: [타임라인 다이어그램](/posts/programming/functional-timeline-diagram/), [액션/계산/데이터](/posts/programming/functional-actions-calculations-data/), [온어니언 아키텍처](/posts/programming/functional-onion-architecture/), [불변 데이터와 구조적 공유](/posts/programming/functional-immutable-data/)*
