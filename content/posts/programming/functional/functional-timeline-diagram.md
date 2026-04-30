---
title: "타임라인 다이어그램: 액션의 실행 순서를 추론하는 법"
date: 2026-04-28T09:00:00+09:00
draft: false
tags: ["함수형 프로그래밍", "Rust", "설계", "타임라인", "Grokking Simplicity"]
categories: ["프로그래밍", "함수형 프로그래밍"]
description: "액션들이 언제, 어떤 순서로 실행되는지 시각화하면 동시성 버그를 코드를 짜기 전에 발견할 수 있습니다."
---

## 이 글을 읽고 나면

- 타임라인 다이어그램이 무엇인지, 어떻게 그리는지 이해합니다.
- 액션들의 실행 순서에서 생기는 버그를 코드 전에 발견할 수 있습니다.
- 타임라인을 단순하게 만드는 설계 원칙을 압니다.

이 글은 [액션/계산/데이터](/posts/programming/functional-actions-calculations-data/), [함수 컴포지션](/posts/programming/functional-composition/), [계층형 설계](/posts/programming/functional-stratified-design/)에 이어지는 네 번째 글입니다.

---

## 왜 액션은 순서가 중요한가

앞선 글들에서 반복한 말이 있습니다.

> 액션은 호출 시점이 중요하다.

계산은 언제 호출해도 같은 결과가 나옵니다. 하지만 액션은 다릅니다.

```rust
fn transfer(from: &mut u64, to: &mut u64, amount: u64) {
    *from -= amount;  // 액션 1: 출금
    *to   += amount;  // 액션 2: 입금
}
```

이 두 줄의 순서가 바뀌거나, 둘 사이에 다른 액션이 끼어들면 잔액이 맞지 않습니다. 계산이었다면 순서는 문제가 되지 않았을 것입니다.

버그가 생기는 건 대부분 이런 순서 문제입니다. 타임라인 다이어그램은 이 순서를 눈에 보이게 만드는 도구입니다.

---

## 타임라인 다이어그램 그리는 법

규칙은 세 가지입니다.

1. **각 스레드(또는 실행 흐름)는 세로 선 하나**
2. **액션은 세로 선 위의 박스**
3. **순서가 보장된 액션은 위아래로, 동시에 실행될 수 있는 액션은 나란히**

단일 스레드에서 순서대로 실행되는 코드:

```
      스레드 1
         │
    ┌────┴────┐
    │  출금   │
    └────┬────┘
         │
    ┌────┴────┐
    │  입금   │
    └────┬────┘
         │
```

두 액션이 순서대로 실행되므로 문제없습니다.

---

## 문제가 생기는 경우: 두 타임라인

온라인 쇼핑몰에서 동시에 두 주문이 들어오는 상황을 생각해보겠습니다.

```rust
static mut STOCK: u32 = 1; // 재고 1개

fn process_order(order_id: u32) {
    let stock = read_stock();         // 액션: 재고 읽기
    if stock > 0 {
        ship_item(order_id);          // 액션: 발송
        write_stock(stock - 1);       // 액션: 재고 차감
    }
}
```

단일 스레드라면 문제없습니다. 하지만 두 요청이 동시에 들어오면:

```
   스레드 1 (주문 A)       스레드 2 (주문 B)
         │                       │
   ┌─────┴─────┐           ┌─────┴─────┐
   │ 재고 읽기  │           │ 재고 읽기  │  ← 둘 다 재고 = 1 읽음
   │  (stock=1)│           │  (stock=1)│
   └─────┬─────┘           └─────┬─────┘
         │                       │
   ┌─────┴─────┐           ┌─────┴─────┐
   │   발송 A  │           │   발송 B  │  ← 둘 다 발송!
   └─────┬─────┘           └─────┬─────┘
         │                       │
   ┌─────┴─────┐           ┌─────┴─────┐
   │ 재고 = 0  │           │ 재고 = 0  │
   └─────┬─────┘           └─────┬─────┘
         │                       │
```

재고가 1개인데 두 번 발송됩니다. 타임라인을 그려보면 문제가 바로 보입니다. 두 타임라인이 공유 상태(`STOCK`)에 동시에 접근하기 때문입니다.

---

## 타임라인을 단순하게 만드는 원칙

에릭 노먼드는 타임라인 문제를 줄이는 원칙을 제시합니다.

### 원칙 1: 타임라인 수를 줄여라

타임라인이 하나면 순서 문제가 생기지 않습니다. 병렬 처리가 필요 없는 곳에서는 굳이 여러 타임라인을 만들지 않습니다.

### 원칙 2: 타임라인 길이를 줄여라

액션 수가 적을수록 순서 조합의 경우의 수가 줄어듭니다. 계산으로 표현할 수 있는 건 계산으로 만들고, 꼭 필요한 액션만 남깁니다.

### 원칙 3: 공유 자원 접근을 한 곳으로 모아라

여러 타임라인이 같은 자원에 접근하는 구간을 최소화합니다.

```
나쁜 구조:                    좋은 구조:
타임라인 1   타임라인 2        타임라인 1   타임라인 2
    │             │                │             │
  읽기          읽기             계산 A        계산 B
    │             │                │             │
  계산A         계산B              └──── 합류 ────┘
    │             │                      │
  쓰기          쓰기                   읽기/쓰기
                                         │
```

오른쪽처럼 공유 자원 접근을 합류 이후 한 곳으로 모으면, 동시성 구간이 줄어들고 추론하기 쉬워집니다.

---

## Rust에서 타임라인 문제 다루기

Rust는 타임라인 문제의 상당 부분을 컴파일 타임에 잡아줍니다. 앞서 본 코드는 Rust에서 컴파일되지 않습니다.

```rust
static mut STOCK: u32 = 1;

// 컴파일 에러: mutable static은 unsafe 없이 접근 불가
fn read_stock() -> u32 {
    STOCK // error[E0133]: use of mutable static is unsafe
}
```

Rust가 강제하는 방식이 곧 "타임라인 원칙 3"입니다. 공유 가변 상태는 반드시 명시적으로 다뤄야 합니다.

### `Mutex`로 공유 상태 접근 직렬화하기

```rust
use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    let stock = Arc::new(Mutex::new(1u32));

    let handles: Vec<_> = (0..2).map(|order_id| {
        let stock = Arc::clone(&stock);
        thread::spawn(move || {
            process_order(order_id, &stock);
        })
    }).collect();

    for h in handles { h.join().unwrap(); }
}

fn process_order(order_id: u32, stock: &Mutex<u32>) {
    let mut s = stock.lock().unwrap(); // 락 획득 — 다른 타임라인 대기
    if *s > 0 {
        println!("주문 {} 발송", order_id);
        *s -= 1;
    } else {
        println!("주문 {} 재고 없음", order_id);
    }
    // 락 해제 — 다른 타임라인 진행
}
```

타임라인 다이어그램으로 보면:

```
   스레드 1 (주문 0)       스레드 2 (주문 1)
         │                       │
   ┌─────┴─────┐                 │  대기
   │  락 획득  │                 │
   │ 재고 읽기 │                 │
   │   발송 0  │                 │
   │ 재고 = 0  │                 │
   │  락 해제  │                 │
   └─────┬─────┘           ┌─────┴─────┐
         │                 │  락 획득  │
         │                 │ 재고 읽기 │
         │                 │ 재고없음  │
         │                 │  락 해제  │
         │                 └─────┬─────┘
         │                       │
```

두 타임라인이 재고 접근 구간에서 직렬화됩니다. 중복 발송이 발생하지 않습니다.

---

## 액션을 줄이면 타임라인이 단순해진다

타임라인 문제를 근본적으로 줄이는 방법은 앞선 글들의 원칙과 같습니다. **액션을 최소화하고 계산으로 대체하기.**

주문 검증 로직을 예로 들겠습니다.

```rust
// 액션이 많은 버전: 각 검증마다 DB 조회
fn validate_order_actions(order_id: u32) -> bool {
    let item_exists  = check_item_in_db(order_id);   // 액션
    let user_valid   = check_user_in_db(order_id);   // 액션
    let stock_ok     = check_stock_in_db(order_id);  // 액션
    item_exists && user_valid && stock_ok
}
```

타임라인:
```
      │
  DB 조회 1   (액션)
      │
  DB 조회 2   (액션)
      │
  DB 조회 3   (액션)
      │
  판단         (계산)
      │
```

액션이 세 번입니다. 각 조회 사이에 상태가 바뀔 수 있고, 각각 실패할 수 있습니다.

```rust
// 액션을 앞으로 모은 버전: 한 번에 필요한 데이터를 수집
struct OrderContext {
    item_exists: bool,
    user_valid: bool,
    stock: u32,
}

fn fetch_order_context(order_id: u32) -> OrderContext {
    // 액션: 필요한 데이터를 한 번에 수집
    OrderContext {
        item_exists: check_item_in_db(order_id),
        user_valid:  check_user_in_db(order_id),
        stock:       fetch_stock_from_db(order_id),
    }
}

fn validate_order(ctx: &OrderContext) -> bool {
    // 계산: 수집된 데이터로만 판단
    ctx.item_exists && ctx.user_valid && ctx.stock > 0
}

fn process(order_id: u32) {
    let ctx = fetch_order_context(order_id); // 액션 구간
    if validate_order(&ctx) {                // 계산
        ship(order_id);                      // 액션
    }
}
```

타임라인:
```
      │
  데이터 수집   (액션, 한 구간으로 묶임)
      │
  검증 판단     (계산, 타임라인에 영향 없음)
      │
  발송 또는 종료 (액션)
      │
```

액션 구간이 명확하게 분리됩니다. `validate_order`는 순수 계산이므로 타임라인 분석에서 신경 쓸 필요가 없습니다.

---

## 타임라인 다이어그램으로 코드 리뷰하기

타임라인 다이어그램은 코드를 짜기 전에 설계를 검토하는 도구로도 쓸 수 있습니다.

다이어그램을 그릴 때 이 질문들을 확인합니다.

**1. 공유 자원에 몇 개의 타임라인이 접근하는가?**
두 개 이상이면, 접근 구간이 겹칠 수 있는지 확인합니다.

**2. 두 액션 사이에 다른 타임라인이 끼어들 수 있는가?**
그렇다면 중간 상태가 노출될 수 있습니다.

**3. 액션의 순서가 바뀌어도 괜찮은가?**
괜찮다면 그 액션들은 독립적입니다. 괜찮지 않다면 순서를 보장하는 장치가 필요합니다.

**4. 계산으로 바꿀 수 있는 액션이 있는가?**
계산은 타임라인에서 투명합니다. 액션 하나를 계산으로 바꾸면 분석해야 할 경우의 수가 줄어듭니다.

---

## 시리즈를 마치며

네 편의 글에 걸쳐 에릭 노먼드의 설계 원칙을 따라왔습니다.

| 글 | 핵심 질문 | 도구 |
|---|---|---|
| [액션/계산/데이터](/posts/programming/functional-actions-calculations-data/) | 이 코드는 무엇인가? | 분류 |
| [함수 컴포지션](/posts/programming/functional-composition/) | 어떻게 연결하는가? | 파이프라인, 모나드 |
| [계층형 설계](/posts/programming/functional-stratified-design/) | 어디에 놓는가? | 의존 방향 |
| **타임라인 다이어그램** (이 글) | 언제 실행되는가? | 시각화, 직렬화 |

네 가지 질문이 함께 작동할 때, 코드는 이해하기 쉽고 변경에 강하며 버그가 적어집니다.

---

*관련 글: [액션/계산/데이터](/posts/programming/functional-actions-calculations-data/), [불변 데이터와 구조적 공유](/posts/programming/functional-immutable-data/), [온어니언 아키텍처](/posts/programming/functional-onion-architecture/), [async/await와 타임라인](/posts/programming/rust-async-timeline/)*

*참고: Eric Normand, [Grokking Simplicity](https://www.manning.com/books/grokking-simplicity), Manning Publications, 2021*
