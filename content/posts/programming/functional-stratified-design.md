---
title: "계층형 설계: 어떤 함수가 어떤 함수 위에 쌓여야 하는가"
date: 2026-04-28T08:00:00+09:00
draft: false
tags: ["함수형 프로그래밍", "Rust", "설계", "계층형 설계", "Grokking Simplicity"]
categories: ["프로그래밍", "함수형 프로그래밍"]
description: "함수들 사이의 의존 방향을 정리하면 변경에 강한 코드가 됩니다. 계층형 설계의 원칙을 Rust 예제로 설명합니다."
---

## 이 글을 읽고 나면

- 함수들을 계층으로 나누는 기준이 무엇인지 이해합니다.
- 어떤 함수가 어떤 함수를 호출해야 하는지 판단할 수 있습니다.
- 변경이 생겼을 때 영향 범위를 예측할 수 있습니다.

이 글은 [액션/계산/데이터 구분](/posts/programming/functional-actions-calculations-data/)과 [함수 컴포지션](/posts/programming/functional-composition/)에 이어지는 세 번째 글입니다.

---

## 문제: 함수를 나눴는데도 바꾸기 어렵다

앞선 글에서 코드를 작은 함수들로 나눴습니다. 그런데 실제로 작업하다 보면 이런 상황을 만납니다.

> 할인 정책 하나를 바꾸려고 했는데, 여기저기 고쳐야 할 곳이 생긴다.

함수를 나눴다고 해서 자동으로 유지보수가 쉬워지지는 않습니다. **어떤 함수가 어떤 함수를 호출하느냐**, 즉 의존 방향이 잘못되어 있으면 변경 하나가 파문처럼 퍼집니다.

계층형 설계(Stratified Design)는 이 문제를 다룹니다.

---

## 계층형 설계의 핵심 아이디어

> **자주 바뀌는 것은 위에, 잘 바뀌지 않는 것은 아래에.**

함수들을 변경 빈도와 추상화 수준에 따라 계층으로 쌓습니다. 위쪽 함수는 아래쪽 함수를 호출하지만, 아래쪽 함수는 위쪽을 모릅니다.

```
┌───────────────────────────────┐
│       비즈니스 규칙 계층        │  ← 자주 바뀜. 정책, 흐름 조율
├───────────────────────────────┤
│       도메인 연산 계층          │  ← 가끔 바뀜. 핵심 계산
├───────────────────────────────┤
│       기반 연산 계층            │  ← 거의 안 바뀜. 원시 타입 조작
└───────────────────────────────┘
```

핵심 규칙은 하나입니다.

> **의존은 아래 방향으로만 흐른다.**

위 계층이 아래 계층을 호출하는 것은 괜찮습니다. 아래 계층이 위 계층을 알면 안 됩니다.

---

## 실전: 쇼핑몰 주문 시스템

주문 처리 코드를 계층별로 설계해 보겠습니다.

### 기반 연산 계층 — 원시 타입에 가까운 연산

이 계층의 함수들은 비즈니스 도메인을 모릅니다. 숫자, 문자열, 컬렉션을 다루는 일만 합니다.

```rust
// 숫자 반올림
fn round_to(value: f64, unit: f64) -> f64 {
    (value / unit).round() * unit
}

// 비율 적용
fn apply_rate(value: f64, rate: f64) -> f64 {
    value * rate
}

// 범위 제한
fn clamp(value: f64, min: f64, max: f64) -> f64 {
    value.max(min).min(max)
}
```

이 함수들은 주문이 뭔지, 할인이 뭔지 모릅니다. 그래서 가장 안정적입니다. 수학적 사실은 바뀌지 않습니다.

---

### 도메인 연산 계층 — 비즈니스 개념을 다루는 연산

이 계층은 `Order`, `Item`, `MemberGrade` 같은 도메인 타입을 알고, 기반 연산을 조합해 도메인 계산을 표현합니다.

```rust
#[derive(Debug, Clone)]
struct Item {
    name: String,
    price: f64,
    quantity: u32,
}

#[derive(Debug, Clone)]
enum MemberGrade { Regular, Silver, Gold }

#[derive(Debug, Clone)]
struct Order {
    items: Vec<Item>,
    grade: MemberGrade,
    coupon_discount: f64, // 쿠폰 할인율 (0.0 ~ 1.0)
}

// 항목 하나의 소계
fn item_subtotal(item: &Item) -> f64 {
    apply_rate(item.price, item.quantity as f64)
}

// 주문 전체 소계
fn order_subtotal(order: &Order) -> f64 {
    order.items.iter().map(item_subtotal).sum()
}

// 회원 등급 할인율
fn grade_discount_rate(grade: &MemberGrade) -> f64 {
    match grade {
        MemberGrade::Regular => 0.0,
        MemberGrade::Silver  => 0.05,
        MemberGrade::Gold    => 0.10,
    }
}

// 총 할인율 (등급 + 쿠폰, 최대 30%)
fn total_discount_rate(order: &Order) -> f64 {
    let rate = grade_discount_rate(&order.grade) + order.coupon_discount;
    clamp(rate, 0.0, 0.30)
}

// 할인 후 금액
fn discounted_price(order: &Order) -> f64 {
    let sub = order_subtotal(order);
    sub - apply_rate(sub, total_discount_rate(order))
}

// 배송비
fn shipping_fee(discounted: f64) -> f64 {
    if discounted >= 50_000.0 { 0.0 } else { 3_000.0 }
}
```

이 계층은 기반 연산만 호출합니다. `apply_rate`, `clamp` 같은 함수들이 아래에 있고, 도메인 연산이 위에서 그것들을 조합합니다.

---

### 비즈니스 규칙 계층 — 정책과 흐름을 조율하는 연산

이 계층은 "주문을 어떻게 처리할 것인가"라는 질문에 답합니다. 도메인 연산들을 엮어서 전체 흐름을 만듭니다.

```rust
#[derive(Debug)]
struct Receipt {
    subtotal:  f64,
    discount:  f64,
    shipping:  f64,
    total:     f64,
}

fn build_receipt(order: &Order) -> Receipt {
    let sub      = order_subtotal(order);
    let disc     = apply_rate(sub, total_discount_rate(order));
    let after    = sub - disc;
    let ship     = shipping_fee(after);
    Receipt {
        subtotal: round_to(sub,   100.0),
        discount: round_to(disc,  100.0),
        shipping: ship,
        total:    round_to(after + ship, 100.0),
    }
}

fn is_vip_eligible(order: &Order) -> bool {
    order_subtotal(order) >= 500_000.0
        || matches!(order.grade, MemberGrade::Gold)
}
```

`build_receipt`는 도메인 연산 여러 개를 조합하지만, `apply_rate`나 `clamp` 같은 기반 연산을 직접 호출하지 않습니다. 계층을 건너뛰지 않는 것이 중요합니다.

---

### 액션 계층 — 외부 세계와 접촉하는 가장 바깥

```rust
fn print_receipt(r: &Receipt) {
    println!("소계:   {:>10.0}원", r.subtotal);
    println!("할인:  -{:>10.0}원", r.discount);
    println!("배송비: {:>10.0}원", r.shipping);
    println!("──────────────────");
    println!("합계:   {:>10.0}원", r.total);
}

fn process_order(order: &Order) {
    let receipt = build_receipt(order);   // 비즈니스 규칙 계층
    print_receipt(&receipt);             // 액션
}
```

---

## 의존 방향을 그려보기

완성된 코드의 호출 관계를 그려보면 이렇습니다.

```
process_order          [액션]
    └─ build_receipt   [비즈니스 규칙]
        ├─ order_subtotal
        │    └─ item_subtotal
        │         └─ apply_rate    [기반 연산]
        ├─ total_discount_rate
        │    ├─ grade_discount_rate
        │    └─ clamp              [기반 연산]
        ├─ apply_rate              [기반 연산]
        ├─ shipping_fee
        └─ round_to                [기반 연산]
```

모든 화살표가 아래를 향합니다. 기반 연산(`apply_rate`, `clamp`, `round_to`)은 누가 자신을 호출하는지 전혀 모릅니다.

---

## 계층을 어기면 어떤 일이 생기나

할인율 계산 로직을 바꾼다고 가정합니다. `total_discount_rate` 함수를 수정합니다.

올바른 계층 구조에서는 영향 범위가 예측 가능합니다.

```
total_discount_rate 변경
    → discounted_price 영향
    → build_receipt 영향
    → process_order 영향
```

위 방향으로만 파급됩니다. 기반 연산(`apply_rate`)은 전혀 영향받지 않습니다.

반면 계층을 어기면 어떻게 될까요. 기반 연산이 도메인 개념을 알게 되면:

```rust
// 잘못된 설계: 기반 연산이 도메인를 알고 있음
fn apply_rate_for_gold(value: f64, grade: &MemberGrade) -> f64 {
    match grade {
        MemberGrade::Gold => value * 0.90,
        _                 => value,
    }
}
```

이제 회원 등급 정책이 바뀌면 기반 연산도 수정해야 합니다. 기반 연산을 믿고 쓰던 다른 모든 코드가 흔들립니다.

---

## 계층을 나누는 기준

실제로 코드를 작성할 때 "이 함수는 어느 계층인가?"를 판단하는 기준은 두 가지입니다.

**1. 무엇을 아는가**

함수가 알고 있는 타입이 구체적일수록 위 계층입니다.

- `f64`만 다룬다 → 기반 연산
- `Order`, `Item`을 다룬다 → 도메인 연산
- 정책 조건(`if 골드 등급이면...`)을 다룬다 → 비즈니스 규칙

**2. 얼마나 자주 바뀌는가**

- 자주 바뀐다 → 위 계층 (변경이 아래로 전파되지 않도록)
- 거의 안 바뀐다 → 아래 계층 (많은 코드가 믿고 쓸 수 있도록)

---

## 테스트도 계층을 따른다

계층이 잘 나뉘면 테스트 전략도 자연스럽게 결정됩니다.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // 기반 연산: 입력/출력만 확인
    #[test]
    fn test_clamp() {
        assert_eq!(clamp(0.35, 0.0, 0.30), 0.30);
        assert_eq!(clamp(0.05, 0.0, 0.30), 0.05);
    }

    // 도메인 연산: 도메인 시나리오로 확인
    #[test]
    fn test_gold_discount_capped_with_coupon() {
        let order = Order {
            items: vec![Item { name: "pen".into(), price: 1000.0, quantity: 1 }],
            grade: MemberGrade::Gold,
            coupon_discount: 0.25, // 10% + 25% = 35% → 30%로 제한
        };
        assert_eq!(total_discount_rate(&order), 0.30);
    }

    // 비즈니스 규칙: 영수증 전체 흐름 확인
    #[test]
    fn test_receipt_total() {
        let order = Order {
            items: vec![Item { name: "notebook".into(), price: 120_000.0, quantity: 1 }],
            grade: MemberGrade::Gold,
            coupon_discount: 0.0,
        };
        let r = build_receipt(&order);
        // 120000 - 12000(10%) + 0(무료배송) = 108000
        assert_eq!(r.total, 108_000.0);
    }
}
```

아래 계층 테스트는 빠르고 단순합니다. 위 계층 테스트는 더 많은 것을 조율하지만, 기반 연산이 이미 검증됐으므로 실패 원인을 좁히기 쉽습니다.

---

## 정리

계층형 설계를 한 문장으로 요약하면 이렇습니다.

> **변경이 위에서 아래로는 퍼지지 않도록, 의존은 아래 방향으로만 흐르게 한다.**

이 원칙을 지키면:
- 어떤 함수를 고쳐도 영향 범위를 예측할 수 있습니다
- 아래 계층 함수들은 믿고 재사용할 수 있습니다
- 테스트가 계층별로 역할이 분명해집니다

세 편의 글에 걸쳐 에릭 노먼드의 설계 원칙을 따라왔습니다.

1. [액션/계산/데이터 구분](/posts/programming/functional-actions-calculations-data/) — 코드를 세 가지로 나누기
2. [함수 컴포지션](/posts/programming/functional-composition/) — 계산들을 파이프라인으로 연결하기
3. **계층형 설계** (이 글) — 파이프라인들을 계층으로 쌓기

이 세 가지가 함께 작동할 때, 코드는 작고 신뢰할 수 있는 조각들로 이루어진 구조가 됩니다.

---

*관련 글: [액션/계산/데이터](/posts/programming/functional-actions-calculations-data/), [함수 컴포지션](/posts/programming/functional-composition/), [온어니언 아키텍처](/posts/programming/functional-onion-architecture/), [중복 제거와 추상화의 기준](/posts/programming/functional-abstraction-barrier/), [함수형으로 상태 기계 만들기](/posts/programming/functional-state-machine/)*

*참고: Eric Normand, [Grokking Simplicity](https://www.manning.com/books/grokking-simplicity), Manning Publications, 2021*
