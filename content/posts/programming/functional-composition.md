---
title: "함수형 프로그래밍의 심장: 함수 컴포지션"
date: 2026-04-28T07:00:00+09:00
draft: false
tags: ["함수형 프로그래밍", "Rust", "설계", "Grokking Simplicity"]
categories: ["프로그래밍"]
description: "액션/계산/데이터 구분 위에서 작은 함수들을 조합해 복잡한 로직을 만드는 방법을 Rust 예제로 설명합니다."
---

## 이 글을 읽고 나면

- 함수 컴포지션이 무엇인지, 왜 함수형 프로그래밍의 핵심인지 이해합니다.
- 작은 계산들을 조합해서 복잡한 로직을 만드는 방법을 압니다.
- 액션/계산/데이터 구분이 컴포지션과 어떻게 연결되는지 봅니다.

이전 글 [코드를 세 가지로 나누면 복잡성이 사라진다](/posts/programming/functional-actions-calculations-data/)를 먼저 읽으면 더 자연스럽게 이어집니다.

---

## 컴포지션이란

컴포지션(Composition)은 **작은 함수들을 조합해 더 큰 함수를 만드는 것**입니다.

수학에서 `h(x) = f(g(x))`로 쓰는 것과 같습니다. `g`의 출력이 `f`의 입력이 됩니다.

코드로 표현하면 이렇습니다.

```rust
fn double(x: i32) -> i32 { x * 2 }
fn add_one(x: i32) -> i32 { x + 1 }

// double과 add_one을 조합
fn double_then_add_one(x: i32) -> i32 {
    add_one(double(x))
}

fn main() {
    println!("{}", double_then_add_one(3)); // (3*2)+1 = 7
}
```

이게 전부입니다. 하지만 이 단순한 원칙을 일관되게 적용하면 코드 구조 전체가 바뀝니다.

---

## 왜 계산만 조합해야 하나

이전 글에서 코드를 액션, 계산, 데이터로 나눴습니다. 컴포지션은 **계산들 사이에서 가장 잘 작동합니다.**

이유는 간단합니다. 계산은 입력과 출력만 있고 외부 영향이 없기 때문에, 조합해도 예측 가능한 결과가 나옵니다.

```
계산 A → 계산 B → 계산 C
```

반면 액션을 조합하면 순서, 타이밍, 부수효과가 얽히면서 추론하기 어려워집니다.

```
액션 A → 액션 B → 액션 C   ← 순서 바뀌면? 실패하면? 중복 실행되면?
```

그래서 함수형 설계의 목표는 이렇게 됩니다.

> **중간 로직은 계산 컴포지션으로 표현하고, 액션은 가장 바깥에서 한 번만 실행한다.**

---

## 실전: 주문 처리 파이프라인

온라인 쇼핑몰의 주문 처리 로직을 단계적으로 만들어 보겠습니다.

각 단계는 독립적인 계산입니다.

1. 주문의 총액을 계산한다
2. 회원 등급에 따라 할인을 적용한다
3. 배송비를 더한다
4. 최종 영수증 데이터를 만든다

### 데이터 정의

```rust
#[derive(Debug, Clone)]
struct Order {
    items: Vec<(String, u32, f64)>, // (상품명, 수량, 단가)
    member_grade: MemberGrade,
}

#[derive(Debug, Clone)]
enum MemberGrade {
    Regular,
    Silver,
    Gold,
}

#[derive(Debug)]
struct Receipt {
    subtotal: f64,
    discount: f64,
    shipping: f64,
    total: f64,
}
```

### 계산 정의: 각 단계를 독립된 함수로

```rust
fn subtotal(order: &Order) -> f64 {
    order.items.iter()
        .map(|(_, qty, price)| *qty as f64 * price)
        .sum()
}

fn discount_rate(grade: &MemberGrade) -> f64 {
    match grade {
        MemberGrade::Regular => 0.0,
        MemberGrade::Silver => 0.05,
        MemberGrade::Gold   => 0.10,
    }
}

fn shipping_fee(subtotal: f64) -> f64 {
    if subtotal >= 50_000.0 { 0.0 } else { 3_000.0 }
}

fn build_receipt(order: &Order) -> Receipt {
    let sub = subtotal(order);
    let disc = sub * discount_rate(&order.member_grade);
    let ship = shipping_fee(sub - disc);
    Receipt {
        subtotal: sub,
        discount: disc,
        shipping: ship,
        total: sub - disc + ship,
    }
}
```

`build_receipt`는 작은 계산 세 개를 조합합니다. 각 계산은 독립적으로 테스트할 수 있고, 조합 결과도 예측 가능합니다.

### 액션은 가장 마지막에

```rust
fn print_receipt(receipt: &Receipt) {
    println!("소계:   {:>10.0}원", receipt.subtotal);
    println!("할인:  -{:>10.0}원", receipt.discount);
    println!("배송비: {:>10.0}원", receipt.shipping);
    println!("─────────────────");
    println!("합계:   {:>10.0}원", receipt.total);
}

fn save_receipt(receipt: &Receipt) {
    println!("[DB] 영수증 저장: 총 {:.0}원", receipt.total);
}

fn process_order(order: &Order) {
    let receipt = build_receipt(order); // 계산
    print_receipt(&receipt);           // 액션
    save_receipt(&receipt);            // 액션
}
```

`build_receipt`까지는 순수한 계산의 파이프라인입니다. 액션은 맨 마지막에 계산 결과를 받아서 처리합니다.

---

## 고차 함수로 컴포지션 일반화하기

위 방식은 잘 작동하지만, 매번 새로운 파이프라인 함수를 만들어야 합니다. 고차 함수(Higher-Order Function)를 쓰면 컴포지션 자체를 추상화할 수 있습니다.

고차 함수란 **함수를 인자로 받거나 함수를 반환하는 함수**입니다.

### `map`과 `filter`로 데이터 변환하기

Rust의 이터레이터는 이미 컴포지션 친화적으로 설계되어 있습니다.

```rust
fn summary(orders: &[Order]) -> f64 {
    orders.iter()
        .map(build_receipt)           // 각 주문을 영수증으로 변환 (계산)
        .filter(|r| r.total > 0.0)   // 유효한 영수증만 (계산)
        .map(|r| r.total)             // 합계만 추출 (계산)
        .sum()                        // 전체 합산 (계산)
}
```

이 체인 전체가 계산입니다. 외부 상태를 바꾸지 않고, 같은 입력이면 항상 같은 출력이 나옵니다.

### 변환 함수를 직접 조합하기

여러 변환 규칙을 동적으로 조합해야 할 때는 함수 슬라이스를 쓸 수 있습니다.

```rust
fn apply_pipeline(value: f64, steps: &[fn(f64) -> f64]) -> f64 {
    steps.iter().fold(value, |acc, step| step(acc))
}

fn add_tax(price: f64) -> f64 { price * 1.1 }
fn round_to_100(price: f64) -> f64 { (price / 100.0).round() * 100.0 }
fn cap_at_million(price: f64) -> f64 { price.min(1_000_000.0) }

fn main() {
    let price = 45_678.0;
    let pipeline: &[fn(f64) -> f64] = &[add_tax, round_to_100, cap_at_million];
    let result = apply_pipeline(price, pipeline);
    println!("{:.0}원", result); // 50200원
}
```

`apply_pipeline`은 어떤 변환 함수든 받아서 순서대로 적용합니다. 새로운 규칙이 생겨도 함수를 하나 추가하기만 하면 됩니다.

---

## 컴포지션이 테스트를 쉽게 만든다

작은 계산들로 분리해 두면, 테스트도 조각조각 할 수 있습니다.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn gold_order() -> Order {
        Order {
            items: vec![("노트북".to_string(), 1, 120_000.0)],
            member_grade: MemberGrade::Gold,
        }
    }

    #[test]
    fn test_subtotal() {
        assert_eq!(subtotal(&gold_order()), 120_000.0);
    }

    #[test]
    fn test_gold_discount() {
        assert_eq!(discount_rate(&MemberGrade::Gold), 0.10);
    }

    #[test]
    fn test_free_shipping_over_50k() {
        assert_eq!(shipping_fee(60_000.0), 0.0);
    }

    #[test]
    fn test_full_receipt() {
        let receipt = build_receipt(&gold_order());
        // 120000 - 12000(10%) + 0(무료배송) = 108000
        assert_eq!(receipt.total, 108_000.0);
    }
}
```

`test_full_receipt`가 실패해도, 어느 단계에서 틀렸는지 위의 단위 테스트들이 바로 알려줍니다. 전체를 다시 디버깅할 필요가 없습니다.

---

## 컴포지션의 한계: 액션이 중간에 필요할 때

모든 로직이 순수 계산으로만 이루어지지는 않습니다. 파이프라인 중간에 DB 조회나 외부 API 호출이 필요한 경우가 있습니다.

```rust
// 주문 처리 중 재고 확인이 필요한 경우
fn check_stock(item: &str) -> bool {
    // DB 조회 — 이건 액션
    println!("[DB] 재고 확인: {}", item);
    true
}
```

이런 경우에는 **액션을 계산 파이프라인 바깥으로 끌어내는 방식**으로 대응합니다.

```rust
// 나쁜 방법: 파이프라인 중간에 액션 삽입
fn build_receipt_bad(order: &Order) -> Option<Receipt> {
    for (item, _, _) in &order.items {
        if !check_stock(item) { return None; } // 계산 안에 액션이 섞임
    }
    Some(build_receipt(order))
}

// 좋은 방법: 액션을 앞으로 분리
fn fetch_stock_status(order: &Order) -> Vec<bool> {
    order.items.iter()
        .map(|(item, _, _)| check_stock(item)) // 액션: 재고 정보를 미리 수집
        .collect()
}

fn all_in_stock(statuses: &[bool]) -> bool {
    statuses.iter().all(|&s| s) // 계산: 수집된 데이터로 판단
}

fn process_order_with_stock(order: &Order) {
    let statuses = fetch_stock_status(order); // 액션 먼저
    if all_in_stock(&statuses) {             // 계산
        let receipt = build_receipt(order);   // 계산
        print_receipt(&receipt);              // 액션
        save_receipt(&receipt);               // 액션
    }
}
```

패턴은 항상 같습니다. **액션으로 데이터를 모으고, 계산으로 처리하고, 액션으로 결과를 내보냅니다.**

```
[액션] 입력 수집 → [계산] 처리 → [액션] 결과 출력
```

---

## 실패할 수 있는 계산의 컴포지션: 모나드

지금까지의 계산들은 모두 항상 성공한다고 가정했습니다. 하지만 실제 파이프라인에서는 각 단계가 실패할 수 있습니다.

- 입력 문자열이 숫자로 변환되지 않을 수 있다
- 조회한 상품이 존재하지 않을 수 있다
- 할인 쿠폰이 만료됐을 수 있다

이런 상황에서 단순하게 함수를 연결하면 문제가 생깁니다.

```rust
fn parse_quantity(s: &str) -> u32 { s.parse().unwrap() }   // 실패하면 패닉
fn find_price(item: &str) -> f64 { /* DB 조회 */ 100.0 }
fn apply_coupon(price: f64, code: &str) -> f64 { price * 0.9 }
```

`unwrap()`을 남발하거나, 단계마다 `if let`으로 에러를 확인하면 파이프라인이 망가집니다.

### `Result`로 실패를 값으로 표현하기

함수형 접근은 실패를 예외로 던지지 않고 **반환값에 담습니다.**

```rust
fn parse_quantity(s: &str) -> Result<u32, String> {
    s.parse::<u32>().map_err(|_| format!("'{}' is not a valid quantity", s))
}

fn find_price(item: &str) -> Result<f64, String> {
    match item {
        "notebook" => Ok(120_000.0),
        "pen"      => Ok(3_000.0),
        other      => Err(format!("item '{}' not found", other)),
    }
}

fn apply_coupon(price: f64, code: &str) -> Result<f64, String> {
    match code {
        "SALE10" => Ok(price * 0.9),
        other    => Err(format!("coupon '{}' is invalid or expired", other)),
    }
}
```

### 문제: `Result`를 연결하면 코드가 지저분해진다

각 단계가 `Result`를 반환하면, 이걸 연결하는 코드가 이렇게 됩니다.

```rust
fn calculate_price_verbose(item: &str, qty_str: &str, coupon: &str) -> Result<f64, String> {
    let qty = match parse_quantity(qty_str) {
        Ok(q)    => q,
        Err(e)   => return Err(e),
    };
    let unit_price = match find_price(item) {
        Ok(p)    => p,
        Err(e)   => return Err(e),
    };
    let total = qty as f64 * unit_price;
    match apply_coupon(total, coupon) {
        Ok(p)  => Ok(p),
        Err(e) => Err(e),
    }
}
```

파이프라인의 본질은 세 줄인데, 에러 처리 boilerplate가 코드를 덮어버립니다.

### `and_then`으로 파이프라인 복원하기

`Result`의 `and_then`은 **성공했을 때만 다음 단계로 넘어가고, 실패하면 그 에러를 그대로 전달합니다.** 이것이 모나드의 `bind`(`>>=`) 연산입니다.

```rust
fn calculate_price(item: &str, qty_str: &str, coupon: &str) -> Result<f64, String> {
    parse_quantity(qty_str)
        .and_then(|qty| find_price(item).map(|p| qty as f64 * p))
        .and_then(|total| apply_coupon(total, coupon))
}
```

각 단계가 성공하면 다음으로, 실패하면 파이프라인 전체가 그 에러를 들고 종료됩니다. 중간에 `match`나 `if let`이 없습니다.

Rust에서는 `?` 연산자로 같은 것을 더 읽기 좋게 쓸 수 있습니다.

```rust
fn calculate_price(item: &str, qty_str: &str, coupon: &str) -> Result<f64, String> {
    let qty        = parse_quantity(qty_str)?;
    let unit_price = find_price(item)?;
    let total      = qty as f64 * unit_price;
    let discounted = apply_coupon(total, coupon)?;
    Ok(discounted)
}
```

`?`는 `Err`이면 즉시 함수를 반환하고, `Ok`이면 안의 값을 꺼냅니다. 파이프라인처럼 읽히면서 에러 처리도 완전합니다.

### 동작 확인

```rust
fn main() {
    println!("{:?}", calculate_price("notebook", "2", "SALE10"));
    // Ok(216000.0)

    println!("{:?}", calculate_price("notebook", "two", "SALE10"));
    // Err("'two' is not a valid quantity")

    println!("{:?}", calculate_price("eraser", "2", "SALE10"));
    // Err("item 'eraser' not found")

    println!("{:?}", calculate_price("notebook", "2", "OLD50"));
    // Err("coupon 'OLD50' is invalid or expired")
}
```

어느 단계에서 실패했는지 메시지가 정확히 나오고, 나머지 단계는 실행되지 않습니다.

### 모나드가 컴포지션에서 하는 역할

모나드는 어렵게 들리지만, 하는 일은 단순합니다.

> **컨텍스트(실패 가능성, 비동기, 부수효과 등)를 가진 값들을 파이프라인으로 연결할 수 있게 해주는 패턴**

`Result<T, E>`는 "실패할 수 있는 값"이라는 컨텍스트를 가집니다. `and_then`(`?`)은 그 컨텍스트를 유지하면서 함수를 연결합니다.

```
f64 → f64            일반 컴포지션     (항상 성공)
Result<f64> → Result<f64>    모나딕 컴포지션  (실패 전파)
```

일반 컴포지션이 "성공하는 계산들의 파이프라인"이라면, 모나딕 컴포지션은 "실패할 수 있는 계산들의 파이프라인"입니다. 구조는 같고, 컨텍스트만 다릅니다.

---

## 정리

| | 컴포지션 가능? | 이유 |
|---|---|---|
| 계산 + 계산 | 자유롭게 | 순서 무관, 예측 가능 |
| 데이터 → 계산 | 자연스럽게 | 데이터는 그냥 입력값 |
| 액션 + 계산 | 분리 후 조합 | 액션을 앞뒤로 밀어내야 함 |
| 액션 + 액션 | 신중하게 | 순서와 타이밍을 명확히 해야 함 |

함수 컴포지션의 힘은 **작은 것들을 신뢰할 수 있을 때** 나옵니다. 계산은 테스트로 신뢰를 증명할 수 있고, 그렇게 신뢰가 쌓인 함수들을 조합하면 복잡한 로직도 단순하게 표현됩니다.

---

## 다음으로

컴포지션을 더 깊이 파고들면 이런 주제들로 이어집니다.

- **계층형 설계**: 어떤 함수가 어떤 함수 위에 쌓여야 하는가 → [계층형 설계](/posts/programming/functional-stratified-design/)
- **`bind`와 모나드 법칙**: 모나딕 컴포지션의 수학적 배경 → [모나드로 배우는 함수형 에러 처리](/posts/programming/monad-intro-error-handling/)

---

*참고: Eric Normand, [Grokking Simplicity](https://www.manning.com/books/grokking-simplicity), Manning Publications, 2021*
