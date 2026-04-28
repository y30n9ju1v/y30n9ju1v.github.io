---
title: "코드를 세 가지로 나누면 복잡성이 사라진다: 액션, 계산, 데이터"
date: 2026-04-28T06:00:00+09:00
draft: false
tags: ["함수형 프로그래밍", "Rust", "설계", "액션 계산 데이터", "Grokking Simplicity"]
categories: ["프로그래밍"]
description: "에릭 노먼드의 'Grokking Simplicity'에서 소개한 액션/계산/데이터 구분법을 Rust 예제로 이해합니다."
---

## 이 글을 읽고 나면

- 코드를 액션, 계산, 데이터 세 가지로 분류할 수 있습니다.
- 왜 이 구분이 테스트와 디버깅을 쉽게 만드는지 이해합니다.
- 복잡한 코드를 어디서부터 단순하게 만들어야 할지 감이 생깁니다.

---

## 함수형 프로그래밍의 진짜 핵심

함수형 프로그래밍을 처음 배우면 "순수 함수", "불변성", "모나드" 같은 단어들을 만납니다. 개념은 알겠는데, 실제 코드에 어떻게 적용해야 할지 막막할 때가 많습니다.

에릭 노먼드(Eric Normand)는 책 **Grokking Simplicity**에서 실용적인 출발점을 제시합니다.

> 코드를 **액션(Action)**, **계산(Calculation)**, **데이터(Data)** 세 가지로 나누는 것.

이 세 가지를 구분하는 것만으로도 코드의 복잡성을 극적으로 줄일 수 있습니다.

---

## 세 가지 정의

### 데이터 (Data)

부르는 것, 저장하는 것, 전달하는 것. 그 자체로는 아무 일도 하지 않습니다.

```rust
struct Order {
    item: String,
    quantity: u32,
    price_per_unit: f64,
}
```

데이터는 **해석되기 전까지는 아무 의미가 없습니다.** 그래서 안전합니다. 언제 만들어도, 얼마나 복사해도 문제가 없습니다.

---

### 계산 (Calculation)

입력을 받아서 출력을 돌려주는 것. **같은 입력에는 항상 같은 출력**을 내고, **외부 세계에 아무런 영향을 주지 않습니다.**

수학의 함수와 같습니다. `f(x) = x * 2`는 항상 같은 결과를 냅니다.

```rust
fn total_price(order: &Order) -> f64 {
    order.quantity as f64 * order.price_per_unit
}

fn apply_discount(price: f64, discount_rate: f64) -> f64 {
    price * (1.0 - discount_rate)
}
```

계산은 **언제 호출해도 안전**합니다. 테스트가 쉽고, 병렬로 실행해도 문제가 없습니다.

---

### 액션 (Action)

**호출 시점이 중요하거나, 외부 세계와 상호작용하는 것.** 같은 코드라도 언제 실행하느냐에 따라 결과가 달라집니다.

```rust
fn send_confirmation_email(email: &str, order: &Order) {
    // 이메일을 실제로 보냄 — 취소할 수 없음
    println!("Sending email to {}: {} ordered", email, order.item);
}

fn save_to_database(order: &Order) {
    // DB에 기록 — 외부 상태를 바꿈
    println!("Saving order to DB: {:?}", order.item);
}

fn get_current_time() -> std::time::SystemTime {
    // 호출할 때마다 다른 값을 반환
    std::time::SystemTime::now()
}
```

액션은 **다루기 가장 어렵습니다.** 테스트하기 어렵고, 실행 순서에 따라 버그가 생기기도 합니다.

---

## 왜 이 구분이 중요한가

코드를 이 세 가지로 나누고 나면 한 가지 패턴이 보입니다.

**버그는 거의 항상 액션에서 납니다.**

- DB에 두 번 저장됐다 → 액션이 잘못 호출됨
- 이메일이 안 갔다 → 액션이 실패함
- 타이밍이 맞지 않는다 → 액션의 순서가 잘못됨

계산은 테스트 한 번이면 신뢰할 수 있습니다. 데이터는 그냥 구조체입니다. 문제는 언제나 액션에 있습니다.

---

## 실전 예제: 주문 처리 시스템

처음부터 뒤섞인 코드를 보겠습니다.

### Before: 뒤섞인 코드

```rust
fn process_order(item: &str, quantity: u32, price: f64, email: &str) {
    // 계산인지 액션인지 불분명
    let total = quantity as f64 * price;
    let discounted = if total > 100.0 {
        total * 0.9
    } else {
        total
    };

    // 여기서 갑자기 외부 세계에 영향을 줌
    println!("Saving order: {} x{} = {:.2}", item, quantity, discounted);
    println!("Sending email to {}", email);
}
```

이 함수는 테스트하기 어렵습니다. 할인 로직이 맞는지 확인하려면 이메일 전송과 DB 저장까지 같이 실행됩니다. 할인율만 바꾸고 싶어도 함수 전체를 건드려야 합니다.

---

### After: 세 가지로 분리

**데이터부터 정의합니다.**

```rust
#[derive(Debug)]
struct Order {
    item: String,
    quantity: u32,
    price_per_unit: f64,
}

#[derive(Debug)]
struct PricedOrder {
    order: Order,
    total: f64,
}
```

**계산을 분리합니다. 순수하게, 외부 의존 없이.**

```rust
fn calculate_total(order: &Order) -> f64 {
    order.quantity as f64 * order.price_per_unit
}

fn apply_discount(total: f64) -> f64 {
    if total > 100.0 {
        total * 0.9
    } else {
        total
    }
}

fn build_priced_order(order: Order) -> PricedOrder {
    let raw_total = calculate_total(&order);
    let final_total = apply_discount(raw_total);
    PricedOrder { order, total: final_total }
}
```

**액션을 가장 바깥쪽으로 밀어냅니다.**

```rust
fn save_order(priced: &PricedOrder) {
    println!("[DB] Saving: {} x{} = {:.2}", 
        priced.order.item, priced.order.quantity, priced.total);
}

fn send_confirmation(email: &str, priced: &PricedOrder) {
    println!("[Email] To: {} — {} ordered for {:.2}", 
        email, priced.order.item, priced.total);
}

fn process_order(item: &str, quantity: u32, price: f64, email: &str) {
    // 액션은 딱 여기 한 곳에서만
    let order = Order {
        item: item.to_string(),
        quantity,
        price_per_unit: price,
    };
    let priced = build_priced_order(order);
    save_order(&priced);
    send_confirmation(email, &priced);
}
```

이제 `calculate_total`과 `apply_discount`는 인수만 넘겨서 테스트할 수 있습니다. DB나 이메일 서버 없이도.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_discount_under_100() {
        let order = Order {
            item: "pen".to_string(),
            quantity: 2,
            price_per_unit: 30.0,
        };
        assert_eq!(calculate_total(&order), 60.0);
        assert_eq!(apply_discount(60.0), 60.0);
    }

    #[test]
    fn test_discount_applied_over_100() {
        let order = Order {
            item: "notebook".to_string(),
            quantity: 3,
            price_per_unit: 50.0,
        };
        let total = calculate_total(&order);
        assert_eq!(apply_discount(total), 135.0); // 150 * 0.9
    }
}
```

---

## 핵심 원칙: 액션을 바깥으로 밀어라

이 설계 방식의 핵심 전략은 하나입니다.

> **액션을 코드의 가장 바깥쪽 레이어로 밀어냅니다.**

안쪽 레이어는 계산과 데이터로만 채웁니다. 바깥쪽 레이어가 액션을 조율합니다.

```
┌─────────────────────────────┐
│         액션 레이어          │  ← process_order, main
│  (DB 저장, 이메일, 시간 읽기) │
├─────────────────────────────┤
│         계산 레이어          │  ← calculate_total, apply_discount
│   (순수 함수, 변환 로직)      │
├─────────────────────────────┤
│         데이터 레이어         │  ← Order, PricedOrder
│    (구조체, 불변 값들)        │
└─────────────────────────────┘
```

이렇게 하면:
- 계산 레이어는 단위 테스트로 완전히 검증할 수 있습니다
- 액션 레이어는 얇아지고, 그 안에 로직이 없어 버그가 줄어듭니다
- 요구사항이 바뀌어도 계산 레이어만 수정하면 됩니다

---

## 분류 연습

다음 코드를 보고 액션/계산/데이터를 구분해보세요.

```rust
// 1번
fn add(a: i32, b: i32) -> i32 { a + b }

// 2번
fn read_user_input() -> String {
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    input
}

// 3번
struct Config { max_retries: u32, timeout_ms: u64 }

// 4번
fn format_greeting(name: &str) -> String {
    format!("Hello, {}!", name)
}

// 5번
fn log_event(msg: &str) {
    eprintln!("[LOG] {}", msg);
}
```

<details>
<summary>정답 보기</summary>

1. `add` → **계산** (같은 입력, 같은 출력, 외부 영향 없음)
2. `read_user_input` → **액션** (실행할 때마다 다른 값, 외부 세계에서 읽음)
3. `Config` → **데이터** (단순 구조체, 그 자체로는 아무 일도 안 함)
4. `format_greeting` → **계산** (순수 변환)
5. `log_event` → **액션** (stderr라는 외부 세계에 기록)

</details>

---

## 마치며

에릭 노먼드의 핵심 통찰은 간단합니다. **코드를 어렵게 만드는 건 대부분 액션입니다.** 액션을 최소화하고, 계산과 데이터로 최대한 표현하면 코드는 자연스럽게 단순해집니다.

이 글에서 다룬 개념은 Grokking Simplicity 1~4장의 핵심 아이디어입니다. 책은 JavaScript 예제를 쓰지만, 원칙은 언어를 가리지 않습니다. Rust처럼 타입이 엄격한 언어에서는 오히려 이 구분이 더 명확하게 드러납니다.

다음 단계로는 이런 주제들이 이어집니다:
- 계층형 설계: 계산들을 어떻게 쌓아 올릴까
- 타임라인 다이어그램: 액션들의 실행 순서를 어떻게 추론할까

---

*관련 글: [일급 함수와 클로저](/posts/programming/functional-first-class-functions/), [함수 컴포지션](/posts/programming/functional-composition/), [계층형 설계](/posts/programming/functional-stratified-design/), [타임라인 다이어그램](/posts/programming/functional-timeline-diagram/), [불변 데이터와 구조적 공유](/posts/programming/functional-immutable-data/), [온어니언 아키텍처](/posts/programming/functional-onion-architecture/)*

*참고: Eric Normand, [Grokking Simplicity](https://www.manning.com/books/grokking-simplicity), Manning Publications, 2021*
