---
title: "온어니언 아키텍처: 액션을 바깥으로, 계산을 안으로"
date: 2026-04-28T23:00:00+09:00
draft: false
tags: ["함수형 프로그래밍", "Rust", "설계", "아키텍처", "Grokking Simplicity"]
categories: ["프로그래밍"]
description: "액션을 시스템 바깥쪽에, 계산을 안쪽에 배치하면 테스트 가능하고 변경에 강한 구조가 됩니다. 에릭 노먼드의 온어니언 아키텍처를 Rust로 구현합니다."
---

## 이 글을 읽고 나면

- 온어니언 아키텍처가 무엇인지, 왜 함수형 설계의 자연스러운 귀결인지 이해합니다.
- 액션과 계산을 어떻게 물리적으로 분리하는지 봅니다.
- 실무 수준의 예제로 전체 구조를 조립하는 방법을 익힙니다.

이 글은 [액션/계산/데이터](/posts/programming/functional-actions-calculations-data/), [계층형 설계](/posts/programming/functional-stratified-design/), [함수형 DI](/posts/programming/functional-dependency-injection/)를 토대로 합니다. 세 글의 원칙들이 하나의 아키텍처로 합쳐지는 지점입니다.

---

## 문제: 액션이 계산 안에 숨어 있다

대부분의 코드는 이렇게 생겼습니다.

```rust
fn process_order(order_id: u32) -> Result<(), String> {
    // DB 읽기 (액션)
    let order = db::fetch_order(order_id)?;

    // 재고 확인 (액션)
    let stock = inventory::check_stock(&order.item)?;

    // 할인 계산 (계산이 될 수 있었지만...)
    let discount = if order.is_vip {
        discount_service::fetch_rate(order_id)? // 액션이 끼어듦
    } else {
        0.0
    };

    // 총액 계산 (계산)
    let total = order.price * (1.0 - discount);

    // 이메일 발송 (액션)
    email::send(&order.user_email, total)?;

    // DB 저장 (액션)
    db::save_order(order_id, total)?;

    Ok(())
}
```

액션과 계산이 뒤섞여 있습니다. 이 함수를 테스트하려면 DB, 재고 시스템, 할인 서비스, 이메일 서버가 모두 필요합니다. 어디서 실패했는지도 추적하기 어렵습니다.

에릭 노먼드는 이것을 "액션이 계산 안에 숨어 있는" 상태라고 부릅니다.

---

## 해법: 양파처럼 겹겹이 쌓기

온어니언 아키텍처(Onion Architecture)는 시스템을 동심원으로 배치합니다.

```
┌─────────────────────────────────────┐
│           액션 (바깥층)              │
│   DB, 네트워크, 파일, 이메일         │
│  ┌───────────────────────────────┐  │
│  │        조율 (중간층)           │  │
│  │   순서 제어, 데이터 조립       │  │
│  │  ┌─────────────────────────┐  │  │
│  │  │      계산 (안쪽층)       │  │  │
│  │  │  순수 함수, 비즈니스 룰  │  │  │
│  │  └─────────────────────────┘  │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

규칙은 하나입니다: **의존은 바깥에서 안으로만 흐릅니다.** 안쪽 계산은 바깥쪽 액션을 알지 못합니다.

- **안쪽층(계산)**: 순수 함수. 외부 시스템을 모릅니다. 테스트가 쉽습니다.
- **중간층(조율)**: 데이터를 모아 계산에 넘기고, 결과를 받아 액션에 전달합니다.
- **바깥층(액션)**: DB, 네트워크, 파일 등 외부 세계와 접촉합니다.

---

## 실전 예제: 주문 처리 시스템

주문 처리를 온어니언 아키텍처로 재구성합니다.

### 도메인 타입 정의

```rust
#[derive(Debug, Clone)]
struct Order {
    id: u32,
    user_email: String,
    item: String,
    price: f64,
    is_vip: bool,
}

#[derive(Debug, Clone)]
struct ProcessedOrder {
    order_id: u32,
    total: f64,
    email: String,
}
```

---

### 안쪽층: 계산 (순수 함수)

외부 의존이 없습니다. `use` 없이 자립합니다.

```rust
mod domain {
    use super::{Order, ProcessedOrder};

    pub fn apply_discount(price: f64, is_vip: bool, discount_rate: f64) -> f64 {
        if is_vip {
            price * (1.0 - discount_rate)
        } else {
            price
        }
    }

    pub fn validate_stock(stock: u32, item: &str) -> Result<(), String> {
        if stock == 0 {
            Err(format!("재고 없음: {}", item))
        } else {
            Ok(())
        }
    }

    pub fn build_processed_order(order: &Order, total: f64) -> ProcessedOrder {
        ProcessedOrder {
            order_id: order.id,
            total,
            email: order.user_email.clone(),
        }
    }

    pub fn render_email(order: &ProcessedOrder) -> String {
        format!(
            "주문 #{} 확인되었습니다. 결제 금액: {:.0}원",
            order.order_id, order.total
        )
    }
}
```

이 함수들은 DB도, 네트워크도 모릅니다. 인자만 주면 결과가 나옵니다.

```rust
#[cfg(test)]
mod tests {
    use super::domain;

    #[test]
    fn vip_gets_discount() {
        let total = domain::apply_discount(10_000.0, true, 0.1);
        assert_eq!(total, 9_000.0);
    }

    #[test]
    fn non_vip_pays_full() {
        let total = domain::apply_discount(10_000.0, false, 0.1);
        assert_eq!(total, 10_000.0);
    }

    #[test]
    fn out_of_stock_returns_error() {
        assert!(domain::validate_stock(0, "pen").is_err());
        assert!(domain::validate_stock(5, "pen").is_ok());
    }
}
```

외부 시스템 없이 테스트가 완결됩니다.

---

### 바깥층: 액션 (외부 세계와의 접촉)

```rust
mod infra {
    use super::Order;

    pub fn fetch_order(order_id: u32) -> Result<Order, String> {
        // 실제 구현에서는 DB 쿼리
        Ok(Order {
            id: order_id,
            user_email: "user@example.com".into(),
            item: "notebook".into(),
            price: 15_000.0,
            is_vip: true,
        })
    }

    pub fn fetch_stock(item: &str) -> Result<u32, String> {
        // 실제 구현에서는 재고 시스템 API 호출
        let _ = item;
        Ok(3)
    }

    pub fn fetch_discount_rate(order_id: u32) -> Result<f64, String> {
        // 실제 구현에서는 할인 서비스 호출
        let _ = order_id;
        Ok(0.1)
    }

    pub fn send_email(to: &str, body: &str) -> Result<(), String> {
        println!("[이메일] to: {} | body: {}", to, body);
        Ok(())
    }

    pub fn save_order(processed: &super::ProcessedOrder) -> Result<(), String> {
        println!("[DB 저장] {:?}", processed);
        Ok(())
    }
}
```

---

### 중간층: 조율 (데이터 수집 → 계산 → 결과 전달)

```rust
fn process_order(order_id: u32) -> Result<(), String> {
    // 1. 데이터 수집 (액션)
    let order         = infra::fetch_order(order_id)?;
    let stock         = infra::fetch_stock(&order.item)?;
    let discount_rate = if order.is_vip {
        infra::fetch_discount_rate(order_id)?
    } else {
        0.0
    };

    // 2. 계산 (순수 함수 호출)
    domain::validate_stock(stock, &order.item)?;
    let total = domain::apply_discount(order.price, order.is_vip, discount_rate);
    let processed = domain::build_processed_order(&order, total);
    let email_body = domain::render_email(&processed);

    // 3. 결과 전달 (액션)
    infra::send_email(&processed.email, &email_body)?;
    infra::save_order(&processed)?;

    Ok(())
}
```

조율 함수의 구조가 명확합니다: **수집 → 계산 → 전달**. 세 단계가 섞이지 않습니다.

```rust
fn main() {
    match process_order(42) {
        Ok(()) => println!("주문 처리 완료"),
        Err(e) => println!("오류: {}", e),
    }
}
```

---

## 테스트 전략: 계층별로 다르게

온어니언 아키텍처는 테스트 전략도 층마다 다릅니다.

```
안쪽층(계산)  →  단위 테스트 — 빠르고, 외부 의존 없음
중간층(조율)  →  함수형 DI로 주입 — infra를 클로저로 교체
바깥층(액션)  →  통합 테스트 — 실제 DB/네트워크 필요, 수 적게
```

중간층 조율 함수를 테스트할 때는 [함수형 DI](/posts/programming/functional-dependency-injection/) 패턴으로 infra를 교체합니다.

```rust
fn process_order_with<FO, FS, FD, FE, FSave>(
    order_id: u32,
    fetch_order: FO,
    fetch_stock: FS,
    fetch_discount: FD,
    send_email: FE,
    save_order: FSave,
) -> Result<(), String>
where
    FO: Fn(u32) -> Result<Order, String>,
    FS: Fn(&str) -> Result<u32, String>,
    FD: Fn(u32) -> Result<f64, String>,
    FE: Fn(&str, &str) -> Result<(), String>,
    FSave: Fn(&ProcessedOrder) -> Result<(), String>,
{
    let order         = fetch_order(order_id)?;
    let stock         = fetch_stock(&order.item)?;
    let discount_rate = if order.is_vip { fetch_discount(order_id)? } else { 0.0 };

    domain::validate_stock(stock, &order.item)?;
    let total     = domain::apply_discount(order.price, order.is_vip, discount_rate);
    let processed = domain::build_processed_order(&order, total);
    let body      = domain::render_email(&processed);

    send_email(&processed.email, &body)?;
    save_order(&processed)?;

    Ok(())
}

#[cfg(test)]
mod orchestration_tests {
    use super::*;

    #[test]
    fn vip_order_applies_discount_and_saves() {
        let mut saved: Option<ProcessedOrder> = None;
        let mut sent_to = String::new();

        let result = process_order_with(
            1,
            |_| Ok(Order { id: 1, user_email: "vip@test.com".into(),
                           item: "pen".into(), price: 10_000.0, is_vip: true }),
            |_| Ok(5),
            |_| Ok(0.2),
            |to, _| { sent_to = to.to_string(); Ok(()) },
            |p| { saved = Some(p.clone()); Ok(()) },
        );

        assert!(result.is_ok());
        assert_eq!(saved.unwrap().total, 8_000.0);
        assert_eq!(sent_to, "vip@test.com");
    }

    #[test]
    fn out_of_stock_returns_error_before_sending_email() {
        let mut email_sent = false;

        let result = process_order_with(
            1,
            |_| Ok(Order { id: 1, user_email: "a@test.com".into(),
                           item: "pen".into(), price: 10_000.0, is_vip: false }),
            |_| Ok(0), // 재고 없음
            |_| Ok(0.0),
            |_, _| { email_sent = true; Ok(()) },
            |_| Ok(()),
        );

        assert!(result.is_err());
        assert!(!email_sent); // 이메일이 발송되지 않았는지 확인
    }
}
```

---

## 온어니언 아키텍처가 주는 것

| 문제 | 온어니언 아키텍처의 해법 |
|---|---|
| 테스트에 DB가 필요 | 계산은 DB를 모름 — 단독 테스트 가능 |
| 어디서 실패했는지 모름 | 층이 분리되어 에러 발생 위치가 명확 |
| 비즈니스 로직이 흩어짐 | 안쪽층에만 집중 |
| 외부 서비스가 바뀌면 전체 수정 | 바깥층만 교체, 계산은 무변 |

---

## Grokking Simplicity와의 연결

에릭 노먼드는 이 구조를 명시적으로 권장합니다.

> "액션을 줄이고, 계산을 늘리고, 액션을 바깥쪽으로 밀어내라."

온어니언 아키텍처는 이 원칙의 아키텍처 수준 표현입니다.

- [ACD 구분](/posts/programming/functional-actions-calculations-data/)은 코드 한 줄 단위의 원칙
- [계층형 설계](/posts/programming/functional-stratified-design/)는 함수 단위의 원칙
- **온어니언 아키텍처**는 시스템 전체 단위의 원칙

세 원칙이 같은 방향을 가리킵니다: **계산을 중심에, 액션을 가장자리에.**

---

## 정리

온어니언 아키텍처의 핵심 규칙:

1. **안쪽(계산)은 바깥(액션)을 모른다** — 의존은 바깥→안 방향으로만
2. **조율 함수는 수집 → 계산 → 전달** 순서를 지킨다
3. **테스트는 층마다 다르게** — 계산은 단위 테스트, 조율은 DI, 바깥은 통합 테스트

이 구조를 갖추면 외부 서비스가 바뀌어도 비즈니스 로직은 그대로입니다. 테스트는 외부 의존 없이 빠르게 돌아갑니다. 에러가 나면 어느 층에서 났는지 즉시 알 수 있습니다.

---

*관련 글: [액션/계산/데이터](/posts/programming/functional-actions-calculations-data/), [계층형 설계](/posts/programming/functional-stratified-design/), [함수형 DI](/posts/programming/functional-dependency-injection/), [타임라인 다이어그램](/posts/programming/functional-timeline-diagram/)*
