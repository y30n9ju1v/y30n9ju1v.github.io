---
title: "함수형으로 상태 기계 만들기: enum과 패턴 매칭"
date: 2026-04-28T18:00:00+09:00
draft: false
tags: ["함수형 프로그래밍", "Rust", "설계", "상태 기계", "Grokking Simplicity"]
categories: ["프로그래밍"]
description: "enum과 패턴 매칭으로 상태 전이를 표현하면, 컴파일러가 빠뜨린 상태를 잡아주는 안전한 상태 기계가 됩니다."
---

## 이 글을 읽고 나면

- 상태 기계를 `enum`으로 표현하는 방법을 이해합니다.
- 잘못된 상태 전이를 런타임이 아니라 컴파일 타임에 막는 방법을 압니다.
- 함수형 설계 원칙(계산/액션 분리)이 상태 기계에 어떻게 적용되는지 봅니다.

이전 글 [액션/계산/데이터](/posts/programming/functional-actions-calculations-data/)와 [계층형 설계](/posts/programming/functional-stratified-design/)를 먼저 읽으면 더 자연스럽게 이어집니다.

---

## 상태 기계란

상태 기계(State Machine)는 **유한한 상태**와 **상태 사이의 전이 규칙**으로 이루어진 모델입니다.

온라인 주문을 예로 들면:

```
주문접수 → 결제완료 → 배송중 → 배송완료
                ↓
             결제실패
```

각 상태에서 가능한 전이가 정해져 있습니다. `배송완료` 상태에서 `결제실패`로 가는 것은 불가능합니다. 이 규칙을 코드로 어떻게 표현하느냐가 핵심입니다.

---

## 나쁜 방법: 문자열이나 정수로 상태 표현

많은 코드에서 상태를 이렇게 표현합니다.

```rust
struct Order {
    id: u32,
    status: String, // "pending", "paid", "shipped", "delivered", "failed"
}

fn transition(order: &mut Order, event: &str) {
    match (order.status.as_str(), event) {
        ("pending", "pay")     => order.status = "paid".into(),
        ("paid",    "ship")    => order.status = "shipped".into(),
        ("shipped", "deliver") => order.status = "delivered".into(),
        ("pending", "fail")    => order.status = "failed".into(),
        _ => println!("잘못된 전이"), // 런타임에 발견
    }
}
```

문제가 여러 가지입니다.

- `"piad"`처럼 오타를 내도 컴파일러가 모릅니다
- 어떤 상태가 존재하는지 타입만 봐서는 알 수 없습니다
- 잘못된 전이가 런타임에서야 발견됩니다
- `match`의 `_` 브랜치가 조용히 실패를 삼킵니다

---

## 좋은 방법: `enum`으로 상태를 타입으로 만들기

상태를 `enum`으로 표현하면 컴파일러가 동맹이 됩니다.

```rust
#[derive(Debug, Clone, PartialEq)]
enum OrderState {
    Pending,
    Paid,
    Shipped,
    Delivered,
    Failed { reason: String },
}
```

상태가 타입이 되는 순간 세 가지가 달라집니다.

1. 오타는 컴파일 에러
2. `match`에서 빠뜨린 상태는 컴파일 에러
3. `Failed`처럼 상태마다 다른 데이터를 담을 수 있음

---

## 이벤트도 `enum`으로

상태 전이를 일으키는 이벤트도 같은 방식으로 표현합니다.

```rust
#[derive(Debug)]
enum OrderEvent {
    PaymentSucceeded,
    PaymentFailed { reason: String },
    Shipped,
    Delivered,
}
```

---

## 전이 함수: 계산으로 표현하기

상태 전이는 **계산**입니다. 현재 상태와 이벤트를 받아서 다음 상태를 돌려줍니다. 외부 상태를 바꾸지 않습니다.

```rust
fn transition(state: OrderState, event: OrderEvent) -> Result<OrderState, String> {
    match (state, event) {
        (OrderState::Pending, OrderEvent::PaymentSucceeded) => {
            Ok(OrderState::Paid)
        }
        (OrderState::Pending, OrderEvent::PaymentFailed { reason }) => {
            Ok(OrderState::Failed { reason })
        }
        (OrderState::Paid, OrderEvent::Shipped) => {
            Ok(OrderState::Shipped)
        }
        (OrderState::Shipped, OrderEvent::Delivered) => {
            Ok(OrderState::Delivered)
        }
        (state, event) => Err(format!(
            "{:?} 상태에서 {:?} 이벤트는 허용되지 않습니다", state, event
        )),
    }
}
```

`transition`은 순수한 계산입니다.

- 같은 `(상태, 이벤트)` 조합이면 항상 같은 결과
- DB도, 이메일도, 로그도 없음
- 테스트가 trivial함

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_payment_success() {
        let next = transition(OrderState::Pending, OrderEvent::PaymentSucceeded).unwrap();
        assert_eq!(next, OrderState::Paid);
    }

    #[test]
    fn test_invalid_transition() {
        let result = transition(OrderState::Delivered, OrderEvent::Shipped);
        assert!(result.is_err());
    }

    #[test]
    fn test_payment_failure_carries_reason() {
        let next = transition(
            OrderState::Pending,
            OrderEvent::PaymentFailed { reason: "잔액 부족".into() },
        ).unwrap();
        assert_eq!(next, OrderState::Failed { reason: "잔액 부족".into() });
    }
}
```

---

## 부수효과(액션)는 전이 바깥에서

상태가 바뀔 때 이메일을 보내거나 DB를 갱신하는 일이 필요합니다. 이것은 액션입니다. 전이 함수 안에 넣으면 계산이 액션으로 오염됩니다.

대신 **전이 결과를 보고 액션을 결정하는 레이어**를 바깥에 둡니다.

```rust
fn apply_event(
    state: OrderState,
    event: OrderEvent,
    save: impl Fn(&OrderState),
    notify: impl Fn(&str),
) -> Result<OrderState, String> {
    let next = transition(state, event)?;   // 계산: 다음 상태 결정

    // 액션: 새 상태에 따라 부수효과 실행
    match &next {
        OrderState::Paid => {
            notify("결제가 완료되었습니다.");
        }
        OrderState::Shipped => {
            notify("상품이 발송되었습니다.");
        }
        OrderState::Failed { reason } => {
            notify(&format!("결제 실패: {}", reason));
        }
        _ => {}
    }

    save(&next);  // DB 저장은 항상
    Ok(next)
}
```

`apply_event`는 세 단계로 명확히 나뉩니다.

```
[계산] transition → 다음 상태 결정
[계산] match next → 어떤 알림을 보낼지 결정
[액션] notify, save → 외부 세계에 반영
```

---

## 타입으로 불가능한 전이 막기

지금 구현은 잘못된 전이를 `Result::Err`로 돌려줍니다. 런타임에서야 알 수 있습니다.

더 나아가면 **타입 자체로 잘못된 전이를 불가능하게** 만들 수 있습니다. 각 상태를 별도 타입으로 만들고, 해당 상태에서만 존재하는 전이 메서드를 구현합니다.

```rust
struct Pending  { order_id: u32 }
struct Paid     { order_id: u32 }
struct Shipped  { order_id: u32 }
struct Delivered { order_id: u32 }
struct Failed   { order_id: u32, reason: String }

impl Pending {
    fn pay(self) -> Paid {
        Paid { order_id: self.order_id }
    }
    fn fail(self, reason: String) -> Failed {
        Failed { order_id: self.order_id, reason }
    }
}

impl Paid {
    fn ship(self) -> Shipped {
        Shipped { order_id: self.order_id }
    }
}

impl Shipped {
    fn deliver(self) -> Delivered {
        Delivered { order_id: self.order_id }
    }
}
```

이제 `Delivered`에서 `ship()`을 호출하는 코드는 **컴파일 자체가 되지 않습니다.** 해당 타입에 그 메서드가 없기 때문입니다.

```rust
fn main() {
    let order = Pending { order_id: 1 };
    let paid  = order.pay();
    let shipped = paid.ship();
    let done  = shipped.deliver();

    // done.ship(); // 컴파일 에러: Delivered에 ship() 없음
}
```

이 패턴을 **타입스테이트(Typestate) 패턴**이라고 합니다. 상태 전이 규칙이 타입 시스템 안에 내재됩니다.

---

## 두 방식 비교

| | `enum` 기반 | 타입스테이트 |
|---|---|---|
| 잘못된 전이 감지 | 런타임 (`Result::Err`) | 컴파일 타임 |
| 여러 상태를 하나의 변수로 | 가능 (`Vec<OrderState>`) | 어려움 |
| 구현 복잡도 | 낮음 | 높음 |
| 적합한 경우 | 상태가 많고 동적인 경우 | 전이 규칙이 엄격하고 중요한 경우 |

대부분의 실무에서는 `enum` 기반으로 시작하는 것이 낫습니다. 전이 규칙이 매우 중요한 도메인(결제, 인증, 프로토콜 구현)에서는 타입스테이트를 고려합니다.

---

## 실무 패턴: 이벤트 소싱과 연결하기

상태 기계는 이벤트 소싱(Event Sourcing)과 자연스럽게 맞습니다. 현재 상태를 직접 저장하는 대신, 이벤트 목록을 저장하고 재생해서 상태를 복원합니다.

```rust
fn replay(events: &[OrderEvent]) -> Result<OrderState, String> {
    events.iter().cloned().try_fold(
        OrderState::Pending,
        |state, event| transition(state, event),
    )
}

fn main() {
    let history = vec![
        OrderEvent::PaymentSucceeded,
        OrderEvent::Shipped,
        OrderEvent::Delivered,
    ];

    let final_state = replay(&history).unwrap();
    println!("{:?}", final_state); // Delivered
}
```

`transition`이 순수한 계산이기 때문에 이벤트 목록만 있으면 언제든 상태를 재현할 수 있습니다. 디버깅, 감사 로그, 타임트래블 디버깅이 모두 자연스럽게 따라옵니다.

---

## 정리

`enum` 기반 상태 기계의 핵심 원칙:

1. **상태와 이벤트를 `enum`으로** — 타입이 문서가 되고, 오타는 컴파일 에러
2. **전이 함수는 순수한 계산으로** — 테스트가 쉽고, 이벤트 소싱과 자연스럽게 연결
3. **부수효과는 전이 바깥에서** — 계산과 액션의 경계를 명확히 유지
4. **`match`의 완전성 검사를 믿어라** — 빠뜨린 상태를 컴파일러가 잡아줌

---

*관련 글: [액션/계산/데이터](/posts/programming/functional-actions-calculations-data/), [계층형 설계](/posts/programming/functional-stratified-design/), [함수형 DI](/posts/programming/functional-dependency-injection/)*
