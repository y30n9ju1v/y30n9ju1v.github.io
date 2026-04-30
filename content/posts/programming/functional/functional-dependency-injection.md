---
title: "함수형 DI: 모킹 없이 테스트 가능한 코드 만들기"
date: 2026-04-28T14:00:00+09:00
draft: false
tags: ["함수형 프로그래밍", "Rust", "설계", "의존성 주입", "Grokking Simplicity"]
categories: ["프로그래밍", "함수형 프로그래밍"]
description: "인터페이스도, 모킹 프레임워크도 없이 액션을 함수 인자로 넘기면 테스트하기 쉬운 코드가 됩니다."
---

## 이 글을 읽고 나면

- 왜 액션이 코드 안에 박혀 있으면 테스트가 어려운지 이해합니다.
- 액션을 함수 인자로 넘기는 것만으로 테스트 가능성이 어떻게 달라지는지 봅니다.
- 전통적인 DI(의존성 주입)와 함수형 DI의 차이를 압니다.

이 글은 [액션/계산/데이터](/posts/programming/functional-actions-calculations-data/) 시리즈와 이어집니다.

---

## 문제: 액션이 안쪽에 박혀 있으면 테스트가 안 된다

주문 확인 이메일을 보내는 코드를 생각해봅니다.

```rust
fn confirm_order(order_id: u32) {
    let order = fetch_order_from_db(order_id); // DB 조회 — 액션
    let email = render_email(&order);          // 이메일 생성 — 계산
    send_email(&order.user_email, &email);     // 이메일 전송 — 액션
    mark_order_confirmed(order_id);            // DB 갱신 — 액션
}
```

`render_email`이 올바른 내용을 만드는지 테스트하고 싶습니다. 그런데 이 함수를 테스트하려면 `confirm_order` 전체를 실행해야 하고, 그러면 DB와 이메일 서버가 필요합니다.

액션이 계산 안에 박혀 있기 때문입니다.

---

## 전통적인 해법: 인터페이스와 모킹

객체지향에서는 이 문제를 인터페이스로 풉니다.

```rust
trait EmailSender {
    fn send(&self, to: &str, body: &str);
}

struct RealEmailSender;
impl EmailSender for RealEmailSender {
    fn send(&self, to: &str, body: &str) { /* 실제 전송 */ }
}

struct MockEmailSender { pub sent: Vec<(String, String)> }
impl EmailSender for MockEmailSender {
    fn send(&self, to: &str, body: &str) {
        // 전송 대신 기록
    }
}

struct OrderService<E: EmailSender> {
    sender: E,
}

impl<E: EmailSender> OrderService<E> {
    fn confirm_order(&self, order_id: u32) {
        // ...
        self.sender.send(&order.user_email, &email);
    }
}
```

작동은 합니다. 하지만 테스트 하나를 위해 트레잇, 구조체 두 개, 제네릭이 생겼습니다. 코드가 늘어났고, 테스트 대상이 아닌 구조가 커졌습니다.

---

## 함수형 해법: 액션을 인자로 넘기기

함수형 접근은 더 단순합니다. **액션을 함수 인자로 넘깁니다.**

```rust
fn confirm_order(
    order_id: u32,
    fetch_order: impl Fn(u32) -> Order,
    send_email:  impl Fn(&str, &str),
    mark_confirmed: impl Fn(u32),
) {
    let order = fetch_order(order_id);
    let email = render_email(&order);        // 계산 — 변하지 않음
    send_email(&order.user_email, &email);
    mark_confirmed(order_id);
}
```

프로덕션에서는 실제 함수를 넘깁니다.

```rust
confirm_order(
    42,
    fetch_order_from_db,
    send_real_email,
    mark_order_confirmed,
);
```

테스트에서는 클로저를 넘깁니다.

```rust
#[test]
fn test_confirm_order_sends_correct_email() {
    let sent = std::cell::RefCell::new(vec![]);

    confirm_order(
        1,
        |_id| Order { user_email: "user@test.com".into(), item: "notebook".into() },
        |to, body| sent.borrow_mut().push((to.to_string(), body.to_string())),
        |_id| {}, // mark_confirmed는 이 테스트에서 관심 없음
    );

    let emails = sent.borrow();
    assert_eq!(emails.len(), 1);
    assert!(emails[0].1.contains("notebook"));
}
```

트레잇도, 모킹 구조체도 없습니다. 클로저 하나로 액션을 교체했습니다.

---

## 핵심 원칙: 계산은 고정하고, 액션은 주입하라

`confirm_order`에서 변하지 않는 부분과 변할 수 있는 부분을 나눠보면:

| 부분 | 종류 | 고정/주입 |
|---|---|---|
| `render_email` | 계산 | 고정 — 항상 같은 로직 |
| `fetch_order` | 액션 | 주입 — DB / 테스트 픽스처 |
| `send_email` | 액션 | 주입 — SMTP / 기록용 클로저 |
| `mark_confirmed` | 액션 | 주입 — DB / no-op |

계산은 함수 안에 두고, 액션은 밖에서 넘겨받습니다. 이것이 함수형 DI의 전부입니다.

---

## 더 복잡한 예제: 여러 단계의 파이프라인

주문 처리 파이프라인이 여러 단계로 이루어진 경우입니다.

```rust
struct Order {
    id: u32,
    user_email: String,
    item: String,
    quantity: u32,
}

// 계산: 재고 부족 여부 판단
fn is_out_of_stock(stock: u32, quantity: u32) -> bool {
    stock < quantity
}

// 계산: 영수증 문자열 생성
fn render_receipt(order: &Order, total: f64) -> String {
    format!("[영수증] {}님, {} {}개 = {:.0}원", order.user_email, order.item, order.quantity, total)
}

// 계산: 총액 계산
fn calculate_total(price_per_unit: f64, quantity: u32) -> f64 {
    price_per_unit * quantity as f64
}
```

파이프라인 함수는 계산을 직접 호출하고, 액션은 인자로 받습니다.

```rust
fn process_order(
    order: &Order,
    fetch_stock:    impl Fn(u32) -> u32,
    fetch_price:    impl Fn(&str) -> f64,
    send_receipt:   impl Fn(&str, &str),
    save_order:     impl Fn(u32),
    notify_failure: impl Fn(u32, &str),
) {
    let stock = fetch_stock(order.id);
    let price = fetch_price(&order.item);

    if is_out_of_stock(stock, order.quantity) {
        notify_failure(order.id, "재고 부족");
        return;
    }

    let total   = calculate_total(price, order.quantity);
    let receipt = render_receipt(order, total);

    send_receipt(&order.user_email, &receipt);
    save_order(order.id);
}
```

테스트는 각 시나리오를 클로저로 조립합니다.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn sample_order() -> Order {
        Order { id: 1, user_email: "a@test.com".into(), item: "pen".into(), quantity: 3 }
    }

    #[test]
    fn test_out_of_stock_notifies_failure() {
        let notified = std::cell::Cell::new(false);

        process_order(
            &sample_order(),
            |_| 1,           // 재고 1개
            |_| 500.0,
            |_, _| panic!("재고 없으면 영수증 보내면 안 됨"),
            |_| panic!("재고 없으면 저장하면 안 됨"),
            |_, msg| {
                assert_eq!(msg, "재고 부족");
                notified.set(true);
            },
        );

        assert!(notified.get());
    }

    #[test]
    fn test_receipt_contains_total() {
        let receipt_sent = std::cell::RefCell::new(String::new());

        process_order(
            &sample_order(),
            |_| 10,          // 재고 충분
            |_| 500.0,       // 단가 500원
            |_, body| *receipt_sent.borrow_mut() = body.to_string(),
            |_| {},
            |_, _| panic!("실패 알림 오면 안 됨"),
        );

        assert!(receipt_sent.borrow().contains("1500"));  // 500 * 3
    }
}
```

각 테스트가 한 가지만 검증합니다. DB도, 이메일 서버도 없습니다.

---

## 언제 트레잇을 쓰고 언제 함수 인자를 쓸까

함수형 DI가 항상 정답은 아닙니다. 둘을 비교하면 이렇습니다.

| | 함수 인자 | 트레잇 |
|---|---|---|
| 코드량 | 적음 | 많음 |
| 유연성 | 호출마다 다른 액션 | 구조체 생성 시 고정 |
| 상태 공유 | 클로저 캡처로 가능 | 구조체 필드로 |
| 적합한 경우 | 단순한 액션 1~3개 | 액션이 많거나 여러 메서드에서 재사용 |

함수 하나에 액션이 1~3개라면 함수 인자가 낫습니다. 액션이 많아지거나 여러 함수에서 같은 액션 묶음을 공유한다면 트레잇으로 묶는 것이 더 명확합니다.

---

## 정리

함수형 DI의 핵심은 단순합니다.

> **계산은 함수 안에 둔다. 액션은 밖에서 받는다.**

이렇게 하면:
- 테스트에서 클로저 하나로 액션을 교체할 수 있습니다
- 트레잇과 모킹 구조체 없이도 테스트가 가능합니다
- 함수 시그니처만 봐도 어떤 액션에 의존하는지 드러납니다

앞선 글들의 원칙과 연결하면, 이것은 "액션을 바깥으로 밀어내기"의 자연스러운 결론입니다. 바깥으로 밀어낸 액션을 인자로 받으면, 호출하는 쪽이 어떤 액션을 쓸지 결정합니다. 프로덕션에서는 실제 액션을, 테스트에서는 기록용 클로저를 넘깁니다.

---

*관련 글: [액션/계산/데이터](/posts/programming/functional-actions-calculations-data/), [일급 함수와 클로저](/posts/programming/functional-first-class-functions/), [계층형 설계](/posts/programming/functional-stratified-design/), [온어니언 아키텍처](/posts/programming/functional-onion-architecture/), [자율주행 모드 전이를 타입으로 만들기](/posts/programming/autonomous-state-machine/), [ROS2 콜백을 함수형으로](/posts/programming/autonomous-ros2-functional/), [시뮬레이션 회귀 테스트 설계](/posts/programming/autonomous-simulation-regression/)*
