---
title: "불변 데이터와 구조적 공유: 데이터를 안전하게 바꾸는 법"
date: 2026-04-28T20:30:00+09:00
draft: false
tags: ["함수형 프로그래밍", "Rust", "설계", "불변성", "Grokking Simplicity"]
categories: ["프로그래밍", "함수형 프로그래밍"]
description: "데이터를 직접 바꾸지 않고 새 값을 만드는 방식이 왜 안전한지, 그리고 Rust에서 어떻게 표현하는지 설명합니다."
---

## 이 글을 읽고 나면

- 왜 데이터를 직접 바꾸는 것이 위험한지 이해합니다.
- 불변 데이터와 구조적 공유가 무엇인지, 어떻게 동작하는지 압니다.
- Rust에서 불변 데이터를 표현하는 패턴을 익힙니다.

이전 글 [액션/계산/데이터](/posts/programming/functional-actions-calculations-data/)에서 데이터는 "언제 만들어도, 얼마나 복사해도 문제없다"고 했습니다. 이 글은 그 말이 어떻게 성립하는지 들여다봅니다.

---

## 문제: 데이터를 직접 바꾸면 생기는 일

장바구니에 상품을 추가하는 코드를 생각합니다.

```rust
fn add_item(cart: &mut Vec<String>, item: String) {
    cart.push(item);
}

fn main() {
    let mut cart = vec!["pen".to_string()];
    add_item(&mut cart, "notebook".to_string());
    println!("{:?}", cart); // ["pen", "notebook"]
}
```

잘 작동합니다. 하지만 이 코드에는 숨겨진 위험이 있습니다.

```rust
fn main() {
    let mut cart = vec!["pen".to_string()];
    let saved = &cart; // 저장해 둔 참조

    add_item(&mut cart, "notebook".to_string());

    // saved가 가리키는 장바구니도 바뀌어 있음
    println!("{:?}", saved); // ["pen", "notebook"] — 의도하지 않은 변경
}
```

`saved`는 변경 전 장바구니를 가리키려 했지만, `cart`를 바꾸자 `saved`도 바뀌었습니다. 데이터를 공유하면서 바꿨기 때문입니다.

이것이 에릭 노먼드가 말하는 **공유 가변 상태**의 문제입니다. 데이터를 공유하는 것도 안전하고, 데이터를 바꾸는 것도 안전하지만, **공유하면서 바꾸는 것**은 위험합니다.

---

## 해법: 바꾸지 말고 새로 만든다

함수형 접근은 원본을 바꾸지 않습니다. 변경된 새 값을 만들어 반환합니다.

```rust
fn add_item(cart: &[String], item: String) -> Vec<String> {
    let mut new_cart = cart.to_vec(); // 복사본 생성
    new_cart.push(item);
    new_cart // 새 값 반환
}

fn main() {
    let cart = vec!["pen".to_string()];
    let new_cart = add_item(&cart, "notebook".to_string());

    println!("{:?}", cart);     // ["pen"] — 원본 그대로
    println!("{:?}", new_cart); // ["pen", "notebook"] — 새 값
}
```

`add_item`은 이제 계산입니다. 입력을 바꾸지 않고, 새 데이터를 반환합니다. 같은 입력이면 항상 같은 결과가 나옵니다.

---

## 비용 문제: 매번 복사하면 느리지 않은가

"매번 전체를 복사하면 메모리와 시간이 낭비되지 않나?"라는 의문이 생깁니다.

실제로 단순한 `clone()`은 전체를 복사합니다. 작은 데이터라면 문제없지만, 큰 컬렉션이라면 비용이 됩니다.

함수형 언어들은 이 문제를 **구조적 공유(Structural Sharing)**로 해결합니다. 변경되지 않은 부분은 복사하지 않고 원본을 공유합니다.

```
원본:  [A] → [B] → [C]
새 값: [A] → [B] → [C] → [D]
                           ↑
                        새로 추가된 부분만 새로 만들고,
                        앞부분([A][B][C])은 공유
```

Rust에서는 `Arc`(Atomic Reference Counted)로 이 패턴을 구현합니다.

---

## `Arc`로 구조적 공유 표현하기

`Arc<T>`는 힙에 있는 값을 여러 곳에서 공유할 수 있게 합니다. 복사할 때 데이터를 복제하지 않고 참조 카운트만 올립니다.

```rust
use std::sync::Arc;

#[derive(Debug)]
struct CartItem {
    name: String,
    price: f64,
}

#[derive(Debug, Clone)]
struct Cart {
    items: Arc<Vec<CartItem>>,
}

impl Cart {
    fn new() -> Self {
        Cart { items: Arc::new(vec![]) }
    }

    fn add(&self, item: CartItem) -> Cart {
        // items의 내용을 복사해서 새 Vec 생성
        let mut new_items = (*self.items).clone();
        new_items.push(item);
        Cart { items: Arc::new(new_items) }
    }

    fn total(&self) -> f64 {
        self.items.iter().map(|i| i.price).sum()
    }
}
```

```rust
fn main() {
    let cart = Cart::new();
    let cart2 = cart.add(CartItem { name: "pen".into(), price: 1_000.0 });
    let cart3 = cart2.add(CartItem { name: "notebook".into(), price: 5_000.0 });

    println!("{:.0}", cart.total());  // 0
    println!("{:.0}", cart2.total()); // 1000
    println!("{:.0}", cart3.total()); // 6000
}
```

`cart`, `cart2`, `cart3`는 서로 독립된 값입니다. `cart2`를 만들어도 `cart`는 바뀌지 않습니다. 그러면서 `Arc`로 인해 공통 부분은 실제 메모리를 공유합니다.

---

## 중첩 데이터 업데이트하기

실무에서 자주 만나는 패턴은 중첩된 구조체의 일부를 바꾸는 경우입니다.

```rust
#[derive(Debug, Clone)]
struct Address {
    city: String,
    zip: String,
}

#[derive(Debug, Clone)]
struct User {
    name: String,
    email: String,
    address: Address,
}
```

도시만 바꾸고 싶을 때, 뮤터블 방식은 이렇습니다.

```rust
// 가변 방식: 원본을 직접 수정
fn update_city_mut(user: &mut User, city: String) {
    user.address.city = city;
}
```

불변 방식은 새 값을 만들어 반환합니다.

```rust
// 불변 방식: 새 값 반환
fn update_city(user: &User, city: String) -> User {
    User {
        address: Address {
            city,
            ..user.address.clone()  // 나머지는 그대로
        },
        ..user.clone()              // 나머지 필드는 그대로
    }
}
```

Rust의 구조체 업데이트 문법(`..`)이 여기서 자연스럽게 맞아떨어집니다. 바꾸고 싶은 부분만 명시하고, 나머지는 원본에서 가져옵니다.

```rust
fn main() {
    let user = User {
        name: "김연준".into(),
        email: "a@example.com".into(),
        address: Address { city: "서울".into(), zip: "04524".into() },
    };

    let moved = update_city(&user, "부산".into());

    println!("{}", user.address.city);  // 서울 — 원본 그대로
    println!("{}", moved.address.city); // 부산 — 새 값
}
```

---

## 불변 데이터가 계산과 만나는 지점

불변 데이터가 중요한 이유는 계산과 직접 연결됩니다.

계산의 조건은 "같은 입력이면 항상 같은 출력"입니다. 입력이 계산 도중에 바뀐다면 이 조건이 깨집니다.

```rust
// 나쁜 예: 계산 도중 외부에서 cart가 바뀔 수 있음
fn total_after_discount(cart: &mut Vec<f64>, rate: f64) -> f64 {
    cart.retain(|&p| p > 0.0); // 원본을 바꿈!
    cart.iter().sum::<f64>() * (1.0 - rate)
}

// 좋은 예: 입력을 바꾸지 않음
fn total_after_discount(prices: &[f64], rate: f64) -> f64 {
    prices.iter()
        .filter(|&&p| p > 0.0)
        .sum::<f64>() * (1.0 - rate)
}
```

두 번째 함수는 `prices`를 바꾸지 않습니다. 어디서 호출해도, 몇 번 호출해도 결과가 같습니다. 이것이 계산입니다.

---

## 언제 불변, 언제 가변?

불변 데이터가 항상 정답은 아닙니다. 판단 기준은 이렇습니다.

| 상황 | 추천 |
|---|---|
| 데이터를 여러 곳에서 참조 | 불변 — 공유 후 변경 위험 없음 |
| 함수 인자로 넘기는 데이터 | 불변 — 계산의 입력은 바뀌면 안 됨 |
| 타임라인이 여러 개 | 불변 — 경쟁 조건 원천 차단 |
| 성능이 중요한 루프 내부 | 가변 허용 — 단, 외부에 노출하지 않음 |
| 함수 내부 임시 변수 | 가변 허용 — 외부에서 공유되지 않음 |

함수 내부에서 임시로 가변 변수를 쓰더라도, 반환값이 불변 데이터라면 괜찮습니다. 중요한 것은 **공유되는 데이터**가 불변인지 여부입니다.

---

## 정리

불변 데이터의 핵심 원칙:

1. **바꾸지 말고 새로 만든다** — 원본을 보존하고, 변경된 새 값을 반환
2. **공유하려면 불변으로** — 공유와 변경을 동시에 하지 않는다
3. **구조적 공유로 비용을 줄인다** — `Arc`, 구조체 업데이트 문법으로 불필요한 복사 최소화
4. **계산의 입력은 항상 불변** — 입력이 바뀌면 계산이 아니다

에릭 노먼드는 데이터가 "안전한" 이유가 불변이기 때문이라고 말합니다. 데이터를 불변으로 유지하면, 공유해도 안전하고, 계산의 입력으로 써도 안전하고, 타임라인이 여러 개여도 안전합니다.

---

*관련 글: [액션/계산/데이터](/posts/programming/functional-actions-calculations-data/), [타임라인 다이어그램](/posts/programming/functional-timeline-diagram/), [함수 컴포지션](/posts/programming/functional-composition/), [온어니언 아키텍처](/posts/programming/functional-onion-architecture/), [로봇 경로 계획과 불변 데이터](/posts/programming/autonomous-path-planning/), [함수형 PID 제어기](/posts/programming/autonomous-pid-controller/)*
