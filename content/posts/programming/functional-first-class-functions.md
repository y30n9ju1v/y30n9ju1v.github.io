---
title: "일급 함수와 클로저: 함수를 값처럼 다루기"
date: 2026-04-28T21:00:00+09:00
draft: false
tags: ["함수형 프로그래밍", "Rust", "설계", "클로저", "Grokking Simplicity"]
categories: ["프로그래밍", "함수형 프로그래밍"]
description: "함수를 값처럼 저장하고, 넘기고, 반환할 수 있으면 코드가 어떻게 달라지는지 Rust 예제로 설명합니다."
---

## 이 글을 읽고 나면

- 일급 함수가 무엇인지, 왜 함수형 프로그래밍의 기반인지 이해합니다.
- 클로저가 일반 함수와 어떻게 다른지 압니다.
- 함수를 값처럼 다루는 것이 컴포지션, DI와 어떻게 연결되는지 봅니다.

이 글은 [함수 컴포지션](/posts/programming/functional-composition/)과 [함수형 DI](/posts/programming/functional-dependency-injection/)의 개념적 토대입니다. 두 글을 먼저 읽었다면 "왜 그게 가능한가"에 대한 답이 이 글에 있습니다.

---

## 일급 함수란

"일급(First-class)"이라는 말은 어떤 값이 언어 안에서 특별한 제약 없이 다뤄질 수 있다는 뜻입니다. 정수나 문자열처럼:

- 변수에 저장할 수 있고
- 함수의 인자로 넘길 수 있고
- 함수의 반환값이 될 수 있다

Rust에서 함수는 일급입니다.

```rust
fn double(x: i32) -> i32 { x * 2 }
fn add_one(x: i32) -> i32 { x + 1 }

fn apply(f: fn(i32) -> i32, value: i32) -> i32 {
    f(value)
}

fn main() {
    let f = double;              // 변수에 저장
    println!("{}", f(3));        // 6

    println!("{}", apply(double, 5));   // 10 — 인자로 전달
    println!("{}", apply(add_one, 5));  // 6
}
```

`double`이라는 함수를 변수 `f`에 담고, `apply`의 인자로 넘겼습니다. 숫자를 다루듯 함수를 다뤘습니다.

---

## 클로저: 주변 값을 기억하는 함수

클로저(Closure)는 자신이 정의된 환경의 변수를 **캡처**할 수 있는 함수입니다.

```rust
fn make_adder(n: i32) -> impl Fn(i32) -> i32 {
    move |x| x + n   // n을 캡처
}

fn main() {
    let add5  = make_adder(5);
    let add10 = make_adder(10);

    println!("{}", add5(3));  // 8
    println!("{}", add10(3)); // 13
}
```

`make_adder`는 함수를 반환합니다. 반환된 클로저는 `n`을 기억합니다. `add5`와 `add10`은 각각 다른 `n`을 캡처한 독립된 함수입니다.

일반 함수로는 이것이 불가능합니다. `fn double`은 정의 시점의 값을 기억하지 못합니다. 클로저만이 주변 환경을 캡처합니다.

---

## 캡처 방식: `move`의 의미

Rust의 클로저는 캡처 방식이 명확합니다.

### 참조로 캡처 (기본)

```rust
let message = String::from("hello");
let greet = || println!("{}", message); // message를 참조로 캡처

greet(); // hello
println!("{}", message); // message는 여전히 유효
```

### 소유권으로 캡처 (`move`)

```rust
let message = String::from("hello");
let greet = move || println!("{}", message); // message를 이동

// println!("{}", message); // 컴파일 에러: message는 이미 이동됨
greet(); // hello
```

`move`는 클로저가 캡처한 값의 소유권을 가져갑니다. 스레드로 클로저를 넘길 때처럼, 클로저가 원본보다 오래 살아야 할 때 씁니다.

---

## 고차 함수: 함수를 받거나 반환하는 함수

함수가 일급이기 때문에 고차 함수(Higher-Order Function)가 가능합니다.

### 함수를 인자로 받기

```rust
fn apply_twice(f: impl Fn(i32) -> i32, x: i32) -> i32 {
    f(f(x))
}

fn main() {
    println!("{}", apply_twice(|x| x * 2, 3)); // 12 (3*2*2)
    println!("{}", apply_twice(|x| x + 1, 3)); // 5  (3+1+1)
}
```

### 함수를 반환하기

```rust
fn make_multiplier(factor: i32) -> impl Fn(i32) -> i32 {
    move |x| x * factor
}

fn main() {
    let triple = make_multiplier(3);
    let prices = vec![1_000, 2_000, 3_000];
    let tripled: Vec<i32> = prices.iter().map(|&p| triple(p)).collect();
    println!("{:?}", tripled); // [3000, 6000, 9000]
}
```

---

## 컴포지션과의 연결

[함수 컴포지션](/posts/programming/functional-composition/) 글에서 이터레이터 체인을 자유롭게 썼습니다.

```rust
orders.iter()
    .map(build_receipt)
    .filter(|r| r.total > 0.0)
    .sum()
```

`map`과 `filter`가 함수(클로저)를 인자로 받을 수 있는 것이 일급 함수 덕분입니다. 함수가 일급이 아니라면 이 체인은 불가능합니다.

```rust
// 일급 함수 없이 같은 결과를 내려면
let mut total = 0.0;
for order in &orders {
    let r = build_receipt(order);
    if r.total > 0.0 {
        total += r.total;
    }
}
```

같은 로직이지만, 변환 단계들이 흩어지고 재사용이 어렵습니다.

---

## DI와의 연결

[함수형 DI](/posts/programming/functional-dependency-injection/) 글에서 액션을 함수 인자로 넘겼습니다.

```rust
fn confirm_order(
    order_id: u32,
    fetch_order: impl Fn(u32) -> Order,
    send_email:  impl Fn(&str, &str),
) {
    let order = fetch_order(order_id);
    send_email(&order.user_email, &render_email(&order));
}
```

`fetch_order`와 `send_email`을 인자로 받을 수 있는 것도 함수가 일급이기 때문입니다. 테스트에서는 다른 클로저를, 프로덕션에서는 실제 함수를 넘깁니다.

```rust
// 테스트
confirm_order(
    1,
    |_| Order { user_email: "test@test.com".into(), item: "pen".into() },
    |to, body| println!("테스트 이메일: {} / {}", to, body),
);
```

---

## 클로저의 트레잇: `Fn`, `FnMut`, `FnOnce`

Rust에서 클로저는 세 가지 트레잇 중 하나(또는 여러 개)를 구현합니다.

| 트레잇 | 의미 | 캡처한 값 |
|---|---|---|
| `Fn` | 여러 번 호출 가능, 캡처 변경 없음 | 불변 참조 또는 `Copy` |
| `FnMut` | 여러 번 호출 가능, 캡처 변경 가능 | 가변 참조 |
| `FnOnce` | 한 번만 호출 가능 | 소유권 이동 |

```rust
// Fn: 캡처한 값을 읽기만 함
let x = 10;
let read = || println!("{}", x);
read(); read(); // 여러 번 호출 가능

// FnMut: 캡처한 값을 변경
let mut count = 0;
let mut increment = || { count += 1; count };
println!("{}", increment()); // 1
println!("{}", increment()); // 2

// FnOnce: 캡처한 값을 소비
let name = String::from("Alice");
let consume = move || println!("Hello, {}!", name);
consume(); // Hello, Alice!
// consume(); // 에러: name은 이미 소비됨
```

함수형 설계 관점에서:
- `Fn` = 계산처럼 동작 (부수효과 없음)
- `FnMut` = 내부 상태를 바꾸는 액션
- `FnOnce` = 한 번만 실행되어야 하는 액션

---

## 함수 합성 직접 만들기

일급 함수가 있으면 컴포지션 연산자를 직접 만들 수 있습니다.

```rust
fn compose<A, B, C>(
    f: impl Fn(A) -> B,
    g: impl Fn(B) -> C,
) -> impl Fn(A) -> C {
    move |x| g(f(x))
}

fn main() {
    let double    = |x: i32| x * 2;
    let add_one   = |x: i32| x + 1;
    let to_string = |x: i32| x.to_string();

    let pipeline = compose(compose(double, add_one), to_string);

    println!("{}", pipeline(3)); // "7" — (3*2)+1 → "7"
}
```

`compose`는 두 함수를 받아 하나의 새 함수를 반환합니다. 이것이 수학의 함수 합성 `g ∘ f`를 코드로 표현한 것입니다.

---

## 정리

일급 함수와 클로저는 함수형 프로그래밍의 기반 메커니즘입니다.

| 개념 | 의미 | 활용 |
|---|---|---|
| 일급 함수 | 함수를 값처럼 다룸 | 고차 함수, 이터레이터 |
| 클로저 | 주변 환경을 캡처하는 함수 | DI, 테스트 픽스처, 지연 실행 |
| 고차 함수 | 함수를 받거나 반환 | `map`, `filter`, `compose` |

에릭 노먼드가 말하는 "액션을 인자로 넘긴다", "계산을 조합한다"는 모두 이 메커니즘 위에 서 있습니다. 함수가 일급이기 때문에 계산을 조합할 수 있고, 액션을 교체할 수 있습니다.

---

*관련 글: [함수 컴포지션](/posts/programming/functional-composition/), [함수형 DI](/posts/programming/functional-dependency-injection/), [Iterator 트레잇 직접 구현하기](/posts/programming/rust-iterator-trait/)*
