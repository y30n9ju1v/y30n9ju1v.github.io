---
title: "중복 제거와 추상화의 기준: 언제 함수로 묶고, 언제 그냥 두는가"
date: 2026-04-28T23:30:00+09:00
draft: false
tags: ["함수형 프로그래밍", "Rust", "설계", "추상화", "Grokking Simplicity"]
categories: ["프로그래밍"]
description: "모든 중복을 제거하면 오히려 코드가 나빠집니다. 에릭 노먼드의 추상화 장벽 개념으로 언제 묶고 언제 그냥 둬야 하는지 판단하는 기준을 설명합니다."
---

## 이 글을 읽고 나면

- "중복은 항상 제거해야 한다"는 규칙이 왜 틀렸는지 이해합니다.
- 추상화가 가치 있는 경우와 해가 되는 경우를 구분할 수 있습니다.
- 에릭 노먼드의 추상화 장벽 개념으로 어디에 경계를 그어야 하는지 판단합니다.

이 글은 [계층형 설계](/posts/programming/functional-stratified-design/)와 [온어니언 아키텍처](/posts/programming/functional-onion-architecture/)의 보완입니다. 계층을 어디서 나눌지, 무엇을 함수로 묶을지 결정할 때 쓰는 기준을 다룹니다.

---

## 문제: 중복 제거가 항상 좋은가

코드를 짜다 보면 비슷한 로직이 두 곳에 나타납니다. 반사적으로 함수로 묶고 싶어집니다.

```rust
// 두 곳에 비슷한 코드가 있다
fn apply_member_discount(price: f64) -> f64 {
    price * 0.9
}

fn apply_employee_discount(price: f64) -> f64 {
    price * 0.9
}
```

"중복이다, 합쳐야 한다"고 생각하고 이렇게 씁니다.

```rust
fn apply_discount(price: f64, rate: f64) -> f64 {
    price * (1.0 - rate)
}
```

지금은 괜찮습니다. 그런데 몇 달 뒤 요구사항이 바뀝니다.

- 회원 할인: 등급에 따라 10~20%, 생일이면 추가 5%
- 직원 할인: 고정 30%, 단 특정 상품 제외

둘의 로직이 완전히 달라졌습니다. `apply_discount` 하나로 묶었기 때문에, 이제 분리하거나 복잡한 분기를 추가해야 합니다. 처음부터 따로 두었다면 각자 독립적으로 바꿀 수 있었습니다.

이것이 에릭 노먼드가 경고하는 **잘못된 추상화**입니다.

---

## 두 종류의 중복

중복에는 두 가지가 있습니다.

### 우연한 중복 (Accidental Duplication)

지금은 같아 보이지만, **이유가 다른** 중복입니다. 회원 할인과 직원 할인이 둘 다 10%인 것은 우연입니다. 미래에 독립적으로 바뀔 것입니다.

우연한 중복을 제거하면 나중에 분리 비용이 생깁니다. **그냥 두는 게 낫습니다.**

### 실질적 중복 (Real Duplication)

같은 이유로 존재하고, **항상 함께 바뀌는** 중복입니다. 예를 들어 여러 곳에서 가격을 원 단위로 포매팅하는 코드가 있다면, 포매팅 정책이 바뀔 때 모두 함께 바뀝니다.

실질적 중복은 제거해야 합니다. 한 곳만 고쳐도 전체가 반영됩니다.

```rust
// 실질적 중복 — 함께 바뀔 것이므로 함수로 묶는다
fn format_price(price: f64) -> String {
    format!("{:.0}원", price)
}
```

---

## 추상화의 기준: 변경 축

중복을 제거할지 말지 결정하는 질문은 이것입니다.

> **"이 두 코드는 같은 이유로 바뀌는가?"**

같은 이유로 바뀐다면 묶어도 됩니다. 다른 이유로 바뀐다면 따로 둡니다.

```rust
// 같은 이유로 바뀌는가? → 묶어도 된다
fn validate_email(email: &str) -> bool {
    email.contains('@') && email.contains('.')
}

// 다른 이유로 바뀌는가? → 따로 둔다
fn validate_user_signup_email(email: &str) -> bool {
    email.contains('@') && email.contains('.')
    // 미래: 도메인 블랙리스트, 일회용 메일 차단 등
}

fn validate_notification_email(email: &str) -> bool {
    email.contains('@') && email.contains('.')
    // 미래: 내부 메일만 허용, 특정 도메인만 허용 등
}
```

지금은 같지만, 회원가입 이메일 검증과 알림 이메일 검증은 다른 정책을 따를 수 있습니다. 묶으면 나중에 분리 비용이 생깁니다.

---

## 추상화 장벽: 어디에 경계를 긋는가

에릭 노먼드는 **추상화 장벽(Abstraction Barrier)**이라는 개념을 씁니다. 장벽은 두 계층 사이의 경계입니다. 장벽 위의 코드는 장벽 아래의 구현 세부사항을 알지 못합니다.

좋은 추상화 장벽은:
- 위쪽이 아래쪽의 구현을 몰라도 됩니다
- 아래쪽이 바뀌어도 위쪽은 수정하지 않아도 됩니다

```rust
// 장벽 아래: 저장소 구현
mod storage {
    pub fn save(key: &str, value: &str) {
        // 지금은 파일에 저장
        println!("[file] {} = {}", key, value);
    }

    pub fn load(key: &str) -> Option<String> {
        // 지금은 파일에서 읽기
        let _ = key;
        Some("cached_value".into())
    }
}

// 장벽 위: 도메인 로직 — storage의 구현을 모른다
mod cache {
    use super::storage;

    pub fn get_or_compute(key: &str, compute: impl Fn() -> String) -> String {
        match storage::load(key) {
            Some(v) => v,
            None => {
                let v = compute();
                storage::save(key, &v);
                v
            }
        }
    }
}
```

나중에 `storage`를 DB로 바꿔도 `cache` 모듈은 건드리지 않아도 됩니다. 장벽이 변경을 막아줍니다.

---

## 나쁜 추상화의 패턴

### 패턴 1: 너무 일찍 묶기

두 번 나왔다고 바로 함수로 만드는 것입니다. "세 번 나오면 그때 묶어라(Rule of Three)"는 경험칙이 있습니다.

```rust
// 두 번 나왔다고 바로 묶으면
fn get_config_value(key: &str) -> String {
    std::env::var(key).unwrap_or_default()
}

// 두 호출처가 완전히 다른 방식으로 진화할 수 있다
// → 처음엔 그냥 두고, 세 번째 나올 때 묶는다
```

### 패턴 2: 구현이 달라야 하는데 같은 인터페이스로 묶기

```rust
// 나쁜 예: 두 계산이 우연히 같은 시그니처를 가졌다고 묶음
fn calculate(a: f64, b: f64) -> f64 {
    a * b
}

// 넓이 계산과 수익 계산이 같은 함수를 쓰면
// 한쪽 정책이 바뀔 때 다른 쪽도 영향받는다
let area   = calculate(width, height);
let profit = calculate(price, quantity); // 수익에 할인이 생기면?
```

### 패턴 3: 추상화가 구현보다 복잡한 경우

추상화는 복잡성을 숨겨야 합니다. 추상화 자체가 더 복잡하면 존재 이유가 없습니다.

```rust
// 이런 추상화는 오히려 이해를 방해한다
fn process<T, F, G>(items: Vec<T>, transform: F, predicate: G) -> Vec<T>
where
    F: Fn(T) -> T,
    G: Fn(&T) -> bool,
{
    items.into_iter().map(transform).filter(predicate).collect()
}

// 그냥 이렇게 쓰는 게 더 명확하다
let result: Vec<_> = items.into_iter()
    .map(|x| x * 2)
    .filter(|x| *x > 10)
    .collect();
```

---

## 실전 기준: 추상화 전 체크리스트

함수로 묶기 전에 이 질문들을 해봅니다.

**묶어도 되는 경우**
- [ ] 두 코드가 항상 함께 바뀌는가?
- [ ] 추상화 이름이 명확하게 정해지는가?
- [ ] 추상화가 호출하는 쪽을 더 단순하게 만드는가?
- [ ] 세 곳 이상에서 같은 패턴이 나타나는가?

**그냥 두는 게 나은 경우**
- [ ] 지금만 우연히 같고, 미래에 독립적으로 바뀔 것 같은가?
- [ ] 묶으면 어색한 이름이 붙는가? (예: `process`, `handle`, `do_thing`)
- [ ] 추상화를 이해하려면 구현을 봐야 하는가?
- [ ] 호출하는 쪽이 오히려 더 복잡해지는가?

---

## 계층형 설계와의 연결

[계층형 설계](/posts/programming/functional-stratified-design/) 글에서 함수들을 변경 빈도에 따라 계층으로 나눴습니다. 추상화 장벽은 그 계층 사이의 경계입니다.

```
[비즈니스 규칙]  ← 자주 바뀜
      │
   장벽 1
      │
[도메인 연산]   ← 가끔 바뀜
      │
   장벽 2
      │
[기반 연산]    ← 거의 안 바뀜
```

장벽이 잘 그어지면:
- 비즈니스 규칙이 바뀌어도 도메인 연산은 그대로입니다
- 기반 구현이 바뀌어도 위쪽은 모릅니다

장벽을 잘못 그으면 (우연한 중복을 제거해서):
- 한쪽 정책이 바뀌면 다른 쪽도 영향받습니다
- 분리하려면 장벽을 다시 그어야 합니다

---

## 정리

| 상황 | 판단 | 이유 |
|---|---|---|
| 같은 이유로 항상 함께 바뀜 | 묶는다 | 실질적 중복 |
| 지금은 같지만 이유가 다름 | 그냥 둔다 | 우연한 중복 |
| 세 곳 이상에서 반복 | 묶는다 | 변경 비용 감소 |
| 두 번만 나옴 | 기다린다 | 섣부른 추상화 방지 |
| 추상화 이름이 안 떠오름 | 그냥 둔다 | 개념이 아직 불명확 |
| 추상화가 구현보다 복잡 | 그냥 둔다 | 추상화의 역할 없음 |

에릭 노먼드의 핵심 메시지: **중복은 비용이지만, 잘못된 추상화는 더 큰 비용입니다.** 묶기 전에 "같은 이유로 바뀌는가"를 먼저 물어봅니다.

---

*관련 글: [계층형 설계](/posts/programming/functional-stratified-design/), [온어니언 아키텍처](/posts/programming/functional-onion-architecture/), [액션/계산/데이터](/posts/programming/functional-actions-calculations-data/)*
