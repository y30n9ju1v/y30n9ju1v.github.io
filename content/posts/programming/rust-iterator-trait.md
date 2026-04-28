---
title: "Iterator 트레잇 직접 구현하기: 지연 평가와 컴포지션의 원리"
date: 2026-04-28T17:00:00+09:00
draft: false
tags: ["함수형 프로그래밍", "Rust", "설계", "이터레이터", "Grokking Simplicity"]
categories: ["프로그래밍"]
description: "map, filter, take가 왜 느리지 않은지, 이터레이터를 직접 구현하며 지연 평가와 zero-cost abstraction의 원리를 이해합니다."
---

## 이 글을 읽고 나면

- `Iterator` 트레잇이 어떻게 구현되는지 직접 만들어보며 이해합니다.
- 왜 이터레이터 체인이 중간 컬렉션을 만들지 않는지 압니다.
- 컴포지션이 왜 Rust에서 "공짜"인지 이해합니다.

이전 글 [함수 컴포지션](/posts/programming/functional-composition/)에서 이터레이터 체인을 자유롭게 썼습니다. 이 글은 그 안에서 무슨 일이 일어나는지 들여다봅니다.

---

## 이터레이터를 쓰면서 생기는 의문

컴포지션 글에서 이런 코드를 썼습니다.

```rust
fn summary(orders: &[Order]) -> f64 {
    orders.iter()
        .map(build_receipt)
        .filter(|r| r.total > 0.0)
        .map(|r| r.total)
        .sum()
}
```

여기서 두 가지 의문이 생깁니다.

1. `.map()` 이후에 새로운 `Vec`이 만들어지는가? 만들어진다면 메모리 낭비 아닌가?
2. `map → filter → map`을 세 번 순회하는가? 느리지 않은가?

정답은 **아니오** 입니다. 이터레이터는 지연 평가됩니다. 중간 컬렉션을 만들지 않고, 한 번만 순회합니다. 어떻게 가능한지 직접 구현하며 알아봅니다.

---

## `Iterator` 트레잇의 구조

Rust의 `Iterator` 트레잇은 핵심이 딱 하나입니다.

```rust
trait Iterator {
    type Item;
    fn next(&mut self) -> Option<Self::Item>;
}
```

`next()`를 호출할 때마다 다음 값을 `Some(value)`로 돌려주고, 끝나면 `None`을 돌려줍니다. 이것이 전부입니다.

`map`, `filter`, `take` 같은 메서드들은 모두 이 `next()` 하나 위에 구현된 기본 메서드입니다.

---

## 직접 구현해보기: `Counter`

1부터 N까지 세는 이터레이터를 만들어봅니다.

```rust
struct Counter {
    current: u32,
    max: u32,
}

impl Counter {
    fn new(max: u32) -> Self {
        Counter { current: 0, max }
    }
}

impl Iterator for Counter {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        if self.current < self.max {
            self.current += 1;
            Some(self.current)
        } else {
            None
        }
    }
}
```

`next()`만 구현하면 `map`, `filter`, `sum` 등 표준 라이브러리의 모든 이터레이터 메서드를 바로 쓸 수 있습니다.

```rust
fn main() {
    // 1부터 5까지의 합
    let total: u32 = Counter::new(5).sum();
    println!("{}", total); // 15

    // 짝수만 두 배로
    let doubled_evens: Vec<u32> = Counter::new(10)
        .filter(|n| n % 2 == 0)
        .map(|n| n * 2)
        .collect();
    println!("{:?}", doubled_evens); // [4, 8, 12, 16, 20]
}
```

---

## 지연 평가: `map`은 무엇을 반환하는가

`counter.map(|n| n * 2)`는 `Vec`을 반환하지 않습니다. `Map`이라는 새로운 구조체를 반환합니다.

```rust
// 표준 라이브러리의 Map 구현을 단순화한 버전
struct Map<I, F> {
    inner: I,
    func: F,
}

impl<I: Iterator, B, F: Fn(I::Item) -> B> Iterator for Map<I, F> {
    type Item = B;

    fn next(&mut self) -> Option<B> {
        self.inner.next().map(|x| (self.func)(x))
    }
}
```

`Map`의 `next()`는 호출될 때만 안쪽 이터레이터에서 값을 하나 꺼내 변환합니다. `.map()`을 호출하는 순간에는 아무 계산도 일어나지 않습니다.

`filter`도 마찬가지입니다.

```rust
struct Filter<I, F> {
    inner: I,
    predicate: F,
}

impl<I: Iterator, F: Fn(&I::Item) -> bool> Iterator for Filter<I, F> {
    type Item = I::Item;

    fn next(&mut self) -> Option<I::Item> {
        loop {
            match self.inner.next() {
                Some(x) if (self.predicate)(&x) => return Some(x),
                Some(_) => continue, // 조건 불만족 → 다음으로
                None    => return None,
            }
        }
    }
}
```

---

## 체인이 실제로 작동하는 방식

```rust
Counter::new(5)
    .map(|n| n * 2)
    .filter(|n| n > 4)
    .sum::<u32>()
```

이 체인의 타입은 이렇게 쌓입니다.

```
Sum(
  Filter<
    Map<
      Counter,
      fn(u32) -> u32
    >,
    fn(&u32) -> bool
  >
)
```

`sum()`이 호출되면 내부적으로 `next()`를 반복 호출합니다. 각 `next()` 호출은 이렇게 전파됩니다.

```
sum이 next() 호출
  → Filter가 next() 호출
    → Map이 next() 호출
      → Counter가 next() 호출 → Some(1)
    → Map이 1 * 2 = 2 반환
  → Filter가 2 > 4? false → 다시 next() 호출
    → Map이 next() 호출
      → Counter가 next() 호출 → Some(2)
    → Map이 2 * 2 = 4 반환
  → Filter가 4 > 4? false → 다시 next() 호출
    ...
  → Filter가 10 > 4? true → Some(10) 반환
sum이 10 누적
...
```

값 하나가 체인 전체를 통과합니다. 중간 컬렉션이 없고, 전체 순회도 한 번입니다.

---

## 나만의 이터레이터 어댑터 만들기

표준 라이브러리에 없는 동작이 필요하면 직접 어댑터를 만들 수 있습니다.

**예제: 연속된 값의 차이를 계산하는 `Diff` 어댑터**

`[1, 3, 6, 10]` → `[2, 3, 4]`

```rust
struct Diff<I: Iterator> {
    inner: I,
    prev: Option<i64>,
}

impl<I: Iterator<Item = i64>> Iterator for Diff<I> {
    type Item = i64;

    fn next(&mut self) -> Option<i64> {
        match (self.prev, self.inner.next()) {
            (Some(p), Some(c)) => {
                self.prev = Some(c);
                Some(c - p)
            }
            (None, Some(c)) => {
                self.prev = Some(c);
                self.next() // 첫 값은 건너뜀
            }
            _ => None,
        }
    }
}

// 트레잇으로 체인에 붙이기
trait DiffExt: Iterator<Item = i64> + Sized {
    fn diff(self) -> Diff<Self> {
        Diff { inner: self, prev: None }
    }
}

impl<I: Iterator<Item = i64>> DiffExt for I {}
```

이제 이터레이터 체인에서 `.diff()`를 쓸 수 있습니다.

```rust
fn main() {
    let data = vec![1i64, 3, 6, 10, 15];
    let diffs: Vec<i64> = data.into_iter().diff().collect();
    println!("{:?}", diffs); // [2, 3, 4, 5]

    // 다른 어댑터와 자유롭게 조합
    let large_diffs: Vec<i64> = vec![1i64, 3, 6, 10, 15]
        .into_iter()
        .diff()
        .filter(|&d| d > 3)
        .collect();
    println!("{:?}", large_diffs); // [4, 5]
}
```

---

## zero-cost abstraction

지금까지 본 것을 정리하면, 이터레이터 체인은:

- 중간 `Vec`을 만들지 않습니다
- 데이터를 한 번만 순회합니다
- 각 단계가 구조체로 표현되어 컴파일러가 인라인할 수 있습니다

Rust 컴파일러는 이 구조체들을 최적화 과정에서 제거하고, 단순한 루프로 변환합니다. 결과적으로 손으로 짠 `for` 루프와 동일한 성능이 나옵니다.

```rust
// 이 코드와
let result: u32 = Counter::new(1000)
    .filter(|n| n % 2 == 0)
    .map(|n| n * n)
    .sum();

// 이 코드의 성능이 동일합니다
let mut result: u32 = 0;
for n in 1..=1000 {
    if n % 2 == 0 {
        result += n * n;
    }
}
```

추상화 비용이 없다는 뜻에서 zero-cost abstraction이라고 부릅니다.

---

## 함수형 컴포지션과의 연결

이터레이터가 함수형 컴포지션과 잘 맞는 이유가 여기 있습니다.

컴포지션 글에서 "같은 입력이면 항상 같은 출력"이라고 했습니다. 이터레이터 어댑터들(`map`, `filter`, `take`)은 모두 이 조건을 만족합니다. 외부 상태를 바꾸지 않고, 입력 시퀀스를 변환해 출력 시퀀스를 만듭니다.

```
입력 시퀀스 → map → filter → map → 출력 시퀀스
```

이 파이프라인 전체가 하나의 계산입니다. 그리고 Rust의 이터레이터는 이 계산을 런타임 비용 없이 표현합니다.

`Iterator` 트레잇은 함수형 컴포지션을 Rust 타입 시스템에 녹여낸 설계입니다. `next()` 하나만 구현하면 수십 개의 컴포지션 메서드를 모두 얻는 것이 그 증거입니다.

---

*관련 글: [함수 컴포지션](/posts/programming/functional-composition/), [Rust 소유권을 함수형 관점으로 읽기](/posts/programming/rust-ownership-functional/)*
