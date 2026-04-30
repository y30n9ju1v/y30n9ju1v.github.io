---
title: "파서 컴비네이터: 컴포지션 철학의 극적인 응용"
date: 2026-04-28T19:00:00+09:00
draft: false
tags: ["함수형 프로그래밍", "Rust", "설계", "파서", "Grokking Simplicity"]
categories: ["프로그래밍", "함수형 프로그래밍"]
description: "작은 파서들을 조합해 복잡한 문법을 처리합니다. 함수 컴포지션 철학이 가장 극적으로 드러나는 패턴입니다."
---

## 이 글을 읽고 나면

- 파서 컴비네이터가 무엇인지, 왜 함수형 컴포지션의 대표 사례인지 이해합니다.
- 작은 파서들을 직접 만들고 조합해서 실용적인 파서를 완성합니다.
- 파서도 계산임을 이해하고, 테스트가 왜 쉬운지 봅니다.

이전 글 [함수 컴포지션](/posts/programming/functional/functional-composition/)과 [모나드로 배우는 함수형 에러 처리](/posts/programming/functional/functional-monad-error-handling/)를 먼저 읽으면 더 자연스럽게 이어집니다.

---

## 파싱의 문제

`"2026-04-28"` 같은 문자열을 날짜 구조체로 변환하려면 어떻게 할까요?

전통적인 방법은 정규식이나 직접 작성한 상태 기계입니다. 둘 다 잘 작동하지만, 문법이 복잡해질수록 코드도 복잡해집니다. 에러 처리가 파싱 로직과 뒤섞이고, 재사용이 어렵습니다.

파서 컴비네이터는 다른 접근을 취합니다.

> **파서를 함수로 표현하고, 작은 파서들을 조합해 큰 파서를 만든다.**

---

## 파서의 타입

파서는 이렇게 정의합니다.

```rust
// 입력 문자열을 받아서
// 성공하면 (남은 입력, 파싱된 값)을 반환하고
// 실패하면 에러 메시지를 반환한다
type Parser<'a, T> = Box<dyn Fn(&'a str) -> Result<(&'a str, T), String> + 'a>;
```

핵심은 반환 타입입니다. `(&'a str, T)`에서 앞의 `&str`은 **아직 파싱하지 않은 나머지 입력**입니다. 파서는 입력을 조금씩 소비하면서 앞으로 나아갑니다.

---

## 기본 파서: 원자 단위

가장 작은 파서들부터 만듭니다. 이것들이 기반 연산 계층입니다.

### 특정 문자 하나 파싱

```rust
fn char_parser(expected: char) -> Parser<'static, char> {
    Box::new(move |input: &str| {
        let mut chars = input.chars();
        match chars.next() {
            Some(c) if c == expected => Ok((chars.as_str(), c)),
            Some(c) => Err(format!("expected '{}', got '{}'", expected, c)),
            None    => Err(format!("expected '{}', got end of input", expected)),
        }
    })
}
```

### 숫자 하나 파싱

```rust
fn digit() -> Parser<'static, char> {
    Box::new(|input: &str| {
        let mut chars = input.chars();
        match chars.next() {
            Some(c) if c.is_ascii_digit() => Ok((chars.as_str(), c)),
            Some(c) => Err(format!("expected digit, got '{}'", c)),
            None    => Err("expected digit, got end of input".into()),
        }
    })
}
```

### 동작 확인

```rust
fn main() {
    println!("{:?}", char_parser('-')("-2026")); // Ok(("2026", '-'))
    println!("{:?}", char_parser('-')("2026"));  // Err("expected '-', got '2'")
    println!("{:?}", digit()("42abc"));          // Ok(("2abc", '4'))
}
```

---

## 컴비네이터: 파서를 조합하는 함수

기본 파서들을 조합하는 함수들입니다. 이것들이 컴비네이터입니다.

### `map`: 파싱 결과를 변환하기

```rust
fn map<'a, A, B>(
    parser: Parser<'a, A>,
    f: impl Fn(A) -> B + 'a,
) -> Parser<'a, B> {
    Box::new(move |input| {
        parser(input).map(|(rest, val)| (rest, f(val)))
    })
}
```

### `and_then`: 두 파서를 순서대로 실행하기

```rust
fn and_then<'a, A, B>(
    first: Parser<'a, A>,
    second: Parser<'a, B>,
) -> Parser<'a, (A, B)> {
    Box::new(move |input| {
        let (rest, a) = first(input)?;
        let (rest, b) = second(rest)?;
        Ok((rest, (a, b)))
    })
}
```

### `many1`: 한 번 이상 반복하기

```rust
fn many1<'a, T: 'a>(parser: Parser<'a, T>) -> Parser<'a, Vec<T>> {
    Box::new(move |input| {
        let mut results = vec![];
        let mut remaining = input;

        loop {
            match parser(remaining) {
                Ok((rest, val)) => {
                    results.push(val);
                    remaining = rest;
                }
                Err(_) => break,
            }
        }

        if results.is_empty() {
            Err("expected at least one match".into())
        } else {
            Ok((remaining, results))
        }
    })
}
```

---

## 조합해서 날짜 파서 만들기

이제 기본 파서와 컴비네이터를 조합해서 `"2026-04-28"`을 파싱합니다.

```rust
#[derive(Debug, PartialEq)]
struct Date {
    year:  u32,
    month: u32,
    day:   u32,
}

fn number_parser<'a>(digits: usize) -> Parser<'a, u32> {
    // digits개의 숫자를 파싱해서 u32로 변환
    Box::new(move |input: &str| {
        if input.len() < digits || !input[..digits].chars().all(|c| c.is_ascii_digit()) {
            return Err(format!("expected {} digits", digits));
        }
        let value: u32 = input[..digits].parse().unwrap();
        Ok((&input[digits..], value))
    })
}

fn date_parser() -> Parser<'static, Date> {
    Box::new(|input| {
        let (rest, year)  = number_parser(4)(input)?;
        let (rest, _)     = char_parser('-')(rest)?;
        let (rest, month) = number_parser(2)(rest)?;
        let (rest, _)     = char_parser('-')(rest)?;
        let (rest, day)   = number_parser(2)(rest)?;

        Ok((rest, Date { year, month, day }))
    })
}
```

```rust
fn main() {
    println!("{:?}", date_parser()("2026-04-28"));
    // Ok(("", Date { year: 2026, month: 4, day: 28 }))

    println!("{:?}", date_parser()("2026-04-28 extra"));
    // Ok((" extra", Date { year: 2026, month: 4, day: 28 }))

    println!("{:?}", date_parser()("2026/04/28"));
    // Err("expected '-', got '/'")
}
```

`date_parser`는 더 작은 파서들의 컴포지션입니다. 각 단계가 실패하면 `?`로 즉시 전파됩니다. 모나드 글에서 본 패턴 그대로입니다.

---

## 파서는 계산이다

파서 컴비네이터의 핵심 통찰은 이것입니다.

> **파서는 계산이다.**

- 같은 입력이면 항상 같은 결과
- 외부 상태를 바꾸지 않음
- 조합해도 예측 가능

계산이기 때문에 테스트가 trivial합니다.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_date() {
        let result = date_parser()("2026-04-28").unwrap();
        assert_eq!(result, ("", Date { year: 2026, month: 4, day: 28 }));
    }

    #[test]
    fn test_wrong_separator() {
        assert!(date_parser()("2026/04/28").is_err());
    }

    #[test]
    fn test_partial_input_leaves_remainder() {
        let (rest, _) = date_parser()("2026-04-28T12:00:00").unwrap();
        assert_eq!(rest, "T12:00:00");
    }

    #[test]
    fn test_digit_parser() {
        assert_eq!(digit()("3abc"), Ok(("abc", '3')));
        assert!(digit()("abc").is_err());
    }
}
```

DB도, 파일도, 외부 서비스도 없습니다. 입력 문자열만 있으면 됩니다.

---

## 컴포지션 철학과의 연결

파서 컴비네이터는 이 시리즈에서 다룬 원칙들이 한꺼번에 드러나는 패턴입니다.

| 원칙 | 파서 컴비네이터에서 |
|---|---|
| 계산/액션 구분 | 파서는 순수한 계산. I/O 없음 |
| 함수 컴포지션 | `and_then`, `map`으로 파서 연결 |
| 계층형 설계 | `digit` → `number_parser` → `date_parser` |
| 모나딕 컴포지션 | 실패를 `Result`로 전파, `?`로 체이닝 |

`digit`은 기반 연산입니다. `number_parser`는 `digit`을 조합한 도메인 연산입니다. `date_parser`는 `number_parser`와 `char_parser`를 조합한 비즈니스 규칙입니다. 계층형 설계 글에서 본 구조 그대로입니다.

---

## 실무에서는

직접 구현한 파서 컴비네이터는 학습 목적으로 좋습니다. 실무에서는 `nom`, `winnow`, `chumsky` 같은 검증된 라이브러리를 씁니다. 이 글에서 만든 `map`, `and_then`, `many1`과 동일한 컴비네이터들을 훨씬 정교하게 구현해 두었습니다.

핵심은 라이브러리가 아니라 패턴입니다. **작은 파서를 만들고, 조합하고, 더 큰 파서를 만든다.** 이 패턴을 이해하면 어떤 파서 라이브러리도 자연스럽게 읽힙니다.

---

*관련 글: [함수 컴포지션](/posts/programming/functional/functional-composition/), [모나드로 배우는 함수형 에러 처리](/posts/programming/functional/functional-monad-error-handling/), [계층형 설계](/posts/programming/functional/functional-stratified-design/)*
