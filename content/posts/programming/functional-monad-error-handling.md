---
title: "모나드로 배우는 함수형 에러 처리"
date: 2026-04-17T17:00:00+09:00
draft: false
tags: ["함수형 프로그래밍", "Rust", "설계", "모나드", "Grokking Simplicity"]
categories: ["프로그래밍"]
description: "try-catch 없이 에러를 처리하는 함수형 프로그래밍의 방식, 모나드를 처음부터 차근차근 설명합니다."
math: true
---

## 이 글을 읽고 나면

- 왜 함수형 프로그래밍에서 `try-catch` 대신 `Result` 타입을 쓰는지 이해할 수 있습니다.
- `bind`(`>>=`)가 뭘 하는 건지 직관적으로 알 수 있습니다.
- "모나드"라는 단어를 들어도 겁먹지 않습니다.

수학 기호는 나오지만, 숫자 계산은 없습니다. 기호는 "이름표"처럼만 읽어도 됩니다.

---

## 문제: 에러가 날 수 있는 함수를 연결하기

프로그래밍을 하다 보면 이런 함수들이 생깁니다.

```python
def 문자열을_정수로(s):
    ...  # "abc" 같은 입력이 오면 에러!

def 역수를_구하기(n):
    ...  # 0이 들어오면 에러!

def 제곱근을_구하기(x):
    ...  # 음수가 들어오면 에러!
```

이 세 함수를 순서대로 연결해서 "문자열 → 정수 → 역수 → 제곱근"을 계산하고 싶습니다.

문제는 각 단계마다 에러가 날 수 있다는 겁니다.

### 보통 방식 — try-catch의 한계

```python
def compute(s):
    try:
        n = 문자열을_정수로(s)
        r = 역수를_구하기(n)
        result = 제곱근을_구하기(r)
        return result
    except Exception as e:
        return f"에러: {e}"
```

나쁜 코드는 아니지만, 문제가 있습니다:

- 어느 단계에서 에러가 났는지 불분명합니다
- 에러 처리가 "예외"라는 별도 메커니즘에 의존합니다
- 함수를 테스트하거나 합치기가 번거롭습니다

함수형 프로그래밍은 다른 접근법을 씁니다.

---

## 핵심 아이디어: 에러를 값으로 취급하기

함수형 프로그래밍의 핵심 발상은 이겁니다.

> 에러는 "프로그램이 터지는 사건"이 아니라, **함수가 돌려주는 값 중 하나**다.

그래서 이런 타입을 만듭니다.

**Haskell**
```haskell
data Result a = Ok a | Err String
--              ^^^^   ^^^^^^^^^^
--           정상값    에러 메시지
```

**Rust** — 표준 라이브러리에 `Result<T, E>`가 내장되어 있습니다.
```rust
// Rust에는 이미 내장되어 있음
// Ok(value) 또는 Err(message) 중 하나
```

`Result`는 두 가지 중 하나입니다.

| 경우 | 의미 | Haskell | Rust |
|------|------|---------|------|
| 정상값 | 계산 성공 | `Ok 42` | `Ok(42)` |
| 에러 | 계산 실패 | `Err "음수입니다"` | `Err("음수입니다")` |

이제 우리 함수들은 이렇게 생깁니다.

**Haskell**
```haskell
parseInt       :: String -> Result Int
safeReciprocal :: Int    -> Result Double
safeSqrt       :: Double -> Result Double
```

**Rust**
```rust
fn parse_int(s: &str)        -> Result<i64, String> { ... }
fn safe_reciprocal(n: i64)   -> Result<f64, String> { ... }
fn safe_sqrt(x: f64)         -> Result<f64, String> { ... }
```

함수가 에러를 "던지지" 않습니다. 그냥 `Err "..."` 값을 돌려줄 뿐입니다.

---

## 문제 발생: 함수를 연결할 수가 없다

`parseInt`는 `Result Int`를 돌려줍니다.
그런데 `safeReciprocal`은 `Int`를 받습니다.

```
parseInt "5"  →  Ok 5
                  ↓
safeReciprocal ?  ← Ok 5를 받을 수가 없음! Int를 기대했는데...
```

`Ok`로 감싸진 값을 꺼내서 다음 함수에 넘겨줘야 합니다.
그런데 에러인 경우도 처리해야 합니다.

직접 짜면 이렇게 됩니다.

**Haskell**
```haskell
compute :: String -> Result Double
compute s =
  case parseInt s of
    Err e1 -> Err e1          -- 에러면 즉시 중단
    Ok n ->
      case safeReciprocal n of
        Err e2 -> Err e2      -- 에러면 즉시 중단
        Ok r ->
          case safeSqrt r of
            Err e3 -> Err e3  -- 에러면 즉시 중단
            Ok y   -> Ok y
```

**Rust**
```rust
fn compute(s: &str) -> Result<f64, String> {
    match parse_int(s) {
        Err(e) => Err(e),                    // 에러면 즉시 중단
        Ok(n) => match safe_reciprocal(n) {
            Err(e) => Err(e),                // 에러면 즉시 중단
            Ok(r) => match safe_sqrt(r) {
                Err(e) => Err(e),            // 에러면 즉시 중단
                Ok(y)  => Ok(y),
            }
        }
    }
}
```

동작은 합니다. 하지만 **같은 패턴이 세 번 반복됩니다**.

> 에러면 즉시 중단, 정상값이면 다음 단계 진행

이 반복을 없앨 수 없을까요?

---

## 해결책: bind

반복되는 그 패턴을 함수로 만들면 됩니다.

**Haskell** — `bind`를 직접 정의합니다.

```haskell
bind :: Result a -> (a -> Result b) -> Result b
bind (Err msg) _ = Err msg   -- 이미 에러면 그냥 에러를 넘긴다
bind (Ok x)    f = f x       -- 정상값이면 다음 함수 f를 실행한다
```

**Rust** — `Result`에 `.and_then()`이 내장되어 있습니다. Haskell의 `bind`와 완전히 같은 역할입니다.

```rust
// and_then = bind
// Ok(x)  → 다음 함수 실행
// Err(e) → 에러 그대로 전달
```

`bind`는 두 가지를 받습니다.

1. 이미 계산된 결과 (`Result a`)
2. 다음에 실행할 함수 (`a -> Result b`)

그리고 "에러면 멈추고, 정상이면 계속" 규칙을 자동으로 적용합니다.

이제 `compute`를 이렇게 쓸 수 있습니다.

**Haskell**
```haskell
-- bind 직접 사용
compute s = parseInt s `bind` safeReciprocal `bind` safeSqrt

-- 내장 >>= 연산자 사용 (같은 의미)
compute s = parseInt s >>= safeReciprocal >>= safeSqrt
```

**Rust** — `.and_then()`으로 체이닝합니다.
```rust
fn compute(s: &str) -> Result<f64, String> {
    parse_int(s)
        .and_then(safe_reciprocal)  // bind와 같은 역할
        .and_then(safe_sqrt)
}
```

한 줄(체인)로 줄었습니다! 구조도 훨씬 명확합니다.

**Haskell** — `do` 표기를 사용하면 더 읽기 쉽습니다.

```haskell
compute s = do
  n <- parseInt s        -- 정수로 변환
  r <- safeReciprocal n  -- 역수 계산
  y <- safeSqrt r        -- 제곱근 계산
  return y
```

**Rust** — `?` 연산자가 `do` 표기와 같은 역할을 합니다.

```rust
fn compute(s: &str) -> Result<f64, String> {
    let n = parse_int(s)?;        // 에러면 즉시 반환, 정상이면 n에 꺼냄
    let r = safe_reciprocal(n)?;  // 에러면 즉시 반환, 정상이면 r에 꺼냄
    let y = safe_sqrt(r)?;        // 에러면 즉시 반환, 정상이면 y에 꺼냄
    Ok(y)
}
```

`?`는 Haskell의 `<-`와 동일한 개념입니다. "Ok이면 값을 꺼내고, Err이면 즉시 함수를 종료하고 에러를 반환"합니다.

---

## 그래서 모나드가 뭔가요?

`Result` 타입 + `bind` + `return`의 조합을 **모나드**라고 합니다.

여기서 `return`이라는 단어가 헷갈릴 수 있습니다. `bind`도 `Result`를 돌려주니까요.

구분하는 방법은 이렇습니다.

- **`return` (모나드 용어)** — 평범한 값을 상자에 *넣는* 함수. 시작점.
- **`bind`** — 상자 안 값을 꺼내서 다음 계산으로 *연결하는* 함수. `Result`를 돌려주긴 하지만, 그건 연결의 결과물이지 "넣기"가 아닙니다.

```
return :  a         -> Result a   -- 평범한 값을 상자에 넣기 (탑승)
bind   :  Result a  -> Result b   -- 상자를 다음 계산으로 연결 (환승)
```

모나드는 다음 세 가지로 구성됩니다.

| 구성요소 | 역할 | Haskell | Rust |
|----------|------|---------|------|
| 타입 생성자 | 값을 담는 상자 타입 | `Result a` | `Result<T, E>` |
| return (unit) | 평범한 값을 상자에 넣기 | `return x` | `Ok(x)` |
| bind (>>=) | 상자를 다음 계산으로 연결 | `>>=` | `.and_then()` |

> `Ok` / `Err` 구분이나 `do` / `?` 표기는 `Result` 모나드의 세부 구현이지, 모나드 자체의 구성요소는 아닙니다.

`Result` 모나드 말고도 다양한 모나드가 있습니다.

- `Maybe` — 값이 없을 수 있는 계산 (`Nothing` / `Just a`)
- `IO` — 입출력이 있는 계산
- `State` — 상태를 들고 다니는 계산
- `List` — 결과가 여러 개일 수 있는 계산

모두 같은 구조입니다. **"상자 + bind + return"**

---

## 범주론으로 다시 읽기

모나드를 수학적으로 공부하면 **범주론**이라는 분야가 나옵니다.
범주론은 어렵게 느껴지지만, 핵심 아이디어는 단순합니다.

> **타입을 대상으로, 함수를 사상으로 보면 프로그램 전체가 하나의 수학적 구조가 된다.**

이 절에서는 지금까지 본 `Result`, `return`, `bind`과 앞으로 설명할 `join`이 범주론에서 각각 무엇인지 차근차근 연결합니다.

---

### 범주론 기본 용어

| 범주론 용어 | 프로그래밍에서의 의미 |
|------------|----------------------|
| 대상(object) | 타입 (`Int`, `String`, `Double` …) |
| 사상(morphism) | 함수 (`f :: A -> B`) |
| 합성(composition) | 함수 합성 (`g ∘ f`) |
| 항등사상(identity) | 아무것도 안 하는 함수 (`id x = x`) |

가장 기본적인 범주는 **Set**입니다. 대상은 집합(타입), 사상은 함수입니다.
순수한 함수 `f :: Int -> String`은 **Set**의 사상입니다.

---

### 1단계: Result는 함자(Functor)다

`Result`는 단순한 타입이 아닙니다. 모든 타입에 "에러 가능성"을 붙여주는 **타입 변환기**입니다.

```
Int    →  Result Int
String →  Result String
Double →  Result Double
```

이 패턴을 수식으로 쓰면 이렇습니다.

$$T(X) = X + \text{Err}$$

각 타입 $X$에 에러 가능성을 추가해 $T(X)$를 만든다는 뜻입니다.

그런데 `Result`는 타입만 변환하는 게 아닙니다. **함수도 같이 들어올릴 수 있습니다.** 평범한 함수 $f : X \to Y$가 있을 때, `Result` 위에서 작동하는 $T(f) : T(X) \to T(Y)$를 이렇게 정의합니다.

$$T(f)(\text{Ok}(x)) = \text{Ok}(f(x)), \qquad T(f)(\text{Err}(e)) = \text{Err}(e)$$

정상값이면 $f$를 적용하고, 에러면 그대로 통과시킵니다.

```haskell
-- fmap: 평범한 함수를 Result 위에서 작동하도록 들어올린다
fmap :: (a -> b) -> Result a -> Result b
fmap f (Ok x)  = Ok (f x)   -- 정상값이면 f를 적용
fmap f (Err e) = Err e       -- 에러면 그대로
```

```rust
// Rust에서는 .map()이 같은 역할을 합니다
Ok(5).map(|x| x * 2)        // Ok(10)
Err::<i32, &str>("오류").map(|x| x * 2)  // Err("오류") — 그대로
```

이처럼 타입도 함수도 모두 변환할 수 있는 구조를 **함자(Functor)**라고 합니다. 단, 이 변환이 "구조를 깨뜨리지 않는다"는 조건이 필요합니다.

1. 항등사상을 보존한다.
$$T(\text{id}_X) = \text{id}_{T(X)}$$

2. 합성을 보존한다.
$$T(g \circ f) = T(g) \circ T(f)$$

`Result`는 이 두 조건을 만족하므로 **Set → Set endofunctor**입니다.

> **endofunctor(자기함자)란?**
> 함자(Functor) 중에서도 출발 범주와 도착 범주가 같은 것을 말합니다.
> `Result`는 Set(타입들의 세계)에서 출발해 Set으로 돌아오므로 *endo*(자기 자신)functor입니다.
> 쉽게 말하면 "타입의 세계를 벗어나지 않는 변환"입니다.

---

### 2단계: return은 자연변환이다

`return :: a -> Result a`는 단순한 함수가 아닙니다. `Int`에도, `String`에도, `Double`에도 — 모든 타입에 대해 **같은 규칙(Ok로 감싸기)**으로 작동합니다.

```haskell
return :: Int    -> Result Int     -- Ok로 감싸기
return :: String -> Result String  -- Ok로 감싸기
return :: Double -> Result Double  -- Ok로 감싸기
```

이처럼 "모든 타입에 대해 일관되게 작동하는 함수들의 묶음"을 **자연변환(natural transformation)**이라고 합니다. 수식으로 쓰면 이렇습니다.

$$\eta : \text{Id} \Rightarrow T$$

$$\eta_X : X \to T(X), \qquad \eta_X(x) = \text{Ok}(x)$$

여기서 $\Rightarrow$는 단순한 함수 화살표가 아니라, 모든 타입(대상)마다 존재하는 수많은 함수의 **가족(family)**을 의미합니다. "어떤 타입을 가져오더라도 동일한 구조적 규칙을 따른다"는 일관성을 수학적으로 표현한 것입니다.

---

### 3단계: join은 중첩된 Result를 한 번 펴는 것이다

`bind`를 설명하기 전에, `join`(또는 `μ`)을 먼저 이해해야 합니다.

$$\mu_X : T(T(X)) \to T(X)$$

예외 모나드에서 $T(T(X)) = (X + \text{Err}) + \text{Err}$의 원소는 세 종류입니다.

$$\mu(\text{Ok}(\text{Ok}(x))) = \text{Ok}(x)$$

$$\mu(\text{Ok}(\text{Err}(e))) = \text{Err}(e)$$

$$\mu(\text{Err}(e')) = \text{Err}(e')$$

```haskell
join :: Result (Result a) -> Result a
join (Ok (Ok x))  = Ok x    -- 둘 다 정상이면 정상
join (Ok (Err e)) = Err e   -- 안쪽 에러를 꺼냄
join (Err e)      = Err e   -- 바깥 에러 그대로
```

```rust
// Rust에서는 .flatten()이 같은 역할입니다
let nested: Result<Result<i32, &str>, &str> = Ok(Ok(42));
nested.flatten()  // Ok(42)

let nested: Result<Result<i32, &str>, &str> = Ok(Err("오류"));
nested.flatten()  // Err("오류")
```

> "안이든 바깥이든 에러가 하나라도 있으면 에러, 둘 다 정상이면 정상"

$$\mu : T^2 \Rightarrow T$$

"함자 $T$를 두 번 적용한 것($T^2$)에서 한 번 적용한 것($T$)으로 가는 자연변환"입니다.

---

### 4단계: bind는 fmap + join이다

이제 핵심입니다. `bind`는 사실 `fmap`과 `join`을 합친 것입니다.

$$m \mathbin{{\gg=}} f \;:=\; \mu_Y(T(f)(m))$$

- $T(f)$ = `fmap f` — 평범한 함수 $f : X \to T(Y)$를 $T(X) \to T(T(Y))$로 들어올림
- $\mu_Y$ = `join` — 두 겹 $T(T(Y))$를 한 겹 $T(Y)$로 펴기

```haskell
m >>= f  =  join (fmap f m)
--           ^^^^  ^^^^^
--           펴기  들어올리기
```

단계별로 보면 이렇습니다.

```
m         :: Result a           -- 현재 결과
T(f)(m)   :: Result (Result b)  -- f를 fmap으로 들어올려 적용 → 두 겹이 됨
μ(...)    :: Result b           -- join으로 두 겹을 한 겹으로 펴기
```

**정상값인 경우** — `Ok 4 >>= safeSqrt`

```haskell
-- 1단계: T(safeSqrt) 적용 (= fmap safeSqrt)
T(safeSqrt)(Ok 4)
  = Ok (safeSqrt 4)
  = Ok (Ok 2.0)        -- 두 겹이 됨

-- 2단계: μ 적용 (= join)
μ(Ok (Ok 2.0))
  = Ok 2.0             -- 한 겹으로 펴짐
```

**에러인 경우** — `Err "오류" >>= safeSqrt`

```haskell
-- 1단계: T(safeSqrt) 적용
T(safeSqrt)(Err "오류")
  = Err "오류"          -- fmap은 에러를 건드리지 않음

-- 2단계: μ 적용
μ(Err "오류")
  = Err "오류"          -- 에러 그대로
```

**안쪽이 에러인 경우** — `Ok (-1) >>= safeSqrt`

```haskell
-- 1단계: T(safeSqrt) 적용
T(safeSqrt)(Ok (-1))
  = Ok (safeSqrt (-1))
  = Ok (Err "negative input")  -- 두 겹, 안쪽이 Err

-- 2단계: μ 적용
μ(Ok (Err "negative input"))
  = Err "negative input"       -- 안쪽 에러를 꺼냄 ← join의 두 번째 케이스
```

즉 `bind`의 "에러면 중단, 정상이면 계속" 동작은 `fmap`과 `join`의 조합에서 **자동으로** 나옵니다.

---

### 5단계: Kleisli 범주 — 에러 가능 함수들의 세계

#### 끊어진 화살표 문제

평범한 함수 $f: A \to B$는 출력($B$)이 다음 함수의 입력($B \to C$)과 딱 맞물립니다. 하지만 에러 가능성이 있는 함수 $f: A \to T(B)$는 출력이 상자($T$)에 갇혀 있어, 다음 함수 $B \to T(C)$와 직접 연결할 수 없습니다.

$$f : X \to T(Y), \qquad g : Y \to T(Z)$$

$$g \circ f \quad \leftarrow \text{불가능! } f\text{의 출력이 }Y\text{가 아니라 }T(Y)\text{이므로}$$

`>>=`는 이 문제를 `Result`에 한해서 해결해줍니다. 하지만 "에러 처리"와 완전히 같은 구조가 **로깅, 상태 전달, 비동기, 리스트** 등에서도 반복해서 나타납니다. 매번 각각의 합성 규칙을 따로 만드는 대신, 이 패턴을 한 번에 추상화할 수 없을까요?

그 답이 **Kleisli 범주**입니다. 타입(대상)은 그대로 두고, **화살표의 종류와 합성 방식만 바꿔** 새로운 범주를 만듭니다.

| | 보통 범주 (**Set**) | Kleisli 범주 ($\mathbf{Set}_T$) |
|---|---|---|
| 대상 | 타입 $X$, $Y$, $Z$ … | 같음 (타입이 그대로 대상) |
| 사상 $X \to Y$ | $f : X \to Y$ | $f : X \to T(Y)$ |
| 합성 | $g \circ f$ | $g \star f$ (아래 정의) |
| 항등사상 | $\text{id}_X : X \to X$ | $\eta_X : X \to T(X)$ |

#### Kleisli 합성의 정의

끊어진 화살표를 잇는 방법은 4단계에서 이미 봤습니다 — `fmap`으로 들어올리고, `join`으로 펴는 것입니다. 이를 합성 연산 $\star$로 정식화합니다.

$$g \star f = \mu_Z \circ T(g) \circ f$$

단계별로 읽으면:
1. $f : X \to T(Y)$를 적용해 $T(Y)$를 얻는다
2. $T(g) : T(Y) \to T(T(Z))$로 들어올린다 (`fmap`)
3. $\mu_Z : T(T(Z)) \to T(Z)$로 펴서 결과를 얻는다 (`join`)

Haskell 코드로는 이렇습니다.

```haskell
-- >=> : Kleisli 합성 ("fish" 연산자)
(>=>) :: (a -> Result b) -> (b -> Result c) -> (a -> Result c)
f >=> g = \x -> join (fmap g (f x))
-- 또는 동일하게 bind로:
f >=> g = \x -> f x >>= g
```

이제 우리 예제를 fish 연산자로 쓸 수 있습니다.

$$\text{compute} = \text{parseInt} \mathbin{\star} \text{safeReciprocal} \mathbin{\star} \text{safeSqrt}$$

```haskell
-- bind 체인 버전
compute s = parseInt s >>= safeReciprocal >>= safeSqrt

-- Kleisli 합성 버전 (같은 의미, 더 수학적)
compute = parseInt >=> safeReciprocal >=> safeSqrt
```

#### 모나드 법칙 = Kleisli 범주가 범주이기 위한 조건

Kleisli 화살표들이 **진짜 범주**를 이루려면 합성이 결합법칙을 만족하고, 항등사상이 존재해야 합니다. 이 조건을 $\mu$와 $\eta$로 쓰면 이렇습니다.

$$\mu_X \circ T(\mu_X) = \mu_X \circ \mu_{T(X)} \qquad \text{(결합법칙)}$$

$$\mu_X \circ T(\eta_X) = \text{id}_{T(X)} \qquad \text{(왼쪽 단위원)}$$

$$\mu_X \circ \eta_{T(X)} = \text{id}_{T(X)} \qquad \text{(오른쪽 단위원)}$$

bind 형식으로는 이렇게 씁니다.

$$\text{return}(x) \mathbin{{\gg=}} f = f(x), \qquad m \mathbin{{\gg=}} \text{return} = m, \qquad (m \mathbin{{\gg=}} f) \mathbin{{\gg=}} g = m \mathbin{{\gg=}} (\lambda x.\, f(x) \mathbin{{\gg=}} g)$$

> 모나드 법칙은 단순한 "좋은 습관"이 아닙니다.
> **Kleisli 범주가 진짜 범주이기 위한 필요충분조건**입니다.

특히 단위원 법칙 덕분에 $\eta_X$가 항등사상이 됩니다. $f : X \to T(Y)$에 대해 `f >=> return = f`를 직접 확인하면 이렇습니다.

```haskell
(f >=> return) x
  = join (fmap return (f x))   -- Kleisli 합성 정의

-- f x = Ok b 인 경우:
  = join (Ok (Ok b))           -- fmap return (Ok b) = Ok (Ok b)
  = Ok b = f x  ✓

-- f x = Err e 인 경우:
  = join (Err e)               -- fmap은 에러를 건드리지 않음
  = Err e = f x  ✓
```

이것이 성립하면, `Result`에서 배운 `>=>` 합성의 성질이 `Maybe`, `IO`, `State`, `List` 등 **모든 모나드에 그대로 적용**됩니다.

---

### 6단계: 모나드를 정의하는 세 가지 동등한 방법

모나드는 세 가지 방법으로 동등하게 정의할 수 있습니다. 각각 강조하는 관점이 다릅니다.

**방법 1: Kleisli 합성으로 정의** (범주론적으로 가장 직접적)

```haskell
class Monad m where
    (>=>)  :: (a -> m b) -> (b -> m c) -> (a -> m c)  -- Kleisli 합성
    return :: a -> m a                                  -- 항등사상
```

**방법 2: bind로 정의** (Haskell 표준, 프로그래밍에 편리)

```haskell
class Monad m where
    (>>=)  :: m a -> (a -> m b) -> m b  -- bind
    return :: a -> m a
```

**방법 3: join + fmap으로 정의** (범주론적 구조를 가장 잘 드러냄)

```haskell
class Functor m => Monad m where
    join   :: m (m a) -> m a  -- 중첩 펴기 (μ)
    return :: a -> m a        -- 값 넣기 (η)
```

세 방법은 서로 변환 가능합니다.

```haskell
-- bind를 join + fmap으로:
m >>= f  =  join (fmap f m)

-- fish를 bind로:
f >=> g  =  \x -> f x >>= g

-- join을 bind로:
join m   =  m >>= id
```

---

### 전체 그림

| 프로그래밍 개념 | 범주론 개념 |
|---|---|
| 타입 `A`, `B`, `C` | 대상(object) |
| 함수 `f :: A -> B` | Set의 사상 |
| `Result` | 함자 $T : \mathbf{Set} \to \mathbf{Set}$ |
| `fmap` / `.map()` | 함자의 사상에 대한 작용 |
| `return` / `Ok(x)` | 자연변환 $\eta : \text{Id} \Rightarrow T$ |
| `join` / `.flatten()` | 자연변환 $\mu : T^2 \Rightarrow T$ |
| `bind` (`>>=`) / `.and_then()` | $\mu \circ T(f)$ = fmap + join |
| `fish` (`>=>`) | Kleisli 합성 연산자 |
| 에러 가능 함수 | Kleisli 범주의 사상 |
| `do` 표기 / `?` 연산자 | bind 연쇄의 문법적 설탕 |
| 모나드 법칙 | Kleisli 범주가 범주이기 위한 조건 |

핵심 메시지는 이겁니다.

> `parse_int(s)? .and_then(safe_reciprocal) .and_then(safe_sqrt)`
>
> 이 코드는 단순한 에러 처리 패턴이 아닙니다.
> **예외 모나드의 Kleisli 범주에서 세 사상을 합성하는 것**입니다.
> 에러 전파는 그 합성의 수학적 구조에서 자동으로 따라옵니다.

---

## 출처

이 글을 작성하면서 참고한 자료들입니다.

- **함수형 프로그래밍의 모나드를 이용한 예외처리의 범주론적 해석** (유투브 채널: 수학의 즐거움의 실리콘 위의 논리)
  - 이 글의 원본. 범주론적 정의(η, μ, Kleisli 범주)와 코드 예시의 기반이 되었습니다.

- **Category Theory for Programmers — Bartosz Milewski**
  - [4장: Kleisli Categories](https://bartoszmilewski.com/2014/12/23/kleisli-categories/) — Writer 모나드를 통한 Kleisli 범주 동기 부여, fish 연산자 `>=>` 도입
  - [20장: Monads — Programmer's Definition](https://bartoszmilewski.com/2016/11/21/monads-programmers-definition/) — 모나드의 세 가지 동등한 정의(`>=>` / `>>=` / `join+return`), `do` 표기법 desugaring
