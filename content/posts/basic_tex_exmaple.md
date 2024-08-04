+++
title = 'Basic Tex Example'
date = 2024-08-04T10:26:29+09:00
draft = false
+++

이 글은 chatGPT의 도움을 받아 정리한 tex의 기본적인 문법에 대한 글입니다.

### 1. 텍스트 서식
* 강조 (Italic): \textit{강조}
* 굵게 (Bold): \textbf{굵게}
* 밑줄 (Underline): \underline{밑줄}

\[\textit{이탤릭체 텍스트}\]
\[\textbf{굵은 텍스트}\]
\[\underline{밑줄 텍스트}\]

### 2. 수식 기호
* 분수: \frac{분자}{분모}
* 근호: \sqrt{내용}
* 지수 및 밑수: a^b, a_b
* 합: \sum_{i=1}^{n} i
* 적분: \int_{a}^{b} f(x) \, dx
* 벡터: \vec{v}, \mathbf{v}
* 수열: a_{n}
* 기타 기호: \alpha, \beta, \gamma, \Delta, \pi, \infty, \partial

~~~
\frac{a}{b}, \sqrt{x}, a^b, a_b
~~~
\[\frac{a}{b}, \sqrt{x}, a^b, a_b\]
~~~
\sum_{i=1}^{n} i
~~~
\[\sum_{i=1}^{n} i\]
~~~
\int_{a}^{b} f(x) \, dx
~~~
\[\int_{a}^{b} f(x) \, dx\]
~~~
\vec{v}, \mathbf{v}
~~~
\[\vec{v}, \mathbf{v}\]
~~~
a_{n}
~~~
\[a_{n}\]
~~~
\alpha, \beta, \gamma, \Delta, \pi, \infty, \partial
~~~
\[\alpha, \beta, \gamma, \Delta, \pi, \infty, \partial\]

### 3. 수식 배열

* 정렬된 수식 배열:
~~~
\begin{align}
E &= mc^2 \\
a &= b + c
\end{align}
~~~
\[
\begin{align}
E &= mc^2 \\
a &= b + c
\end{align}
\]

* 분활된 수식:
~~~
\begin{split}
a + b &= c \\
d + e &= f
\end{split}
~~~
\[
\begin{split}
a + b &= c \\
d + e &= f
\end{split}
\]

### 4. 행렬

* 기본 행렬:
~~~
\begin{matrix}
a & b \\
c & d
\end{matrix}
~~~
\[
\begin{matrix}
a & b \\
c & d
\end{matrix}
\]

* 괄호로 둘러싼 행렬:
~~~
\begin{pmatrix}
a & b \\
c & d
\end{pmatrix}
~~~
\[
\begin{pmatrix}
a & b \\
c & d
\end{pmatrix}
\]

* 중괄호로 둘러싼 행렬:
~~~
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix}
~~~
\[
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix}
\]

### 5. 수식 환경

* 수식 번호 매기기:
~~~
\begin{equation}
E = mc^2
\end{equation}
~~~
\[
\begin{equation}
E = mc^2
\end{equation}
\]

* 수식 번호 없는 환경:
~~~
\begin{equation*}
E = mc^2
\end{equation*}
~~~
\[
\begin{equation*}
E = mc^2
\end{equation*}
\]

### 6. 다양한 연산 기호

* 상한과 하한:
~~~
\sum_{i=1}^{n} i^2, \prod_{i=1}^{n} i
~~~
\[\sum_{i=1}^{n} i^2, \prod_{i=1}^{n} i\]

* 점 곱:
~~~
\cdot, \times
~~~
\[
\cdot, \times

* 논리 연산:
~~~
\land, \lor, \neg, \implies, \iff
~~~
\[
\land, \lor, \neg, \implies, \iff
\]

### 7. 기타 수식 기능

* 케이스 나누기:
~~~
\begin{cases}
1 & \text{if } x > 0 \\
0 & \text{if } x = 0 \\
-1 & \text{if } x < 0
\end{cases}
~~~
\[
\begin{cases}
1 & \text{if } x > 0 \\
0 & \text{if } x = 0 \\
-1 & \text{if } x < 0
\end{cases}
\]
