+++
title = "Spearmans rank correlation coefficient"
date = 2024-08-14T12:40:11+09:00
draft = false
+++

이 글은 https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient 을 번역 및 정리한 글입니다.

통계에서, 스피어만의 순위 상관 계수 또는 스피어만의 ρ는 Charles Spearman의 이름을 따서 명명되었으며, 종종 그리스 문자 ρ (rho) 또는 rₛ로 표기됩니다.
이는 두 변수의 순위 간의 상관 관계(즉, 두 변수 순위 간의 통계적 의존성)를 측정하는 비모수적 방법입니다.

스피어만의 상관 계수는 두 변수의 순위 값 간의 피어슨 상관 계수와 동일합니다.
피어슨 상관 계수가 선형 관계를 평가하는 반면, 스피어만 상관 계수는 단조 관계(선형일 수도 있고 아닐 수도 있는)를 평가합니다.
데이터 값이 중복되지 않는다면, 스피어만 상관 계수 +1 또는 -1의 완벽한 상관은 각 변수가 다른 변수의 완벽한 단조 함수일 때 발생합니다.

직관적으로, 두 변수 간의 스피어만 상관 계수는 두 변수 간의 순위(즉, 변수 내에서의 관측치의 상대적 위치: 1위, 2위, 3위 등)가 유사하거나 (상관이 1인 경우) 동일할 때 높고, 순위가 다르거나 (상관이 -1인 경우) 완전히 반대일 때 낮아집니다.

스피어만의 계수는 연속형 변수와 이산형 서열 변수를 모두 평가하는 데 적합합니다.
스피어만의 ρ와 켄달의 τ는 모두 보다 일반적인 상관 계수의 특수한 사례로서 공식화될 수 있습니다.

### Applications
이 계수는 데이터가 모델에 얼마나 잘 맞는지를 판단하거나, 텍스트 문서의 유사성을 판단하는 데 사용할 수 있습니다.

### Definition and calculation
스피어만 상관 계수는 순위 변수 간의 피어슨 상관 계수로 정의됩니다.

크기가 \( n \)인 표본의 경우, \( X_i, Y_i \)의 원시 점수는 순위 \( R(X_i), R(Y_i) \)로 변환되며, 그 후 \( r_s \)는 다음과 같이 계산됩니다.
\[\displaystyle r_{s}=\rho _{\operatorname {R} (X),\operatorname {R} (Y)}={\frac {\operatorname {cov} (\operatorname {R} (X),\operatorname {R} (Y))}{\sigma _{\operatorname {R} (X)}\sigma _{\operatorname {R} (Y)}}}\]

여기서 \( \rho \)는 일반적인 피어슨 상관 계수를 나타내지만, 여기서는 순위 변수에 적용됩니다.
\( \operatorname{cov} (\operatorname{R}(X), \operatorname{R}(Y)) \)는 순위 변수들의 공분산을 의미하며, \( \sigma _{\operatorname {R} (X)} \)와 \( \sigma _{\operatorname {R} (Y)} \)는 각각 순위 변수 \( \operatorname{R}(X) \)와 \( \operatorname{R}(Y) \)의 표준 편차를 나타냅니다.
 
모든 \( n \)개의 순위가 서로 다른 정수일 경우에만, 다음의 유명한 공식을 사용하여 계산할 수 있습니다.
\[\displaystyle r_{s}=1-{\frac {6\sum d_{i}^{2}}{n(n^{2}-1)}}\]

여기서 \( d_i = \operatorname{R}(X_i) - \operatorname{R}(Y_i) \)는 각 관측치의 두 순위 간의 차이를 나타내며, \( n \)은 관측치의 수입니다.

#### Proof
공분산 Cov(X, Y) = E[XY] - E[X]E[Y] 식과 \({\displaystyle (R(X_{i}),R(Y_{i}))=(R_{i},S_{i})}\) 로  위 식은 다음과 같이 변환할 수 있습니다.
\[\displaystyle r_{s}={\frac {{\frac {1}{n}}\sum _{i=1}^{n}R_{i}S_{i}-{\overline {R}}\,{\overline {S}}}{\sigma _{R}\sigma _{S}}}\]
여기에 \({d_i}\) = \({R_i}\) - \({S_i}\)로 정의하고 우리가 가지고 있는 \( R \)과 \( S \)는 1, 2, ..., n 집합에서 균등하게 분포된 랜덤 변수 \( U \)처럼 분포된다고 볼 수 있어 \( \overline{R} = \overline{S} = \mathbb{E}[U] \)가 됩니다.
그래서 위 식의 분자는 다음과 같은 계산될 수 있습니다.

\[{\displaystyle {\begin{aligned}{\frac {1}{n}}\sum _{i=1}^{n}R_{i}S_{i}-{\overline {R}}{\overline {S}}&={\frac {1}{n}}\sum _{i=1}^{n}{\frac {1}{2}}(R_{i}^{2}+S_{i}^{2}-d_{i}^{2})-{\overline {R}}^{2}\\&={\frac {1}{2}}{\frac {1}{n}}\sum _{i=1}^{n}R_{i}^{2}+{\frac {1}{2}}{\frac {1}{n}}\sum _{i=1}^{n}S_{i}^{2}-{\frac {1}{2n}}\sum _{i=1}^{n}d_{i}^{2}-{\overline {R}}^{2}\\&=({\frac {1}{n}}\sum _{i=1}^{n}R_{i}^{2}-{\overline {R}}^{2})-{\frac {1}{2n}}\sum _{i=1}^{n}d_{i}^{2}\\&=\sigma _{R}^{2}-{\frac {1}{2n}}\sum _{i=1}^{n}d_{i}^{2}\\&=\sigma _{R}\sigma _{S}-{\frac {1}{2n}}\sum _{i=1}^{n}d_{i}^{2}\\\end{aligned}}}\]

\(\sigma _{R}^{2}=\sigma _{S}^{2}=\mathrm {Var} (U)=\mathbb {E} [U^{2}]-\mathbb {E} [U]^{2}\) 인데
\(\mathbb{E}[U^2] : \text{Var}(U) = \frac{(n+1)(2n+1)}{6}\) 이고 \(\mathbb{E}[U] : (\mathbb{E}[U])^2 = \left(\frac{n+1}{2}\right)^2 = \frac{(n+1)^2}{4}\)

이제 위에서 구한 식을 넣으면 전체 식은 아래와 같이 정리 될 수 있습니다.

\[{\displaystyle r_{s}={\frac {\sigma _{R}\sigma _{S}-{\frac {1}{2n}}\sum _{i=1}^{n}d_{i}^{2}}{\sigma _{R}\sigma _{S}}}=1-{\frac {\sum _{i=1}^{n}d_{i}^{2}}{2n\cdot {\frac {n^{2}-1}{12}}}}=1-{\frac {6\sum _{i=1}^{n}d_{i}^{2}}{n(n^{2}-1)}}}\]

### Related quantities
관측치 쌍 간의 통계적 의존성 정도를 정량화하는 여러 다른 수치적 측정 방법들이 있습니다.
그중 가장 흔한 것은 피어슨 적률 상관 계수로, 이는 스피어만의 순위 상관과 유사한 상관 방법이지만, 순위가 아닌 원시 숫자 간의 '선형' 관계를 측정합니다.

스피어만 순위 상관의 또 다른 이름은 '등급 상관'입니다.
여기서 '순위' 대신 '등급'이 사용됩니다.
연속 분포에서 관측치의 등급은 관례적으로 순위보다 절반이 낮아지며, 따라서 이 경우 등급 상관과 순위 상관은 동일합니다.
보다 일반적으로, 관측치의 '등급'은 주어진 값보다 작은 모집단의 비율에 대한 추정치에 비례하며, 관측된 값에서 절반의 관측치 조정이 이루어집니다.
이는 동순위 처리의 한 방법에 해당합니다.
'등급 상관'이라는 용어는 드물지만 여전히 사용되고 있습니다.

### Interpretation
스피어만 상관 계수의 부호는 독립 변수 \( X \)와 종속 변수 \( Y \) 간의 연관성 방향을 나타냅니다.
만약 \( X \)가 증가할 때 \( Y \)도 증가하는 경향이 있다면, 스피어만 상관 계수는 양수입니다.
반대로 \( X \)가 증가할 때 \( Y \)가 감소하는 경향이 있다면, 스피어만 상관 계수는 음수입니다.
스피어만 상관 계수가 0이면 \( X \)가 증가할 때 \( Y \)가 증가하거나 감소할 경향이 없음을 나타냅니다.
스피어만 상관 계수는 \( X \)와 \( Y \)가 서로 완벽한 단조 함수에 가까워질수록 절대값이 증가합니다.
\( X \)와 \( Y \)가 완벽하게 단조 관계에 있을 때, 스피어만 상관 계수는 1이 됩니다.
완벽한 단조 증가 관계는 두 데이터 값 쌍 \( X_i, Y_i \)와 \( X_j, Y_j \)에 대해 \( X_i - X_j \)와 \( Y_i - Y_j \)가 항상 같은 부호를 가진다는 것을 의미합니다.
완벽한 단조 감소 관계는 이러한 차이들이 항상 반대 부호를 가지는 것을 의미합니다.

![Positive and negative Spearman rank correlations](/posts/spearman_correlation.png)

스피어만 상관 계수는 종종 '비모수적'이라고 설명됩니다.
이는 두 가지 의미를 가질 수 있습니다.
첫째, 스피어만 상관 계수는 \( X \)와 \( Y \)가 어떤 단조 함수로도 관련이 있을 때 완벽한 값을 나타냅니다.
이는 \( X \)와 \( Y \)가 선형 함수로만 관련될 때 완벽한 값을 주는 피어슨 상관 계수와 대조됩니다.
스피어만 상관 계수가 비모수적이라는 또 다른 의미는 \( X \)와 \( Y \)의 결합 확률 분포의 매개변수를 알 필요 없이, 그 정확한 표본 분포를 얻을 수 있다는 점입니다.

### Example
이 예제에서는 아래 표에 있는 임의의 원시 데이터를 사용하여 한 사람의 IQ와 주당 TV 앞에서 보낸 시간의 상관 관계를 계산합니다 [가상의 값 사용].

| **IQ (Xᵢ)** | **Hours of TV per week (Yᵢ)** |
|:---:|:---:|
| 106 | 7   |
| 100 | 27  |
| 86  | 2   |
| 101 | 50  |
| 99  | 28  |
| 103 | 29  |
| 97  | 20  |
| 113 | 12  |
| 112 | 6   |
| 110 | 17  |

먼저 \(d_{i}^{2}\)을 평가하십시오. 이를 위해 아래 표에 반영된 단계를 사용하십시오.
1. 데이터를 첫 번째 열 \(X_{i}\)에 따라 정렬합니다. 새로운 열 \(x_{i}\)를 생성하고, 이 열에 순위값 1, 2, 3, …, n을 할당합니다.
2. 다음으로, \(x_{i}\)가 추가된 데이터를 두 번째 열 \(Y_{i}\)에 따라 정렬합니다. 네 번째 열 \(y_{i}\)를 생성하고, 이 열에 동일하게 순위값 1, 2, 3, ..., n을 할당합니다.
3. 다섯 번째 열 \(d_{i}\)를 생성하여 두 순위 열 \(x_{i}\)와 \(y_{i}\)의 차이를 저장합니다.
4. 마지막으로 열 \(d_{i}^{2}\)를 생성하여 열 \(d_{i}\)의 값을 제곱한 값을 저장합니다.

| IQ  | Hours of TV per week | Rank \(x_{i}\) | Rank \(y_{i}\) | \(d_{i}\) | \(d_{i}^{2}\) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 86  | 2  | 1  | 1  | 0  | 0  |
| 97  | 20 | 2  | 6  | -4 | 16 |
| 99  | 28 | 3  | 8  | -5 | 25 |
| 100 | 27 | 4  | 7  | -3 | 9  |
| 101 | 50 | 5  | 10 | -5 | 25 |
| 103 | 29 | 6  | 9  | -3 | 9  |
| 106 | 7  | 7  | 3  | 4  | 16 |
| 110 | 17 | 8  | 5  | 3  | 9  |
| 112 | 6  | 9  | 2  | 7  | 49 |
| 113 | 12 | 10 | 4  | 6  | 36 |
 
\(d_{i}^{2}\) 값을 찾은 후, 이들을 합하여 \( \sum d_{i}^{2} = 194 \)를 구합니다.
n의 값은 10입니다. 이 값을 다음 식에 대입할 수 있습니다:

\[ \rho = 1 - \frac {6 \sum d_{i}^{2}}{n(n^{2}-1)} \]

이를 통해 다음과 같이 계산할 수 있습니다:

\[ \rho = 1 - \frac {6 \times 194}{10(10^{2}-1)} \]

계산 결과, \( \rho = -29/165 = -0.175757575...\) 입니다.

값이 0에 가까운 것은 IQ와 TV 시청 시간 사이의 상관관계가 매우 낮다는 것을 보여주지만, 음수 값은 TV 시청 시간이 길수록 IQ가 낮아지는 경향이 있음을 시사합니다.
