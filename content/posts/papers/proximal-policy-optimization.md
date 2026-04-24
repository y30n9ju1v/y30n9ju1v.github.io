---
title: "Proximal Policy Optimization Algorithms (PPO)"
date: 2026-04-19T23:30:00+09:00
draft: false
categories: ["Papers", "Deep Learning", "Reinforcement Learning"]
tags: ["reinforcement learning", "policy gradient", "PPO", "deep learning", "OpenAI"]
---

## 개요
- **저자**: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (OpenAI)
- **발행년도**: 2017
- **arXiv**: 1707.06347v2
- **주요 내용**: 신뢰 영역(Trust Region) 방법의 안정성을 유지하면서도 1차 최적화만으로 구현 가능한 강화학습 정책 최적화 알고리즘 PPO 제안. MuJoCo 연속 제어 및 Atari 게임에서 SOTA 달성

## 한계 극복

이 논문은 기존 정책 경사(Policy Gradient) 방법들의 한계를 극복하기 위해 작성되었습니다.

- **기존 Vanilla Policy Gradient의 한계**: 데이터 샘플 1개당 1번의 그래디언트 업데이트만 가능. 동일 데이터로 여러 번 업데이트하면 정책이 불안정하게 크게 변함
- **기존 TRPO의 한계**: 2차 최적화(켤레 기울기법)가 필요해 구현이 복잡하고, dropout이나 파라미터 공유가 있는 아키텍처에 적용 불가
- **기존 DQN의 한계**: 이산(discrete) 행동 공간에만 적합, 연속 제어 문제에 부적합
- **이 논문의 접근 방식**: 확률 비율(probability ratio)을 **클리핑(clipping)**하여 정책 변화를 암묵적으로 제한하는 단순한 1차 최적화 목적함수 $L^{CLIP}$ 제안. 같은 데이터로 여러 에포크 학습 가능

## 목차
- Section 1: Introduction
- Section 2: Background — Policy Gradient & TRPO
- Section 3: Clipped Surrogate Objective — PPO의 핵심
- Section 4: Adaptive KL Penalty Coefficient
- Section 5: Algorithm — Actor-Critic Style PPO
- Section 6: Experiments — MuJoCo, Atari 벤치마크
- Section 7: Conclusion

---

## Section 1: Introduction

**요약**

강화학습에서 신경망 함수 근사기를 사용하는 세 가지 주류 접근법(DQN, Vanilla PG, TRPO) 중 어느 것도 확장성, 데이터 효율, 강건성을 동시에 만족하지 못했습니다. 이 논문은 **TRPO의 신뢰성을 가지면서도 vanilla PG처럼 간단한 구현**을 가능하게 하는 PPO를 제안합니다.

핵심 아이디어는 새로운 정책이 이전 정책에서 너무 멀어지지 않도록, 확률 비율을 특정 범위로 **클리핑**하는 대리 목적함수(surrogate objective)를 사용하는 것입니다.

**핵심 개념**
- **Policy Gradient**: 정책의 로그 확률에 이득(advantage)을 곱한 기댓값을 최대화하는 방법
- **Trust Region**: 정책 업데이트가 안전한 범위 내에 있도록 제한하는 영역
- **Surrogate Objective**: 실제 목적함수를 직접 최적화하는 대신, 근사적으로 대체하는 함수

---

## Section 2: Background — Policy Optimization

**요약**

PPO의 기반이 되는 두 가지 기존 방법을 소개합니다.

### 2.1 Policy Gradient Methods

가장 기본적인 정책 경사 추정량:

$$\hat{g} = \hat{\mathbb{E}}_t\left[\nabla_\theta \log \pi_\theta(a_t \mid s_t) \hat{A}_t\right]$$

**수식 설명**:
- **$\hat{g}$**: 추정된 정책 기울기 (그래디언트)
- **$\pi_\theta(a_t \mid s_t)$**: 파라미터 $\theta$를 가진 정책. 상태 $s_t$에서 행동 $a_t$를 선택할 확률
- **$\nabla_\theta \log \pi_\theta$**: 정책의 로그 확률의 기울기. "이 행동을 더 자주/덜 하도록 파라미터를 어떻게 바꿔야 하는가"
- **$\hat{A}_t$**: 어드밴티지 함수 추정값. "이 행동이 평균보다 얼마나 좋은가" (양수 = 좋음, 음수 = 나쁨)
- **$\hat{\mathbb{E}}_t[\ldots]$**: 샘플들에 대한 경험적 평균

이를 자동 미분으로 구현하기 위한 목적함수:

$$L^{PG}(\theta) = \hat{\mathbb{E}}_t\left[\log \pi_\theta(a_t \mid s_t) \hat{A}_t\right]$$

> 문제: 동일 데이터로 $L^{PG}$를 여러 번 최적화하면 정책이 과도하게 크게 업데이트되어 불안정해집니다.

### 2.2 Trust Region Methods (TRPO)

TRPO는 정책 업데이트 크기를 KL divergence로 제한합니다:

$$\underset{\theta}{\text{maximize}} \quad \hat{\mathbb{E}}_t\left[\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}\hat{A}_t\right]$$

$$\text{subject to} \quad \hat{\mathbb{E}}_t[\text{KL}[\pi_{\theta_\text{old}}(\cdot \mid s_t), \pi_\theta(\cdot \mid s_t)]] \leq \delta$$

**수식 설명**:
- **$\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}$**: 확률 비율(probability ratio). 새 정책이 이전 정책보다 해당 행동을 얼마나 더/덜 선택하는지
- **$\delta$**: KL divergence 허용 임계값. 정책이 이 이상 변하면 안 됨
- **KL divergence 제약**: 두 정책의 분포 차이를 $\delta$ 이하로 유지

또한 이론적으로는 페널티 형태로도 표현 가능합니다:

$$\underset{\theta}{\text{maximize}} \; \hat{\mathbb{E}}_t\left[\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}\hat{A}_t - \beta \, \text{KL}[\pi_{\theta_\text{old}}(\cdot \mid s_t), \pi_\theta(\cdot \mid s_t)]\right]$$

**수식 설명**:
- **$\beta$**: KL 페널티 계수. 정책 변화에 얼마나 강하게 패널티를 부과할지
- 문제: 최적 $\beta$를 고정값으로 정하기 어려워 TRPO는 하드 제약(hard constraint)을 선택

**핵심 개념**
- **확률 비율 $r_t(\theta)$**: $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}$. $\theta = \theta_\text{old}$이면 항상 1
- **KL Divergence**: 두 확률 분포의 차이 측정. 정책이 얼마나 변했는지의 척도

---

## Section 3: Clipped Surrogate Objective (핵심 기여)

**요약**

PPO의 핵심인 **클리핑된 대리 목적함수**를 제안합니다. TRPO의 복잡한 2차 최적화 없이도 안정적인 정책 업데이트를 달성합니다.

### Conservative Policy Iteration (CPI) 목적함수

먼저 확률 비율 $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}$를 정의하면:

$$L^{CPI}(\theta) = \hat{\mathbb{E}}_t\left[\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}\hat{A}_t\right] = \hat{\mathbb{E}}_t\left[r_t(\theta)\hat{A}_t\right]$$

**수식 설명**:
- **$r_t(\theta)$**: 확률 비율. 새 정책/이전 정책. 1이면 변화 없음, 2이면 해당 행동 확률이 2배
- 제약 없이 $L^{CPI}$를 최대화하면 $r_t$가 1에서 너무 멀어져 정책이 과도하게 변함

### PPO의 핵심: Clipped Surrogate Objective

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\left[\min\left(r_t(\theta)\hat{A}_t, \;\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

**수식 설명**:
- **$\epsilon$**: 클리핑 하이퍼파라미터 (보통 $\epsilon = 0.2$). 확률 비율이 $[1-\epsilon, 1+\epsilon]$ 밖으로 나가지 못하게 제한
- **$\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)$**: 확률 비율을 $[0.8, 1.2]$ 범위로 강제 제한
- **$\min(\ldots)$**: 클리핑된 버전과 원래 버전 중 **더 작은 값(비관적 하한)** 선택

> 직관적 설명:
> - **어드밴티지 $\hat{A}_t > 0$** (좋은 행동): $r_t$를 $1+\epsilon$보다 크게 키워봤자 목적함수가 더 이상 늘어나지 않음 → 과도한 업데이트 방지
> - **어드밴티지 $\hat{A}_t < 0$** (나쁜 행동): $r_t$를 $1-\epsilon$보다 작게 줄여봤자 목적함수가 더 이상 줄어들지 않음 → 과도한 억제 방지
> - 결과적으로 "정책을 개선하되, 너무 급격히 바꾸지 말라"는 효과

이 목적함수는 $L^{CPI}$의 **비관적 하한(pessimistic lower bound)**으로, 정책 변화에 자동으로 페널티를 부여합니다.

**핵심 개념**
- **Clipping**: 확률 비율의 범위를 $[1-\epsilon, 1+\epsilon]$으로 제한하여 정책 변화의 크기를 암묵적으로 통제
- **Pessimistic Lower Bound**: min을 취함으로써 목적함수가 실제 성능보다 낙관적으로 추정하지 않도록 보장

---

## Section 4: Adaptive KL Penalty Coefficient

**요약**

클리핑 대신 KL 페널티를 적응적으로 조절하는 대안 방법을 제시합니다. 실험에서는 클리핑이 더 좋은 성능을 보였지만, 중요한 베이스라인으로 소개합니다.

### Adaptive KL 목적함수

$$L^{KLPEN}(\theta) = \hat{\mathbb{E}}_t\left[\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}\hat{A}_t - \beta \, \text{KL}[\pi_{\theta_\text{old}}(\cdot \mid s_t), \pi_\theta(\cdot \mid s_t)]\right]$$

각 정책 업데이트 후 KL divergence $d$를 계산하여 $\beta$를 동적 조정:
- $d < d_\text{targ}/1.5$ 이면: $\beta \leftarrow \beta/2$ (패널티 완화 — 정책 변화가 너무 작음)
- $d > d_\text{targ} \times 1.5$ 이면: $\beta \leftarrow \beta \times 2$ (패널티 강화 — 정책 변화가 너무 큼)

**수식 설명**:
- **$d_\text{targ}$**: 목표 KL divergence 값 (사용자 설정)
- **$\beta$**: 자동으로 조절되는 KL 패널티 계수
- **핵심**: 고정 $\beta$의 단점을 극복하여 다양한 문제에 자동 적응

---

## Section 5: Algorithm — Actor-Critic Style PPO

**요약**

실용적인 PPO 구현은 **Actor-Critic** 구조를 사용합니다. N개의 병렬 환경에서 T 타임스텝 데이터를 수집하고, K 에포크 동안 미니배치 SGD로 최적화합니다.

### 전체 목적함수

정책 손실 + 가치 함수 손실 + 엔트로피 보너스를 결합합니다:

$$L_t^{CLIP+VF+S}(\theta) = \hat{\mathbb{E}}_t\left[L_t^{CLIP}(\theta) - c_1 L_t^{VF}(\theta) + c_2 S[\pi_\theta](s_t)\right]$$

**수식 설명**:
- **$L_t^{CLIP}(\theta)$**: 클리핑된 정책 목적함수 (최대화)
- **$L_t^{VF}(\theta) = (V_\theta(s_t) - V_t^\text{targ})^2$**: 가치 함수의 평균 제곱 오차 (최소화)
- **$S[\pi_\theta](s_t)$**: 정책의 엔트로피 보너스. 탐색(exploration)을 장려
- **$c_1, c_2$**: 각 항의 가중치 계수

### GAE (Generalized Advantage Estimation)

어드밴티지를 추정하기 위해 GAE를 사용합니다:

$$\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + \cdots + (\gamma\lambda)^{T-t+1}\delta_{T-1}$$

$$\text{where} \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

**수식 설명**:
- **$\delta_t$**: TD(시간차) 오차. "한 스텝 앞을 보았을 때 예상보다 보상이 얼마나 좋았는가"
- **$\gamma$**: 할인율(discount factor). 미래 보상의 현재 가치 ($\gamma = 0.99$이면 미래 보상을 거의 전부 고려)
- **$\lambda$**: GAE 파라미터. 편향(bias)과 분산(variance) 간 트레이드오프 조절 ($\lambda = 0$: 낮은 분산/높은 편향, $\lambda = 1$: 높은 분산/낮은 편향)
- **$V(s_t)$**: 상태 $s_t$의 가치 함수 추정값 (critic이 예측)

### PPO 알고리즘 (Algorithm 1)

```
for iteration = 1, 2, ... do
    for actor = 1, 2, ..., N do
        이전 정책 π_old로 T 타임스텝 환경 실행
        어드밴티지 추정값 Â_1, ..., Â_T 계산
    end for
    NT 타임스텝 데이터로 surrogate L을 K 에포크, 미니배치 크기 M ≤ NT로 최적화
    θ_old ← θ
end for
```

**핵심 개념**
- **Actor-Critic**: Actor(정책)가 행동을 선택하고, Critic(가치함수)이 행동의 좋고 나쁨을 평가
- **병렬 Actor**: N개의 환경을 동시에 실행하여 데이터 수집 효율화
- **미니배치 SGD**: 수집한 NT 타임스텝을 작은 배치로 나눠 K 에포크 학습

---

## Section 6: Experiments

**요약**

MuJoCo 연속 제어 태스크 7개와 Atari 49개 게임에서 PPO의 성능을 검증합니다.

### 6.1 대리 목적함수 비교 (MuJoCo)

| 알고리즘 | 평균 정규화 점수 |
|---------|--------------|
| No clipping or penalty | -0.39 |
| Clipping, $\epsilon=0.1$ | 0.76 |
| **Clipping, $\epsilon=0.2$** | **0.82** |
| Clipping, $\epsilon=0.3$ | 0.70 |
| Adaptive KL $d_\text{targ}=0.01$ | 0.74 |
| Fixed KL, $\beta=1$ | 0.71 |

- **$\epsilon=0.2$ 클리핑**이 모든 설정 중 최고 성능
- 클리핑 없이 페널티도 없으면 음수 점수 (불안정한 학습)

### 6.2 연속 제어 도메인 비교

MuJoCo 7개 환경(HalfCheetah, Hopper, Walker2d 등)에서 비교:
- **PPO > A2C, TRPO, CEM, Vanilla PG** (거의 모든 환경에서)
- 1백만 타임스텝 학습, 3개 랜덤 시드로 평가

### 6.3 3D 휴머노이드 (Roboschool)

고차원 연속 제어(3D 휴머노이드 달리기/방향 전환)에서도 안정적으로 학습:
- RoboschoolHumanoid: 1억 타임스텝 학습
- 32~128개 병렬 Actor 사용

### 6.4 Atari 게임 비교

49개 Atari 게임에서 A2C, ACER와 비교:

| 기준 | A2C | ACER | **PPO** |
|------|-----|------|---------|
| 전체 훈련 평균 보상 | 1 | 18 | **30** |
| 마지막 100 에피소드 | 1 | **28** | 19 |

- 전체 훈련 기간 기준으로 PPO가 49개 중 **30개**에서 승리 (A2C: 1개, ACER: 18개)
- 최종 성능은 ACER가 약간 우세하지만, PPO가 훨씬 빠르게 학습

**실험 하이퍼파라미터 (MuJoCo)**:
- Horizon T = 2048, Adam stepsize = $3\times10^{-4}$
- Epochs = 10, Minibatch = 64, $\gamma = 0.99$, GAE $\lambda = 0.95$

---

## 핵심 개념 정리

| 개념 | 설명 |
|------|------|
| **확률 비율 $r_t(\theta)$** | 새 정책/이전 정책의 행동 확률 비율. 1 = 변화 없음 |
| **$L^{CLIP}$** | 확률 비율을 $[1-\epsilon, 1+\epsilon]$으로 클리핑한 대리 목적함수. PPO의 핵심 |
| **클리핑 파라미터 $\epsilon$** | 정책 변화의 허용 범위. 기본값 0.2 (±20%) |
| **어드밴티지 $\hat{A}_t$** | 해당 행동이 평균 대비 얼마나 좋은지. GAE로 추정 |
| **GAE** | Generalized Advantage Estimation. 편향-분산 트레이드오프를 $\lambda$로 조절 |
| **Actor-Critic** | 정책(Actor)과 가치함수(Critic)를 함께 학습. 파라미터 공유 가능 |
| **엔트로피 보너스** | 정책의 무작위성을 유지하여 충분한 탐색을 장려 |
| **적응적 KL 페널티** | 실제 KL 발산이 목표값과 다르면 $\beta$를 자동 조정하는 대안 방법 |

---

## 결론 및 시사점

**논문의 결론**

PPO는 **TRPO의 신뢰성과 Vanilla PG의 단순성을 동시에 달성**합니다. 단 몇 줄의 코드 변경만으로 기존 정책 경사 구현에 적용 가능하며, MuJoCo 연속 제어와 Atari 이산 제어 모두에서 우수한 성능을 보입니다.

**실무적 시사점**

1. **구현 용이성**: TRPO처럼 2차 최적화 불필요. `min(r * A, clip(r, 1-ε, 1+ε) * A)` 한 줄로 핵심 구현 가능
2. **데이터 효율**: 동일 데이터를 K=10 에포크 재사용하여 샘플 효율성 향상
3. **범용성**: 연속/이산 행동 공간, 단일/공유 네트워크 아키텍처 모두 지원
4. **현재 위치**: ChatGPT/GPT-4 등 LLM의 RLHF(인간 피드백 강화학습) 파인튜닝에서 현재까지 표준 알고리즘으로 사용됨
5. **자율주행 응용**: 복잡한 연속 제어 문제(조향, 가속)에 직접 적용 가능한 안정적인 프레임워크 제공
6. **하이퍼파라미터**: $\epsilon=0.2$, $\gamma=0.99$, $\lambda=0.95$가 대부분의 문제에서 잘 동작하는 기본값
