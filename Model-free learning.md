---
title: Model-free learning
draft: true
slug: model-free
---

# Введение

В model-free алгоритмах мы аппроксимируем функцию ценности действия $Q^\pi(s, a)$ или саму политику $\pi(a \mid s)$ в value-based и policy-based подходах соответственно. Также выделяют методы, объединяющие эти два подхода: actor-critic. Они включают в себя как обучение политики, так и функции ценности действия, что делает их более устойчивыми и качественными.

Основное преимущество value-based обучения заключается в том, что оно позволяет агенту планировать, оценивать, что может произойти при разных возможных действиях, и явно выбирать между альтернативами. Затем агент может преобразовать результаты такого планирования в оптимальную политику, что может давать лучшие результаты.

С другой стороны, агенту, как правило, неизвестна истинная модель окружения, поэтому всё, что ему остаётся, — строить аппроксимацию на основе собственного опыта. При обучении на недостаточном количестве итераций существует риск того, что агент будет использовать смещение в модели, демонстрируя хороший результат на обучении, но плохой — при инференсе в реальной среде. Кроме того, воссоздание модели среды само по себе неэффективно и вычислительно затратно.

# Policy-based

В policy-based подходах мы аппроксимируем функцию политики $\pi(a \mid s)$ некоторой функцией $\pi_{\theta}(a \mid s)$, параметризованной $\theta$, и ищем такие параметры $\theta$, которые максимизируют ожидаемую награду $J(\pi_{\theta})$. Один из распространённых методов такой оптимизации — градиентный подъём по градиенту ожидаемой награды.

Пусть $J(\pi_{\theta}) = \mathbb{E}_{\tau \sim \pi_{\theta}}[R(\tau)]$,  тогда перепишем математическое ожидание как интеграл:  
$$
J(\pi_{\theta}) = \int R(\tau) P_{\theta}(\tau) d\tau,
$$
где $P_{\theta}(\tau)$ — вероятность траектории $\tau$, которая вычисляется как  $P_{\theta}(\tau) = p(s_{0}) \prod_{t=0}^{T} \pi_{\theta}(a_{t} \mid s_{t}) P(s_{t+1} \mid s_{t}, a_{t})$.

Запишем градиент по $\theta$:
$$
\nabla_{\theta} J(\pi_{\theta})=\nabla_{\theta} \int R(\tau) P_{\theta}(\tau) d\tau=\int R(\tau) \nabla_{\theta} P_{\theta}(\tau) d\tau.
$$

Рассмотрим градиент $\nabla_{\theta} P_{\theta}(\tau)$. Перепишем его следующим образом:
$$
\nabla_{\theta} P_{\theta}(\tau)=P_{\theta}(\tau)\nabla_{\theta} \ln P_{\theta}(\tau).
$$

Тогда
$$
\nabla_{\theta} \ln P_{\theta}(\tau)=\nabla_{\theta}
\left(
\ln p(s_{0})+
\sum_{t=0}^{T}
\ln \pi_{\theta}(a_{t} \mid s_{t})+\sum_{t=0}^{T}\ln P(s_{t+1} \mid s_{t}, a_{t})
\right)
$$

Так как динамика среды и распределение начального состояния не зависят от $\theta$, остаётся:
$$
\nabla_{\theta} \ln P_{\theta}(\tau)=\sum_{t=0}^{T}
\nabla_{\theta} \ln \pi_{\theta}(a_{t} \mid s_{t}).
$$

Подставляя это в выражение для градиента, получаем теорему о policy gradient:
$$
\nabla_{\theta} J(\pi_{\theta})=
\mathbb{E}_{\tau \sim \pi_{\theta}}
\left[
\sum_{t=0}^{T}
\nabla_{\theta} \ln \pi_{\theta}(a_{t} \mid s_{t})
R(\tau)
\right].
$$

Используя тот факт, что $Q^\pi(s_t, a_t) = \mathbb{E}[R(\tau) \mid s_t, a_t]$, и что вычитание базовой функции $V^\pi(s_t)$ не изменяет математическое ожидание, перепишем градиент:
$$
\nabla_{\theta} J(\pi_{\theta})=
\mathbb{E}_{\tau \sim \pi_{\theta}}
\left[
\sum_{t=0}^{T}
\nabla_{\theta} \ln \pi_{\theta}(a_{t} \mid s_{t})
A^\pi(s_t, a_t)
\right],
$$
где $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$ — функция преимущества.

Итеративное обновление параметров:
$$
\theta_{k+1}
\leftarrow
\theta_k+
\alpha
\nabla_{\theta} J(\pi_{\theta_k}).
$$

При выполнении стандартных условий (достаточная гладкость, корректный шаг обучения, конечная дисперсия оценок градиента и т.д.) алгоритм сходится к локальному оптимуму

**Другие методы оптимизации политики:**

Градиентные:
- REINFORCE  
- Actor-Critic  
- Natural Policy Gradient (NPG)  
- Trust Region Policy Optimization (TRPO)  
- Proximal Policy Optimization (PPO)  
- Deep Deterministic Policy Gradient (DDPG)  
- Soft Actor-Critic (SAC)  

Неградиентные:
- Cross-Entropy Method (CEM)  
- Evolution Strategies (ES)  
- Covariance Matrix Adaptation Evolution Strategy (CMA-ES)  

**Ссылки:**
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning (REINFORCE).
- Kakade, S. (2001). A Natural Policy Gradient.
- Schulman et al. (2015). Trust Region Policy Optimization.
- Schulman et al. (2017). Proximal Policy Optimization Algorithms.
- Lillicrap et al. (2016). Continuous control with deep reinforcement learning (DDPG).
- Haarnoja et al. (2018). Soft Actor-Critic.
- Rubinstein (1999). The Cross-Entropy Method.
- Salimans et al. (2017). Evolution Strategies as a Scalable Alternative to Reinforcement Learning.
- Hansen (2001). CMA-ES.

# Value-based (Q-learning)

В value-based подходах мы аппроксимируем функцию ценности действия $Q^\ast(s, a)$ и строим политику как жадную по этой функции:
$$
\pi(s) = \arg\max_a Q(s, a)
$$

Основой является уравнение Беллмана для оптимальной функции действия:
$$
Q^\ast(s, a)=
\mathbb{E}
\left[
r_t+
\gamma
\max_{a'} Q^\ast(s_{t+1}, a')
\mid
s_t = s, a_t = a
\right].
$$

Алгоритм Q-learning использует стохастическое приближение:
$$
Q(s_t, a_t)
\leftarrow
Q(s_t, a_t)
+
\alpha
\left(
r_t
+
\gamma
\max_{a'} Q(s_{t+1}, a')
-
Q(s_t, a_t)
\right).
$$

Особенность Q-learning — off-policy характер: обновление использует максимум по действиям, независимо от текущей поведенческой политики.

При использовании нейросетевой аппроксимации получаем Deep Q-Network (DQN), где минимизируется функция потерь:
$$
\mathcal{L}(\theta)=
\mathbb{E}
\left[
\left(
r_t
+
\gamma
\max_{a'} Q_{\theta^-}(s_{t+1}, a')
-
Q_{\theta}(s_t, a_t)
\right)^2
\right].
$$

**Ссылки:**
- Watkins, C. J. C. H. (1989). Learning from Delayed Rewards.
- Mnih et al. (2015). Human-level control through deep reinforcement learning (DQN).
- Hasselt (2010). Double Q-learning.
- Wang et al. (2016). Dueling Network Architectures.

# Actor-critic

В actor-critic методах одновременно обучаются:

- актор — политика $\pi_{\theta}(a \mid s)$,
- критик — функция ценности $V_{w}(s)$ или $Q_{w}(s, a)$.

Критик обучается минимизацией TD-ошибки:
$$
\delta_t=
r_t
+
\gamma V_w(s_{t+1})
-
V_w(s_t).
$$

Актор обновляется по градиенту:
$$
\nabla_{\theta} J(\pi_{\theta})
\approx
\mathbb{E}
\left[
\nabla_{\theta}
\ln \pi_{\theta}(a_t \mid s_t)
\delta_t
\right].
$$

Таким образом, критик снижает дисперсию оценки градиента, а актор напрямую оптимизирует политику