# Deep Reinforcement Learning for Stock Portfolio Optimisation

*Undergraduate dissertation — Mahidol University International College, 2023*

Link to full report: [pdf](https://github.com/pavanpreet-gandhi/portfolio-optimization-rl/blob/main/Report.pdf)

---

## Overview

This dissertation asks whether a straightforward **Deep Q‑Learning** (DQN) agent can match— or even outperform—a classical mean‑variance portfolio when dynamically rebalancing a basket of ten Dow Jones stocks. By framing portfolio management as a reinforcement‑learning task, the study measures how purely reward‑driven decisions stand up against two traditional benchmarks under real‑world market conditions.

---

## Experimental Setup

| Component              | Design Choices                                                                                              |
| ---------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Universe**           | 10 Dow Jones constituents (AAPL, DIS, JNJ, … ) plus cash.                                                   |
| **Timeline**           | 2010‑2019 train · 2020‑2021 validate · 2022 test.                                                           |
| **Features per stock** | Daily return, 12/26/60‑day rolling return, 20‑day rolling volatility, volume Δ. 6 × 10 stocks → 60 signals. |
| **State vector**       | 10‑day window of features + current 10 asset weights + risk‑free rate → 621‑D observation.                  |
| **Action space**       | 21 discrete actions (buy / sell each stock in 10 % steps + hold).                                           |
| **Reward**             | Cumulative portfolio return, scaled by episode progress (no risk penalty).                                  |
| **Agent**              | DQN (MLP 2×64) from *Stable‑Baselines3*, 3 M timesteps, ε‑greedy exploration.                               |
| **Baselines**          | (1) DJIA buy‑and‑hold, (2) daily‑rebalanced maximum‑Sharpe portfolio.                                       |

---

## Performance Snapshot

| Split                       | Metric   | DJIA       | Max‑Sharpe | **DQN**    |
| --------------------------- | -------- | ---------- | ---------- | ---------- |
| **Train (2010‑19)**         | Return   | 10.3 %     | 19.1 %     | **20.6 %** |
|                             | Risk (σ) | 14.1 %     | **17.0 %** | 21.6 %     |
|                             | Sharpe   | 0.73       | **1.13**   | 0.95       |
| **Validate (2020‑21)**      | Return   | 12.1 %     | **31.8 %** | 27.8 %     |
|                             | Risk (σ) | **27.7 %** | 31.9 %     | 33.3 %     |
|                             | Sharpe   | 0.44       | **1.00**   | 0.84       |
| **Test (2022 — recession)** | Return   | −7.9 %     | −3.7 %     | **+8.2 %** |
|                             | Risk (σ) | 20.2 %     | 19.6 %     | **17.2 %** |
|                             | Sharpe   | −0.39      | −0.19      | **0.47**   |

*Figures reproduced from Tables 4.1–4.3 of the dissertation.*

---

## Key Insights

* **Resilience in Turbulence** – The RL agent was the *only* strategy to finish the recession‑hit 2022 test year in positive territory.
* **Competitive Risk‑Adjusted Returns** – While the purpose‑built maximum‑Sharpe portfolio tops the Sharpe ratio on average, DQN’s out‑performance during drawdowns yields a more stable growth path.
* **Reward Design Matters** – Omitting an explicit risk penalty let the agent chase raw returns; integrating drawdown or variance into the reward could lift the Sharpe further.

---

## Limitations & Future Work

* **Universe Selection Bias** – The study uses just ten blue‑chip DJIA stocks that have historically out‑performed; thus, the emphasis should be on *relative* method comparison rather than the absolute returns achieved.
* Add drawdown/volatility terms in the reward to respect investor risk appetite.
* Incorporate realistic frictions (commissions, slippage).
* Explore continuous action methods (DDPG, SAC), sequence‑aware architectures (GRU/Transformer), **and systematic hyper‑parameter tuning**.
* Test on diverse asset classes (small‑caps, crypto, bonds) to stress‑test generality.

---

### Acknowledgements

This project was completed as part of my undergraduate dissertation under the supervision of **Assoc. Prof. Dr. Chatchawan Panraksa** (major advisor) and **Dr. Sunsern Cheamanunkul** (co‑advisor).

---

### Contact

Feel free to reach out at **[pavanpreet.gandhi@gmail.com](mailto:pavanpreet.gandhi@gmail.com)** or connect on [GitHub](https://github.com/pavangandhi).

*This work is for research & educational purposes only and does **not** constitute financial advice.*
