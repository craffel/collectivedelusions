# Intermediate Evaluation 4: Experimental Evaluation Check

## 1. The Synthetic Sandbox Limitation
The primary empirical validation of Dirichlet-PAC is conducted inside the **Analytical Coordinate Sandbox (ICS)**, a custom-designed, 14-layer simulation with $D=192$ and $K=4$ task experts. 
- While synthetic environments are useful for controlled sweeps, they are highly prone to "cherry-picked" configurations. 
- For instance, the representational interference noise in the sandbox is modeled as directly proportional to the ensembling entropy: $\text{noise\_scale} = \eta \cdot \text{entropy}(\boldsymbol{\alpha})$. 
- By explicitly injecting noise proportional to the routing entropy, the authors have hard-coded a penalty for high-entropy (uniform) routing. It is highly circular and unsurprising that a entropy-regularized router (like Dirichlet-PAC) outperforms Uniform Merging (which has maximum ensembling entropy) under this specific noise model. The evaluation is essentially pre-conditioned to favor the proposed method.

## 2. Inadequate and Weak Baselines
- **Omission of Simple Regularization:** The paper compares Dirichlet-PAC against unregularized Temp-Only ERM to show that complexity control is necessary. However, they completely omit standard regularization techniques like **L2 regularization (weight decay)** or **L1 regularization** on the log-temperature parameters $\mathbf{w}$. Comparing against unregularized ERM is a "strawman" comparison; the authors must demonstrate that their complex PAC-Bayesian penalty outperforms standard, simpler L2/L1 penalties to justify its overhead.
- **SABLE (Raw Coords) Superiority:** In Table 1, the simple, static baseline **SABLE (Raw Coords)** ($\tau = 0.05$) achieves **79.02% ± 0.98%** on orthogonal manifolds, which actually **outperforms** the proposed supervised **Dirichlet-PAC** (**77.88% ± 1.19%**). On overlapping manifolds, they are statistically tied (**76.66%** vs. **76.32%**). This shows that the complex PACbound-derived optimization fails to beat a simple, static uncalibrated heuristic.

## 3. Major Shortcoming: Real-World Underperformance
The most damning empirical finding in the paper is in the **Real-World Validation on Pre-Trained BERT Backbones** (Table 3).
- On **BERT-Tiny**, SABLE Norm achieves **96.00%**, while the proposed Dirichlet-PAC and PEM-Div achieve **94.67%** (underperforming the uncalibrated baseline).
- On **BERT-Medium**, SABLE Norm and static Uniform Merging both achieve **96.00%**, while Dirichlet-PAC drops to **89.33%** and PEM-Div drops to **88.67%** (a significant degradation of **--6.67%** and **--7.33%**, respectively).
- On **BERT-Base**, Uniform Merging achieves **95.33%** and SABLE Norm achieves **94.67%**, while Dirichlet-PAC drops to **92.00%** and PEM-Div to **91.33%** (underperforming by **--3.33%** and **--4.00%**).
- **Crucial Analysis:** In every single real-world backbone scale (Tiny, Medium, Base), the proposed Dirichlet-PAC and PEM-Div **underperform** the simpler, static baselines (Uniform Merging, SABLE Norm).
- The authors attempt to explain this away by arguing that the physical BERT tasks are perfectly orthogonal, where static priors represent a "local ceiling" and that Dirichlet-PAC is only superior in overlapping settings. However, the authors **never evaluate the BERT models under overlapping or noisy settings.** 
- Thus, the paper contains **zero physical evidence** that Dirichlet-PAC is superior on real-world networks under any conditions. Its claim of superiority rests entirely on a custom, synthetic sandbox pre-conditioned to favor entropy-regularized routing. On real models and standard tasks, the proposed method actually hurts performance compared to simple parameter averaging or uncalibrated static routing, while introducing optimization latency and high mathematical complexity.
