# 2. Novelty Check

## Key Novel Aspects
The primary novelty of **PAC-Kinetics** lies in its **interdisciplinary synthesis of four separate fields**:
1. **Dynamic Model Serving:** Addressing the "routing jitter paradox" and "cascading representation collapse" in sequential, multi-expert LoRA ensembling.
2. **Chemical Reaction Kinetics:** Modeling routing trajectories as continuous concentration dynamics.
3. **Control-Theoretic Stability:** Formulating a stateful linear state-space recurrence and proving Global Asymptotic Stability (GAS) and Input-to-State Stability (ISS) using Lyapunov candidate functions.
4. **PAC-Bayesian Generalization Theory:** Deriving parameter-space out-of-sample expected risk bounds under non-i.i.d. $\beta$-mixing stochastic sequences, utilizing the **Even/Odd Block Splitting** technique to prevent exploding Total Variation (TV) penalties.

A major algorithmic novelty is the **Adaptive Online Kinetics** mechanism. This feature dynamically scales down retention coefficients using the cosine similarity of consecutive coordinate vectors ($Sim_t$), effectively turning off memory during rapid switches to eliminate "inertial drag" or routing lag.

## The 'Delta' from Prior Work

### 1. Comparison to Stateless Routers (SABLE, SPS-ZCA, PAC-ZCA)
- **Prior Work:** SABLE and SPS-ZCA perform sample-wise activation blending by extracting coordinate representation signals from early layers and mapping them immediately to routing weights in a stateless manner. PAC-ZCA applies a PAC-Bayesian framework but restricts optimization to a static, stateless Gibbs temperature parameter.
- **Delta:** PAC-Kinetics introduces a stateful linear recurrence $s_t = \mathbf{A} s_{t-1} + W \mathbf{e}_t$. Rather than processing queries in isolation, it maintains a running state vector $s_t$ that acts as a robust low-pass filter over raw coordinate signals, drastically reducing routing jitter.

### 2. Comparison to Heuristic Stateful Routers (ChemMerge)
- **Prior Work:** ChemMerge treats model ensembling as a continuous-time multi-component chemical reactor using first-order reaction kinetics ODEs, successfully smoothing trajectories. However, it is purely heuristic; parameters are statically set and lack any stability proofs or out-of-sample generalization guarantees, causing severe accuracy drops under heterogeneous workloads.
- **Delta:** PAC-Kinetics derives the state recurrence from a continuous-time chemical kinetics state space model, but maps the log-parameters to unconstrained, learnable parameters. It provides formal proofs of GAS and ISS, optimizes all kinetics and routing parameters via gradient descent on a PAC-Bayesian objective, and introduces Adaptive Online Kinetics to prevent rigid memory collapses.

### 3. Comparison to Standard PAC-Bayesian Theory
- **Prior Work:** Traditional PAC-Bayesian bounds (e.g., McAllester, Catoni) assume that training and test samples are independent and identically distributed (i.e.d.). Some prior works (e.g., Alquier 2013) have established PAC-Bayesian bounds for stationary mixing sequences, but these have not been applied to parameter-space optimization of stateful dynamical networks for test-time expert routing.
- **Delta:** The authors derive a novel Catoni-type PAC-Bayesian bound specifically for stateful sequential ensembling. They successfully resolve a critical theoretical vulnerability: when coupling dependent sequences inside unbounded exponential moments, direct application of Yu's coupling lemma incurs an exploding TV penalty scaling as $\exp(\lambda a)$, making the bound vacuous. PAC-Kinetics resolves this by employing the **Even/Odd Block Splitting** technique to apply concentration inequalities on independent, coupled blocks.

## Scholarly Positioning & Missing Prior Art/Attributions
While the submission is mathematically rigorous and highly original, a scholarly evaluation reveals some critical areas where the historical context and related work can be better situated:

### 1. Historical Context of Sequential Model Ensembling (Online Learning)
The paper claims to propose "the first learning-theoretic framework for stateful, sequential model ensembling." However, the rich literature on **Online Learning and Prediction with Expert Advice** has studied stateful, sequential model ensembling with rigorous mathematical guarantees for decades. 
- Algorithms like the **Hedge Algorithm** / **Multiplicative Weights Update (MWU)** (Freund & Schapire 1997, Littlestone & Warmuth 1994) and **Online Convex Optimization** (Zinkevich 2003, Hazan 2016) maintain a running state of weights based on historical expert performances, updating them recursively.
- These algorithms provide strict, non-asymptotic *regret bounds* under arbitrary, adversarial (non-i.i.d.) streams without requiring any stationarity or mixing assumptions.
- **Delta & Connection:** PAC-Kinetics operates in the parameter space of a dynamical router and optimizes expected risk under mixing processes, which is a different setting from online regret minimization. However, acknowledging and contrasting PAC-Kinetics with classical online learning (Hedge, FTRL, FTL) is crucial for a complete scholarly characterization of the stateful sequential ensembling landscape.

### 2. Control Theory in Deep Learning & State-Space Models (SSMs)
The stateful recurrence $s_t = \mathbf{A} s_{t-1} + W \mathbf{e}_t$ is mathematically equivalent to a multi-dimensional linear State-Space Model (SSM). 
- In deep learning, there is a prominent line of work modeling sequential dependencies using linear SSMs, such as **Structured State Spaces (S4)** (Gu et al. 2021) and **Mamba** (Gu & Dao 2023). 
- Furthermore, modeling neural network blocks or controllers as continuous-time dynamical systems (e.g., Neural ODEs, Chen et al. 2018) is highly related.
- **Delta & Connection:** PAC-Kinetics applies a linear SSM specifically as a routing controller rather than a sequence backbone. Highlighting this connection would situate the work within the modern trend of using SSMs to replace or augment attention mechanism representations.

### 3. Classic Control-Theoretic Attributions
The proofs of Global Asymptotic Stability (GAS) and Input-to-State Stability (ISS) in Section 3.7 are correct but represent direct applications of classical linear systems theory and Lyapunov stability analysis.
- These proofs should explicitly cite foundational control theory literature, such as **Lyapunov's Direct Method** (Khalil 2002) and Sontag's foundational formulation of **Input-to-State Stability (ISS)** (Sontag 1989).

## Characterization of Novelty
The novelty of the paper is **significant but integrative**. Rather than proposing an entirely new machine learning paradigm from scratch, it masterfully merges distinct mathematical disciplines—non-equilibrium chemical kinetics, linear control systems, and PAC-Bayesian concentration theory—to solve a critical, real-world systems bottleneck in PEFT serving. By bridging these fields, it establishes a new standard of mathematical safety for dynamic model ensembling.
