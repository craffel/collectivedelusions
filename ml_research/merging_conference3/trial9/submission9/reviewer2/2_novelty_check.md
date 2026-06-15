# 2. Novelty Check and Delta from Prior Work

## Key Novel Aspects
PAC-Kinetics introduces several innovative concepts at the intersection of machine learning theory, control theory, and system serving:
1. **Stateful Dynamic Routing with Provable Stability:** Unlike existing heuristic or stateless methods, PAC-Kinetics models the dynamic ensembling weights as state concentrations in a continuous-time chemical reactor. Its discretized discrete-time linear recurrence is proven to be **Globally Asymptotically Stable (GAS)** and **Input-to-State Stable (ISS)** under Lyapunov control theory.
2. **PAC-Bayesian Bound for Dependent Streams ($\beta$-mixing):** Standard PAC-Bayesian frameworks assume independent and identically distributed (i.i.d.) data, which is heavily violated in multi-task query streams. PAC-Kinetics derives a Catoni-type PAC-Bayesian bound customized for stationary $\beta$-mixing stochastic processes, resolving the technical issue of exploding Total Variation (TV) penalties through **Even/Odd Block Splitting**.
3. **Adaptive Online Kinetics:** To suppress the "inertial drag" (routing lag) inherent to low-pass filters during abrupt task transitions, the framework introduces a differentiable, self-modulating retention mechanism. It dynamically scales the state-retention coefficient $a_t$ in real-time based on the cosine similarity of consecutive coordinate vectors.
4. **Theoretical Characterization of the Deterministic Surrogate Gap:** The paper presents a formal sensitivity bound proving that the trajectory discrepancy of the stateful recurrence under parameter perturbations scales quadratically as $(1-\rho)^{-2}$. This mathematically explains why the served deterministic surrogate remains stable while the randomized Gibbs posterior collapses under parameter perturbations.

---

## Delta from Closely Related Prior Work

### 1. Versus Stateless Dynamic Routers (SABLE, SPS-ZCA, Stateless PAC-ZCA)
* **Delta:** SABLE and PAC-ZCA are stateless; they map individual queries directly to routing weights without temporal memory. This makes them highly susceptible to query-level noise, causing severe high-frequency oscillations (routing jitter). 
* **Novelty Level: Significant.** PAC-Kinetics replaces these stateless mappings with a stateful dynamical system that acts as a robust low-pass filter over intermediate representations, slashing routing jitter by up to $16.0\times$ while matching or exceeding their peak accuracies.

### 2. Versus Heuristic Stateful Routers (ChemMerge)
* **Delta:** ChemMerge (2026) is stateful and uses first-order chemical reaction kinetics ODEs to smooth routing. However, it is purely heuristic: its parameters are statically set and do not adapt, causing a catastrophic accuracy collapse (down to $70.49\%$) under heterogeneous streams. It also lacks any learning-theoretic or stability guarantees.
* **Novelty Level: Significant.** PAC-Kinetics replaces ChemMerge's static heuristics with a unified learning framework. By minimizing the PAC-Bayesian generalization bound, PAC-Kinetics learns optimal, task-specific decay rates and cross-task coupling parameters that are robust to heterogeneous streams. It also mathematically proves the contractive and Lyapunov stability of the dynamics, which ChemMerge lacks.

### 3. Versus Traditional PAC-Bayesian Generalization Theory
* **Delta:** Classical PAC-Bayesian bounds (McAllester, Catoni) assume i.i.d. samples. While some theoretical works (e.g., Alquier, 2013) have established bounds for mixing processes, PAC-Kinetics is the first to apply this theory to the parameter-space optimization of stateful, sequential model ensembling. Furthermore, it explicitly resolves the "exploding TV penalty" that occurs when coupling unbounded exponential moments under Catoni's inequality.
* **Novelty Level: Moderate-to-Significant.** The mathematical integration of Even/Odd Block Splitting with Catoni's bound for stateful linear recurrences represents a rigorous, non-trivial theoretical contribution.

---

## Characterization of Novelty
The overall novelty of the paper is **significant**. Instead of proposing an incremental heuristic tweak, the authors have successfully integrated concepts from three distinct disciplines—control-theoretic Lyapunov stability, chemical reaction kinetics, and PAC-Bayesian generalization theory—to address a major, real-world systems bottleneck in dynamic model serving. The resulting framework is both mathematically elegant and empirically performant, providing a strong blueprint for stable test-time ensembling.
