# Intermediate Evaluation 1: Summary of the Paper

## Main Topic and Motivation
The paper addresses the challenge of serving multiple task-specific Parameter-Efficient Fine-Tuning (PEFT) expert adapters (e.g., LoRA) on a highly heterogeneous, sample-by-sample online stream where task labels are unknown. Static weight merging causes parameter interference and representation blurring under such conditions. While dynamic Mixture-of-Experts (MoE) ensembling with stateless similarity-based routers (such as SABLE) can compute routing weights on-the-fly, these stateless systems suffer from **routing jitter**—high-frequency layer-to-layer oscillations in ensembling weights triggered by representational noise. This jitter degrades final accuracy by causing cascading representational drift. 

To stabilize these routing trajectories, previous state-of-the-art work (ChemMerge) introduced a continuous-time framework modeling ensembling coefficients as chemical concentrations governed by biochemical kinetics and Ordinary Differential Equations (ODEs) solved via numerical integrators. The motivation of this paper is to apply Occam's razor to this continuous stateful formulation, asking whether the biochemical metaphors and ODE solvers are unnecessarily complex.

## Proposed Approach
The authors propose **Momentum-Merge**, a training-free and single-parameter dynamic ensembling framework. Instead of a complex system of biochemical rate equations, Momentum-Merge stabilizes routing trajectories using a standard, discrete **constant Exponential Moving Average (EMA)** update on ensembling weights across network depth. It requires a single hyperparameter (momentum coefficient $\beta \in [0, 1]$) and can be written in a single line of code.

The paper introduces two variants of Momentum-Merge:
1. **Base Variant:** Employs global early-layer centroids for similarity matching, a constant momentum parameter $\beta$, and a uniform boundary initialization ($\alpha_k^{(L_{\text{frozen}})} = 1/K$).
2. **Advanced Variant:** Incorporates **Layer-wise Centroid Calibration** (computing task centroids layer-by-layer across network depth to align similarities within local coordinate spaces) and **Raw Boundary Initialization** (initializing the recurrence with the first adapted layer's raw routing weight to eliminate early-layer damping and transient lag).

## Key Findings and Claims
1. **Mathematical Equivalence:** Under assumptions of constant temperature and uniform activation energies across tasks, ChemMerge's continuous ODE integrated via explicit Euler discretization is mathematically equivalent to the constant EMA formulation of Momentum-Merge.
2. **The Accuracy-Stability Trade-off:** Symmetrically evaluating stateless and stateful ensembling systems exposes a fundamental trade-off:
   - Stateless calibrated routing (SABLE + Layer Centroids) achieves the highest joint accuracy (**77.24%**) due to absolute local expert plasticity, but exhibits high routing jitter (**0.0285**).
   - Stateful temporal smoothing (EMA or ODE kinetics) acts as a low-pass filter, trading off a minor fraction of local classification accuracy to stabilize representation trajectories.
3. **Empirical Superiority with Minimalist Design:** Evaluated in the Analytical Coordinate Sandbox (ICS):
   - **Momentum-Merge (Base)** achieves **74.85%** joint accuracy and slashes routing jitter to **0.0128** (a 5.7$\times$ reduction over tuned SABLE), matching or exceeding the fully-tuned ChemMerge SOTA (**74.71%** accuracy, **0.0153** jitter) without any systems-biochemistry or virtual-time ODE integration overhead.
   - **Momentum-Merge (Advanced)** reaches **74.98%** joint accuracy (recovering 92.41% of the Oracle expert ceiling) while virtually eliminating routing oscillations, dropping jitter to a near-zero **0.000374** (a 195.7$\times$ reduction over tuned SABLE and a 41.1$\times$ reduction over calibrated ChemMerge SOTA).
4. **Physical Interpretability of $\beta$:** Sweeping $\beta$ maps a smooth Pareto frontier from stateless routing ($\beta=0$) to static uniform merging ($\beta=1$), peaking at $\beta = 0.60$.

## Explicitly Claimed Contributions (with Evidence)
- **Mathematical Deconstruction:** Proof of Theorem 3.1, showing the formal mathematical dualism between continuous chemical kinetics and a discrete constant momentum filter.
- **Momentum-Merge Framework:** The parsimonious formulation of Eq. 4, with extensions to Layer-wise Centroid Calibration (Eq. 5) and Raw Boundary Initialization (Eq. 7).
- **Comprehensive Empirical Validation:** Double-blind multi-seed evaluations within the Analytical Coordinate Sandbox (ICS) across 10 random seeds. Evidence is provided in Table 1, showing that Momentum-Merge matches or exceeds SOTA baselines.
- **Robustness sweeps in Appendix:** Systematic hyperparameter sweeps ($\beta \in [0, 1]$, temperature $\tau$, calibration subset size $|\mathcal{C}_k|$, task-asymmetric noise regimes, depth-wise scheduling sweeps, and high task density $K=10$), supporting the strength and parsimony of the proposed constant-inertia EMA.
