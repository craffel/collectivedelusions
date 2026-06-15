# Intermediate Review Step 1: Paper Summary

## Main Topic and Approach
The paper introduces **ChemMerge**, a training-free, continuous-time framework for dynamic, activation-space model ensembling on resource-constrained edge hardware. The core objective is to resolve the trade-off between ensembling accuracy and representation stability (routing weight jitter across layers) when serving multi-task models under highly heterogeneous, noisy, and rapidly switching input streams.

Rather than treating sequential network layers as decoupled, independent execution blocks—as is standard in existing post-hoc dynamic ensembling methods like SABLE and SPS-ZCA—ChemMerge models the activation flow through the network's depth as a continuous multi-component chemical reactor. The proposed method consists of three key components:
1. **Catalytic Zero-Shot Alignment (C-ZCA):** Pre-computes task-specific representation centroids from early, adapter-free shared layers to act as "catalytic enzymes."
2. **Non-Equilibrium Kinetic Routing (NEKR):** Maintains a continuous, sample-wise expert concentration state vector $C_b^{(l)} \in [0, 1]^K$ across successive layers. The dynamics are governed by a system of first-order linear differential equations, where the forward reaction rates are determined by a temperature-scaled, Softmax-normalized Arrhenius rate equation, and the backward rate determines representation decay.
3. **Catalytic Activation Blending (CAB):** Blends expert output activations dynamically in a single parallel forward pass using mass-action ensembling weights proportional to the normalized active concentrations.

The paper provides two operational modes: *Single-Centroid Mode* (anchoring to early layers for memory and parameter savings) and *Multi-Centroid Mode* (layer-specific centroids for highly non-linear foundation models). It also explores an *Active Representation Coupling* mechanism to gently warp intermediate representations toward active centroids.

---

## Key Findings
1. **Performance in Simulated Sandbox (ICS):** Inside a 14-layer, 192-dimensional Analytical Coordinate Sandbox (ICS), ChemMerge achieves a Joint Mean accuracy of **78.11%** (homogeneous) and **78.06%** (heterogeneous) across 10 random seeds. This recovers **98.81%** of the theoretical Expert Oracle ceiling (79.00%), outperforming stateless nearest-centroid routing (SPS-ZCA) by up to **+8.22%** and static uniform merging by **+17.41%**, while remaining statistically comparable to SABLE.
2. **Robustness to Streaming Collapses:** Unlike parametric routers which collapse catastrophically under heterogeneous serving (e.g., QWS-Merge dropping to 34.58%), ChemMerge maintains flat, robust performance across batch sizes from $B=256$ down to $B=1$, demonstrating complete immunity to both *Heterogeneity Collapse* and *Vectorization Collapse*.
3. **Profound Jitter Suppression:** In routing-only simulations on a pre-trained Vision Transformer ($\text{ViT-B/16}$) over a randomized PIL-generated shape stream, ChemMerge reduces layer-to-layer ensembling weight routing jitter by **9.9$\times$** compared to stateless nearest-centroid routing and by over **2.15$\times$** compared to SABLE (under identical sensitivities).
4. **Discretization and Solver Stability:** The exact analytical *Exponential Integrator* derived in the paper ensures absolute numerical stability without requiring heuristic projection clipping, maintaining stable accuracy even under extremely large step sizes ($\Delta t = 10.0$).
5. **Efficiency and Scaling:** Vectorized implementations of ChemMerge execute a 16-expert routing update in only 19.9ms, which is **42.1%** faster than SABLE and **49.4%** faster than SPS-ZCA, demonstrating excellent computational scalability with expert density.

---

## Explicitly Claimed Contributions and Evidence
- **Contribution 1: A training-free continuous-time chemical kinetics paradigm for dynamic model ensembling.**
  - *Evidence:* Formal mathematical formulation of first-order reversible kinetics (Eq. 7), Arrhenius rate equations (Eq. 5), and the Law of Mass Action for activation blending (Eq. 11, 12).
- **Contribution 2: Non-Equilibrium Kinetic Routing (NEKR) to resolve layer-to-layer routing weight jitter.**
  - *Evidence:* Quantitative routing-only simulation on pre-trained $\text{ViT-B/16}$ showing a 9.9$\times$ jitter reduction over SPS-ZCA and 2.15$\times$ over SABLE (Table 3), accompanied by trajectory visualizations (Figure 10).
- **Contribution 3: Complete immunity to Heterogeneity Collapse and Vectorization Collapse under constant $O(1)$ edge latency.**
  - *Evidence:* Empirical evaluations inside the ICS Sandbox (Table 1) and batch size sweeps (Figure 1b) showing flat accuracy curves across all batch sizes, compared to the severe performance degradation of static merging (heterogeneity) and parametric routing (vectorization).
- **Contribution 4: Derivation and evaluation of an exact analytical Exponential Integrator.**
  - *Evidence:* Section 3.3 (Eq. 9) derives the integration scheme exactly, and Section 4 (Figure 9) provides an empirical ablation confirming its absolute numerical stability up to $\Delta t = 10.0$.
- **Contribution 5: Highly hardware-friendly and efficient vectorized scaling.**
  - *Evidence:* CPU-bound NumPy benchmarks in Section 4.5.1 showing that ChemMerge executes 16-expert routing updates in 19.9ms, outperforming stateless SABLE and SPS-ZCA which rely on interpreter-bound sample-by-sample loops.
