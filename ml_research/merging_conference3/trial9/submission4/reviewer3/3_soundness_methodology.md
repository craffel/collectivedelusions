# Intermediate Evaluation 3: Soundness & Methodology

## Clarity of Description
The methodology is exceptionally well-structured and described with rigorous mathematical clarity. The problem formulation, unit-norm anchored similarity routing, layer-wise centroids, and Momentum-Merge dynamics are detailed in explicit equations. The boundary conditions are clearly laid out, and the deconstruction proof in Theorem 1 is presented in a step-by-step fashion. 

## Appropriateness of Methods
- **Exponential Moving Average (EMA):** Using a constant-parameter EMA across network depth is highly appropriate for smoothing routing trajectories. It represents a classical, low-pass filter that effectively suppresses high-frequency noise while preserving the low-frequency task-specific signal.
- **Unit-Norm Calibration (UNC):** Cosine similarity is a standard and effective metric for measuring representational alignment, bypassing magnitude expansion.
- **Layer-wise Centroid Anchoring:** Calibrating task centroids layer-by-layer ($\mu_k^{(l)}$) is mathematically sound and essential for handling representational rotation and coordinate drift across depth.

## Potential Technical Flaws & Deeper Mathematical Analysis

As a theory-minded reviewer, we perform a deep mathematical analysis of the paper's core claims and proofs. We identify several critical nuances, hidden assumptions, and potential limitations:

### 1. The Normalization/Simplex Constraint in Theorem 1's Proof
Theorem 1 states that under uniform activation energy ($E_{a,k} = E_a$) and constant temperature, the discretized continuous-time ODE of ChemMerge is mathematically equivalent to the constant EMA of Momentum-Merge.

In the proof, the authors write:
> "For the concentration vector $C^{(l)}$ to remain on the probability simplex (preserving $\sum_k C_k^{(l)} = 1$ given $\sum_k w_k^{(l)} = 1$ and $\sum_k C_k^{(l-1)} = 1$), the coefficients must satisfy conservation of mass. This physical constraint forces: $\kappa \Delta t = k_{\text{decay}} \Delta t = \gamma$, where $\gamma \in [0, 1]$ is a positive scaling factor."

This implies that the physical constraint $\kappa = k_{\text{decay}}$ is mathematically required to satisfy the simplex constraint. 

**Our Independent Derivation:**
However, if we do not assume $\kappa = k_{\text{decay}}$ a priori, and instead perform a step-wise simplex projection (normalization) on the discretized concentration vector $C_k^{(l)}$ (which any probability-conserving system must do):
$$C_k^{(l)} = (\kappa \Delta t) w_k^{(l)} + (1 - k_{\text{decay}} \Delta t) \alpha_k^{(l-1)}$$
Summing over $k$ (where $\sum_k w_k^{(l)} = 1$ and $\sum_k \alpha_k^{(l-1)} = 1$):
$$\sum_k C_k^{(l)} = \kappa \Delta t + 1 - k_{\text{decay}} \Delta t = Z$$
If we normalize to obtain the ensembling weights $\alpha_k^{(l)} = C_k^{(l)} / Z$:
$$\alpha_k^{(l)} = \frac{\kappa \Delta t}{Z} w_k^{(l)} + \frac{1 - k_{\text{decay}} \Delta t}{Z} \alpha_k^{(l-1)}$$
Letting $\gamma = \frac{\kappa \Delta t}{Z}$. Then:
$$1 - \gamma = 1 - \frac{\kappa \Delta t}{\kappa \Delta t + 1 - k_{\text{decay}} \Delta t} = \frac{1 - k_{\text{decay}} \Delta t}{Z}$$
Substituting these back yields:
$$\alpha_k^{(l)} = \gamma w_k^{(l)} + (1 - \gamma) \alpha_k^{(l-1)}$$
This is *exactly* equivalent to a constant-inertia EMA with momentum parameter $\beta = 1 - \gamma$, where $\gamma$ is a constant because $\kappa$, $k_{\text{decay}}$, and $\Delta t$ are all constants!

**Theoretical Insight:**
This is an incredibly profound theoretical result: **step-wise normalization automatically guarantees mathematical equivalence to a constant EMA for *any* values of $\kappa$ and $k_{\text{decay}}$, without requiring the artificial and physically strained assumption that $\kappa = k_{\text{decay}}$!** 

In physical chemistry, there is no thermodynamic reason why the rate of species creation ($\kappa$) must be identical to its rate of degradation ($k_{\text{decay}}$). The authors correctly point out this strain in the metaphor, but our derivation proves that the metaphor is even more redundant than they claim: step-wise normalization completely bypasses the need for the thermodynamic rate-matching assumption while preserving the exact EMA recurrence.

### 2. Recurrence Trapping under Scarce Calibration Data
The authors introduce **Raw Boundary Initialization** (Eq. 7):
$$\alpha_k^{(L_{\text{frozen}})} = w_k^{(L_{\text{frozen}}+1)}$$
While this is shown to eliminate transient startup jitter and reduce routing jitter by up to 70.1$\times$, our analysis of Appendix E exposes a significant theoretical vulnerability.

Under small calibration subset sizes ($|\mathcal{C}_k| \le 16$), the computed centroids are highly noisy. This noise is directly transferred to the initial ensembling weight $w_k^{(L_{\text{frozen}}+1)}$, meaning the recurrence is initialized with a highly inaccurate starting state. 
Because Momentum-Merge possesses stateful momentum (low-pass filter memory), this initial boundary error propagates through the network depth, trapping the ensembling coefficients in highly sub-optimal states throughout the forward pass. This **Recurrence Trapping** collapses joint accuracy to **71.20%** when $|\mathcal{C}_k| = 8$ (a **4.80% absolute degradation** compared to the stateless SABLE + Layer Centroids which achieves **76.00%**).

This is a critical theoretical limitation: **stateful smoothing via momentum introduces a trade-off where the system becomes highly vulnerable to initialization errors (recurrence trapping) when the quality of representation anchors is low.** In contrast, stateless systems evaluate representations independently at each layer, making errors localized and allowing the network to recover in subsequent layers.

## Reproducibility
The paper is highly reproducible. The authors provide the exact mathematical equations, detailed hyperparameters ($\beta = 0.60$, $\tau = 0.005$ for Advanced, etc.), a complete listing of physical parameters in the Analytical Coordinate Sandbox (ICS) environment, and report results over 10 independent random seeds with statistical significance tests. The inclusion of extensive sensitivity sweeps over the temperature parameter $\tau$, the calibration subset size $|\mathcal{C}_k|$, and the task-asymmetric noise scales ensures high transparency.
