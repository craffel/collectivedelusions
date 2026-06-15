# Experimental Verification and Critical Check

## Critical Evaluation of the Experimental Setup
1. **The Synthetic Simulator:**
   * *Dimensions and Scale:* The simulator models a 14-layer network with a dimension of $D=192$ and $K=4$ tasks. This is a highly controlled, synthetic, closed-loop setup.
   * *Generalization Gap Modeling:* The simulator's biggest flaw is modeling the test-time generalization gap analytically using the formula $\text{Gap}_k = \eta_{\text{noise}} \|W_k\|_2 \|V_k\|_F$. This directly matches the objective minimized by SR3, making the comparison on this simulator inherently biased/circular.
   * *Accuracy Decay Function:* Accuracies are mapped from parameter-space distances using an exponential decay function. While standard for simulation, this is a highly idealized assumption of how physical neural networks degrade under weight changes.
2. **The Physical PyTorch MLP:**
   * *Dataset:* Uses the small, toy `load_digits` dataset from scikit-learn (1797 samples of size $8\times 8$).
   * *Model:* A shallow 2-layer MLP (TinyMLP).
   * *Scale:* This setup is exceptionally small. While it breaks the analytical circularity, its toy scale means that random initialization seeds play a massive role, as acknowledged by the authors. This limits the strength of any empirical conclusions drawn.

## Analysis of Baselines
The paper includes a robust set of baselines:
* Static Uniform Merging
* Unregularized Linear Router
* Isotropic $L_2$ Weight Decay
* TSAR (Centroid Anchoring)
* VR-Router (Variance Minimization)
* PFSR (Parameter-Free Subspace Routing)

This represents a fair and comprehensive comparison. However, the performance differences between these baselines and the proposed methods are critical to analyze.

## Do the Results Support the Claims?

### Claim 1: SR3 family achieves highly competitive joint accuracy comparable to state-of-the-art heuristics on the simulator.
* **Result:** In the simulated results (Table 1), the highest performing model is actually the heuristic **TSAR at 79.90%**. Isotropic $L_2$ weight decay achieves **79.71%**, which is practically identical to the best proposed variant, **SR3-S at 79.72%** (a difference of only 0.01%) and **SR3-S-L1-Sched at 79.71%** (a difference of 0.00%).
* **Critique:** The empirical gains over a standard, uniform $L_2$ regularizer are virtually zero on the simulator. This raises a major question: *Is it worth introducing complex learning-theory derivations, asymmetric precomputed multipliers, smoothed L1 formulas, and warm-up schedules if the end performance is identical to standard, uniform $L_2$ weight decay?* The results do not support the necessity of this added complexity.

### Claim 2: SR3 variants are superior or highly stable in physical networks.
* **Result:** In the physical MLP experiment (Table 2, 10-seed average):
  * Isotropic $L_2$ Reg: **$92.13\% \pm 2.47\%$**
  * TSAR: **$92.13\% \pm 2.92\%$**
  * SR3-F (Frobenius): **$90.50\% \pm 1.36\%$**
  * SR3-S (Spectral): **$90.93\% \pm 1.94\%$**
  * SR3-H (Hybrid): **$91.20\% \pm 1.81\%$**
* **Critique:** Standard $L_2$ decay and TSAR **outperform all proposed SR3 variants on average by substantial margins (up to 1.63%)**. While SR3-F has a lower standard deviation, its absolute accuracy is significantly degraded. The hybrid controller (SR3-H) improves the average to 91.20% but still fails to reach the performance of standard $L_2$ decay. This directly refutes the claim of empirical superiority in physical settings, proving that when the generalization gap is real (natural data classification) rather than synthetic (Rademacher penalty), uniform regularization is superior.

### Claim 3: The Spectral variant (SR3-S) is tighter and outperforms the Frobenius variant (SR3-F).
* **Result:** On the simulator, SR3-S (79.72%) marginally outperforms SR3-F (79.61%) by $0.11\%$. On the physical experiment (10-seed average), SR3-S ($90.93\%$) outperforms SR3-F ($90.50\%$) by $0.43\%$.
* **Critique:** While the direction of the claim holds, the margins are tiny and lie well within the standard deviations of the runs ($\pm 1.36\% - \pm 1.94\%$). Thus, the claimed superiority of Spectral over Frobenius is statistically weak and heavily overshadowed by the fact that both are worse than standard $L_2$ weight decay.

### Claim 4: The Hybrid Controller resolves the specialization-generalization tension (recovering SVHN accuracy).
* **Result:** On the simulator, the high-norm expert (SVHN, $v_k = 8.0$) accuracy is:
  * VR-Router: **66.24%**
  * SR3-S (Spectral): **62.24%**
  * SR3-S-Hybrid: **62.34%**
* **Critique:** The hybrid controller's recovery is negligible ($+0.10\%$), leaving it far below VR-Router's performance on the complex expert (66.24%). Minimizing Rademacher complexity asymmetrically causes severe under-activation of complex experts. The hybrid controller fails to resolve this "Double-Edged Sword" of asymmetric regularization in a meaningful way, demonstrating the fundamental limitation of the theoretical framework.
