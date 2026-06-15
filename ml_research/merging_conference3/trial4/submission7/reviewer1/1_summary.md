# Evaluation Phase 1: Paper Summary

## Main Topic and Problem Addressed
This paper critically audits the current paradigm of adaptive weight-space model merging, focusing specifically on Test-Time Adaptation (TTA) methods such as AdaMerging and PolyMerge. These methods dynamically adjust layer-wise merging coefficients at test time over unlabeled target streams to optimize multi-task performance without training costs. The authors expose a severe, previously unreported evaluation confounder termed **"Task Suite Bias"** where current evaluation protocols rely on a single, arbitrary combination of four datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) that masks fundamental limitations of unconstrained online adaptation.

To address this, the paper introduces **SuiteMerge**, a systematic methodological audit that partitions the task pool into five distinct multi-task evaluation suites along axes of domain distance and representational conflict. It also proposes **Offline Few-Shot Validation Tuning (OFS-Tune)** using continuous low-degree polynomial trajectories as a robust, regularized, and computation-free alternative for practical deployment.

---

## Proposed Approach
The authors investigate the following key methodologies:
1. **SuiteMerge Framework:** Decomposing the standard evaluation set into five suites:
   - *Suite A (Highly Homogeneous - Low Conflict):* MNIST + FashionMNIST
   - *Suite B (Highly Heterogeneous - High Conflict):* CIFAR-10 + SVHN
   - *Suite C (Cross-Domain Digits):* MNIST + SVHN
   - *Suite D (Cross-Domain Objects):* FashionMNIST + CIFAR-10
   - *Suite E (Full 4-Task Suite - Control):* MNIST + FashionMNIST + CIFAR-10 + SVHN
2. **Offline Few-Shot Validation Tuning (OFS-Tune):** Leveraging a small labeled validation set ($M=10$ samples per task) to optimize stable layer-wise trajectories offline prior to deployment. The trajectories are constrained to low-degree continuous polynomial curves (specifically linear $d=1$ and quadratic $d=2$) to act as a noise-rejection filter. Optimization is performed using the Nelder-Mead derivative-free local search algorithm.
3. **Alternative Parameterizations:** Expanding the trajectory framework to include *Piecewise Linear Splines* (with knots at block boundaries) and *Block-wise Parameter Sharing* (grouping MHA and MLP blocks) to capture localized non-smooth sensitivity spikes without overfitting.
4. **Physical Weight-Space Validation:** Evaluated on a 5-layer CNN on CPU using MNIST and FashionMNIST under two initializations: (Regime A) Scratch-trained experts in disjoint loss basins, and (Regime B) Fine-tuned experts in a shared pre-trained basin, examining the impact of "Unsupervised TTA" versus "Privileged TTA" (oracle task routing).

---

## Key Findings and Empirical Evidence
- **Task Suite Bias Confirmed:** The relative ranking of model-merging methods is highly sensitive to the chosen task suite. In the highly homogeneous Suite A, Uniform merging is extremely competitive. In the high-conflict Suite B, unconstrained online TTA (AdaMerging) overfits to local stream noise, lagging behind OFS-Tune in simulation and collapsing catastrophically below the Uniform baseline in physical weight-space deployments.
- **Transductive Overfitting of Online TTA:** Unconstrained online TTA (AdaMerging) over-parameterizes on local stream noise. Restricting coefficients to continuous polynomial trajectories (PolyMerge and OFS-Tune) serves as an essential regularizer. In Suite B, unconstrained AdaMerging averages $62.58\% \pm 5.71\%$, whereas polynomial-constrained OFS-Tune ($d=2$) achieves $68.62\% \pm 2.45\%$.
- **OFS-Tune Performance and Efficiency:** OFS-Tune ($d=2$) matches or exceeds PolyMerge across all suites in simulation, and outclasses both online PolyMerge and AdaMerging in physical validation (Regime B) by up to $3.70\%$ and $4.20\%$, respectively, while requiring **zero test-time compute, zero backpropagation latency, and zero privileged task-routing assumptions** at deployment.
- **Ablation baseline (OFS-Unconstrained):** Isolating the polynomial constraint from the validation data access shows that unconstrained offline tuning overfits to small-sample validation noise ($\epsilon_{\text{val}}$), achieving $60.42\% \pm 4.98\%$ in Suite B, compared to the $68.62\% \pm 2.45\%$ of quadratic OFS-Tune ($d=2$).
- **Physical Validation Insights:**
  - Standard linear merging fundamentally requires a pre-trained shared initialization (Regime B). Scratch-trained experts (Regime A) collapse to random guessing ($\sim 12.20\%$ accuracy).
  - Online TTA suffers from a "privilege trap": in unsupervised mixed streams without oracle labels, it collapses due to joint entropy minimization across heads. OFS-Tune naturally bypasses this by keeping parameters static.
  - Adding temporal Exponential Moving Averages (EMA) to online TTA improves cooperative performance but still lags behind the static Uniform baseline ($82.20\%$) and OFS-Tune ($83.00\%$), while retaining test-time compute overhead.

---

## Explicitly Claimed Contributions
1. **Exposing Task Suite Bias:** Uncovering how a single monolithic multi-task suite serves as a dangerous confounding variable in the model-merging literature.
2. **Formulation of Transductive Overfitting:** Mathematically defining and empirically demonstrating how unconstrained online TTA overfits to correlated stream-level statistics.
3. **Establishing OFS-Tune as an Analytical Filter:** Proving that low-degree continuous polynomial trajectory constraints act as robust offline low-pass regularizers, offering superior generalizability across task relations with zero test-time overhead.
4. **Providing Multi-Seed Comparative Benchmark:** Running a robust 30-seed simulation benchmark alongside concrete physical weight-space validation to call for a transition toward multi-suite validation.
5. **LLM Scaling Roadmap:** Outlining concrete, actionable strategies (representative subsets, first-order OFS-Adam, CPU offloading) to scale OFS-Tune to billion-parameter models.
