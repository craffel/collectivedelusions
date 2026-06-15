# Intermediate Evaluation: 1. Summary of the Paper

## Main Topic and Approach
This paper addresses the problem of **spatial weight-space interference** in weight-space model merging. When task-specific expert models, which are fine-tuned on conflicting or orthogonal domains from a shared pre-trained initialization, are merged using standard linear interpolation (Task Arithmetic) or sign-consistent averaging (TIES-Merging), they experience representation collapse due to signal erasure. 

To mitigate this, the authors propose **Exclusive Parameter Merging (EPM)**, a training-free model fusion operator that operates at the individual coordinate level. EPM is built on two core techniques:
1. **Soft Exclusive Parameter Allocation (Soft-EPA):** Routes each parameter coordinate primarily to the "dominant" expert model (defined as the model with the largest absolute standardized update magnitude at that coordinate) while retaining a fraction $\gamma = 0.2$ of the standard linear blend of other experts as a "structural glue." To level the playing field between tasks with large gradient norms (complex color datasets) and simple tasks (grayscale digits), the authors introduce **Task Vector Standardization** (both Global and Layer-wise variants).
2. **Task-Level Coefficient Tuning (TLC-Tune):** A gradient-free optimization method that tunes only $K$ global scaling factors (one per expert) using a (1+1) Evolution Strategy on a small offline validation set (128 samples per task). TLC-Tune optimizes a non-differentiable validation minimax accuracy score to raise the worst-performing task's floor.

Under parameter-constrained sparse merging regimes, the authors introduce **Dynamic Coherence Scheduling (DCS)**, which dynamically scales the coherence retention factor $\gamma$ as a quadratic function of target sparsity $p$.

---

## Key Findings and Empirical Results
- **Mitigation of Collapses:** EPM successfully prevents the representational collapse seen in Task Arithmetic and TIES-Merging under severe task conflicts. 
- **Empirical Outperformance:** EPM (TLC-Tune) achieves up to **46.19%** joint mean test accuracy under dense merging on a ViT-Tiny backbone across four conflicting datasets (MNIST, FashionMNIST, CIFAR-10, SVHN), outperforming Task Arithmetic (40.96%) and TIES-Merging (20.55%).
- **Generalization and Stability:** The authors evaluate TLC-Tune against high-dimensional optimization baselines (AdaMerging with 56 parameters, ZipMerge with 70 parameters). They show that under a zero-order search, these high-dimensional baselines suffer from absolute optimization failure and fail to converge, whereas TLC-Tune's 4-dimensional search space converges within 40 steps and generalizes flawlessly to the test set.
- **Exclusivity vs. Cooperation Trade-off:** Sensitivity analysis of the coherence factor $\gamma$ shows that pure exclusivity ($\gamma=0.0$) degrades performance on complex tasks, whereas a small background blend ($\gamma=0.2$) restores activation coherence and t-SNE clustering, raising joint accuracies.

---

## Explicitly Claimed Contributions (with Evidence)
1. **The EPM Routing Operator:** Introduces a training-free coordinate routing protocol based on standardized absolute magnitude. *Evidence:* PyTorch implementation description and quantitative results in Tables 1-3.
2. **Task Vector Standardization:** Mitigates the "Rich Task" Dominance Trap, leveling the scale differences between simple and complex tasks. *Evidence:* MNIST accuracy rises from ~12% to ~48% in Table 1; coordinate-level analysis shows SVHN overrides MNIST at 297k parameters.
3. **Task-Level Coefficient Tuning (TLC-Tune):** Bypasses the "Overfitting-Optimizer Paradox" by optimizing only $K$ global scaling factors using a zero-order (1+1) ES. *Evidence:* Systematic optimization budget and validation size sweeps (Tables 5 and 6) showing TLC-Tune converges and generalizes stably.
4. **Dynamic Coherence Scheduling (DCS):** Dynamically adjusts $\gamma$ to resolve the capacity starvation of non-dominant experts under high sparsity. *Evidence:* Lift in 80% sparse joint mean accuracy from 24.11% to 26.41% (Table 3).
5. **Deconstruction of Baseline Failures:** Empirically dissects the performance degradation of AdaMerging and ZipMerge under zero-order search as an optimization failure (under-convergence) rather than pure transductive overfitting. *Evidence:* Optimization sweep trajectories up to 500 steps (Figure 2, Table 5) showing flat validation and test curves.
