# Intermediate Review File 1: Summary of the Revised Paper

## 1. Overview of the Submission
The submission, titled **"Sparse Task Arithmetic (STA): Deconstructing the Redundancy of Sign-Resolution in Model Merging"**, presents a deconstructive critique of training-free sparse model merging techniques (such as **TIES-Merging** and **DARE**). The authors challenge the prevailing research trend of introducing complex, multi-stage heuristics (such as coordinate-wise sign voting, sign consensus enforcement, and stochastic drop-and-rescale) to resolve parameter interference in weight space.

Guided by Occam's razor, the authors propose a minimalist baseline called **Sparse Task Arithmetic (STA)**. STA consists of two basic steps:
1. **Layer-wise Magnitude Pruning:** Pruning each task vector to retain only the top-$s$\% largest absolute updates in each layer.
2. **Linear Addition:** Directly summing the sparse updates and adding them to the pre-trained base model, without any sign voting, sign consensus, or disjoint merging.

To address the **under-scaling confounder** (where zeroing out updates at low survival densities reduces update energy and degrades performance), the authors propose two scale-preserving variants:
- **Rescaled STA (R-STA):** Surviving updates are divided by the survival density $s/100$, scale-preserving them exactly.
- **Tuned STA:** The scaling coefficient $\lambda$ is dynamically adjusted (e.g., to $\lambda = 0.8$ at $s = 20\%$) to match the optimal weight space energy level.

The paper evaluates these variants on a 4-task vision-transformer multi-task suite (MNIST, FashionMNIST, CIFAR-10, SVHN) using a pre-trained ViT-B-32 backbone.

## 2. Status of Revisions and Key Claim Updates
Following a previous review cycle that highlighted a severe hyperparameter tuning confounder, the authors have revised their paper to include a **fully symmetric hyperparameter sweep** across all methods. The revised paper now presents a highly rigorous evaluation:
- **Un-tuned configurations ($\lambda = 0.3$, $s = 20\%$):** Standard STA (82.91%) lags behind TIES-Merging (85.02%).
- **Symmetrically Tuned Peak Performance (Table 1):**
  - **Tuned Task Arithmetic (s=100%):** Peaks at $\lambda^* = 0.5$ with **88.64%** average accuracy.
  - **Tuned DARE (s=20%):** Peaks at $\lambda^* = 0.4$ with **88.95%** average accuracy.
  - **Tuned TIES-Merging (s=20%):** Peaks at $\lambda^* = 0.5$ with **90.16%** average accuracy.
  - **Tuned STA (Ours, s=20%):** Peaks at $\lambda^* = 0.8$ with **90.53%** average accuracy.
- **Key Claims:** 
  1. Once update energy is balanced via symmetric scaling, Tuned STA outperforms Tuned TIES-Merging by **+0.37%** and Tuned DARE by **+1.58%** without requiring any sign consensus.
  2. Sparsity (e.g., $s \le 20\%$) naturally reduces the coordinate-wise mask overlap across tasks to extremely low levels (3.1%–4.3%), matching the theoretical independence bound of 4%, making sign voting mathematically moot for over 96% of the parameters.
  3. Where conflicts do overlap, direct addition allows the dominant signal to naturally suppress the weaker opposite sign, and TIES's hard zeroing-out of conflicting weights is structurally harmful (e.g., on SVHN).
  4. Magnitude pruning is conceptualized as an isotropic noise filter that removes high-frequency gradient noise accumulated during fine-tuning, rather than resolving sign conflicts.

## 3. Structural and Formatting Review
The paper is well-structured and conforms to standard conference template guidelines. It includes an Abstract, Introduction, Related Work, Methodology (Section 3), Experimental Evaluation (Section 4), and Conclusion (Section 5). It includes professional figures (Figure 1) and tables (Table 1) that are integrated cleanly into the narrative.
