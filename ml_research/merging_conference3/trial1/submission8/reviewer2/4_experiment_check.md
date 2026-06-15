# 4. Experimental Check & Empirical Evaluation

## Evaluation of the Experimental Setup & Datasets
The authors evaluate their methods on:
1. **Split-MNIST (using a 3-layer MLP, $d=256$):** A standard disjoint multi-task setting (Task 1: digits 0--4, Task 2: digits 5--9).
2. **Split-MNIST (using a custom Vision Transformer, ViT):** Validates the methods on attention-based architectures.
3. **Split-CIFAR-10 (using a 3-layer MLP, $d=256$):** A harder classification benchmark with 3-channel spatial inputs (3072 dimensions) to evaluate generalizability (Appendix M).
4. **Multi-Task Scaling ($N=5$ experts on Split-MNIST):** Evaluates how the merging methods handle increasing numbers of tasks (Task 1: 0--1, Task 2: 2--3, Task 3: 4--5, Task 4: 6--7, Task 5: 8--9).

### Strengths of the Experimental Setup
* **Dual Training Regimes:** Evaluating under both Standard (non-OFT) and Orthogonal Regularization ($\lambda_{ortho} = 2.0$) regimes is crucial for validating the role of the orthogonality condition.
* **Extensive Baselines:** The paper compares against standard flat Euclidean methods (**Task Arithmetic (TA)**), advanced sparsification techniques (**TIES-Merging**, **DARE**), test-time adaptive optimization (**AdaMerging**), Euclidean isotropic balancing (**SAIM**), and geometric manifold merging (**OrthoMerge**).
* **Multiple Solvers Evaluated:** Comparing SVD, real Schur decomposition, and complex Hermitian solvers across accuracy and execution latency benchmarks provides a comprehensive systems-level view.
* **Multi-Seed Robustness Check:** Sweeping three random seeds ($\text{seed} \in \{42, 100, 2026\}$) ensures that the observed trends are not random optimization artifacts.

---

## Do the Results Support the Claims?
Yes, the empirical results strongly support the paper's core claims:
1. **The Orthogonality Condition:** Moving from standard training to orthogonal regularization doubles the performance of OrthoMerge and RIMO ($t=1.0$) from $42.07\%$ to $84.55\%$. Additionally, the naive post-hoc SVD projection experiment shows a catastrophic collapse to $15.00\%$, proving that native manifold-respecting models are required.
2. **The Tangent Space Spectral Pitfall:** SVD-based spectral balancing (RIMO with $t > 1.0$) collapses performance to $13.66\%$ on MLP and $18.44\%$ on ViT, empirical proof of the destructive noise propagation under SVD-based coordinate inflation.
3. **Bypassing the Pitfall via RIMO-Pruned:** In both MLP and ViT environments, RIMO-Pruned ($\text{keep}=0.2$) successfully avoids the collapse, recovering high accuracies of $90.47\%$ (Standard MLP), $91.49\%$ (Orthogonal MLP), and $88.16\%$ (ViT), outperforming standard OrthoMerge.
4. **Schur and Complex Solver Equivalences:** The real Schur and Complex Hermitian solvers yield identical accuracies to SVD, proving that the spectral pitfall is a fundamental mathematical property of Cayley maps, while the Complex solver achieves a $12.2\times$ speedup, resolving practical latency concerns.
5. **Representational Decay of Task Arithmetic:** Figure 4 empirically validates that as $N$ scales to 5, Task Arithmetic suffers from representational magnitude decay ($O(1/\sqrt{N})$), while RIMO-Pruned preserves the manifold energy ($O(1)$).

---

## Constructive Scholarly Critique & Weaknesses

### 1. Low Complexity of Experimental Benchmarks
While the authors extended their results to Split-CIFAR-10 and a Vision Transformer (ViT) in the appendix to address scale and generalizability, the datasets (MNIST and CIFAR-10) and models (3-layer MLPs, custom small ViT) remain small by modern standards.
* **Critique:** The paper argues in Appendix C and I that the spectral balancing pitfall worsens quadratically with representation size $d$, making it highly relevant to large models like LLaMA-7B ($d=4096$) combined with LoRA ($k=8$). However, there are no empirical evaluations on large pre-trained foundation models (such as RoBERTa-large, CLIP-ViT-B/16, or LLaMA-7B) fine-tuned on standard benchmarks (e.g., GLUE, VTAB, or instruction tuning). Conducting a small-scale PEFT/LoRA model-merging experiment in these settings would turn this from a highly theoretical toy study into a high-impact paper for the broader ML community.

### 2. Absolute Performance Gap with Task Arithmetic
Under both standard and orthogonal regularization training, flat Euclidean averaging (Task Arithmetic, SAIM, TIES-Merging, DARE) consistently outperforms manifold-level merging (OrthoMerge, RIMO).
* **Critique:** Under orthogonal regularization, Task Arithmetic achieves **94.00%**, while OrthoMerge achieves **84.55%** (a $9.45\%$ gap). RIMO-Pruned reduces this gap but still achieves only **91.49%** (a $2.51\%$ gap). Even under hard orthogonal constraints (where residual norms are exactly zero), OrthoMerge achieves only **72.08%** compared to Task Arithmetic's **94.00%**. 
* The authors honestly discuss this gap, attributing it to non-convex loss barriers and path divergence on the curved manifold. However, this is a major practical limitation: geometric model merging is mathematically elegant but yields inferior absolute performance to simple Euclidean averaging in standard settings. This performance gap must be discussed more prominently in the main text as a major limitation, rather than being confined to the discussion.

### 3. Missing Quantitative Details for Multi-Task Scaling ($N=5$)
Figure 4 displays the empirical multi-task scaling curves for $N=5$ experts, illustrating representational magnitude preservation.
* **Critique:** While the curves visually show Task Arithmetic's magnitude decay, there is no corresponding quantitative table in the appendix showing the individual task accuracies and average multi-task accuracies for $N=5$ merged experts across the various methods (TA, SAIM, OrthoMerge, RIMO-Pruned). Providing these numbers would strengthen the claim that geometric model merging is superior in high-$N$ settings.
