# Experimental and Empirical Validation Evaluation

## 1. Quality of Empirical Evidence and Baselines
The empirical evaluation is highly comprehensive in its breadth of ablation studies, sweeps, and baselines:
* **Baselines Considered:** Includes a broad set of baselines spanning direct Euclidean addition (Task Arithmetic), advanced Euclidean pruning/dropping (TIES-Merging, DARE), test-time adaptive merging (AdaMerging), and Euclidean spectral balancing (SAIM), in addition to the manifold-level base model (OrthoMerge).
* **Architecture Generalization:** Evaluates on both a Multi-Layer Perceptron (MLP) and a custom, attention-based Vision Transformer (ViT) architecture, as well as the Split-CIFAR-10 classification benchmark.
* **Sweeps and Ablations:** Includes:
  * Multi-task scaling ($N=5$ experts) to verify representation energy decay of Task Arithmetic vs. exact preservation of RIMO.
  * Block size sensitivity sweeps ($b \in \{32, 64, 128, 256\}$).
  * Latency comparisons between sequential CPU solvers (SVD, Schur) and the proposed GPU-compatible Complex Hermitian solver on an NVIDIA H100 GPU.
  * Statistical significance checks across 3 independent random initializations.
  * Evaluation of unsupervised test-time adaptive merging (AdaMerging) under disjoint multi-task setups.

---

## 2. Key Empirical Findings and Insights
* **The Tangent Space Spectral Pitfall** is exceptionally well-supported empirically. RIMO with $t > 1.0$ (spectral balancing) consistently collapses to random guess level ($13.66\%$ on MLP and $18.44\%$ on ViT), proving the catastrophic impact of inflating smaller singular values under the Cayley map.
* **RIMO-Pruned** (Rank-Preserving Spectral Pruning) is shown to be a highly effective and robust alternative, achieving $90.47\%$ accuracy in standard training and $91.49\%$ in orthogonally regularized training, outperforming standard OrthoMerge ($84.55\%$) and matching Euclidean baselines.
* **AdaMerging's Vulnerability** is clearly demonstrated: it overfits to a single task calibration batch in disjoint setups, collapsing accuracy on inactive tasks to $0.00\%$ ($47.61\%$ average). This is a strong, high-signal negative result for test-time optimization.
* **The GPU-compatible Complex Hermitian Solver** is shown to be extremely fast, executing in just $7.66$ ms on an NVIDIA H100 GPU ($12.2\times$ faster than Schur, $8.1\times$ faster than SVD), proving that the execution speed bottleneck of geometric model merging is practically solvable.

---

## 3. Major Weaknesses and Critical Practical Limitations

Despite the extensive experiments, we identify three critical empirical weaknesses:

### Weakness A: Toy-Scale Evaluation (Split-MNIST and Tiny Custom Architectures)
While the paper purports to extend its results to modern attention-based architectures and more complex datasets, a close inspection of the codebase reveals that the experimental setup is extremely small-scale:
1. **Low-Complexity Datasets:** The main evaluations are conducted on **Split-MNIST**, a toy dataset with $28 \times 28$ resolution. For the Split-CIFAR-10 evaluation, the codebase uses a simple MLP rather than a standard convolutional or transformer backbone, and only retains a **20% subset of the dataset** to keep CPU training fast.
2. **Custom "Toy" Vision Transformer:** The Vision Transformer (ViT) utilized in Section 4.5 is not a standard foundation model (e.g., ViT-B/16). It is an extremely small, custom toy model with an embedding dimension of **32**, a depth of only **1**, and **2** attention heads, trained on $28 \times 28$ grayscale Split-MNIST images.
* **Criticism:** It remains unproven whether the proposed RIMO-Pruned or the GPU Complex Hermitian solver scales or generalizes to real-world foundation models (e.g., LLaMA, RoBERTa, or standard ViTs) on high-complexity, diverse tasks.

### Weakness B: Persistent Performance Gap with Task Arithmetic
In the orthogonally regularized regime (Table 2), simple Euclidean Task Arithmetic achieves **$94.00\%$ average accuracy**, and SAIM achieves **$93.07\%$**.
* **Criticism:** Meanwhile, the proposed RIMO-Pruned ($t=1.0$) only achieves **$91.49\%$** accuracy. While RIMO-Pruned represents a major boost over standard OrthoMerge ($84.55\%$), it still underperforms the simple, flat Euclidean Task Arithmetic baseline by $2.51\%$.
* **Impact on Paper:** In practice, a practitioner would still prefer simple Task Arithmetic over RIMO-Pruned due to this persistent performance gap, unless the number of experts $N$ is very large (where they show Task Arithmetic decays). This reduces the practical utility of the proposed method in standard, low-$N$ scenarios.

### Weakness C: Underperformance and Practical Difficulty of Hard Orthogonal Constraints
The authors conduct a pilot experiment using hard orthogonal constraints during training (projected SGD on the Stiefel manifold) to eliminate the residual component ($\|\rho_k\|_F = 0$) and prevent coordinate warp.
* **Criticism:** Individual experts achieve high task accuracies ($96.01\%$ and $93.15\%$), but OrthoMerge on these experts achieves only **$72.08\%$ average accuracy**.
* **Impact on Paper:** While $72.08\%$ is an improvement over soft-regularized training without residuals ($7.11\%$), it is still a massive drop (approx. $22\%$) compared to the soft-regularized Task Arithmetic ($94.00\%$) and the individual expert performance. This indicates that hard-constrained optimization on the Stiefel manifold severely restricts the models' representational capacity or is extremely prone to getting trapped in bad local minima. This major performance drop casts doubt on whether hard orthogonal constraints are truly a viable future direction for practical model merging.

## 4. Overall Rating
**Experiment Rating: Good**
The experimental sweep is exceptionally detailed, featuring deep ablations, statistical robustness checks, and high-performance GPU latency benchmarks. However, the reliance on extremely small-scale, toy-scale architectures and datasets, combined with the persistent performance gap with flat Euclidean Task Arithmetic, prevents an "Excellent" rating.
