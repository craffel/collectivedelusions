# Summary of the Paper

## Main Topic and Approach
The paper addresses the challenge of deploying multiple task-specific expert models on resource-constrained edge and IoT devices. While weight-space merging via Task Arithmetic enables multi-task capabilities by storing task-specific delta vectors, dense weights are still too large for edge hardware. To resolve this, the paper proposes **Norm-Preserved Budgeted Task-Vector Pruning (NP-BTVP)**, a training-free post-hoc weight sparsification and merging framework.

Specifically, the framework extracts task vectors ($\tau_k = \theta_k - \theta_{\text{base}}$) and sparsifies them using magnitude-based pruning. It evaluates two schemes:
1. **Uniform Pruning (NP-BTVP-U):** Prunes each task vector globally to retain the top $p\%$ largest updates.
2. **Adaptive Saliency-Based Pruning (NP-BTVP-S):** Dynamically allocates parameter budgets to layers based on their average update intensity ($L_1$-norm).

Crucially, the framework incorporates **norm-preserving rescaling** (scaling active updates by $1/p$ or $1/p_l$) to prevent update norm shrinkage and preserve update signal strength. This scaling is conceptually positioned as a "signal-strength boost" that prevents task-specific updates from being drowned out by the base pre-trained model during multi-task fusion.

---

## Key Findings
1. **Flatness and Pruning Resilience:** Training-stage loss landscape flatness (via Sharpness-Aware Minimization, or SAM) does not inherently provide an additional coordinate-aligned pruning buffer under well-converged regimes compared to standard AdamW. Under the proposed rescaling, both AdamW and SAM experts demonstrate nearly identical and high resilience to heavy sparsification (retaining high classification accuracy even at 90-95% sparsity).
2. **Competitive Performance of NP-BTVP-U:** Under a tight 90% sparsity budget ($p=0.10$), Uniform Pruning with rescaling (NP-BTVP-U) achieves 90.32% (SAM) and 90.34% (AdamW) average accuracy across four datasets. This is remarkably close to the fully dense unpruned baselines (91.00% and 90.94%), performs competitively with the advanced stochastic DARE-Merging baseline, and outperforms TIES-Merging by over 3.6% at half the parameter budget.
3. **The Saliency Double-Bind:** Saliency-Based Pruning (NP-BTVP-S) is trapped in a trade-off between severe inter-layer scale imbalance (under global scaling) and local noise amplification (under layer-wise scaling). As a result, the simpler, global Uniform Pruning (NP-BTVP-U) is the more stable and robust option.
4. **Storage Footprint Savings:** Storing the sparse task vectors in compressed formats like Coordinate (COO) or Compressed Sparse Row (CSR) reduces the edge storage footprint by 5$\times$ to 20$\times$, enabling efficient multi-task capabilities on memory-bounded edge hardware.

---

## Explicitly Claimed Contributions and Supporting Evidence
The authors explicitly claim the following contributions, supported by empirical or theoretical evidence in the text:
* **The NP-BTVP Framework:** Supported by the mathematical formulation in Section 3 and implementation details for both Uniform (NP-BTVP-U) and Saliency-Based (NP-BTVP-S) pruning.
* **Rigorous Empirical Evaluation:** Evaluated across 3 independent random seeds on 4 datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) using a pre-trained CLIP ViT-B/32 backbone. The results in Table 2 and Table 3 show that NP-BTVP-U at $p=0.10$ matches unpruned performance within 0.70% and outperforms TIES-Merging by over 3.8%.
* **Separation of Geometry and Coordinate Sparsification:** Supported by the comparative analysis of SAM vs. AdamW experts under pruning (Table 2). The finding that flatness-aware experts do not exhibit superior pruning resilience compared to AdamW under well-converged regimes is a valuable, counter-intuitive geometric insight.
* **Analysis of the Saliency Double-Bind:** Supported by comparing NP-BTVP-S under global scaling and layer-wise scaling (Table 2), showing how layer-wise scaling degrades or destabilizes performance due to noise blowup (especially when combined with INT8 quantization as discussed in Appendix E).
* **Theoretical Derivations of Rescaling:** Derived in Appendix A, demonstrating that under Laplace and Gaussian weight-update distributions, a $1/p$ scaling factor on the top-$p$ elements mathematically amplifies the expected $L_1$ update norm (by 3.30$\times$ and 2.58$\times$ respectively for $p=0.10$), acting as a beneficial signal-strength boost.
* **Storage Footprint Quantification:** Supported by the calculations in Section 4.5, showing that 90% sparse CLIP experts occupy ~22.96 MB in CSR/COO format compared to 114.8 MB for the dense versions.
