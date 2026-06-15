# 4. Experimental Setup and Empirical Evaluation Check

## Evaluation of the Experimental Setup & Datasets
- **Toy Benchmarks and Toy Models:** The primary experimental environment is **Split-MNIST**, a highly simplified, low-dimensional toy benchmark. The authors utilize a 3-layer Multi-Layer Perceptron (MLP) with a hidden dimension of $d=256$. 
- **Toy "Vision Transformer" (ViT):** Although the authors claim to generalize their findings to attention-based architectures, their "Vision Transformer" is an extremely down-scaled, custom toy model: input patches are projected into sequence of size 16, embedding dimension $d=32$, 2 attention heads, and FFN expansion dimension 32. This model contains fewer than 50K parameters, which is a toy model, and it's also trained on Split-MNIST.
- **Split-CIFAR-10:** Mentioned in Appendix J, but with very little architectural detail, and the performance reported ($28.50\%$ accuracy) is extremely poor for CIFAR-10, suggesting a very weak underlying model or severe optimization issues.
- **Critical Limitation:** Modern deep learning operates at the scale of billions of parameters (e.g., LLaMA, ViT-B/16) and on complex multi-task or multi-domain datasets. Evaluating geometric model merging solely on Split-MNIST/Split-CIFAR-10 using tiny MLPs and toy ViTs is a major limitation. The claims of "quadratic scaling of the pitfall to LLM scales" are based purely on theoretical derivations, without any actual empirical validation on standard modern models.

---

## Critical Evaluation of the Baselines
- The authors include a good range of standard Euclidean model merging baselines: **Task Arithmetic (TA)**, **DARE**, **TIES-Merging**, and **SAIM**.
- However, their comparison with **AdaMerging** is a strawman evaluation:
  - AdaMerging is an unsupervised test-time adaptive method that optimizes coefficients based on unlabeled data. 
  - The authors evaluate AdaMerging under a disjoint multi-task setup where they feed it a test calibration batch belonging entirely to a single task (e.g., Task 1's inputs). Unsurprisingly, AdaMerging converges to a coefficient of $\sim 0.95$ for Task 1 and $\sim 0.00$ for Task 2, collapsing Task 2 accuracy to $0.00\%$.
  - This "catastrophic overfitting" is a direct consequence of violating AdaMerging's fundamental assumption of a representative calibration batch. Evaluating an adaptive method on highly biased data to claim "unsupervised entropy minimization is highly prone to task-specific overfitting" is methodologically unfair.

---

## Do the Results Actually Support the Claims?

### 1. Claim: RIMO-Pruned is a Robust, High-Performance Merging Method
- **No.** The empirical results in Tables 1 and 2 show that under *both* training regimes, the proposed **RIMO-Pruned** method is consistently and significantly outperformed by standard, flat Euclidean **Task Arithmetic (TA)**:
  - **Standard Training:** TA achieves **$91.11\%$** vs. RIMO-Pruned's **$90.47\%$**.
  - **Orthogonal Training:** TA achieves **$94.00\%$** vs. RIMO-Pruned's **$91.49\%$** (a $2.51\%$ gap).
  - Furthermore, in the orthogonal regime, advanced flat-space baselines like **DARE ($93.89\%$)** and **TIES-Merging ($93.34\%$)** also easily outperform RIMO-Pruned.
  - This indicates that RIMO-Pruned is not a practical or high-performance merging method. Rather, it is a computationally expensive, highly restricted alternative that fails to match the performance of simple linear averaging.

### 2. Claim: SVD Projection and Coordinate Gauges are the Primary Causes of the Pitfall
- **No.** As discussed in Soundness, the authors' own experiments with **RIMO-Schur-Balanced** and **RIMO-Complex-Balanced** completely disprove this claim. Because the projection-free, gauge-consistent Schur-Balanced method also collapses to $\sim 12\%$ accuracy, the "SVD projection distortion" (Theorems 3.2 and 3.3) is empirically shown to be irrelevant to the model collapse. The collapse is entirely driven by the non-linear coordinate inflation under the Cayley map, which is common to both Schur and SVD.

### 3. Claim: Complex Hermitian Solver Achieves Spectacular Acceleration (8.1x over SVD)
- **Partially Supported, but Unfair Comparison:**
  - The authors report a latency of **$7.66$ ms** for the Complex Hermitian solver compared to **$62.37$ ms** for sequential SVD and **$93.34$ ms** for sequential Schur.
  - However, looking closely at the hardware details: sequential SVD and Schur were benchmarked on a **single-threaded CPU** (Intel(R) Xeon(R) Platinum @ 2.90GHz), while the Complex Hermitian solver was benchmarked on a massively parallel, high-end **NVIDIA H100 Tensor Core GPU** using batched PyTorch operations.
  - Comparing a sequential, single-threaded CPU implementation to a massively parallelized GPU accelerator is highly misleading and mathematically unfair. To support a claim of "spectacular acceleration" due to the algorithm, the authors must compare parallel SVD on GPU with parallel Complex Hermitian eigensolver on GPU under identical hardware and batching conditions.
