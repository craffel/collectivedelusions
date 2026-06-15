# 4. Experimental Evaluation and Results Check

## Evaluation Setup
* **Backbone:** `vit_tiny_patch16_224` (5.7M parameters, 12 layers). This is an appropriate, highly lightweight backbone for edge-device profiling.
* **Datasets:** MNIST, FashionMNIST, CIFAR-10, and SVHN (10 classes each). These represent a diverse set of visual domains (digits, clothing, natural objects, street numbers).
* **Baselines:** Robust and highly representative, covering:
  1. Static Uniform Merging
  2. Task Arithmetic (with optimized scale factor $\lambda = 0.3$)
  3. Linear Router (standard dynamic baseline)
  4. QWS-Merge (state-of-the-art quantum-inspired dynamic router)

## Key Results and Claims Validation
1. **Batch-Independence and Shuffled Stream Evaluation:**
   Table 1 evaluates joint test accuracy under a shuffled mixed-task stream as batch size $B$ sweeps from 1 to 256. 
   * **The Claim:** SLD-Merge is batch-independent and avoids performance collapse.
   * **The Evidence:** SLD-Merge maintains a perfectly flat, peak joint accuracy of **63.87%** across all batch sizes. In contrast, the Linear Router (59.28% at $B=1$) and QWS-Merge (56.93% at $B=1$) show flat performance curves that approximate the Uniform Merging baseline (55.37%) as the batch size increases.
   * **The Analysis:** The authors provide an honest explanation of this "soft collapse" (rather than a drop to random 10% guessing), attributing it to the pre-trained ViT backbone serving as a capacity buffer. This is highly credible and represents a high level of academic integrity.
2. **Task-wise Performance (Table 2):**
   * SLD-Merge achieves the most balanced performance, setting SOTA on FashionMNIST (**76.17%**) and CIFAR-10 (**77.34%**).
   * On SVHN, previous dynamic methods like the Linear Router collapse to **16.80%** (below the static baseline of **23.44%**). SLD-Merge robustly routes SVHN to **26.56%** (close to the standalone ceiling of **29.30%**), proving the regularizing power of the bounded cosine router under domain shift.
   * Overall, SLD-Merge preserves **93.0%** of the standalone expert average (**68.66%**) while running merged.
3. **SVD Rank Sensitivity (Table 3):**
   * Shows a highly predictable and monotonic recovery of performance as target rank $r$ scales ($r=4 \rightarrow 58.98\%$, $r=8 \rightarrow 63.87\%$, $r=16 \rightarrow 66.50\%$).
   * This validates the SVD formulation and enables system developers to make predictable trade-offs between hardware memory budgets and accuracy.
4. **Zero-Shot Initialization vs. Optimized Router:**
   * Proves that the zero-shot Activation-Space Mean Initialization (**63.87%**) is incredibly effective, recovering **99.5%** of the performance of a Straight-Through Estimator (STE) optimized router (**64.16%**). This is a massive win for real-world deployment, as it eliminates expensive backpropagation training during calibration.
5. **SVD Regularization Effect (Full-Rank Baseline):**
   * The authors compare rank-16 SLD-Merge (**66.50%**) against a Full-Rank + Top-1 routing baseline (**65.12%**). 
   * **The Claim:** SVD low-rank truncation acts as a heavy implicit regularizer in data-scarce regimes.
   * **The Evidence:** SVD rank-16 actually *outperforms* the full-rank baseline by **+1.38%** by discarding noisy, overfitted parameter dimensions from the low-resource experts. This is a profound and highly useful scholarly insight.
6. **Autonomous vs. Oracle Head Selection:**
   * The autonomous selection rule achieves **62.99%** accuracy, recovering **98.6%** of the privileged oracle-head baseline (**63.87%**). This validates the router's domain classification accuracy (**93.26%**) and proves the feasibility of fully autonomous deployment.

## Potential Gaps in the Empirical Evaluation
* **Dataset Scale:** 
   The evaluation is limited to subsampled datasets (256 train, 128 validation, 256 test samples per domain). While the authors justify this as a low-shot streaming stress-test representing rapid edge specialization, it is important to verify if the proposed methods and their SVD regularization effects scale to standard, full-scale datasets (e.g., full ImageNet, full CIFAR-100, or NLP benchmarks).
* **Model Scale:**
   The backbone model is ViT-Tiny (5.7M parameters). Modern production systems deploy models that are several orders of magnitude larger (e.g., ViT-Huge, LLaMA-8B, etc.). Demonstrating SLD-Merge on larger models would significantly increase its industry relevance and impact.
