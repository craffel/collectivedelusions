# Comprehensive Paper Summary

## Main Topic and Motivation
This submission addresses the critical deployment bottlenecks of **batch-dependency** and **heterogeneity collapse** in dynamic weight-space model merging. Model merging is a powerful paradigm to combine multiple task-specific expert networks into a single multi-task model without training or fine-tuning from scratch. While static methods (e.g., Task Arithmetic, TIES-Merging, DARE) suffer from task interference due to parameter conflicts, dynamic merging approaches (e.g., AdaMerging, QWS-Merge, Linear Routers) dynamically adjust merging coefficients based on input features. 

However, existing dynamic merging methods reconstruct a single set of dense weights by averaging routing coefficients across the batch dimension. This introduces a critical limitation:
1. **I.I.D. Violation and Prediction Shifting:** A sample's prediction varies based on what other samples are evaluated with it in the same batch, preventing deterministic, predictable deployment.
2. **Heterogeneity Collapse:** Under shuffled, mixed-task streams, batch averaging averages opposing expert routing signals, producing a flat uniform weight state that degrades performance toward the baseline of uniform static merging.
3. **Computational and Memory Overhead:** Reconstructing large dense matrices at runtime is costly for edge devices.

## Proposed Approach: SLD-Merge
To resolve these limitations, the paper introduces **Sparse Low-Rank Dynamic Merging (SLD-Merge)**. SLD-Merge shifts dynamic routing from weight-reconstruction space to sample-wise activation-space routing using lightweight low-rank adapters. It comprises the following components:

1. **Offline Task Vector SVD Factorization:** 
   The difference between the specialized expert weights and pre-trained base weights ($V_k^{(l)} = W_k^{(l)} - W_{\text{base}}^{(l)}$) is decomposed offline via Singular Value Decomposition (SVD):
   $$V_k^{(l)} \approx B_k^{(l)} A_k^{(l)}$$
   where $B_k^{(l)}$ and $A_k^{(l)}$ are truncated to a low rank $r \ll \min(D_{\text{out}}, D_{\text{in}})$. This reduces task-specific parameter storage by over 92.5%.

2. **Bounded Cosine-Similarity Router:** 
   Input activations $X$ are spatially pooled to create a sample representation $z(x)_b$. A bounded cosine-similarity score is computed against task routing basis vectors $\Phi_k^{(l)}$ representing each expert's signature feature space. This formulation maps representations onto a bounded spherical cosine space to regularize and suppress high-frequency activation noise.

3. **Top-1 Sparse Gating & Vectorized Parallel Forward Pass:** 
   Instead of using a soft dense combination of experts, a Top-1 sparse gating (hard gating) selects only the most relevant expert task adapter per sample. The forward pass is executed via:
   $$Y = X W_{\text{base}}^{(l)} + \sum_{k=1}^K \alpha_k \odot ((X A_k^{(l)}) B_k^{(l)})$$
   Since the gating coefficient $\alpha_{k, b}$ is computed and applied sample-by-sample, the execution is mathematically batch-independent.

4. **Activation-Space Mean Initialization:** 
   The non-differentiability of $\arg\max$ is bypassed by initializing the basis vectors $\Phi_k^{(l)}$ as the centroid (mean activation) representing each task on a tiny unlabeled calibration split (e.g., 128 samples per task). This provides highly effective zero-shot routing without backpropagation.

5. **Autonomous Classification Head Selection:** 
   To avoid using an oracle or ground-truth labels at inference, the average routing score across specialized late blocks is used to autonomously select the appropriate classification head for each sample.

## Key Findings and Claims
- **Batch-Independence:** SLD-Merge completely eliminates batch dependency and heterogeneity collapse. In shuffled mixed-task streams, joint accuracy remains a stable **63.87%** across all batch sizes ($B \in \{1, 4, 16, 64, 256\}$), outperforming dynamic baselines by up to **+8.50%** under large-batch deployment.
- **Task-Wise Performance:** Evaluated on a 4-dataset Vision Transformer benchmark (MNIST, FashionMNIST, CIFAR-10, SVHN) using a `vit_tiny_patch16_224` backbone. SLD-Merge recovers **93.0%** of the specialized standalone expert ceiling average (63.87% vs. 68.66%).
- **Low-Rank Regularization Benefit:** At rank $r=16$, SLD-Merge achieves **66.50%** joint accuracy, outperforming the "Full-Rank + Top-1 Gating" baseline (65.12%) by **+1.38%**, suggesting that SVD truncation serves as an implicit regularizer that filters out training noise/overfitting artifacts.
- **On-Device Efficiency:** SLD-Merge reduces task-specific storage by over **92.5%** (0.295M vs 3.96M parameters for blocks 9-11 specialization) and adds only **8.3%** computational overhead (FLOPs) during inference, enabling efficient edge deployment.
- **Autonomous Head Selection Robustness:** The autonomous classification head selection achieves **93.26%** domain classification accuracy, yielding a joint accuracy of **62.99%** (recovering 98.6% of the privileged oracle-head baseline performance of 63.87%).
