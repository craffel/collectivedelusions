# Paper Summary: Prior-Driven Classical Routing for Dynamic Model Merging

## 1. Core Objectives and Context
The paper addresses the paradigm of **test-time dynamic model merging**, where specialized, task-specific expert neural networks are merged on the fly at inference time using a lightweight routing network. Dynamic routing aims to adapt model parameters sample-by-sample, activating relevant expert pathways based on individual input characteristics.

However, the authors expose a critical, previously unrecognized vulnerability in standard dynamic merging evaluation protocols, which they term the **Batch-Average Smoothing Confounder** and **Vectorization Collapse**:
- **Batch-Average Smoothing Confounder**: Standard evaluation of dynamic routers on heterogeneous (mixed-task) batches of size $B > 1$ typically averages the predicted merging coefficients across the batch. This averaging acts as an implicit smoothing operator that masks severe router overfitting.
- **Vectorization Collapse**: When deployed in real-time, low-latency, batch-independent, or sample-wise vectorized pipelines (where the batch size is $B=1$), this smoothing mask is removed. Consequently, unregularized dynamic routers suffer catastrophic performance degradation (e.g., dropping to 41.09% accuracy on the synthetic sandbox, nearly 17% below a naive static uniform merging baseline).

To address these vulnerabilities, the paper proposes a mathematically simpler, highly stable, and robust ensembling framework called **Prior-Driven Classical Routing**.

---

## 2. Methodology
To isolate the core mechanics of parameter-space routing from confounding visual pre-training variables, the authors evaluate their methods on a controlled, 192-dimensional synthetic **Analytical Coordinate Sandbox** across 10 independent random seeds. They complement this with a real-world MNIST and FashionMNIST expert merging validation.

The methodology consists of several key elements:
1. **Low-Dimensional Unit-State Feature Projection**: High-dimensional latent representations $z(x)_b \in \mathbb{R}^D$ are projected to a compact $d$-dimensional subspace ($d = K \ll D$) using a static, frozen normalized random projection matrix $P$ (satisfying the Johnson-Lindenstrauss lemma). The projected state is normalized onto a unit sphere to obtain $\psi(x)_b$.
2. **Layer-wise Classical Routing**: Standard Softmax activation is applied over linear projections to compute sample-wise ensembling coefficients $\alpha_{k, b}(l)$. This avoids the non-monotonic, non-convex optimization landscapes introduced by complex quantum-inspired wave-interference activations (e.g., QWS-Merge).
3. **Task-Variance Regularization ($\mathcal{L}_{VR}$)**: An explicit group-level loss penalty that minimizes the intra-task variance of predicted coefficients across homogeneous sub-groups within a heterogeneous batch, acting as a soft-clustering constraint.
4. **Sequential Smoothness Regularization ($\mathcal{L}_{\text{smooth}}$)**: A penalty designed to minimize rapid layer-to-layer ensembling weight fluctuations across adjacent layers, suppressing representation misalignment ("routing jitter") in deep multi-layer sequential architectures.
5. **Prior-Driven Zero-Initialization**: Trainable routing weights and biases are initialized to exact zeros. This acts as a maximum-entropy uniform prior (yielding equal coefficients $1/K$). Under $L_2$ weight decay and brief training epochs, routing parameters are held close to this uniform starting state, preventing overfitting.
6. **Vectorized Parameter Assembly**: True sample-specific weight assembly during calibration and deployment (using PyTorch `torch.vmap` or optimized `einsum` operations) to ensure that each input is processed by its own dynamically assembled parameters, bypassing batch-averaging bottlenecks.
7. **Systems-Level Scaling via Dynamic LoRA**: Restricting sample-wise parameter ensembling to low-rank adapter (LoRA) matrices, which avoids the $O(B \cdot M)$ memory expansion and high latency overhead of full-parameter dynamic assembly.

---

## 3. Key Findings and Results
- **Catastrophic Vectorization Collapse**: Unregularized routers (such as random-initialized L3-Softmax) drop to **41.09% $\pm$ 3.73%** joint accuracy at $B=1$ (compared to 58.00% $\pm$ 1.13% for Uniform Merging), showing that their high heterogeneous $B=256$ performance (59.35%) was purely an artifact of batch-averaging.
- **Robustness of Prior-Driven Routing**: Zero-initialized Softmax routing with weight decay (`L3_Softmax_WellReg`) and its variance-regularized variant (`VR_Router`) completely resolve Vectorization Collapse. Both maintain stable, robust joint accuracies of **59.16% $\pm$ 1.17%** and **59.14% $\pm$ 1.18%** respectively across all batch sizes ($B=1$ to $B=512$).
- **The Dynamic Routing Paradox**: To avoid Vectorization Collapse under extreme data scarcity (64 calibration samples), the dynamic router must be so heavily regularized (by the zero-initialized prior) that its predicted coefficients stay in a tight neighborhood of the uniform baseline (MAD of only 0.0236 from 0.25). This heavy constraint limits its functional flexibility, yielding a marginal +1.16% gain over static Uniform Merging, which stands in contrast to the systems-level overhead of dynamic assembly.
- **Breaking the Paradox**: The paradox is a direct consequence of data scarcity. When the calibration budget is scaled to 1024 samples, the Prior-Driven Dynamic router successfully breaks the paradox, yielding a significant **+4.28%** accuracy boost over Uniform Merging.
- **Equivalence of Baselines**: `L3_Softmax_WellReg` and `VR_Router` perform statistically identically across all seeds, sensitivity sweeps, and stress tests. This reveals that the explicit task-variance regularization ($\mathcal{L}_{VR}$) loss is empirically redundant once the zero-initialized, maximum-entropy Softmax prior is established.
- **Dynamic LoRA Effectiveness**: Restricting assembly to LoRA adapters with rank $r \ge 10$ (the algebraic rank of the expert classifiers) completely recovers the full-parameter baseline accuracy (59.26% vs 59.39%) while eliminating VRAM expansion and reducing latency overhead to a mere $1.01\times$.
- **Real-World Expert Merging**: Merging MNIST and FashionMNIST experts on a shared CNN backbone using `L3_Softmax_WellReg` beats static Uniform Merging (82.40% vs 80.30% at $B=1$), demonstrating the transferability of the paper's insights to physical deep learning models.
