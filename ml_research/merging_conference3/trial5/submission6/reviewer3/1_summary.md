# 1. Summary of the Paper

## Main Topic
The paper addresses a critical deployment bottleneck in dynamic weight-space model merging, which the authors term **batch-dependency** and **heterogeneity collapse**. Traditional dynamic model merging methods reconstruct a single set of merged weights by averaging routing coefficients across the batch dimension. This causes a sample's classification to shift depending on other samples in the batch (violating the I.I.D. assumption) and degrades performance when processing heterogeneous, mixed-task inference streams. To resolve this, the paper proposes **Sparse Low-Rank Dynamic Merging (SLD-Merge)**, a batch-independent and parameter-efficient dynamic weight merging framework.

## Approach
SLD-Merge operates in three main phases:
1. **Offline Task-Vector SVD Factorization:** It computes the weight-space parameter shifts (task vectors) for specialized experts: $V_k^{(l)} = W_k^{(l)} - W_{base}^{(l)}$. It then performs Singular Value Decomposition (SVD) on these task vectors offline, truncating them to a predefined low rank $r$ (e.g., $r=8$ or $r=16$) to construct lightweight low-rank adapters $B_k^{(l)}$ and $A_k^{(l)}$, such that $V_k^{(l)} \approx B_k^{(l)} A_k^{(l)}$.
2. **Bounded Cosine-Similarity Router:** During the forward pass, the input activations are spatially pooled to form a global representation vector $z(x)_b$. A cosine-similarity score is computed between $z(x)_b$ and task routing basis vectors $\Phi_k^{(l)}$ to strictly bound scores in $[-1, 1]$.
3. **Top-1 Sparse Gating and Parallel Forward Pass:** It applies hard Top-1 gating to activate only the single most relevant low-rank task adapter per sample. It implements a parallel PyTorch forward pass: $Y = X W_{base}^{(l)} + \sum_{k=1}^K \alpha_k \odot \left( (X A_k^{(l)}) B_k^{(l)} \right)$ to process samples batch-independently.
4. **Activation-Space Mean Initialization:** Instead of training the routing basis vectors $\Phi_k^{(l)}$, they are initialized to the empirical mean activation of each task computed over a tiny, unlabeled calibration split.
5. **Autonomous Classification Head Selection:** To avoid relying on an oracle task label at test time, the model dynamically routes the sample to the best-aligned classification head by averaging routing scores across the final blocks.

## Key Findings
1. **Heterogeneity Collapse in Baselines:** Traditional dynamic model merging methods (Linear Router, QWS-Merge) average coefficients across the batch, leading to a performance collapse down to the static uniform merging level (55.37%) as the batch size increases.
2. **Batch-Independence of SLD-Merge:** SLD-Merge maintains a stable, peak joint accuracy of **63.87%** across all batch sizes ($B \in \{1, \dots, 256\}$), completely avoiding heterogeneity collapse and outperforming baselines by up to **+8.50%**.
3. **Fidelity of Low-Rank Representation:** Under a truncation rank of $r=8$, SVD preserves over 92% of the expert parameter shifting signal (reconstruction error $<8.3\%$). Rank-16 SLD-Merge achieves **66.50%** accuracy, which actually outperforms the full-rank baseline by **+1.38%**, which the authors attribute to SVD low-rank truncation acting as an implicit regularizer.
4. **Routing Stability:** Layer-wise routers achieve perfect agreement (selecting the same expert across layers) on **96.48%** of the evaluation samples.
5. **Hardware Efficiency:** Hardware profiling on a Raspberry Pi 4 shows that SLD-Merge reduces average latency by **85.2%** (185ms vs. 1250ms) and lowers peak RAM utilization by **42.1%** compared to dense weight-reconstruction methods.

## Claimed Contributions
1. Characterization and empirical exposure of the "heterogeneity collapse" and batch-dependency of existing dynamic merging methods.
2. The SLD-Merge framework, combining offline SVD task-vector decomposition, bounded cosine-similarity routing, and sample-level hard Top-1 gating.
3. The Activation-Space Mean Initialization technique for high-quality, zero-shot routing without gradient descent calibration.
4. Empirical validation on a 4-dataset Vision Transformer benchmark (MNIST, FashionMNIST, CIFAR-10, SVHN) demonstrating stable peak joint accuracy, storage reduction of over **92.5%**, and minor computational overhead (**+8.3%** FLOPs).
