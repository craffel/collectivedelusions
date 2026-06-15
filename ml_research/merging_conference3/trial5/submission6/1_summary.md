# 1. Summary of the Paper

## 1.1. Context and Problem Statement
Weight-space model merging has emerged as a promising, training-free paradigm to consolidate multiple task-specific expert neural networks into a single multi-task model. This eliminates the linear scaling cost of storing and maintaining independent models for every new task. However, existing dynamic model-merging methods (such as QWS-Merge and Linear Routers) suffer from a critical deployment bottleneck: **batch-dependency** and **heterogeneity collapse**.

Traditional dynamic routers compute input-dependent weight-merging coefficients for a given batch and average them across the batch dimension to reconstruct a single, merged dense weight matrix. This design results in three major failure modes:
1. **Violation of the I.I.D. Assumption:** Reconstructing weights with batch-averaged coefficients causes a sample's classification to shift dynamically depending on other co-packaged samples in the same batch, which is unacceptable for reliable and safe deployment.
2. **Heterogeneity Collapse:** Under highly mixed, heterogeneous streaming inputs, averaging opposing task-expert routing coefficients produces an intermediate, uniform-like weight state. This state performs poorly on all tasks, collapsing the model's accuracy back to the static merging baseline.
3. **Compute and Memory Bottlenecks:** On-the-fly reconstruction of massive, dense weight matrices during inference introduces severe memory bandwidth and latency bottlenecks, making it completely unviable for resource-constrained edge hardware.

## 1.2. Proposed Solution: SLD-Merge
To resolve these limitations, the paper introduces **Sparse Low-Rank Dynamic Merging (SLD-Merge)**, a parameter-efficient, completely batch-independent dynamic weight-merging framework designed for robust on-device deployment. 

SLD-Merge shifts the dynamic adaptation from heavy parameter weight-reconstruction to lightweight, sample-wise activation-space routing. It operates in three main phases:
1. **Offline SVD Task-Vector Decomposition:** For each specialized expert, the parameter shift relative to the base model (the task vector $V_k^{(l)}$) is computed and factorized offline using Singular Value Decomposition (SVD). Truncation to a low rank $r \ll D$ (e.g., $r=8$) yields compact matrices $B_k^{(l)}$ and $A_k^{(l)}$, reducing the additional task-specific parameter footprint by over **92.5%**.
2. **Bounded Cosine-Similarity Router:** During the forward pass, input activations are spatially averaged to form a global sample representation $z(x)_b$. The cosine similarity is computed between $z(x)_b$ and a set of task routing basis vectors $\Phi_k^{(l)}$. Mapping features to a bounded $[-1, 1]$ spherical cosine space suppresses high-frequency activation noise and acts as a strong regularizer.
3. **Top-1 Sparse Gating and Parallel Forward Pass:** Instead of executing a soft combination, SLD-Merge applies hard Top-1 expert selection to route each sample completely independently through only its selected low-rank expert adapter. Vectorized PyTorch implementation ensures complete mathematical batch-independence and zero cross-sample leakage.

The paper also introduces:
- **Activation-Space Mean Initialization:** An elegant zero-shot calibration technique that sets the routing basis vectors $\Phi_k^{(l)}$ to the empirical mean activations of each task on a tiny unlabeled validation set (e.g., 128 samples per task), avoiding unstable, complex optimization.
- **Autonomous Classification Head Selection:** Eliminates the privileged "oracle" head selection of prior work. By averaging the routing scores across final blocks, the model autonomously routes representations to the predicted classification head at inference time.

## 1.3. Key Results and Achievements
The paper evaluates the proposed framework on a 4-dataset Vision Transformer benchmark (MNIST, FashionMNIST, CIFAR-10, SVHN) using a `vit_tiny_patch16_224` backbone.
- Under heterogeneous, shuffled mixed-task streams, SLD-Merge maintains a perfectly stable, peak joint accuracy of **63.87%** (or **64.16%** with optimized basis) across all batch sizes ($B \in \{1, 4, 16, 64, 256\}$), completely eliminating heterogeneity collapse.
- It outperforms Uniform/Task Arithmetic by **+8.50%** and Linear Router/QWS-Merge by up to **+6.84%** under large-batch deployment.
- SVD rank sensitivity shows a predictable, monotonic trade-off: rank $r=8$ preserves over 92% of the expert parameter shift signal and recovers 93.0% of standalone expert capability.
- Zero-shot Activation-Space Mean Initialization performs within 0.3% of a fully optimized basis (63.87% vs. 64.16%).
- Autonomous head selection achieves **62.99%** accuracy, recovering **98.6%** of the privileged oracle-head performance.
- On-device profiling on a Raspberry Pi 4 demonstrates an **85.2% latency reduction** and a **42.1% RAM reduction** compared to dense weight-reconstruction baselines.
