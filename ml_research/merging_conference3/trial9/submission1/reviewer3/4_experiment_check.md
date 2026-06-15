# 4. Experimental Check and Evaluation

## Critique of the Experimental Setup & Datasets
From a highly critical, adversarial perspective, the empirical evaluation in this paper has several major weaknesses, relying heavily on artificial setups and showing minimal practical benefits in real-world scenarios:

### 1. Primary Reliance on a Stylized Simulation (Analytical Coordinate Sandbox)
The core quantitative results of the paper (Tables 1 and 2) are conducted entirely within a **simulated 14-layer, 192-dimensional Analytical Coordinate Sandbox (ICS)**. 
- **The Critic's View:** A simulated sandbox is highly stylized and artificial. The authors design the representation propagation, manifold contraction, and noise levels themselves. This makes it trivial to engineer the simulation to perfectly match their assumptions (e.g., that noise propagates in a specific Markovian way across depth, or that representations reside in distinct linear principal subspaces). 
- **The Implication:** Success in a toy simulation is a very weak empirical foundation. It fails to demonstrate whether the method can handle the complex, chaotic, and non-stationary representations found in real, large-scale deep learning models (e.g., modern LLMs or ViTs on real-world datasets).

### 2. Zero Accuracy Improvement in Real-World Validation
The authors present a "Real-World Validation on Pre-trained Vision Transformer (ViT-B/16)" in Section 4.4 to show the practical applicability of PAC-STM. 
- **The Critic's View:** Looking closely at Table 4, under a heterogeneous serving stream on MNIST and CIFAR-10, SABLE (PCA), Temp-Only ERM, and PAC-STM all achieve the **exact same classification accuracy of $86.25\%$**.
- **The Implication:** The proposed PAC-STM does not show *any* accuracy improvement over the unregularized ERM or SABLE baselines on real-world representations. The only metric where PAC-STM outperforms ERM is "trajectory smoothness" (0.1095 vs. 0.2754). However, "smoothness" is an auxiliary optimization metric, not a primary performance goal. If a highly complex learning-theoretic framework does not yield any improvement in actual accuracy or generalization on real data, its practical utility in real-world deployment is highly questionable.

### 3. Oversimplified, Binary Toy Task Benchmark
The real-world validation uses a simple ensembling of $K=2$ tasks: MNIST (handwritten digits) and CIFAR-10 (natural images).
- **The Critic's View:** MNIST and CIFAR-10 are extremely simple toy datasets for a pre-trained Vision Transformer (`ViT-B/16`). Moreover, a binary expert setup ($K=2$) is trivial. Realistic PEFT multi-task serving frameworks (such as S-LoRA or Punica) are designed to route across dozens or hundreds of expert adapters ($K \gg 10$) with complex task relationships. 
- **The Implication:** Restricting the real-world evaluation to $K=2$ toy tasks is a major limitation. It fails to test the routing head under realistic, high-dimensional multi-task serving loads, leaving the scalability and robustness of PAC-STM unverified in practice.

### 4. Modest and Fragile Gains in the Sandbox
Even within the highly stylized coordinate sandbox simulation, the performance gains are very modest:
- Under orthogonal manifolds (Table 1), PAC-STM achieves $73.62\%$ accuracy, which is only a **$2.05\%$** improvement over Temp-Only ERM ($71.57\%$).
- Under overlapping manifolds (Table 2), PAC-STM achieves $72.15\%$ accuracy, which is only a **$2.10\%$** improvement over Temp-Only ERM ($70.05\%$).
- **The Critic's View:** In a highly controlled simulation where the model assumptions are perfectly aligned, a mere 2% improvement is remarkably small. This suggests that the practical benefit of depth-wise trajectory regularization is marginal and likely to be washed out by real-world feature noise.

### 5. Practical Failure of the "Contrastive Projection Head" (UN-CPH-SEP)
To address the high serving-time latency of Kernel PCA (UN-KPCA-SEP), the authors propose a parameterized Contrastive Projection Head (UN-CPH-SEP) as a low-latency alternative.
- **The Critic's View:** In Table 5, while the contrastive projection head is indeed $22.24\times$ faster than Kernel PCA, its task routing accuracy is only **$45.98\%$**, which is barely better than standard linear PCA ($45.35\%$) and significantly worse than Kernel PCA ($51.98\%$). 
- **The Implication:** The fast alternative (UN-CPH-SEP) loses almost all of the accuracy benefits ($+6.63\%$) that Kernel PCA achieves. This reveals a major practical trade-off: to handle non-linear representations, one must either accept high serving-time latency (Kernel PCA) or settle for poor routing accuracy (Contrastive Projection), undermining the authors' claims of a seamless, high-performance solution.
