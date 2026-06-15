# Soundness and Methodology Evaluation

## 1. Clarity of Description
The paper is exceptionally well-structured and written with high technical clarity. The methodology is explained in detail with mathematical formulas (Eq. 1 to 14), a comprehensive procedural algorithm (Algorithm 1), and structured qualitative audits (Table 1, Table 2). The authors are highly transparent about their systems-level choices and explicitly document their configurations, making the narrative easy to follow.

## 2. Appropriateness of Methods
The proposed methods (PFSR, UNC, Class-Size Scaling Calibration, MBH) are appropriate for the specific co-designed algorithm-systems context they target. The co-design of parameter-free routing and stream-level partitioning addresses the key memory and representation interference bottlenecks. However, several critical limitations and methodological concerns are present when evaluated under a skeptical lens.

## 3. Potential Technical Flaws & Methodological Weaknesses

### A. The "Zero Training/Calibration" Contradiction
The authors claim their framework requires "zero trainable parameters" and "zero calibration split data" (Abstract, Introduction). However, a closer look at their proposed enhancements reveals significant data-dependence and training requirements:
- **Unsupervised Non-Classification Centroids:** For non-classification or generative experts (regression, diffusion), they must introduce a calibration split to fit unsupervised $K$-means centroids (Eq. 11, 12).
- **OOD Density Estimation:** Their primary OOD detection method relies on fitting a Gaussian Mixture Model (GMM) on a calibration split (Section 4.4, Table 9). They admit this "slightly relaxes our strict 'zero calibration data' claim."
- **Representational Drift Mitigations:** To resolve representational drift in fully fine-tuned models, they suggest using a "Lightweight Calibration Projection" (a 1-layer MLP trained on a 64-sample calibration split) or adding "Representation Alignment Objectives" during training (Eq. 10). If experts must be retrained with a customized alignment objective to remain compatible, the method is no longer zero-shot or training-free.

### B. Severe Dependency on the PEFT (LoRA) Paradigm
The spatial viability of dynamic model merging heavily depends on the PEFT (LoRA) assumption. Without LoRA:
- Keeping $K$ full-parameter expert models concurrently in VRAM would be memory-prohibitive ($>70$ GB for LLaMA-7B experts).
- Dynamically loading full weights from host CPU to GPU would incur a massive PCIe transfer latency of over 5,000 ms per forward pass (Table 3).
Thus, the entire framework is only viable under the PEFT/LoRA co-design. For practitioners working with fully fine-tuned, full-parameter expert networks, this method is physically non-viable.

### C. Infrastructure-Serving Complexity Trade-off
The authors claim to apply Occam's razor to simplify model-merging by stripping away trainable routing parameters. However, this is a **complexity shift** rather than a true simplification. They shift the complexity from the model parameters to the data-serving infrastructure. 
On-the-fly stream partitioning, dynamic weight-merging, sequential micro-batch dispatching, and index-based scatter-gather output re-assembly require a highly sophisticated serving layer. Integrating Punica/SGMV kernels for parallel execution requires custom CUDA compilation, specific PyTorch bindings, and dedicated GPU architectures. This represents a substantial systems engineering overhead that may outweigh the training costs of a simple, classical parametric linear router.

### D. Over-Simplified Analytical Proof of Layer-Averaging Collapse
The mathematical proof of Layer-Averaging Collapse (Section 3.6) relies on several highly restrictive and unrealistic assumptions:
- The base network's layer-wise Jacobians $J_b^{(m)}$ act as a sequence of contractive operators that strongly dominate and dampen any layer-dependent semantic variance.
- The intermediate representation manifolds in deep layers stabilize and become approximately collinear across layers ($h_{base, b}^{(l-1)} \approx c_l \bar{h}_{base,b}$).
In actual, highly non-linear deep neural networks (e.g., LLaMA-7B), early layers and late layers are semantically highly distinct (early layers process low-level structural/syntax details, while late layers process high-level task semantics). The assumption of near-collinearity and contractive projection across all layers is a massive over-simplification. Thus, the "mathematical redundancy" of layer-wise routing is only proven for a highly idealized toy model and does not hold rigorously for real deep networks.

### E. Independent Gaussian Assumption for Class-Size Scaling Calibration
The Class-Size Scaling Calibration factor ($\sqrt{2\log C_k / d}$) is derived under the assumption of independent random Gaussian variables in high dimensions. In actual networks, deep representations and classification weights are highly structured, correlated, and far from independent random variables. While the calibration factor shows empirical utility on their sandbox, the theoretical justification relies on a statistical assumption that is violated in practice.

## 4. Reproducibility
The authors provide detailed mathematical formulations, a clear algorithmic block (Algorithm 1), and explicit descriptions of the hyperparameters used (e.g., $\tau = 0.001$, $C_{sub}=256$, $\gamma_{OOD}=0.4$, etc.). The experimental setup is described clearly. However, there is no mention of open-sourcing the code or repository links in the text, which is standard practice for ensuring full reproducibility.
