# Experimental Validation Check

## 1. Experimental Design Quality: Excellent
The experimental design is exceptionally thorough, comprehensive, and statistically rigorous. The authors do not rely on a single selective seed or isolated hyperparameter setting. Instead, they execute:
- A **Statistical Significance Sweep** across 10 independent random seeds.
- A **Regularization Sensitivity Sweep** over 10 values of $\lambda_{var}$.
- An **Inference Stream Heterogeneity Stress Test** across 5 batch sizes $B \in \{1, 8, 32, 128, 512\}$.
- An **Exhaustive Ablation Study** of loss components ($\mathcal{L}_{CE}, \mathcal{L}_{reg}, \mathcal{L}_{VR}$).
- A **Subspace Overlap Sensitivity Sweep** over 7 values of geometric overlap $\rho \in [0.0, 0.90]$.
- A **Projection Subspace Dimension Sweep** over $d \in \{2, 4, 8, 16\}$.
- A **Sequential Smoothness Regularization Sweep** over $\gamma_{\text{smooth}} \in \{0.0, 0.01, 0.1, 1.0, 10.0\}$.
- A **Dynamic LoRA Rank Sweep** over $r \in \{2, 4, 8, 10, 12\}$.
- A **Real-World Model Merging Validation** on MNIST + FashionMNIST experts.

This multi-dimensional audit ensures that their findings are highly reproducible and structurally sound.

---

## 2. Baseline Comprehensiveness: Excellent
The baselines evaluated in this paper are comprehensive and highly appropriate:
1. **Static Uniform Merging**: Serves as a strong, training-free baseline representing a fixed parameter compromise.
2. **Global Linear Router (Unregularized)**: Standard unregularized baseline, which collapses due to extreme overfitting.
3. **QWS-Merge (Quantum-Inspired, SOTA)**: Represents the state-of-the-art complex routing activation.
4. **L3-Linear**: Represents classical layer-wise linear routing (Muqeeth et al., 2023).
5. **L3-Softmax**: Layer-wise routing with random-initialized Softmax.
6. **L3-Softmax (Well-Reg.)**: This baseline is of **exceptional scientific quality**. By training a standard Softmax router under the same zero-initialized and weight-decayed hyperparameters as VR-Router but setting $\mathcal{L}_{VR} = 0$, the authors isolate the precise impact of the architectural prior from the explicit loss penalty.

---

## 3. Support for Claims and Key Findings
The experimental results fully and robustly support all of the authors' primary claims:
- **Vectorization Collapse**: Table 3 and Table 5 clearly document this catastrophic collapse. Standard random-initialized L3-Softmax achieves a strong 59.35% accuracy under $B=256$, but plummets to **41.09%** under vectorized $B=1$ streaming (nearly 17% below Uniform Merging).
- **The Batch-Average Smoothing Confounder**: The stress test results (Table 5) prove that unregularized Softmax has high accuracy under large batch sizes because averaging coefficients over a batch collapses the predicted weights back to the uniform compromise.
- **Efficacy of Zero-Initialization & Well-Reg Baseline**: Both `L3_Softmax_WellReg` and `VR_Router` maintain an exceptionally stable, flatline accuracy of **59.16% $\pm$ 1.17%** and **59.14% $\pm$ 1.18%** respectively across all batch sizes from $B=1$ to $B=512$, completely resolving Vectorization Collapse.
- **Equivalence of Baselines**: The statistical equivalence between `L3_Softmax_WellReg` and `VR_Router` in Table 3, Table 5, and Table 6 clearly proves that explicit task-variance regularization ($\mathcal{L}_{VR}$) is empirically redundant once the zero-initialized Softmax prior is established.
- **Dynamic LoRA Capacity**: Table 10 provides elegant proof that setting the LoRA rank $r \ge 10$ (recovering the algebraic rank of the expert classifiers) completely eliminates reconstruction error, achieving accuracy identical to the full-parameter baseline (59.26% vs 59.39%).

---

## 4. Empirical Gaps and Weaknesses

### 4.1. Real-World Scale Gap (The CLIP/Transformer Boundary)
Although the authors provide a "Real-World Validation" section (Section 4.13), the evaluation is conducted on a very simple MNIST + FashionMNIST classification task using a shared 2-layer CNN backbone with a router of only 56 parameters.
* **The Gap**: Modern parameter-space merging is primarily applied to large-scale, pre-trained transformer foundation models (such as LLMs or Vision-Language models like CLIP/ViT), where representations are processed through multi-billion parameter networks.
* **Reviewer Critique**: While the MNIST+FashionMNIST experiment successfully serves as a small-scale proof of concept, it does not confirm whether Vectorization Collapse and Prior-Driven Classical Routing behave identically at the scale of large transformer-based foundation models. The paper contains a highly detailed CLIP ViT-B/16 protocol and roadmap in Appendix A, but does not present actual empirical results at this scale in the main paper. Running experiments on CLIP or other large foundation models would dramatically strengthen the empirical grounding of the paper.

### 4.2. Circular Nature of the Smoothness Accuracy Evaluation in the Sandbox
In Section 4.12, the authors evaluate the Sequential Smoothness Regularizer ($\mathcal{L}_{\text{smooth}}$) and report in Table 9 that scale variations in $\gamma_{\text{smooth}}$ have "zero degradation in multi-task classification accuracy."
* **Reviewer Critique**: In the synthetic sandbox, the expert classifiers are single linear layers. To apply the predicted layer-wise coefficients to this single-layer setup, the authors average the coefficients over the layer dimension (i.e., $\bar{\alpha}_k = \frac{1}{L} \sum_{l=1}^L \alpha_k(l)$). Because the routing coefficients are average-collapsed over the layer dimension, layer-to-layer routing weight jitter has **no functional impact** on classification accuracy in the sandbox. Therefore, the flatline accuracy in Table 9 is a mathematical consequence of the sandbox's average-collapsing design, and does not empirically prove that sequential smoothness has "zero degradation on accuracy" in a true deep sequential network where layer-wise parameters are applied sequentially without average-collapsing. This limitation should be explicitly highlighted.
