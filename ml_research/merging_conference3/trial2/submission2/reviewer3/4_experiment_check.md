# Intermediate Evaluation 4: Experimental Evaluation

This document provides a critical evaluation of the experimental setup, datasets, baselines, and empirical results, analyzing whether they align with and support the theoretical claims of the paper.

## 1. Experimental Setup & Datasets
The authors evaluate SVS on the complete 86M parameter visual backbone of the **CLIP-ViT-B/32** vision encoder, rather than restricting evaluation to a few layers. Merging all parameters across a deep, multi-layer Vision Transformer is a robust protocol that tests the ability of SVS to handle cascading approximation errors.

The downstream experts are trained on four diverse and widely-used classification datasets:
* **MNIST** (Handwritten digits)
* **FashionMNIST** (Clothing)
* **CIFAR-10** (Natural objects)
* **SVHN** (Street View House Numbers)

The choice of 1,000 test samples per dataset (4,000 total test samples) provides sufficient statistical significance to eliminate evaluation noise and support the empirical comparisons.

---

## 2. Baseline Comparison
The baseline suite is excellent and includes standard, representative, and state-of-the-art training-free offline model merging operators:
1. **Zero-Shot Base CLIP:** Establishes the pre-trained starting point.
2. **Individual Experts:** Represents the empirical performance upper bound.
3. **Task Arithmetic (TA):** The standard linear merging baseline.
4. **TIES-Merging:** State-of-the-art coordinate pruning via magnitude thresholding and sign consensus.
5. **DARE:** State-of-the-art randomized coordinate-basis dropout.

The authors also clearly outline the scientific distinction between data-free offline operators and test-time optimization methods (AdaMerging, FoldMerge), establishing a fair comparison under identical resource and data constraints.

---

## 3. Analysis of Empirical Claims vs. Evidence

### A. SVS Matches or Outperforms Task Arithmetic
* **Claim:** SVS with rank $k=128$ matches or exceeds full-rank Task Arithmetic.
* **Evidence:** SVS ($74.83\%$) strictly outperforms Task Arithmetic ($74.78\%$) on average (Table 1). It matches or outperforms TA on MNIST ($88.3\%$ vs. $87.9\%$) and SVHN ($48.9\%$ vs. $48.8\%$), while being highly competitive on FashionMNIST ($82.5\%$ vs. $82.6\%$) and CIFAR-10 ($79.6\%$ vs. $79.8\%$). This provides convincing proof that the core trajectory of fine-tuning updates resides on a low-rank spectral manifold.

### B. The Redundancy of BWN in Normalized Architectures
* **Claim:** Explicit scale-preservation algorithms are redundant in normalized architectures like CLIP-ViT because feature normalization layers cancel out global weight scaling.
* **Evidence:** Standard SVS and SVS+BWN achieve identical performances ($74.83\%$) across all ranks (Table 1 and Figure 2). This empirical finding strongly supports the scale-invariance argument, even if the residual blocks make it an approximation rather than an exact identity (the computed average scale $\alpha = 0.998$ is so close to 1.0 that any residual block scale shifts are negligible).

### C. Validation of BWN in Un-normalized Environments
* **Claim:** BWN successfully preserves weight/activation scales and boosts performance in un-normalized neural networks where global scaling cancellation does not hold.
* **Evidence:** In a controlled 3-layer MLP experiment (Section 4.5), the authors show that at low scaling regimes ($\lambda=0.1$), the activation norm collapses to $1.37$ without BWN. Applying BWN analytically restores it to $1.62$ ($+17.8\%$), which translates directly to a downstream multi-task average accuracy boost from $29.50\%$ to $30.25\%$ (Figure 4). Similarly, at SVS rank $k=8$, BWN improves accuracy from $37.25\%$ to $37.50\%$. This is excellent empirical validation of the boundary conditions.

### D. Information-Theoretic Entropy-SVS Pareto-Optimality
* **Claim:** Shannon spectral entropy successfully captures layer-wise informational complexity and allows dynamic rank allocation, tracing a robust Pareto frontier.
* **Evidence:** In Table 2 and Figure 5, the authors demonstrate that Entropy-SVS under multiplier $m_{\text{entropy}}=1.0$ compresses the average rank to $108.74$ ($15.05\%$ compression) with virtually no drop in accuracy ($74.80\%$). More remarkably, at $m_{\text{entropy}}=0.4$, they achieve **$65.7\%$** rank compression (average rank of $43.90$ per layer) with negligible accuracy degradation ($74.55\%$). This strongly validates the spectral complexity hypothesis.

---

## 4. The Intellectual Honesty of the "Representation Gap"
A commendable aspect of the experimental section is the honest discussion regarding why coordinate-basis pruning methods like TIES-Merging ($77.98\%$) and DARE ($75.18\%$) outperform SVS ($74.83\%$). 

Rather than glossing over this gap, the authors provide a highly insightful conceptual analysis: SVS filters high-frequency noise in the continuous spectral domain but results in dense updates that still overlap entirely in the spatial coordinate basis, leading to cross-layer representation interference in deep sequential transformer backbones. In contrast, coordinate-wise pruning (TIES/DARE) completely eliminates parameter overlap at specific spatial coordinate locations, which is more effective at preventing cross-layer representation interference. 

This is a brilliant, high-signal conceptual insight that highlights a fundamental trade-off in weight-space representation and guides future hybrid merging research.
