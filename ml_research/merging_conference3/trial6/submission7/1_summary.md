# 1_summary.md - Systematic Summary of PFSR + MBH Framework

This document provides a high-signal technical summary of the submission, detailing its core contributions, mathematical formulations, systems-level co-designs, and primary empirical claims.

---

## 1. Problem Statement and Motivation
Dynamic weight-space routing is a powerful paradigm for merging multi-task neural network experts into a single unified network, performing specialized on-the-fly parameter blending at test time based on representation activations. However, the authors identify two critical and overlooked failure modes in existing architectures:
1. **Optimization Bloat and Out-of-Distribution (OOD) Overfitting:** Existing dynamic routers (such as the wave-inspired QWS-Merge) rely on complex, highly-parameterized multi-layer neural networks trained via iterative optimization (e.g., AdamW). This over-parameterization leads to transductive overfitting on small calibration splits, causing catastrophic collapse when presented with OOD tasks (e.g., QWS-Merge drops to $10.00\%$ accuracy on SVHN).
2. **Heterogeneity Collapse in Mixed-Task Streams:** When a batch contains a mixture of different tasks, standard dynamic routers average sample-wise coefficients across the batch dimension to comply with hardware accelerator requirements. This averaging forces coefficients toward a flat, uniform distribution, completely destroying task specificity (e.g., the Linear Router drops from $51.00\%$ to $43.40\%$).

---

## 2. Key Methodological Contributions
To resolve these issues under a strict philosophy of **The Minimalist** (Occam's razor), the authors propose a zero-shot, completely non-parametric framework containing **zero trainable parameters** and requiring **zero calibration split data**:

### 2.1 Parameter-Free Subspace Routing (PFSR)
PFSR projects high-dimensional intermediate features $z_b \in \mathbb{R}^D$ from the penultimate layer of the backbone onto a frozen, low-dimensional task coordinate space. It does so by computing the maximum cosine similarity between the expert block features $z_{k, b}$ and the rows of the pre-trained expert classification weights $W_k \in \mathbb{R}^{C \times d}$:
$$u_{k, b} = \max_{c \in \{1, \dots, C_k\}} \frac{W_{k, c} \cdot z_{k, b}}{\|W_{k, c}\|_2 \|z_{k, b}\|_2}$$
The routing coefficients are derived directly from these coordinates via a temperature-scaled Softmax:
$$\alpha_{k, b} = \frac{\exp(u'_{k, b} / \tau)}{\sum_{j=1}^K \exp(u'_{j, b} / \tau)}$$

### 2.2 Micro-Batch Homogenization (MBH)
To shield model merging from heterogeneity collapse, MBH handles stream heterogeneity at the data-stream level. It dynamically partitions mixed-task streams into homogeneous micro-batches on the fly based on the dominant task coordinates:
$$k_b^* = \arg\max_k u_{k, b}$$
$$X^{(g)} = \{x_b \in X \mid k_b^* = g\}$$
For each active micro-batch $X^{(g)}$, coefficients are aggregated by averaging only within the micro-batch size $|X^{(g)}|$:
$$\bar{\alpha}_k^{(g)} = \frac{1}{|X^{(g)}|} \sum_{x_b \in X^{(g)}} \alpha_{k, b}$$
The parameters are merged specifically for each micro-batch, and the final predictions are re-assembled into the original sequential order.

### 2.3 Statistical Class-Size Scaling Calibration
In heterogeneous expert registries with asymmetrical output space sizes (e.g., $C_1 = 32,000$ for an LLM vs. $C_2 = 10$ for digits), the expected maximum of random similarities is statistically biased to be larger for larger vocabulary sizes. To resolve this maximum bias, the authors normalize the raw coordinates:
$$u'_{k, b} = \frac{u_{k, b}}{\sqrt{2\log C_k / d}}$$

### 2.4 Mitigation Strategies for Massive Vocabulary spaces ($C \ge 32,000$)
To prevent systems bottlenecks in LLM weight merging, the authors introduce:
1. **Subspace Dimension Reduction:** Projecting features and weights to a bottleneck dimension $M=128$.
2. **Sub-Vocabulary Prototype Selection:** Selecting $C_{sub}=256$ high-variance tokens across experts based on parameter-space variance, completely bypassing text corpus dependencies.

### 2.5 Unit-Norm Calibration (UNC)
UNC pre-normalizes both representation features and classification prototypes to unit-norm, mitigating local representation scale imbalances and stabilizing the similarity manifold structure.

### 2.6 Out-of-Distribution (OOD) Rejection
To handle OOD tasks, the authors propose a **Cosine Rejection Threshold** $\gamma_{OOD}$ or a multi-component **Gaussian Mixture Model (GMM) Density Estimator** with diagonal covariance and positive-definite ridge regularization ($\epsilon I$), routing OOD samples to a frozen base backbone with uniform merging weights.

### 2.7 Dynamic Temperature Scheduling
For ambiguous samples situated at task boundaries, a dynamic scheduler scales the Softmax temperature on the fly to soften routing:
$$\tau_b = \frac{\tau_{base}}{\Delta_b + \epsilon}$$
where $\Delta_b = s_{1, b} - s_{2, b}$ is the similarity margin between the top two tasks.

---

## 3. Primary Empirical Claims
1. **OOD Generalizability:** PFSR + MBH achieves a high **75.00% Joint Mean accuracy** under homogeneous streams in a synthetic sandbox, completely resolving OOD overfitting.
2. **Heterogeneity Robustness:** Under heterogeneous mixed-task streams ($B=256$), PFSR + MBH maintains a collapse-free **71.60% Joint Mean accuracy**, while parametric alternatives degrade severely (Linear Router collapses to $43.40\%$, QWS-Merge collapses to $43.30\%$).
3. **Real-World Scalability:** On Vision Transformers (DomainNet) and LLaMA-7B (NLP), PFSR + MBH + UNC recovers up to **$97.5\%$** and **$96.8\%$** of the expert standalone ceilings ($78.50\%$ and $79.12\%$ Mean accuracy), systematically outperforming strong static (TIES-Merging) and parametric dynamic routing baselines.
4. **Systems Viability:** Running low-rank parameter merging of adapters executes in less than a millisecond on GPU, and SGMV parallel kernel execution achieves true constant-time $O(1)$ batch execution with a mere **$5.71\%$** systems overhead.
5. **Dynamic Task Adaptation:** PFSR enables instantaneous registration or retirement of task experts in massive model hubs with **zero retraining or joint calibration**, establishing a robust plug-and-play production registry.
