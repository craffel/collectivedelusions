# Peer Review Evaluation - Summary (1_summary.md)

## 1. Main Topic and Scope of the Paper
The paper addresses a critical and previously undocumented vulnerability in **dynamic model merging**: catastrophic low-data overfitting of lightweight routing layers during the post-hoc calibration phase. Model merging combines specialized expert neural networks into a single multi-task model. Dynamic model merging uses input-dependent routing to predict sample-specific merging coefficients on the fly. However, because calibration data is often extremely scarce in practice (e.g., $B_{cal} \le 64$ samples across tasks), unconstrained routing parameters aggressively overfit to local noise, causing representation-space collapse and a severe drop in out-of-distribution (OOD) performance.

To solve this optimization failure, the authors propose **Task-Space Anchor Regularization (TSAR)**, a simple, geometrically grounded classical regularizer. TSAR computes stable, task-specific feature centroids (anchors) from the pre-trained expert representations and applies a quadratic spatial penalty to keep the layer-wise routing weights aligned with these centroids during calibration.

## 2. Proposed Approach and Mathematical Formulation
The TSAR framework operates in three main steps:
1. **Low-Dimensional State Representation:** High-dimensional visual features $z(x)_b \in \mathbb{R}^{192}$ are projected onto a compact, low-dimensional coordinate space of dimension $d = K$ (where $K$ is the number of tasks) using either unsupervised Principal Component Analysis (PCA) or a data-independent Random Gaussian projection. The projected features are normalized onto the unit sphere to ensure scale invariance:
   $$\psi(x)_b = \frac{z(x)_b P}{\|z(x)_b P\|_2 + \epsilon}$$
2. **Task Feature Anchors:** Centroids are computed offline over the small calibration split for each task $k$:
   $$\bar{\psi}_k = \frac{1}{|X_{cal, k}|} \sum_{b \in X_{cal, k}} \psi(x)_b$$
3. **Anchor-Guided Optimization Loss:** During calibration, the cross-entropy loss is augmented with standard $L_2$ weight decay and the TSAR quadratic distance penalty, pulling each layer $l$'s routing weights $W_{l, k}$ toward its corresponding task anchor $\bar{\psi}_k$:
   $$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda_{wd} \sum_{l=1}^L \sum_{k=1}^K \left( \|W_{l, k}\|_2^2 + B_{l, k}^2 \right) + \lambda_{anchor} \sum_{l=1}^L \sum_{k=1}^K \| W_{l, k} - \bar{\psi}_k \|_2^2$$

To resolve multi-task gradient cross-talk where hard tasks dominate the gradients and corrupt easy-task routing parameters, the authors integrate **Projecting Conflicting Gradients (PCGrad)** during optimization, projecting conflicting gradients onto normal planes when their cosine similarity is negative.

To address **heterogeneity collapse** in streaming (where positive and negative routing coefficients cancel out under mixed-task batches), the authors propose a **scaled Sigmoid activation** bounded at $[0, 1.5]$ as a non-negative, unconstrained alternative to Softmax.

## 3. Key Findings and Quantitative Results
* **Mitigating Overfitting:** Under $B_{cal}=64$, unconstrained linear routers collapse to a poor **23.20%** Joint Mean accuracy. Introducing the TSAR penalty ($\lambda_{anchor}=0.1$) stabilizes calibration, boosting Joint Mean to **54.10%** and completely resolving overfitting.
* **Gradient Conflict Resolution:** Incorporating PCGrad with TSAR resolves multi-task gradient dominance, yielding a new state-of-the-art Joint Mean of **57.06%** (surpassing Static Uniform Merging by **+5.20%** and the wave-based QWS-Merge SOTA by **+17.18%**).
* **Streaming Deployment Robustness:** Under mixed-task streaming ($B=256$), standard unconstrained TSAR suffers from coefficient cancellation, dropping to **43.10%** (heterogeneity collapse). The proposed scaled Sigmoid activation fully resolves this with zero serving latency or parameter overhead, achieving a stable **50.80%** accuracy.
* **Physical Weight-Space Validation:** Merging the classification heads of a real pre-trained Vision Transformer (ViT-Tiny) using TSAR + PCGrad outperforms Static Uniform Merging by **+13.90%** on synthetic stimuli and by a spectacular **+23.60%** on raw, uncurated natural image manifolds (MNIST and CIFAR-10).
* **Representational Redundancy:** The authors mathematically prove and empirically verify that under linear batch-averaged routing, layer-wise over-parameterized routers ($L=14$) collapse to a single-layer global router ($L=1$) during deployment, allowing a **92.8%** reduction in parameters (down to only 20 parameters) with negligible performance loss.

## 4. Explicitly Claimed Contributions and Supporting Evidence
* **Systematic Exposure of Low-Data Overfitting:** Documented via unregularized router baselines in Table 1, showing catastrophic collapse on MNIST (37.84%) and CIFAR-10 (9.52%) under $B_{cal}=64$.
* **Mathematical Formulation of TSAR:** Fully detailed in Section 3, proving how centroids provide a stable coordinate system on the unit sphere to guide optimization.
* **Empirical Dominance over Baselines:** Validated in Table 1, showing TSAR + PCGrad outperforms Static Uniform, AdaMerging, and QWS-Merge SOTA across 5 independent seeds.
* **Characterization and Resolution of Heterogeneity Collapse:** Exposed in Table 4 and Figure 4, and resolved using scaled Sigmoid activations with zero serving overhead.
* **Physical Weight-Space Model Merging:** Demonstrated on a real Vision Transformer (ViT-Tiny) in Table 5 and Table 6, establishing clear real-world utility and practical generalizability.
