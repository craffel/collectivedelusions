# Paper Summary: Deconstructing "Layer-Averaging Collapse" in Dynamic Model Merging

## 1. Main Topic and Scope
This paper conducts a critical methodological audit of the "Layer-Averaging Collapse" (rank-1 collapse) hypothesis in weight-space dynamic model merging. Dynamic model merging computes sample-specific coefficients on the fly to merge multiple specialized expert models sharing a common initialization. A key structural design question is the spatial resolution of routing—specifically, whether different layers should have specialized routing policies (layer-wise routing) or use a single model-wide configuration (global routing). 

Prior work (referred to as an anonymous theoretical study) claimed that layer-wise routing is mathematically redundant because routing coefficient trajectories across layers inevitably collapse to a collinear, rank-1 subspace. This paper audits this claim, arguing that the "Layer-Averaging Collapse" theorem is a pure artifact of over-simplified, linear representation-space sandboxes and low-conflict environments.

The paper investigates these dynamics empirically on physical neural network architectures (DeepMLP-12 and TinyCNN-4) across task suites of varying semantic conflict, using Split-MNIST digit subsets and conducting extensive diagnostic analysis (SVD, pairwise inter-layer cosine similarity maps).

---

## 2. Methodology and Approach
To isolate and audit the routing space dimensionality, the authors establish a physical dynamic weight-space model merging pipeline:
* **Expert Models:** Pre-trained on disjoint Split-MNIST subsets (Task 0: digits 0/1; Task 1: digits 2/3; Task 2: digits 4/5; Task 3: digits 6/7) from a shared base initialization to ensure local loss basin alignment.
* **Backbone Architectures:** DeepMLP-12 ($L=12$ layers) and TinyCNN-4 ($L=4$ blocks/stages).
* **Task Suites:**
  * Low-Conflict ($K=2$): Task 0 + Task 1
  * High-Conflict ($K=2$): Task 2 + Task 3
  * Cross-Domain ($K=4$): Task 0 + Task 1 + Task 2 + Task 3
* **Bounded Sigmoid (BSigmoid) Router:** To overcome Softmax's zero-sum bottleneck, the authors project the flattened input $x \in \mathbb{R}^D$ into a low-dimensional state $\psi(x) \in \mathbb{R}^d$ ($d=8$) using a random frozen Gaussian projection matrix. Trainable linear projections compute logits, followed by independent sigmoid activations:
  $$\tilde{\alpha}_{l, k}(x) = \sigma(\alpha_{l, k}(x))$$
  A post-gating sum-to-1 normalization is then applied to preserve representation scale across deep layers:
  $$\lambda_{l, k}(x) = \frac{\tilde{\alpha}_{l, k}(x)}{\sum_j \tilde{\alpha}_{l, j}(x) + \epsilon}$$
* **Calibration Protocol:** Calibrated on a balanced dataset of 128 samples per task over 40 steps using Adam ($lr=0.01$) and $L_2$ weight decay ($\gamma=10^{-4}$), minimizing cross-entropy classification loss of the physically merged model.
* **Spectral Diagnostics:**
  * **Batch-Averaged Layer-wise Coefficient Matrix:** $A \in \mathbb{R}^{L \times K}$ is constructed by passing the test dataset through the calibrated router and averaging coefficients over the batch.
  * **SVD Collinearity Ratio:** $\rho_{collinear} = \frac{\sigma_1}{\sum_i \sigma_i}$, where $\sigma_i$ are the singular values of $A$. A ratio near $1.0$ indicates perfect rank-1 collapse (highly collinear routing), whereas a lower ratio (e.g., $<0.6$) indicates multi-dimensional routing.
  * **Inter-Layer Cosine Similarity Matrix:** $S \in \mathbb{R}^{L \times L}$, measuring the pairwise directional alignment of routing coefficients across layers.

---

## 3. Key Findings
* **Refutation of Rank-1 Collapse:** Under Cross-Domain task conflict, the Collinearity Ratio $\rho_{collinear}$ drops to $0.4987 \pm 0.08$ on DeepMLP-12 and $0.5673 \pm 0.03$ on TinyCNN-4, demonstrating that routing trajectories occupy a multi-dimensional subspace. The rank-1 collapse is therefore shown to be a localized phenomenon of low-conflict settings.
* **Emergence of Depth-Specialized Blocks:** Inter-layer cosine similarity maps reveal that while low-conflict suites yield uniform routing across layers ($S \approx 1.0$), cross-domain conflict forces the networks to specialize their routing into distinct block-diagonal structures (e.g., layers 1–4, 5–8, and 9–12 on DeepMLP-12).
* **Decoupled Optimization in BSigmoid:** The BSigmoid router outclasses standard Softmax routing (e.g., by $+24.19\%$ on Cross-Domain TinyCNN-4). The authors prove that this is due to the decoupling of gradient paths during backward propagation, which prevents joint optimization collapse when learning rates/gradients clash between easy and hard tasks.
* **Capacity-Variance Trade-offs:** On TinyCNN-4, the offline static baseline (OFS-Tune) consistently outperforms the dynamic router under tight few-shot calibration budgets (e.g., $53.40\%$ vs $52.52\%$ on Cross-Domain). This is because the global router/OFS-Tune has minimal parameters (very low variance), whereas layer-wise routing has higher spatial capacity but exhibits high optimization variance under scarce data. This variance is controlled, and the dynamic router crosses over to outperform the static baseline, when calibration budgets are scaled from 64 to 1024 samples per task.
* **The Batch-Averaged Multi-Task Inference Paradox:** The authors expose a crucial serving bottleneck in dynamic weight-space merging: averaging routing coefficients over a mixed-task batch collapses dynamic routing back to static, uniform-like merging, whereas homogeneous batching requires knowing the labels beforehand (making merging redundant compared to direct expert routing).
* **Representational Failure of Deep MLPs:** In DeepMLP-12 Cross-Domain, all merged models perform near or below the random guessing threshold (12.5%), showing that full-parameter linear interpolation of deep, dense, fully connected layers under multi-task conflict is a failed paradigm due to unconstrained activation drift and coordinate misalignment.

---

## 4. Explicitly Claimed Contributions (with Evidence)
1. **Deconstruction of "Layer-Averaging Collapse" (Rank-1 Collapse):** Supported by SVD spectral analysis across 5 independent seeds showing the Collinearity Ratio dropping below 0.50 in Cross-Domain task environments.
2. **First Map of Depth-Specialized Routing Trajectories:** Supported by inter-layer pairwise cosine similarity heatmaps displaying clear transitioning block-diagonal structures under cross-domain semantic conflict.
3. **Introduction of the Bounded Sigmoid (BSigmoid) Router:** Conceptually grounded as an activation function with decoupled gradient paths, and empirically verified via classification accuracy improvements of $+20\%$ to $+25\%$ on TinyCNN-4 compared to Softmax.
4. **Empirical Validation of the Capacity-Variance Dilemma & Calibration Scaling:** Proved via scaling experiments showing a performance crossover where the high-capacity Layer-wise Router outclasses OFS-Tune as calibration samples increase.
5. **Formalization of the "Batch-Averaged Multi-Task Inference Paradox":** Conceptually framing the systems-level bottlenecks of dynamic model merging and offering Concrete Pathways forward (e.g., Sample-Specific Low-Rank Adaptive Merging, Task-Aware Bucketing).
6. **Supportive Proofs of Concept:** Standard-deviations for SVD ratios showing robust results; ablation studies on projection dimension; empirical validation on natural images (CIFAR-10/SVHN); and preliminary ViT-B/16 LoRA simulation proving that PEFT-level routing yields even lower collinearity (0.34) and deeper spatial specialization.
