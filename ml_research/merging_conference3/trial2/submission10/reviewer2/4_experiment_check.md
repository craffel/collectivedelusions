# 4. Experimental Evaluation and Results Support

This assessment provides a critical analysis of the paper's experimental setup, dataset choice, baselines, and whether the presented empirical data actually support the authors' central claims.

---

## Evaluation of the Experimental Setup and Datasets

The experimental setup is exceptionally well-designed for the research questions under investigation:
1. **The Gradient of Difficulty:** The authors selected a four-task visual benchmark consisting of:
   * **MNIST** and **FashionMNIST**: Easy, highly homogeneous tasks with clean backgrounds.
   * **CIFAR-10**: Moderate difficulty, natural images with diverse classes.
   * **SVHN**: Hard, highly heterogeneous real-world street numbers featuring drastic variations in illumination, fonts, background clutter, and distracting digits.
   This specific selection is crucial. A homogeneous task suite (e.g., merging only digits) would not have triggered the significant prediction entropy and gradient imbalances that exposed the **Spatial Averaging Paradox**. The inclusion of SVHN is particularly valuable for measuring representation collapse.
2. **The Backbone choice:** Using **CLIP ViT-B/32** is a strong standard baseline. Since CLIP is trained via multi-modal contrastive pre-training, its visual representations are highly generalizable, providing a stable foundation for testing task vectors and model merging.

---

## Critical Analysis of Baselines

The paper evaluates a comprehensive and highly rigorous set of baselines:
* **Static Baselines:** Includes **Task Arithmetic** (the baseline to beat), **TIES-Merging**, and **DARE-Merging**. It is noteworthy that TIES-Merging and DARE-Merging collapse significantly on this heterogeneous benchmark (dropping to $77.54\%$ and $73.67\%$ average accuracy, and crashing to $49.46\%$ and $40.61\%$ on SVHN respectively). This is a strong, high-signal finding that shows static pruning heuristics can destroy critical representation directions in highly disparate domains.
* **State-of-the-Art Adaptive Baselines:** Includes SOTA **Layer-wise AdaMerging** under both zero-order (1+1 ES) and first-order (Adam GD) optimization.
* **Proposed Formulations and Controls:** Includes **Task-wise AdaMerging**, **Calibrated Task-wise AdaMerging**, and two diagnostic controls: **Intra-Task Layer Shuffling** and **Spatially Averaged** models.
* **Scale Sweeps:** The paper also includes a complete grid sweep over static scaling factors for standard Task Arithmetic ($\lambda \in \{0.1, 0.2, 0.3, 0.4, 0.5\}$) in Section 4.2.4 to verify whether an oracle-tuned static scale could beat the adaptive models.

This selection of baselines is outstanding and exceeds the standard of typical model-merging papers.

---

## Do the Results Support the Claims?

The authors' claims are fully and rigorously supported by the presented empirical data:

### Claim 1: Layer-wise AdaMerging captures hierarchical routing but suffers from test-time overfitting.
* **Empirical Support:**
  * **Intra-Task Layer Shuffling** collapses average performance from $88.05\%$ down to $78.61\%$ (Adam GD), demonstrating that the learned coefficients are deeply tied to the network's structural hierarchy. Shuffling breaks this architectural alignment, resulting in a severe performance collapse.
  * **Spatial Averaging** reduces the parameter space to 4 scalars but achieves $84.96\%$ (Adam GD). Although this represents a $3.09\%$ performance trade-off compared to the unconstrained $88.05\%$, it still successfully outperforms Task Arithmetic ($84.64\%$). This proves that post-hoc averaging acts as a powerful regularizer, smoothing away transductive overfitting from the tiny calibration batch while preserving the robust global task-level scales.

### Claim 2: Direct Task-wise AdaMerging fails spectacularly (The Spatial Averaging Paradox).
* **Empirical Support:**
  * **Task-wise AdaMerging (Adam GD)** collapses average accuracy to **81.19%**, which is **3.45% lower than its unoptimized uniform initialization** (Task Arithmetic baseline of $84.64\%$).
  * Why? The task-level accuracies show that while MNIST and F-MNIST remain high, the harder tasks collapse: CIFAR-10 drops from $89.93\%$ to $81.45\%$, and SVHN collapses from $69.94\%$ to $63.71\%$. Under uncalibrated prediction entropy, the optimizer is dominated by easy tasks (MNIST/F-MNIST), scaling up their coefficients to drive joint entropy down and causing destructive weight interference on CIFAR-10 and SVHN.

### Claim 3: The failure is due to a structural bottleneck rather than pure gradient initialization imbalance.
* **Empirical Support:**
  * **Calibrated Task-wise AdaMerging (Adam GD)** only achieves **80.59%** (and **80.78%** under 1+1 ES), continuing to degrade performance below the baseline.
  * Since normalizing the losses at initialization fails to restore performance, this proves that the pathology is structural: a global, low-dimensional weight bottleneck is fundamentally incompatible with joint prediction entropy minimization, as forcing the network to produce sharp outputs on a joint unlabeled calibration batch without labels leads to overconfident misclassifications in shared early projection layers.

### Claim 4: The optimization landscape is flat and robust to noise.
* **Empirical Support:**
  * In the noise sensitivity sweep (Figure 5b), both AdaMerging and Task-wise AdaMerging remain highly stable under Gaussian noise perturbations up to $\gamma = 0.50$, confirming that weight merging landscapes are highly flat.

### Claim 5: Early layers maintain representation alignment, while late layers specialize.
* **Empirical Support:**
  * The Linear CKA results in Table 2 and the layer-by-layer curves in Figure 6 confirm near-perfect CKA ($CKA > 0.995$) in early layers (Layers 1--4) across all methods, which are baseline properties of task vector scaling ($\lambda \approx 0.3$).
  * In contrast, late layers (Layers 8--12) show decreased CKA similarity for optimized models, empirically confirming that high-dimensional adaptive optimizers leverage local layer degrees of freedom to route representations through late task-specialized layers while keeping early general representations intact.
