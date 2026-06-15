# Experimental Evaluation

## Experimental Setup
The experimental evaluation is highly systematic, well-designed, and exhaustive:
* **Diverse Datasets:** Evaluated on a high-conflict multi-task vision suite consisting of MNIST, FashionMNIST, CIFAR-10, and SVHN.
* **Standard Backbone:** Utilizes a standard, pre-trained Vision Transformer backbone (`vit_tiny_patch16_224`) and extracts feature representations from the early semantic layer (Layer 3).
* **Systematic sweeps:** Sweeps representation noise $\sigma^2 \in [0.0, 0.20]$, calibration sample sizes $N \in [8, 256]$, and task registry dimensions $K \in [4, 64]$.
* **Rigorous Seeds:** Reports mean and standard deviation across 20 independent random seeds.
* **Ecosystem and Physical Validation:** Extends the evaluation to end-to-end input-level image corruptions, physical multi-task registries (sub-tasks via KMeans), and physical MCU-emulated resource benchmarks.

## Baselines
The baselines are comprehensive and include:
1. **Raw Cosine Thresholding:** A non-parametric, single-coordinate similarity threshold.
2. **Unregularized Diagonal GMM (SPS-ZCA):** The standard maximum likelihood estimator.
3. **Ridge Diagonal GMM:** A standard L2-regularized diagonal GMM with static regularization ($\gamma = 10^{-4}$).
4. **Tuned Ridge Diagonal GMM:** A highly competitive baseline that dynamically selects the optimal ridge coefficient $\gamma$ per task using a 3-fold cross-validation scheme directly over the calibration splits.

## Do the Results Support the Claims?
Yes, the results support the specific claims that unregularized GMMs are highly sensitive to low-resource calibration overfitting and covariate shift, and that **SRC-DE** successfully stabilizes these density boundaries, reducing estimator variance and improving multi-component GMM AUC under noise.

However, the experimental results also expose a **fundamental conceptual weakness** of the entire parametric coordinate density framework:
* **The Dominance of the Non-Parametric Baseline:** In Table 2 (Robustness to Covariate Shift) and Table 4 (Sample Complexity), the simple, non-parametric **Raw Cosine** baseline consistently and massively outperforms *all* parametric GMM models—including their proposed regularized SRC-DE. Under $N=64$ and moderate noise ($\sigma^2 = 0.05$), Raw Cosine achieves **0.9040** AUC, while the best SRC-DE GMM achieves only **0.7648** AUC. Under severe noise ($\sigma^2=0.20$), Raw Cosine retains **0.7915** AUC, while SRC-DE drops to **0.6059** AUC.
* **The Inactive Dimension Noise Accumulation:** The authors deconstruct this performance gap by revealing that joint multi-dimensional GMMs suffer from the "curse of dimensionality" under noise: representation noise accumulates over the $K-1$ inactive coordinate dimensions, burying the active routing signal under a sum of noisy dimensions with collapsed variances. 
* **The High-Dimensional Collapse:** When scaling coordinate dimensions to $K=64$ (Table 6), the joint GMM models collapse catastrophically, with the best regularized model dropping to **0.6180** AUC. To resolve this, the authors are forced to propose alternative architectures like **Independent 1D GMMs** (which bypass high-dimensional noise but are completely blind to semantic overlaps) or **Hierarchical Hybrid Routing**.

In summary, the experiments successfully validate that SRC-DE is a superior regularizer *for diagonal GMMs* compared to unregularized or static Ridge GMMs. However, they also reveal that diagonal joint GMMs are a fundamentally fragile and suboptimal choice for coordinate-space OOD task rejection compared to simpler, robust 1D similarity thresholding schemes under realistic serving noise.
