# Soundness and Methodology Evaluation

## Clarity of the Description
- **Strengths:** The methodology is exceptionally well-structured and described with high mathematical clarity. All terms, variables, and objectives are clearly defined:
  - Eq. 3 defines the linear parameter-space task vector merging.
  - Eq. 4 (Elastic Spatial Regularization) clearly delineates the Proximity Penalty ($\beta$) and the Spatial Deviation Penalty ($\gamma$).
  - Eq. 5 (Class-Capacity Normalization) mathematically defines the normalized prediction entropy.
  - Eq. 6 (Scale-Normalized Entropy Weighting) provides the formula for baseline inverse-entropy scaling weights.
  - Eq. 7 shows the unified joint loss function.
- **Areas for Improvement:** While the math is rigorous, there is limited high-level pseudocode or architectural flow diagrams illustrating how the calibration stream is processed, how $w_k$ is cached, and how the optimization loop executes at test-time.

## Appropriateness of Methods
- **SNEW & CCN:** These are highly appropriate and elegant solutions for addressing gradient imbalance in multi-task setups. Class-Capacity Normalization addresses the difference in classification label-spaces (scaling by $1/\log C_k$), while SNEW scales by baseline entropy to prevent "easy" tasks from dominating the optimization landscape.
- **ESR:** This is a structurally logical way to control parameter drift. It bridges the gap between uniform static merging (zero spatial degrees of freedom) and unconstrained layer-wise optimization (which is highly prone to transductive overfitting).

## Potential Technical Flaws & Practitioner Critiques
From a **Practitioner's** perspective, there are several key concerns and potential flaws in the methodology and its real-world viability:
1. **Optimization Latency and Inference Overhead:** Test-time adaptation requires optimizing merging coefficients on the fly. The paper discusses running first-order gradient descent (Adam GD) and derivative-free evolution (1+1 ES). However, in practical deployment, inference latency is a primary constraint. The paper completely omits runtime benchmarks (e.g., time in seconds, FLOPs, or GPU memory usage) for the optimization phase. This is a critical omission for real-world deployment where on-the-fly optimization could cause unacceptable latency spikes.
2. **The Shuffling Anomaly in Evolutionary Optimization:** In Table 1, under *Unconstrained AdaMerging (1+1 ES)*, the model achieves 59.77% Joint Mean accuracy. When the optimized coefficients are spatially shuffled (*Shuffled 1+1 ES*), the accuracy actually *increases* to 60.45%! The authors do not explain this anomaly. If shuffling *improves* performance, it strongly indicates that the 1+1 ES optimizer failed to find a stable localized representation and was instead trapped in local minima or noisy parameter states, casting doubt on the soundness of derivative-free optimization in this context.
3. **Complexity vs. Performance Trade-off:** The *Calibrated Spatial Mean (Cal-Mean)* baseline (Method 9) optimizes only 1 scalar per task ($K=4$ variables) and achieves a high Joint Mean accuracy of 61.13%. Our proposed flagship *CalMerge* (Method 8) optimizes layer-wise parameters ($K \times L = 52$ variables) and achieves 61.82%. The absolute difference is only **0.69%**. In a practical engineering environment, optimizing 4 variables is vastly simpler, faster, safer, and less prone to overfitting than maintaining and optimizing 52 variables. The paper does not justify whether this marginal 0.69% accuracy boost is practically worth the added architectural complexity of layer-wise coefficient optimization and regularizing with ESR.
4. **Sensitivity of SNEW to Calibration Batch Noise:** SNEW computes task weights $w_k$ using the baseline entropy on a single calibration batch of size 16. In practice, a tiny batch of size 16 can be highly noisy, containing unrepresentative outliers. If the initial calibration batch is skewed, $w_k$ could be computed incorrectly, leading to biased joint optimization. The paper lacks a sensitivity analysis evaluating how SNEW and CCN perform across different calibration batch sizes (e.g., 4, 8, 32, 64) or in the presence of sample noise.

## Reproducibility
- **Strengths:**
  - High level of detail regarding the model backbone (CLIP ViT-B/32, 86M parameters), layer groups ($L=13$), task experts, and datasets (MNIST, FashionMNIST, CIFAR-10, SVHN).
  - Explicit reporting of hyperparameters (Adam learning rate $10^{-3}$, calibration batch size 16, test size 256 per domain, seeds 42, 43, 44).
  - The paper honestly reports deterministic outcomes ($\pm 0.00\%$ standard deviation) for Adam GD and explains its mathematical cause (cached splits and fixed initializations), reflecting strong scientific integrity.
  - Section 4.3.3 provides a concrete, step-by-step heterogeneous class count simulation, showing exact math and numerical calculations for replication.
- **Weaknesses:**
  - No public repository link or code artifact is provided, although the paper references "our concrete code implementation."
