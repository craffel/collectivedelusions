# 2. Novelty and Positioning Check

## Assessment of Novelty
The overall novelty of the paper is **good to fair**. While the individual mathematical components of OmniMerge are largely adapted from existing fields—such as stochastic quantization, data augmentation, and test-time adaptation—the proposed synthesis and its specific application to the problem of cross-schema model merging robustness are highly creative, practical, and novel.

### Key Novelty Dimensions
1. **Identification of the "Cross-Schema Performance Degradation" Phenomenon:** 
   The paper's primary conceptual contribution is identifying and demonstrating that optimizing merging coefficients under a single static quantization schema (as done by Q-Merge) leads to acute overfitting to that schema's rounding boundaries. This is a highly pragmatic and valuable observation that has not been systematically analyzed in prior model merging literature.
2. **Stochastic Operator Sampling (SOS) as Parameter-Space Augmentation:**
   Stochastic quantization is a known technique in Quantization-Aware Training (QAT). However, using a stochastically sampled pool of *coarse structural operators* (switching between Symmetric Per-Tensor, Symmetric Per-Channel, Asymmetric Per-Tensor, and Asymmetric Per-Channel) during test-time calibration represents a novel and elegant interpretation of parameter-space augmentation. It represents a highly effective way to enforce compiler-invariant optimization.
3. **Scale and Zero-Point Noise Perturbation (SZNP) as Landscape Smoothing:**
   Adding Gaussian noise to scale factors and zero points is conceptually similar to variational noise injection or weight noise. Its use during STE backpropagation to smooth the highly discontinuous "rounding cliffs" in the loss landscape is an intuitive and mathematically sound way to stabilize the Adam optimizer during short test-time adaptation sweeps.
4. **Task-Consensus Regularization (TCR):**
   TCR is a more incremental contribution. Restricting parameter drift from initial configurations is a common practice in test-time adaptation (e.g., in AdaMerging, RegCalMerge, and EATA). Penalizing task-wise deviations from a layer-wise consensus average ($\bar{\lambda}^l$) is a sensible heuristic for ensembling, but represents a minor, straightforward modification of existing spatial regularization techniques.

---

## Positioning Relative to Prior Art
The paper is exceptionally well-positioned and does a commendable job of contextualizing its contributions within the broader machine learning literature. 

### Strengths in Positioning
- **Comprehensive Related Work Section:** 
  Section 2 provides a thorough overview of three related domains: (1) Model Merging (Task Arithmetic, Model Soups, Fisher Merging, RegMean, TIES-Merging, DARE, etc.), (2) Quantization-Aware Model Merging (Q-Merge, ZipMerge, RegCalMerge), and (3) Post-Training Quantization (PTQ) and Test-Time Adaptation (TTA) (AdaRound, BRECQ, SmoothQuant, Tent, EATA, etc.).
- **Clear Distinction from Closely Related Work:**
  The paper clearly distinguishes OmniMerge from standard model merging (which operates in FP16/FP32 and ignores quantization) and standard quantization-aware merging (which optimizes coefficients under a static, single operator). It explicitly notes that prior methods rely on a static, single simulated operator, making them vulnerable to catastrophic cross-schema overfitting.
- **Accurate Mathematical Mapping:**
  The paper accurately positions its math relative to standard asymmetric and symmetric post-training quantization, making the connection between the proposed noise perturbations and the standard formulas transparent and easy to follow.

### Gaps and Opportunities in Positioning
- **Lack of Discussion on Quantization Noise Literature:**
  The paper would benefit from citing and discussing prior work that uses quantization noise during training, such as **Quantization Noise (QN)** by Fan et al. (2020) or **DiffQ** (stochastic quantization noise). Positioning SZNP relative to these works would strengthen the theoretical foundation of the paper.
- **Vague Treatment of "CMA-ES/Stochastic Random-Walk" Baselines:**
  In Section 3.4, the paper discusses zeroth-order optimization methods (CMA-ES, stochastic random-walk search) as alternatives to STE, criticizing their prohibitive sample complexity. However, the paper does not actually evaluate a zeroth-order optimization baseline in the experiments. Evaluating or citing a specific zeroth-order model merging benchmark (e.g., Evolutionary Model Merging) would ground this critique more firmly.
- **Acknowledge Overlap with Q-Merge's Evaluation:**
  The paper mentions that "The robustness audit conducted in [qmergeaudit] highlighted that such single-schema optimization makes the model highly vulnerable to catastrophic cross-schema overfitting." Given that this "robustness audit" is a highly influential reference for this paper's premise, the authors should be more explicit about how OmniMerge's formulation directly builds upon or departs from the audit's findings (e.g., whether the audit proposed any solutions, or if OmniMerge is the first constructive solution).
