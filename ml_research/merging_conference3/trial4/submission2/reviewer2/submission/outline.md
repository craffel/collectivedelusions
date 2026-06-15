# Paper Outline: OmniMerge

## 1. Title & Abstract
- **Title:** OmniMerge: Multi-Schema Stochastic Co-Optimization for Robust Model Merging on Heterogeneous Edge Hardware
- **Authors & Affiliation:** Dr. Marcus Thorne (Georgia Institute of Technology, marcus.thorne@gatech.edu)
- **Abstract:**
  - Introduce on-device multi-task ensembling via model merging as a zero-overhead alternative to deploying multiple large-scale models.
  - Detail the critical bottleneck: Edge hardware and runtime compilers employ highly heterogeneous, incompatible post-training quantization (PTQ) standards.
  - Highlight the failure of existing quantization-aware merging methods (like Q-Merge): they suffer from catastrophic cross-schema overfitting, where coefficients optimized for one hardware schema (e.g., Symmetric Per-Channel) collapse to random-guess performance on other hardware target schemas.
  - Introduce **OmniMerge**, which resolves this bottleneck via **Stochastic Operator Sampling (SOS)** and **Scale/Zero-Point Noise Perturbation (SZNP)** during test-time calibration.
  - Summary of results: Evaluated on `ViT-Tiny` across 5 target hardware schemas under robust 8-bit quantization. OmniMerge achieves up to 50.78% average accuracy and completely closes the cross-schema generalization gap, outperforming the state-of-the-art Q-Merge baseline across all 5 schemas.

## 2. Introduction
- **Pragmatic Context:** Deploying deep learning models on resource-constrained edge devices (IoT, smartphones, automotive DSPs) requires aggressive compression (quantization, pruning) and multi-task capability.
- **The Solution:** Model merging (e.g., Task Arithmetic, Model Soup) combines multiple specialized task experts into a single backbone, requiring zero inference-time compute or memory overhead.
- **The Bottleneck:** To maintain accuracy under quantization, quantization-aware model merging (Q-Merge) optimizes merging coefficients under a simulated hardware operator. However, real-world deployment targets have heterogeneous hardware and compilers (TPU, Apple Neural Engine, DSP, GPU) using mismatched quantization schemas.
- **The Failure (Cross-Schema Collapse):** Standard optimization overfits to the rounding boundaries of a specific "source" operator. When deployed onto mismatched target hardware, accuracy collapses catastrophically (e.g., down to 8.79%).
- **Our Contribution (OmniMerge):**
  - We propose Multi-Schema Stochastic Co-Optimization.
  - Key technique 1: Stochastic Operator Sampling (SOS) which stochastically samples active quantization operators from a discrete pool at each optimization step.
  - Key technique 2: Scale and Zero-Point Noise Perturbation (SZNP) which adds noise to the dynamic rounding grid to smooth out the quantized loss landscape.
  - Empirical Validation: Demonstrates robust, stable performance across all heterogeneous target hardware schemas without requiring extra hardware metadata or test-time compute.

## 3. Related Work
- **Model Merging:** Discuss Task Arithmetic, Model Soups, AdaMerging. Mention that these are typically optimized or designed in full-precision (FP16/FP32).
- **Quantization-Aware Merging:** Discuss Q-Merge and ZipMerge. Point out that they optimize for a specific, single quantization target operator (usually per-channel symmetric).
- **Quantization Robustness:** Discuss post-training quantization (PTQ), Straight-Through Estimators (STE). Point out the lack of work on robustness of merged models to heterogeneous runtime environments.

## 4. Methodology (OmniMerge)
- **Problem Formulation:**
  - Define weight blending using task vectors: $\theta^l_{\text{merged}}(\Lambda) = \theta^l_{\text{pre}} + \sum_{k=1}^K \lambda^l_k \tau^l_k$.
- **Stochastic Operator Pool:**
  - Discrete pool of hardware schemas $\mathcal{Q}$ (Symmetric/Asymmetric, Per-Channel/Per-Tensor).
- **Quantization Equations:**
  - Formal mathematical definitions of Symmetric and Asymmetric quantization operators.
- **Stochastic Co-Optimization Techniques:**
  - *Stochastic Operator Sampling (SOS)*: $Q^{(t)} \sim \mathcal{Q}$.
  - *Scale & Zero-Point Noise Perturbation (SZNP)*: $\epsilon_s \sim \mathcal{N}(0, \sigma^2_{\text{scale}})$, $\epsilon_z \sim \mathcal{N}(0, \sigma^2_{\text{zero}})$.
- **Optimization Objective:**
  - Unsupervised Shannon prediction entropy over a small calibration stream ($N=64$ images).
  - Task-Consensus Regularization (TCR) with absolute proximity and group consensus penalties to control parameter drift and encourage inter-task consensus.
- **Gradient Flow:**
  - Straight-Through Estimator (STE) with "operator-dropout" effect.

## 5. Experiments
- **Experimental Setup:**
  - Backbone: `ViT-Tiny` patch16_224.
  - 4 Task Experts: MNIST, FashionMNIST, CIFAR-10, SVHN.
  - Target bit-width: Robust 8-bit quantization.
  - Optimization settings: $N=64$ unlabeled images per task, 15 steps of Adam.
- **Baselines:**
  - FP16 Task Arithmetic
  - Naive Merge-then-Quantize (M-then-Q)
  - Quantized AdaMerging
  - Q-Merge (Symmetric Per-Channel)
- **Results & Analysis:**
  - Present the Cross-Schema Accuracy Retention Matrix.
  - Analyze the catastrophic collapse of Q-Merge under mismatched operators.
  - Highlight OmniMerge's superior performance (up to 50.78% average accuracy) and robust retention across all 5 schemas.
  - Discuss the impact of scale/zero-point perturbations in smoothing the discretization loss landscape.
  - Refer to `fig1.png` for a visual comparison of accuracy across target schemas.

## 6. Conclusion
- Summary of OmniMerge as a simple, training-free, and deployment-friendly solution for multi-hardware edge deployment.
- Emphasize how OmniMerge addresses a major real-world bottleneck of AI deployment on heterogeneous device fleets.
- Outline future work (e.g., extensions to LLMs, sub-4-bit quantization).
