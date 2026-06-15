# Evaluation Task 4: Experimental Evaluation Check

## Datasets, Baselines, and Experimental Setup
1. **Datasets**:
   * **Synthetic Sandbox (ACS)**: Simulates a 4-task visual stream mapping representation distributions for **MNIST**, **FashionMNIST**, **CIFAR-10**, and **SVHN**.
   * **Real-World Validation**: Merges actual Vision Transformer (ViT-B/16) expert checkpoints fine-tuned on **CIFAR-10** and **CIFAR-100**.
2. **Baselines**:
   * **Static Uniform**: Simple parameter average (weight $1/K = 0.25$ or $0.5$ across all layers).
   * **Globally-Scaled Task Arithmetic ($d=0$)**: Optimizes a single global scalar per task.
   * **Offline Unconstrained**: Optimizes layer-wise weights independently on the calibration set.
   * **RBPM ($d=2$)**: Quadratic polynomial trajectory mapping with a learning-theoretic penalty.
3. **Calibration Budget**: Extreme few-shot calibration (10 samples per task, for a total of 40 samples in the sandbox and 20 samples in the real-world validation).

---

## Critical Evaluation of the Results: Do They Support the Claims?

1. **Mitigation of Boundary Runaway (Claim Supported)**:
   The results support the authors' claim that polynomial trajectories suffer from boundary runaway. Inside the sandbox, the quadratic polynomial competitor **RBPM ($d=2$)** performs poorly on Deep12LayerCNN (**39.30%** categorical accuracy) and CLIP ViT-B/16 (**63.50%**). On actual ViT checkpoints, RBPM gets **70.70%** accuracy, which is lower than the parameter-free Static Uniform baseline (**71.30%**). This suggests that global polynomial constraints indeed cause severe boundary oscillations that degrade representation propagation in deep architectures. Our proposed Fourier and DCT variants exhibit much higher stability, confirming the utility of sinusoidal bounding.

2. **The Static Uniform Dominance Paradox (Contradiction of Motivation)**:
   In the synthetic sandbox (Table 1), **Static Uniform** achieves **85.10%** categorical accuracy on CNN and **83.75%** on CLIP. This is **substantially higher** than *all* tuned and adaptive methods, including our best proposed spectral method (RB-FTM achieves 70.70% on CNN and 72.70% on CLIP). 
   * This represents a major experimental contradiction: the primary evaluation sandbox used to study ensembling trajectories demonstrates that **the simplest, parameter-free, zero-tuning baseline is actually the best method**.
   * While the authors try to justify this as an "anisotropic shearing pathology" of aligned spaces, it means the sandbox is an unrealistic, idealized environment that fails to demonstrate any practical benefit of adaptive ensembling. 

3. **Real-World Performance Gains (Modest and Complex)**:
   On actual ViT checkpoints (Table 2), where coordinate alignment is imperfect, **RB-DCTM (F=2)** achieves the highest accuracy of **74.90%**, outperforming Static Uniform (71.30%) and Offline Unconstrained (69.80%).
   * While this supports the claim that spectral trajectory optimization mitigates overfitting in real-world settings, the absolute gain over the extremely simple **Globally-Scaled Task Arithmetic ($d=0$)** baseline (72.50%) is only **2.40%** (74.90% vs 72.50%).
   * Since Globally-Scaled uses only 2 parameters and requires no Fourier or DCT mapping, no Rademacher complexity bounds, and no boundary-condition safeguards, a minimalist would argue that the massive mathematical complexity of RB-DCTM is not justified by this small marginal gain.

---

## Technical and Scientific Gaps in the Experiments

1. **Complete Absence of Error Bars and Significance Testing**:
   The entire paper conducts extremely few-shot calibration (10-shot, i.e., 10 samples per task). Few-shot optimization is notoriously sensitive to the specific choice of the calibration samples, and the variance across different calibration splits is typically very high.
   * **The Flaw**: Neither Table 1 nor Table 2 contains standard deviations, error bars, or statistical significance testing (such as p-values or t-tests) across multiple random calibration seeds.
   * **The Risk**: Without showing performance over multiple random runs (e.g., 5 or 10 seeds), the reported +2.40% gain of RB-DCTM over Globally-Scaled could be an artifact of a single lucky calibration split. The lack of rigorous statistical validation is a major scientific weakness, especially under a 10-shot regime.

2. **The "Dual-Dataset" Footprint for ZipIt!**:
   The real-world validation relies on ZipIt! to align expert coordinates before merging. The authors disclose that ZipIt! requires an additional, unlabelled calibration footprint of **100 samples per task** to estimate stable covariance matrices and prevent rank-deficiency.
   * This means the pipeline is not truly "10-shot" (20 samples total). It actually requires a footprint of **220 samples** (200 unlabelled samples for coordinate alignment, plus 20 labelled samples for trajectory optimization).
   * This dual-dataset footprint is a significant practical bottleneck that is glossed over in the main narrative of sample efficiency.

3. **Comparison Against Unmerged Experts (The Pragmatic Gap)**:
   The unmerged expert models achieve accuracies of **89.50%** (CIFAR-10) and **71.20%** (CIFAR-100), averaging **80.35%**. 
   * The best merged model (RB-DCTM, F=2) achieves **74.90%**, representing a **5.45% absolute accuracy drop** compared to the unmerged models.
   * If a user actually has the resources to fine-tune experts, they would likely prefer to run ensembling (keeping the experts separate) to preserve the full 80.35% average performance. Weight merging is only justified if the inference-time cost of running two models is prohibitive, but suffering a 5.45% accuracy penalty is a high price to pay.
