# Peer Review: Riemannian Curvature-Regularized Test-Time Model Merging (RCR-Merge)

## 1. Summary of the Paper
The paper addresses a critical challenge in adaptive, test-time model merging (TTA). direct parameter-space merging of task-specific expert models (which are fine-tuned from a shared pre-trained base model $\theta_0$) has emerged as a computationally efficient alternative to multi-task training from scratch. Recent adaptive model-merging methods like *AdaMerging* introduce learnable layer-wise merging coefficients $\boldsymbol{\lambda}$ that are optimized online during deployment using gradient-based minimization of predicted entropy on unlabeled, local test data streams.

The authors identify and define a fundamental failure mode of this approach, termed the **Overfitting-Optimizer Paradox**: unconstrained unsupervised optimization of merging coefficients online is highly susceptible to transductive noise and local distribution shifts. To minimize entropy on local streams, the optimizer introduces high-frequency spatial oscillations in merging coefficients across network depth. Under the physical laws of representation learning, these uncoordinated coefficient fluctuations across adjacent layers disrupt the internal representation manifold of the model, leading to catastrophic representation collapse and a severe drop in generalization performance (often performing worse than simple static uniform baselines).

To resolve this paradox, the paper proposes **Riemannian Curvature-Regularized Test-Time Model Merging (RCR-Merge)**. The core philosophy is to model the deep neural network parameter space as a Riemannian manifold where distance is locally scaled by the second-order curvature of the loss landscape, rather than a flat, isotropic Euclidean surface.

The practical execution of RCR-Merge consists of a three-step sequential pipeline:
1. **Offline Base Curvature Estimation**: The diagonal trace of the Fisher Information Matrix (FIM) for each layer of the pre-trained base model $\theta_0$ is estimated offline using a minuscule calibration batch (e.g., $|D_{\text{cal}}| = 64$ samples). The resulting curvature vector is normalized across depth to act as a spatial Riemannian metric.
2. **Online Test-Time Adaptation**: As unlabeled test data streams arrive, the merging coefficients $\boldsymbol{\lambda}$ are optimized online via entropy minimization.
3. **Riemannian Curvature-Weighted Total Variation (RCR-TV) and Absolute Anchoring**: The joint optimization objective includes a spatial regularizer that penalizes the squared difference of merging coefficients between adjacent layers, scaled by the geometric mean of their pre-trained base curvatures ($\sqrt{c_l c_{l-1}}$). To prevent joint-drift (where all coefficients shift in tandem under correlated noise), an absolute coordinate anchoring penalty ($\gamma \|\boldsymbol{\lambda} - \boldsymbol{\lambda}_0\|_2^2$) is also added.

Additionally, the authors propose **Gradient Norm Balancing (GNB)** to dynamically initialize the spatial regularization strength $\beta$ at step 0 in a fully unsupervised manner. By evaluating the gradient of the regularizer at a worst-case spectral perturbation (alternating signs representing maximum spatial frequency), GNB standardizes the optimization coordinates to a unit perturbation sphere and calculates a scale-invariant ratio of the initial loss gradient to the regularizer's peak sensitivity.

---

## 2. Strengths and Weaknesses

### Strengths
1. **Pioneering Problem Formalization**: The paper clearly defines and formalizes the **Overfitting-Optimizer Paradox** in adaptive model merging, identifying a critical vulnerability of online entropy minimization that leads to catastrophic representation collapse.
2. **Deep Mathematical Grounding**: The proposed framework is supported by a rich and exceptionally complete theoretical foundation:
   - *Lemma 3.1 (Coordinate-Level Barrier)* proving curvature-guided spatial TV limits coefficient variation in sensitive regions.
   - *Theorem 3.2 (Activation-Level Drift Bound)* establishing a causal link between coordinate variations and global representation drift.
   - *Spectral Graph Theory* showing that RCR-Merge acts as a physical Laplacian low-pass filter filtering out transductive noise.
3. **Fully Unsupervised Scale-Invariance (GNB)**: Proposes Gradient Norm Balancing, a mathematically principled scale-invariant method to dynamically scale regularization strength $\beta$ at step 0 without requiring ground-truth labels.
4. **Computational and Memory Efficiency**: By utilizing offline, pre-computed diagonal Fisher trace approximations, RCR-Merge operates on a conformal flat subspace during adaptation, keeping computational overhead to $O(1)$ and storage to $O(L)$ (less than 128 bytes of metadata), making it edge-friendly.
5. **Outstanding Presentation and Clarity**: Extremely well-written, containing a clear conceptual schematic (Figure 1), detailed algorithm block (Algorithm 1), and a standalone PyTorch recipe in the appendix.

### Weaknesses
Despite its mathematical beauty and excellent writing, the paper exhibits several critical empirical weaknesses that must be addressed before acceptance:

1. **Predominant Reliance on a Handcrafted Synthetic Simulator**:
   The primary quantitative evaluations (Table 1, Figure 2, Table 2, Table 4) are performed entirely on a synthetic **Coupled Model II Landscape** emulator. While controlled simulations enable high-throughput statistical validation (30 seeds) and causal variables isolation, a synthetic emulator is an idealized, low-dimensional mathematical toy model. A 12-layer, 4-task emulator does not capture the complex, highly non-convex loss landscapes of real deep networks. The transferability of these simulated results to actual deep learning systems remains highly questionable.

2. **Toy Scale and Lack of Rigor in Real-World Pilot Studies**:
   To address the simulation-only constraint, the authors describe two real-world pilot studies (BERT-Base and ViT-B/16). While these are welcome additions, their execution lacks scientific rigor:
   - *BERT-Base Pilot Study*: The authors fine-tune sentiment classification and topic classification experts, but evaluate on a tiny simulated stream with simulated expert perturbations. They report a "perfect 100.00% accuracy across both tasks" under RCR-Merge. In real-world NLP, achieving a perfect 100% accuracy is virtually impossible, suggesting that the evaluation was performed on a trivial, small, or heavily sanitized subset of data, which lacks empirical credibility.
   - *ViT-B/16 Pilot Study*: This experiment is executed in "less than 15 seconds on a standard CPU," indicating that it was likely run on only a single batch of size 16 or a tiny handful of samples. This is far from a rigorous empirical validation.
   - *Complete Absence of Statistical Significance*: While the authors run 30 seeds on the synthetic simulator, they report zero statistical significance, zero standard deviations, and zero random seeds for the BERT and ViT pilot studies. This makes it impossible to know if the observed stabilization is statistically sound on real models or just a lucky run.

3. **Absence of Standard Out-of-Distribution (OOD) Benchmarks**:
   Standard test-time adaptation and adaptive model-merging papers (e.g., AdaMerging, PolyMerge, Tent) evaluate their methods on widely accepted, large-scale OOD benchmarks. For vision, this includes **ImageNet-C**, **ImageNet-R**, or **DomainNet** streams. For language, this includes **GLUE** or **MMLU** streams under temporal drift. The complete lack of evaluations on these standard benchmarks is a major empirical gap. Without running on actual standard benchmark datasets, we cannot verify if RCR-Merge actually delivers robust performance under real distribution shifts.

4. **Incomplete Baselines Comparison in Real-World Settings**:
   On BERT-Base and ViT-B/16, the authors *only* compare RCR-Merge against unconstrained AdaMerging. They completely omit PolyMerge and TV-Regularized AdaMerging. This is a critical omission. Without comparing RCR-Merge against flat spatial Total Variation on real architectures, we cannot verify if the proposed *curvature-weighting* ($\sqrt{c_l c_{l-1}}$) actually provides any real-world benefit over isotropic smoothing, or if the stabilization is driven purely by flat Total Variation.

---

## 3. Ratings

- **Soundness**: **Fair**
  The theoretical foundation is excellent and mathematically sound. However, the experimental support for real-world claims is currently weak. The reliance on a synthetic simulator and toy-scale CPU pilot studies with suspicious 100% accuracy does not meet the standards of a top-tier machine learning publication.
- **Presentation**: **Excellent**
  The paper is exceptionally well-written, clear, and structured. The visual schematics, algorithmic blocks, and PyTorch recipes are highly professional.
- **Significance**: **Good**
  Parameter-space model merging is highly relevant and growing rapidly in popularity. RCR-Merge could have broad impact if its empirical claims are validated on real benchmarks.
- **Originality**: **Excellent**
  The problem formalization (Overfitting-Optimizer Paradox), the curvature-weighted spatial TV, the GNB initialization, and the threshold-triggered local charting are highly novel and creative contributions.

---

## 4. Overall Recommendation

**Overall Recommendation**: **3: Weak Reject**

The paper presents an exceptionally well-designed, mathematically rigorous, and novel second-order geometric framework for online test-time model merging. However, because modern machine learning is heavily empirical, the lack of a rigorous, multi-seed evaluation on standard, real-world benchmarks (such as ImageNet-C/R or GLUE/MMLU streams) is a critical limitation that outweighs the paper's theoretical merits in its current form. 

The small, toy-scale CPU pilot studies with reported "perfect 100% accuracy" are insufficient to support the high-dimensional real-world deployment claims. If the authors can conduct a rigorous evaluation on standard datasets, compare RCR-Merge against all baselines with proper statistical significance (standard deviations and random seeds), and confirm that curvature weighting outperforms flat Total Variation on real architectures, this paper would easily transition to a **Strong Accept**.

---

## 5. Constructive Comments and Questions for the Authors

To help the authors improve their work, I provide the following actionable and constructive steps required for acceptance:

1. **Evaluate on Standard OOD Benchmarks**:
   Evaluate RCR-Merge and all baselines on standard real-world streaming benchmarks:
   - *For Vision*: Merge ViT-B/16 experts fine-tuned on different tasks (e.g., DomainNet or PACS) and adapt them on incoming streams under corruptions (ImageNet-C) or domain shifts (ImageNet-R).
   - *For Language*: Merge BERT or LLaMA-7B task-specific experts and adapt online on streaming GLUE or MMLU benchmarks.
2. **Provide Statistical Significance on Real Models**:
   Execute the BERT and ViT experiments over at least 5 to 10 independent random seeds. Report the average multi-task accuracy along with standard deviations and confidence intervals, matching the empirical rigor of the synthetic simulator results.
3. **Compare Against All Baselines on Real Models**:
   Include **TV-Regularized AdaMerging** and **PolyMerge** in the real-world BERT/ViT experiments. Verify and discuss whether RCR-Merge's curvature-guided spatial TV ($\sqrt{c_l c_{l-1}}$) outperforms flat spatial TV on actual neural architectures.
4. **Conduct a Sensitivity Study on Calibration Size**:
   Provide a sensitivity study or analysis evaluating how robust the estimated base curvatures $\bar{c}_l$ are to the size and sample choice of the calibration batch $D_{\text{cal}}$. Demonstrate that $|D_{\text{cal}}| = 64$ is stable and robust to sample sampling variance.
5. **Analyze the Universality of the "U-shaped" Sensitivity**:
   The Coupled Model II Landscape simulator assumes highly sensitive early and late layers and robust middle layers. Is this "U-shaped" sensitivity profile universally true across different architectures (e.g., CNNs vs. Transformers) and tasks? Providing empirical FIM trace plots of real networks across depth to validate this assumption would greatly strengthen the paper's scientific foundation.
