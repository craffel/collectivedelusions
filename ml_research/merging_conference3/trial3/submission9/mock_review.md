# Peer Review Report: FlatQ-Merge (Flatness-Aware Quantization-Aware Model Merging)

---

## 1. Summary of the Paper

This paper presents **FlatQ-Merge**, a rigorous, multi-axial empirical study that investigates the relationship between the loss landscape flatness of task-specific expert models and their subsequent resilience to post-training quantization (PTQ) and test-time blending coefficient optimization. 

To systematically address this question, the authors fine-tune expert models using Sharpness-Aware Minimization (SAM) across 5 different perturbation radii ($\rho \in \{0.0, 0.01, 0.05, 0.1, 0.2\}$). These experts are merged using a layer-wise dynamic blending scheme and compressed to 8-bit or extreme 4-bit precision via symmetric uniform PTQ. Blending coefficients are optimized at test-time under an unsupervised joint prediction entropy objective via the Straight-Through Estimator (STE). 

The paper uncovers a powerful, precision-dependent **Flatness-Robustness Synergy**: under standard 8-bit precision, standard SGD-trained experts are robust and flatness yields negligible benefits; however, under extreme 4-bit precision, pre-training with an optimal SAM radius ($\rho=0.05$) yields a massive **+7.44%** absolute multi-task accuracy gain. Strikingly, the authors show that a simple uniform merge on flat experts outperforms sophisticated test-time coefficient optimization on standard sharp experts by **+6.03%** absolute accuracy, proving that pre-merging landscape geometry is significantly more critical to quantized merging success than downstream adaptation complexity. They also identify a critical non-linear **Over-Perturbation Threshold** ($\rho \ge 0.1$) where performance collapses due to "representation convergence," backed by pairwise cosine similarity and task vector analyses.

---

## 2. Key Contributions

- **Novel Geometric Link**: Establishes and mathematically derives the relationship between weight-space and coefficient-space flatness: $H_{\Lambda} = T^T H_{\theta} T$. This proves that pre-training experts via SAM to minimize weight Hessian spectral norms directly bounds and flattens both the trace and spectral norm of the test-time coefficient Hessian, guaranteeing smooth optimization dynamics.
- **Exhaustive Multi-Axial Evaluation**: Conducts extensive sweeps across 5 SAM radii, 2 quantization levels, and 3 independent random seeds, providing exceptionally reliable empirical findings.
- **Uncovering Core Design Rules**: Demonstrates that pre-merging expert geometry (flatness) dominates downstream test-time adaptation, providing a simple yet highly potent design recipe for practitioners (pre-train with SAM $\rho=0.05$ for extreme 4-bit merging).
- **Geometric Collapse Explanation**: Identifies and characterizes the over-perturbation threshold ($\rho \ge 0.1$) where excessive SAM perturbations trigger "representation convergence," causing task-specific experts to lose their specialized features and align with the same scarce wide basins of the pre-trained base model.
- **Comprehensive Baseline Comparison**: Evaluates and compares the proposed framework against a wide array of competitive baselines, including SGD, NaiveUniform, AdaMerging-PostQ, SWA, DARE, and high-dimensional TENT-style adaptation.

---

## 3. Assessment of Strengths and Weaknesses

### Strengths

1. **Outstanding Theoretical Rigor**: The mathematical derivation connecting weight-space Hessians to coefficient-space Hessians is exceptionally elegant and correct. Furthermore, the characterization of the quantized loss landscape as piecewise-constant demonstrates high intellectual maturity and honesty, avoiding standard continuous approximations where they are mathematically invalid.
2. **Exceptional Empirical Completeness**: The paper goes far beyond standard empirical evaluations. Sweeping across 3 independent random seeds with mean and standard deviation for all entries is highly commendable. Additionally, implementing multiple orthogonal baselines (SWA, DARE, TENT-style weight adaptation, independent clipping vs. Softmax, direct weight-space flatness measurements) isolates the exact mechanisms of the observed phenomena.
3. **High Presentation Quality**: The paper is exceptionally well-written, clearly structured, and easy to follow. The "Limitations and Scope" section is exemplary in its transparency, proactively addressing potential criticisms (scale of backbone, weight-only quantization, absolute accuracy gaps).
4. **Actionable Systems-Level Insights**: Uncovering the peak-memory trade-offs of direct quantized optimization (FlatQ-Merge) over post-hoc unquantized optimization (AdaMerging-PostQ) provides highly valuable, practical guidelines for deploying merged models onto RAM-constrained edge microcontrollers.

### Weaknesses & Areas for Improvement (Constructive Feedback)

While this is an exceptionally high-quality submission, there are three primary limitations that the authors should address to make the findings even more compelling:

1. **Scale of Backbone and Datasets**:
   The empirical sandbox is restricted to a very small Vision Transformer backbone (`vit_tiny_patch16_224`, 5.7M parameters) fine-tuned on a tiny budget of 512 images per task. Consequently, the absolute multi-task accuracies are relatively low (e.g., individual unquantized experts achieve ~64.28% on this budget). While this simulates extremely low-resource edge deployment, evaluating on standard, larger vision backbones (e.g., ViT-B or ResNet-50) or showing preliminary scaling to small autoregressive LLMs (e.g., LLaMA-1B/3B) would make the empirical findings much more compelling and representative of mainstream model-merging practice.
   
2. **Exclusion of Joint Weight-Activation Quantization**:
   The evaluation focuses strictly on weight-only post-training quantization (W8A32, W4A32). Physical edge hardware typically requires integer-only arithmetic, which necessitates joint weight-activation quantization (e.g., W8A8 or W4A4). Extending the framework to evaluate joint quantization, and discussing how flatness suppresses activation outliers, would bridge the gap between simulation and hardware deployment.
   
3. **Artificial Task Combination**:
   Merging MNIST, FashionMNIST, CIFAR-10, and SVHN onto a single backbone is a highly artificial and heterogeneous multi-task combination. The massive domain shift between these datasets introduces severe parameter interference (as shown by the large performance gap between individual experts and merged models). It would be valuable to evaluate on a more aligned and standard domain-merging setting (e.g., DomainNet or Office-Home experts fine-tuned on different sub-domains) where parameter interference is naturally lower.

---

## 4. Ratings and Recommendation

- **Overall Recommendation**: **5: Accept**  
  *Justification*: This is an exceptionally polished, mathematically sound, and empirically complete paper. The paper introduces deep geometric insights into model merging in low-precision weight spaces and supports all claims with overwhelming empirical evidence. The minor limitations (such as scale and weight-only quantization) are proactively and transparently discussed by the authors, and the paper represents a substantial contribution that the community will build upon.
- **Soundness Rating**: **Excellent**  
  *Justification*: The claims are theoretically backed by sound mathematical derivations, the optimization dynamics under Straight-Through Estimator and low-dimensional search spaces are clearly analyzed, and the empirical results are statistically validated across 3 random seeds.
- **Presentation Rating**: **Excellent**  
  *Justification*: The paper is beautifully written, clearly structured, and highly transparent. The figures provide immediate visual context, and the limitations are discussed with an exemplary level of intellectual honesty.
- **Significance Rating**: **Good**  
  *Justification*: The paper addresses an important and timely problem (low-precision deployment of merged models). The insights provided (such as the dominance of pre-merging geometry and the over-perturbation threshold) are highly valuable and will guide future research.
- **Originality Rating**: **Excellent**  
  *Justification*: Connecting pre-merging expert flatness to quantized model merging resilience, deriving the weight-to-coefficient Hessian projection, and characterizing the over-perturbation threshold via representation convergence are highly original contributions.

---

## 5. Detailed Questions and Suggestions for the Authors

1. **Generalization to Joint Quantization**: In your future work discussion, you mention that SAM-induced flatness might suppress activation outliers by bounding the Lipschitz constants of layers. Have you conducted any preliminary experiments with joint weight-activation quantization (such as W8A8)? Can you elaborate on how much activation clipping noise is mitigated under flat experts?
2. **LLM Feasibility**: You mention that LLM post-training quantization (such as AWQ or GPTQ) suffers from extreme activation outliers. How feasible is it to incorporate SAM-like objectives into the instruction-tuning or pre-training phases of 1B-7B parameter autoregressive models? Would the computational overhead of SAM ($2\times$ forward/backward passes) be a significant blocker at that scale?
3. **Out-of-Distribution Sensitivity of Test-Time Adaptation**: The joint prediction entropy minimization is performed on a balanced, unlabeled calibration batch of 64 images. How sensitive is the test-time adaptation to extreme distribution shifts or class imbalances in the calibration batch (e.g., if the calibration batch consists entirely of CIFAR-10 images with zero SVHN/MNIST images)? Does the implicit structural regularization of the 56-parameter coefficient matrix still protect the model from task collapse under such extreme scenarios?
4. **Visualizations**: The curvature profiling table (Table 3 in Appendix B / Figure 1c) shows that expected prediction entropy changes under perturbation are extremely small and fluctuate between positive and negative values. While your piecewise-constant landscape explanation is highly convincing, it would be valuable to show a 2D visualization or contour plot of the coefficient-space loss landscape around the optimized coefficients $\Lambda^*$ to make this step-like geometry visually intuitive for readers.
