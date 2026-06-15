# 4. Experimental Setup and Baseline Evaluation Check

## 4.1 Evaluation Rigor and Design
The experimental evaluation in this paper is outstandingly thorough and sets a very high standard for empirical research:
- **Statistical Significance**: All key experiments are evaluated across 3 independent random seeds (42, 100, 2026), and both the mean and standard deviation are reported.
- **Multidimensional Grid Sweep**: The authors systematically sweep multiple axes—5 SAM perturbation radii ($\rho \in \{0.0, 0.01, 0.05, 0.1, 0.2\}$), 2 quantization precisions (8-bit, 4-bit), and 4 core merging configurations (FlatQ-Merge, NaiveUniform, AdaMerging-PostQ, Individual-Quantized).
- **Extensive Ablation and Baseline Suites**: Beyond the core sweep, the paper includes:
  - **SWA Baseline**: Isolate SAM's worst-case adversarial flatness from trajectory average flatness (SWA).
  - **DARE Baseline**: Confirm compatibility with state-of-the-art sign-conflict and parameter-pruning techniques.
  - **TENT Baseline**: Compare low-dimensional coefficient adaptation against high-dimensional model weight adaptation.
  - **Softmax Normalization Baseline**: Validate the use of independent clipping bounds $[0, 1]$ over convex normalized schemes.
  - **Curvature Profiling**: Perturb coefficients to map the test-time loss landscape.
  - **Hessian Trace Proxy**: Measure weight-space loss increases under random weight perturbations to empirically verify Hessian curvature.

This level of empirical completeness is highly commendable.

## 4.2 Support for Claims
The central claims of the paper are fully and robustly supported by the experimental results:
- **Flatness-Robustness Synergy**: Table 2 clearly shows that under extreme 4-bit quantization, flat experts ($\rho=0.05$) achieve 30.44% (FlatQ) and 30.78% (Ada-PQ), representing a massive improvement over standard SGD experts ($\rho=0.0$, 23.00% and 24.16% respectively).
- **Dominance of Pre-Merging Geometry**: The NaiveUniform baseline with flat experts ($\rho=0.05$) achieving 29.03% (outperforming test-time optimized standard experts at 23.00% by +6.03% absolute accuracy) provides undeniable proof that pre-merging landscape conditioning is far more critical than the downstream optimization algorithm.
- **Over-Perturbation Collapse**: The rapid drop-off in accuracy beyond $\rho=0.1$ is clearly documented. The explanation of "representation convergence" is empirically supported by the pairwise cosine similarity increase of the task vectors.

## 4.3 Areas for Empirical Improvement (Constructive Criticism)
Despite the exceptional thoroughness of the evaluation, there are three main limitations that could be addressed to strengthen the paper:

1. **Backbone and Dataset Scale**:
   The experiments are conducted on a very small Vision Transformer (`vit_tiny_patch16_224`, 5.7M parameters) and use a tiny training budget of 512 images per task. This results in relatively low absolute accuracies (e.g., individual unquantized experts achieve ~64.28% on this budget). While this budget simulates extremely low-resource edge deployment, evaluating on standard full-scale datasets (such as full CIFAR-10 or ImageNet subsets) and using larger models (such as ViT-B or small LLMs like LLaMA-1B/3B) would significantly make the empirical findings more compelling and representative of mainstream model-merging practice.
   
2. **Weight-Activation Quantization**:
   The paper focuses exclusively on weight-only post-training quantization (W8A32, W4A32). Real-world edge deployment on microcontrollers or integer-only accelerators requires joint weight-activation quantization (e.g., W4A4 or W8A8) to avoid floating-point operations in the forward pass. Evaluating weight-activation PTQ and discussing how flatness suppresses activation outliers would bridge the gap between simulation and physical edge deployment.

3. **Artificial Task Composition**:
   Merging MNIST, FashionMNIST, CIFAR-10, and SVHN onto a single tiny backbone is a highly artificial and heterogeneous multi-task combination. The massive domain shift between these datasets introduces severe parameter interference (as shown by the large performance gap between individual experts and merged models). It would be valuable to evaluate on a more aligned and standard domain-merging setting (e.g., DomainNet or Office-Home experts fine-tuned on different sub-domains) where parameter interference is naturally lower.

## 4.4 Experimental Check Rating
- **Rating**: **Excellent**
- **Justification**: The experimental design is exceptionally thorough, the baselines are highly appropriate, and the statistical rigor (3 random seeds, mean and std reported for all entries) is outstanding. Every major claim is backed by overwhelming empirical evidence, and the authors go above and beyond by implementing multiple additional baselines (SWA, DARE, TENT, Softmax normalization, Hessian trace proxy) to isolate the exact mechanisms of their findings.
