# Mock Review: Cross-Attention Multi-Expert Routing for Dynamic Model Merging

## 1. Summary of the Paper
The paper introduces **Cross-Attention Multi-Expert Routing (CAM-Router)**, a dynamic model-merging framework for multi-task expert fusion on compact Vision Transformers. The paper identifies three key limitations of existing dynamic model-merging methods:
1. **Vulnerability to spatial occlusion:** caused by collapsing spatial token sequence representations immediately via global average pooling (GAP) before routing.
2. **Task heterogeneity collapse:** under mixed-task batching where average pooling across heterogeneous inputs dilutes domain-specific features and collapses coefficients back to static uniform fusions.
3. **Softmax competitive bottleneck:** which forces an artificial zero-sum competitive pressure among task-experts, restricting multi-expert concurrent activation.

To resolve these issues, CAM-Router retains the full un-pooled spatial token sequence from the early layers of the transformer backbone (resolving the "First-Block Paradox" via static early weights). It introduces trainable task-expert queries ($Q \in \mathbb{R}^{K \times D}$) that attend to patch tokens via Multi-Head Cross-Attention (MHCA), enabling localized and spatially adaptive feature extraction. It utilizes Softmax-free, independent Bounded Sigmoid gating to allow multi-expert activation, and proposes Decoupled Historical Gating (DHG) to smooth and stabilize coefficients across batched inference. It claims a Joint Mean Accuracy of **53.07%** across four disparate datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) on a compact 14-layer Vision Transformer backbone, alongside high resilience to spatial occlusions (maintaining **50.57%** accuracy at 80% patch masking) and heterogeneous batching (maintaining **54.30%** accuracy at $B=256$).

---

## 2. Key Strengths
* **Highly Relevant and Impactful Concept:** Exploring spatially aware, query-based routing for weight-space model merging is an exceptionally fresh and well-motivated direction. Moving away from rigid global average pooling is a valuable conceptual contribution.
* **Excellent Writing and Presentation Structure:** The paper is exceptionally well-written, polished, and structured. The narrative flow is engaging, the background information is clearly detailed, and the mathematical formulations are clearly stated and easy to follow.
* **Thorough Positioning and Literature Review:** The related work section is complete and clearly positions the proposed method relative to weight-space merging, Mixture-of-Experts (MoE) routing, and recent dynamic model-merging baselines (such as QWS-Merge and BSigmoid-Router).
* **Elegant Sequence Masking Mechanism:** The handling of masked tokens in cross-attention (filling masked indices with `-1e9` prior to Softmax) is an elegant, emergent architectural feature that mathematically explains why the model is robust to spatial occlusions.

---

## 3. Major Weaknesses and Critical Flaws (Identify Up to 3)

### Flaw 1: Synthetic "Simulation" Presented as Real-World ViT Experiments
The paper's narrative repeatedly claims that the method is evaluated on a real 14-layer compact Vision Transformer (`vit_tiny_patch16_224` with 5.7M parameters) backbone across physical datasets (MNIST, FashionMNIST, CIFAR-10, SVHN). 
However, an inspection of the source code (`run_experiments_new.py`) reveals that **no real models are trained, no real datasets are processed, and no physical model merging is ever performed.** 
Instead, the entire evaluation is run on a highly artificial, toy simulator where image features $H_0 \in \mathbb{R}^{B \times N \times D}$ are generated from random Gaussian noise with hardcoded task-specific masks. Class prediction accuracies are calculated using a closed-form algebraic formula based on a hardcoded "ceiling" value (`prob_correct = 0.1317 + (ceiling - 0.1317) * norm_score`).
While the authors briefly mention a "controlled coordinate-space sandbox" in Section 4, the overall paper framing is highly misleading to the reader. Reporting a "53.07% Joint Mean Accuracy" as if it were an emergent property of physically merged networks violates scientific transparency standards.

### Flaw 2: Evaluation Discrepancy & Structural Bias in the Batch Size / Task Heterogeneity Sweep
There is a severe structural bias in how the models are evaluated in Sweep 3 (Batch Size & Task Heterogeneity Sweep):
- **Physical Weight-Merging Constraint:** When deploying a merged model in weight-space on a GPU, weights are loaded once in memory, and the entire batch must be processed using those weights. Thus, a dynamic router must predict a single set of merging coefficients of shape `[K]` for the entire batch. Standard routers (like `BSigmoidRouter`) adhere to this constraint by average pooling their predicted coefficients over the batch dimension, returning a single coefficient vector of shape `[K]`.
- **Bypassing the Constraint:** In Sweep 3, `BSigmoidRouter` is evaluated under this physical constraint (average-pooled coefficients), causing its accuracy to collapse because the mixed-task batch averages out task-specific signatures. However, `CAMRouter` is evaluated with `return_sample_alphas=True`, which returns individual, sample-specific coefficients of shape `[B, K]`.
- **Unfair Comparison:** By evaluating `CAMRouter` using sample-specific coefficients and comparing it with a baseline restricted to batch-averaged coefficients, the authors introduce an unfair structural advantage. `CAMRouter` completely bypasses the physical batch weight-merging constraint. If `CAMRouter` were evaluated under the same physical constraint as `BSigmoidRouter` (returning a single averaged coefficient vector over the batch), it would also suffer from "heterogeneity collapse."
- **DHG Evaluation Omission:** While the authors propose "Decoupled Historical Gating (DHG)" in Section 3.3 as their core solution to batched inference in production, they do not actually evaluate `CAMRouter` using DHG in Sweep 3. Instead, they bypass it by using sample-specific alphas directly, leaving DHG completely unverified empirically under the physical batch-merging constraint.

### Flaw 3: Presentation and Integrity Concerns Regarding Speculative GPU Claims
Section 3.3 discusses custom Triton kernels and coefficient quantization caching compiled to resolve dynamic model-merging latency. The paper states: *"we develop custom Triton kernels that fuse the parameter summation and the linear projection operation"* and presents a *"Coefficient Quantization and Model Caching Design."*
However, **there is zero implementation of custom Triton kernels, quantization bins, or weight caches in the codebase**, and no profiling or latency benchmarks are reported in the experiments. Presenting unimplemented, speculative engineering designs as completed contributions is highly misleading and violates basic scientific reporting standards.

---

## 4. Minor Suggestions & Questions
1. **Training-Inference Discrepancy in DHG:** Section 3.3 states that during training/calibration, *"average-pooling is utilized over the small calibration batch as an efficient gradient-smoothing mechanism."* However, during inference, the router is evaluated using a sample-level EMA (DHG) that completely bypasses batch pooling. Training a model under one feature-aggregation distribution (batch average pooling) and deploying it under a different sequential distribution (historical EMA) represents a major methodological flaw that can lead to severe covariate shift.
2. **High Statistical Noise in Sandbox Evaluation:** The simulator evaluates models using a test set of only 100 samples per task averaged over only 3 seeds. Given this extremely small sample size, the standard error of the joint accuracy is extremely high ($\approx 2.9\%$). The authors make detailed claims and sweeps based on small fluctuations of $2\%$ to $4\%$ in accuracy (e.g., Sweep 1 and Sweep 5), which are well within standard statistical noise and do not represent robust optimization discoveries.
3. **Hyperparameter Inconsistency:** In Table 1, CAM-Router achieves **53.07%** Joint Mean Accuracy. According to Sweep 5, this corresponds to $\lambda_{wd} = 0.0$ or $\lambda_{wd} = 10^{-4}$. However, Sweep 5 marks $\lambda_{wd} = 10^{-3}$ as the **Default** (which achieves a lower **47.40%**). Why was a sub-optimal hyperparameter configuration reported as the main baseline results in Table 1?
4. **Design Inconsistency in Attention Heads:** Sweep 1 shows that $h=1$ attention heads achieves **56.73%**, outperforming the default $h=4$ configuration at **53.07%**. Why is the sub-optimal $h=4$ configuration designated as the default if a single head is better?

---

## 5. Actionable Recommendations for Authors
To elevate this work to a publishable standard, the authors should undertake the following revisions:
1. **Implement on Real Models:** Re-run the entire evaluation suite on actual pre-trained Vision Transformers (e.g., `vit_tiny_patch16_224` from `timm`) on actual datasets (MNIST, FashionMNIST, CIFAR-10, SVHN). Train the routing parameters on a real calibration set, physically merge the model weights, and report the actual test accuracies.
2. **Remove or Validate Speculative GPU Claims:** Either fully implement the custom Triton kernels and coefficient quantization caching, providing comprehensive GPU latency and memory-bandwidth benchmarks, or move Section 3.3's latency discussion to a "Discussion / Future Work" section, framing it clearly as a conceptual proposal rather than an existing contribution.
3. **Resolve the Structural Evaluation Discrepancy:** In Sweep 3, evaluate both `CAMRouter` and `BSigmoidRouter` under the same physical constraint (either both use batch-averaged coefficients of shape `[K]`, or both are evaluated under a sequential sample-by-sample mode). Compare the actual EMA-smoothed DHG mode of `CAMRouter` directly against an EMA-smoothed version of the baseline to verify if DHG provides a genuine, fair performance improvement.
4. **Be Honest About the Simulator:** If the authors choose to keep the simulator, they must rename the paper to clearly indicate its theoretical/simulated nature (e.g., "A Simulation Analysis of Spatial Cross-Attention Routing...") and remove all claims of having run on a real ViT backbone, clearly framing the simulator as a toy sandbox rather than a "high-fidelity" evaluation.

---

## 6. Rating and Decision
**Overall Recommendation:** **2: Reject**
**Soundness:** **Poor** (Due to evaluation discrepancy, synthetic simulation framing, and training-inference discrepancy)
**Presentation:** **Good** (Well-written, but misleading in framing and speculative contributions)
**Significance:** **Poor** (Synthetic proxy does not generalize to real parameter-space manifolds)
**Originality:** **Good** (Concept of spatially aware query routing is highly interesting and novel)
