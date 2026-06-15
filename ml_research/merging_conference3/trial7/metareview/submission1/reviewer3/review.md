# Peer Review

## Summary of the Paper
The paper investigates weight-space dynamic model merging, which interpolates task-specific neural networks on the fly during inference using sample-specific coefficients computed by an input-dependent routing network. The core goal of the work is to deconstruct a recent theoretical claim of "Layer-Averaging Collapse" (rank-1 collapse), which asserted that learned layer-wise routing trajectories must inevitably become perfectly collinear across layers, rendering layer-wise dynamic routing redundant. 

To address this, the authors establish an empirical framework on Split-MNIST digit subsets using two small architectures: DeepMLP-12 (a 12-layer MLP) and TinyCNN-4 (a 4-layer CNN). They propose a Bounded Sigmoid (BSigmoid) routing pipeline using frozen random Gaussian projections and introduce spectral diagnostics—specifically, Singular Value Decomposition (SVD) of the Batch-Averaged Layer-wise Coefficient Matrix and inter-layer pairwise cosine similarity—to analyze routing dimensionality. Based on their evaluations, the authors argue that the rank-1 collapse is an artifact of low-conflict, simplified representation sandboxes, showing that routing trajectories occupy a multi-dimensional subspace under severe task conflict. They also highlight a fundamental systems-level "Batch-Averaged Multi-Task Inference Paradox" and discuss regularization/optimization trade-offs under scarce few-shot calibration budgets.

---

## Strengths
1. **Exceptional Clarity and Presentation:** The paper is exceptionally well-written, logically structured, and mathematically detailed. The narrative is easy to follow, equations are formatted cleanly, and terms are defined precisely.
2. **Outstanding Intellectual Honesty:** The authors are highly commended for their extreme transparency regarding the limitations of their work. They do not shy away from exposing the "Normalization Paradox" of their router, the "Batch-Averaged Multi-Task Inference Paradox" (which identifies when dynamic merging collapses back to static compromise), and the "Random Guessing Barrier" where their MLP merged model fails catastrophically. This candor is refreshing and highly scholarly.
3. **Methodological Rigor in Evaluation:** The paper performs multi-seed evaluations across 5 independent random seeds and reports means and standard deviations, ensuring that results are statistically stable and robust to initialization.
4. **Systems-Level Depth:** The analysis of memory-bandwidth bottlenecks (Section 8.1) and the preliminary proof-of-concept for PEFT-level dynamic merging on Vision Transformers are valuable and highly insightful contributions to the systems literature of model serving.

---

## Weaknesses

### 1. Fundamental Mathematical Flaw in the SVD Collinearity Ratio
The core diagnostic of the paper is the SVD Collinearity Ratio ($\rho_{collinear}$), computed on the Batch-Averaged Layer-wise Coefficient Matrix $A \in \mathbb{R}^{L \times K}$ defined as:
$$A_{l, k} = \frac{1}{B} \sum_{b=1}^B \lambda_{l, k}(x_b)$$
However, under a balanced, heterogeneous test stream containing an equal mixture of all task samples, **any well-calibrated, symmetric dynamic router must mathematically converge to a constant uniform matrix as the batch size $B \to \infty$**:
$$\lim_{B \to \infty} A_{l, k} = \frac{1}{K} \quad \forall l, k$$
Because a constant matrix where all elements equal $1/K$ has exactly rank 1, **its SVD Collinearity Ratio $\rho_{collinear}$ must mathematically converge to 1.0 (perfect collapse), regardless of how dynamic, sample-specific, and non-collinear the routing actually is for individual samples.**
* **The Implication:** A perfect, 100%-accurate sample-specific router on a balanced mixed batch will yield a collinearity ratio of exactly 1.0 (perfect collapse). Conversely, a lower collinearity ratio ($\rho_{collinear} < 1.0$) can only be obtained if the batch-averaged coefficients differ across layers, which occurs if and only if the router possesses a *systematic, input-independent layer-wise bias* (e.g., early layers always route to Expert 0, deep layers always route to Expert 1, regardless of the input sample).
* **The Conclusion:** Therefore, the SVD Collinearity Ratio on the batch-averaged matrix is mathematically incapable of measuring dynamic routing capacity. A lower collinearity ratio actually indicates a *static layer-wise bias* (making the routing sample-independent and static but layer-dependent) rather than true dynamic, sample-specific specialization. This invalidates the authors' core spectral diagnostic method and deconstruction claims.

### 2. Extreme Reliance on Over-Simplified Toy Sandboxes
The authors critique prior work for relying on "over-simplified, linear representation-space sandboxes." However, they replace it with another toy sandbox:
* **Dataset:** Experiments are conducted on **Split-MNIST**, a grayscale handwritten digit dataset that is trivial, low-dimensional, and lacks the semantic hierarchies, textures, and rotational variances of real-world datasets.
* **Architectures:** The backbones are miniature toy models—**DeepMLP-12** (64 hidden units, ~100k parameters) and **TinyCNN-4** (3 conv layers, 1 linear layer). 
These models are far removed from modern high-capacity architectures (e.g., ResNets, ViTs, LLMs) where model merging is actually utilized. Drawing general architectural conclusions about "hierarchical representation space" and "semantic depth-specialization" from a 12-layer MLP with 64 hidden units is highly speculative.

### 3. Empirical Dominance of Simple Static Baselines
On the convolutional backbone (TinyCNN-4) in Table 2, the **simple static, 4-parameter baseline OFS-Tune consistently and significantly outperforms the proposed high-capacity dynamic Layer-wise Router across all three task-conflict suites**:
* *Low-Conflict:* OFS-Tune **82.85% ± 11.52%** vs. Layer-wise **78.70% ± 14.56%** ($+4.15\%$ delta).
* *High-Conflict:* OFS-Tune **90.75% ± 1.58%** vs. Layer-wise **81.30% ± 9.69%** ($+9.45\%$ delta).
* *Cross-Domain:* OFS-Tune **53.40% ± 7.16%** vs. Layer-wise **52.52% ± 5.95%** ($+0.88\%$ delta).
The authors argue that scaling the data budget to 1024 samples per task (Figure 4) allows the dynamic router to finally cross over. However, the improvement is a marginal ~1% and requires large calibration datasets, which directly violates the "few-shot" calibration premise. Introducing $L \times (d \cdot K + K)$ parameters and complex dynamic gating for near-zero or negative utility under the target few-shot regime indicates a severe lack of practical value.

### 4. Analysis of Non-Functional Merged MLP Models
On DeepMLP-12 under Cross-Domain task conflict (Table 1), the proposed dynamic router achieves **16.15% ± 5.60%** accuracy, while standard Uniform gets **11.80%**, and L1-Global gets **11.68%**. While their method is "statistically superior," all merged models perform below or near the **12.5% random guessing threshold** for the 8-class subset. 
A classification accuracy of 16% on digit recognition is a complete and catastrophic functional collapse. The weights are scrambled, and coordinate projections are destroyed. Conducting a spectral analysis on the routing coefficients of a completely broken, non-functional model to draw major conclusions about "the emergence of depth-specialized routing policies as a semantic necessity" is methodologically meaningless, as the router is merely learning to navigate random optimization noise.

### 5. Flawed Dismissal of Strong Baselines
The authors do not compare against prominent weight-space alignment and merging baselines like **ZipIt!** or **TIES-Merging**. They claim that because the experts are fine-tuned from a shared initialization, these advanced alignment techniques "mathematically collapse to standard arithmetic interpolation." 
* **The Flaw:** This is false. TIES-Merging trims parameter updates based on magnitude (e.g., keeping only the top 20% of parameter deltas) and resolves sign conflicts. Even when starting from a shared base initialization, different experts will update parameters in different directions, and trimming small parameter changes can remove optimization noise. Thus, TIES-Merging does *not* collapse to standard averaging and should have been included as a baseline. Bypassing these methods represents a significant baseline omission.

---

## Detailed Comments and Questions for the Authors
1. **Redefine the SVD Diagnostic:** To mathematically validate the true dimensionality of the routing space, you must perform SVD on the *sample-specific* routing matrices $A(x) \in \mathbb{R}^{L \times K}$ directly rather than the batch-averaged matrix. Please report the sample-wise collinearity ratio:
   $$\rho_{collinear}(x) = \frac{\sigma_1(A(x))}{\sum_{i} \sigma_i(A(x))}$$
   and then average this ratio across the test set: $\bar{\rho}_{collinear} = \frac{1}{B} \sum_{b=1}^B \rho_{collinear}(x_b)$. This will correctly capture whether individual samples utilize multi-dimensional routing trajectories.
2. **Incorporate TIES-Merging and ZipIt!:** Please run empirical evaluations comparing your method against TIES-Merging and ZipIt! to demonstrate whether they actually "collapse" or if their parameter pruning/alignment properties provide a superior baseline on your task suites.
3. **Scale up the Empirical Framework:** To prove your claims about "depth-specialized routing policies," you must scale beyond Split-MNIST. Please evaluate a ResNet-18/50 or ViT-B/16 on more complex multi-task suites (e.g., CIFAR-100, Oxford Flowers, or Stanford Cars) under your dynamic merging pipeline.

---

## Ratings

### Soundness: Fair
While the paper's mathematical formulation of equations and router implementations are correct, the soundness is rated as "fair" due to:
* The fundamental mathematical flaw in their primary spectral diagnostic ($\rho_{collinear}$ on batch-averaged matrix $A$).
* Drawing major conclusions about architectural spatial routing on fully collapsed, non-functional models (DeepMLP-12 scoring near random guessing).
* Omission of key alignment baselines.

### Presentation: Excellent
The writing, structure, figures, and clarity of the paper are exceptional. The authors are incredibly honest about their work's limitations, which makes the paper highly readable and easy to follow.

### Significance: Fair
The significance is rated as "fair." While the theoretical discussion of the Batch-Averaged Paradox is highly valuable, the actual proposed dynamic merging pipeline underperforms simple static baselines (OFS-Tune) on CNNs, collapses on MLPs, and is evaluated only on Split-MNIST with tiny networks. Thus, its practical utility and impact on the machine learning community are highly limited in its current form.

### Originality: Good
The introduction of SVD diagnostics (once mathematically corrected to be sample-specific) and the systems-level analysis of memory-bandwidth overhead of dynamic merging represent creative and original contributions to the model merging literature.

---

## Overall Recommendation: 3: Weak Reject
The paper is an exceptionally well-written, honest, and mathematically polished submission. It identifies a crucial debate (Layer-Averaging Collapse) and introduces highly interesting systems insights (the Batch-Averaged Paradox). 

However, the core diagnostic of the paper is mathematically flawed, the empirical evaluation is limited to toy sandboxes (Split-MNIST on miniature networks), the proposed dynamic router underperforms a simple 4-parameter static baseline, and the MLP models suffer from complete representational collapse. The paper has clear merits, but these weaknesses outweigh them in its current form. The authors must address the mathematical formulation of their SVD metric and scale their experiments to standard backbones on realistic datasets before this work can be built upon by the community.
