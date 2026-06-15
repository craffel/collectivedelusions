# Peer Review of GranMerge: Deconstructing the Generalization-Granularity Trade-off in Adaptive Model Merging

---

## 1. Summary of the Paper
This paper investigates **multi-task model merging** under test-time adaptive conditions. It addresses a fundamental and previously unstudied structural question: *At what level of physical/structural granularity should merging coefficients be defined and optimized, and how does this choice affect multi-task generalization?*

To answer this, the authors introduce **GranMerge**, a unified empirical framework designed to dissect the **Generalization-Granularity Trade-off** across five nested levels of parameter resolution (Level 1 Global to Level 5 Tensor-wise) across four visual classification tasks (MNIST, FashionMNIST, CIFAR-10, SVHN) using a 12-layer Vision Transformer backbone. The optimization of these merging coefficients is conducted on a small, unlabeled calibration stream ($N=256$) at test-time using an unsupervised surrogate loss (prediction entropy). 

The paper systematically analyzes two distinct optimizer families: first-order gradient descent (Adam) and zero-order black-box optimization (1+1 Evolution Strategies). It also evaluates the stabilization effects of two soft L2 regularizers: Elastic Spatial Regularization (ESR) and Depth-wise Total Variation (TV) smoothness.

### Key Findings:
1. **The Generalization-Granularity Trade-off:** Coarse-grained merging (L1 Global) leads to low performance due to underfitting. Intermediate granularities (L2 Layer-wise to L4 Component-wise) offer stable improvement. Finer-grained, unregularized optimization (L5 Tensor-wise) suffers from severe **transductive overfitting** on the compact local calibration batch ($N=256$), which exploits local entropy and degrades global multi-task generalization.
2. **Optimizer Overfitting Trajectories:** First-order Adam is highly vulnerable to rapid generalization collapse. In contrast, zero-order 1+1 ES maintains robust performance, which the authors deconstruct as a combination of *isotropic implicit regularization* and *optimization sluggishness (underfitting)* in high-dimensional spaces.
3. **Regularization Performance:** ESR and TV soft regularizers successfully stabilize 1+1 ES (recovering Level 5 performance close to the baseline) but are insufficient to arrest the unconstrained updates of first-order Adam.
4. **Surrogate Loss Misalignment and Static Baseline Superiority:** Under small calibration budgets, no test-time adaptive merging configuration (even when regularized) outperforms the zero-overhead, static **Uniform Task Arithmetic baseline (30.41%)**. The authors attribute this to a fundamental misalignment where minimizing prediction entropy produces "confident but incorrect" predictions on transductive noise.

---

## 2. Strengths of the Paper
1. **High Intellectual Honesty and Diagnostic Depth:** The paper is exceptionally candid about its "negative results" (i.e., that none of the adaptive configurations beat the zero-overhead static uniform baseline). Rather than trying to obscure this outcome, the authors embrace it as a key diagnostic finding, leading to an insightful deconstruction of surrogate loss misalignment.
2. **Unified Taxonomic Continuum:** By defining and nesting five distinct levels of parameter resolution (Global, Layer-wise, Block-wise, Component-wise, and Tensor-wise), the paper brings excellent clarity to a previously fragmented literature where merging granularities were chosen heuristically.
3. **Rigorous Analysis of Optimizer Behavior:** The dual evaluation of first-order and zero-order optimizers is highly rigorous. The deconstruction of zero-order 1+1 ES's robustness into the "sluggishness hypothesis" (underfitting near initialization due to the curse of dimensionality) represents a very mature, mathematically sound analysis.
4. **Physically Grounded Regularizers:** Elastic Spatial Regularization (ESR) and Depth-wise Total Variation (TV) smoothness penalties are physically intuitive, simple to implement, and elegant tools to constrain parameter drift during test-time adaptation.

---

## 3. Weaknesses of the Paper
While the paper is conceptually elegant and written with high clarity, it exhibits significant limitations regarding **practical deployability, scalability, and real-world utility**:

1. **Extreme Latency and Computational Overhead of Test-Time Adaptation:**
   - From a deployment perspective, the computational overhead of test-time adaptation is extremely high. Running 100 steps of 1+1 ES or 60 steps of Adam backpropagation at test-time on a batch of 256 samples per task is incredibly expensive.
   - For 100 steps of 1+1 ES across 4 tasks, this requires $100 \times 4 \times 256 = 102,400$ forward passes. On resource-constrained edge devices (the natural deployment domain for model merging), this massive compute overhead translates into prohibitive test-time latency (seconds or even minutes), which is highly impractical.
   - Given that the resulting adapted model does not even outperform the zero-overhead static uniform baseline (30.17% vs 30.41%), the proposed adaptation paradigm exhibits negative practical utility and unnecessary energy/compute waste in this regime. The paper needs a dedicated cost-benefit and execution latency analysis to address this.

2. **Toy-Scale Experimental Setup and Poorly Converged Experts:**
   - The experiments are restricted to a toy-scale Vision Transformer (\texttt{ViTTiny} with $d_{\text{model}}=64$, 2 heads, 12 layers) and extremely weak, poorly converged downstream experts.
   - The expert models perform exceptionally poorly: MNIST is only **61.03%** (vs. standard >99%), FashionMNIST is **62.47%** (vs. standard >92%), CIFAR-10 is **24.93%** (vs. standard >85%), and SVHN is **17.50%** (vs. standard >90%).
   - Because these experts are underfitted and weak, their task vectors are bound to contain substantial high-frequency parameter noise. This noise artificially amplifies the model's vulnerability to transductive representation collapse during entropy minimization. In real-world deployment scenarios using highly converged foundation models (e.g., CLIP-Large, LLaMA), representations are far more robust and stable, and test-time adaptive merging (e.g., AdaMerging) has been shown to successfully outperform static baselines. Thus, presenting the failure of test-time adaptation as a general truth without validation on standard, high-fidelity experts is a major methodological limitation that limits the broader impact of the work.

3. **Omission of Calibration Stream Size ($N$) Sweeps:**
   - Overfitting is directly governed by the volume of available data. The paper evaluates a single, highly compact calibration batch size ($N=256$). 
   - Sweeping $N$ across a wider range (e.g., $N \in \{32, 64, 128, 256, 512, 1024, 2048\}$) would reveal how the "Generalization-Granularity" curve shifts. In practical settings, if a developer has access to 1024 unlabeled calibration samples, does Level 5 Tensor-wise merging finally overcome the static baseline? The omission of this scaling analysis is a missed opportunity to provide highly actionable guidelines for practitioners.

---

## 4. Evaluation of Dimensions

### Soundness: Good
The technical claims and methodology are mathematically rigorous and correct. The experimental design is consistent and averaged across 3 random seeds with reported standard deviations. However, the rating is capped at "Good" rather than "Excellent" because the reliance on highly underfitted expert models makes it unclear whether the core conclusions (e.g., that adaptive merging cannot beat static blending) generalize to standard, high-performance expert models.

### Presentation: Excellent
The paper is exceptionally well-structured and written with exemplary clarity. Figure 1 provides a beautiful, stylized summary of the central thesis, and Table 1 is cleanly organized and complete. The authors deserve major credit for their intellectual honesty in highlighting and thoroughly analyzing their negative results.

### Significance: Good
The paper provides high diagnostic value by acting as a "cautionary tale" that warns against unnecessary test-time adaptation and reveals the limits of entropy-based surrogate losses. The 5-level taxonomic hierarchy of merging granularities is a valuable, unifying framework that future researchers can build upon. However, its practical significance is limited because the method is evaluated on a toy scale, does not beat the static baseline, and lacks any execution latency analysis.

### Originality: Good
The paper does not introduce a fundamentally new algorithm, as it combines existing components (Task Arithmetic, AdaMerging's entropy loss, 1+1 ES, and standard L2 penalties). However, its originality lies in its creative, systematic taxonomic continuum and its deep conceptual investigation of optimizer-driven overfitting trajectories.

---

## 5. Overall Recommendation

**Rating: 4 (Weak Accept)**

**Justification:**
This is a technically solid, highly rigorous, and exceptionally well-written paper that addresses a critical structural question in model merging. It brings excellent clarity and taxonomic order to a fragmented field. Its deep diagnostic analysis of transductive overfitting, optimizer behaviors (including the "sluggishness hypothesis" for ES), and surrogate loss misalignment is highly valuable and intellectually honest. 

However, its impact is limited by its toy-scale setup (ViT-Tiny), extremely weak expert models, and the lack of execution latency/compute overhead analysis. From a practical perspective, deploying a test-time adaptive optimizer that requires tens of thousands of forward passes and fails to beat a zero-overhead static uniform baseline is highly impractical. Despite these limitations, the paper is highly valuable as a diagnostic "cautionary tale" and establishes a rigorous empirical foundation that others will build on.

---

## 6. Constructive Questions and Suggestions for the Authors

1. **Test-Time Latency and Compute Analysis:** 
   - Could you provide a table or discussion detailing the wall-clock execution time and the total number of forward/backward passes required by each configuration at test-time? A comparison of the computational costs (in FLOPs or seconds) between running 100 steps of 1+1 ES, 60 steps of Adam backprop, and the static Uniform baseline would greatly help ground the practical deployment trade-offs.
2. **Varying Calibration Batch Sizes ($N$):** 
   - How does the Generalization-Granularity curve shift as the size of the calibration batch $N$ increases (or decreases)? Providing an ablation or curve showing L5 performance as a function of $N \in \{64, 128, 256, 512, 1024\}$ would be incredibly high-signal and would help practitioners determine when test-time adaptation becomes viable.
3. **Validation on Fully Converged Experts:** 
   - To address the concern that the observed representation collapse is an artifact of poorly converged experts, would it be possible to run a subset of experiments (e.g., L2 AdaMerging vs. L5 GranMerge) using fully converged expert models (e.g., standard ResNet-18 or ResNet-50 models achieving standard accuracies of >90% on CIFAR-10/SVHN)? Confirming that the same trade-offs hold in a high-fidelity regime would significantly strengthen the paper's generalizability and impact.
