# Mock Review: Zero-Shot Calibration-Free Model Merging: Opportunities, Limits, and Hybrid Solutions

**Overall Recommendation:** 5: Accept (Technically solid paper with high practical impact, thorough evaluation, excellent systems-level grounding, and exceptional reproducibility)  
**Soundness:** Excellent  
**Presentation:** Excellent  
**Significance:** Excellent  
**Originality:** Excellent  

---

## 1. Summary of the Paper

This paper presents a comprehensive, systems-ML comparative study of **Zero-Shot Calibration-Free Model Merging** for test-time ensembling of Low-Rank Adaptation (LoRA) experts. The central objective of this work is to eliminate the severe operational bottleneck of state-of-the-art dynamic ensembling frameworks (such as SPS-ZCA and SABLE), which rely on pre-computing task-routing centroids using offline, labeled calibration splits (typically $|\mathcal{C}_k|=64$ samples). The paper focuses on privacy-restricted, zero-downtime, and streaming edge environments where labeled calibration data is entirely unavailable and on-device backpropagation is too resource-intensive.

The authors formulate, evaluate, and compare two primary calibration-free serving paradigms:
1. **Zero-Shot Expert Entropy Routing (EER) [Accuracy-First]:** A direct-routing approach that processes incoming samples through all $K$ specialized expert adapters in parallel, computes a proposed scale-invariant *Normalized Shannon Entropy* to measure prediction confidence, and routes 100% of the compute to the expert exhibiting the minimum entropy.
2. **Entropy-Pseudo-Labeled Online Centroid Adaptation (EPL-OCA) [Efficiency-First]:** A centroid-routing approach that maintains running task centroids on-the-fly to enable single-pass serving. It pseudo-labels incoming samples using EER to identify the active expert, updates the corresponding task centroid via a running average, and performs single-pass activation ensembling via Single-Pass Activation-Space Dynamic Blending (SPS) based on cosine similarity to the centroids.

The paradigms are evaluated in a synthetic 192-dimensional multi-task sandbox (modeling MNIST, FashionMNIST, CIFAR-10, and SVHN as orthogonal Gaussian manifolds) and validated on real 512-dimensional ResNet-18 embeddings across 5 independent random seeds.

### Key Empirical Findings:
* **EER Robustness:** Under synthetic shuffled streams, EER achieves **71.38 ± 4.05%** Joint Mean accuracy (outperforming supervised SPS-ZCA by **+4.62%** absolute) and maintains complete robustness under extreme representation drift ($d=0.45$, **71.18 ± 3.74%** accuracy).
* **The Representational Sparsity Paradox:** Hard centroid ensembling (EPL-OCA Hard, $\tau=0.001$) falls to **49.88%** accuracy due to class orthogonality within tasks, which soft ensembling (EPL-OCA Soft, $\tau=0.5$) successfully mitigates to **61.62%** (+11.74% absolute improvement) by acting as a spatial regularizer.
* **Systems Mitigations:** To address the high complexity of EER ($0.25 + 0.75K$ passes), the authors introduce **Amortized Pseudo-Labeling**, which exploits temporal task locality to reduce CPU latency from $6.52\times$ to **$1.57\times$** ($0.2211$ ms per sample) and slashes edge energy footprint by **$4.14\times$** while preserving an outstanding **71.20%** accuracy.
* **Real Embeddings Evaluation:** On real ResNet-18 features, where EER suffers from out-of-distribution (OOD) overconfidence (the Entropy Calibration Discrepancy), the authors propose **Centroid-Gated Entropy Routing (CG-EER)**, achieving **61.50 ± 0.18%** accuracy, outperforming the offline-supervised SPS-ZCA baseline by **+0.70%** absolute accuracy. Attempting to make CG-EER completely calibration-free via unsupervised online centroids (**UCG-EER**) yields only **28.45 ± 1.59%** accuracy, collapsing due to a self-referential pseudo-label corruption loop.

---

## 2. Strengths and Weaknesses

### Strengths
- **High Systems-ML Relevance:** The paper is exceptionally well-grounded in physical serving constraints. It does not treat ensembling in a theoretical vacuum; instead, it profiles actual wall-clock execution speed (milliseconds per sample) on CPU (AMD EPYC 7763) and conducts detailed DRAM vs. SRAM memory bandwidth and energy footprint analyses on edge nodes.
- **Methodological Honesty and Rigor:** The authors deserve high praise for identifying and resolving the chronological data leakage bottleneck in online centroid updates, ensuring that routing and predictions use only historical centroid states.
- **Scale-Invariant Normalized Shannon Entropy Formulation:** The proposed *Normalized Shannon Entropy* successfully corrects the inherent mathematical bias of raw entropy toward experts with smaller vocabularies under heterogeneous registries.
- **Thorough Multi-Dimensional Sweeps:** The paper includes a wide array of highly informative sweeps, including Softmax temperature ablations, registry scalability sweeps ($K \in \{4, 8, 12\}$), temporal task locality block sizes, and linear covariate shifts.
- **Robust Empirical Validation of Edge Cases:** In response to previous reviews, the authors have added two crucial new evaluations that dramatically strengthen the paper's claims:
  1. **Test-Time Adaptation (TTA) Baseline (TENT):** The authors quantitatively evaluate TENT on their real ResNet-18 stream. They show that TENT suffers from catastrophic representation collapse, falling to **20.00%** accuracy on shuffled heterogeneous streams. This empirically validates the authors' theoretical critique that online backpropagation is unstable under realistic serve-time streaming conditions.
  2. **Warm-up Window Sensitivity Ablation:** Sweeping $T_{\text{warmup}} \in \{10, 50, 100, 200\}$, the authors show that ensembling performance is incredibly robust even with an ultra-short warm-up window of only **10 steps** (yielding 59.98% vs 62.12% for 200 steps). This proves that the unsupervised task centroids converge and stabilize almost instantly under the entropy-pseudo-label feedback loop.
- **Statistical Rigor on Real Embeddings:** The ResNet-18 experiments are evaluated over 5 independent random seeds and report standard deviations, confirming the statistical significance of CG-EER's outperformance.

### Weaknesses (Up to 3 Critical Points for Improvement)

#### Point 1: The "Calibration-Free" Claim vs. Hybrid Nature of CG-EER (The Real-World Bottleneck)
The gating threshold in CG-EER is based on pre-computed centroids. Although the authors honestly re-classify CG-EER as a hybrid method, the title ("Zero-Shot Calibration-Free Model Merging") and the introduction still suggest that the entire paper is completely calibration-free.
- **The Issue:** Since CG-EER is the *only* proposed method that functions effectively on real embeddings (achieving 61.50% compared to 35.38% for pure EER), the paper's most viable real-world solution is *not* calibration-free or zero-shot. It re-introduces the exact offline calibration data dependency ($|\mathcal{C}_k|=64$) that the paper claims to eliminate.
- **Action:** The authors must make this hybrid classification more prominent in the Abstract and Intro. Declaring its semi-supervised nature early on will prevent readers from feeling misled about the "calibration-free" premise of the paper.

#### Point 2: Total Collapse of the "Efficiency-First" Paradigm (EPL-OCA) on Real Features
The authors propose EPL-OCA as a major online ensembling paradigm designed to achieve $1.3\times$ amortized serving complexity. While it performs reasonably well in the highly orthogonal synthetic sandbox (61.62% for EPL-OCA Soft), it **completely collapses on real ResNet-18 embeddings**:
- **The Issue:** EPL-OCA Hard yields $27.45 \pm 1.34\%$ and EPL-OCA Soft yields $31.52 \pm 1.37\%$ Joint Mean accuracy. Given that static **Uniform Weight Merging** achieves **$31.66 \pm 0.91\%$** on these same features, EPL-OCA Soft is actually *worse* than or statistically equivalent to a simple uniform average of the expert weights, while EPL-OCA Hard is significantly worse. EPL-OCA relies entirely on the EER pseudo-labeler to direct online centroid updates. However, on real features, the pseudo-labeler is heavily corrupted by the Entropy Calibration Discrepancy (routing 75.2% of all samples to MNIST). This corrupts the MNIST running centroid with SVHN and CIFAR-10 features, while other expert centroids are never updated, resulting in a total collapse of the centroid space.
- **Action:** This is a major methodological limitation of EPL-OCA. It proves that the "Efficiency-First" online centroid adaptation paradigm is **completely non-functional on real features** under standard serving conditions without offline calibration data or pre-computed spatial anchors. The paper should state this limitation and collapse more directly in the main text and abstract.

#### Point 3: Overlapping Class Label Namespace and Evaluation Bias
In both the synthetic sandbox and the real-world ResNet-18 experiments, all $K=4$ tasks have exactly $C=10$ classes, and their labels are represented by overlapping integers $\{0, \dots, 9\}$. 
- **The Issue:** Since the model's prediction is evaluated via a simple argmax match `(pred == y).item()`, if a sample from CIFAR-10 with class index 3 ("cat") is routed incorrectly to the MNIST expert, and the MNIST expert happens to predict class index 3 ("3"), the evaluation script will count this as a **correct** prediction.
- **Action:** While orthogonal representations in the synthetic sandbox and well-separated ResNet-18 features limit the probability of such cross-task prediction matches, they are still theoretically possible (occurring with a background probability of $\approx 10\%$). This introduces a slight optimistic bias in absolute accuracy values across all evaluated ensembling models. The authors should explicitly discuss this label overlap characteristic and its evaluation implications in Section 4.1.

---

## 3. Detailed Constructive Suggestions for the Authors

### 1. Clarifying the "Calibration-Free" Claim for CG-EER (Section 4.10)
- **Observation:** In Section 4.10, the authors state that CG-EER achieves 61.50 ± 0.18% accuracy by applying a representation-space cosine similarity threshold to pre-computed centroids. They honestly re-classify CG-EER as a semi-supervised hybrid method.
- **Action:** Please make this hybrid classification more prominent in the Abstract and Intro. Since CG-EER is the most effective method on real embeddings, declaring its semi-supervised nature early on will prevent readers from feeling misled about the "calibration-free" premise.

### 2. Sandbox Simplification Discussion (Section 4.11)
- **Observation:** The synthetic sandbox is highly orthogonal and isotropic, which isolates the Representational Sparsity Paradox but may make the centroid-routing collapse in EPL-OCA Hard appear more extreme than what might occur on smoother, highly correlated real-world representation manifolds.
- **Action:** Please add a brief discussion in Section 4.11 clarifying how non-orthogonal class layouts or smoother class-manifolds in real LLM or ViT representation spaces would affect centroid-based ensembling.

### 3. SVHN Calibration and Joint Mean Behavior (Section 4.1)
- **Observation:** SVHN's expert ceiling is intentionally set to 39.44% via a high noise scale (0.56). This makes SVHN performance extremely low across all methods, which skews the Joint Mean downward.
- **Action:** Briefly justify why SVHN was calibrated to such a low ceiling (e.g., as an aggressive real-world stress-test for out-of-task noise rejection) and discuss if this exaggerates or mitigates the difference between EER and the other baselines.

---

## 4. Questions for the Authors

1. **Warm-up Window Length:** Since the sensitivity ablation (Section 4.6) shows that EPL-OCA Soft achieves near-optimal performance (59.98% accuracy) with a warm-up window of only **10 steps** (1% of the stream), why is the default warm-up window set to 200 steps (20% of the stream)? Setting a shorter default (e.g., 50 steps) would minimize the duration of the lower-performing fallback policy (Uniform Merging, 56.72%) and boost the overall serving accuracy. Have you considered using a shorter default warm-up?
2. **UCG-EER Stream Length:** Does the *self-referential pseudo-label corruption loop* in UCG-EER persist or worsen over even longer stream lengths (e.g., 5,000 or 10,000 steps)? Is there a simple threshold or confidence gating mechanism that could prevent the overconfident MNIST expert from corrupting the centroids during the warm-up phase?
3. **LLM/ViT Scale Applicability:** How do you expect the *Representational Sparsity Paradox* and *Entropy Calibration Discrepancy* to scale as we transition to larger models (e.g., LLaMA-3 8B or ViT-B/16)? Would the larger representation capacity reduce spatial jitter or exacerbate OOD overconfidence?

---

## 5. Summary of Ratings

- **Soundness:** Excellent
- **Presentation:** Excellent
- **Significance:** Excellent
- **Originality:** Excellent
- **Overall Recommendation:** 5 (Accept)

---

## Conclusion

This is an exceptionally high-quality, technically rigorous, and honest systems-ML paper. It tackles a critical, real-world deployment bottleneck in dynamic model ensembling, analyzes physical edge hardware constraints with impressive depth, and addresses prior reviews meticulously. The paper is solid and highly recommended for publication.
