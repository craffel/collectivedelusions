# Peer Review

## 1. Summary of the Submission
The submission proposes **Norm-Preserved Budgeted Task-Vector Pruning (NP-BTVP)**, a post-hoc, training-free weight sparsification and merging framework designed to compress specialized multi-task expert models for resource-constrained edge and IoT environments. Under weight-space model merging (Task Arithmetic), task-specific updates (task vectors) are merged onto a shared pre-trained base backbone. However, storing multiple dense experts remains a significant storage and memory bottleneck on edge hardware. 

To address this deployment challenge, the authors introduce deterministic, magnitude-based pruning schemes: global **Uniform Pruning (NP-BTVP-U)** and layer-wise **Adaptive Saliency-Based Pruning (NP-BTVP-S)**. Crucially, they incorporate **norm-preserving rescaling** (scaling surviving active elements by $1/p$) as a deterministic signal-strength preservation heuristic to prevent update norm shrinkage. Mathematically, the authors derive that this $1/p$ scale factor actually amplifies (boosts) the expected $L_1$ update norm by $2.58\times$ (Gaussian) to $3.30\times$ (Laplace) at 90% sparsity, acting as a highly beneficial "signal-strength boost" that prevents highly sparse task vectors from being drowned out by the base model weights.

Evaluating the framework using a pre-trained CLIP ViT-B/32 backbone across 4 classification datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) over 3 independent random seeds, the authors uncover several high-signal findings:
1. **Curvature vs. Sparsification Separation:** Loss landscape flatness (via Sharpness-Aware Minimization, or SAM) does not provide an additional coordinate-aligned pruning buffer under well-converged regimes compared to standard AdamW. Rather, under norm-preserving rescaling, both AdamW and SAM experts demonstrate extraordinary and nearly identical resilience to post-hoc coordinate magnitude pruning.
2. **The Saliency Double-Bind:** Saliency-Based Pruning (NP-BTVP-S) is slightly outperformed by global Uniform Pruning (NP-BTVP-U) because layer-specific budgets $p_l$ introduce scale instability. Saliency-Global suffers from inter-layer scale mismatch, while Saliency-Layer suffers from extreme local noise amplification.
3. **High Empirical Performance:** At a tight 90% sparsity budget ($p=0.10$), global Uniform Pruning (NP-BTVP-U) achieves **90.34%** (AdamW) and **90.32%** (SAM) average accuracy, performing close to the dense unpruned baselines (**90.94%** and **91.00%**). It is highly competitive with DARE-Merging (operating at 80% sparsity) and completely crushes TIES-Merging by **3.81%** average accuracy under SAM while using only half of TIES's parameter budget (10% vs 20%).
4. **Quantization Synergy:** NP-BTVP-U integrates seamlessly with 8-bit quantization (INT8), achieving a massive **40$\times$ storage footprint reduction** (reducing a 114.8 MB CLIP expert to only **5.74 MB**) with a negligible accuracy drop of only **0.12%** under SAM (retaining **90.20%** average accuracy).

---

## 2. Strengths and Weaknesses

### Strengths
1. **Exceptional Practical and Edge Utility:**
   The paper is highly focused on real-world edge-deployment constraints. The proposed method is completely training-free, requires zero forward/backward passes during weight fusion, avoids any dependence on calibration datasets, and has a computational complexity of $O(d \log d)$ due to simple sorting. This enables on-device expert switching and over-the-air expert updates on severely bandwidth-constrained edge hardware.
2. **Deep and Rigorous Theoretical Foundations:**
   Unlike many purely empirical model-merging papers, this work provides a beautiful, formal mathematical analysis (Appendix A). Deriving the expected $L_1$ norm ratios under Laplace and Gaussian update distributions and computing the expected $L_2$ reconstruction error provides a rock-solid mathematical justification for why deterministic reciprocal rescaling ($1/p$) acts as a vital "signal-strength boost."
3. **Flawless and Unbiased Experimental Rigor:**
   The evaluation is exceptionally fair and statistical. All experiments are conducted over 3 independent random seeds with reported standard deviations. The authors completely avoid "baseline-handicapping" by individually sweeping and optimizing the merging coefficient $\lambda \in [0.1, 1.0]$ with a step size of 0.1 for *each baseline on each configuration*, ensuring that both TIES-Merging and DARE-Merging are evaluated at their absolute peak performance.
4. **High-Signal Scientific Insights:**
   The paper reports and analyzes a valuable negative/counter-intuitive result: training-stage loss landscape flatness (SAM) does *not* provide an additional coordinate-aligned pruning buffer under well-converged regimes. This helps separate the geometry of dense weight merging from unstructured coordinate sparsification.
5. **Practical Synergy with Quantization:**
   The demonstration of the "Saliency Double-Bind" under 8-bit quantization (where Saliency-Layer completely collapses due to noise amplification, while Uniform holds strong) is a fantastic, deployment-ready blueprint that will save Edge AI practitioners from unnecessarily complex layer-wise budget optimization.

### Weaknesses
1. **Scale of Empirical Evaluation:**
   The empirical validation is focused on a pre-trained CLIP ViT-B/32 backbone fine-tuned on 28.7 million parameters across disjoint datasets containing 1024 samples. While the statistical rigor is outstanding (3 seeds, mean, and standard deviation), evaluating the framework on larger models (such as LLaMA-7B or ViT-L) on standard benchmarks would strongly reinforce the scalability of their conclusions.
2. **Conceptual Refinement of the Name "Norm-Preserved":**
   As the authors transparently acknowledge, the name "Norm-Preserved" is conceptually a slight misnomer. Since they apply $1/p$ to the *largest* (deterministically selected) updates, they mathematically amplify the expected $L_1$ update norm (e.g., $3.30\times$ for Laplace, $2.58\times$ for Gaussian at $p=0.10$). Thus, the method acts as a **"Signal-Strength Boost"** rather than a strict norm-preservation identity. Although this over-scaling is highly beneficial in practice, referring to it as **"Norm-Scaled"** or **"Signal-Boosted" Budgeted Task-Vector Pruning** would reflect the mathematical reality more accurately.

---

## 3. Detailed Dimension Ratings

### Soundness: Excellent
The submission is technically sound. The mathematical proofs in Appendix A are correct, based on reasonable and standard parametric assumptions (Laplace and Gaussian distributions), and offer clear derivations of expected L1 norms and L2 reconstruction errors. The empirical methodology is exemplary: testing across 3 random seeds, avoiding baseline handicapping through exhaustive individual baseline searches, and including a complete ablation study that isolates and proves the critical role of norm-preserving rescaling.

### Presentation: Excellent
The paper is exceptionally well-written, clearly structured, and easy to follow. The transition from task-vector extraction to SAM optimization, pruning formulations, and empirical sweeps is logical. The mathematical notations are standard and consistent. The figures (e.g., the pruning resilience curves and baselines comparison) are clean, informative, and perfectly labeled, summarizing complex multi-variable sweeps clearly.

### Significance: Excellent
The paper addresses a highly important, real-world deployment problem in decentralized machine learning. By establishing a robust, training-free, highly compressed weight-space merging pipeline, this work has a direct, massive impact on Edge AI and IoT. Practitioners can immediately deploy specialized multi-task expert models at a fraction of the storage and network bandwidth (10--40$\times$ reduction), unlocking dynamic on-device expert switching on resource-constrained platforms.

### Originality: Good
The individual components—magnitude-based pruning, uniform rescaling, and binary search—are classical compression techniques. However, the unique combination of these techniques in a training-free, post-hoc weight-space merging context, the conceptual framing of deterministic rescaling as a "signal-strength boost," the discovery of the "Saliency Double-Bind," and the deconstruction of SAM's coordinate-wise resilience represent a highly solid, creative, and significant conceptual advance.

---

## 4. Overall Recommendation

**Rating: 5 (Accept)**

### Detailed Rationale
This is a technically solid, exceptionally well-written, and pragmatically grounded paper that makes a highly significant contribution to the field of weight-space model merging and Edge AI. It successfully solves a pressing deployment problem: compressing specialized multi-task experts down to a fraction of their size (yielding 10--40$\times$ raw storage reductions) without adding any training overhead or test-time computational latency. 

The paper's theoretical derivations are rigorous, the empirical evaluations are unbiased and statistically powerful, and the insights (the "Saliency Double-Bind" and the separation of loss-landscape flatness from coordinate sparsification) are highly valuable for both researchers and practitioners. While evaluating on a larger-scale model (such as LLaMA-7B) would be a valuable addition, the overall technical quality, practical utility, and exceptional clarity of the manuscript easily meet the standard for acceptance.

---

## 5. Constructive Feedback and Questions for the Authors

1. **Scalability to LLMs:** 
   In Appendix B, you discuss the potential scalability of NP-BTVP to Large Language Models (LLMs) and suggest that the "Saliency Double-Bind" would be even more pronounced due to severe inter-layer scale mismatch and structural heterogeneity. Do you have any preliminary empirical results or plans to evaluate this framework on a compact LLM (such as LLaMA-3-8B or OPT-125M) on standard NLP datasets? This would demonstrate the generalizability of your findings beyond visual encoders.
2. **Conceptual Nomenclature:**
   Since the $1/p$ scaling factor mathematically amplifies the expected $L_1$ norm by $2.58\times$ to $3.30\times$ (as elegantly proven in Appendix A), the term "Norm-Preserved" is slightly misleading. Have you considered renaming the framework to something like **"Norm-Scaled"** or **"Signal-Boosted" Budgeted Task-Vector Pruning** to reflect this beneficial signal amplification more accurately? 
3. **Open-Source Implementation:**
   The paper does not mention an open-source repository. Given the immediate practical value of this framework to Edge AI and IoT developers, do you plan to release a clean, open-source library or integrate this framework with popular community tools like `mergekit`? This would drastically maximize the adoption and impact of your work.
4. **Comparison with Strict $L_1$-Preservation Scale Factors:**
   In Section 3.3 and Appendix A, you mathematically analyze that the $1/p$ scale factor over-scales the remaining coordinates and acts as a signal boost, and you mention that this is empirically superior to strict $L_1$-preservation scale factors (such as $S_p = \frac{1}{p(1-\ln p)}$ under Laplace, which preserves the exact norm). Could you provide the specific empirical accuracies of the strict $L_1$-preserving scale factor $S_p$ in the main paper or appendix to visually demonstrate this performance gap to the readers?
