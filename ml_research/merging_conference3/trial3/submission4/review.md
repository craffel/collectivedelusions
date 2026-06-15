# Peer Review

**Paper Title:** ZipMerge: Joint Weight Pruning and Test-Time Coefficient Tuning for On-Device Model Merging
**Recommendation:** 5: Accept
**Soundness:** Excellent
**Presentation:** Excellent
**Significance:** Excellent
**Originality:** Excellent

---

## 1. Summary of the Paper

This paper presents **ZipMerge**, a training-free framework designed to produce highly compressed, sparse multi-task merged models for resource-constrained edge-device deployment. ZipMerge co-optimizes layer-wise merging coefficients ($\Lambda$) and dynamic magnitude-pruning boundaries ($M$) simultaneously at test-time on a tiny, unlabeled calibration dataset (16 images per task) using an unsupervised Shannon entropy minimization loss. To navigate the non-differentiability of the pruning threshold on-the-fly, the authors investigate two alternative optimization paradigms: (1) **ZipMerge (STE)**, which uses an Identity-pass Straight-Through Estimator to flow gradients back through pruned connections during first-order optimization, and (2) **ZipMerge (ES)**, which uses a derivative-free $1+1$ Evolution Strategy to explore the coefficient space. To stabilize optimization against transductive overfitting on tiny samples, the authors also propose **Reg-ZipMerge**, which incorporates structural distance penalties and functional KL distillation constraints.

Rather than presenting a curated success story, the authors conduct a highly rigorous, honest empirical study and "post-mortem" failure analysis evaluating a compact Vision Transformer (`vit_tiny_patch16_224`) backbone across four highly disparate tasks (MNIST, FashionMNIST, CIFAR-10, SVHN). Their findings expose critical physical boundaries of weight-space merging:
1. **Catastrophic Representational Collapse:** Merging highly orthogonal domains onto a compact backbone using linear task arithmetic causes severe mutual parameter cancellation, collapsing all merged configurations (Uniform, AdaMerging, and ZipMerge) to random guessing levels (~10% to 14% accuracy). This persists even when scaling model capacity by over $15\times$ (to ViT-Base) and across Convolutional Neural Networks (ResNet-18), proving it is a fundamental geometric limitation.
2. **Prune-then-Merge (P-then-M) Baseline Outperformance:** A simple, unoptimized decoupled baseline, P-then-M (which prunes task vectors individually *prior* to merging), consistently and significantly outperforms the test-time optimized joint merging methods because pre-merging pruning acts as a spatial regularizer that zero's out conflicting parameter noise.
3. **The Overfitting-Optimizer Paradox:** Unconstrained minimum-entropy test-time adaptation on tiny calibration sets overfits transductively, successfully minimizing entropy while destroying generalizable features.
4. **Noisy Expert Noise Injection:** Incorporating a poorly converged expert (e.g., SVHN at 19.59% accuracy) acts as a "poison pill" that corrupts the shared parameter space and collapses the performance of other accurate experts.

To bridge the gap between theory and physical systems, the authors translate these boundaries into actionable architectural guidelines, demonstrating that:
- Restricting merging to domain-aligned task groups (e.g., DomainNet) maintains high performance (up to 74.20% joint mean at 50% sparsity).
- Restricting fine-tuning and merging to **low-rank PEFT/LoRA adapters** delivers a massive, highly significant improvement of over **+29% absolute** (reaching 42.30% Joint Mean for dense and 51.45% for ZipMerge-ES at 50% sparsity).
- Practical edge CPU sorting overheads can be completely mitigated using **Delayed Thresholding** ($10\times$ speedup) and **Histogram-based Quantile Estimation** ($17.4\times$ speedup) with zero accuracy degradation.
- A hardware-friendly **structured block pruning** variant of ZipMerge successfully converges to 13.50% joint mean, within -0.50% of its unstructured counterpart.

---

## 2. Strengths and Weaknesses

### Strengths
1. **Outstanding Scientific Honesty and Intellectual Integrity:** In an academic landscape where negative results are frequently buried or "spun" as successes, this paper is highly refreshing and exemplary. The authors present an objective, transparent, and thoroughly validated post-mortem analysis of their proposed algorithm's failures under extreme task shifts. Exposing these boundaries provides immense practical value to edge practitioners.
2. **Deep Practical Systems and Hardware Orientation:** The paper is written from the perspective of real-world edge deployment, which is highly appreciated. Rather than treating pruning and merging as purely mathematical abstractions, it actively addresses physical system limits:
   - Evaluates and mitigates the **Storage-RAM Paradox** of unstructured pruning by implementing and validating a structured block-pruning variant of ZipMerge.
   - Diagnoses the $O(N \log N)$ **On-Device Sorting Overhead** of dynamic percentile sorting on edge CPUs and profiles highly effective mitigations (Delayed Thresholding and Histogram Quantile Estimation), achieving up to $17.4\times$ physical speedup.
3. **Exemplary Experimental Rigor and Variable Isolation:** The paper is a masterclass in thorough empirical validation. The authors go to extraordinary lengths to isolate every potential confounding factor, evaluating:
   - **Reg-ZipMerge ablated sweeps** over structural distance scale $\gamma$ and functional scale $\beta$.
   - **Low-conflict domains (DomainNet)** to confirm the algorithmic correctness of ZipMerge.
   - **Backbone scaling (ViT-Base)** to rule out capacity-induced collapse.
   - **CNN architecture diversity (ResNet-18)** to prove findings are backbone-agnostic.
   - **Seed sensitivity analysis (over 5 seeds)** to establish statistical significance.
   - **Calibration set size sweeps ($B \in \{8, 16, 32, 64, 128\}$)**.
   - **Expert convergence studies** (re-training the SVHN expert to 82.15% to isolate convergence from spatial incompatibility).
4. **Actionable Architectural Guidelines:** The paper does not merely report failure; it translates empirical boundaries into highly practical, concrete design guidelines for edge-systems engineers (e.g., advocating for low-rank adapter merging, domain-aligned task clustering, and explicit test-time regularizers).

### Weaknesses & Areas for Improvement
1. **Lack of Preliminary Generative Language Model Evaluation:** While the authors outline an exciting and highly impactful future direction for applying ZipMerge to large autoregressive language models (such as LLaMA or Mistral) in the conclusion, providing even a preliminary, small-scale evaluation on a tiny language model (e.g., 100M to 1B parameter model) would significantly broaden the paper's reach and impact, given the current dominance of LLM model merging.
2. **Sign-Conflict Resolution inside the Co-Optimization Loop:** While the authors compare against Uniform, AdaMerging, and Prune-then-Merge, they do not explore whether advanced conflict-resolution methods like TIES-Merging or DARE can be integrated directly inside the ZipMerge co-optimization loop to resolve sign conflicts on-the-fly before masking. This represents a minor gap in the methodological exploration.

---

## 3. Detailed Review and Critique of the Methodology

The mathematical formulation of ZipMerge is elegant, sound, and highly precise. Defining the pruning mask as a dynamic function of the continuous layer-wise coefficients ($M(\Lambda)$) is an intuitive way to co-optimize compression and merging. 

The authors' analysis of the **Prune-then-Merge (P-then-M) Superiority** is mathematically and physically compelling. By zeroing out 50% to 80% of the smallest parameter updates in each task vector independently *prior* to merging, P-then-M removes task-specific parameter shifts that do not align with the shared base model. This acts as an extreme spatial regularizer, dramatically reducing overlapping noise and spatial collisions in weight space before the fusion occurs. This finding is highly significant, as it shows that a simple, unoptimized decoupled baseline outperforms complex, unconstrained test-time joint optimization under high task conflict.

The identification of the **Overfitting-Optimizer Paradox** is another major contribution. It highlights that minimizing Shannon entropy on a tiny, unlabeled calibration set (64 images total) is a transductive task; without regularization, the optimizer adjusts the continuous merging coefficients to output highly peaky distributions on those specific 64 images, completely destroying the generalizable feature representations learned during expert training. The proposed **Reg-ZipMerge** formulations (distance penalty and functional KL distillation) are highly appropriate, and the ablated sweeps show a clean, concave sensitivity profile that confirms their effectiveness in mitigating transductive overfitting.

---

## 4. Minor Suggestions and Recommendations for the Authors

1. **Continuous Sweep of the Global Scaling Factor:** In the baseline experiments, the global task vector scaling factor is kept fixed at $\lambda = 0.3$. In high-conflict regimes, exploring a continuous sweep of $\lambda \in [0.0, 1.0]$ could be a simple baseline to see if reducing the scaling factor (e.g., $\lambda \approx 0.1$ as done in ViT-Base) preserves base representations better, even on the compact ViT-Tiny backbone.
2. **Preliminary LLM Demonstration:** If possible, include a preliminary text sequence generation experiment on a tiny autoregressive model (e.g., GPT-2 or a small OPT model) in the camera-ready version. Minimizing next-token prediction entropy or perplexity over a tiny set of generic prompts would demonstrate the generalizability of ZipMerge beyond visual backbones.
3. **Hybrid TIES-ZipMerge Formulation:** Consider discussing or formulating a hybrid model where TIES-Merging's sign-voting and parameter filtering are applied to the task vectors *prior* to ZipMerge's test-time co-optimization. This would combine the spatial noise reduction of P-then-M with the adaptive co-optimization of ZipMerge.

---

## 5. Final Recommendation

This is an **exceptionally high-quality, rigorous, and intellectually honest paper** that represents a major contribution to the field of on-device model deployment and model merging. Its deep system pragmatism, exhaustive experimental backing, and valuable post-mortem analysis make it a highly significant and publishable work. I enthusiastically recommend **Accept**.
