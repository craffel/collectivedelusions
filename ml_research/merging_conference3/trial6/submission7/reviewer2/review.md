# Peer Review

## 1. Summary of the Paper
This paper addresses the problem of dynamic weight-space routing for merging multi-task neural network experts. It identifies and seeks to resolve two key failure modes in existing dynamic routing architectures:
1. **Optimization Bloat and Out-of-Distribution (OOD) Overfitting:** Previous parametric routers require training via iterative gradient optimization on small calibration sets, causing severe transductive overfitting and catastrophic collapse on OOD tasks.
2. **Heterogeneity Collapse:** Under mixed-task streaming deployment, dynamic routers average sample-wise routing coefficients across the batch dimension to satisfy hardware accelerator constraints, flattening the coefficients and destroying expert specialization.

To resolve these limitations, the authors present a co-designed algorithm-systems framework consisting of:
- **Parameter-Free Subspace Routing (PFSR):** A non-parametric, training-free routing method that projects penultimate-layer features onto the coordinate subspace of frozen pre-trained expert classification heads via cosine similarity, obtaining routing coefficients via a temperature-scaled Softmax.
- **Unit-Norm Calibration (UNC):** A training-free normalization step to resolve representation and scale mismatches across independently fine-tuned experts.
- **Class-Size Scaling Calibration:** A statistical normalization factor ($O(\sqrt{\log C_k / d})$) to neutralize routing bias toward experts with larger output vocabularies.
- **Micro-Batch Homogenization (MBH):** A data-stream scheduling mechanism that dynamically partitions mixed-task input batches into homogeneous micro-batches based on dominant task coordinates, executes specialized merged inference on each, and re-assembles the outputs.
- **Bounded Top-$k$ Routing and Sub-Vocabulary Prototype Selection:** Optimizations to scale the framework to large expert counts and large vocabulary tasks (such as LLMs) with low systems latency.

The authors evaluate their framework on a synthetic sandbox and present "simulated" large-scale evaluations on Vision Transformers (DomainNet) and LLaMA-7B experts.

---

## 2. Strengths
- **Refreshing Philosophical Deconstruction (Occam's Razor):** The paper takes a bold and highly valuable stance against hyper-parameterization and convoluted routing metaphors (e.g., QWS-Merge). Proving that simple classical $L_2$ regularization on standard linear primitives easily replicates or outperforms wave-inspired routing architectures is a high-signal scientific contribution.
- **Rigorous Systems-ML Co-Design:** Unlike many model-merging papers that ignore deployment realities, this work explicitly co-designs the algorithm alongside systems constraints, considering GPU VRAM footprints, PCIe host-to-device copying bandwidth, Punica-style SGMV parallel kernels, and sequential-vs-parallel parameter materialization trade-offs. The "Systems Deployment Decision Matrix" (Table 4) is exceptionally well-conceived.
- **Attention to Statistical Calibration:** The introduction of Unit-Norm Calibration (UNC) and Class-Size Scaling Calibration ($O(\sqrt{\log C_k / d})$) demonstrates high technical sophistication in resolving representation and output-space scale imbalances across asymmetrical expert registries.
- **Detailed Analytical Intuition:** The first-order mathematical proof of Layer-Averaging Collapse (Section 3.6) provides an elegant, structured explanation of why layer-wise dynamic parameters are redundant under joint classification constraints.

---

## 3. Weaknesses (Detailed Criticisms)

### A. The "Simulation" Illusion (Major Empirical Gap)
The most severe, glaring weakness of this submission is the **complete reliance on simulated representations instead of live model execution** for both the Vision Transformer (DomainNet) and LLaMA-7B benchmarks.
As explicitly disclosed in Sections 4.7 and 4.8:
- *DomainNet Benchmark:* "...these real-world benchmarks are evaluated using **simulated penultimate feature representation manifolds** modeled after actual ViT-Base domain feature distributions... rather than live fine-tuned Vision Transformer weights on full datasets during each simulation pass."
- *LLaMA-7B Benchmark:* "...these large-scale LLM evaluations are **simulated using representative feature embeddings and pre-calculated statistical expert ceilings** rather than running live 7-billion parameter active inference over raw text corpora..."

This is an unacceptable empirical shortcut for a top-tier machine learning conference. The authors claim in their contributions to have "validated their framework on large-scale NLP experts (LLaMA-7B) and standard computer vision (DomainNet)," but they did not actually run these models or perform real weight-merging on live weights. Instead, they ran mathematical simulations of the penultimate representation spaces. 
In a real deployment, representation manifolds of large-scale models are highly non-linear, dynamic, subject to noise, and prone to complex representation drift. Simulating feature spaces (presumably using simple parametric distributions) fails to capture the true complexity of live deep networks. Consequently, the claimed results on DomainNet (Table 5) and LLaMA-7B (Table 6) are purely speculative and lack scientific rigor, as they do not constitute genuine end-to-end empirical validation.

### B. Contradiction in the "Zero Training/Calibration" Value Proposition
The abstract and introduction heavily emphasize a "completely non-parametric framework that contains zero trainable parameters and requires zero calibration split data in standard registries." However, a closer look at their proposed enhancements reveals significant data-dependence and training requirements:
- **Unsupervised Non-Classification Centroids:** To extend the framework to non-classification or generative experts (regression, diffusion), the authors must introduce a calibration split to fit unsupervised $K$-means centroids (Eq. 11, 12).
- **OOD Density Estimation:** Their primary OOD detection method relies on fitting a Gaussian Mixture Model (GMM) on a calibration split (Section 4.4, Table 9). The authors admit this "slightly relaxes our strict 'zero calibration data' claim."
- **Representational Drift Mitigations:** To resolve representational drift in fully fine-tuned models, the authors suggest using a "Lightweight Calibration Projection" (a 1-layer MLP trained on a 64-sample calibration split) or adding "Representation Alignment Objectives" during training (Eq. 10). If experts must be retrained with a customized alignment objective to remain compatible, the method is no longer zero-shot or training-free.
These contradictions undermine the core value proposition of the paper.

### C. The Complexity-Shift Paradox
The authors claim to simplify model-merging by aggressively stripping away trainable routing parameters. However, this is a **complexity shift** rather than a true simplification. They shift the complexity from the model parameters to the underlying data-serving infrastructure. 
On-the-fly stream partitioning, dynamic weight-merging, sequential micro-batch dispatching, and index-based scatter-gather output re-assembly require a highly sophisticated, robust serving layer. Integrating Punica/SGMV kernels for parallel execution requires custom CUDA compilation, specific PyTorch bindings, and dedicated GPU architectures. This represents a substantial systems engineering overhead that may outweigh the training and serving costs of a simple, classical parametric linear router.

### D. Over-Simplified Analytical Proof of Layer-Averaging Collapse
The mathematical proof of Layer-Averaging Collapse (Section 3.6) relies on several highly restrictive and unrealistic assumptions:
- The base network's layer-wise Jacobians $J_b^{(m)}$ act as a sequence of contractive operators that strongly dominate and dampen any layer-dependent semantic variance.
- The intermediate representation manifolds in deep layers stabilize and become approximately collinear across layers ($h_{base, b}^{(l-1)} \approx c_l \bar{h}_{base,b}$).
In actual, deeply hierarchical networks (e.g., LLaMA-7B), early layers and late layers are semantically highly distinct (early layers process low-level structural/syntax details, while late layers process high-level task semantics). The assumption of near-collinearity and contractive projection across all layers is a massive over-simplification. Thus, the "mathematical redundancy" of layer-wise routing is only proven for a highly idealized toy model and does not hold rigorously for real deep networks.

### E. Severe Dependency on the PEFT (LoRA) Paradigm
The spatial viability of the dynamic model-merging framework heavily depends on the PEFT (LoRA) assumption. Without LoRA, keeping $K$ full-parameter expert models concurrently in VRAM would be memory-prohibitive ($>70$ GB for LLaMA-7B experts), and dynamically loading full weights from host CPU to GPU would incur a massive PCIe transfer latency of over 5,000 ms per forward pass (Table 3). For practitioners working with fully fine-tuned, full-parameter expert networks, this method is physically non-viable.

---

## 4. Ratings and Justifications

### Soundness: Fair
The theoretical co-design is systems-literate, and the statistical calibrations (UNC and Class-Size Scaling) are technically sound. However, the soundness is severely undermined by the **complete lack of live end-to-end model execution** on real datasets for the DomainNet and LLaMA-7B benchmarks, relying instead on speculative representation simulations. Additionally, there are clear contradictions in the "zero training/calibration data" claims, and the proof of Layer-Averaging Collapse relies on unrealistic collinearity assumptions.

### Presentation: Excellent
The paper is exceptionally well-structured, clear, and professional. The equations are detailed, the algorithms are complete, and the qualitative tables (such as Table 1 and Table 2) are highly polished. The systems trade-offs and decision matrices are beautifully presented.

### Significance: Fair
While the deconstruction of wave-inspired routing is a valuable contribution, the significance of the proposed PFSR + MBH method is limited. Its validation is simulation-only, its spatial viability is restricted to PEFT (LoRA) adapters, and its substantial serving infrastructure overhead (dynamic partitioning, sequential dispatching) makes it difficult to adopt in standard, out-of-the-box deep learning serving pipelines.

### Originality: Good
The critique and deconstruction of predecessor models are excellent and highly valuable. The proposed PFSR mechanism is an incremental application of Prototypical Networks (Snell et al., 2017) to routing in model merging. Micro-Batch Homogenization is an engineering-level serving heuristic (conceptually connected to request batching in systems-ML) applied to weight-space merging, rather than a major algorithmic breakthrough.

---

## 5. Questions and Constructive Feedback for the Authors
1. **Can you provide live, end-to-end empirical evaluations on real datasets using actual fine-tuned Vision Transformers and LLaMA-7B weights?** Evaluating the framework on actual weights is critical to proving that the proposed similarity projection and micro-batching work on real, non-linear representation manifolds.
2. **How does the Gaussian Mixture Model (GMM) density estimator for OOD rejection scale and perform on real high-dimensional LLM representation spaces?** Please provide empirical validation of GMM rejection on actual (or high-fidelity) LLM feature spaces.
3. **If representational drift is severe in fully fine-tuned models, requiring a "Lightweight Calibration Projection" (a 1-layer MLP trained on 64 samples) or "Representation Alignment Objectives" during fine-tuning, how do you reconcile this with your claim of a "completely training-free, zero-shot framework requiring zero calibration data"?**
4. **Why is the Expert Ceiling on the SVHN task in the synthetic sandbox so low (31.20%)?** A standard visual classifier should easily achieve $>90\%$ accuracy on SVHN. Does this low ceiling indicate that the synthetic sandbox represents an excessively noisy, artificial, or under-trained setup?

---

## 6. Overall Recommendation
**Recommendation: 2: Reject**

**Justification:** While the paper has high technical merit, beautiful presentation, and provides an outstanding, much-needed critique of over-parameterized "quantum" routing, it fails to meet the bar for empirical completeness and scientific rigor required for a top-tier machine learning conference. Specifically, the **simulation-only nature of the real-world benchmarks (DomainNet and LLaMA-7B)** leaves its actual end-to-end performance on real models entirely unproven. Additionally, the paper's core claims of being "completely training-free, requiring zero calibration data" are contradicted by its own OOD GMM estimators and representational drift mitigations, which explicitly require calibration splits and auxiliary training objectives. Finally, the "minimalism" of the method is a complexity-shift paradox that moves the burden from model parameters to highly complex, non-trivial data-serving infrastructure. For these reasons, the paper in its current form is not ready for acceptance and requires genuine end-to-end empirical validation on actual model weights.
