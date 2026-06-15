# Peer Review of Conference Submission

## Paper Summary
The submission presents a rigorous, systematic, and highly practical deconstruction of the Sharpness-Aware Isotropic Merging (SAIM) framework for continual learning. SAIM is a modern, complex framework consisting of a coordinate-restricted sharpness-aware optimizer (SA-BCD) and an SVD-based Adaptive Isotropic Merging algorithm. 

To evaluate these components, the authors conduct a decoupled multi-axial grid ($5 \times 3$ configuration) on Split CIFAR-100 with a Vision Transformer backbone, crossing 5 optimizers and 3 merging strategies. They evaluate these under two distinct mixing conditions: sequential fine-tuning parity ($\lambda = 0.0$) and active weight mixing ($\lambda = 0.2$). Additionally, they extend their analysis to Parameter-Efficient Fine-Tuning (PEFT) by proposing and profiling **LoRA-SAM**, conducting scale validation on a ViT-Base (86M parameters) model, and providing an elegant mathematical bound linking training flatness to post-hoc pruning robustness.

The paper reveals that:
1. Optimization-stage flatness (using standard SAM) is the primary driver of merging performance.
2. SVD-based isotropic merging is highly sensitive to boundary conditions; it degrades performance under sequential fine-tuning parity ($\lambda = 0.0$) but acts as a helpful regularizer under active mixing ($\lambda = 0.2$).
3. The proposed SA-BCD optimizer contains a fatal algebraic bug that leads to training divergence, and even when corrected, is computationally and empirically suboptimal compared to standard SAM.
4. LoRA-SAM is highly efficient ($<2.5\%$ wall-clock time and $<1.5\%$ VRAM overhead) and renders expensive $O(d^3)$ SVD-based merging redundant on low-rank manifolds, enabling zero-overhead model consolidation.

---

## Strengths and Weaknesses

### Strengths
- **Exemplary Scientific Rigor and Ablation Design:** The decoupled multi-axial $5 \times 3$ grid is an outstanding example of how complex deep learning pipelines should be audited. It successfully isolates the individual contribution of the optimizer from the post-hoc merging strategy.
- **Hardware-Aware Practical Insights:** The paper provides exceptional practical value by highlighting that coordinate-wise optimizers (like SA-BCD), which are theoretically designed for efficiency, actually increase wall-clock training time by $18.5\%$ on GPU. This is due to sequential sorting, indexing, and masking operations that break tensor parallelization. Highlighting this gap between theoretical design and hardware reality is extremely valuable.
- **Uncovering and Correcting a Fatal Typo:** The paper identifies and mathematically details a critical algebraic bug in SAIM's published SA-BCD optimizer formula (multiplying by the raw perturbed gradient again), which leads to immediate training divergence. The authors implement corrected variants to provide a fair and thorough baseline comparison.
- **Elegant Theoretical Unification (Proposition 3.1):** The authors prove a second-order Taylor bound showing that the loss increase from post-hoc weight consolidation (like pruning in TIES-Merging or random dropouts in DARE) is directly bounded by the spectral norm of the Hessian, $\lambda_{\max}(H)$. This provides a clean, rigorous explanation for the dramatic synergy observed when crossing SAM with post-hoc pruning methods (such as DARE).
- **Scalable, High-Utility PEFT Proposal (LoRA-SAM):** The introduction and profiling of LoRA-SAM is highly practical. Showing that LoRA-SAM matches SAM's merging performance while adding negligible training overhead ($<2.5\%$ wall-clock and $<1.5\%$ VRAM) and rendering expensive SVD computations redundant on low-rank manifolds offers a direct, ready-to-deploy, and highly scalable pipeline for foundation models.
- **Scale Validation:** Validating the core findings on a larger ViT-Base (86M parameters) backbone ensures that the deconstruction findings hold as parameter capacity scales by over $17\times$ from ViT-Tiny.

### Weaknesses
- **Task-Incremental vs. Class-Incremental Continual Learning:** The paper evaluates under the *Task-Incremental* setting, where an oracle task ID is provided at test time to swap in task-specific classification heads. While this isolates backbone weight merging from head interference, it is a less realistic setting than *Class-Incremental* learning, where task IDs are unknown at evaluation. Real-world machine learning systems must operate over a joint output space without relying on an oracle task ID.
- **Single-Seed Scale Validation for ViT-Base:** Due to high computational costs, the scale validation results on ViT-Base are reported with a single seed. Although the absolute margins ($+3.89\%$) far exceed the standard deviations observed in the ViT-Tiny sweeps, having multi-seed standard deviations for ViT-Base would improve statistical confidence.
- **Lack of Diverse Architectures/Domains in Empirical Evaluation:** While the authors outline a very detailed, concrete NLP experimental design (BERT on GLUE tasks) and discuss hidden-layer scaling limits of SVD, actually executing a subset of these NLP or CNN experiments would further strengthen the claims of cross-domain generalizability.

---

## Detailed Category Ratings

### Soundness: Excellent
The paper's soundness is methodologically flawless. The multi-axial grid crosses all major variables, and the evaluation of boundary conditions ($\lambda=0.0$ vs. $\lambda=0.2$) is mathematically and conceptually sound. Introducing control baselines like Norm-Matching and Scale-Calibrated successfully isolates SVD's spectral effects from simple global weight updates. The theoretical proof for Proposition 3.1 is correct and provides a strong foundation for the observed empirical results. The profiling of LoRA-SAM (training time, VRAM, and hyperparameter sensitivity) is exceptionally thorough and realistic.

### Presentation: Excellent
The paper is exceptionally clear, logical, and structured. The narrative flow is engaging and maintains a highly objective, scholarly skepticism. The mathematical notation is precise and consistent. Tables are clean, well-formatted, and feature standard deviations for almost all configurations. The discussion section is mature, honestly addressing limitations (such as model scale and task-incremental setups) and outlining concrete experimental designs to support future research.

### Significance: Excellent
The paper has high practical and research significance. For practitioners, it simplifies model merging by showing that standard SAM + Task Arithmetic is highly competitive, and that LoRA-SAM achieves excellent linear mode connectivity without the memory or runtime overhead of standard SAM. It also saves practitioners from implementing complex, mathematically flawed custom optimizers or running expensive $O(d^3)$ SVD spectral reconstructions when they are not needed. For researchers, it establishes a clear mathematical bound (Proposition 3.1) that unifies optimization-stage flatness with post-hoc pruning, showing that training-stage sharpness minimization is a mandatory pre-requisite.

### Originality: Excellent
For an audit and deconstruction paper, this work is exceptionally original. It goes far beyond a simple baseline comparison by identifying a fatal algebraic typo in prior literature, exposing hardware parallelization bottlenecks of coordinate-restricted optimizers, formulating a rigorous mathematical proof linking flatness to pruning resilience, and proposing a lightweight PEFT variant (LoRA-SAM) that completely bypasses the SVD bottleneck.

---

## Overall Recommendation

**Rating: 6: Strong Accept**

**Justification:**
This is an outstanding, technically flawless paper that provides exceptional impact on the highly active area of model merging. The deconstruction is executed with exemplary scientific rigor, and the findings yield clear, actionable design principles for building efficient and scalable merging pipelines. By exposing algebraic bugs and hardware bottlenecks in "complex" custom optimizers, the authors save practitioners and researchers from false research paths. Furthermore, the introduction of LoRA-SAM, the GPU profiling of VRAM and training time, and the theoretical unification of flatness and pruning (Proposition 3.1) make this paper highly complete, extremely valuable, and fully ready for publication.

---

## Constructive Feedback and Questions for Authors

1. **Class-Incremental Evaluation:** Have the authors considered evaluating their configurations under a *Class-Incremental* setting? Given that a practitioner's ultimate goal is to deploy these consolidated models in settings where oracle task IDs are unavailable at test time, showing how training-stage flatness impacts representation drift and joint-head classification accuracy in class-incremental scenarios would be a major addition.
2. **LoRA-SAM Scaling to Billions of Parameters:** The GPU profiling for LoRA-SAM on ViT-Tiny is highly encouraging ($<2.5\%$ training time and $<1.5\%$ VRAM overhead). Do the authors have any preliminary profiles or expectations on how these percentages scale as the backbone scales to billions of parameters (e.g., LLaMA-7B)? Since the active parameter space of LoRA remains extremely small ($<1\%$) regardless of the backbone scale, do you expect the overhead of LoRA-SAM to remain negligible on large LLMs?
3. **Multi-Seed Scale Validation:** If computational resources permit, adding multi-seed results (with standard deviations) for the ViT-Base scale validation in Table 4 would further strengthen the statistical confidence of the scaling claims.
4. **NLP Generalization:** The outlined NLP experimental design (BERT on GLUE tasks) is excellent. Do the authors plan to execute and release these NLP results in the final open-source artifact, or is it reserved for future work? Even a single-seed BERT scale validation would go a long way in establishing cross-domain generalizability.
