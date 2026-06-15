# 5. Presentation, Impact, and Suggestions for Improvement

## Major Strengths
1. **Exceptional Scientific Rigor:**
   The paper avoids the common pitfall of introducing a convoluted pipeline with un-ablated modules. By designing a decoupled $5 \times 3$ grid, the authors successfully isolate the individual contributions of the optimizer (SA-BCD/SAM) and the merging algorithm (SVD Isotropic Merging/Task Arithmetic).
2. **Identification of an Algebraic Bug and Hardware Bottleneck:**
   Exposing the typesetting/code discrepancy in SAIM's SA-BCD optimizer formula is a major contribution that prevents community confusion. Crucially, pointing out that coordinate-restricted optimizers actually *increase* wall-clock training time by $18.5\%$ on GPU due to sorting and indexing bottlenecks is a stellar, hardware-aware practical insight.
3. **Elegant Mathematical Unification (Proposition 3.1):**
   The formal proof linking optimization flatness (spectral norm of the Hessian) to post-hoc weight consolidation (pruning/sparsification perturbations) is clean, elegant, and provides a powerful explanation for the dramatic synergy observed when crossing SAM with methods like DARE ($+16.89\%$ accuracy boost).
4. **Lightweight, High-Utility PEFT Extension (LoRA-SAM):**
   Proposing LoRA-SAM as a resource-efficient training alternative is highly practical. The authors show that LoRA-SAM matches SAM's merging performance, renders SVD isotropic merging redundant, and runs with negligible GPU overhead ($<2.5\%$ wall-clock time and $<1.5\%$ VRAM overhead), making it an ideal choice for deploying large models.
5. **Scale Validation:**
   Evaluating on a larger ViT-Base (86M parameters) backbone ensures that the core deconstruction findings hold as parameter capacity scales by over $17\times$ from ViT-Tiny.

## Areas for Improvement
1. **Transition to Class-Incremental Continual Learning:**
   The paper evaluates under the *Task-Incremental* setting, where an oracle task ID is provided at test time to select task-specific classification heads. This is slightly contrived compared to the *Class-Incremental* setting, where task IDs are unknown at evaluation and the model must classify over all joint classes. Evaluating under Class-Incremental settings would make the findings even more convincing for real-world applications.
2. **Multi-Seed Results for ViT-Base Scale Validation:**
   Due to high computational costs, the scale validation on ViT-Base (86M parameters) is reported using a single seed. Although the margins are large, having multi-seed standard deviations would improve statistical confidence.
3. **Empirical Validation in NLP:**
   While the authors outline a very detailed, concrete experimental design for NLP practitioners (BERT on GLUE tasks), actually executing a subset of these NLP experiments would elevate the paper from a vision-centric study to a truly multi-modal deconstruction.

## Overall Presentation Quality
The presentation quality is **excellent**:
- The paper is exceptionally well-written, with a clear and engaging narrative flow that treats complex multi-component frameworks with healthy scientific skepticism.
- The mathematical notation is rigorous and consistent.
- Tables are clean, well-formatted, and feature standard deviations for almost all results.
- The discussion section (Section 5.3) is highly mature, acknowledging limitations and providing actionable future paths for the community.

## Potential Impact and Significance
This paper has **high practical and research significance**:
- **For Practitioners:** It simplifies model merging pipelines by demonstrating that standard SAM + Task Arithmetic is highly competitive, and that LoRA-SAM achieves excellent linear mode connectivity without the memory or runtime overhead of standard SAM. It also saves practitioners from implementing complex, mathematically flawed custom optimizers or running expensive $O(d^3)$ SVD spectral reconstructions.
- **For Researchers:** It establishes a clear mathematical bound (Proposition 3.1) that unifies optimization-stage flatness with post-hoc pruning, showing that training-stage sharpness minimization is a mandatory pre-requisite for advanced merging algorithms. It also sets a high standard for how ablation studies and pipeline audits should be conducted in the future.
