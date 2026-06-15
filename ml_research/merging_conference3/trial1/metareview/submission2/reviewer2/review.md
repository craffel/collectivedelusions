# Peer Review

## Summary
This paper presents a systematic methodological deconstruction of Sharpness-Aware Isotropic Merging (SAIM), a recent dual-stage model merging framework for continual learning. SAIM claims that both a coordinate-restricted sharpness-aware optimizer (SA-BCD) and an SVD-based adaptive isotropic merging algorithm are necessary to prevent representation collapse during consolidation. 

The authors systematically audit these claims by constructing a rigorous 5 (optimizers) $\times$ 3 (merging strategies) grid evaluation on Split CIFAR-100 with a Vision Transformer (ViT-Tiny). They also evaluate active parameter-mixing regimes ($\lambda = 0.2$), validate their findings on a larger 86M-parameter ViT-Base backbone, generalize their deconstruction to PEFT via a newly proposed LoRA-SAM method, and formalize the synergy between optimizer flatness and pruning operators (such as TIES and DARE).

The key findings are:
1. **Flatness is Primary:** Standard, globally-perturbed Sharpness-Aware Minimization (SAM) combined with naive Task Arithmetic represents an extremely strong, often ignored baseline that yields a massive +9.87% accuracy boost under sequential parity ($\lambda=0.0$) and a +12.30% boost under active mixing ($\lambda=0.2$).
2. **SVD is Boundary-Condition-Sensitive:** Under standard sequential parity ($\lambda=0.0$), post-hoc SVD isotropic merging is redundant and acts as a distortive operator on un-mixed parameters, dropping average accuracy by 3% to 12%. However, under active mixing ($\lambda=0.2$), SVD isotropic merging successfully acts as a regularizer against parameter interference, boosting SAM's performance to the overall best score of 76.42%.
3. **SA-BCD is Mathematically Broken and Suboptimal:** The authors expose a fatal algebraic typo in SAIM's published SA-BCD optimizer formula that causes complete optimization divergence. Furthermore, they prove that coordinate-restricted perturbation is suboptimal compared to global SAM (+5.37% advantage for SAM) and introduces an 18.5% training wall-clock slow-down due to coordinate-sorting and sparse indexing operations.
4. **PEFT Generalization:** They propose LoRA-SAM, showing that on a low-rank manifold, post-hoc SVD isotropic merging is redundant (yielding a negligible +0.73% improvement) because the adapters are already flat and low-rank. LoRA-SAM achieves excellent performance (74.12% accuracy) with negligible wall-clock (<2.5%) and VRAM (<1.5%) overhead.
5. **Synergy with Post-Hoc Consolidation:** The authors theoretically formalize (Proposition 3.1) and empirically demonstrate that pre-merging optimization flatness (SAM) bounds the loss increase under post-hoc coordinate-wise pruning, making parameters structurally robust to high dropout rates (50%) in TIES and DARE.

---

## Strengths and Weaknesses

### Strengths
1. **Outstanding Empirical and Methodological Rigor:** The paper is an exceptionally disciplined, modular, and thorough peer audit. Instead of proposing another hyper-complex pipeline, it systematically breaks down an existing one to evaluate the true causal drivers of performance.
2. **Exhaustive and Well-Isolated Baselines:** The authors introduce a series of highly clever baselines—such as *Scalar Update Decay*, *Norm-Matching*, and *Scale-Calibrated* baselines—which successfully decouple SVD's unique singular-spectrum variance reduction mechanism from global weight scaling or magnitude-preservation artifacts.
3. **Identification of a Critical Bug:** Detecting and formalizing the mathematical typo in the published SA-BCD formula is of great service to the community, preventing researchers from wasting compute resources trying to reproduce a broken formula.
4. **Strong Theoretical Underpinnings:** The inclusion of Proposition 3.1, which shows that the loss change from pruning-induced parameter perturbations is upper-bounded by Hessian curvature, provides a robust, elegant mathematical foundation that explains the observed synergy with TIES and DARE.
5. **Highly Practical PEFT Generalization:** Proposing and validating LoRA-SAM on a low-rank adapter manifold is highly valuable. The detailed GPU/CPU execution time benchmarks and VRAM profiling demonstrate that optimization-stage flatness can be unlocked in large foundation models with virtually zero overhead (<2.5% time and <1.5% VRAM), bypassing post-hoc SVD bottlenecks.
6. **Multi-Scale Validation:** Validating the key configurations on a 17x larger ViT-Base (86M parameters) backbone ensures that the deconstruction findings translate to larger parameter capacities.
7. **Statistically Sound Evaluation:** Primary experiments are averaged over 3 random seeds and include confidence intervals (standard deviations), demonstrating a high level of empirical rigor.

### Weaknesses
1. **Single-Seed Scale Validation:** For the ViT-Base scale validation (Table 3), the authors report single-seed results due to computational constraints. While the authors explicitly acknowledge this as an empirical limitation, having multi-seed statistics would make this scaling analysis more robust.
2. **Dataset Diversity:** The main empirical evaluation is restricted to Split CIFAR-100. While the authors outline a concrete, feasible experimental design for NLP tasks on GLUE to encourage cross-domain verification, actual empirical results on larger-scale datasets (such as ImageNet) or language models would make the findings even more definitive.

---

## Evaluation Categories

### Soundness
* **Rating:** **Excellent**
* **Justification:** The paper's empirical methodology is exceptionally sound. The crossed 5 $\times$ 3 grid is an excellent experimental design to isolate the contribution of individual components. The custom-designed *Norm-Matching* and *Scale-Calibrated* baselines are methodologically brilliant and allow the authors to isolate SVD's spectral effects with high precision. All primary claims are backed by solid empirical data with statistical error bounds, and the theoretical derivations in the appendix and main text are mathematically correct.

### Presentation
* **Rating:** **Excellent**
* **Justification:** The paper is written with extreme clarity and professionalism. The logical flow is seamless, the tables are beautifully organized, and the captions provide thorough context. The mathematical exposition is highly precise, and the deconstruction of the SA-BCD algebraic typo is described clearly and constructively.

### Significance
* **Rating:** **Excellent**
* **Justification:** This paper addresses an important and active area of research (model merging and continual learning) and delivers an essential, sobering lesson on the inflation of complex frameworks. By showing that a heavily-tuned standard baseline (SAM + Task Arithmetic) can outperform or match complex multi-stage pipelines, it establishes a solid, transparent foundation for future work and raises the bar for peer review. The practical utility of LoRA-SAM also provides immediate benefits to PEFT practitioners.

### Originality
* **Rating:** **Excellent**
* **Justification:** While deconstruction papers are sometimes viewed as purely evaluative, this work introduces significant original contributions: the mathematical formalization of optimizer-driven flatness and pruning synergy (Proposition 3.1), the introduction of LoRA-SAM with thorough resource profiling, and the design of novel baselines (*Norm-Matching* and *Scale-Calibrated*) to isolate SVD mechanisms.

---

## Overall Recommendation
* **Rating:** **5: Accept**
* **Justification:** This is an exceptionally high-quality, technically solid, and methodologically rigorous paper. It presents a disciplined, comprehensive deconstruction of a multi-component merging pipeline, exposing fatal formulas, redundant components, and highlighting the critical role of optimizer-driven flatness (SAM). The paper's evaluation is thorough, its baselines are extremely strong, and the proposed LoRA-SAM provides highly practical and scalable value. It is a highly scholarly, polished contribution that sets an excellent standard for empirical rigor in the model-merging literature.

---

## Questions and Constructive Suggestions

1. **Multi-Seed Scale Validation:** While the computational constraints of running sequential fine-tuning on a 17x larger backbone (ViT-Base, 86M parameters) are understandable, could the authors provide results for at least 2 or 3 random seeds for Table 3 in the camera-ready version? Even a 2-seed average with standard deviation would provide stronger statistical confidence for the scaling analysis.
2. **Expansion to ImageNet-Scale:** For future work, it would be highly beneficial to evaluate key configurations on a more challenging vision dataset (e.g., TinyImageNet or ImageNet-1k) to verify if the observed singular-value spectrum and loss landscape dynamics generalize beyond Split CIFAR-100.
3. **Sparsity Amortization in Full-Parameter SAM:** In Section 6, the authors discuss several strategies to mitigate the double-backward pass overhead of full-parameter SAM, such as applying sharpness-aware updates only every $k$ steps or restricting perturbations to specific late layers. Have the authors run any preliminary experiments with these amortization strategies? Reporting even a small ablation on these techniques in the appendix would be of great value to full-parameter practitioners.
