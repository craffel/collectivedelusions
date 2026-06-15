# 3. Soundness & Methodology

An evaluation of the clarity of the description, appropriateness of methods, potential technical flaws, and reproducibility.

## Clarity of the Description
The methodology is described with exemplary clarity and mathematical rigor. The paper clearly formalizes:
- The multi-task model-merging paradigm.
- The input state projection and normalization (unsupervised PCA).
- The exact mathematical formulations of the QWS-Merge cosine routing equations.
- The three proposed classical alternative routing channels (L3-Linear, L3-Tanh, and L3-Softmax).
- The global single-layer classical Linear Router baseline, in both vector and element-wise forms.
- The closed-form proof of layer-averaging collapse.

The notation is consistent, precise, and highly readable. The inclusion of tables comparing parameter counts and detailed appendices covering limitations, optimization audits, and mathematical derivations makes the paper exceptionally comprehensive.

## Appropriateness of Methods
1. **Isolating Coordinate Sandbox:** Large-scale model merging benchmarks typically couple routing performance with weight space alignment conflicts, making it impossible to diagnose whether a failure is due to an unstable router or coordinate misalignment in backbone weights. The authors' design of an **Isolating Coordinate Sandbox** (using MNIST, FashionMNIST, CIFAR-10, and SVHN prototypes in orthogonal feature spaces) is a highly appropriate and elegant methodological control. It successfully isolates and analyzes the fundamental behavior of the dynamic routing equations.
2. **Empirical CLIP Scale-Validation Pilot:** To address the potential "toy" nature of the sandbox, the authors perform a scale-validation pilot merging actual task-specific Vision-Language models (CLIP-ViT-B/16 image encoders, 86M parameters each). This scale-validation pilot is highly appropriate and confirms that the trends isolated in the sandbox translate directly to commercial weight manifolds.
3. **Advanced Audits in Appendices:** The authors anticipate and address multiple potential methodological questions through:
   - *Optimization Sensitivity Audit (Appendix E):* Sweeping QWS-Merge learning rates to prove its collapse is structural, not an artifact of poor tuning.
   - *Task Correlation and Overlap Audit (Appendix G):* Testing non-orthogonal task prototypes to prove classical routing still dominates when tasks overlap.
   - *True Layer-by-Layer Merging Audit (Appendix H):* Evaluating deep multi-layer experts without coefficient averaging to prove classical methods maintain their advantage even when the layer-averaging collapse is bypassed.
   - *Multi-Seed Robustness Audit (Appendix F):* Verifying statistical stability across five random seeds.

## Potential Technical Flaws / Limitations
From a **practitioner's perspective**, there are no major technical flaws, but several structural limitations should be highlighted:
1. **Negligible Absolute Parameter Savings:** The 16.7% parameter footprint reduction (saving 56 parameters, 280 vs 336) is practically negligible on hardware when compared to backbone model scales (e.g., 86M parameters for CLIP-ViT-B/16 or 7B/8B for LLMs). The authors honestly acknowledge this in Section 4.3 and Appendix B, noting it is of high theoretical and structural significance rather than of practical memory-saving value.
2. **Implementation Hurdles of Triton-based Dynamic Assembly:** In Appendix A.3, the authors propose Triton-based dynamic weight assembly to bypass heterogeneity collapse by performing sample-specific merging on-the-fly in SRAM. While theoretically elegant, loading $K$ separate sets of LoRA weights into SRAM from High Bandwidth Memory (HBM) at runtime introduces significant memory synchronization bottlenecks, warp scheduling overheads, and HBM bandwidth saturation on modern GPUs (e.g., NVIDIA H100). The authors acknowledge this as an "active engineering frontier," which is a highly realistic and grounded assessment.
3. **Scale and Task Complexity of CLIP Pilot:** The CLIP pilot merges only $K=3$ relatively simple visual classification tasks (MNIST, FashionMNIST, CIFAR-10). When scaling to highly complex, diverse real-world tasks and massive LLMs, weight-space coordinate misalignment ($\text{Error}_{alignment}$) scales significantly due to non-linear representation drift across independent fine-tuning paths, which could introduce additional routing instabilities.

## Reproducibility
The reproducibility of this work appears to be extremely high. The authors provide:
- Exact data generation details, dimensionalities ($D=192$, $D=768$, $d=4$).
- Specific optimizer settings (AdamW, learning rate $10^{-2}$, weight decay $10^{-3}$, 100 epochs).
- Exact initializations and stability controls matching the original QWS-Merge paper (Appendix E).
- Complete mathematical formulations for both the sandbox and the CLIP-ViT-B/16 pilot.
- All experimental statistics and standard deviations across five seeds (Appendix F).
- Step-by-step algorithms and mathematical derivations.
Practitioners would have no difficulty reproducing these results on standard deep learning libraries.
