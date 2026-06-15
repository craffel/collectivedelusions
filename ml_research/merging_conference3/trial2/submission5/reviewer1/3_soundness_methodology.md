# 3. Soundness and Methodology Evaluation

## Clarity of the Description
The methodology of **Norm-Equalized Task Arithmetic (NETA)** is described with exceptional clarity. The paper provides a clear mathematical formulation of:
- Standard Task Arithmetic and its vulnerability to task vector norm disparity (Equations 1, 2).
- The layer-wise Frobenius norm equalization (Equations 3, 4, 5, 6).
- The continuous $\alpha$-relaxation framework (Equation 7).
- The noise-damping stabilizer $\beta$ to prevent noise amplification in intermediate layers with negligible updates.
- The physical/geometric reasoning behind the composite layer grouping of visual input parameters (Group 0).
- The closed-form scale compensation factor $\gamma^l$ to correct directional norm contraction (Equation 13).

Algorithm 1 provides a concise, step-by-step procedural pipeline that makes implementing NETA straightforward. The active parameter scope (i.e., applying NETA only to the 13 visual encoder parameter groups of CLIP ViT-B/32 while merging frozen text encoder parameters using standard Task Arithmetic) is also clearly delineated.

## Appropriateness of Methods
From a mathematical and geometric standpoint, layer-wise Frobenius norm equalization is a highly appropriate and intuitive solution to weight-space imbalance. Since the shared-basin assumption holds that fine-tuned expert models reside in a connected, low-loss manifold, balancing the scale of their updates ensures isotropic alignment of their representations in weight space. 

Moreover, the layer-wise design is far more appropriate than model-wide normalization because deep neural networks process representations hierarchically. It is physically correct that shallow layers extract general features with relatively small parameter shifts for simple tasks, whereas deep layers specialize. Standardizing updates layer-by-layer prevents high-magnitude early-layer updates (like those of SVHN) from overwhelming other tasks' representations in the early visual stream.

## Potential Technical Flaws & Limitations (Practitioner's Perspective)

While the methodology is sound, several aspects raise concerns regarding real-world generalizability:

1. **Ad-Hoc Architecture-Specific Heuristic (Group 0)**:
   The composite grouping of input-stage projections, embeddings, and the first Transformer block into a single "visual input block" is a highly manual, architecture-specific heuristic designed specifically for CLIP's visual encoder. While the physical justification (preventing positional/class embedding distortions from negligible updates) is sensible, the paper does not provide a systematic or automated rule for grouping layers in other architectures (such as ConvNets, Swin Transformers, or Large Language Models). For a practitioner looking to deploy NETA on a new model, this introduces manual engineering effort and trial-and-error grouping.

2. **Linear Scale Sensitivity and the Global Hyperparameter ($\lambda_0$)**:
   Although NETA eliminates task-wise and layer-wise merging coefficients, it does not remove the need to tune the global scaling hyperparameter $\lambda_0$. In fact, because NETA's norm-equalization contracts the overall merged update vector (as mathematically qualified in Section 3.4), the optimal $\lambda_0$ for NETA is likely different from that of standard Task Arithmetic. While the authors propose the closed-form $\gamma^l$ scale compensation factor to resolve this contraction analytically, the grid search in Table 3 shows that NETA's average accuracy is still highly sensitive to $\lambda_0$.

3. **Active Parameter Scope Complexity**:
   Applying NETA only to active visual encoder parameters while merging text encoder parameters with standard Task Arithmetic is practical, but it introduces an architectural hybrid. If both visual and textual encoders were task-adapted, or if some layers were frozen and others fine-tuned, the implementation boundaries of NETA become complex, demanding careful custom masking of parameters.

## Reproducibility
The reproducibility of NETA is **excellent**. Unlike test-time adaptation methods (which depend on stochastic optimization, learning rate schedules, batch sizes, and the quality of calibration datasets), NETA is entirely closed-form. Given the pre-trained and expert weights, a practitioner can implement the 3-line weight transformation and obtain identical merged weights every time. The authors have provided clear parameter definitions (e.g., $\beta = 10^{-6}$), mapping rules, and baseline hyperparameters, making this work highly reproducible.
