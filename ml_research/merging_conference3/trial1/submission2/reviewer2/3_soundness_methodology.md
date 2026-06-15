# Soundness and Methodology Evaluation

## Clarity of Description
- **High-Quality Writing:** The paper is exceptionally well-written, structured, and easy to follow. Each section has a clear purpose, transitioning logically from problem formulation to methodology, empirical results, and discussion.
- **Precise Formulations:** The authors provide rigorous mathematical definitions of all evaluated methods, including SAIM, Scalar Update Decay, the two corrected SA-BCD variants, standard SAM, and their proposed Norm-Matching and Scale-Calibrated baselines.
- **Exposing the Algebraic Bug:** The critique of the SA-BCD optimizer's literal formula is clear, demonstrating mathematically why multiplying the Adam-scaled step by the raw perturbed gradient again causes complete divergence.

## Appropriateness of Methods
- **Comprehensive Grid Evaluation:** The $5 \times 3$ grid crossing 5 optimizers and 3 merging strategies is highly appropriate for decoupling the individual contributions of the optimization phase and the post-hoc merging phase.
- **Rigorous Baselines:** The introduction of the *Scalar Update Decay*, *Norm-Matching*, and *Scale-Calibrated* baselines is a methodical tour de force. It allows the authors to definitively prove that SVD-based isotropic merging's regularizing effect is due to selective singular-spectrum variance reduction (spectrum flattening) rather than global magnitude preservation or scaling artifacts.
- **Two Mixing Regimes:** Evaluating the grid under both the sequential parity regime ($\lambda = 0.0$) and the active weight-mixing regime ($\lambda = 0.2$) is highly appropriate to identify the boundary-condition-sensitive nature of post-hoc SVD regularizations.
- **LoRA-SAM PEFT Integration:** Evaluating LoRA-SAM on a low-rank adapter manifold is highly appropriate for checking whether the findings extend to parameter-efficient fine-tuning, which is of great interest to modern practitioners.

## Potential Technical Flaws and Limitations
- **Single-Seed Scale Validation:** For the ViT-Base validation (Table 3), the authors report single-seed results due to the computational cost of running sequential fine-tuning on an 86M parameter backbone over 5 sequential tasks. While understandable, reporting single-seed numbers lacks the statistical confidence intervals present in the main sweeps (which use 3 seeds). The authors do, however, explicitly acknowledge this as an empirical limitation.
- **Dataset and Domain Scope:** The main empirical findings are restricted to Split CIFAR-100. While highly effective as a controlled proof-of-concept, the behavior of singular value spectrums and loss landscapes can differ in large language models or different modalities. The authors mitigate this limitation by providing a highly detailed, concrete experimental design for NLP practitioners using BERT-Base on GLUE tasks to encourage cross-domain verification.

## Reproducibility
- **Excellent Hyperparameter Detail:** The paper provides exact hyperparameters for all experiments: model architectures, learning rates (5e-4), batch sizes (128), weight decay (1e-4), perturbation radii ($\rho = 0.05$ for SAM/SA-BCD, $\rho_{\text{LoRA}} = 0.08$ for LoRA-SAM), rank ($r=8$), and coordinate selection ratio ($p = 0.3$).
- **Commitment to Open-Source:** The authors state they will release their complete modular $5 \times 3$ evaluation suite and corrected SA-BCD implementations as public, open-source artifacts, further ensuring high reproducibility.
