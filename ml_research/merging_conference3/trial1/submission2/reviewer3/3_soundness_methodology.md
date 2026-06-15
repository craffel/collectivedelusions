# 3. Soundness and Methodology Evaluation

## Clarity of the Description
The methodology of the paper is exceptionally clear, rigorous, and logically structured:
- The problem formulation of continual model merging is mathematically formalized under two distinct regimes ($\lambda = 0.0$ and $\lambda = 0.2$).
- SVD-based Isotropic Merging is laid out step-by-step with clear equations, and the Scalar Update Decay baseline is cleanly contrasted against it.
- The algebraic bug in SAIM's SA-BCD optimizer is clearly isolated, and two alternative corrected variants—SA-BCD (Std Adam) and SA-BCD (Adam GT)—are explicitly defined, making it easy to understand how they work.
- The theoretical synergy section (Section 3.4) features a mathematically sound proof of the linear bound on loss increase ($\Delta L$) in terms of maximum Hessian curvature ($\lambda_{\max}(H)$), establishing a clear link between optimization flatness and weight pruning/sparsification.
- The PEFT extension (LoRA-SAM) is mathematically described with clear perturbation formulas for the adapters $A$ and $B$.

## Appropriateness of Methods
The experimental methodology is highly appropriate and designed with exemplary scientific rigor:
- **Multi-Axial Evaluation Grid ($5 \times 3$):** This grid is the gold standard for auditing a two-stage pipeline, ensuring that every possible optimizer is crossed with every possible merging strategy.
- **Active Parameter Mixing vs. Sequential Parity:** Evaluating under both $\lambda=0.0$ and $\lambda=0.2$ is mathematically sound. It isolates sequential adaptation from active parameter consolidation, which serves as a crucial boundary-condition sanity check.
- **Baseline Isolation (Norm-Matching and Scale-Calibrated):** Introducing these baselines is an outstanding methodological choice. It successfully separates SVD's unique *spectral variance reduction* (flattening the singular values) from simple *global weight update scaling*, which has been a major confounding variable in prior literature.
- **LoRA-SAM Profiling:** Benchmarking GPU memory (VRAM) and wall-clock training time is exactly what is needed to validate the practical utility of LoRA-SAM.

## Potential Technical Flaws and Limitations
While the methodology is solid, we identify a few minor areas of improvement and realistic boundaries from a practitioner's perspective:
1. **Task-Incremental Continual Learning (Oracle Head):** 
   The authors evaluate under the standard *Task-Incremental* setting, where a task-specific classification head is swapped in using an oracle task ID at evaluation time. While this successfully isolates backbone weight merging from head-level interference, it is a less realistic setting than *Class-Incremental* learning, where task IDs are unknown at test time. For real-world deployments in industry, classification models must operate over a joint output space without relying on an oracle task ID.
2. **Taylor Series Approximation Assumptions:**
   The proof for Proposition 3.1 assumes a converged local minimum ($\nabla_\theta L(\theta^*) \approx 0$) and relies on a second-order Taylor series approximation neglecting higher-order terms of $o(\|\delta\theta\|_2^2)$. While standard in deep learning theory, in practice, models are rarely optimized to exactly zero gradient, and large pruning perturbations (e.g., zeroing out 50% or more coordinates in TIES-Merging) might violate the small-perturbation assumption, causing higher-order terms to become non-negligible.
3. **Single-Seed Scale Validation on ViT-Base:**
   Due to the high computational cost of full-parameter sequential fine-tuning on ViT-Base (86M parameters), the authors only report single-seed results for their scale validation. While the margin of improvement ($+3.89\%$) is much larger than the standard deviation of the ViT-Tiny results, having multi-seed standard deviations for ViT-Base would make the scaling claims even more bulletproof.

## Reproducibility
The paper is exceptionally strong regarding reproducibility:
- **Explicit Hyperparameters:** Every crucial training hyperparameter is provided (3 epochs, batch size 128, learning rate 5e-4, weight decay 1e-4, default perturbation radius $\rho=0.05$ for SAM/SA-BCD, coordinate ratio $p=0.3$).
- **PEFT/LoRA Setup:** The authors detail the exact rank ($r=8$), targeting layers (query, key, value projections in all self-attention blocks), and the optimal perturbation radius ($\rho_{LoRA} = 0.08$) after an exhaustive sweep.
- **Commitment to Open Source:** The authors state that they will release their complete modular $5 \times 3$ grid evaluation suite and corrected, verified implementations of the SA-BCD optimizer as public, open-source artifacts, which guarantees high reproducibility.
