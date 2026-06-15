# 4. Experiment Check

## Experimental Setup and Evaluation Design
The paper presents a comprehensive and well-designed experimental validation consisting of two distinct parts:

1. **Part I: Simulation Sweeps (30 Seeds):**
   - *Design:* Evaluated across 30 seeds on two simulated loss landscapes: Model I (Convex Landscape with diagonal Hessians) and Model II (Coupled, Non-Convex Landscape with dense, non-diagonal Hessians).
   - *Relevance:* This setup isolates the effects of loss landscape coupling and non-convexity from other architectural confounding factors, providing a clean testbed to verify mathematical assumptions.

2. **Part II: Physical Validation on Vision Transformers (ViT-Tiny):**
   - *Design:* Evaluated on a real pre-trained Vision Transformer (`vit_tiny_patch16_224`, 14 parameter groups) fine-tuned on four diverse classification tasks: MNIST, FashionMNIST, CIFAR-10, and SVHN.
   - *Relevance:* Validates the methods under a realistic **Low-Data Deployment Regime** (2048 fine-tuning images, 10 epochs). The authors resolve the autograd graph disconnection of standard Test-Time Adaptation (TTA) frameworks to run them fairly on the physical ViT.

## Comprehensiveness of Baselines
The baselines used are highly comprehensive and represent the current state-of-the-art:
- **Upper Bound:** Task Experts (evaluating independent models on their respective datasets).
- **Uniform Baselines:** Task Arithmetic (extensively tuned across a scale factor sweep to select the optimal scale of 0.4).
- **Diagonal Curvature Baselines:** Fisher Merging (which uses the diagonal of the Fisher Information Matrix).
- **Optimization-based / Test-Time Adaptation Baselines:** AdaMerging, PolyMerge, and RegCalMerge (successfully implemented in a fully PyTorch-compatible manner).

## Hyperparameter Selection and Scientific Integrity
To prevent test-set leakage, the authors employ a rigorous, unsupervised few-shot validation split heuristic for hyperparameter selection:
- They use a tiny calibration set of 32 samples per task.
- 24 samples are allocated for projected Hessian/gradient estimation.
- 8 samples are reserved as a local validation split to select the Ridge regularization parameter $\gamma$ and Lasso penalty parameter $\mu$ by minimizing validation entropy/loss.
- This ensures zero test-set tuning data leakage, maintaining absolute scientific integrity.

## Do the Results Support the Central Claims?
Yes, the empirical results strongly support the theoretical claims of the paper:

1. **Claim: Capture of off-diagonal (cross-parameter) interactions is essential.**
   - *Evidence:* On physical ViT-Tiny, **ACM-GlobalNorm** (57.76%) and **ACM-Norm** (58.89%) outperform **Fisher Merging** (56.03%), which uses a diagonal approximation. Specifically, on CIFAR-10, ACM-GlobalNorm achieves **77.05%** accuracy compared to Fisher Merging's **66.60%** (a **+10.45%** absolute gain). This directly supports the claim that capturing off-diagonal, cross-parameter terms in the projected subspace yields superior merges.

2. **Claim: Test-Time Adaptation suffers from transductive overfitting and instability.**
   - *Evidence:* On simulated Model II, AdaMerging shows high variance ($\pm 4.58\%$). On physical ViT-Tiny, the TTA baselines collapse or degrade heavily (AdaMerging: 55.42%, PolyMerge: 38.96%, RegCalMerge: 54.27%) due to representational drift and transductive overfitting under unsupervised entropy minimization. Meanwhile, ACM is highly stable, achieving robust performance without test-time optimization.

3. **Claim: ACM behaves as a stable, training-free, and physically compliant solver.**
   - *Evidence:* Layer-wise analysis shows that ACM automatically solves untrained frozen layers (0 to 9) to exactly 0.000. For highly coupled, low-parameter layers like Layer 13 LayerNorm, it solves for negative coefficients (e.g., SVHN: -95.691), which physically corresponds to a directional interference cancellation mechanism.

4. **Claim: A local-global gap limits curvature-aware methods on highly non-convex physical manifolds.**
   - *Evidence:* The authors honestly report that an exhaustively tuned Task Arithmetic baseline achieves 60.72% (scale 0.4) on the physical ViT, slightly outperforming ACM-GlobalNorm (57.76%) and performing comparably to Vanilla ACM (60.89%). This perfectly aligns with their theoretical derivation of the local-global gap, showing that when task vectors have large magnitudes, the local quadratic approximation error scales cubically, allowing global uniform interpolation (Task Arithmetic) to act as a stronger regularizer.
