# 4. Experimental Setup and Baseline Evaluation

## Critical Evaluation of the Experimental Setup
The experimental setup is systematically structured, but contains a few significant limitations that warrant a critical scholarly review:
1. **Severe Under-Training of Task Experts:**
   - As the authors transparently acknowledge in the limitations section, the task experts are severely under-trained due to a highly restricted local compute budget. They are fine-tuned on only 256 training images for 3 epochs.
   - This leads to exceptionally low individual validation accuracies: 82.03% (MNIST), 81.25% (FashionMNIST), 74.22% (CIFAR-10), and 28.91% (SVHN). 
   - In particular, the SVHN expert’s accuracy of 28.91% is barely above random guessing (~10%). This drag lowers the uniform FP16 ensembling ceiling to a modest 38.67%.
   - **Critique:** Undertrained experts have highly distinct weight distributions, smaller task vector magnitudes, and unstable decision boundaries compared to fully converged models. While this low-compute setup is a valuable proxy for analyzing PTQ sensitivity, it remains an open question whether OmniMerge’s findings (such as the quantization denoising effect and the cross-schema performance gap) hold true when using highly optimized, fully converged task experts. Evaluating on fully converged experts is crucial to validate the generalizability of the framework.

2. **Calibration and Evaluation Size:**
   - The calibration set uses $N_{\text{cal}} = 64$ unlabeled images per task (256 total), and the evaluation set uses $N_{\text{eval}} = 256$ images per task (1024 total).
   - This evaluation set size is relatively small (e.g., standard CIFAR-10 test set is 10k images). With only 1024 total images, the binomial standard error is approximately $\sqrt{p(1-p)/N} \approx 1.56\%$ for a performance of ~50%. Small performance fluctuations (such as the +0.39% gain from quantization) lie well within this margin of error.

3. **Selective Quantization Policy:**
   - Quantizing only the weights of transformer encoder blocks and projection layers, while leaving biases, LayerNorm layers, patch embeddings, and classification heads in FP16, is standard for maintaining high-precision representation in sensitive layers. However, this means the model is not fully quantized to 8-bit integers, which should be explicitly noted when discussing edge deployment footprint reductions.

## Baseline Quality
The set of baselines chosen for comparison is outstanding and highly rigorous:
- **FP16 Task Arithmetic (Ceiling):** Uniform blending (0.3) in FP16, acting as the baseline ensembling ceiling.
- **AdaMerging (FP16):** Optimizes coefficients in FP16 using prediction entropy and TCR on the calibration set.
- **Quantized AdaMerging:** Optimizes coefficients in FP16 (under AdaMerging) and then applies post-hoc quantization to the target hardware schema. This is an essential baseline that isolates the benefits of stochastically co-optimizing *directly* under simulated quantization operators.
- **Q-Merge (Symmetric Per-Channel):** Optimizes coefficients directly under a single static quantization operator (Symmetric Per-Channel) using STE.

## Do the Results Support the Claims?
Yes, the results convincingly support the primary claims of the paper:
- **Overcoming Cross-Schema Degradation:** Table 1 demonstrates that standard Q-Merge optimized under Symmetric Per-Channel suffers a drop to 45.90% under Symmetric Per-Tensor. In contrast, OmniMerge achieves 50.39% on Symmetric Per-Tensor, representing a massive **+4.49%** improvement.
- **Out-of-Pool Generalization:** Crucially, under the Double Quantization schema (which was excluded from OmniMerge’s stochastic operator pool during test-time co-optimization), OmniMerge achieves **50.29%**, outperforming Q-Merge (46.58%) and Quantized AdaMerging (46.68%). This provides convincing empirical evidence of true schema-invariance and generalizable robustness.
- **Ablation Validity:** Table 2 isolates the benefits of SOS (+3.50% gain) and SZNP (+3.77% gain) over the baseline + TCR, showing that both mechanisms are essential. The discussion of the slight "compound stochasticity" over-regularization effect (causing a minor 0.12% drop when combining both on the training stream) is exceptionally honest and scientifically mature.

## The "Denoising" Hypothesis and FP16 Performance
An intriguing finding is that OmniMerge quantized under Symmetric Per-Channel achieves **50.78%** accuracy, outperforming its own unquantized FP16 continuous counterpart (**50.39%**) by **+0.39%** absolute.
- The authors hypothesize that weight discretization (quantization) can act as a beneficial noise filter or hard-thresholding operator that reduces destructive task vector interference in merged weight-spaces.
- While this hypothesis is highly compelling and mathematically possible, the authors are highly transparent that this +0.39% improvement is statistically modest and lies within the binomial standard error ($\approx 1.56\%$).
- Crucially, the coefficients optimized by OmniMerge (using SOS + SZNP) achieve **50.39%** in FP16, which is significantly superior to standard AdaMerging (FP16) which achieves **46.68%**. This indicates that the main benefit of OmniMerge is that the stochastic operator sampling and scale noise perturbations act as powerful test-time regularizers, guiding the coefficient optimizer toward a fundamentally flatter, more generalizable ensembling minimum, even if the model is ultimately deployed in full-precision (FP16).
