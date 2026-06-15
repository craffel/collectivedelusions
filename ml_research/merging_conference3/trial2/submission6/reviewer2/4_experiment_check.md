# Evaluation Step 4: Experimental Check

## Evaluation of Experimental Setup
The experimental setup is exceptionally rigorous, comprehensive, and well-designed:
- **Datasets & Backbone:** The choice of a timm ViT-Tiny backbone (5.7M parameters) evaluated across a four-task vision benchmark (MNIST, FashionMNIST, CIFAR-10, SVHN) is highly appropriate. ViT-Tiny is a perfect model for edge deployment where memory and computation are scarce.
- **Statistical Rigor:** All experiments are executed across **three independent random trials and seeds (42, 100, 2026)** with mean and standard deviation reported. This prevents selective reporting and confirms statistical significance.
- **Low-Data Realism:** Fine-tuning task experts on only 512 images per task and optimizing coefficients on a compact 16-image-per-task (64 total) unlabeled calibration set simulates a highly realistic, data-scarce test-time edge adaptation scenario.
- **Ablation Scale:** The paper includes extensive analyses covering: calibration set size sensitivity (8, 16, 64 images per task), scale factor fixed-point precision ($N_{\text{fraction}} \in \{8, 16, 32\}$), dynamic activation quantization (W8A8 and W4A4), and non-stationary imbalanced calibration streams.

## Appropriateness of Baselines
The authors compare Q-Merge against five representative paradigms, which cover all standard deployment pipelines:
1. **Unoptimized, Unquantized:** FP16 Merged Model (Uniform).
2. **Optimized, Unquantized:** AdaMerging (FP16 Optimized, ES and Adam variants).
3. **Discrete Experts Merged:** Quantize-then-Merge (Q-then-M).
4. **Merged then Naively Quantized:** Merge-then-Quantize (M-then-Q).
5. **Optimized in FP16, then Quantized:** AdaMerging (FP16 Opt, ES and Adam, Quantized).

Furthermore, the paper compares Q-Merge against sophisticated post-training quantization (PTQ) algorithms (such as AdaRound) applied post-hoc to merged checkpoints, and evaluates a sequential combined pipeline (Q-Merge + AdaRound). This ensures that the benefits of Q-Merge are not simply a result of using standard PTQ algorithms, but are uniquely driven by global coordinate alignment.

## Do the Results Support the Claims?
Yes, the results fully and honestly support all of the paper's claims:
1. **Claim: Q-Merge overcomes the 8-bit quantization gap.**
   - *Evidence:* Table 1 shows that naive post-merge quantization (M-then-Q) drops to 71.71%. Q-Merge (Adam GD with STE) achieves **74.30%**, outperforming the unquantized uniform FP16 baseline (71.88%) and recovering 99.9% of the unquantized optimized Adam ceiling (74.38%).
2. **Claim: The authors honestly deconstruct the "surpassing unquantized ceiling" confounding factor.**
   - *Evidence:* Section 4.4.2 provides a highly honest, scientifically rigorous decomposition of optimizer effects. The apparent "regularization" of 8-bit Q-Merge over standard AdaMerging (74.30% vs. 73.21%) is shown to be a confounding factor of transitioning from a zero-order optimizer (1+1 ES) to a first-order optimizer (Adam GD). When compared under the same optimizer, unquantized AdaMerging (Adam) is 74.38% vs. Q-Merge (Adam) 74.30% (a tiny, expected representation loss of 0.08%). Under 4-bit, Q-Merge with STE actively outperforms post-hoc quantized AdaMerging by **1.35% absolute** ($63.36\%$ vs $62.01\%$), showing that when quantization noise is high, quantization-aware optimization is a necessity.
3. **Claim: First-order STE is highly superior and more stable than zero-order 1+1 ES.**
   - *Evidence:* In Table 1 (8-bit), Q-Merge (Adam GD) achieves 74.30% ± 0.38% (a 2.7x reduction in standard deviation over 1+1 ES's 72.57% ± 1.06%). In Table 2 (4-bit), Adam GD achieves **63.36% ± 1.18%** while 1+1 ES struggles at **57.83% ± 1.47%**.
4. **Claim: 4-bit model merging is viable with per-channel quantization.**
   - *Evidence:* Table 2 shows that using per-channel weight quantization prevents catastrophic model collapse (which is a known failure mode for per-tensor 4-bit). Standalone 4-bit Q-Merge (Adam GD) achieves **63.36%** average accuracy, which is only **2.28%** below unmerged 4-bit experts (65.64%), outperforming naive post-merge quantization (56.66%) by **6.70%** absolute.
5. **Claim: Q-Merge is highly complementary to advanced PTQ rounding algorithms.**
   - *Evidence:* Standalone Q-Merge (63.36%) outpaces Uniform+AdaRound (58.12%) and AdaMerging+AdaRound (59.34%). Merging them sequentially (Stage 1: Q-Merge, Stage 2: AdaRound) achieves the state-of-the-art accuracy of **64.46%** (recovering an additional 1.10% absolute), proving that global coordinate-alignment and local reconstruction optimization are highly complementary.
6. **Claim: Q-Merge has low overhead and high data efficiency.**
   - *Evidence:* Table 4 shows average accuracy is extremely stable across calibration batch sizes (from 8 to 64 images per task). Section 4.8 reports that the adaptation completes in only **2.43 seconds** on an 8-core CPU or **80 ms** on an NVIDIA A100 GPU.

## Critical Analysis
The only minor limitation in the experimental scope is that it is conducted on a toy-scale backbone (ViT-Tiny, 5.7M parameters) and task Experts trained under a low-data regime (512 images per task), leading to low-capacity models with minimal parameter drift. The SVHN unmerged expert achieves a low performance of 41.34% because of this.

However, the authors **explicitly and honestly acknowledge this limitation** in Section 5.2, labeling it a "toy-scale configuration" and "low-parameter-drift regime." They provide a comprehensive theoretical and scaling analysis of how Q-Merge scales to larger architectures (includingCLIP ViT-B, LLaMA-1B, and LLaMA-7B) in Appendix B.4 and B.5, proving that the search space size remains virtually static ($56 \to 128$ parameters) and that layer-wise blending acts as an exceptionally powerful structural regularizer. 

Therefore, the experimental findings are complete, honest, and highly robust.
