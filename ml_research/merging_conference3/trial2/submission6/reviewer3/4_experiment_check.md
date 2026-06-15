# 4. Experimental Evaluation and Results Check

## Evaluation of the Experimental Setup
The experimental setup designed by the authors is remarkably thorough, systematic, and intellectually honest. All experiments are averaged across **three independent random seeds (42, 100, and 2026)** with reported standard deviations. This ensures statistical rigor and prevents "seed-cherry-picking" which is a common issue in machine learning papers.

### Confounding Factors Decoupled:
The primary strength of the empirical evaluation is how carefully the authors decouple multiple potential confounding factors to isolate the source of their method's performance gains:
1. **The Merging Penalty vs. Quantization Penalty:** By evaluating individual, unmerged experts (both unquantized and quantized) alongside merged models, the authors isolate the exact performance losses attributable to task interference (weight-space sharing) versus quantization rounding noise.
2. **The Optimizer Confounding Factor:** Rather than comparing their Adam GD Q-Merge against standard 1+1 ES AdaMerging and claiming a "quantization regularization" effect, the authors implement an unquantized *AdaMerging (Adam GD)* baseline. This fair, optimizer-controlled comparison reveals that:
   - Under 8-bit quantization, Q-Merge is nearly lossless relative to the unquantized Adam ceiling (74.30% vs. 74.38%). The apparent "boost" over standard AdaMerging is mostly due to the transition to a superior first-order optimizer, which the Straight-Through Estimator (STE) elegantly enables.
   - Under 4-bit quantization, Q-Merge optimized with Adam GD + STE achieves **63.36%**, strictly outperforming the unquantized Adam GD baseline followed by post-hoc quantization (**62.01%**). This proves that when quantization noise is high, optimization-aware merging is a scientific necessity.
3. **PTQ Rounding vs. Global Coordinate Alignment:** The authors compare Q-Merge against advanced PTQ reconstruction algorithms like **AdaRound** (applied post-hoc to merged weights). They demonstrate that Q-Merge outperforms AdaMerging+AdaRound (63.36% vs. 59.34%), proving that global weight-manifold alignment is far more effective than local discrete rounding. They also show the two paradigms are complementary (Q-Merge + AdaRound yields the highest overall score of 64.46%).

---

## Datasets and Backbones
* **Backbone:** The paper uses a pre-trained **timm ViT-Tiny** backbone (`vit_tiny_patch16_224`, 5.7M parameters). This is an appropriate and practical choice representing lightweight edge deployment.
* **Datasets:** The evaluation spans a diverse 4-task classification benchmark: MNIST, FashionMNIST, CIFAR-10, and SVHN.
* **Scale Limitation (Explicitly Acknowledged):** As the authors openly discuss in Section 5.2, this is a relatively small/toy-scale setup. The experts are trained under a low-data regime (512 images per task), leading to low-parameter drift from the base pre-trained checkpoint. While this setup is ideal for rapid iterative research, multi-seed ablation, and deep controlled scientific analysis, further validation on large-scale generative models (e.g., multi-billion parameter LLMs or high-resolution Diffusion experts) under high-parameter drift is necessary to fully confirm generalizability. The authors' transparency about this limitation is highly commendable.

---

## Do the Results Support the Claims?
Yes, the quantitative results in Table 1 and Table 2, alongside the ablation studies, provide overwhelming and watertight support for every single claimed contribution:
* **8-Bit Performance:** 8-bit Q-Merge (Adam GD with STE) achieves **74.30%** average accuracy, nearly matching the unquantized Adam ceiling of **74.38%** (99.9% recovery) and outperforming the uniform FP16 baseline (**71.88%**).
* **First-Order vs. Zero-Order Stability:** First-order Adam GD with STE consistently outperforms the derivative-free 1+1 ES across both bitwidths. It achieves higher mean accuracy (74.30% vs. 72.57% in 8-bit; 63.36% vs. 57.83% in 4-bit) and over **2.7x lower seed-to-seed standard deviation** (0.38% vs. 1.06% in 8-bit). This demonstrates that STE-based backpropagation is highly stable.
* **Correcting the 4-Bit Collapse:** Under per-tensor INT4, standard merging collapses to random guess levels (~11-12%). However, using standard per-channel (channel-wise) weight quantization, the naive baseline achieves **56.66%**, and Q-Merge with STE elevates this to **63.36%**, proving that extreme low-bit model merging is indeed viable.
* **Low Overhead and Calibration Stability:**
  - Standard deviation across seeds remains low.
  - High accuracy is stable down to a tiny calibration split of only 8 images per task (76.95% in 8-bit, 59.77% in 4-bit).
  - Wall-clock adaptation is extremely fast, requiring only **2.43 seconds** on CPU and **80 milliseconds** on GPU.
  - On-device task stream skew is successfully mitigated using the Confidence-Based FIFO Stratification heuristic (Scenario C), restoring average accuracy to 76.95% (8-bit) and 59.77% (4-bit) under extreme unbalanced inputs.

The empirical evidence is complete, highly detailed, and thoroughly validated. No claims are overstated, and all baselines are correctly and fairly represented.
