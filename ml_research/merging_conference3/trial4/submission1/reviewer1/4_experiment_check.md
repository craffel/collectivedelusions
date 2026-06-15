# 4. Experimental Evaluation & Empirical Integrity

## Critique of the Experimental Setup
The experimental setup of this paper is extremely limited and does not meet the standard conventions of modern model-merging research. It can be characterized as a "toy-scale" sandbox evaluation:

1. **Inadequate Model Scale:**
   * The paper evaluates strictly on `vit_tiny_patch16_224`. This is an extremely small, obsolete model (approximately 5.7 million parameters).
   * Modern model-merging literature evaluates on substantially larger models (e.g., ViT-Base, ViT-Large, and multi-billion parameter LLMs like LLaMA-7B or Mistral-7B) to demonstrate practical utility. A tiny 5M parameter model is insufficient to prove that the proposed frequency-domain method scales or generalizes to modern architectures.

2. **Toy Datasets and Severe Data Constraints:**
   * The downstream tasks are limited to four simple image classification datasets: MNIST, FashionMNIST, CIFAR-10, and SVHN.
   * To establish the "experts," each model is trained on a mere **500 samples**. 
   * More critically, the test sets are subsampled to only **100 samples per task**. 

3. **High Statistical Noise and Marginal Significance:**
   * Evaluating on a test set of 100 samples introduces extreme statistical noise. A $1\%$ accuracy change corresponds to exactly 1 sample being correctly or incorrectly classified.
   * This tiny test set size explains the very high standard deviations reported in the paper (e.g., $\pm 3.65\%$ or $\pm 4.92\%$). Because the variance across random seeds is so high relative to the performance gaps, many of the paper's claims are statistically marginal or insignificant.

## Evaluation of Claims vs. Actual Results

### Claim 1: "U-PhaseMerge exhibits exceptional robustness under extreme 4-bit quantization."
* **The Reality (Table 1):** In the 4-bit PTQ regime, PolyMerge achieves $43.42 \pm 1.30\%$ accuracy, while U-PhaseMerge achieves $37.42 \pm 1.94\%$, and AdaMerging achieves $37.50 \pm 1.22\%$. 
* U-PhaseMerge is actually **worse** than the unconstrained real-space AdaMerging baseline by $0.08\%$ and is **6.00% absolute accuracy worse** than PolyMerge. Calling this "exceptional robustness" is a massive overstatement; it performs comparably to or worse than standard real-space optimization.

### Claim 2: "PhaseMerge resolves the Overfitting-Optimizer Paradox."
* **The Reality (Table 2):** In the sample complexity sweep under 8-bit PTQ:
  * At $M=4$, U-PhaseMerge ($42.42 \pm 1.64\%$) is slightly better than AdaMerging ($40.75 \pm 1.24\%$), but the difference is within the overlapping standard deviations. Meanwhile, PolyMerge is dramatically better at $47.25 \pm 1.87\%$.
  * As the calibration stream size increases to $M=32$, U-PhaseMerge's performance actually **degrades** to $40.67 \pm 3.65\%$ (even when using the $L_2$ phase decay regularization described in Section A.4.2). 
  * In a stable, regularized system, more calibration data should improve or maintain performance. The fact that U-PhaseMerge's performance drops as $M$ increases from 4 to 32, accompanied by a doubling of the standard deviation (from $1.64\%$ to $3.65\%$), indicates **severe optimization instability** and a failure to resolve the Overfitting-Optimizer Paradox.

### Claim 3: "PhaseMerge exhibits strong generalizability to target deployment schema shift."
* **The Reality (Table 3):** Under 8-bit calibration and 4-bit deployment:
  * PolyMerge achieves $43.42 \pm 1.30\%$.
  * AdaMerging achieves $37.50 \pm 1.22\%$.
  * U-PhaseMerge achieves $37.42 \pm 1.94\%$.
  * PhaseMerge achieves $36.92 \pm 0.92\%$.
* Again, the proposed methods (U-PhaseMerge and PhaseMerge) are worse than both PolyMerge and AdaMerging under target schema shift. There is no empirical evidence of "strong generalizability" compared to standard baselines.

## Baseline Suitability
The choice of baselines (Uniform TA, FREE-Merging, AdaMerging, and PolyMerge) is appropriate and represents the standard benchmarks. However, the fact that the strongest baseline (PolyMerge) consistently outperforms PhaseMerge/U-PhaseMerge by a large margin across *all* experiments severely weakens the paper's core motivation. The paper fails to make a compelling empirical case for adopting complex-valued frequency-domain phase rotations over simpler real-space depth-wise polynomial constraints.
