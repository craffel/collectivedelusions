# 4_experiment_check.md: Experimental Setup and Result Validation

## Experimental Setup Evaluation
The experimental design is thorough and rigorous:
- **Backbone and Datasets:** A pre-trained Vision Transformer (`vit_tiny_patch16_224`) is evaluated across four highly diverse downstream benchmarks: MNIST, FashionMNIST, CIFAR-10, and SVHN. This domain shift serves as an excellent, realistic stress-test for model merging under severe representation mismatch.
- **Data Partitions:** Using a low-data regime (512 training images, 16 unlabeled calibration images, and 2000 test images per task) directly emulates resource-constrained, data-scarce industrial IoT/edge environments.
- **Seeds and Statistics:** Evaluated across three random seeds (42, 100, 2026), reporting both the Mean and Standard Deviation. This adds immense statistical credibility and ensures the reported improvements are robust and not an artifact of random seed luck.

---

## Baselines Evaluation
The paper compares Q-PolyMerge against 23 baseline configurations, representing an exceptionally comprehensive evaluation. The main categories of baselines include:
- **Full-Precision Ceilings (FP16):** Individual Experts, Uniform Merged, AdaMerging (ES/Adam), and PolyMerge (Adam), establishing the absolute upper bounds.
- **Static Quantized Merges:** Q-then-M and M-then-Q (8-bit/4-bit), representing the unoptimized post-training quantization baseline.
- **Traditional Adaptive Merging (Post-Hoc Quantized):** AdaMerging optimized in full precision and subsequently quantized (8-bit/4-bit), which isolates the severe "representation alignment loss" caused by post-hoc rounding.
- **Quantization-Aware Merging (Q-Merge):** Direct, unconstrained optimization under the quantization operator using 56 parameters (Adam STE or 1+1 ES), isolating the exact benefits of our continuous polynomial prior.

---

## Technical Discrepancies and Reporting Anomalies
A meticulous inspection of the empirical results in the text versus the tables has revealed that the paper is highly consistent and statistically honest. The results match perfectly across the Abstract, Intro, Experiments, and Conclusion sections (e.g., Q-PolyMerge (Adam STE) is reported consistently as 59.76% under 8-Bit and 48.87% under 4-Bit PTQ). 

However, two very minor details and reporting choices are worth noting:

### 1. Minor Discrepancy in Continuous vs. Block-wise Constant Scaling Improvement
In the experiments section (Section 4.4, summary bullet points), the authors write:
> *"Our continuous trajectory strictly outperforms block-wise constant scaling by +2.13% absolute accuracy under the exact same 12-parameter budget..."*

However, in Appendix B.4, Table B.3 (and the accompanying discussion), the actual numbers are:
- `Q-PolyMerge (Adam STE)`: **`48.87%`**
- `Block-wise Constant (Adam STE)`: **`46.72%`**
The difference is `48.87% - 46.72% = 2.15%`. The main text reports `+2.13%`, which appears to be a very minor typo (or a rounding artifact from a previous run) and should be updated to `+2.15%` to match the exact table values.

### 2. Selective Reporting on the Zero-Order Block-wise Scaling Anomaly
In Section 4.4 (and Table B.3 in the appendix), the comparison between the proposed continuous polynomial trajectory and block-wise constant scaling is analyzed.
While continuous polynomial trajectory strictly outperforms block-wise constant scaling under first-order search (Adam STE) by `2.15%`, we observe the following under zero-order ES:
- `Polynomial Continuous (ES, Ours)` achieves **`43.05 \pm 1.90%`**.
- `Block-wise Constant (ES)` achieves **`43.33 \pm 4.49%`**.
Under zero-order ES, block-wise constant scaling actually slightly *outperforms* our continuous polynomial trajectory by **`0.28%`**. 
The authors do not mention this zero-order anomaly in the main text summary of Section 4.4, selectively highlighting only the first-order gradient gains. While the appendix (Section B.4) provides an excellent, mathematically rich explanation for this (attributing it to the non-smooth step-like PTQ rounding landscape where hard step-boundaries help isotropic random search escape local plateaus), the main text should ideally acknowledge this nuance or qualify the statement to refer specifically to first-order optimization.

---

## Do the Results Support the Claims?
Yes, overall the claims are strongly supported, and the empirical results are outstandingly robust:
- **Overfitting Mitigation:** Q-PolyMerge (Adam STE) reduces standard deviation in 8-bit PTQ by over 47% (1.22% vs. 2.36% for Q-Merge) and stabilizes 4-bit trajectories (1.42% vs. 4.03% for Q-Merge).
- **Extreme Parameter Efficiency:** Bypassing high-dimensional search enables 1+1 ES to converge in just 100 iterations.
- **SRAM Footprint Reduction:** The modeled SRAM footprint reduction (>97% reduction) is highly plausible and mathematically proven in the appendix.
- **Generalization:** Outperforming the naive post-merge baseline by +5.95% under 4-bit PTQ confirms that restricting the merging coefficients to a low-dimensional continuous polynomial subspace regularizes adaptation under extreme quantization rounding.
