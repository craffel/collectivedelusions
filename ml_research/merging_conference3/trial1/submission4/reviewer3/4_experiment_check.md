# Evaluation Component 4: Experimental Evaluation and Claims Check

## 1. Experimental Setup and Benchmarking
The experimental evaluation is highly rigorous, comprehensive, and stands out for its methodical design:
* **Architecture & Datasets:** The authors utilize a standard Vision Transformer backbone (**ViT-B-32** with 86M parameters) across a robust, diverse benchmark of 8 standard image classification datasets (SUN397, Stanford Cars, RESISC45, EuroSAT, SVHN, GTSRB, MNIST, DTD). This is an appropriate and standard benchmark for evaluating multi-task model merging in computer vision.
* **Baseline Coverage:** The baselines evaluated are extensive and highly representative of the state-of-the-art:
    * *Dynamic TTA Baselines:* SyMerge (Jung et al., 2025), AdaMerging (Yang et al., 2024), and Task Surgery (Zhang et al., 2024).
    * *Static Merging Baselines:* Task Arithmetic (Ilharco et al., 2023), Ties-Merging (Yadav et al., 2024), and OrthoMerge (Yang et al., 2026).
    * *Control Ablations:* "Static TA + Head-Only Tuning" and "L2 Weight Anchoring (at TA)".
* **Standardization:** All test-time adaptation methods are optimized under identical, standardized TTA batch settings (1000 randomly sampled validation images, batch size 32). This guarantees a fair, apples-to-apples comparison.

---

## 2. Evaluation Protocols and Insightful Analyses

The paper features two distinct evaluation protocols that yield deep diagnostic insights:

### A. The Synergy-Refinement Protocol (Main Evaluation)
All methods (including FluidMerge and the three baselines) are initialized starting from the Task Arithmetic weight-space average ($\theta(0) = \theta_{\text{TA}}$). 
* This is a highly robust and fair protocol because it places all methods inside a high-performing multi-task basin from the outset, evaluating their capacity to refine and align representations.
* **Results Support Claims:** Under this protocol, FluidMerge-Fisher (Ours) achieves **59.34%** average Top-1 accuracy, outperforming:
    * Static Task Arithmetic (**57.74%**, +1.60% absolute gain)
    * Static TA + Head Tuning control (**58.12%**, +1.22% absolute gain)
    * AdaMerging at TA (**58.04%**, +1.30% absolute gain)
    * Task Surgery at TA (**58.23%**, +1.11% absolute gain)
    * SyMerge at TA (**58.42%**, +0.92% absolute gain)
    * L2 Weight Anchoring (**58.48%**, +0.86% absolute gain)
    * FluidMerge with Spatial Laplacian (**54.76%**, proving that grid-based Laplacians suffer from representation tearing).
* **Calibration Preservation:** FluidMerge-Fisher achieves the lowest Expected Calibration Error (**7.18%** vs. **8.75%** for L2 Anchoring and **9.23%** for Head-Tuning), proving that function-sensitive regularized updates successfully stabilize confidence calibration.

### B. The Boundary Stress-Test (Diagnostic Protocol)
All methods are initialized starting from the raw, unadapted pretrained base encoder weights ($\theta_0$) to evaluate their capacity to adapt representations post-hoc from scratch.
* **Scientific Insights:** This protocol is a diagnostic stress-test. It reveals that both FluidMerge and all three baselines fail completely, staying locked at random-guessing levels (~5%). 
* **Calibration Collapse:** The authors diagnose that unconstrained self-training from scratch causes the Expected Calibration Error (ECE) to explode to over 90% because classification heads quickly overfit to teacher pseudo-labels on top of unaligned representations. This is an extremely valuable and honest scientific analysis that defines clear boundaries for post-hoc adaptation.

---

## 3. Mathematical Rigor and Statistical Significance
The empirical results are backed by strong mathematical validation:
* **Statistical Significance:** The authors conduct paired two-tailed t-tests across the 8 datasets, demonstrating that FluidMerge-Fisher's improvements are highly statistically significant:
    * vs. Static TA: $t = 11.573, p = 8.0 \times 10^{-6}$ ($p < 0.0001$)
    * vs. L2 Weight Anchoring: $t = 7.883, p = 1.0 \times 10^{-4}$ ($p < 0.001$)
    * vs. SyMerge: $t = 9.040, p = 4.1 \times 10^{-5}$ ($p < 0.0001$)
* **Variability:** The authors report results over 3 random seeds, showing extremely low run-to-run variability (standard deviation $\le 0.15\%$). This establishes that the reported gains are robust and not artifacts of optimization noise.

---

## 4. Critique of Claims and Limitations

* **Claim: FluidMerge "resolves" the domain shift barrier.**
    * *Critique:* The physical "fluid flow" itself does **not** resolve the domain shift barrier; rather, the **Task Arithmetic initialization** does. Under the Boundary Stress-Test (initialized at $\theta_0$), FluidMerge fails completely. Therefore, the continuous-time trajectory is an *adaptive refinement* step on top of a viable initialization, not an independent solver that can traverse extreme representation gaps from scratch. The authors are mostly honest about this, but they should tone down statements that imply "FluidMerge" alone resolves the barrier.
* **Is the modest improvement worth the cost?**
    * *Critique:* FluidMerge-Fisher requires **20.5 minutes** of premium A100 GPU compute and **14.8 GB** of memory. It yields a **1.60%** improvement over static Task Arithmetic (0 seconds compute) and a **1.22%** improvement over the "Static TA + Head Tuning" baseline (which only takes seconds). In practical edge deployments, this tiny gain does not justify the massive full-encoder backpropagation cost. While the authors transparently acknowledge this and frame FluidMerge as a "high-capacity research tool" or "upper bound," this practical limitation is a major bottleneck.
