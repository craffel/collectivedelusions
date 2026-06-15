# Soundness and Methodology Evaluation: "Barycentric Proximity-Anchored Merging: A Critical, Deconstructive Audit of Parameter-Frugal Test-Time Model Merging"

## 1. Description Clarity and Reproducibility
The mathematical formulation and description of the BPAM optimization pipeline are clear, structured, and easy to follow. The authors provide the exact equations, optimization constraints, learning rates, and epochs, making the study theoretically reproducible. However, several critical methodological and technical flaws undermine the soundness and validity of their approach.

## 2. Technical and Methodological Flaws

### A. Major Omission of the Zero-Shot CLIP Baseline
The authors dedicate substantial text to explaining the **"0-Weight Performance Mystery"**—i.e., how the model achieves 88.09% accuracy on MNIST and 78.15% on SVHN even when their task coefficients are exactly $0.0000$ and the base model coefficient is $0.0$. They propose elaborate hypotheses regarding "representation sharing" and "compact weight basins," backed by Centered Kernel Alignment (CKA) metrics.

However, this "mystery" is highly likely to be a trivial consequence of a missing baseline: **the zero-shot capability of the pre-trained CLIP ViT-B/32 model**. 
Standard, untouched pre-trained CLIP ViT-B/32 already possess robust zero-shot classification performance on SVHN and MNIST. By completely omitting the zero-shot base model accuracies from Table 1, the authors hide this essential baseline. If the base model already achieves ~85% on MNIST and ~75% on SVHN, then the merged model getting 88% and 78% is not a profound "reconstructed representation space" mystery; it is simply the default zero-shot capacity of the pre-trained encoder. Failing to report the zero-shot base model performance on all 8 datasets is a major methodological oversight that invalidates their "0-Weight Mystery" narrative.

### B. Unconstrained Scaling Outperforms the Proposed Convex Simplex
The core mathematical contribution of BPAM is the **Convex Barycentric Simplex Projection**, which the authors claim functions as a "critical structural safeguard" that prevents "activation scale distortion" and "catastrophic representation collapse."

Yet, their own empirical results (Table 1) show that **Unconstrained Scaling** (which completely strips away the simplex constraints and proximity penalty) consistently and substantially outperforms BPAM:
*   Under frozen heads, Unconstrained Scaling achieves **71.51%** average accuracy (+2.30% absolute improvement over BPAM-Static's 69.21%).
*   Under active heads, Unconstrained + Head Tuning achieves **77.12%** average accuracy (+1.90% absolute improvement over BPAM-Full's 75.22%).

This means their proposed scale-preserving constraint actually **degrades** performance. To justify this degradation, the authors claim that unconstrained scaling is "highly vulnerable to weight-norm instability" under larger task ensembles or longer epochs. However, **they provide absolutely no empirical evidence of this "collapse" occurring in any setting.** Methodologically, introducing a complex constraint that significantly hurts performance based on a purely theoretical fear—without demonstrating a single instance where unconstrained scaling actually fails—is highly unsound.

### C. Redundancy of the Mean-Field Proximity Penalty
The authors introduce the **Mean-Field Proximity Penalty** ($\mathcal{R}(\Lambda)$) as a key stabilizer to prevent transductive overfitting. However, their split-test ablation study (Table 4) shows that setting the regularization coefficient $\beta = 0.0$ vs. $\beta = 10^{-2}$ yields a negligible performance difference (calibration split accuracy is 68.80% vs. 68.77%, and unseen test accuracy is 69.30% vs. 69.29%). 

This proves that under standard test-time adaptation settings, **their proposed regularizer is completely redundant and useless.** The authors attempt to rescue this regularizer by evaluating an "extreme low-data" scenario (5 samples per class, Table 5) where they claim it provides stability. But introducing a complex geometric constraint for standard TTA that is only useful under highly contrived, extreme limits (5 samples) violates the very Occam's razor they advocate.

### D. Ad-Hoc Ray-Scaling vs. Exact Orthogonal Projection
To project updated coefficients back onto the simplex, the authors employ an ad-hoc **ray-scaling ($L_1$-normalization)** heuristic (Equation 13) instead of an exact orthogonal Euclidean projection. 
While they argue that ray-scaling preserves "directional ratios" and avoids the sparsification (zeroing-out) of expert weights, standard projected gradient descent (PGD) is mathematically grounded on orthogonal projections to preserve convergence guarantees. By employing an ad-hoc scaling heuristic, the authors alter the gradient optimization trajectory in a way that lacks rigorous theoretical convergence guarantees. This mathematical loose end is swept under the rug with speculative assertions.

### E. Unsound Joint Co-Adaptation of Vastly Different Scales
In **BPAM-Full**, the authors optimize 8 scalar parameters and 388,096 classification head parameters concurrently using a uniform learning rate ($\eta = 10^{-3}$) and a standard Adam optimizer. 
This is highly unsound from an optimization perspective. The classification head parameters outnumber the weight-space scalars by nearly five orders of magnitude, meaning their gradient magnitudes, landscape curvatures, and optimization dynamics are completely different. A uniform joint optimization is highly likely to cause the classification heads to rapidly overfit and dominate the loss, rendering the weight-space optimization stagnant and useless. 

The authors claim they "extended their codebase" to support asymmetric co-adaptation schedules (separate learning rates) to address this. However, **they do not present any systematic experimental results or ablations in their tables** evaluating this schedule. Making speculative claims about code extensions without presenting any concrete empirical data to back them up is methodologically weak.
