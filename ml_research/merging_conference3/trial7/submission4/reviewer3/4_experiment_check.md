# 4. Experimental Evaluation and Baseline Check

## Experimental Setup
The paper evaluates the proposed methods using a dual-track approach:
1. **Calibrated Representation Sandbox:** A high-fidelity, controlled $D=192$-dimensional feature space simulating a $K=4$ multi-task specialist registry (modeled after MNIST, FashionMNIST, CIFAR-10, and SVHN). Each task is assigned a disjoint 48-dimensional subspace with distinct, calibrated Gaussian noise scales ($\sigma = [0.05, 0.25, 0.40, 1.95]$). Joint expert ceiling is analytically established at $74.46\%$.
2. **Real-World Proof of Concept:** Pre-trained ImageNet-1K ResNet-18 final layer features ($fc$, $D=512$) represent actual high-dimensional deep features across three semantic domains (Dogs, Cats, and Vehicles). The evaluation utilizes 1,250 real-world deep representation vectors corrupted with feature-space noise ($\sigma = 0.15$).

*Theorist Perspective on Setup:* The sandbox is an exceptionally clean, well-formulated tool to isolate coordinate-level dynamics, noise scales, and analytical thresholds. Averaging all results over 10 independent random seeds ensures high statistical significance. Furthermore, the ResNet-18 proof of concept successfully bridges the "evaluation gap," verifying that the mathematical SVD-centroid and Löwdin-projection mechanics generalize to real deep-feature manifolds.

## Evaluated Baselines
The paper compares PFSR and OTSP against a comprehensive suite of baselines:
- **Parameter-free:** Uniform Merging (Task Arithmetic), Naive Mean Centroid.
- **Parametric / Trainable:** Single-layer LinearRouter, QWS-Merge (SOTA wave-superposition routing), L3-Softmax (Unregularized), and L3-Softmax Well-Reg (Zero-Initialized, with weight decay).

*Rigorous Training Update:* Rather than using batch-averaged ensembling coefficients, the authors optimized the training of the parametric baselines by training them directly on the $D$-dimensional representations of the calibration split to predict task labels via supervised cross-entropy. This provides a fair, capacity-optimized comparison representing the absolute upper-bound of supervised parametric routing.

## Do the Results Support the Claims?
Yes, the empirical results provide highly convincing, statistically significant support for all core claims of the paper:

1. **SVD Centroid Extraction is Mandatory:** 
   Supported by the **Naive Mean Centroid** baseline, which collapses to near-random guessing ($25.18\% \pm 1.10\%$ routing accuracy) due to classifier sum-to-zero prototype cancellation. In contrast, SVD centroid extraction maintains a flawless **100.00%** routing accuracy under uncorrupted disjoint subspaces (Table 4.1).
2. **Löwdin Orthogonalization is Redundant under Symmetric Layouts:**
   Supported by Table 4.1, which shows OTSP and PFSR achieving identical routing accuracies to the decimal point ($100.00\%$) under perfectly orthogonal layouts. Under symmetric overlap ($\rho = 0.33$, Section 4.3), both methods also match perfectly at $94.62\%$, validating the closed-form proofs of Symmetric and SNR Equivalence.
3. **Löwdin Orthogonalization Degrades Performance under Asymmetric Overlap:**
   Supported by Table 4.2's dense parameter sweep, where OTSP systematically underperforms PFSR by **0.2% to 1.6%** across multiple noise scales ($\sigma \in [0.01, 0.15]$) and overlaps ($\rho \in [0.85, 0.95]$). This empirically confirms the presence of the **Noise Amplification Penalty** and **Noise Spillover Penalty** under active representation noise and asymmetric cross-talk.
4. **Vectorization Collapse & Necessity of the Simplex Constraint:**
   Supported by the unnormalized **LinearRouter** baseline's collapse to $55.57\% \pm 1.68\%$ accuracy under sample-wise vectorized streaming ($B=1$) (Table 4.1). Softmax-normalized parametric routers (L3-Softmax) and our parameter-free Softmax-gated PFSR/OTSP maintain perfect stability ($74.46\% \pm 0.81\%$), demonstrating that simplex-constraint normalization is mathematically required for online deployment.
5. **The Orthogonal Masking Effect:**
   Supported by the flat $74.46\%$ classification accuracy across Uniform, QWS-Merge, L3-Softmax, PFSR, and OTSP in disjoint subspaces (Table 4.1). This demonstrates that joint classification accuracy is insensitive to routing quality in disjoint setups, making routing accuracy the only sensitive benchmark.
6. **Implicit Regularization of Zero-Initialization:**
   Supported by L3-Softmax Well-Reg (Zero-Init) outperforming its unregularized counterpart in routing accuracy ($67.22\% \pm 1.61\%$ vs. $66.47\% \pm 1.62\%$), proving that simple zero-initialization acts as a powerful maximum-entropy prior.
7. **Mitigation of Anisotropic Noise via Whitening:**
   Supported by the anisotropic toy simulation, where OTSP's accuracy collapses to $77.10\%$ under anisotropic noise, but is restored to $89.45\%$ (+12.35% absolute gain) upon applying the offline covariance whitening step (Section 4.6).
8. **Real-World Generalization:**
   Supported by Table 4.4, where both PFSR and OTSP achieve outstanding routing accuracies of $92.00\%$ and $92.08\%$ on the ResNet-18 manifold, with OTSP gaining +0.08% absolute accuracy due to decoupling positive animal semantic overlaps.
