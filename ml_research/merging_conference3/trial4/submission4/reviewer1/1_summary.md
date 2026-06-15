# 1. Summary of the Paper

## Main Topic and Domain
The submission operates in the domain of post-hoc model merging, which aims to combine multiple task-specific expert neural networks (fine-tuned from a shared pre-trained base model) into a single multi-task network without joint retraining. Specifically, the paper focuses on parameterized model merging, where task-combining coefficients are optimized layer-by-layer using validation or streaming data.

## Proposed Approach: SpectralMerge
The authors challenge the standard "spatial coordinate paradigm," where layer-wise merging coefficients are treated as independent variables across the network depth. They argue that this representation is redundant, ill-conditioned, and highly prone to overfitting validation noise. 

To address this, they propose **SpectralMerge**, which re-parameterizes the merging coefficients in the frequency domain:
1. **Discrete Cosine Transform (DCT-II):** Each task's layer-wise coefficient vector $\vec{\alpha}_k \in \mathbb{R}^L$ is treated as a 1D discrete spatial signal. A forward DCT-II transforms this signal into orthogonal spectral coordinates $\vec{c}_k \in \mathbb{R}^L$.
2. **Two Regularization Schemes:**
   - **SpectralMerge-LP (Low-Pass Hard Cutoff):** Restricts optimization to the first $F$ low-frequency components ($F < L$, e.g., $F=3$), forcing all high-frequency components to zero.
   - **SpectralMerge-Reg (Soft Spectral Regularization):** Optimizes all components but penalizes high frequencies using a quadratic Spectral Decay Penalty ($\lambda_j = \mu \cdot j^2$) added to the loss function.
3. **Inverse Transform (IDCT-III):** Projects the spectral coordinates back to the spatial domain to reconstruct the physical layer coefficients $\vec{\alpha}_k$ for model weight combination.

## Explicitly Claimed Contributions and Reported Evidence
The authors explicitly claim the following contributions:

1. **New Frequency-Domain Parameterization:** Transitioning model-merging trajectory optimization from physical spatial depth to an orthogonal frequency space using DCT-II.
   - *Evidence:* Mathematical formulations of DCT-II and IDCT-III mappings, highlighting the even-symmetry boundary extension that guarantees zero-slope derivatives at the network boundaries.
2. **Structural Regularization and Orthonormality:** Presenting SpectralMerge-LP and SpectralMerge-Reg to provide analytical regularization and perfect numerical conditioning.
   - *Evidence:* A theoretical comparison of the matrix condition numbers of standard power-series polynomials (PolyMerge) and Chebyshev polynomials versus the orthonormal DCT-II basis. The authors claim that the DCT basis maintains a condition number of exactly $1.0$ at all scales, producing isotropic, well-conditioned loss contours.
3. **Empirical Advantage on Simulation Landscape:** Demonstrating superior multi-task performance over online and offline baselines.
   - *Evidence:* Evaluations on a Vision Transformer (ViT-B/32) simulation landscape (Model II) over 30 random seeds, where SpectralMerge-LP ($F=3$) and SpectralMerge-Reg achieve state-of-the-art simulated accuracies of $85.32\%$ and $85.17\%$ under online test-time adaptation, and $86.46\%$ and $86.44\%$ under offline few-shot validation tuning (OFS-Tune, $M=10$).
4. **Resolution of the "Overfitting-Optimizer Paradox" and Resilience to Bias:** Refuting the notion that optimization on small validation sets is inherently harmful, and demonstrating stability under non-stationary streams and biased validation sets.
   - *Evidence:* Sweeps over validation sample size $M \in \{5, 10, 20, 50\}$, showing that SpectralMerge avoids the validation overfitting that collapses unconstrained spatial search. Sweeps over isotropic and structured validation selection bias (up to $30\%$) show graceful degradation ($>85\%$ accuracy) compared to unconstrained spatial search.
5. **Physical Network and Real Checkpoint Validation:** Demonstrating that the findings translate to actual neural network parameters.
   - *Evidence:* 
     - A 12-layer physical PyTorch MLP model merging experiment (3 tasks), showing that SpectralMerge-Reg achieves $60.42\%$ accuracy vs $50.42\%$ for unconstrained spatial search.
     - A pre-trained ResNet-18 model ($L=18$) evaluated on CIFAR-10 tasks (2 tasks, $M=15$ samples), where SpectralMerge-Reg achieves $54.00\%$ accuracy, preventing the catastrophic collapse to $29.00\%$ (random/majority guessing) suffered by unconstrained spatial search and PolyMerge.
