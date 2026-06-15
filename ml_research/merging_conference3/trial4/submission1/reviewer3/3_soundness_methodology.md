# 3. Soundness and Methodology Evaluation

## Clarity of Description
The mathematical formulation of PhaseMerge and U-PhaseMerge is described with excellent clarity. The step-by-step deconstruction of weights, projection into 2D Real FFT space, decomposition into amplitude and phase, and the continuous phase rotations are easy to follow. Crucially, the mathematical proof in Section 3.4 establishing the spatial domain dual (the equivalence of uniform phase-rotation to a coordinate-dependent directional Hilbert transform) is rigorous, elegant, and highly informative.

## Appropriateness of Methods
1. **Representational Mismatch:** The primary methodological concern is the choice of the 2D Fast Fourier Transform (RFFT2D) on dense linear weight matrices (`qkv`, `proj`, `fc1`, `fc2`). Fourier transforms operate on spatial grids where neighboring dimensions have geometric correlations (e.g., in images or convolutions). Dense layer matrices are permutation-invariant and do not have an inherent 2D topology. While the authors address this by locking the rows/columns to the pre-trained initialization backbone ($W_{\text{pre}}$) and viewing it as a matrix-basis regularizer, it remains a structurally contrived assumption. This is highlighted by the fact that the upsampled 2D spatial frequency smoothing grid ($r=2$) performs *worse* than the uniform phase rotation ($r=1$).
2. **Evaluation Scale (Toy Benchmarks):** Evaluating only on a tiny model (`vit_tiny_patch16_224`, ~5.7M parameters) and four small-scale, highly conflicting vision datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) limits the generalizability of the findings. In modern machine learning, weight-space merging is most critical for Large Language Models (LLMs) and massive vision-language foundations. It remains unproven whether the complex Fourier transformations and Straight-Through Estimators (STE) can scale effectively to massive multi-billion parameter architectures.

## Potential Technical Flaws
1. **Unjustified Empirical Claims:** The paper claims that U-PhaseMerge ($r=1$) "*exhibits outstanding robustness to extreme 4-bit quantization, outperforming traditional real-space layer-wise optimizations*" (such as AdaMerging). However, looking closely at **Table 1** (under 4-bit PTQ) and **Table 3** (Target Schema Shift under 4-bit PTQ):
   - **Table 1 (4-bit PTQ):** AdaMerging achieves $37.50 \pm 1.22\%$, while U-PhaseMerge achieves $37.42 \pm 1.94\%$.
   - **Table 3 (Target Schema Shift under 4-bit PTQ):** AdaMerging achieves $37.50 \pm 1.22\%$, while U-PhaseMerge achieves $37.42 \pm 1.94\%$.
   
   In both cases, U-PhaseMerge actually *underperforms* AdaMerging (the traditional real-space layer-wise optimization baseline) by $0.08\%$ absolute accuracy, while showing significantly higher variance (standard deviation of $1.94\%$ vs. $1.22\%$). Thus, the paper's claims regarding U-PhaseMerge's superiority under extreme 4-bit quantization and target schema shifts are empirically contradicted by its own tables.

2. **Severe Optimization Instability (Scaling Failure):** In **Table 2** (Sample Complexity Sweep under 8-bit PTQ):
   - As the calibration dataset size $M$ increases from $16$ to $32$:
     - AdaMerging's accuracy increases from $41.67 \pm 1.45\%$ to $42.50 \pm 1.59\%$.
     - PhaseMerge ($r=2$) increases from $40.83 \pm 1.18\%$ to $42.00 \pm 1.34\%$.
     - PolyMerge remains stable at $47.83 \pm 1.03\%$.
     - **U-PhaseMerge ($r=1$), however, drops significantly from $42.33 \pm 1.76\%$ to $40.67 \pm 3.65\%$**, and its standard deviation more than doubles to $3.65\%$.
   
   This indicates a fundamental scalability and stability issue. In standard machine learning optimization, having more calibration data should improve generalization (as seen with AdaMerging and PhaseMerge $r=2$). The fact that U-PhaseMerge's performance collapses and variance spikes when data is increased to $M=32$ suggests that its optimization trajectory is highly unstable and prone to local minima or overfitting-optimizer dynamics that its $L_2$ phase decay penalty fails to adequately regularize. This instability is a major technical flaw and a practical blocker for real-world deployment.

## Reproducibility
The paper provides a high level of detail regarding hyperparameters, optimizer selection (Adam with learning rate $1\times 10^{-2}$ for 5 steps), backbone architecture (`vit_tiny_patch16_224`), target layer selections (48 dense weight matrices), and calibration set sizes ($M \in \{4, 16, 32\}$). The mathematical algorithms are thoroughly written. The work is highly reproducible for an expert reader.
