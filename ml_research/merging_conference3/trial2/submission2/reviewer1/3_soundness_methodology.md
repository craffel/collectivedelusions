# 3. Soundness and Methodology Evaluation

## Methodological Clarity
The mathematical formulation of **Singular Value Slicing (SVS)** is exceptionally clear, precise, and standard. The step-by-step description of:
*   Defining task vectors ($T_t = W_t - W_0$).
*   Performing SVD ($T_t = U_t \Sigma_t V_t^T$).
*   Truncating to rank $k$ ($\mathcal{S}_k(T_t)$) using the Eckart-Young-Mirsky Theorem.
*   Linearly combining the low-rank task vectors.
*   Barycentric Weight Normalization (BWN).
is highly transparent and mathematically rigorous.

## Appropriateness of Methods
*   **SVD as a noise filter:** SVD is a theoretically sound and well-established method for low-rank matrix approximation. It is highly appropriate for filtering out "high-frequency noise" (represented by small singular values) under the assumption that fine-tuning trajectories reside in a low-rank subspace.
*   **Tensor flattening:** The decision to flatten higher-dimensional parameter tensors (e.g., convolutional kernels, patch embeddings) by grouping output channels and flattening all other dimensions is standard in deep learning SVD contexts. However, the authors note that this choice can affect the singular spectrum, and they did not conduct a sensitivity analysis on alternative flattening axes, which is a minor methodological gap.
*   **Scale-Invariance Proofs:** The mathematical derivations in Section 3.4 for L2-normalization, LayerNorm, and RMSNorm are elegant, rigorous, and correct. The conclusion that global positive weight scaling is neutralized by these layers is technically sound and represents a valuable insight for the model-merging literature.
*   **Entropy-SVS:** Defining spectral complexity via normalized Shannon entropy of singular values is a solid information-theoretic approach. It allows dynamic, rather than uniform, rank allocation.

## Technical Flaws and Empirical Weaknesses (The Empiricist Perspective)

Despite the clean mathematical formulation, there are major weaknesses in the empirical validation of the methodology:

### 1. Extremely Weak performance and Poor Training of the MLP Baselines
To validate BWN in un-normalized architectures (where scaling is not neutralized), the authors train a 3-layer MLP on MNIST and FashionMNIST (Section 4.5). However, the reported performances of the individual experts are exceptionally poor:
*   **Expert A (MNIST):** $77.00\%$ accuracy. (A simple linear model or 2-layer MLP on MNIST should easily achieve $>95-98\%$ accuracy).
*   **Expert B (FashionMNIST):** $69.00\%$ accuracy. (A standard MLP should easily get $>85\%$ accuracy).
These extremely low expert accuracies suggest that the training was severely under-optimized, terminated prematurely, or implemented incorrectly. 
Evaluating model merging on "broken" or poorly trained experts undermines the validity of the conclusions. If the experts themselves have not converged, their task vectors represent incomplete optimization trajectories rather than converged semantic manifolds, making any merging analysis highly suspect.

### 2. Statistically Insignificant Gains
The performance improvements attributed to BWN in the MLP environment are extremely marginal and lack any statistical validation:
*   At $\lambda=0.1$, accuracy improves from $29.50\%$ (without BWN) to $30.25\%$ (with BWN) — an improvement of only **$0.75\%$** at an overall accuracy level that is barely above random guessing.
*   At rank $k=8$, accuracy improves from $37.25\%$ to $37.50\%$ — an improvement of only **$0.25\%$**.
Because there are **no multiple random seeds, confidence intervals, or standard deviations** reported, these tiny differences are statistically indistinguishable from random noise. An empiricist reviewer cannot accept these claims as "empirical validation of scale preservation."

### 3. Residual Block Boundary Approximation
The authors acknowledge that global scale-invariance does not strictly hold across residual connections of the form $\mathbf{y} = \mathbf{x} + \alpha \mathcal{F}(\text{LN}(\mathbf{x}))$. They justify omitting BWN because the computed scaling factor is extremely close to $1.0$ (mean $\alpha = 0.998$). While this is a reasonable empirical justification for CLIP-ViT, it means that for larger models with larger fine-tuning steps, BWN's scaling factor could deviate from 1.0, and the residual path ratio would be altered. This boundary condition is discussed theoretically but not analyzed empirically.

## Reproducibility Assessment
The paper has **significant reproducibility gaps**:
*   **No Fine-Tuning Details:** There is absolutely no description of how the CLIP experts were fine-tuned. Crucial hyperparameters such as the learning rate, epochs, optimizer (e.g., AdamW, SGD), batch size, weight decay, and learning rate scheduler are entirely missing.
*   **No MLP Training Details:** Similarly, the training parameters of the 3-layer MLP are omitted.
*   **No Codebase:** Although the paper provides a GitHub URL (`https://github.com/anonymous/svs`), this is a placeholder and contains no actual code.
An expert reader would not be able to reproduce these results from the information provided in the paper.
