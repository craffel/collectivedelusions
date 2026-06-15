# Technical Summary: FoldMerge (Neural Origami)

## Main Topic
This paper proposes **FoldMerge (Neural Origami)**, an exploratory, non-linear coordinate-transformation framework that re-conceptualizes model merging as a continuous weight-space warping process. It aims to challenge and move beyond the traditional "flat-space" paradigm of model merging, which relies on simple linear combinations or rigid projections in Euclidean weight space.

## Approach and Methodology
FoldMerge addresses the problem of merging task-specific expert neural networks fine-tuned from a shared pre-trained base model. It is formulated as follows:
1. **Differentiable Coordinate Warp ($g_\phi$):** Maps parameter vectors from raw Euclidean weight space to a latent, unified "Origami Space" ($z$-space) using a diffeomorphism (smooth, bijective, invertible mapping).
2. **Normalizing Flow Parameterization:** Implements $g_\phi$ using a cascade of $M = 4$ RealNVP affine coupling layers with standard feed-forward scaling ($s_\phi$) and translation ($t_\phi$) MLPs (2 layers, 512 hidden size, GELU activations).
3. **Scale Bounding:** Binds the scale network outputs using a hyperbolic tangent activation ($\bar{s}_\phi = \tanh(s_\phi)$) to prevent exponential scaling and ensure numerical stability.
4. **Origami Space Merging Formulations:**
   - **Absolute Additive (Default):** $\bar{z} = 1.0 \cdot z_{base} + \sum_k \lambda_k z_k$, decoded via $g_\phi^{-1}$.
   - **Barycentric Latent Merging:** $\bar{z} = (1.0 - \sum_k \lambda_k) z_{base} + \sum_k \lambda_k z_k$ (energy-scale preserving).
   - **Latent Task Vector Warping:** $\theta_{MTL} = \theta_{base} + g_\phi^{-1}\left( \sum_k \lambda_k g_\phi(\tau_k) \right)$ (bypasses base-model scale distortion).
5. **LoRA-Flow:** A parameter-efficient formulation that decomposes the MLP weights inside the coupling layers using low-rank matrices ($r = 8$). This compresses the trainable parameters by $27\times$ (from $2.6\text{M}$ to $96\text{K}$) and serves as an implicit regularizer.
6. **Unsupervised Test-Time Optimization:** Employs an unsupervised self-labeling protocol (guided by expert teacher predictions) on unlabeled test streams. It optimizes flow parameters $\phi$ and coefficients $\lambda$ by minimizing prediction Kullback-Leibler (KL) divergence over 500 steps using Adam.
7. **Implicit Flow Regularization:** Introduces an $\ell_2$ parameter weight-decay penalty ($\gamma \sum \|p\|_2^2$) to pull scale/translation parameters towards zero, encouraging the flow to stay close to the identity mapping. This avoids computing expensive $O(d^2)$ Jacobians.

## Key Findings and Claims
- **State-of-the-Art Accuracy:** FoldMerge achieves an average accuracy of **89.76%** on an 8-task classification benchmark using CLIP ViT-B/32, performing on par with state-of-the-art SyMerge (89.74%) and outperforming it on 5 out of 8 individual tasks.
- **Superior Formulations:** The mathematically rigorous **Latent Task Vector Warping** achieves **89.77%**, while the **Barycentric Latent Merging** achieves **89.74%**.
- **LoRA-Flow Performance:** The compressed **LoRA-Flow** ($r=8$) reduces the trainable parameter count by $27\times$ and achieves a superior accuracy of **89.82%** (outperforming full-rank flow's 89.77%), proving that low-rank constraints act as robust structural regularizers.
- **Deterministic and Robust Optimization:** Under Test-Time Adaptation (TTA), FoldMerge displays $100\%$ deterministic reproducibility (zero run-to-run variance) and is robust to the temporal order of task streams ($\pm 0.03\%$ variance under shuffling).
- **Inherent Alignment Capacity:** Under a frozen classifier-head setting (\texttt{args.classifier\_train = False}), FoldMerge achieves an Average Accuracy of **83.56%**, performing on par with SyMerge (83.56%) and outperforming it on 3 out of 8 tasks, confirming genuine, non-linear representation alignment.

## Explicitly Claimed Contributions (With Evidence)
1. **A New Geometric Perspective for Model Merging:** Proposes formulating model merging as a non-linear coordinate warping process. This is backed by detailed geometric explanations, visualizations, and mathematical formulations of Origami Space.
2. **Differentiable Coordinate Warping via Normalizing Flows:** Builds a working RealNVP-based pipeline with bounded scale and implicit weight-decay regularization. Supported by ablation studies exploring coupling depth ($M$) and regularization coefficient ($\gamma$).
3. **Empirical Proof-of-Concept on Multi-Task Benchmark:** Achieves highly competitive classification accuracies across 8 diverse datasets on a ViT-B/32 backbone. Evidence is provided in detailed comparative tables (Tables 1, 2, 4, 5).
4. **Candid Discussion of Architectural and Practical Limitations:** Transparently details limitations such as coordinate-dependence (lack of permutation equivariance), row-wise slicing category errors, and parameter/computational overhead, suggesting theoretical pathways forward (Glow/Neural Splines, pre-alignment matching, low-rank compression).
