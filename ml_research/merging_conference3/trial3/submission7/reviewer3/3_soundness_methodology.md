# 3. Soundness and Methodology

## Clarity of Description
The methodology in the paper is exceptionally well-described and structured. The mathematical formulation of the task vectors, the nested structural granularities, the unsupervised surrogate objective, the optimization dynamics (Adam vs. 1+1 ES), and the spatial-depth regularizers (Elastic Spatial Regularization and Total Variation) are clearly laid out with explicit equations. The appendix provides highly detailed hyperparameter settings, architecture details (ViT-Tiny), and pre-training/fine-tuning protocols, making the paper highly transparent and readable.

## Appropriateness of Methods
The methods chosen are highly appropriate for the empirical goals of the paper:
* **Unification**: The implementation of the 5-level nested hierarchy allows for a direct, apples-to-apples comparison of parameter resolution.
* **Optimization families**: Comparing analytical first-order gradient descent (Adam) with derivative-free zero-order evolution (1+1 ES) is a smart way to deconstruct how different optimization trajectories handle transductive overfitting in high dimensions.
* **Regularizers**: ESR and TV smoothness penalties are well-suited to control spatial and depth-wise parameter fluctuations.

## Potential Technical Flaws and Pragmatic Limitations
As a practitioner focused on deployment and scaling, there are a few notable limitations in the methodology, though none of them constitute logical or mathematical flaws:
* **Highly Compact, Toy Model and Small Datasets**: The experiments are conducted on a custom, extremely small `ViTTiny` model ($d_{\text{model}}=64$, 2 heads, 12 layers) on MNIST, FashionMNIST, CIFAR-10, and SVHN, with only 500 samples per task. While the authors defend this as a "low-resource edge warm-start setting," this is far from standard real-world deployment. In industry, model merging is typically applied to large-scale, fully converged foundation models (e.g., CLIP-Large, LLaMA) on high-fidelity, massive datasets. 
* **Generalizability of Findings**: It is highly possible that the "Generalization-Granularity Trade-off" behave differently in high-fidelity foundation model regimes. Fully-converged, large-scale models possess highly stable, structured, and noise-free feature spaces. The "catastrophic collapse" observed at higher granularities (L5 Tensor-wise) in this paper may be amplified by the high-frequency parameter noise of the small, poorly converged experts. Thus, while the paper's findings are extremely robust for this low-fidelity regime, they may not directly translate to larger production systems.
* **Misalignment of Entropy Loss**: The authors' observation of prediction entropy misalignment is excellent, but it highlights a fundamental limitation of unsupervised test-time adaptation. The paper relies heavily on entropy minimization; if the surrogate loss itself is highly flawed, any optimization (especially fine-grained) is bound to exploit it, leading to "confident but incorrect" decision boundaries.

## Reproducibility
The reproducibility of this paper is **excellent**. The authors have provided:
* The exact architectural hyperparameters of `ViTTiny` (number of blocks, embedding dimension, heads, patch size, etc.).
* Exceeded task-wise details of pre-training (15 epochs, joint pool, Adam, learning rate $1\times 10^{-3}$, weight decay) and independent fine-tuning (25 epochs, Adam, $5\times 10^{-4}$).
* Complete test-time adaptation settings (batch size, optimizer steps, learning rates, sliding window for ES, and regularization constants).
* Complete tables showing standard deviations across 3 independent random seeds.
A practitioner could easily reproduce these experiments and build upon the findings.
