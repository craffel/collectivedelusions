# Intermediate Evaluation: Novelty Check

## Assessment of Key Novel Aspects
The primary novelty of GSC-Merge lies in treating multi-task model merging as finding an optimal shared low-rank consensus subspace on the Grassmannian manifold $\mathbf{Gr}(r, d_{out})$, rather than applying heuristic coordinate-wise operations (such as magnitude-based pruning or coordinate-wise sign voting). 

Specifically, the novel components include:
1. **Horizontal Concatenation for Consensus**: Constructing a unified joint update matrix $\mathbf{M}^{(l)}$ by horizontally concatenating task vectors across distinct tasks, and performing SVD to extract joint principal directions.
2. **Left-Singular Projection**: Utilizing the left-singular vectors of SVD to construct an orthogonal projection operator $P^{(l)}$ in the output activation space $\mathbb{R}^{d_{out}}$. This is justified as direct alignment of output activations and provides mathematical optimality under the Frobenius norm via the Eckart-Young-Mirsky Theorem.
3. **Low-Rank Spectral Regularization**: Using the Grassmannian subspace projection to constrain the parameter search space during few-shot validation-set coefficient tuning, thereby preventing the optimizer from fitting high-frequency task-specific noise (resolving the "Overfitting-Optimizer Paradox").

## Delta from Prior Work
The paper positions GSC-Merge relative to two main classes of prior work:
1. **Heuristic Merging Methods (TIES, STA)**: These methods "denoise" task vectors on a strictly coordinate-wise basis (e.g., pruning small-magnitude coordinates or determining a consensus sign). The delta is that GSC-Merge treats parameters as geometrically correlated, utilizing continuous spectral operators (SVD) and low-rank manifold projections instead of non-differentiable discrete heuristics.
2. **Low-Rank Parameter Adaptation (LoRA)**: LoRA constrains parameter updates during training to a low-rank factorization. The delta is that GSC-Merge is a post-hoc merging framework that identifies a shared low-rank consensus subspace from already fully fine-tuned, dense parameter weights.

## Characterization of Novelty
The novelty of this work is **incremental to moderate**:
* **Theoretically elegant but conceptually close to existing ideas**: Applying SVD to extract low-rank subspaces from neural network weights, activations, or updates is a highly standard technique in machine learning (e.g., in matrix factorization, adapter compression, or LoRA merging). For example, recent concurrent works (such as MADE-IT or GAM) already map expert principal subspaces to Grassmann manifolds using SVD to align adapters. 
* **Straightforward Integration**: Combining SVD-based projection with standard layer-wise coefficient tuning (OFS-Tune) is a clean and logical progression. While the mathematical framing using the Eckart-Young-Mirsky Theorem and Grassmannian geometry is exceptionally rigorous, the core technical delta is relatively straightforward: it replaces coordinate-wise thresholding with SVD-based projection before optimizing coefficients.
* **Reliance on Task-Conditional Swapping**: To achieve competitive results, the framework relies heavily on task-conditional parameter swapping of lightweight parameters (norms, biases, embeddings). This represents a hybrid routing setup rather than a fully task-agnostic single weight configuration, which diminishes the practical novelty of the weight-merging aspect (as discussed in the Soundness and Methodology section).
