# Soundness and Methodology Evaluation

## Methodological Strengths
1. **Mathematical Clarity:** The methodology is written with commendable precision. The offline SVD factorization of task vectors, the bounded cosine-similarity router, the Top-1 sparse gating, and the parallel forward pass are all clearly defined and easy to mathematically trace.
2. **Elegant Zero-Shot Initialization:** Bypassing the non-differentiable nature of the $\arg\max$ operation in Top-1 gating through **Activation-Space Mean Initialization** is an elegant, training-free, and highly practical solution. Using representative centroids in activation space is computationally lean and requires only a small calibration set.
3. **Autonomous Head Selection:** Introducing a sample-wise, layer-averaged classification head selection rule ($\hat{k}_b = \arg\max_k \bar{s}_{k,b}$) removes the reliance on an "oracle" during inference. This resolves a common critique of dynamic model-merging methods (which often assume a privileged oracle at inference time to select classification heads) and makes the system truly autonomous.

## Potential Technical Flaws and Points of Discussion

### 1. Vectorized Parallel Forward Pass vs. True Computational Savings
The authors write:
> "Rather than using a soft, dense combination of experts (which would require executing $K$ low-rank forward passes per sample), we enforce extreme computational efficiency by performing Top-1 sparse routing (hard gating)."

However, they subsequently present the vectorized parallel forward pass equation as:
$$Y = X W_{\text{base}}^{(l)} + \sum_{k=1}^K \alpha_k \odot \left( (X A_k^{(l)}) B_k^{(l)} \right)$$
In standard deep learning libraries like PyTorch, if this equation is implemented as-written (i.e., broadcasting and summing over $k \in \{1, \dots, K\}$), the system will still evaluate the matrix multiplications $(X A_k) B_k$ for **all $K$ experts** before multiplying by the one-hot $\alpha_k$ tensor. 
While this is mathematically equivalent to Top-1 gating and ensures batch independence, it does **not** deliver the $O(1)$ computational scaling of sparse gating during execution unless specialized conditional indexing or gather/scatter kernels are used (e.g., grouping samples in the batch by their selected expert and executing only the corresponding adapter).
- **Critique:** The paper should clarify whether their PyTorch implementation actually achieves $O(1)$ expert execution per sample via indexing/masking, or if it naively evaluates all $K$ paths in parallel. If the latter, the claimed computational savings (+8.3% FLOPs) are still true for a single sample ($B=1$), but for larger batches, the parallel compute cost would scale with $O(K)$ in practice.

### 2. Representation Shift during Calibration
To compute the task basis vectors $\Phi_k^{(l)}$, the paper runs a calibration forward pass of the model. However, during calibration, the model is configured with *uniform merging weights*. At evaluation, the model runs with *sparse low-rank weights*. 
- **Block 9 (First specialized layer):** Since early layers (blocks 0--8) are frozen and identical across all experts, the inputs and activations at the entrance of block 9 are perfectly identical during calibration and evaluation. Thus, $\Phi_k^{(9)}$ is perfectly aligned with the evaluation representation.
- **Blocks 10 & 11:** For subsequent blocks, the activations $X^{(10)}$ and $X^{(11)}$ will depend on the low-rank adapter selected and applied at block 9 (and block 10). Because calibration was done under uniform weights, there is a mathematical representation shift between the calibration activations and the actual evaluation activations at blocks 10 and 11.
- **Critique:** While the authors note in Section 4.4 and Appendix B that this representation shift has virtually zero negative impact due to high domain separability, a more rigorous treatment should discuss how this shift scales as the number of specialized blocks increases. For deeper specialized subnetworks (e.g., full 12-block networks), this representation shift could compound, potentially degrading downstream routing quality.

### 3. Discarding Activation Magnitudes in Cosine Router
The bounded cosine-similarity router normalizes both the activation $z(x)_b$ and the basis $\Phi_k^{(l)}$, projecting them onto a unit sphere. While this provides excellent regularization under domain shifts, it completely discards representation magnitude/scale. For highly distinct domains, this works perfectly. However, if two tasks have overlapping activation directions but differ significantly in their activation magnitudes (a common pattern in hierarchical or fine-grained tasks), the cosine router may fail to differentiate them. This limitation should be explicitly discussed.

## Reproducibility Assessment
The reproducibility of the methodology is **excellent**. The mathematical formulation is complete, including details like splitting singular values equally ($\sqrt{\Sigma_k}$) and using a stabilization constant $\epsilon = 10^{-8}$. The use of standard ViT backbones and public datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) further ensures that a practitioner can easily reimplement and reproduce the findings.
