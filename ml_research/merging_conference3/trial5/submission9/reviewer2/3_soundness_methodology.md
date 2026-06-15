# Soundness and Methodology Evaluation of GSC-Merge

## Clarity of Description
The description of the GSC-Merge framework is exceptionally clear, structured, and mathematically rigorous.
- **Formulation:** The authors transition smoothly from defining standard task vectors to constructing the joint multi-task update matrix $\mathbf{M}^{(l)}$, performing SVD, and defining the left-singular projection operator $P^{(l)}$.
- **Target Layers:** The paper clearly specifies the target layers (theQuery-Key-Value projection `blocks.i.attn.qkv.weight`, attention projection `proj`, and MLP layers `fc1` and `fc2`), leaving no ambiguity about which parameters are merged.
- **Notation:** The mathematical notation is highly consistent and precise. Step-by-step derivations are provided for the projection operations and the final multi-task blended weight formulation.

## Appropriateness of Methods
- **SVD for Subspace Consensus:** Using Singular Value Decomposition on the joint multi-task update matrix is a highly appropriate and mathematically principled way to identify the shared directions of parameter variation across different downstream tasks.
- **Left-Singular Projection:** The choice to construct the projection operator using the left-singular vectors (output space $d_{out}$) rather than right-singular vectors (input space $d_{in}$) is thoroughly and elegantly justified. Since linear layers are structured as $y = W x$, left-singular vectors represent the principal output representation coordinates. Projecting onto this space directly aligns activation coordinates, which is where representation mismatch occurs. Furthermore, since $d_{out} \ll K \cdot d_{in}$ for modern architectures, this choice is computationally and memory-wise much more efficient.
- **Offline Few-Shot Validation Tuning (OFS-Tune):** Using a tiny validation set (16 samples per task, 64 total) to optimize layer-wise merging coefficients using the Adam optimizer is a standard and robust approach to avoid static coefficient bias.

## Technical Correctness and Proofs
We reviewed the mathematical proofs presented in the methodology section:
1. **Eckart-Young-Mirsky Theorem (Theorem 1):** The proof that the Grassmannian projection $P^{(l)} \mathbf{M}^{(l)}$ yields the unique global minimizer of the reconstruction error under the Frobenius norm among all rank-$r$ matrices is technically flawless and standard in linear algebra.
2. **Proposition 1 (Spectral and Frobenius Norm Non-Strict Contraction):**
   - The proof that GSC-Merge is a non-strict contraction ($\|\Delta W_{gsc}^{(l)}\|_2 \le \|\Delta W_{uncon}^{(l)}\|_2$ and $\|\Delta W_{gsc}^{(l)}\|_F \le \|\Delta W_{uncon}^{(l)}\|_F$) is technically sound. Since $P^{(l)}$ is an orthogonal projector, its spectral norm is exactly 1, and it is a contraction on Euclidean vector spaces of matrices.
   - The conceptual derivation showing that restricting the active optimizer steps to the $r$-dimensional Grassmannian manifold (where $r \ll d_{out}$) filters out high-frequency validation noise is highly convincing and provides a strong theoretical basis for resolving the Overfitting-Optimizer Paradox.
3. **Scalability and LoRA Adapter Discussions (Sections 3.7 & 3.9):** The authors show remarkable foresight by discussing standard numerical methods (randomized SVD, power iteration) to scale GSC-Merge, and how to apply GSC-Merge to LoRA adapters without reconstructing full-rank weight matrices. These extensions are technically sound and highly practical.

## Reproducibility
The level of detail provided in the methodology guarantees an **excellent** standard of reproducibility:
- **Architecture Details:** The paper specifies the exact model backbone (`vit_tiny_patch16_224` from `timm`) and targets exactly 48 major linear layers across 12 blocks.
- **Data Configuration:** The use of 16 labeled validation samples per task is explicitly stated, alongside the specific 4 datasets evaluated (MNIST, FashionMNIST, CIFAR-10, SVHN).
- **Optimization Hyperparameters:** The initialization of $\alpha_k^{(l)} = 0.25$, learning rate ($10^{-2}$), weight decay ($10^{-4}$), optimizer (Adam), and number of steps (100) are all clearly provided.
- **Statistical Rigor:** The use of 5 independent random validation calibration splits (with mean and standard deviation reported) further solidifies the reliability and reproducibility of the results.

## Potential Minor Technical Flaws / Limitations
- **Resetting non-target parameters in Truly Task-Agnostic Setting:** In the truly task-agnostic setting, all non-target parameters (linear biases, layer norms, patch projections) are reset to their pre-trained base values. Since the experts were fine-tuned on all parameters (presumably including layer norms and biases), resetting them to pre-trained base values causes a massive performance drop (e.g., from $43.88\%$ to $20.61\%$). A more comprehensive baseline would be to fine-tune the experts *while keeping non-target parameters frozen* at base values from the beginning, which would prevent this statistic mismatch. However, the authors' evaluation of resetting them post-hoc is still an honest and instructive experiment that highlights the critical role of normalization layers in vision models.
