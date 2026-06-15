# Idea Proposal: Block-wise Weight-Sharing Routing Sweep (BWS-Router)

## 1. Persona Alignment
True to the traits of **The Empiricist**, this project is fundamentally driven by a philosophy of exhaustive empirical validation and systematic hyperparameter sweeps rather than speculative mathematical analogies. Rather than assuming that any particular layer-wise granularity is theoretically superior, we argue that the optimal level of layer specialization can only be uncovered through large-scale, rigorous experimental testing. 
By designing a highly modular codebase, we will run parallel Slurm sweeps over the complete space of layer-grouping sizes ($M \in \{1, 2, 3, 4, 6, 12\}$), activation functions (Linear, Tanh, Softmax, Sigmoid), learning rates, random seeds, and regularization scales. We will present overwhelming empirical evidence across distinct visual classification tasks to systematically map the capacity-robustness trade-off, verifying every component of our proposed method with exhaustive ablation tables.

## 2. Core Techniques
*   **Block-wise Parameter Sharing:** Instead of learning independent routing parameters for every single layer (which leads to layer-averaging collapse and optimization instability) or relying on a single global router, we group the $L$ layers of the model into $G = L / M$ uniform block groups and share routing weights within each block.
*   **Low-dimensional Input State Projection:** We compress the globally pooled feature representations of the backbone to a low-dimensional space ($d = K = 4$) using an unsupervised PCA projection matrix and normalize them onto the unit sphere, extending the formulation from the L3-Router (Trial 5, Submission 5).
*   **Bounded Independent Sigmoidal Gating:** To bypass the zero-sum competitive bottleneck of Softmax routing highlighted in the BSigmoid-Router (Trial 5, Submission 4), we apply independent, bounded Sigmoidal activations to compute task-specific merging coefficients in the range $[0, 0.3]$.

## 3. Mathematical Formulation
Let the layers of the model be indexed by $l \in \{1, \dots, L\}$. For a chosen block size $M$, we partition the layers into $G = L / M$ block groups:
$$
\mathcal{G}_g = \{(g-1)M + 1, \dots, gM\} \quad \text{for } g \in \{1, \dots, G\}
$$
Within each block group $g$, the routing parameters are shared. We define the trainable group weights $W_{group}^{(g)} \in \mathbb{R}^{K \times d}$ and biases $B_{group}^{(g)} \in \mathbb{R}^K$.

For any layer $l \in \mathcal{G}_g$, the dynamic merging coefficient for task $k$ and sample $b$ in a batch is computed using the shared block parameters:
$$
\alpha_{k, b}^{(l)} = \alpha_{k, b}^{(g)} = \lambda_{max} \times \text{Sigmoid}\left( \langle \psi(x)_b, W_{group, k}^{(g)} \rangle + B_{group, k}^{(g)} \right)
$$
where $\psi(x)_b \in \mathbb{R}^d$ is the unit-sphere projected low-dimensional input representation for sample $b$, and $\lambda_{max} = 0.3$ is the individual task scaling ceiling.

We average the sample-wise coefficients across the batch of size $B$ to compute the batch collapsed coefficient:
$$
\bar{\alpha}_k^{(l)} = \bar{\alpha}_k^{(g)} = \frac{1}{B} \sum_{b=1}^B \alpha_{k, b}^{(g)}
$$
The dynamically merged weights at layer $l$ are then assembled as:
$$
W_{merged}^{(l)}(x) = W_{base}^{(l)} + \sum_{k=1}^K \bar{\alpha}_k^{(l)} V_k^{(l)}
$$
where $V_k^{(l)} = W_k^{(l)} - W_{base}^{(l)}$ is the task vector.

During the 64-sample calibration phase, we train the router parameters ($W_{group}$ and $B_{group}$) by minimizing the cross-entropy loss augmented with $L_2$ weight decay regularization:
$$
\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda_{wd} \sum_{g=1}^G \sum_{k=1}^K \left( \|W_{group, k}^{(g)}\|_2^2 + (B_{group, k}^{(g)})^2 \right)
$$

### Proof of Mitigating Layer-Averaging Collapse
In post-hoc ensembling of classifiers, the final classification head is merged using the average coefficient across layers:
$$
\bar{\alpha}_k = \frac{1}{L} \sum_{l=1}^L \alpha_{l, k}
$$
Under block-wise parameter sharing, substituting our shared coefficients into the averaging formula yields:
$$
\bar{\alpha}_k = \frac{1}{G \cdot M} \sum_{g=1}^G \sum_{l \in \mathcal{G}_g} \alpha_{l, k} = \frac{1}{G \cdot M} \sum_{g=1}^G M \alpha_{g, k} = \frac{1}{G} \sum_{g=1}^G \alpha_{g, k}
$$
For $M = L$, this reduces to a single global block ($G = 1$), which simplifies to:
$$
\bar{\alpha}_k = \alpha_{1, k}
$$
This completely eliminates the layer-averaging noise. For intermediate block sizes $M$, the averaging summation is reduced from $L$ independent terms to $G$ terms, which significantly lowers the degrees of freedom and guides the optimizer toward more stable, lower-variance representations while preserving coarse layer specialization (e.g., distinguishing early representation layers from late semantic layers).

## 4. Architecture Specifications
*   **Backbone:** Vision Transformer (`vit_tiny_patch16_224`) consisting of $L_{total}=14$ layer groups, with $L=12$ core Transformer blocks where we apply our block-wise grouping. The hidden feature dimension is $D = 192$.
*   **Compression layer:** Unsupervised PCA projection matrix $P \in \mathbb{R}^{D \times d}$ where the target dimension is $d = K = 4$.
*   **Routing Network:** Linear projection layer mapping the low-dimensional state $\psi(x)_b \in \mathbb{R}^d$ to $K$ task logits per block group, activated with an independent Sigmoid function scaled by $\lambda_{max} = 0.3$.
*   **Parameters footprint:** For $G$ groups, the parameter shape is $(G \times K \times d)$ for weights and $(G \times K)$ for biases.
    *   **$M=1$ (Fully Unshared Baseline):** $G=12 \implies 240$ trainable parameters.
    *   **$M=3$ (Our Proposed Block):** $G=4 \implies 80$ trainable parameters.
    *   **$M=4$ (Our Proposed Block):** $G=3 \implies 60$ trainable parameters.
    *   **$M=12$ (Fully Shared Global Block):** $G=1 \implies 20$ trainable parameters.

## 5. Baselines
*   **Static Uniform Merging:** Assigns a fixed, static task vector scale of $0.3$ to all tasks, providing a baseline ceiling for unadapted multi-task fusion.
*   **Un-regularized Global Linear Router:** High-dimensional direct linear mapping from $z(x)_b$ to routing coefficients, serving as the simplest global routing comparison.
*   **Quantum Wavefunction Superposition Merging (QWS-Merge):** The current SOTA wave-inspired method, helping us evaluate whether complex non-monotonic activations have any real advantage over regularized classical alternatives.
*   **Layer-wise Low-dimensional Classical Router (L3-Router):** Our primary unshared baseline (equivalent to block size $M=1$), allowing us to isolate and demonstrate the benefits of block-wise parameter sharing in preventing collapse.

## 6. Step-by-Step Interaction
1.  **Feature Extraction:** Extract the globally pooled $192$-dimensional feature representation $z(x)_b \in \mathbb{R}^D$ from the first block of the ViT backbone.
2.  **PCA Projection:** Multiply $z(x)_b$ by the unsupervised PCA projection matrix $P \in \mathbb{R}^{D \times d}$ to get low-dimensional representations.
3.  **Unit Sphere Normalization:** Normalize the low-dimensional representations to the unit sphere, adding $\epsilon = 10^{-8}$ for stability, yielding the input state $\psi(x)_b \in \mathbb{R}^d$.
4.  **Block Gating Logits:** For each sample $b$ and block group $g \in \{1, \dots, G\}$, project the input state to task logits:
    $$
    o(x)_{b, k}^{(g)} = \langle \psi(x)_b, W_{group, k}^{(g)} \rangle + B_{group, k}^{(g)}
    $$
5.  **Sigmoidal Activation:** Map the logits to independent sample-level coefficients:
    $$
    \alpha_{k, b}^{(g)} = \lambda_{max} \times \text{Sigmoid}\left( o(x)_{b, k}^{(g)} \right)
    $$
6.  **Batch Averaging:** Average sample-level coefficients across the batch dimension to obtain batch collapsed coefficients $\bar{\alpha}_k^{(g)}$.
7.  **Layer Assignment:** Assign the block coefficient to all constituent layers inside that block:
    $$
    \bar{\alpha}_k^{(l)} = \bar{\alpha}_k^{(g)} \quad \forall l \in \mathcal{G}_g
    $$
8.  **Weight Merging:** Assemble the dynamically merged weights $W_{merged}^{(l)}(x)$ and perform the forward pass.
