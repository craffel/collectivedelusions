# Quantum Wavefunction Superposition Merging (QWS-Merge)

## 1. Persona Alignment
This proposal is a direct manifestation of **The Visionary** persona. Instead of making incremental tweaks to static linear coefficient search spaces, QWS-Merge completely rethinks the foundational assumption of weight-space model merging—that merged model parameters must remain static and input-independent. 

By drawing inspiration from quantum mechanics, we treat task experts as defining a Hilbert space of parameter state wavefunctions. Merging is represented as a coherent quantum-like superposition of these states, which dynamically "collapses" (measures) to a specific localized weight configuration based on the phase-overlap of the incoming input features. This radical, out-of-the-box paradigm shifts model merging from a static parameter-blending problem to a dynamic, input-guided wavefunction resonance process. It prioritizes extreme high-concept novelty and massive multi-task performance potential over safe, incremental linear compromises.

---

## 2. Core Techniques
*   **Quantum Wavefunction Superposition (QWS):** We model the fine-tuned task-specific experts not as static points in Euclidean weight-space, but as task eigenstates $\{|\psi_k\rangle\}$ in a parameter Hilbert space.
*   **Dynamic Wavefunction Collapse (DWC):** We define dynamic, input-dependent probability amplitudes (merging coefficients) determined by the cosine phase overlap between a low-dimensional input state and task-specific phase basis vectors.
*   **Lightweight Phase Projector (LPP):** A tiny, frozen linear projection layer that maps the global representation of the input batch (extracted from the backbone's initial patch embedding) into a low-dimensional phase-state space.
*   **Parameter-Efficient Amplitude & Phase Tuning:** We optimize layer-wise scaling amplitudes $R_k^{(l)}$, phase vectors $\Phi_k^{(l)}$, and phase biases $\phi_k^{(l)}$ using a tiny offline few-shot validation set, bypassing the Overfitting-Optimizer Paradox of online test-time adaptation (TTA) entirely.

---

## 3. Mathematical Formulation

Let $W_{base}^{(l)}$ represent the pre-trained base network weights at layer $l \in \{1, \dots, L\}$, and $V_k^{(l)} = W_k^{(l)} - W_{base}^{(l)}$ represent the task vector of expert $k \in \{1, \dots, K\}$ at layer $l$. 

### Input Phase State
Let $x \in \mathbb{R}^{B \times C \times H \times W}$ be the input batch. We extract a global representation $z(x) \in \mathbb{R}^D$ from the first layer's activations (e.g., global average pooling across the spatial/patch tokens of the ViT's patch embedding). We project $z(x)$ into a $d$-dimensional phase-state space via a frozen random projection $P \in \mathbb{R}^{d \times D}$ and normalize it to the unit sphere:
$$
\tilde{\psi}(x) = P z(x) \in \mathbb{R}^{B \times d}
$$
$$
\psi(x)_b = \frac{\tilde{\psi}(x)_b}{\|\tilde{\psi}(x)_b\|_2} \quad \forall b \in \{1, \dots, B\}
$$

### Quantum Phase-Coherent Overlap
For each layer $l$ and task $k$, we define a trainable phase basis vector $\Phi_k^{(l)} \in \mathbb{R}^d$ and normalize it to the unit sphere:
$$
\hat{\Phi}_k^{(l)} = \frac{\Phi_k^{(l)}}{\|\Phi_k^{(l)}\|_2}
$$
The dynamic probability amplitude (merging coefficient) $\alpha_k(x, l)$ for task $k$ at layer $l$ is determined by wave-like phase interference:
$$
\alpha_{k, b}(l) = R_k^{(l)} \cos\left( \omega \langle \psi(x)_b, \hat{\Phi}_k^{(l)} \rangle + \phi_k^{(l)} \right) \quad \forall b \in \{1, \dots, B\}
$$
where:
*   $R_k^{(l)} \in \mathbb{R}$ is the learned scaling amplitude.
*   $\omega \in \mathbb{R}$ is a fixed wave frequency scaling factor (default: $\pi$).
*   $\phi_k^{(l)} \in \mathbb{R}$ is a learned task-specific phase offset bias.

### Weight Measurement/Collapse
To ensure classical hardware compatibility and eliminate batch-looping computational overhead, we perform a mean-measurement over the input batch to collapse the wavefunction:
$$
\bar{\alpha}_k(l) = \frac{1}{B} \sum_{b=1}^B \alpha_{k, b}(l)
$$
The final collapsed merged weight matrix for layer $l$ used to process the batch $x$ is:
$$
W_{merged}^{(l)}(x) = W_{base}^{(l)} + \sum_{k=1}^K \bar{\alpha}_k(l) V_k^{(l)}
$$

---

## 4. Architecture Specifications
*   **Backbone:** Vision Transformer `vit_tiny_patch16_224` (5.7M parameters, $L=14$ layer groups: Patch Embedding, 12 Transformer blocks, and final Layer Normalization).
*   **Phase Projector ($P$):** A frozen linear layer mapping the 192-dimensional ViT patch-embedding output (mean-pooled across patch tokens, $D=192$) to a low-dimensional phase space of $d=4$ dimensions.
*   **Parameter Dimensionality:**
    *   Amplitudes $R_k^{(l)} \in \mathbb{R}$ (initialized to 0.3, total of $K \times L = 56$ parameters).
    *   Phases $\Phi_k^{(l)} \in \mathbb{R}^d$ (initialized randomly from $\mathcal{N}(0, 1)$, total of $K \times L \times d = 224$ parameters).
    *   Biases $\phi_k^{(l)} \in \mathbb{R}$ (initialized to 0.0, total of $K \times L = 56$ parameters).
    *   Total trainable parameters: $336$ parameters, representing an incredibly parameter-efficient subspace.
*   **Dynamic Assembly:** The merged weight $W_{merged}^{(l)}(x)$ is assembled dynamically in the forward pass of the backbone prior to executing the corresponding block. Since weight reconstruction only requires scalar-tensor operations, it adds negligible computational latency.

---

## 5. Baselines
*   **Uniform Merge (Task Arithmetic):** Static, uniform merging coefficients ($\lambda_k = 0.3$) across all layers.
*   **AdaMerging (Dense):** Unconstrained online TTA using unsupervised entropy minimization to optimize 56 layer-wise coefficients.
*   **PolyMerge / SplineMerge:** Restricting merging coefficients to continuous polynomial subspaces.
*   **OFS-Tune (Offline Few-Shot Validation Tuning):** Finding optimal, static coefficients $\alpha_k(l)$ using a tiny labeled validation set.
*   **Prune-then-Merge (P-then-M):** Separately pruning expert task vectors prior to uniform merging.

---

## 6. Step-by-Step Interaction

1.  **Input Feed:** An input batch of images $x \in \mathbb{R}^{B \times 3 \times 224 \times 224}$ is fed into the Vision Transformer backbone.
2.  **State Projection:** The images are processed by the patch embedding layer, yielding patch tokens $H_0 \in \mathbb{R}^{B \times 196 \times 192}$. We compute the spatial average $z(x) = \text{mean}(H_0, \text{dim}=1) \in \mathbb{R}^{B \times 192}$.
3.  **Wavefunction Map:** $z(x)$ is projected and L2-normalized to produce the input batch wavefunction state $\psi(x) \in \mathbb{R}^{B \times 4}$.
4.  **Layer-wise Dynamic Assembly:** For each layer group $l \in \{1, \dots, 14\}$:
    a.  Compute the inner products $\langle \psi(x)_b, \hat{\Phi}_k^{(l)} \rangle$ for each task $k$ and batch sample $b$.
    b.  Compute the wave phase modulation to find individual sample-level coefficients $\alpha_{k, b}(l)$.
    c.  Average across the batch to measure/collapse the state into a batch-level coefficient $\bar{\alpha}_k(l)$.
    d.  Assemble the layer weight tensor $W_{merged}^{(l)} = W_{base}^{(l)} + \sum_k \bar{\alpha}_k(l) V_k^{(l)}$.
5.  **Block Execution:** Process the token representation through the $l$-th Transformer block using the dynamically assembled weights $W_{merged}^{(l)}$.
6.  **Final Prediction:** Output the predictions from the task-specific classification heads.
