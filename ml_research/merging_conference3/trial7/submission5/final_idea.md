# Parameter-Free Activation Blending (PFAB) for Efficient Multi-Task Model Merging

## 1. Persona Alignment
This technical implementation represents a direct, uncompromising application of **The Minimalist** research philosophy:
- **Occam's Razor over Infrastructure Bloat:** Prior work in Trial 6 (PFSR + MBH) resolved heterogeneity collapse under mixed-task streams by shifting the burden of robustness from the model parameters to a complex serving infrastructure layer (Micro-Batch Homogenization, sequential batch partitioning, index sorting, and output re-sorting). PFAB applies Occam's razor to aggressively prune this entire data-orchestration infrastructure. It handles mixed-task streams **in a single forward pass of the shared model backbone** with zero batch partitioning or sorting.
- **Complexity Pruning:** Rather than dynamically compiling $G \le K$ separate merged parameter states on the fly and executing multiple sequential forward passes, PFAB performs sample-wise activation-space blending of lightweight expert adapter outputs at each layer. This reduces the inference FLOPs from $G$ full backbone passes to exactly 1 backbone pass, achieving a state-of-the-art multi-task solution by stripping away unnecessary systems components.
- **Zero Trainable Parameters:** Consistent with our taste for simple, clean architectures, PFAB introduces **zero trainable parameters** and requires **zero calibration split data** during test-time adaptation, preserving the mathematical elegance of non-parametric task identification.

---

## 2. Core Techniques
PFAB introduces and combines three core techniques:
1. **Parameter-Free Subspace Cosine Projection (PFSR):** A non-parametric, zero-shot coordinate routing mechanism that projects penultimate representations onto pre-trained expert classification weights to derive routing coefficients on a per-sample basis.
2. **Unit-Norm Calibration (UNC):** A training-free, spatial alignment layer that normalizes intermediate activations and classification heads, resolving cross-expert representation scale drift.
3. **Activation-Space Adapter Blending (ASAB):** A sample-wise feature modulation layer that blends lightweight LoRA expert outputs directly in activation space using sample-wise coefficients, completely bypassing parameter-space weight merging and eliminating Micro-Batch Homogenization (MBH).

---

## 3. Mathematical Formulation

### A. Non-Parametric Gating Coordinates
Let $z_b \in \mathbb{R}^D$ be the penultimate representation of sample $b \in \{1, \dots, B\}$ in a batch $X$, extracted from the final layer of the base model backbone. We first apply Unit-Norm Calibration (UNC) to normalize both $z_b$ and the expert classification weight vectors $W_{k,c} \in \mathbb{R}^D$ (where $c \in \{1, \dots, C_k\}$ is the class prototype index for Expert $k$):
$$\tilde{z}_b = \frac{z_b}{\|z_b\|_2}, \quad \tilde{W}_{k,c} = \frac{W_{k,c}}{\|W_{k,c}\|_2}$$

We compute the raw task coordinate $u_{k, b}$ using maximum cosine similarity:
$$u_{k, b} = \max_{c \in \{1, \dots, C_k\}} \tilde{W}_{k, c} \cdot \tilde{z}_b$$

We correct statistical extreme-value bias across asymmetrical vocabulary spaces $C_k$ and feature dimensions $D$ using Class-Size Scaling Calibration:
$$u'_{k, b} = \frac{u_{k, b}}{\sqrt{2\log C_k / D}}$$

The sample-specific blending coefficient $\alpha_{k, b}$ for Expert $k$ is derived via temperature-scaled Softmax:
$$\alpha_{k, b} = \frac{\exp(u'_{k, b} / \tau)}{\sum_{j=1}^K \exp(u'_{j, b} / \tau)}$$
where $\tau > 0$ is a static scaling temperature (set to $\tau = 0.001$ to perform near-discrete task identification).

### B. Activation-Space Sample-Wise Blending
For each attention or MLP projection layer $l$ equipped with fine-tuned Parameter-Efficient Fine-Tuning (PEFT/LoRA) adapters, let the input activations to the layer be $H^{(l-1)} \in \mathbb{R}^{B \times D}$.
The output of the shared base model layer is:
$$X_{base}^{(l)} = H^{(l-1)} W_{base}^{(l)} \in \mathbb{R}^{B \times D}$$

For each specialized expert $k \in \{1, \dots, K\}$, its low-rank adapter output is computed as:
$$X_k^{(l)} = \left( H^{(l-1)} B_k^{(l)} \right) A_k^{(l)} \in \mathbb{R}^{B \times D}$$
where $B_k^{(l)} \in \mathbb{R}^{D \times r}$ and $A_k^{(l)} \in \mathbb{R}^{r \times D}$ represent the LoRA adapter matrices of Expert $k$ with rank $r \ll D$.

We perform sample-wise blending directly in activation space by weighting each expert's adapter contribution by its corresponding sample-specific coefficient $\alpha_{k, b}$:
$$H^{(l)}_b = X_{base, b}^{(l)} + \sum_{k=1}^K \alpha_{k, b} X_{k, b}^{(l)}$$
for $b \in \{1, \dots, B\}$.

In vectorized tensor form, this is formulated as:
$$H^{(l)} = X_{base}^{(l)} + \sum_{k=1}^K \mathbf{diag}(\alpha_k) X_k^{(l)}$$
where $\alpha_k = [\alpha_{k, 1}, \dots, \alpha_{k, B}]^T \in \mathbb{R}^B$ is the batch-wide coefficient vector for Expert $k$.

---

## 4. Architecture Specifications
- **Backbone Model:** Vision Transformer (ViT) or standard multi-layer perceptron/attention block structure. In the synthetic sandbox, we operate on a $L=14$ layer representation space with intermediate dimension $D=192$.
- **Adapters:** Each expert $k \in \{1, \dots, 4\}$ consists of lightweight LoRA-like projection adapters injected at every layer, parameterized by low-rank matrices $B_k^{(l)} \in \mathbb{R}^{192 \times r}$ and $A_k^{(l)} \in \mathbb{R}^{r \times 192}$ with rank $r=8$.
- **Classification Heads:** Pre-trained frozen classification heads $W_k \in \mathbb{R}^{C \times 192}$, where $C=10$ classes for standard image classification tasks.
- **Inference Pipeline:** 
  - **Input:** Heterogeneous batch $X \in \mathbb{R}^{B \times D_{in}}$.
  - **Routing Step:** Extract penultimate representations $Z \in \mathbb{R}^{B \times D}$, compute sample-wise coefficient matrix $\alpha \in \mathbb{R}^{B \times K}$ once per batch.
  - **Forward Pass:** Execute base model layers, adding the sample-wise blended adapter outputs at each layer in a single vectorized forward pass.
  - **Output:** Joint prediction logits $Y \in \mathbb{R}^{B \times C}$.

---

## 5. Baselines
To demonstrate the ultimate simplicity and high-performance of PFAB, we compare against several prominent static and dynamic model merging baselines:
1. **Uniform Merging:** A static baseline that merges experts by naively averaging task-specific weights with uniform coefficients ($\alpha_k = 1/K$).
2. **Linear Router (Unregularized / Regularized):** A parametric dynamic router that learns a linear projection layer from penultimate features to task coefficients on a 64-sample calibration split.
3. **QWS-Merge (SOTA predecessor):** Quantum Wavefunction Superposition Merging, which models task experts as quantum eigenstates and uses complex wave phase-overlap equations to compute routing coefficients.
4. **L3-Linear / Softmax (Reg):** Parametric multi-layer routers that learn independent layer-wise coefficients, suffering from severe overfitting and layer-averaging collapse.
5. **PFSR + MBH (Trial 6 Winner):** Non-parametric subspace routing combined with Micro-Batch Homogenization. While highly performant, it serves as our primary systems baseline to demonstrate the massive latency and complexity savings of moving from parameter-space batch-partitioning to activation-space sample-blending.

---

## 6. Step-by-Step Interaction
The flow of data through PFAB is highly streamlined, executing in a single, parallelized pass:

1. **Feature Extraction & Routing Projection (Offline/Pre-activation):**
   - The input batch $X \in \mathbb{R}^{B \times D_{in}}$ is passed through the frozen base model backbone.
   - At the penultimate layer, the high-dimensional representation matrix $Z \in \mathbb{R}^{B \times D}$ is extracted.
   - We apply **Unit-Norm Calibration (UNC)** to obtain normalized features $\tilde{Z}$ and normalized classification heads $\tilde{W}_k$.
   - We compute the raw cosine similarity matrix $U \in \mathbb{R}^{B \times K}$, where $U_{b, k} = \max_c ( \tilde{W}_{k, c} \cdot \tilde{z}_b )$.
   - We scale $U$ by Class-Size Scaling Calibration to correct statistical asymmetrical vocabulary bias, and pass it through temperature-scaled Softmax to derive the sample-wise blending coefficients $\alpha \in \mathbb{R}^{B \times K}$.

2. **vectorized Forward Propagation (Layer-wise Activation Blending):**
   - For each layer $l \in \{1, \dots, L\}$:
     - The base layer weights $W_{base}^{(l)}$ execute a standard forward pass on input activations $H^{(l-1)}$, producing $X_{base}^{(l)} = H^{(l-1)} W_{base}^{(l)}$.
     - Concurrently, the input activations are passed through the $K$ lightweight parallel expert adapters to produce the task-specific feature deltas $X_k^{(l)} = H^{(l-1)} B_k^{(l)} A_k^{(l)}$.
     - We compute the fused activations $H^{(l)}$ by performing a vectorized, sample-wise weighted addition:
       $$H^{(l)} = X_{base}^{(l)} + \sum_{k=1}^K \mathbf{diag}(\alpha_k) X_k^{(l)}$$
       which scales extremely fast using simple tensor broadcast multiplications without any serial batch-splitting or custom CUDA kernels.
     - $H^{(l)}$ is passed as the input to the next layer $l+1$.

3. **Classification & Prediction:**
   - The final layer activations $H^{(L)}$ are passed through the respective classification heads to generate highly precise, multi-task predictions $Y$ in a single constant-time forward pass.
