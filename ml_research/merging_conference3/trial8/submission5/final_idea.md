# PEAR: Patch-Embedding Activation Routing for Multi-Task Expert Merging

## 1. Persona Alignment
PEAR is the ultimate embodiment of **The Minimalist** persona. Guided by Occam's razor, PEAR argues that prior state-of-the-art dynamic ensembling methods (like SABLE or SPS-ZCA) have introduced unnecessary system-level or structural complexity to resolve the "Routing Paradox" (where executing a model twice is required to route). SABLE solves this by freezing and leaving early layers (e.g., Layers 1 to 10) completely unadapted, discarding early task-specific features. SPS-ZCA restricts LoRAs to mid-to-late layers, creating a training-test mismatch. Other methods use external auxiliary models, adding memory and parameter bloat.

In contrast, PEAR performs dynamic, sample-wise activation ensembling in a **strictly single-pass, parameter-free, closed-form** manner over **100% of the network depth (all layers adapted)**. It achieves this by routing inside the base model's first layer—the frozen Patch Embedding layer (Layer 0). By using the base model's own early projection as a zero-cost, high-fidelity task-space feature extractor, PEAR resolves:
1. **The Routing Paradox:** Routing coefficients are computed at Layer 0, requiring zero redundant backbone passes.
2. **The Representational Alignment Paradox:** Cosine similarities are computed against centroids pre-calculated in the same early Patch-Embedding space, ensuring perfect structural alignment.
3. **The Early-Feature Loss Trade-Off:** Since routing coefficients are available from Layer 0, PEAR can activate and blend LoRA adapters across **all 12 layers** of the network, preserving and utilizing early-stage specialized features with zero trainable parameters.

---

## 2. Core Techniques
PEAR consists of the following interconnected, non-parametric techniques:
- **Layer 0 (Patch-Embedding) Extraction:** The input images are projected through the shared, frozen Patch Embedding layer of the Vision Transformer, mapping pixels into the network's aligned representational space at virtually zero extra compute.
- **Zero-Shot Patch Centroids (ZPC):** Computes representative task centroids in the Patch-Embedding output space offline using only 64 samples per task, requiring zero trainable parameters and zero gradients.
- **Unit-Norm Calibration (UNC):** Formulates the routing similarity using cosine projection on a unit-hypersphere to ensure perfect scale-invariance.
- **Intra-Task Dispersion Calibration (IDC):** Standardizes similarities by expected in-distribution dispersion scales to resolve asymmetric task manifold variance.
- **Single-Pass Activation Blending (SPS):** Layer-wise, sample-specific LoRA adapter activation blending on-the-fly, spanning **all 12 layers** of the network, maintaining flat $O(1)$ latency and zero dynamic memory buffers.

---

## 3. Mathematical Formulation

### A. Early Representation Extraction
For an incoming batch of images $X = \{x_1, \dots, x_B\}$, each sample $x_b$ is projected via the pre-trained, frozen Patch Embedding layer (Layer 0) of the base model:
$$Z_b = \text{PatchEmbed}(x_b) \in \mathbb{R}^{N \times D}$$
where $N$ is the number of patches, and $D = 192$ is the feature dimension of the base ViT-Tiny.
We extract a single global early representation vector $z_b \in \mathbb{R}^D$ by taking the spatial mean over the patch dimension:
$$z_b = \frac{1}{N} \sum_{i=1}^N Z_{b, i, :} \in \mathbb{R}^D$$

### B. Offline Calibration of Early Centroids
During a one-time, offline, training-free calibration phase, using a tiny calibration set $\mathcal{C}_k$ of $|\mathcal{C}_k| = 64$ samples for each task $k \in \{1, \dots, K\}$, we compute the early task centroids in the Patch Embedding space:
$$\mu_k = \frac{1}{|\mathcal{C}_k|} \sum_{s \in \mathcal{C}_k} z_s \in \mathbb{R}^D$$

### C. Subspace Cosine Projection (with Unit-Norm Calibration)
At test time, the raw similarity $s_{k, b}$ is computed by taking the cosine similarity between $z_b$ and the pre-computed centroids $\mu_k$:
$$s_{k, b} = \text{cos\_sim}(z_b, \mu_k) = \frac{z_b \cdot \mu_k}{\|z_b\|_2 \|\mu_k\|_2}$$
By restricting vectors to a unit-norm sphere, we eliminate scale drift and representational magnitude biases.

### D. Intra-Task Dispersion Calibration (IDC)
To handle asymmetric task manifold variance, we compute the expected in-distribution similarity scale $d_k$ of the calibration samples to their respective centroid for each task $k$:
$$d_k = \frac{1}{|\mathcal{C}_k|} \sum_{s \in \mathcal{C}_k} \text{cos\_sim}(z_s, \mu_k)$$
We then calibrate the raw similarity by dividing it by the expected dispersion scale:
$$u_{k, b} = \frac{s_{k, b}}{d_k}$$

### E. Temperature-Scaled Softmax Routing
For in-distribution samples, the calibrated similarities are converted into sample-specific routing coefficients $\alpha_{k, b}$ using a temperature-scaled Softmax:
$$\alpha_{k, b} = \frac{\exp(u_{k, b} / \tau)}{\sum_{j=1}^K \exp(u_{j, b} / \tau)}$$
where $\tau = 0.05$ is the default temperature hyperparameter under standard and non-linear settings, and we set $\tau = 0.001$ under highly optimized, low-noise expert regimes. If the maximum similarity is below an OOD threshold ($\max_j s_{j, b} < \gamma_{\text{OOD}}$), we execute a uniform ensembling fallback ($\alpha_{k, b} = 1/K \quad \forall k$) to preserve representational diversity.

### F. Dynamic Activation Blending (All Layers)
Because routing coefficients $\alpha_{k, b}$ are computed at Layer 0, they are immediately available to guide activation blending across **all subsequent layers** $l \in \{1, \dots, 12\}$ in a single forward pass.
For each transformer block $l$, the output activation $h_b^{(l)}$ is computed on-the-fly as:
$$h_b^{(l)} = h_b^{(l-1)} W_{\text{base}}^{(l)} + \sum_{k=1}^K \alpha_{k, b} \left( h_b^{(l-1)} A_k^{(l)} B_k^{(l)} \right)$$
where $A_k^{(l)}$ and $B_k^{(l)}$ are the rank $r=8$ down- and up-projection matrices of expert LoRA $k$.

---

## 4. Architecture Specifications
- **Base Backbone:** Vision Transformer (`vit_tiny_patch16_224`), FP32, $L=12$ transformer blocks, intermediate feature dimension $D=192$.
- **Expert LoRA Adapters:** $K=4$ experts (MNIST, Fashion-MNIST, CIFAR-10, SVHN). LoRA rank $r=8$, scaling factor $\alpha_{\text{lora}}=16$. Adapters are inserted into **all 12 layers** (from Layer 1 to Layer 12) in parallel.
- **Calibration Split Size:** $|\mathcal{C}_k| = 64$ samples per task.
- **Routing Feature Space:** Layer 0 Patch Embedding output space ($\mathbb{R}^{192}$).
- **Hyperparameters:** Temperature $\tau = 0.05$ (standard) / $\tau = 0.001$ (low noise). OOD rejection threshold $\gamma_{\text{OOD}} = 0.05$.

---

## 5. Baselines
PEAR will be evaluated against five core baselines inside the Isolating Coordinate Sandbox (ICS):
1. **Expert Ceiling (0 params):** Oracle routing to isolated specialists.
2. **Uniform Merging (0 params):** Static weight averaging of expert adapters.
3. **Linear Router (Reg) (10,752 params):** L2-regularized parametric classical router.
4. **PFSR + MBH SOTA (0 params):** Non-parametric classification-head routing with sequential Micro-Batch Homogenization (MBH) partitioning.
5. **SABLE SOTA (0 params):** Activation-space ensembling with Mid-Layer Routing (Late Adaptation), leaving Layers 1--10 completely unadapted.

---

## 6. Step-by-Step Interaction
1. **Input Batching:** The edge device receives a heterogeneous batch of images $X = \{x_1, \dots, x_B\}$.
2. **Patch Embedding & Pooling:** Images are processed through the frozen Patch Embedding layer (Layer 0). Features are average-pooled spatially to extract early representations $z_b \in \mathbb{R}^{192}$.
3. **Subspace Cosine Similarity:** Cosine similarities $s_{k, b}$ are computed between $z_b$ and the pre-computed early task centroids $\mu_k \in \mathbb{R}^{192}$.
4. **Dispersion Calibration:** Cosine similarities are calibrated by dividing by expected task dispersion scales $d_k$, producing $u_{k, b}$.
5. **Coefficient Normalization:** A temperature-scaled Softmax ($\tau=0.05$ or $\tau=0.001$) converts similarities into routing coefficients $\alpha_{k, b}$.
6. **Unified Forward Pass:** The batch is propagated through Layers 1 to 12. At each layer, base model compute is executed exactly once, and parallel LoRA adapter activations are blended sample-wise using $\alpha_{k, b}$.
7. **Task-Agnostic Head Blending:** Final outputs are evaluated only on the top classification heads to yield final prediction logits, bounding complexity and completing execution in a strictly single pass.
