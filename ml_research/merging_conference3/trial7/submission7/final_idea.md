# Idea Proposal: One-Pass Dynamic Model Merging via Early-Layer Adaptive Task Identification (ELATI)

## 1. Persona Alignment
This technical implementation directly aligns with the **Pragmatist** persona:
- **Inference Latency Reduction:** Previous dynamic model-merging routers (such as PFSR+MBH) suffer from a severe "two-pass" latency penalty. Because they route at the penultimate layer, they require a full, throw-away forward pass of the base model backbone just to obtain routing coefficients, followed by a second forward pass (or multiple sequential passes in MBH) to produce predictions. This requires a minimum of $2\times$ model forward latency. ELATI resolves this by routing at an **early layer** (e.g., Layer 2), allowing the downstream layers (Layers 3--14) to be dynamically merged and executed on-the-fly during a single, cohesive **one-pass** execution. This cuts dynamic routing latency almost in half (close to $1.0\times$ baseline latency) for homogeneous streams.
- **VRAM and Compute efficiency:** It maintains a strict PEFT (LoRA) adapter footprint (requiring only $\approx 1.04\times$ base model memory), keeping all experts loaded in memory and merging parameters dynamically in RAM/VRAM via highly optimized on-the-fly sequential materialization.
- **Robustness to Real-World Streaming Constraints:** In edge-AI and real-time streaming, latency is a first-class citizen. ELATI is a Relentless application of systems-level Occam's razor, making dynamic model merging highly viable on resource-constrained consumer-grade hardware.

## 2. Core Techniques
ELATI introduces three core techniques to establish a training-free, parameter-free, low-latency dynamic model-merging system:
1. **One-Pass Subspace Routing (OPSR):** Shifting the parameter-free subspace similarity projection from the penultimate layer of the backbone to an early layer $l_{route} \ll L$ (specifically, $l_{route}=2$ or $l_{route}=3$ in our $L=14$ layer network). This eliminates the expensive first-pass backbone execution.
2. **Early-Layer Representative Mapping (ELRM):** Because early representations $z_b^{(l_{route})}$ have not reached the final semantic classification layer, we compute early representative centroids to act as projection heads. For each expert $k$, we compute the mean activation centroid at layer $l_{route}$ over the 64-sample calibration split:
   $$W'_{k} = \frac{1}{|X^{(k)}|} \sum_{x \in X^{(k)}} \text{Backbone}_{1 \dots l_{route}}(x)$$
   where $X^{(k)}$ is the calibration split for task $k$. These frozen, data-free centroids capture the early representation-space characteristics of each expert, allowing precise task identification at early stages of the network with zero trainable parameters.
3. **Downstream-Only Micro-Batch Homogenization (DO-MBH):** For mixed-task batches, we partition the intermediate activations $z_{b}^{(l_{route})}$ into homogeneous micro-batches on the fly. We dynamically merge only the *downstream* layers $l > l_{route}$ for each active micro-batch, completely avoiding any redundant forward computation on early layers $l \le l_{route}$.

## 3. Mathematical Formulation
Let the network backbone have $L = 14$ layers and intermediate representation dimension $D = 192$. Let $K = 4$ represent the number of task experts (MNIST, F-MNIST, CIFAR-10, SVHN).

### Step 1: Early-Layer Forward Pass
An incoming heterogeneous batch $X = \{x_1, \dots, x_B\}$ of size $B$ is passed through the first $l_{route}$ layers of the shared base model backbone to extract intermediate representations:
$$z_b^{(l_{route})} = \text{Backbone}_{1 \dots l_{route}}(x_b) \in \mathbb{R}^D, \quad \forall b \in \{1, \dots, B\}$$

### Step 2: Unsupervised Early Representative Cosine Similarity Projection
We compute the cosine similarity between the intermediate representations and the pre-computed early expert centroids $W'_k \in \mathbb{R}^D$:
$$u_{k, b} = \frac{W'_{k} \cdot z_b^{(l_{route})}}{\|W'_{k}\|_2 \|z_b^{(l_{route})}\|_2}, \quad \forall k \in \{1, \dots, K\}$$
This yields a $K$-dimensional task coordinate vector $u_b = [u_{1,b}, \dots, u_{K,b}]^T \in \mathbb{R}^K$.

### Step 3: Temperature-Scaled Routing & Stream Partitioning
We apply temperature-scaled Softmax to obtain sample-specific routing coefficients:
$$\alpha_{k, b} = \frac{\exp(u_{k, b} / \tau)}{\sum_{j=1}^K \exp(u_{j, b} / \tau)}$$
where $\tau > 0$ is a pre-defined routing temperature.
For each sample, we identify its dominant task:
$$k_b^* = \arg\max_{k \in \{1, \dots, K\}} u_{k, b}$$
We dynamically partition the batch activation indices into $G \le K$ homogeneous micro-batches:
$$I^{(g)} = \{b \in \{1, \dots, B\} \mid k_b^* = g\}$$

### Step 4: Downstream Tailored Weight Merging & Forward Dispatch
For each active micro-batch $g \in \{1, \dots, G\}$ with corresponding indices $I^{(g)}$, we compute the batch-average routing coefficient:
$$\bar{\alpha}_k^{(g)} = \frac{1}{|I^{(g)}|} \sum_{b \in I^{(g)}} \alpha_{k, b}$$
We dynamically interpolate and materialize the downstream parameters for layers $l > l_{route}$:
$$W_{merged}^{(l), (g)} = W_{base}^{(l)} + \sum_{k=1}^K \bar{\alpha}_k^{(g)} V_k^{(l)}, \quad \forall l \in \{l_{route}+1, \dots, L\}$$
where $V_k^{(l)}$ represents the LoRA expert task adapter weights.
We dispatch the subset of intermediate activations $z_{I^{(g)}}^{(l_{route})}$ through the merged downstream layers:
$$Y^{(g)} = \text{Backbone}_{l_{route}+1 \dots L}\left( z_{I^{(g)}}^{(l_{route})}; \{W_{merged}^{(l), (g)}\}_{l=l_{route}+1}^L \right)$$

### Step 5: Output Scatter Re-assembly
We instantiate a blank output tensor $Y$ of size $B$ and populate its rows using the stored micro-batch indices to maintain sequential ordering:
$$Y[I^{(g)}] = Y^{(g)}$$

## 4. Architecture Specifications
- **Model Backbone:** $L=14$ layers, intermediate hidden representation dimension $D=192$, feedforward inner dimension $D_{ff}=768$, multi-head attention with $H=3$ heads.
- **Routing Layer Index ($l_{route}$):** Set to Layer 2 ($l_{route}=2$). This minimizes early computation while ensuring the representations are sufficiently rich to perform highly accurate task routing.
- **Downstream Dynamically Merged Layers:** Layers $3 \dots 14$ are dynamically merged on the fly based on early-layer routing coefficients.
- **Active Memory Buffer:** Exactly one scratch weight buffer of size $D \times D$ is allocated in VRAM to perform sequential on-the-fly materialization, restricting the peak hardware memory footprint to a strict maximum of $1.04\times$ the base model size.

## 5. Baselines
We evaluate ELATI against the following key baselines:
1. **Expert Ceiling:** Evaluates each sample on its corresponding active expert model, representing the empirical performance ceiling (79.80% Joint Mean).
2. **Static Uniform Merging:** Computes a static parameter average across all layers ($\bar{\alpha}_k = 0.25$), serving as the classical baseline (42.90% Joint Mean).
3. **Linear Router (Reg):** A classical calibrated dynamic router optimized on the 64-sample calibration split with $L_2$ regularization (51.00% Joint Mean).
4. **PFSR + MBH (Penultimate):** The state-of-the-art "two-pass" dynamic routing model from Trial 6 Submission 7, which projects features from the penultimate layer (Layer 13) and runs a separate first-pass forward execution (75.00% homogeneous Joint Mean, 71.60% heterogeneous Joint Mean).

## 6. Step-by-Step Interaction
1. **Input Batch Ingestion:** A shuffled, heterogeneous batch of samples $X$ is ingested by the model.
2. **Early Feature Extraction:** The batch $X$ is propagated through the first two layers ($l=1, 2$) of the frozen base model backbone to extract the intermediate representation matrix $Z^{(2)} \in \mathbb{R}^{B \times D}$.
3. **Subspace Similarity Gating:** For each sample representation $z_b^{(2)}$, we calculate the cosine similarity against the $K=4$ pre-computed early expert centroids $W'_k$, producing the $B \times K$ task coordinate matrix $U$.
4. **Active Stream Partitioning:** The task coordinate matrix is passed through a temperature-scaled Softmax to derive sample-wise routing coefficients. Samples are grouped into homogeneous micro-batches $X^{(1)}, \dots, X^{(G)}$ on-the-fly based on their argmax task coordinate.
5. **Dynamic Parameter Interpolation:** For each active micro-batch $g$, the sample coefficients are averaged. The downstream LoRA task adapters for Layers $3 \dots 14$ are linearly interpolated with the base weights and materialized into a single scratch buffer.
6. **Downstream Forward Propagation:** The intermediate activations $z_{I^{(g)}}^{(2)}$ are propagated through the remaining layers ($3 \dots 14$) of the dynamically merged model to compute the prediction logits $Y^{(g)}$.
7. **Transparent Scatter Re-assembly:** The predictions from all micro-batches are scattered back to their original index positions, returning a single, properly-ordered prediction tensor $Y$ to the outer application with zero PCIe latency or double-pass forward bottlenecks.
