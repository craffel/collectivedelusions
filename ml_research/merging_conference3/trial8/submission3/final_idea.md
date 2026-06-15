# Idea Proposal: Scale-Aligned Quantized Activation Blending (SA-QAB)

## 1. Persona Alignment
This project is deeply grounded in the core philosophy of **The Pragmatist**:
- **Solving Real-World Bottlenecks:** It directly addresses the primary edge-deployment obstacle of model merging: the severe memory and storage footprint. To serve merged models on resource-constrained devices, quantization (e.g., INT4 or INT8) is mandatory. However, standard post-merge quantization collapses multi-task performance due to weight-space scale mismatch and quantization noise.
- **Pragmatic and Resource-Bounded Design:** Instead of employing heavy, complex test-time optimization loops or fragile Straight-Through Estimators (STEs) that require backpropagation and overfit catastrophically to the source quantization operator, SA-QAB is a training-free, pure forward-pass method. It executes the base backbone and low-rank experts in their native integer formats (INT4/INT8) and blends their activations, maintaining a constant $O(1)$ memory footprint and constant latency.
- **Robustness to Hardware Shifts:** By decoupling the quantization of the heavy base weights from the task-specific adapters, the system is highly robust to cross-schema shifts. If an edge chip requires a different quantization target (e.g., INT4 symmetric vs. asymmetric), our Activation Scale Alignment (ASA) generalizes immediately without requiring any coefficient re-optimization.

---

## 2. Core Techniques
SA-QAB introduces three primary inter-connected techniques:
1. **Decoupled Heterogeneous Quantization (DHQ):** The heavy, shared base model weights ($W_{\text{base}}$) are aggressively quantized to INT4 (using symmetric per-channel post-training quantization) to minimize memory transfer latency, while the lightweight LoRA adapters ($\Delta W_k = A_k B_k$) are quantized to INT8 to preserve task-specific representation precision.
2. **Quantized Zero-Shot Centroid Alignment (Q-ZCA):** Routing coordinates are extracted at an early stage (Layer 3) of the quantized base model using INT8-quantized features and INT8-quantized pre-computed task centroids, preventing the circular dependency of late-stage routing.
3. **Activation Scale Alignment (ASA):** A lightweight scale-alignment factor ($\beta_k^{(l)}$) is computed offline during a tiny 64-sample calibration phase. It adjusts the scale of the quantized LoRA activations to perfectly match the quantized base model activations, neutralizing scale drift and preventing activation bleeding.
4. **Single-Pass Quantized Activation Blending:** Merging the outputs of the quantized base model and the active LoRA experts dynamically in activation space in a single forward pass, completely avoiding parameter-space weight interpolation and dequantization overhead.

---

## 3. Mathematical Formulation

### 3.1 Heterogeneous Quantization Operator
Let $W$ be a weight matrix, and let $b$ be the bit-width. We define the symmetric per-tensor quantization operator $Q_b(W)$ and its dequantization scale factor $S$ as:
$$S = \frac{\max(|W|)}{2^{b-1} - 1}$$
$$Q_b(W) = \text{round}\left( \text{clip}\left(\frac{W}{S}, -2^{b-1}, 2^{b-1} - 1\right) \right)$$
For the base model weights, we use $b_{\text{base}} = 4$ (INT4), yielding $Q_4(W_{\text{base}}^{(l)})$ and $S_{\text{base}}^{(l)}$.  
For the LoRA adapter down-projection and up-projection matrices, we use $b_{\text{LoRA}} = 8$ (INT8), yielding $Q_8(A_k^{(l)})$, $S_{A, k}^{(l)}$, $Q_8(B_k^{(l)})$, and $S_{B, k}^{(l)}$.

### 3.2 Quantized Zero-Shot Centroid Alignment (Q-ZCA)
During calibration, we extract early-stage features $h_s^{(3)}$ at Layer 3 on full-precision models, and pre-compute the centroids:
$$\mu_k^{(3)} = \frac{1}{|\mathcal{C}_k|} \sum_{s \in \mathcal{C}_k} h_s^{(3)}$$
We quantize the centroids to INT8: $Q_8(\mu_k^{(3)})$ with scale $S_{\mu, k}$. During on-device serving, the early-stage activation $h_b^{(3)}$ is extracted and quantized to INT8: $Q_8(h_b^{(3)})$ with scale $S_{h}$.
The routing similarity score is computed directly in integer space:
$$u_{k, b} = \frac{Q_8(h_b^{(3)}) \cdot Q_8(\mu_k^{(3)})}{\|Q_8(h_b^{(3)})\|_2 \|Q_8(\mu_k^{(3)})\|_2}$$
The dynamic sample-wise coefficients $\alpha_{k, b}$ are obtained via a temperature-scaled Softmax:
$$\alpha_{k, b} = \frac{\exp(u_{k, b} / \tau)}{\sum_{j=1}^K \exp(u_{j, b} / \tau)}$$

### 3.3 Activation Scale Alignment (ASA)
Since the base model runs in INT4 and the adapters in INT8, their outputs have vastly different scales. To prevent scale drift, we compute the expectation of the L2 norms of the activations over the calibration set $\mathcal{C}_k$ offline:
$$\beta_k^{(l)} = \frac{\mathbb{E}_{s \in \mathcal{C}_k} \left[ \| \text{Base\_FP}(h_s^{(l-1)}) \|_2 \right]}{\mathbb{E}_{s \in \mathcal{C}_k} \left[ \| \text{Adapter\_FP}_k(h_s^{(l-1)}) \|_2 \right]}$$
where $\text{Base\_FP}(h) = h W_{\text{base}}^{(l)}$ and $\text{Adapter\_FP}_k(h) = h A_k^{(l)} B_k^{(l)}$ are the full-precision activations.

The dynamic blended output at layer $l$ is calculated as:
$$h_b^{(l)} = \text{GEMM\_INT4}\left(h_b^{(l-1)}, Q_4(W_{\text{base}}^{(l)})\right) \cdot S_{\text{base}}^{(l)} + \sum_{k=1}^K \alpha_{k, b} \beta_k^{(l)} \left( \text{GEMM\_INT8}\left(\text{GEMM\_INT8}\left(h_b^{(l-1)}, Q_8(A_k^{(l)})\right), Q_8(B_k^{(l)})\right) \cdot S_{A, k}^{(l)} S_{B, k}^{(l)} \right)$$

---

## 4. Architecture Specifications
- **Backbone Model:** Vision Transformer (`vit_tiny_patch16_224`) with $L=12$ blocks, $14$ total layer groups (Patch Embeddings, 12 blocks, head), feature dimension $D=192$.
- **Downstream Expert Tasks ($K=4$):** MNIST (grayscale, $28 \times 28$), Fashion-MNIST ($28 \times 28$), CIFAR-10 ($3 \times 32 \times 32$), SVHN ($3 \times 32 \times 32$).
- **Adapter Configuration:** LoRA adapters inserted in attention projection layers (query, key, value, projection) across Blocks 4 to 12. Rank $r=8$.
- **Quantization Schemas:**
  - Base model $W_{\text{base}}$: 4-bit INT4 per-channel symmetric quantization.
  - LoRA adapters $A_k, B_k$: 8-bit INT8 per-tensor symmetric quantization.
- **Routing Layer:** Centroid extraction at Layer 3 (Block 3 output), keeping Blocks 1--3 frozen and task-agnostic to prevent any train-test mismatch.

---

## 5. Baselines
We compare SA-QAB against five rigorous baselines:
1. **Expert Ceiling (FP16):** Fully isolated task-specific experts running in full FP16 precision.
2. **Post-Merge Quantization (PMQ - 4bit):** Standard uniform model merging in FP16 followed by aggressive 4-bit quantization of the merged weights.
3. **Q-Merge (STE - 4bit):** Quantization-Aware Model Merging optimized using the Straight-Through Estimator under 4-bit quantization.
4. **Q-Merge Cross-Schema Shift (4bit to 8bit):** Evaluating the robustness of Q-Merge's learned coefficients when the runtime hardware changes from 4-bit to 8-bit quantization.
5. **SPS-ZCA (FP16):** Full-precision single-pass activation-blending routing without any quantization constraints, acting as our performance ceiling.

---

## 6. Step-by-Step Interaction
1. **Offline Calibration (One-time):**
   - Extract Layer 3 features on 64 calibration samples per task. Compute the task centroids $\mu_k^{(3)}$ and quantize them to INT8.
   - For each late layer $l \in \{4, \dots, 12\}$, compute the expected activation norms for the base path and each adapter path to pre-calculate the Activation Scale Alignment factors $\beta_k^{(l)}$.
2. **Inference Pipeline:**
   - Input sample $x_b$ is passed through Patch Embedding and early blocks (Layers 1--3) of the base model in INT4 format.
   - At the output of Layer 3, the representation $h_b^{(3)}$ is quantized to INT8, and ZCA similarity coordinates $u_{k, b}$ are computed against pre-computed centroids.
   - Dynamic Softmax coefficients $\alpha_{k, b}$ are derived.
   - For late blocks (Layers 4--12), the INT4 base path and active INT8 LoRA expert paths are executed concurrently.
   - The LoRA activations are scaled by $\beta_k^{(l)}$ and blended using $\alpha_{k, b}$.
   - The blended representation is passed to the next layer.
   - Penultimate features are passed to the classification head to output the final prediction.
