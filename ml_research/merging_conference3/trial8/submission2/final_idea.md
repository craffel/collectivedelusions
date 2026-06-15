# Q-SPS: Quantized Activation-Space Dynamic Blending of Low-Rank Experts for Ultra-Low Footprint and High-Throughput Edge Serving

## 1. Persona Alignment
This project aligns perfectly with **The Pragmatist** persona:
*   **Real-world Edge Constraints:** On-device serving (e.g., on smart watches, IoT nodes, mobile phones) is severely memory-bound. While PEFT/LoRA reduces storage, serving $K$ unquantized experts in FP16/FP32 still occupies substantial on-chip SRAM/L1 cache and incurs heavy memory-bandwidth overheads during dynamic weight switching.
*   **Inference Latency & Energy:** Quantized integer operations (INT8/INT4) are natively accelerated by edge Neural Processing Units (NPUs), ARM Cortex CPUs, and microcontrollers, consuming up to $10\times$ less energy and providing substantial throughput speedups compared to floating-point operations.
*   **Direct Deployability:** Q-SPS enables serving dozens of concurrent experts within a tiny, constant memory budget without requiring expensive GPU acceleration, making multi-task foundation models accessible and robust in the wild.

## 2. Core Techniques
We introduce **Q-SPS** (Quantized Single-Pass Activation-Space Dynamic Blending), a training-free, memory-efficient serving framework that implements the following key mechanisms:
1.  **Low-Rank Quantized Adapters (QLoRA-SPS):** Low-bitwidth integer quantization (INT8/INT4 symmetric weight-only and activation quantization) applied directly to expert LoRA down-projection $A_k^{(l)}$ and up-projection $B_k^{(l)}$ matrices.
2.  **Integer-Precision Activation-Space Blending:** LoRA matrix multiplications executed entirely in quantized integer arithmetic inside a single forward pass, postponing dequantization scaling to the final adapter output to prevent precision loss.
3.  **Quantization-Aware Scale Calibration (QASC):** A training-free calibration mechanism using our 64-sample splits to compute optimal, task-specific activation scaling bounds, neutralizing the precision-degradation penalty of low bit-width representations.
4.  **Zero-Shot Centroid Alignment (ZCA) with GMM Coordinate Shield:** A highly robust early-stage (Layer 3) routing framework that uses Unit-Norm Calibration (UNC), Intra-Task Dispersion Calibration (IDC), and low-dimensional diagonal GMM coordinate density estimation to reject OOD noise before execution.

## 3. Mathematical Formulation
Let the shared base model weights in layer $l$ be $W_{\text{base}}^{(l)} \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$ (maintained in FP16 or FP32).
For each task expert $k \in \{1, \dots, K\}$, the down-projection adapter $A_k^{(l)} \in \mathbb{R}^{D_{\text{in}} \times r}$ and up-projection adapter $B_k^{(l)} \in \mathbb{R}^{r \times D_{\text{out}}}$ are quantized to INT8 or INT4 symmetric integers:

$$\bar{A}_k^{(l)} = \text{round}\left( \frac{A_k^{(l)}}{s_{A, k}^{(l)}} \right), \quad \bar{B}_k^{(l)} = \text{round}\left( \frac{B_k^{(l)}}{s_{B, k}^{(l)}} \right)$$

where $s_{A, k}^{(l)}, s_{B, k}^{(l)} \in \mathbb{R}$ are the floating-point quantization scale factors.

During inference, let $h_b^{(l-1)}$ be the FP16 input activation for sample $b$. To execute LoRA purely in integer precision, we dynamically quantize the activation to INT8 using a sample-wise scale $s_{h, b}^{(l-1)}$:

$$\bar{h}_b^{(l-1)} = \text{round}\left( \frac{h_b^{(l-1)}}{s_{h, b}^{(l-1)}} \right) \in \mathbb{Z}^{1 \times D_{\text{in}}}$$

The parallel quantized LoRA computation is executed purely in integer arithmetic:

$$\bar{z}_{b, k}^{(l)} = \left( \bar{h}_b^{(l-1)} \times \bar{A}_k^{(l)} \right) \times \bar{B}_k^{(l)} \in \mathbb{Z}^{1 \times D_{\text{out}}}$$

The intermediate low-rank representations are rescaled back to FP16 at the output of the adapter block using the combined scaling factor:

$$z_{b, k}^{(l)} = \bar{z}_{b, k}^{(l)} \cdot \left( s_{h, b}^{(l-1)} \cdot s_{A, k}^{(l)} \cdot s_{B, k}^{(l)} \right) \in \mathbb{R}^{1 \times D_{\text{out}}}$$

The final blended activation for layer $l$ is computed as:

$$h_b^{(l)} = h_b^{(l-1)} W_{\text{base}}^{(l)} + \sum_{k=1}^K \alpha_{k, b} z_{b, k}^{(l)}$$

where $\alpha_{k, b}$ are the sample-wise ZCA routing coefficients computed at Layer 3:

$$\alpha_{k, b} = \frac{\exp(u'_{k, b} / \tau)}{\sum_j \exp(u'_{j, b} / \tau)}, \quad u'_{k, b} = \frac{\text{cos\_sim}(h^{(3)}_b, \mu^{(3)}_k)}{s_k}$$

where $\mu^{(3)}_k$ is the pre-computed task centroid, $s_k$ is the expected intra-task dispersion calibration (IDC) scale, and $\tau = 0.001$.

## 4. Architecture Specifications
*   **Backbone:** 12-layer Vision Transformer (ViT-Tiny / ViT-Small) or transformer-based LLM (e.g. GPT-2).
    *   Hidden Dimension: $D = 192$, Attention Heads: 3, Feed-Forward Dimension: 768.
*   **Adapter Configuration:** LoRA adapters inserted in Layers 4 to 12. Rank $r = 8$, scaling factor $\alpha_{\text{lora}} = 16$.
*   **Quantization Formats:**
    *   Base model weights $W_{\text{base}}$: FP16/FP32.
    *   LoRA weights $A_k, B_k$: Quantized to INT8 or INT4 symmetric format.
    *   Activations $h^{(l-1)}$: Dynamically quantized to INT8 for the adapter path.
*   **Memory Footprint:** Quantizing the adapters to 4-bit cuts their SRAM storage and DRAM transfer footprint by **4$\times$** (INT8) to **8$\times$** (INT4) compared to FP32, enabling large multi-expert suites to fit natively on low-power microcontroller SRAM (e.g. $<512$ KB).

## 5. Baselines
We compare Q-SPS against the following baselines:
1.  **SPS-ZCA (FP32):** The unquantized floating-point activation-blending baseline, representing the theoretical accuracy ceiling.
2.  **PFSR + MBH (FP32):** The state-of-the-art non-parametric head-based routing with sequential micro-batch homogenization, which suffers from severe latency scaling.
3.  **Static Uniform Merging with Quantization:** Statically averaging quantized expert LoRA weights, which serves as a lower bound demonstrating "heterogeneity collapse".
4.  **Raw QLoRA:** Standard quantized LoRA adapters executed with sequential batch splitting.

## 6. Step-by-Step Interaction
Data flows through the Q-SPS serving pipeline as follows:
1.  **Early Execution:** Input batch $X = \{x_1, \dots, x_B\}$ of size $B$ is processed task-agnostically by the shared frozen early-stage layers (Layers 1--3) of the base model backbone, where no LoRA adapters are present.
2.  **ZCA Routing Extraction:** At the output of Layer 3, the representation $h^{(3)}_b$ for each sample is extracted.
    *   We compute the cosine similarity to the robust, pre-computed early centroids $\mu^{(3)}_k$.
    *   We divide the raw coordinates by the expected in-distribution dispersion scale $s_k$ (IDC) to handle task manifold variance.
    *   The coordinate GMM evaluates the log-likelihood of each sample; if below $\eta$, the query is rejected as OOD and routed to a fallback prediction flow (setting $\alpha_{k,b} = 0$).
    *   In-distribution samples are converted to sharp, sample-wise coefficients $\alpha_{k, b}$ using a temperature-scaled Softmax ($\tau = 0.001$).
3.  **Quantized Mid-to-Late Execution:** For each subsequent block $l \in \{4, \dots, L\}$:
    *   **Base Pathway:** The shared FP16 backbone executes the heavy GEMM $H W_{\text{base}}^{(l)}$ exactly once for the entire batch.
    *   **Quantized Adapter Pathway:**
        *   The input activation $h_b^{(l-1)}$ is dynamically quantized to INT8 using $s_{h, b}^{(l-1)}$.
        *   The quantized activations are routed to the active experts according to coefficients $\alpha_{k, b}$ and multiplied by the INT8/INT4 weights $\bar{A}_k^{(l)}, \bar{B}_k^{(l)}$ using highly optimized, low-power integer GEMM kernels on edge hardware.
        *   The integer output $\bar{z}_{b, k}^{(l)}$ is scaled back to FP16 in-place using the product scale $s_{h, b}^{(l-1)} s_{A, k}^{(l)} s_{B, k}^{(l)}$.
    *   **Activation Blending:** The FP16 base activation and scaled FP16 adapter activations are blended in-place using the coefficients $\alpha_{k, b}$, producing $h_b^{(l)}$.
4.  **Final Output:** The final Layer $L$ activations are passed to task-specific heads or generative fallback pipelines to output the predictions.
