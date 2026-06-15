# Summary of the Submission

## Main Topic and Objective
The submission addresses the challenge of deploying multi-task merged models onto resource-constrained edge platforms (such as microcontrollers, wearables, and IoT devices). Standard model merging techniques consolidate task-specific adapters (such as low-rank adapters, or LoRA) into a single shared backbone to save memory. However, deploying such merged models requires post-training quantization (PTQ) to low-bit integer formats (e.g., INT4/INT8) to fit within tight SRAM and flash limits. 

The paper identifies two main limitations of existing methods:
1. **Linear Scaling of Ensembling**: Parallel activation-space execution of multiple ($K$) expert adapters scales compute and memory footprint linearly ($O(K)$), which is prohibitively expensive for edge platforms.
2. **Representation Collapse under Non-linearities**: Static parameter-space merging (Post-Merge Quantization or PMQ) runs in constant $O(1)$ compute, but collapses catastrophically under realistic non-linear activation functions (e.g., GELU) due to weight-space interference and representation misalignment across non-linear layers. Furthermore, static merging lacks modularity, requiring a complete offline re-merging and re-quantization cycle to add or update tasks.

To address these limitations, the paper proposes **Scale-Aligned Quantized Activation Blending (SA-QAB)**, a training-free (though they later employ Quantization-Aware Fine-Tuning/QAT to close the accuracy gap), forward-pass-only framework designed for multi-task edge execution.

---

## Technical Approach (SA-QAB)
SA-QAB is designed to keep the base model and task experts decoupled during quantization and execute them in their native integer formats, dynamically blending activations on-the-fly. The key technical components are:

1. **Decoupled Heterogeneous Quantization (DHQ)**: 
   - Compresses the heavy, shared base backbone weights ($W_{\text{base}}$) per-channel to 4-bit signed integer format (**INT4**): $Q_4(W_{\text{base}}^{(l)})$.
   - Keeps the lightweight task-specific LoRA expert weights ($A_k, B_k$) in **INT8** format to preserve task representation boundaries: $Q_8(A_k^{(l)})$ and $Q_8(B_k^{(l)})$.
   
2. **Quantized Zero-Shot Centroid Alignment (Q-ZCA)**:
   - An early-stage dynamic routing layer operating at Layer 3 to perform dynamic sample-wise routing entirely on the integer manifold.
   - Computes task centroids offline over a 64-sample calibration set: $\mu_k^{(3)} = \frac{1}{|\mathcal{C}_k|} \sum_{s \in \mathcal{C}_k} h_s^{(3)}$, and quantizes them to INT8.
   - During on-device inference, the feature representation $h_b^{(3)}$ is quantized to INT8, and cosine similarity is computed in integer space via dot products to yield routing weights $\alpha_{k, b}$ using a temperature-scaled Softmax:
     $$\alpha_{k, b} = \exp(u_{k, b} / \tau) \Big/ \sum_{j=1}^K \exp(u_{j, b} / \tau)$$
   - On microcontroller hardware, hard-argmax is used to achieve sparse routing (active expert weight is set to 1, others to 0), ensuring that only the single active expert pathway is executed ($O(1)$ expert compute footprint).

3. **Quantization Scale Recovery (QSR)**:
   - Quantization induces scale contraction and representation distortion. QSR pre-computes scaling factors $\beta_k^{(l)}$ as the expected $L_2$ norm ratio of unquantized (FP16) to quantized (INT8) adapter activations across a calibration dataset of 64 samples:
     $$\beta_k^{(l)} = \frac{\mathbb{E}_{s \in \mathcal{C}_k} \left[ \| \text{Adapter\_FP}_k(h_s^{(l-1)}) \|_2 \right]}{\mathbb{E}_{s \in \mathcal{C}_k} \left[ \| \text{Adapter\_Quant}_k(h_s^{(l-1)}) \|_2 \right]}$$
   - These pre-computed factors are applied on-the-fly during inference to scale the INT8 adapter activations, correcting scale mismatches.

4. **Out-of-Distribution (OOD) GMM Rejection Gate**:
   - Trains a diagonal Gaussian Mixture Model (GMM) on Layer 3 features offline.
   - If log-likelihood falls below a threshold $\eta$, the sample is classified as OOD, dynamic routing is bypassed ($\alpha_{k, b} = 0$), and the input is processed solely by the frozen base backbone.

---

## Key Findings and Claims
- **Catastrophic Failure of Static Merging**: Static parameter merging (such as 4-bit Post-Merge Quantization or Q-Merge) suffers from near-random joint classification accuracy (18.60% and 22.20% respectively) in a non-linear network topology, while SA-QAB maintains a robust joint accuracy of **77.50%** (a **+58.90% absolute improvement**).
- **Physical Microcontroller Profiling Emulation**: Emulating an STM32H753XI microcontroller running CMSIS-NN kernels, SA-QAB requires **360.8 KB** of active SRAM (well within the 1 MB limit, whereas FP16 Ensembling requires 1224.8 KB and exceeds it). It achieves a latency of **0.836 ms** and energy of **0.3035 mJ** per inference (a **2.3x speedup** and **57% energy saving** over ensembling) with only a **3.7% latency overhead** (0.03 ms) over the collapsed Static 4-bit model.
- **Dimensional Isomorphism and Real-Pixel Transfer**: While the main evaluation is performed in a synthetic 192-dimensional Coordinate Sandbox (calibrated to represent MNIST, F-MNIST, CIFAR-10, and SVHN), they argue that this is dimensionally isomorphic to a physical ViT-Tiny model. They validate this on real pixels using ViT-Tiny and ResNet-18, reporting a joint accuracy of **84.80%** (3.6% below unquantized blending).
- **Mitigating PTQ Quantization Gap via Frozen-Base QAT**: Pure post-training quantization yields an accuracy of 50.00% (or 70.10% with SmoothQuant-like pre-scaling). Applying 5 epochs of Straight-Through Estimation (STE) fine-tuning to the lightweight INT8 adapters and classification heads (with a frozen INT4 base backbone) recovers performance to **77.50%**.

---

## Explicitly Claimed Contributions and Accompanying Evidence
1. **Decoupled Heterogeneous Quantization (DHQ)**: Evidence is provided in Table 2, where compressing the base to INT4 and adapters to INT8 is shown to prevent representation collapse.
2. **Quantization Scale Recovery (QSR)**: Formulated in Equation 4 and evaluated as part of the pipeline. They show it corrects scale contraction in low-bit adapters (such as 4-bit adapters), providing a +0.70% improvement.
3. **Quantized Zero-Shot Centroid Alignment (Q-ZCA)**: Formulated in Equations 2-3 and evaluated for routing specificity on both synthetic and real pixel suites (achieving 90% routing accuracy on real pixels under ViT-Tiny).
4. **Evaluation and Systems Trade-off Analysis**: Demonstrated through emulated microcontroller profiling (Table 3), Python-level PyTorch CPU benchmarking, task overlap sweeps, batch-size heterogeneity sweeps, and OOD GMM sensitivity sweeps.
