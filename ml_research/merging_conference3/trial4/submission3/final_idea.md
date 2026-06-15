# Deconstructing the "Re-Quantization Silence": A Methodological Audit of QLoRA Adapter Merging

## 1. Persona Alignment
This project aligns directly with **The Methodologist** persona. It focuses on exposing and analyzing a widespread "silent failure" in modern deep learning deployment pipelines: the naive merging of Parameter-Efficient Fine-Tuning (PEFT) adapters (specifically QLoRA/LoRA) back into quantized base models. 
* **Skepticism of SOTA Claims:** Popular tutorials and papers claim that QLoRA models can be "seamlessly merged and deployed." We critically audit this claim, exposing that the standard post-hoc re-quantization step silently obliterates low-rank updates.
* **Methodological Rigor:** Instead of proposing a flashy new architecture, we conduct a systematic, multi-axial audit of this "Re-Quantization Collapse" across multiple quantization schemas, bit-widths, and datasets. We evaluate simple, mathematically rigorous baselines and test-time optimization corrections to find a robust, generalizable solution.

---

## 2. Core Techniques
We introduce and evaluate three primary techniques to study and mitigate Re-Quantization Collapse:
1. **Re-Quantization Auditing Framework (RQA):** A diagnostic framework that computes the exact representation drift and performance loss when full-precision LoRA updates are merged into dequantized base weights and subsequently re-quantized back to low-bit (e.g., 4-bit) target configurations. We evaluate this across multiple quantization schemas (Symmetric vs. Asymmetric, Per-Tensor vs. Per-Channel) and bit-widths (4-bit, 8-bit).
2. **Quantization-Aware Adapter Coefficient Search (QA-ACS):** Rather than optimizing coefficients in full-precision space and applying post-hoc quantization (which suffers from quantization-operator overfitting), QA-ACS performs test-time optimization of layer-wise merging coefficients $\Lambda \in [0, 1]^{K \times L}$ *through* the re-quantization operator. We leverage the Straight-Through Estimator (STE) and 1+1 Evolution Strategy (ES) on a tiny calibration set ($N=16$) to find coefficient configurations that actively guide continuous weight updates to survive discrete quantization thresholds.
3. **Scale-Adaptive Weight Shifting (SAWS):** A zero-data, closed-form mathematical mitigation. SAWS computes a layer-wise scale factor based on the ratio of the L2-norms of the base weight matrix $W_0$ and the merged low-rank update matrix $\Delta W_{\text{merged}}$. By scaling up the low-rank updates prior to re-quantization and applying a corresponding correction to the layer outputs, SAWS ensures that the subtle adapter signals are not truncated to zero by the quantization grid.

---

## 3. Mathematical Formulation

### 3.1. QLoRA Merging & Naive Re-Quantization
Let $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ represent the pre-trained weights of a base model. Under QLoRA, $W_0$ is quantized to $Q_b(W_0)$ using a $b$-bit quantization operator. 
For $K$ task-specific experts, we fine-tune low-rank adapters in high-precision (FP16):
$$\Delta W_k = A_k B_k, \quad A_k \in \mathbb{R}^{d_{\text{out}} \times r}, B_k \in \mathbb{R}^{r \times d_{\text{in}}}$$
During model merging, we first dequantize the base model back to high-precision:
$$\tilde{W}_0 = \text{Dequantize}(Q_b(W_0))$$
The continuous merged weight matrix at layer $l$ under blending coefficients $\Lambda$ is defined as:
$$W^l_{\text{merged}}(\Lambda) = \tilde{W}^l_0 + \sum_{k=1}^K \lambda^l_k (A^l_k B^l_k)$$
To deploy the merged model back into low-bit memory constraints, the naive pipeline re-quantizes the merged matrix:
$$W^l_{\text{re-quant}}(\Lambda) = Q_b\left(W^l_{\text{merged}}(\Lambda)\right)$$
Under uniform symmetric quantization, the quantizer scale factor $s$ is dynamically computed as:
$$s = \frac{\max(|W^l_{\text{merged}}(\Lambda)|)}{2^{b-1} - 1}$$
The re-quantized discrete weights are:
$$W_{\text{quant}} = \left[ \left\lfloor \frac{W^l_{\text{merged}}(\Lambda)}{s} \right\rceil \right]_{-2^{b-1}+1}^{2^{b-1}-1}$$
and the dequantized weights used during inference are:
$$\tilde{W}^l_{\text{final}}(\Lambda) = s \cdot W_{\text{quant}}$$

### 3.2. Quantization-Aware Adapter Coefficient Search (QA-ACS)
We formulate QA-ACS to optimize the layer-wise coefficient matrix $\Lambda \in [0, 1]^{K \times L}$ by minimizing prediction entropy directly on the re-quantized model weights $\tilde{W}_{\text{final}}(\Lambda)$:
$$\mathcal{L}_{\text{entropy}}(\Lambda) = -\frac{1}{N \cdot K} \sum_{k=1}^K \sum_{i=1}^N \sum_{c=1}^C p_{k, i}(c) \log p_{k, i}(c)$$
where $p_{k, i}(c)$ represents predictions using $\tilde{W}^l_{\text{final}}(\Lambda)$. Under gradient-based optimization, we propagate gradients through the rounding operator using the Straight-Through Estimator (STE):
$$\frac{\partial \lfloor x \rceil}{\partial x} \approx 1$$
yielding:
$$\Lambda^{(t+1)} = \Lambda^{(t)} - \eta \cdot \text{Adam}\left(\nabla_{\Lambda} \mathcal{L}_{\text{entropy}}(\Lambda^{(t)})\right)$$

### 3.3. Scale-Adaptive Weight Shifting (SAWS)
To prevent the low-rank updates from being crushed, SAWS applies a closed-form scaling. We define the layer-wise adaptation norm ratio $\gamma^l$:
$$\gamma^l = \alpha \cdot \frac{\| \tilde{W}^l_0 \|_F}{\| \sum_{k=1}^K \lambda^l_k A^l_k B^l_k \|_F}$$
where $\| \cdot \|_F$ is the Frobenius norm and $\alpha$ is a hyperparameter scaling constant (e.g., $\alpha = 0.1$). We scale up the adapter updates before merging:
$$W^l_{\text{saws}}(\Lambda) = \tilde{W}^l_0 + \gamma^l \sum_{k=1}^K \lambda^l_k A^l_k B^l_k$$
and re-quantize $W^l_{\text{saws}}(\Lambda)$. To preserve the original output scale of the layer for an input activation $X$, we apply a diagonal correction scaling to the output of the dequantized layer:
$$Y = \left( X \cdot \tilde{W}^l_{\text{final, saws}}(\Lambda)^T \right) \odot V^l$$
where $V^l$ is a dimension-wise correction vector computed during calibration to balance the scale shift, or simply a scalar correction $1/\gamma^l$ applied to the task vector activation component.

---

## 4. Architecture Specifications
* **Backbone Model:** Vision Transformer (`vit\_tiny\_patch16\_224`) from the `timm` library (5.7M parameters).
* **Layer Partitioning:** $L=14$ layers (Patch Embeddings, 12 Attention Blocks, and the final Layer Normalization layer).
* **PEFT Module:** Low-Rank Adaptation (LoRA) of rank $r=8$ targeting query ($W_q$), key ($W_k$), and value ($W_v$) projections in all 12 self-attention blocks.
* **Quantization Specifications:** 4-bit and 8-bit uniform symmetric and asymmetric quantization (both per-tensor and per-channel granularity).
* **Trainable Parameters:** Merging coefficient matrix $\Lambda \in [0, 1]^{4 \times 14}$ (56 parameters).

---

## 5. Baselines
We evaluate our proposed framework against the following critical baselines to isolate the source of failure and verify the effectiveness of our mitigations:
1. **Unmerged FP16 LoRA Experts:** The upper bound of task-specific performance before merging (MNIST, FashionMNIST, CIFAR-10, SVHN).
2. **Naive FP16 LoRA Merge:** Full-precision Uniform Task Arithmetic merging of the LoRA adapters, establishing the unquantized merging ceiling.
3. **Naive Re-Quantized Merge (Naive-RQ):** Uniform merging followed by post-hoc re-quantization back to 4-bit target schemas without any mitigation, representing the current flawed industry practice.
4. **Post-Hoc Quantized AdaMerging:** Optimizing continuous coefficients in FP16 to minimize entropy, followed by post-hoc re-quantization. This isolates whether quantization-aware optimization is necessary.
5. **Decoupled "Quantize-then-Merge" (Q-then-M):** Separately quantizing the base weights and the adapter weights, and executing a dual-path forward pass at test-time. This baseline serves as a resource/latency comparison strawman.

---

## 6. Step-by-Step Interaction
The flow of data and computations through our proposed evaluation and mitigation pipeline is as follows:

1. **Expert Training & Initialization:** LoRA adapters are trained in high-precision (FP16) on their respective tasks with the backbone frozen.
2. **Dequantization of Base Model:** The 4-bit base model weights $Q_b(W_0)$ are dequantized to FP16 to enable weight-space addition.
3. **Mitigation/Optimization Hook:**
   * *If running SAWS:* We compute the Frobenius norm ratio $\gamma^l$ and scale up the merged adapter updates before addition.
   * *If running QA-ACS:* We initialize $\Lambda$ and pass calibration data through step 4-5.
4. **Weight-Space Fusion:** The dequantized base weights and the (potentially scaled) merged adapter updates are summed together in continuous FP16 space.
5. **Re-Quantization Operator:** The merged continuous weights are passed through the $b$-bit quantization grid (computing dynamic scales $s$ and zero-points $z$), and immediately dequantized to produce $\tilde{W}_{\text{final}}(\Lambda)$.
6. **Inference & Prediction:** Input activations flow through the network utilizing the final dequantized weights $\tilde{W}_{\text{final}}(\Lambda)$.
7. **Test-Time Optimization Loop (for QA-ACS):** Prediction entropy is computed on the calibration set. Gradients of the loss with respect to continuous coefficients $\Lambda$ are backpropagated using the Straight-Through Estimator (STE) to update $\Lambda$. Steps 4-7 are repeated for 100 steps.
8. **Final Evaluation:** The optimized/mitigated re-quantized model is evaluated on the full test sets of all 4 tasks to report final multi-task accuracy and Cross-Schema Generalization Gap.
