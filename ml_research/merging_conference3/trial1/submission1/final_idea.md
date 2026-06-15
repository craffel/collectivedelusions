# QP-Merge: Quantization-Preserving Task Vector Merging

## 1. Persona Alignment
This proposal is directly guided by **The Pragmatist** persona. In real-world enterprise applications, models are rarely deployed in FP16 or FP32; instead, they are quantized to INT4 or INT8 to fit edge devices, satisfy inference latency SLA constraints, and drastically reduce serving costs. 
Traditional model merging methods (like Task Arithmetic, SyMerge, and OrthoMerge) focus solely on floating-point performance, ignoring the catastrophic degradation that occurs when merged models are subsequently quantized. 

`QP-Merge` directly addresses this practical deployment bottleneck by:
1. **Preserving low-bit quantization capability (INT4/INT8):** It decouples extreme weight outlier components (the top $\le 1\%$) that typically stretch and ruin quantization scales, while representing them in a highly sparse, high-precision format that has near-zero computational or storage overhead.
2. **Requiring zero labeled data or costly fine-tuning:** It optimizes quantization and merging scaling factors over a lightweight set of unlabeled calibration samples, making it extremely robust and fast to deploy in the wild where labeled downstream data is scarce.
3. **Simplicity and Ease of Integration:** It modifies only post-merging weights and introduces a standard sparse-dense split, which is trivial to implement in PyTorch and integrates seamlessly with common deployment runtimes (e.g., TensorRT, vLLM).

---

## 2. Core Techniques
`QP-Merge` introduces two key synergistic techniques:

1. **Outlier-Residual Decoupling (ORD):**
   Weight merging processes (especially subtraction from pre-trained weights to create task vectors) often generate heavy-tailed distributions with extreme outliers. In low-bit quantization (e.g., symmetric linear quantization), these outliers stretch the quantization scale factor, causing massive precision loss (distortion) for the standard weight range. We identify these task-specific outliers and decouple them into a highly sparse FP16 tensor ($W_{\text{outlier}}$) with a density threshold $\tau = 1\%$, while the dense, low-range remainder is merged with $W_{\text{base}}$ to form $W_{\text{dense}}$ for low-bit quantization.

2. **Quantization-Error Aware Scale Calibration (QE-Calib):**
   Because different tasks have distinct activation scale dynamics, uniform merging coefficients lead to scaling mismatches under quantization. We use a tiny calibration set of $M = 128$ unlabeled domain samples to optimize a layer-wise diagonal weight scaling matrix $D$ and task weights $\lambda$ using a mean-squared error (MSE) reconstruction objective on the activations. This is a form of post-training quantization (PTQ) calibration tailored specifically for model merging.

---

## 3. Mathematical Formulation

Let $W_{\text{base}} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ be the pre-trained weight matrix of a layer, and let $\Delta W_t \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ be the task vector for task $t \in \{1, \dots, T\}$, where:
$$\Delta W_t = W_t - W_{\text{base}}$$

### Technique 1: Outlier-Residual Decoupling (ORD)
We define the outlier mask $M_t \in \{0, 1\}^{d_{\text{out}} \times d_{\text{in}}}$ for each task vector based on a percentile threshold $\gamma$ (e.g., $99$-th percentile of absolute weights):
$$(M_t)_{i,j} = \begin{cases} 1 & \text{if } |\Delta W_{t, i, j}| \ge \text{Percentile}(|\Delta W_t|, \gamma) \\ 0 & \text{otherwise} \end{cases}$$

The task vector is decomposed into outlier and dense components:
$$\Delta W_{t, \text{outlier}} = M_t \odot \Delta W_t$$
$$\Delta W_{t, \text{dense}} = (1 - M_t) \odot \Delta W_t$$

The merged weight matrix $W$ before quantization is:
$$W = W_{\text{quantized\_base}} + W_{\text{outlier}}$$
where:
$$W_{\text{quantized\_base}} = W_{\text{base}} + \sum_{t=1}^T \lambda_t \Delta W_{t, \text{dense}}$$
$$W_{\text{outlier}} = \sum_{t=1}^T \lambda_t \Delta W_{t, \text{outlier}}$$

### Quantization Operator
The $b$-bit symmetric uniform quantization of $W_{\text{quantized\_base}}$ is defined as:
$$Q_b(W_{\text{quantized\_base}}) = \text{clamp}\left( \text{round}\left( \frac{W_{\text{quantized\_base}}}{S} \right), -2^{b-1}, 2^{b-1}-1 \right) \cdot S$$
where the scale factor $S$ is computed block-wise or channel-wise as:
$$S_c = \frac{\max_r |(W_{\text{quantized\_base}})_{r,c}|}{2^{b-1}-1}$$

Because the extreme outliers have been removed from $\Delta W_{t, \text{dense}}$, the range of $W_{\text{quantized\_base}}$ is tightly bounded, ensuring that the quantization bin size $S_c$ remains small, resulting in minimal quantization noise.

### Technique 2: Quantization-Error Aware Scale Calibration (QE-Calib)
Given input activations $X_l$ for layer $l$ from the calibration dataset, we optimize a layer-wise diagonal scaler $D_l = \text{diag}(d_{l, 1}, \dots, d_{l, d_{\text{in}}})$ and merging weights $\lambda$ to minimize the reconstructed activation mean-squared error:
$$\mathcal{L} = \sum_{l=1}^L \mathbb{E}_X \left[ \| X_l W_{l, \text{FP32}} - X_l \left( Q_b(W_{l, \text{base}} + D_l \sum_{t=1}^T \lambda_t \Delta W_{t, l, \text{dense}}) + \sum_{t=1}^T \lambda_t \Delta W_{t, l, \text{outlier}} \right) \|_F^2 \right]$$
where $W_{l, \text{FP32}} = W_{l, \text{base}} + \sum_{t} \lambda_t \Delta W_{t, l}$ is the unquantized merged weight. This optimization runs for only 100 iterations of Adam, taking a few minutes on CPU/GPU.

---

## 4. Architecture Specifications
For each transformer block's linear layer (e.g., Query, Key, Value, Out projections in Self-Attention, and MLP Gates/Projections):

- **Dense Quantized Path ($W_{\text{quantized\_base}}$):**
  - **Bit-width ($b$):** 4-bit (INT4) or 8-bit (INT8) linear weights.
  - **Scale Type:** Per-channel (or per-block of size 128) symmetric quantization scales.
- **Sparse Path ($W_{\text{outlier}}$):**
  - **Data Type:** FP16 (or BF16 depending on base hardware).
  - **Sparsity:** $\ge 99\%$ sparse (density $\le 1\%$).
  - **Storage Format:** Coordinate-list (COO) format or structured sparse indices.
- **Activations:**
  - Standard floating-point representations (FP16 or BF16) throughout the transformer block.

**Layer Representation:**
```
            Input Activations (X)
             /                \
            /                  \
     Dense Quantized Path   Sparse Outlier Path
     [Q_b(W_dense)]         [W_outlier]
     (INT4/INT8 gemm)       (FP16 SpMM / sparse MM)
            \                  /
             \                /
              \              /
            Addition of Outputs (Y)
```

---

## 5. Baselines
The proposed `QP-Merge` method will be compared against the following baselines:

1. **FP32 Merged Upper Bound:** Standard model merging (Task Arithmetic, SyMerge, and OrthoMerge) without any quantization. This serves as the reference maximum performance.
2. **Naive Quantized Merging:** Models are merged using standard Task Arithmetic in FP32, and the resulting merged weight matrix is directly quantized to 4-bit or 8-bit using standard symmetric per-channel post-training quantization. This will demonstrate the extent of quantization-induced degradation.
3. **AdaRound / GPTQ Post-Merging:** The merged model is quantized using advanced post-training quantization algorithms (e.g., AdaRound or GPTQ/AWQ equivalent block-wise scaling) without separating the outliers or optimizing merging-specific scales.
4. **OFT / OrthoMerge + Naive Quantization:** Orthogonal merging followed by direct low-bit quantization, showing whether structural manifolds are robust to quantization without explicit outlier decoupling.

**Metrics:**
- **Accuracy:** Task performance across standard datasets (e.g., vision classification accuracy, NLP GLUE scores).
- **Latency:** Average inference latency per sample (or token) on edge/CPU configurations.
- **Memory Footprint:** Size of the model in storage and GPU/CPU VRAM.

---

## 6. Step-by-Step Interaction

Given a pre-trained base model and $T$ task-specific models fine-tuned on different tasks:

1. **Task Vector Extraction:**
   For each linear layer, compute the task vectors:
   $$\Delta W_t = W_t - W_{\text{base}}$$

2. **Outlier Partitioning:**
   - Sort the absolute weights of each task vector $\Delta W_t$.
   - Create a mask $M_t$ marking the top $1\%$ largest values.
   - Separate the task update into outlier component $\Delta W_{t, \text{outlier}} = M_t \odot \Delta W_t$ and dense component $\Delta W_{t, \text{dense}} = (1 - M_t) \odot \Delta W_t$.

3. **Calibration Data Preparation:**
   - Sample $M = 128$ unlabeled calibration instances from the target task domains (e.g., 16 samples per task if 8 tasks are merged).

4. **Quantization-Error Scale Calibration:**
   - Initialize diagonal scale matrix $D_l = I$ for each layer.
   - Initialize merging coefficients $\lambda_t = 0.3$ (or uniform scaling).
   - Feed the calibration samples through the FP32 model to capture intermediate activation outputs.
   - Perform 100 steps of Adam optimizer to update $D_l$ and $\lambda$ to minimize the L2 difference between the FP32 activations and the output of our hybrid Quantized-Dense + Sparse-Outlier layer formulation.

5. **Final Low-Bit Compression:**
   - Compute the final dense weight:
     $$W_{\text{dense\_final}} = W_{\text{base}} + \sum_{t=1}^T \lambda_t \Delta W_{t, \text{dense}}$$
   - Quantize $W_{\text{dense\_final}}$ to 4-bit or 8-bit integers using the optimized scale parameters.
   - Compress the outlier tensors $W_{\text{outlier\_final}} = \sum_{t=1}^T \lambda_t \Delta W_{t, \text{outlier}}$ using a sparse tensor layout.

6. **Inference Execution:**
   For any incoming test batch input $X$:
   - Compute the dense quantized gemm: $Y_{\text{dense}} = X \cdot Q_b(W_{\text{dense\_final}})$ (highly accelerated on hardware supporting low-bit tensor cores).
   - Compute the sparse gemm: $Y_{\text{outlier}} = X \cdot W_{\text{outlier\_final}}$.
   - Sum the outputs: $Y = Y_{\text{dense}} + Y_{\text{outlier}}$ and pass to the next layer/activation.
