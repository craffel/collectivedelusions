# 3. Soundness of Methodology

## Mathematical Rigor and Correctness
The mathematical formulation of PEAR is rigorous, complete, and logically consistent. The pipeline is structured as follows:

1. **Early Representation Extraction:**
   $$Z_b = \text{PatchEmbed}(x_b) \in \mathbb{R}^{N \times D}$$
   $$z_b = \frac{1}{N} \sum_{i=1}^N Z_{b, i, :} \in \mathbb{R}^D$$
   The spatial average-pooling of Layer 0 patch tokens is clearly formulated. The paper correctly points out that this is mathematically equivalent to a linear projection of the global average color/brightness of the input image, leading to the **Global Average Color Routing Paradox** in real-world visual datasets.

2. **Zero-Shot Patch Centroids (ZPC):**
   $$\mu_{k, c} = \frac{1}{|\mathcal{C}_{k, c}|} \sum_{s \in \mathcal{C}_{k, c}} z_s^{(0)} \in \mathbb{R}^D$$
   Reference task coordinates are defined as class-wise means over the calibration split. This ensures a zero-shot, training-free anchoring system in the exact early-layer representational manifold.

3. **Subspace Cosine Projection (Unit-Norm Calibration):**
   $$s_{k, b} = \max_{j} \text{cos\_sim}(z_b, \mu_{k, j}) = \max_{j} \frac{z_b \cdot \mu_{k, j}}{\|z_b\|_2 \|\mu_{k, j}\|_2}$$
   The use of cosine similarity enforces scale-invariance, preventing representational magnitude biases from dominating the routing decisions.

4. **Intra-Task Dispersion Calibration (IDC):**
   $$d_k = \frac{1}{|\mathcal{C}_k|} \sum_{s \in \mathcal{C}_k} \text{cos\_sim}(z_s, \mu_{k, c_s})$$
   $$u_{k, b} = \frac{s_{k, b}}{d_k}$$
   This step is mathematically sound. It standardizes the similarity scores by dividing by the expected in-distribution similarity, which successfully corrects for asymmetric task manifold densities (e.g., highly concentrated tasks having artificially higher cosine similarities than diverse ones).

5. **Temperature-Scaled Softmax and OOD Rejection:**
   $$\alpha_{k, b} = \frac{\exp(u_{k, b} / \tau)}{\sum_{j=1}^K \exp(u_{j, b} / \tau)}$$
   $$\alpha_{k, b} = \frac{1}{K} \quad \forall k \quad \text{if } \max_{j} s_{j, b} < \gamma_{\text{OOD}}$$
   The fallback to a uniform distribution prevents logit nullification or predictable biases toward class 0 that would occur if adapter scales were simply set to zero in a multi-head expert setup.
   
6. **Adaptive Task-Specific Thresholding:**
   $$\gamma_{\text{OOD}, k} = \eta \cdot d_k$$
   By scaling the security thresholds dynamically with each task's expected representational density $d_k$, the OOD rejection rule becomes task-adaptive. A query is rejected if:
   $$s_{k, b} < \gamma_{\text{OOD}, k} \quad \forall k \in \{1, \dots, K\}$$
   This resolves the security-selectivity trade-off elegantly without requiring manual hyperparameter hand-tuning under varying task distributions.

7. **Hard Edge Rejection and Generalist Head:**
   To handle resource-constrained edge NPUs where loading $K$ parallel paths concurrently under a uniform fallback is impossible, the authors introduce a **Hard Edge Rejection** fallback. Here, $\alpha_{k,b}=0$ for all $k$, and the query is routed solely through the frozen base backbone to a dedicated **Generalist Classification Head** (a lightweight single-layer linear projection trained on base representations). This head is lightweightly optimized on the combined calibration split ($K \times B_{\text{cal}} = K \times 64$ samples) for 5--10 epochs on CPU, requiring $<1$ second of execution and zero external data. This prevents prediction logit nullification, providing edge operators with a mathematically sound and extremely frugal fallback option.

8. **Dynamic Activation Blending (All Layers):**
   $$h_b^{(l)} = h_b^{(l-1)} W_{\text{base}}^{(l)} + \sum_{k=1}^K \alpha_{k, b} \left( h_b^{(l-1)} A_k^{(l)} B_k^{(l)} \right)$$
   This equation correctly models standard parallel LoRA scaling and blending. Since $\alpha_{k, b}$ is computed early at Layer 0 (or slightly deeper), it can be applied to all blocks $l \in \{l_{\text{route}}, \dots, L\}$, bypassing the routing paradox and avoiding any mid-layer activation caching.

9. **Early-Layer Freezing during Training (ELFT):**
   When shifting the routing boundary deeper to layer $l_{\text{route}} \geq 1$, a training-serving representational mismatch arises at block $l_{\text{route}}-1$:
   $$h_{b,\text{serving}}^{(l_{\text{route}})} = \text{Block}_{l_{\text{route}}-1}\left(h_b; W_{\text{base}}\right)$$
   $$h_{b,\text{ideal}}^{(l_{\text{route}})} = \text{Block}_{l_{\text{route}}-1}\left(h_b; W_{\text{base}} + \Delta W_k\right)$$
   To resolve this boundary discrepancy, ELFT freezes early blocks during fine-tuning:
   $$\Delta W_k^{(l)} = 0 \quad \forall l < l_{\text{route}}, \quad \forall k \in \{1, \dots, K\}$$
   By aligning the training-time architecture with the serving-time inference path, ELFT completely neutralizes the boundary mismatch while preserving full-depth ensembling benefits for subsequent blocks.

## Systems-Level Feasibility and Realism
The paper exhibits exceptional systems-level awareness, which is a rare strength in algorithmic papers. The authors make several important systems-aware observations:
- **Sequential Latency vs. Compute Overhead:** The paper clearly distinguishes between *sequential timeline delay* and *computational FLOPs overhead*. When routing at Layer 1 or Layer 2, the first block(s) are executed as part of the standard forward pass and cached, meaning the actual FLOPs overhead is negligible ($<0.05$ ms), while the sequential latency delay is only $\sim 11\%$--$20\%$.
- **Hardware Bottlenecks (FLOPs and Memory Bandwidth Scaling Limits):** The authors explicitly discuss physical hardware limitations. While sequential depth complexity is flat $O(1)$, loading and executing $K$ parallel adapters concurrently introduces an $O(K)$ computational and memory bandwidth footprint. On resource-constrained edge NPUs or mobile devices with narrow memory bus widths, physical memory transfer serialization and thread concurrency limits can degrade serving speeds, rendering the flat $O(1)$ latency model invalid under massive expert counts $K$. This is an extremely thorough, systems-aware, and honest analysis.
- **Overfitting Mitigation Guidelines:** The authors recognize the risk of overfitting hyperparameters ($\tau$, $\gamma_{\text{OOD}}$) to the tiny calibration set. They propose highly practical, validation-free calibration methods, such as **Calibration-Relative OOD Thresholding** and **Validation-Free Temperature Calibration** using Shannon entropy targeting ($H(\alpha) \approx \rho \cdot \log(K)$). These guidelines provide robust engineering blueprints for real-world deployments.

## Overall Soundness Rating
The soundness of the methodology is **excellent**. The formulations are precise, the systems-level assumptions are grounded in real hardware characteristics, and the limitations are addressed with concrete, mathematically sound, and elegantly validated countermeasures (ELFT, Adaptive Thresholding, and Generalist Heads).
