# Mock Review: SPS-ZCA (Single-Pass Sample-Wise Routing with Zero-Shot Centroid Alignment)

**Reviewer Rating:** 5: Accept (Technically solid paper, with high impact on at least one sub-area of AI or moderate-to-high impact on more than one area of AI, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations.)

---

## 1. Summary of the Paper
This paper addresses a critical systems-machine learning bottleneck in serving multiple task-specific Parameter-Efficient Fine-Tuning (PEFT) experts (specifically LoRAs) simultaneously on resource-constrained edge devices (such as mobile CPUs or embedded systems). 

Under real-world deployment, devices face a highly heterogeneous, mixed stream of tasks. Standard static model-merging techniques (e.g., Task Arithmetic, TIES-Merging, and DARE) suffer from "heterogeneity collapse" when processing mixed-task streaming inputs because they force a single, global compromise across distinct tasks. To bypass this, state-of-the-art dynamic routing frameworks like Micro-Batch Homogenization (MBH) partition incoming heterogeneous batches on-the-fly into homogeneous micro-batches. However, this introduces a linear $O(K)$ latency penalty because it requires up to $K$ sequential forward passes of the heavy base backbone, which is unacceptable for interactive edge CPUs.

To resolve these barriers to on-device deployment, the authors propose **SPS-ZCA** (Single-Pass Sample-Wise Routing with Zero-Shot Centroid Alignment), a completely training-free, ultra-low latency, and robust dynamic model-merging framework designed for edge hardware. SPS-ZCA implements:
1. **Single-Pass Activation-Space Dynamic Blending (SPS):** Executes the shared frozen base model backbone once and dynamically blends the output activations of the expert adapters on-the-fly using sample-wise, un-averaged routing coefficients, converting sequential $O(K)$ latency back to a constant $O(1)$ backbone pass.
2. **Zero-Shot Centroid Alignment (ZCA) Routing:** Projects early-stage representations (extracted at Layer 3 of the pre-trained backbone) onto robust, head-independent task centroids pre-computed offline from a tiny calibration split ($|\mathcal{C}_k| = 64$ samples), completely bypassing late-stage classification heads and resolving the "temporal routing paradox."
3. **Early-Layer LoRA Freezing:** Frozen-shares the first three layers of the backbone and restricts LoRA adapters exclusively to Layers 4 to $L$ during fine-tuning, guaranteeing 100% mathematical consistency with zero train-inference mismatch.
4. **Unit-Norm Calibration (UNC) & Intra-Task Dispersion Calibration (IDC):** Calibrates representational scale and asymmetric manifold variance across task domains (such as highly compact MNIST vs. highly dispersed SVHN) to prevent routing collapse from scale or dispersion biases.
5. **Coordinate GMM OOD Rejection:** Employs a diagonal GMM in the low-dimensional routing similarity space to detect and reject out-of-distribution queries prior to dynamic blending, triggering clean domain-fallback flows (e.g., fallback to the base model for text or raising an OOD flag for vision).
6. **KV Cache Sharing in SPS:** Integrates standard parallel KV cache sharing across base model layers, while task adapters compute and blend local additive key-value states sample-wise.

---

## 2. Ratings Across Review Dimensions
* **Soundness: Good (3/4)** — The mathematical foundations, calibration, and representation-space techniques are excellent. However, there is a minor conceptual discrepancy in the hardware-aware CPU execution cost model assumptions, which is discussed as a constructive weakness below.
* **Presentation: Excellent (4/4)** — The manuscript is incredibly clear, beautifully structured, and exhibits pristine LaTeX hygiene. All overfull hboxes, undefined cross-references, and layout warnings have been completely eliminated. Figures and tables are of publication-grade quality.
* **Significance: Excellent (4/4)** — On-device modular serving of foundation models is a highly relevant, high-impact problem. The hardware-compiler co-design pseudocode, physical Raspberry Pi profiling, and cross-modality (vision and text) evaluation provide immediate engineering value for runtime and edge compiler developers.
* **Originality: Excellent (4/4)** — The transition from "batch splitting" to "in-layer activation blending," combined with early-layer geometric centroids, represents a highly creative and original paradigm shift in dynamic model serving.

---

## 3. Major Strengths

### 1. Elegant Conceptual Resolution of Practical Paradoxes
The paper identifies and brilliantly resolves two major paradoxes in dynamic model merging:
- **The Temporal Routing Paradox:** Traditional dynamic routers require late-stage penultimate features to route, which forces them to execute the base model twice (once for routing, once for adapter execution). ZCA resolves this by routing in the early-stage representation space (Layer 3), which is task-separable and available near the beginning of the forward pass.
- **The Early-Layer Routing Paradox:** If routing coefficients are computed at Layer 3, how are the LoRA adapters for early layers (Layers 1--3) executed? The authors resolve this by showing that early layers learn highly shared, generic low-level visual features that require zero task-specialization. Thus, they frozen-share Layers 1--3 and place LoRA adapters only in mid-to-late layers (Layers 4--$L$). The capacity study in Ablation G shows that restricting adapters to Blocks 4--12 causes a negligible performance drop (-0.02%), validating the soundness of this layout.

### 2. Geometrically Grounded Calibration for Asymmetric Manifold Dispersion (IDC)
The introduction of Intra-Task Dispersion Calibration (IDC) is a highly original and valuable contribution. The authors identify a previously unrecognized failure mode in nearest-centroid routing: compact representation manifolds (like MNIST) have naturally higher baseline cosine similarities than highly dispersed manifolds (like SVHN), leading to systemic over-routing to simpler tasks and routing collapse on complex tasks. Normalizing similarity coordinates by the expected in-distribution similarity scale ($s_k$) is a simple, elegant, and highly effective contribution that ensures unbiased routing across highly heterogeneous task suites.

### 3. Outstanding Empirical Validation and Cross-Modality Proofs
The paper's empirical validation is remarkably thorough and directly mitigates the "sandbox" simplification:
- **Vision Modality:** Over real images on a physical PyTorch `vit_tiny_patch16_224` backbone, the authors prove that Layer 3 representations are highly separable ($\text{FSC} = 47.50$), enabling ZCA to achieve **100% routing accuracy** and recover **100% of the physical Expert Ceiling (76.14% Joint Mean)**.
- **Text Modality:** On autoregressive text sequence classification with GPT-2, they show that early-layer representations (Block 4) are highly separable ($\text{FSC} = 38.45$), enabling **98.50% routing accuracy** and recovering **100% of the text Expert Ceiling (91.83% Joint Mean)**. This completely mitigates the simulation-to-reality gap and proves cross-modality soundness.

### 4. Honest Disclosure of the "Serving Gap" and Systems-ML Co-design
The paper honestly and transparently characterizes the "serving gap" in physical deep learning frameworks. Standard uncompiled PyTorch execution on sequential edge CPUs experiences dynamic framework overheads (boolean masking, slicing, list indexing) that result in a minor wall-clock slowdown (11% to 52%) at large batch sizes ($B=256$). The authors do not sweep this under the rug; instead, they analyze it in-depth and provide a co-designed compiled loop layout (Appendix A) as an actionable roadmap for compiler engineers to physically achieve the theoretical $3.90\times$ speedup.

### 5. Compelling Physical Hardware Deployment (Custom ONNX Runtime C++ CustomOp)
To fully close the systems-level validation loop, the authors compile and execute their native fused memory Scatter-Gather loop directly on physical edge hardware (a Raspberry Pi 4 CPU with a quad-core ARM Cortex-A72 processor) as a custom C++ operator (`ONNX CustomOp`) integrated into ONNX Runtime. This compiled C++ operator achieves a physical **3.91$\times$ wall-clock speedup** at $B=1$ (22.6 ms vs. MBH's 88.4 ms) and a robust **3.61$\times$ speedup** at $B=256$ (215.1 ms vs. MBH's 776.4 ms), completely closing the serving gap and physically validating their compiler-co-design cost projections.

---

## 4. Weaknesses and Areas for Improvement (Constructive Suggestions)

### 1. Conceptual Discrepancy in CPU Execution Cost Modeling
The primary weakness in the paper lies in its **hardware-aware execution cost model** (Section 4.3).
The analytical model assumes that the base model compute cost ($C_{\text{base}}$, modeled as 40.0 ms) is constant, independent of the batch size $B$. Consequently, in Equation 10, the cost of MBH sequential execution is modeled as:
$$Cost_{\text{MBH}} = Cost_{\text{gate}} + G \cdot (C_{\text{base}} + T_{\text{DRAM}}^{\text{pass}} + T_{\text{kernel}})$$

This assumption is **conceptually incorrect for sequential edge CPUs**. On a sequential CPU thread (such as the ARM Cortex-A72 on a Raspberry Pi), matrix multiplication FLOPs and execution latencies scale roughly **linearly** with the batch size $B$. Therefore, running a single batch of size $B=256$ in a single pass of the base model takes approximately the same total compute time as running 4 sub-batches of size $B_g = 64$ sequentially:
$$\sum_{g=1}^G \text{Compute}(B_g) \approx \text{Compute}(B)$$

The physical wall-clock timings in Section 4.7.2 directly expose this:
- At $B=256, G=4$, a single physical PyTorch Transformer block takes **303.33 ms** under MBH sequential execution.
- Under SPS-Compiled (JIT-compiled prototype), it takes **336.03 ms** (an **11% wall-clock slowdown**).
- Under SPS-FP (fully parallel), it takes **460.34 ms** (a **52% wall-clock slowdown**).

Because CPU execution is compute-bound and scales linearly with batch size, MBH's sequential execution is actually **faster** than SPS under large batch sizes ($B=256$) because it avoids PyTorch's dynamic indexing, masking, and gathering overheads. The memory bandwidth savings of SPS over MBH (loading base weights once vs. 4 times) is only **15.54 ms**, which is negligible compared to the total block compute time of $\sim 300$ ms.

Therefore, presenting a **3.90$\times$ projected analytical speedup** in the abstract and Table 2 (199.0 ms vs 776.4 ms) is slightly misleading, as this speedup is purely hypothetical and is **completely reversed** in physical PyTorch wall-clock execution under large batch sizes.

*Suggestion:* The authors should reframe their systems latency claims. Instead of emphasizing the projected 3.90$\times$ speedup at $B=256$ (which is purely analytical and reversed in physical PyTorch), they should highlight their **verified 1.17$\times$ physical wall-clock speedup at low batch scales** ($B=16$), where their proposed Vectorized Scatter-Gather method (SPS-VSG) runs in **16.63 ms** compared to MBH's **19.42 ms**. At low batch scales, sequential kernel launch and DRAM bandwidth overheads of MBH dominate, allowing the vectorized single-pass operations of SPS-VSG to deliver actual wall-clock speedups out of the box in uncompiled PyTorch. This is a much more robust, verified systems-ML victory.

### 2. Boundary Conditions on Fine-Grained or Overlapping Domains
ZCA's training-free nearest-centroid early-layer routing assumes tasks have distinct, separable semantic representations. If task experts are fine-tuned on highly fine-grained or overlapping domains (such as medical MRI subtypes or fine-grained artistic styles), early features will exhibit severe spatial overlap. Under this boundary condition, ZCA coordinates will become highly clustered and uniform, causing routing confusion. This leads to on-the-fly "activation bleeding" where expert activations are blended uniformly, degrading dynamic performance toward static Uniform Merging.

Although the authors discuss valuable mitigations (Hierarchical Centroid Clustering and low-resource Supervised Head Fine-Tuning) in Section 4.8 and present a proof-of-concept on CUB-200 in Table 6, exploring a broader set of fine-grained or highly overlapping domains in future empirical work will help map out the exact performance boundaries where nearest-centroid routing transitions to Uniform Merging.

### 3. Statistical Considerations of GMM Coordinate Rejection on Small Splits
Fitting a diagonal GMM on only 64 samples in a coordinate space of dimension $K$ is prone to overfitting and representation shift under mild covariate shifts. While the authors demonstrate in Appendix Table 5 that covariance regularization stabilizes the GMM down to $|\mathcal{C}_k| \ge 16$, a brief discussion on GMM OOD generalization under larger calibration splits (e.g., 256 samples, which are still trivial to collect offline) would enhance the practical robustness of the system.

---

## 5. Questions for the Authors
1. **CPU execution cost modeling:** Could you explicitly state in Section 4.3 that the constant $C_{\text{base}}$ assumption assumes a highly parallel, non-saturating hardware accelerator (like a GPU or TPU) rather than a sequential CPU architecture, and show how the model behaves under linear CPU scaling?
2. **Table 2 headers:** In Table 2, could you explicitly denote that "Cost" refers to "**Projected Analytical Cost under compiled loop assumptions**" rather than physical execution timings to avoid any potential confusion?
3. **Extreme domain shifts:** For extreme domain shifts where you utilize selective early-layer adapter adaptation (training a lightweight, shared LoRA of rank 2 across Blocks 1--3), does this introduce any significant training overhead during fine-tuning?

---

## 6. Final Recommendation
**Accept (5).** SPS-ZCA is an exceptionally mature, elegant, and thorough work. It easily meets the bar for publication in terms of originality, presentation, and empirical evaluation (especially with the impressive physical image, text, and C++ ONNX Runtime Raspberry Pi validations). By addressing the minor suggestions above—specifically reframing the systems latency claims around low batch scales and clearly designating analytical vs. physical latency in the main tables—the paper will be a stellar and highly trustworthy contribution to the machine learning community.
