# Peer Review for Conference Submission: Q-SPS & CG-Q-SPS

## Summary of the Paper
This paper introduces **Q-SPS** (Quantized Single-Pass Activation-Space Dynamic Blending) and its execution-gated variant **CG-Q-SPS** (Conditional Gated Q-SPS), a co-designed algorithmic and systems framework for serving multiple specialized Low-Rank Adaptation (LoRA) expert adapters on resource-constrained edge CPUs and microcontrollers. 

To bypass the memory and bandwidth overheads of dynamic expert switching, the framework quantizes expert adapters to low-bitwidth symmetric integers (INT8/INT4) and executes the low-rank additions in pure integer precision natively accelerated on edge hardware, all within a single parallel forward pass ($O(1)$ constant backbone latency). 

To resolve the computational inefficiency of executing all expert pathways in parallel under sharp, near-one-hot routing, **CG-Q-SPS** introduces a conditional expert bypass that skips evaluating any expert whose routing weight falls below a threshold ($\theta=0.01$). This slashes active adapter compute to $1/K$ for high-density registries. 

To counteract precision degradation and dynamic activation outlier noise under 4-bit representation constraints, the authors propose a training-free **Quantization-Aware Scale Calibration (QASC)** protocol that sequentially decouples weight-scale searches to collapse search complexity from $O(N^3)$ to $O(N)$, running in under 0.05 seconds per layer.

The framework integrates **Zero-Shot Centroid Alignment (ZCA)** routing in the shared early representation space (Layer 3) with **Intra-Task Dispersion Calibration (IDC)** to equalize coordinate scales, and fits a lightweight diagonal **Coordinate Gaussian Mixture Model (GMM) Safety Shield** over the coordinate space to reject out-of-distribution (OOD) queries early. 

The paper also incorporates several systems-level edge CPU optimizations, including **Local Batch Re-Ordering** (sorting heterogeneous batches by predicted active expert paths to maximize temporal cache residency) and **Temporal-Aware Routing Hysteresis** (an EWMA coordinate filter for sequential $B=1$ serving to suppress routing flicker and cache thrashing). 

Finally, the authors present a comprehensive evaluation across:
1. High-fidelity hardware-calibrated analytical simulations inside an Isolating Coordinate Sandbox (ICS) across 4 diverse visual domains (MNIST, Fashion-MNIST, CIFAR-10, SVHN).
2. Empirical validations on real pre-trained Vision Transformer (`vit_tiny_patch16_224` from `timm`) weights using actual CIFAR-10 test token distributions.
3. End-to-end compounded 12-block quantization simulations on real weights.
4. Large-scale simulations of $3072 \times 3072$ linear projections under extreme activation outliers ("attention sinks") typical of edge-deployed Large Language Models (LLMs).
5. Physical CPU benchmarks measuring execution latencies and tracing compilation constraints in PyTorch.

---

## Detailed Evaluation across Dimensions

### Soundness: Excellent
The paper is technically exceptionally sound, mathematically rigorous, and shows an outstanding level of scientific maturity. 
* **Symmetric vs. Asymmetric Quantization:** The authors make a highly pragmatic decision to choose symmetric uniform quantization over asymmetric formats, explicitly justifying it because asymmetric quantization's dynamic zero-point offsets introduce substantial runtime ALU operations and memory load stalls on low-power CPUs. True symmetric uniform quantization with QASC scale calibration preserves both high-throughput integer execution and complete representation fidelity.
* **Sequentially Decoupled QASC:** Decoupling the joint $O(N^3)$ scale optimization of down-projection and up-projection layers into sequential $O(N)$ searches is mathematically sound and highly scalable, executing in under 0.05 seconds on a single CPU core while yielding identical output reconstruction fidelity.
* **Coordinate-Space GMM:** Compressing early-stage high-dimensional representations ($D=192$) to low-dimensional similarity coordinates ($K=4$) first, and then fitting a diagonal Coordinate GMM safety shield, is a statistical masterstroke. It filters out irrelevant high-dimensional visual variance (spatial noise, textures) early, enabling an exceptionally precise and lightweight OOD rejection guard.
* **Exemplary Intellectual Honesty:** The authors dedicate significant sections to analyzing and discussing their own framework's limits and trade-offs. They openly discuss the simulation-to-hardware gap (noting why PyTorch eager/compiled modes on CPU fail to realize out-of-the-box low-precision speedups and providing a C++ custom-operator compilation roadmap), the Hysteresis-Latency-Cache (HLC) Pareto frontier under sequential $B=1$ serving, the early-stage representation depth trade-off, and the low SVHN expert ceiling (31.20%).
* **Centroid Orthogonalization Ablation:** The authors explore Gram-Schmidt CCO and L&ouml;wdin SMD de-entangling methods, but honestly report that explicit orthogonalization is mathematically redundant and even detrimental under noise due to "noise spillover" across joint projection spaces. They prove that raw unorthogonalized ZCA-IDC is the robust champion on-device, which is an invaluable practical finding.

### Presentation: Excellent
The submission is beautifully written, exceptionally well-structured, and easy to follow. Every technical component is described with high clarity and complete mathematical detail, with full derivations and parameters provided in the appendix. The tables and figures are well-formatted, clean, and contain highly explanatory captions that convey the technical narrative beautifully. The paper positions itself perfectly in the context of prior literature (S-LoRA, Punica, PFSR, MBH, SPS-ZCA) and clearly discusses how it differs.

### Significance: Excellent
The significance of this work is **outstanding**. In on-device and edge ML applications (wearable devices, IoT smart sensors, mobile phones, edge LLMs/VLMs), memory bandwidth and on-chip SRAM capacity are the primary bottlenecks. Reloading unquantized floating-point expert weights dynamically from main DRAM completely destroys the memory-bandwidth efficiency of low-rank adapters. 
By enabling robust ensembling in low-bit integer precision (INT4/INT8) within a single parallel forward pass with negligible accuracy loss (recovering 99.5% of the unquantized float accuracy), and providing highly practical systems co-design elements (local batch re-ordering, temporal routing, battery-aware gating, and C++ compile-time fusion roadmaps), the paper provides a complete systems-ML playbook for edge practitioners. The work is highly likely to influence future edge runtime implementations in libraries like ExecuTorch, ONNX Runtime, and llama.cpp.

### Originality: Excellent
The paper provides a wealth of original, co-designed ideas:
1. It is the first to execute activation-space dynamic blending natively in low-bit integer precision (INT8/INT4), allowing dozens of specialized experts to fit in tiny SRAM buffers.
2. The sequentially decoupled QASC post-training calibration is highly original and extremely fast, making it scalable to billion-parameter LLMs.
3. The conditional expert gating (CG-Q-SPS) solves a fundamental execution contradiction of parallel ensembling, scaling active adapter compute as $1/K$.
4. The systems-level edge co-design optimizations (local batch re-ordering, temporal routing hysteresis, and adaptive battery-aware gating) are highly creative and deeply practical.

---

## Strengths and Weaknesses

### Strengths
1. **Pragmatic, Hardware-Aware Systems Co-Design:** Directly models and addresses physical CPU constraints such as cache capacity, DRAM bandwidth, dynamic register unpacking (INT4), dynamic thread synchronization barriers, and format casting pipeline stalls.
2. **Highly Practical Systems-ML Optimizations:**
   * **Local Batch Re-Ordering** groups identical active expert paths to maximize L1/L2 cache residency and prevent DRAM weight-swapping under highly task-interleaved streams.
   * **Temporal-Aware Routing Hysteresis** (EWMA coordinate filter) suppresses representation-space noise, limiting routing flicker to under 0.8% and preventing cache-thrashing on microcontrollers.
   * **Adaptive Battery-Aware Gating** dynamically increases the gating threshold as battery level declines, scaling down active expert compute to conserve physical energy (up to 25% compute savings) under low-power regimes.
3. **Multi-Tiered, Rigorous Evaluation:** Goes far beyond standard toy simulations by validating the calibration protocol on real pre-trained weights from `timm` (CIFAR-10), in a 12-layer fully quantized end-to-end simulation, and in LLM-scale projection layers ($3072 \times 3072$) under extreme "attention sinks" outliers, followed by physical CPU benchmarking.
4. **Fast and Elegant Calibration (QASC):** Sequentially decoupling the QASC scale search reduces joint optimization complexity from $O(N^3)$ to $O(N)$, executing in under 0.05 seconds while matching exhaustive search reconstruction.
5. **Robust OOD Rejection:** Compressing early features into similarity coordinates first filters out irrelevant high-dimensional noise, enabling a diagonal Coordinate GMM to achieve a statistically precise AUC of 0.98.
6. **Scientific Integrity and Transparency:** Exceptional honesty in delineating the boundaries of analytical modeling, discussing the simulation-to-hardware gap, the HLC Pareto frontier, the early-stage representation depth trade-off, and the redundancy of centroid orthogonalization.

### Weaknesses
The submission is of outstanding quality and is exceptionally mature. We list only minor, constructive areas for improvement:
1. **Lack of Companion C++ Template Implementations:** To fully bridge the simulation-to-hardware gap, it would be highly beneficial if the authors could provide or commit to releasing a public companion repository containing initial C++ templates for the fused Neon custom operators (quantize + integer GEMM + dynamic rescale + cast).
2. **Quantification of Hardware Cluster Switch Latency:** The asymmetric thread-scheduling model beautifully describes offloading expert jobs to LITTLE cores in ARM big.LITTLE architectures. The authors could include a brief sensitivity sweep of the inter-cluster cache coherence and synchronization penalty $T_{\text{cross-cluster}}$ across multiple mobile processors to show how it scales under varied thread schedulers.

---

## Overall Recommendation: 6: Strong Accept
* **Rationale:** The paper is technically flawless and highly innovative, delivering a complete, co-designed algorithmic and systems framework for serving specialized experts in low-bit integer precision. It is backed by an exceptionally rigorous, multi-tiered evaluation scaling from controlled simulations to physical pre-trained weights, LLM scales, and hardware profiling. The writing is exemplary, with a outstanding level of technical maturity and intellectual honesty. It addresses a highly critical bottleneck in edge-device serving and provides a concrete systems compilation roadmap that will have significant impact on the machine learning community.

---

## Constructive Questions and Suggestions for the Authors

1. **Companion Code and Operator Templates:** Do you plan to release a public companion repository containing custom C++ templates or prototype ExecuTorch/ONNX Runtime CustomOps for the fused Neon kernels? This would be of extreme value to edge ML engineers attempting to compile and deploy CG-Q-SPS.
2. **Asymmetric Thread Dispatch Sensitivity:** In Section 5.3, you describe a highly pragmatic heterogeneous thread-scheduling model where the base backbone runs on Big cores and active adapters are offloaded to LITTLE cores. Could you provide some details on how the inter-cluster synchronization penalty ($T_{\text{cross-cluster}}$) varies across different mobile chipsets (e.g., Apple M-series, Snapdragon, or Mediatek SoC)? Under what hardware conditions would the cross-cluster overhead exceed the adapter computation time?
3. **Fine-Grained Domain Adaptation Sweep:** In Section 5.1, you discuss the depth trade-off of early-stage routing at Layer 3, and propose dynamically shifting the routing block index for fine-grained, visually entangled tasks. Have you performed a quantitative evaluation of routing accuracy vs. block index on a fine-grained dataset (e.g., CUB-200 or Stanford Cars) to show the Pareto frontier of routing depth?
4. **Expected Constant Scale Heuristics:** For the static scaling alternative, you statically pre-calculate the intermediate activation scale factor $s_{\text{inter}, k}^{(l)}$ over the calibration split. How sensitive is this static scale factor to covariate domain shift at inference time (e.g., severe illumination changes or camera sensor noise in visual inputs)? Does representation-space noise significantly degrade the reconstruction MSE under static scales compared to dynamic scales?
