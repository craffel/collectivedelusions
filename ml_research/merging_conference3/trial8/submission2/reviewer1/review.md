# Peer Review

## 1. Summary of the Paper
This paper addresses the critical bottleneck of deploying multi-task foundation models on resource-constrained edge CPUs and microcontrollers. While Parameter-Efficient Fine-Tuning (PEFT) adapters like LoRA minimize storage, serving dozens of concurrent experts in high-precision floating-point formats (FP16/FP32) quickly exceeds the tiny on-chip SRAM of edge devices and incurs heavy DRAM-SRAM transfer latencies during dynamic weight switching. To resolve these issues, the authors introduce **Q-SPS** (Quantized Single-Pass Activation-Space Dynamic Blending) and its execution-gated variant **CG-Q-SPS** (Conditional Gated Q-SPS). 

The proposed framework quantizes expert LoRA adapters to low-bitwidth symmetric integers (INT8/INT4) and executes the low-rank additions in pure integer precision within a single, parallel forward pass ($O(1)$ constant backbone latency). To prevent precision degradation under 4-bit quantization, the authors introduce **Quantization-Aware Scale Calibration (QASC)**, a post-hoc training-free protocol that sequentially decouples down-projection and up-projection scale optimization to reduce search complexity from $O(N^2)$ to $O(N)$. Furthermore, **CG-Q-SPS** applies a gating threshold ($\theta = 0.01$) to the dynamic routing coefficients ($\alpha_{k, b}$) derived from early-stage (Layer 3) Zero-Shot Centroid Alignment (ZCA) with **Intra-Task Dispersion Calibration (IDC)**. If an expert's routing weight falls below $\theta$, its execution is bypassed, scaling expert compute overhead down to the active experts. Finally, a Coordinate GMM safety shield is fitted over ZCA coordinates to detect and reject out-of-distribution (OOD) queries early.

The authors evaluate their methods using a hardware-calibrated analytical simulation of a 12-layer Vision Transformer (ViT-Tiny) across four diverse visual domains (MNIST, Fashion-MNIST, CIFAR-10, SVHN). In high-fidelity simulations, CG-Q-SPS (INT4) recovers **99.5%** of the unquantized FP32 expert ceiling, slashes expert memory footprints by **87.5%**, achieves a projected **3.97$\times$ physical speedup** over sequential micro-batch routing (PFSR+MBH SOTA), and reduces dynamic execution energy by **56.2%**.

---

## 2. Major Strengths
*   **Deep Systems-ML Co-Design:** The paper is highly commendable for its deep hardware-aware co-design. Rather than treating machine learning and systems engineering in isolation, the authors co-design their ensembling pipelines to exploit the physical register limits and cache structures of low-power CPUs (e.g., symmetric uniform quantization to avoid zero-point correction overhead, local batch re-ordering to maintain cache locality, and Neon instruction optimizations).
*   **Exemplary Intellectual Honesty and Theoretical Rigor:** The paper's theoretical exploration of task-representation entanglement is outstanding. The authors mathematically formulate and evaluate Gram-Schmidt Cross-Centroid Orthogonalization (GS-CCO) and L{\"o}wdin Symmetric Manifold De-Entangling (SMD) as theoretical extensions. Instead of presenting them as magical performance-boosting mechanisms, they show that explicit orthogonalization is actually mathematically redundant and even detrimental under noise due to "noise spillover" / "representation coupling." This negative finding is intellectually honest and highly valuable for researchers in coordinate-space routing.
*   **Highly Solid Reproducibility:** The paper provides complete mathematical formulations, detailed step-by-step algorithms, precise hyperparameter configurations, and hardware parameters, making the results highly reproducible.
*   **Outstanding Multi-Dimensional Trade-offs:** The proposed CG-Q-SPS achieves an exceptional Pareto-frontier, combining a **3.97$\times$ speedup**, **87.5% memory footprint savings**, **56.2% energy savings**, and a precise **95.2% TPR at 4.3% FPR** for OOD rejection, while recovering **99.5%** of the unquantized joint mean accuracy.

---

## 3. Weaknesses and Areas for Improvement
*   **SABLE / LoraHub Citation Conflation (Critical Bibliography Correction):** In Section 2 (Related Work), under the heading *"Activation-Space Blending and SPS-ZCA"*, the authors write:
    > *"SABLE (Sample-wise Activation Blending of Low-Rank Experts) \cite{huang2024lorahub} blends adapter activations layer-by-layer..."*
    
    However, the cited reference `huang2024lorahub` is **LoraHub** (*"LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition"*, COLM 2024). LoraHub is a gradient-free dynamic composition framework and is not abbreviated as "SABLE." SABLE appears to be a different framework or a direct acronym conflation by the authors. This bibliography expansion error must be corrected to maintain strict citation accuracy and proper historical mapping.
*   **Lack of Physical Benchmarking:** The primary limitation is that all reported accuracy and hardware metrics are simulated (inside the Isolating Coordinate Sandbox) rather than measured on real physical hardware. Although the cost model is calibrated against real Broadcom BCM2711 specifications, a small physical benchmark (e.g., executing a single INT4 vs INT8 vs FP16 low-rank GEMM using ARM Neon intrinsics on a Raspberry Pi 4) would greatly strengthen the paper's systems claims and verify the real impact of register unpacking instruction overheads.
*   **Framing of "Analytical Simulation" in Abstract/Title:** Given that this work is a simulated evaluation, the title and abstract should explicitly use the word *"Simulated"* or *"Analytical Simulation"* to manage reader expectations. The authors are highly transparent about this in the introduction and limitations sections, but moving this clarity to the title/abstract prevents any misinterpretation by readers expecting physical CPU benchmarks.
*   **Task-Separability at Early Blocks:** The paper utilizes early Layer 3 representations for task-agnostic routing. While highly sufficient for coarse-grained domains, this representation-depth trade-off would collapse for fine-grained, visually entangled tasks (such as medical imaging modalities or biological taxonomies) where early representations are near-identical. While briefly noted in the limitations, the paper would be significantly improved if the authors proposed or discussed a dynamic calibration protocol to determine the optimal routing block index based on task separation vs. latency.

---

## 4. Rating Categories

### Soundness: Good
The mathematical derivations, decoupled scale optimizations, and low-temperature stabilization are technically flawless. The hardware cost model is highly comprehensive, incorporating cache sizes, synchronization barriers, and register unpacking penalties. However, because the entire evaluation is situated within a simulated sandbox environment rather than a physical deployment, a rating of "Excellent" is held back. Nevertheless, the soundness is highly robust within the simulated boundaries.

### Presentation: Excellent
The paper is exceptionally well-written, clear, and logically structured. The narrative flow is easy to follow, and the mathematical formulations are mathematically complete and highly rigorous. Figures and tables are extremely rich in data and offer great visual clarity. Acknowledging relevant prior works and clearly discussing limitations represents a high standard of academic writing.

### Significance: Excellent
The proposed integer activation blending with sparse conditional gating represents a major practical breakthrough for edge serving and TinyML. Slashing expert memory footprints by 87.5% enables dozens of experts to fit natively inside tiny embedded SRAM buffers ($<512$ KB). Furthermore, proving that activation-space blending is completely immune to the collapse that destroys parameter-space model merging will likely influence future research in dynamic model serving.

### Originality: Excellent
The co-design of quantized PEFT serving with sparse conditional gating and sequential post-training scale calibration is highly original. The theoretical analysis of basis orthogonalization (GS-CCO, L{\"o}wdin SMD) and the characterization of the Hysteresis-Latency-Cache (HLC) Pareto frontier for sequential $B=1$ streams provide profound, highly original insights.

---

## 5. Overall Recommendation
**5: Accept**  
The paper is a technically solid, highly rigorous, and extremely well-written manuscript. It makes a significant, high-impact contribution to the sub-fields of Edge AI, TinyML, and Model Merging. While evaluated primarily in simulation and containing a minor bibliography conflation regarding SABLE/LoraHub, the depth of the systems-ML co-design, the rigor of the theoretical de-entangling exploration, and the outstanding quantitative trade-offs make this paper a strong candidate for publication. Addressing the citation error and clarifying the simulation framing in the title/abstract would make this paper flawless.

---

## 6. Questions for the Authors
1.  **SABLE/LoraHub Citation:** Could you clarify if "SABLE" is indeed a separate concurrent paper that was conflated with the `huang2024lorahub` citation in Section 2, and ensure the reference is corrected?
2.  **Physical Benchmark:** Do you have any plans (or preliminary data) to execute a physical micro-benchmark of the INT4 register unpacking and GEMM instruction loops using ARM Neon vector intrinsics on real hardware, such as a Raspberry Pi 4, to validate the simulated 15% compute penalty?
3.  **Dynamic Router Depth:** For fine-grained task registries where early Layer 3 representation space is entangled, how would the performance-latency Pareto frontier scale if the routing block index were dynamically shifted to a deeper block (e.g., Layer 6 or Layer 9)?
4.  **Scaling to LLMs:** While the scaling properties of CG-Q-SPS to larger edge-deployed LLMs (such as LLaMA-3.2-1B/3B) are theoretically analyzed, does the decoupled QASC scale calibration require a larger calibration split than 64 samples to handle the higher representation variance in deeper LLM layers?
