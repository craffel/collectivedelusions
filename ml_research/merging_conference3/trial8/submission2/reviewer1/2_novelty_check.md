# Evaluation Phase 2: Novelty Check and Literature Context

## 1. Characterization of Novelty
The paper offers **significant and highly pragmatic novelty**. Instead of introducing complex, parameter-heavy neural architectures, the authors focus on a deep systems-ML co-design that combines and adapts existing techniques—such as low-rank adapters, post-training quantization, centroid-based routing, and Gaussian mixture models—specifically to satisfy physical edge-deployment constraints. 

Moreover, the novelty is elevated by the authors' **intellectual honesty and theoretical rigor**:
*   Instead of presenting their basis orthogonalization methods (GS-CCO and L{\"o}wdin SMD) as magical performance boosts, they conduct a rigorous, principled exploration of task-representation entanglement. They empirically demonstrate that explicit basis orthogonalization is actually mathematically redundant and even detrimental under noise due to "noise spillover" / "representation coupling."
*   They provide a complete systems-ML co-design that explicitly addresses physical hardware bottlenecks like cache locality degradation, pipeline stalls, and heterogeneous thread scheduling (e.g., Big.LITTLE architectures). They characterize the **Hysteresis-Latency-Cache (HLC) Pareto Frontier** for sequential $B=1$ streams, which is a highly novel and valuable practical insight.

---

## 2. Delta from Prior Work
The paper positions itself very clearly and carefully in the context of existing literature:

### A. Parameter-Space Model Merging (Task Arithmetic, TIES-Merging, DARE, ZipIt!)
*   **Prior Work:** Statically merges independent expert weights in parameter space, requiring zero runtime memory or compute overhead.
*   **Delta:** These static methods suffer from "heterogeneity collapse" when evaluated under realistic, interleaved multi-task streams because the merged model cannot adjust its weights dynamically per-sample, resulting in catastrophic cross-task interference. Q-SPS completely bypasses this via sample-wise activation-space blending, maintaining separate integer-precision expert pathways.

### B. High-Throughput Server-Side Serving (S-LoRA, Punica)
*   **Prior Work:** Uses specialized CUDA paging and memory-management kernels to serve thousands of concurrent LoRA adapters on high-end server GPU clusters.
*   **Delta:** These are designed strictly for server clusters. At the resource-constrained edge (CPUs, microcontrollers), GPU-centric paging and heavy orchestration overheads make these systems completely impractical. Q-SPS is explicitly co-designed for edge CPU register structures, cache line limits, and instruction-level integer pipelines.

### C. Edge-Routing SOTA (PFSR + MBH SOTA)
*   **Prior Work:** Parameter-Free Subspace Routing (PFSR) combined with Micro-Batch Homogenization (MBH) avoids cross-task interference by partitioning heterogeneous batches into task-homogeneous sub-batches.
*   **Delta:** Sequential micro-batch dispatching requires the heavy pre-trained base model backbone to run sequentially $G$ times (where $G$ is the number of active tasks in the batch), scaling latency linearly with the number of tasks. CG-Q-SPS executes dynamic sample-wise expert ensembling inside a single, parallel forward pass, achieving $O(1)$ constant backbone latency, and using conditional gating to scale the expert path overhead down to $O(1)$ under confident routing.

### D. Unquantized Activation Blending (SPS-ZCA, SABLE)
*   **Prior Work:** SPS-ZCA routes inputs task-agnostically at early layers and blends expert activations on-the-fly.
*   **Delta:** Prior activation-blending frameworks are restricted to high-precision floating-point execution (FP32/FP16), failing to exploit edge-native low-power integer accelerators. Q-SPS is the first to execute activation-space dynamic blending in pure integer precision (INT4/INT8) and introduces a training-free Quantization-Aware Scale Calibration (QASC) protocol to recover precision loss.

### E. Quantized PEFT / QLoRA (Dettmers et al., 2023)
*   **Prior Work:** Keeps the massive base model in 4-bit (using the NF4 data type) and trains/runs adapters in FP16/BF16 precision.
*   **Delta:** In standard QLoRA inference, the adapters are kept in floating point. Q-SPS/CG-Q-SPS quantizes the *adapters* themselves to low-bitwidth symmetric integers (INT8/INT4) and executes low-rank matrix multiplications entirely in pure integer precision.

---

## 3. Literature Attribution Check (Critical Scholarly Critique)
Upon a rigorous review of the bibliography and the citations in the related work, there is a minor **attribution conflation** that the authors should address:

*   **The SABLE Citation Error:** In Section 2 (Related Work), under the heading *"Activation-Space Blending and SPS-ZCA"*, the authors write:
    > *"SABLE (Sample-wise Activation Blending of Low-Rank Experts) \cite{huang2024lorahub} blends adapter activations layer-by-layer rather than merging weights, executing serving in a single forward pass."*
    
    However, checking `references.bib`, entry `huang2024lorahub` is:
    > *Chengsong Huang, Qian Liu, Bill Yuchen Lin, Tianyu Pang, Chao Du, Min Lin. "LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition" (First Conference on Language Modeling, COLM, 2024).*
    
    **The Critique:** The authors have conflated the name of **LoraHub** with the acronym "SABLE" (Sample-wise Activation Blending of Low-Rank Experts). LoraHub is a gradient-free, few-shot adapter composition technique, and does not carry the acronym SABLE. While SABLE might refer to a separate or concurrent work, citing `huang2024lorahub` for "SABLE" is a direct citation expansion error. The authors must correct this to ensure proper attribution of ideas and accurate historical mapping of the field.
