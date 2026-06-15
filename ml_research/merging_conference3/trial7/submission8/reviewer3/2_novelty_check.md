# Evaluation Component 2: Novelty Check and Delta from Prior Work

## 1. Key Novel Aspects of the Submission
1. **Confidence-Gated Hybrid Routing (CGHR):** A novel hybrid routing architecture that combines a parametric router with a non-parametric fallback (Parameter-Free Subspace Routing, PFSR). By dynamically gating the pathways based on the prediction confidence of the parametric model, the system maintains high performance under extreme data scarcity and gains expressive strength as data scales.
2. **Identification of Heterogeneity Collapse:** The paper introduces and formalizes the phenomenon of "heterogeneity collapse," which occurs when dynamic routers process mixed-task batches. This is a critical and highly practical contribution, as real-world inference streams are rarely homogeneous.
3. **Micro-Batch Homogenization (MBH):** A library-level, hardware-agnostic scheduling and partitioning technique that dynamically groups mixed-task batches into homogeneous sub-batches on the fly. This prevents logit smoothing and ensures task-specific expert blending.
4. **Systems-Level Optimizations for Dynamic Merging:** The paper proposes and evaluates several systems-level innovations:
   * **Homogeneity Bypass:** Bypasses partitioning for single-sample or homogeneous streams to eliminate scheduling overhead.
   * **Fusion Weight Caching:** Discretizes ensembling coefficients to allow cache hits ($98.2\%$ at $0.10$ step size) and reuse pre-fused weights.
   * **Warp Batch Padding / Segmented-BGEMM Triton Outlines:** Systems-level designs to maximize occupancy and minimize warp divergence on parallel GPUs.

---

## 2. 'Delta' from Prior Work
The paper clearly positions itself relative to three lines of research:
* **Vs. Static Model Merging (Task Arithmetic, TIES-Merging, DARE):** Static methods compute a single, fixed set of merging coefficients during offline fusion. The delta is that CGHR performs dynamic, input-dependent blending on a sample-by-sample basis at inference time.
* **Vs. Dynamic Model Merging (VR-Router, TSAR):** Existing dynamic routers are either purely parametric (vulnerable to small-data overfitting) or purely non-parametric (unable to scale with more calibration data). The delta is CGHR's confidence-gated hybrid pathway, which bridges these regimes.
* **Vs. Multi-Tenant Serving Engines (S-LoRA, Punica):** Punica and S-LoRA require custom, low-level CUDA compilers and specific hardware to execute multiple LoRAs in parallel. The delta is that MBH is a hardware-agnostic, library-level scheduling design pattern that runs out-of-the-box on standard deep learning libraries (e.g., PyTorch).

---

## 3. Characterization of Novelty
The novelty of this paper can be characterized as **significant in conceptual design and systems integration, but limited in empirical verification**. 

* **Conceptual/Systems Novelty (Significant):** The hybrid routing framework (CGHR), the identification of heterogeneity collapse, and the scheduling-level solution (MBH) are very solid and represent a highly complete, end-to-end design. The paper goes far beyond typical "toy" algorithmic proposals by thoroughly analyzing PCIe bus contention, CUDA kernel warp divergence, caching memory footprints, and Triton Segmented-BGEMM execution.
* **Empirical Novelty (Limited):** The entire quantitative verification is conducted within a 1-layer synthetic sandbox using simulated noise instead of real-world datasets (MNIST, Fashion-MNIST, CIFAR-10, SVHN are simulated with Gaussian noise, not actually run). The lack of empirical validation on real deep models (such as modern Transformer-based LLMs or Vision Transformers) or real-world datasets limits the practical novelty, as the actual behavior of these methods under non-orthogonal, non-linear representation spaces remains unverified in practice.
