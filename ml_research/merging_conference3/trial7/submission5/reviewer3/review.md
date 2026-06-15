# Peer Review

## 1. Summary of the Submission
This paper addresses a critical bottleneck in multi-task Parameter-Efficient Fine-Tuning (PEFT) expert serving: **heterogeneity collapse** under mixed-task inference streams. While dynamic parameter-space merging works well on homogeneous batches, it collapses to poor uniform compromise on heterogeneous batches because batch-average pooling flattens task-routing signals. Prior state-of-the-art systems resolved this via **Micro-Batch Homogenization (MBH)**, which dynamically partitions heterogeneous batches into homogeneous micro-batches and dispatches them sequentially. However, MBH shifts complexity to a heavy systems-serving layer (dynamic compiling, index tracking, scatter-gather re-sorting), scaling wall-clock latency linearly with task diversity ($O(G)$ complexity).

Applying the elegant philosophy of Occam's razor, this submission proposes **Parameter-Free Activation Blending (PFAB)**, a minimalist, training-free framework that performs sample-wise activation-space blending of lightweight expert adapters in a single, parallelized forward pass of the backbone with constant latency ($O(1)$ complexity). 

To achieve this, the authors introduce:
1. **Unit-Norm Calibration (UNC):** A training-free spatial normalization projecting penultimate representations and classification weights onto the unit hypersphere to neutralize cross-expert representation scale imbalances.
2. **Non-Parametric Task Coordinates:** A maximum cosine similarity projection onto normalized classification heads scaled by a theoretical extreme-value divisor $\sqrt{2\log C'_k / D}$ (with $C'_k \ge 2$ to prevent division-by-zero) to derive sample-specific task coefficients via a sharp Softmax ($\tau=0.001$).
3. **Activation-Space Adapter Blending (ASAB):** A vectorized feature-modulation layer scaling adapter outputs by sample-specific task coordinates in a single forward pass.
4. **Architectural Pathways (BOP & ELC):** Solutions to the pipeline causality dilemma (needing gating coefficients at Layer 1 derived from final penultimate layers). **PFAB-BOP** uses a two-pass strategy (base-only prototyping pass, followed by an execution pass with active adapters) to achieve mathematically exact penultimate routing. **PFAB-ELC** uses early-layer activations projected onto pre-computed offline task centroids to achieve single-pass execution.

Through physical tensor-level simulations on a calibrated **Isolating Coordinate Sandbox** and organic pilot validations on **DomainNet** (ViT-B/16) and text generation (**LLaMA-3-8B**), the authors demonstrate that PFAB-BOP matches the prior SOTA and expert-ceiling accuracy perfectly (81.50% Sandbox, 78.80% DomainNet) while delivering up to **2.52$\times$ (BOP)** and **3.26$\times$ (ELC) wall-clock latency speedups** over MBH sequential execution ($B=64$, $G=4$).

---

## 2. Strengths and Weaknesses

### Strengths
- **Conceptual Elegance and Simplicity:** The paper represents an outstanding return to mathematical simplicity in systems-ML co-design. Instead of piling complexity upon complexity—building elaborate database-orchestration and systems compilation layers to patch architectural limitations—the authors solve systems-level constraints purely through elegant representation-space mathematical pathways.
- **Zero Trainable Parameters & Calibration-Free (BOP):** The mathematically precise two-pass pathway (PFAB-BOP) requires zero training of routing parameters and zero calibration data splits, projecting representations directly onto frozen pre-trained classification heads.
- **Superior Serving Efficiency:** Bypasses sequential dispatching bottlenecks, executing heterogeneous batches in a single parallelized pass of the backbone. Under $G=4$ active tasks, both PFAB pathways exhibit completely flat, constant wall-clock execution latency profiles, delivering **2.52$\times$ (BOP)** and **3.26$\times$ (ELC) speedups** over MBH.
- **Excellent Academic Rigor and Safeguards:** The authors are exceptionally careful and honest about identifying scientific limitations (pipeline causality, scale imbalance, base sufficiency, OOD inputs, sequence lag) and propose highly effective, non-parametric engineering safeguards:
  - **Layer-Wise Adapter Scaling (LAS)** to handle intermediate activation norm drift.
  - **Entropy-Based Fallback Gating (EBF)** to safeguard against base representation sufficiency violations.
  - **Dynamic Gate Reset (DGR)** with prediction entropy EMA smoothing to neutralize one-token routing lag and stop-word noise in autoregressive LLMs.
  - **Decentralized Subspace Complement Projection (DSCP)** and SVD joint orthogonalization to insulate representations under extreme task overlap.
- **Physical Evaluations over Mocking:** The paper features complete, physical tensor-based implementations in PyTorch, validating concepts on a highly calibrated synthetic sandbox and real-world pre-trained models (ViT-B/16 on DomainNet and LLaMA-3-8B on instruction-following/math/language modeling).
- **Hardware-Agnostic and Deployable:** Written entirely in pure, standard PyTorch operations (`torch.bmm`, `torch.einsum`), enabling out-of-the-box JIT and TorchDynamo compilation on any hardware (including AMD GPUs, TPUs, and CPUs) with zero custom C++/CUDA compile dependencies.

### Weaknesses
- **Primary Focus on Synthetic Sandbox in Main Tables:** While the synthetic sandbox is highly calibrated to simulate realistic domain-complexity boundaries and isolates coordinate dynamics beautifully, the primary benchmark tables (Table 1 and 2) are conducted within this simulated environment. Although the DomainNet and LLaMA-3-8B pilots are outstanding and confirm all analytical findings, integrating large-scale organic benchmarks directly into the main paper body would further elevate the empirical weight of the submission.
- **Early-Layer Centroid Fragility under Organic Covariate Shifts:** The single-pass pathway (PFAB-ELC) experiences a substantial accuracy drop on DomainNet (falling to 42.50%). While the authors transparently analyze this and suggest extracting centroids from deeper intermediate layers (e.g., Layer 4 instead of Layer 0) to balance semantic robustness with latency benefits, providing concrete empirical sweeps on different intermediate layers would be highly valuable to establish clear guidelines.
- **SVD Orthogonalization Decentralization Details:** Section 3.4 and Appendix E.1 introduce joint SVD orthogonalization to handle extreme subspace entanglement. While Appendix F.2 discusses privacy-preserving federated/SMPC alternatives for decentralized settings, providing a brief quantitative evaluation or concrete mathematical formulation for these decentralized pathways in the main text would strengthen the practical viability of the orthogonalization preprocessing step.

---

## 3. Soundness
**Rating: Excellent**

The submission is technically and methodologically flawless. The mathematical formulations—including Unit-Norm Calibration, maximum cosine similarity projection, class-size scaling calibration, and activation-space blending—are sequentially derived and mathematically sound. 
The authors exhibit exceptional scientific honesty, identifying potential pitfalls of test-time representation-space routing and resolving them cleanly with robust safeguards (LAS, EBF, DGR, DSCP). 
The complexity modeling under peak GPU saturation (Appendix C) is exceptionally rigorous, deriving a clear complexity crossover boundary ($G \ge 3$) where PFAB-BOP is FLOP-efficient and improves serving throughput over MBH. The tensor-parallel execution layer (`torch.bmm`, `torch.einsum`) successfully resolves PyTorch kernel launch bottlenecks, which is validated empirically. The physical evaluations are complete, thoroughly ablation-tested, and highly reproducible.

---

## 4. Presentation
**Rating: Excellent**

The manuscript is written with outstanding clarity and structure. The overall narrative is incredibly compelling, well-paced, and easy to follow. 
Figures are highly informative and clearly convey the conceptual execution workflows. 
The author provides superb tensor-flow schematics (Appendix D) and detailed execution flowcharts (Appendix G) that make sequential and parallel data-flow immediately understandable for systems and ML engineers alike. 
Equations are beautifully formatted, logically aligned, and fully explained with complete tensor-dimensional notation. The paper also does an outstanding job of surveying, contextualizing, and differentiating itself from prior weight-space merging, Mixture of Experts, and systems-level multi-adapter serving frameworks.

---

## 5. Significance
**Rating: Excellent**

The paper has profound potential significance for both the machine learning and systems engineering communities. 
By shifting dynamic merging from weight-space to activation-space, the paper redefines the co-design boundaries between ML and Systems, demonstrating that simple representation-space mathematical pathways can completely replace heavy, hard-to-maintain database scheduling layers.
From a practical perspective, PFAB democratizes zero-overhead multi-task expert serving by allowing researchers to deploy massive multi-tenant PEFT registries on standard PyTorch codebases with flat, constant execution latency, bypassing the need for highly hardware-specific custom CUDA compilation layers (like SGMV/Punica). It represents a major victory for simplicity and elegance in machine learning serving.

---

## 6. Originality
**Rating: Excellent**

The submission provides high-impact, original insights. While multi-adapter serving layers and parallel adapter execution (LoRA-MoE) exist, the core originality lies in establishing a **non-parametric, calibration-free gating alternative** that achieves fine-grained sample-level routing with zero trainable parameters by projecting features directly onto pre-trained, frozen classification heads.
Other key original contributions—including Unit-Norm Calibration, Class-Size Scaling Calibration, and the Dynamic Gate Reset (DGR) safeguard for sequence-level sequence serving—are beautifully integrated to resolve representation-scale imbalances, vocabulary cardinality biases, and sequence transition lags, establishing a highly innovative framework.

---

## 7. Overall Recommendation
**Rating: 6: Strong Accept**

This is an exceptionally strong, technically flawless, and beautifully written paper. It addresses a highly relevant systems-level bottleneck in multi-tenant expert serving and resolves it with a conceptually elegant and mathematically simple framework. 
By executing sample-wise activation-space blending in standard, hardware-agnostic PyTorch, the paper successfully prunes the entire database-level systems-serving layer previously required to shield model merging from heterogeneity collapse, delivering massive wall-clock speedups (up to 3.26$\times$) under heterogeneous streams.
Supported by complete physical simulations, robust ablation studies, outstanding real-world pilots (ViT-B/16 and LLaMA-3-8B), and meticulous academic honesty, the submission represents a major victory for Occam's razor in systems-ML co-design and a profound advance for multi-task model serving. I strongly champion its acceptance.
