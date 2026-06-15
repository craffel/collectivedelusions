# Detailed Paper Outline: Parameter-Free Activation Blending (PFAB)

## 1. Title & Abstract
* **Title:** Parameter-Free Activation Blending: Applying Occam's Razor to Heterogeneous Multi-Task Model Merging
* **Author:** Leo Vance (Stanford University)
* **Abstract:** 
  - Context: Multi-task learning via Parameter-Efficient Fine-Tuning (PEFT) and weight merging.
  - Challenge: Heterogeneous mixed-task inference streams suffer from "heterogeneity collapse" when using standard dynamic routing (batch averaging). Previous SOTA (PFSR + MBH) shields against this by dynamically partitioning streams (Micro-Batch Homogenization), but shifts the complexity to heavy serving-infrastructure and introduces sequential execution latency ($O(G)$ for $G$ active tasks).
  - Solution: We propose **Parameter-Free Activation Blending (PFAB)**, an elegant, non-parametric framework that performs sample-wise activation blending of expert outputs directly in feature space on-the-fly.
  - Benefits: Completely prunes systems-level data-orchestration complexity. Executes in a single, parallelized forward pass of the backbone with flat, constant wall-clock latency (independent of task diversity). Introduces zero trainable parameters and zero calibration data.
  - Key Results: Improves Joint Mean accuracy from 73.30% to 76.30% (+3.00% absolute gain) under heterogeneous streams compared to SOTA, while achieving a 1.41x speedup at $G=4$ active tasks and eliminating systems bloat.

## 2. Section 1: Introduction
* Context: The proliferation of specialized large models and task-specific adapters (e.g., LoRA).
* The model merging paradigm: combining weights to resolve multi-task requests without retraining.
* Evolution of routing: From static merging to dynamic test-time routing. Mention previous milestones (SAM, RegCalMerge, PolyMerge, QWS, PFSR + MBH).
* The Systems-Complexity Dilemma: Prior SOTA (PFSR + MBH) requires heavy database partitioning, sequential model compilation, and scatter-gather operations.
* The Minimalist Philosophy: True progress lies in simplifying pipelines, not complicating the serving layer. We ask: *Can we handle heterogeneous streams in a single backbone pass with zero trainable parameters and zero serving-level partitioning?*
* Introducing PFAB: 
  - Non-Parametric Gating Coordinates via Unit-Norm Calibration (UNC).
  - Activation-Space Adapter Blending (ASAB) directly at each layer.
* Summary of Contributions:
  - Propose PFAB, a training-free, parameter-free activation blending framework.
  - Empirically demonstrate that sample-level feature-space blending outperforms batch-level parameter-space merging.
  - Quantify major latency reductions (constant wall-clock latency) and system simplification.

## 3. Section 2: Related Work
* **Model Merging in Parameter Space:** Task arithmetic, TIES-Merging, DARE. Limitations of static compromises.
* **Dynamic Test-Time Routing:** Parametric routers, unregularized/regularized linear/layer-wise routers, and their vulnerability to transductive overfitting/collapse under heterogeneous batches.
* **Mixture of Experts (MoE):** Traditional MoE vs. Parameter-Efficient MoE.
* **Systems Bottlenecks in Model Merging:** SGMV/Punica, Micro-Batch Homogenization. Why complex serving layers are undesirable in low-compute or standard deployment environments.

## 4. Section 3: Methodology
* **Problem Definition:** Heterogeneous multi-task inference over batches of mixed tasks.
* **Step 1: Non-Parametric Gating Coordinates:**
  - Penultimate feature extraction $z_b$.
  - Unit-Norm Calibration (UNC) for features and classification weights.
  - Raw Cosine Similarity Projection (PFSR).
  - Correction of vocabulary bias and Softmax scaling to yield sample-wise coefficients $\alpha_{k, b}$.
* **Step 2: Activation-Space Adapter Blending (ASAB):**
  - Highlighting why standard LoRA merging performs weight blending: $W_{merged} = W_{base} + \sum \alpha_k \Delta W_k$.
  - Shifting to sample-wise activation-space blending: $H_b = X_{base, b} + \sum \alpha_{k,b} X_{k,b}$.
  - Mathematical proof of equivalence for linear layers under single-sample conditions.
  - Discussion of why this avoids MBH: because activation blending is intrinsically sample-bound, whereas weight merging is batch-bound.
* **Complexity & Systems Comparison:** Mathematical analysis of FLOPs, sequential vs. parallel dispatch, and CUDA/Punica dependencies.

## 5. Section 4: Experiments & Results
* **Experimental Environment:** The Isolating Coordinate Sandbox ($L=14$, $D=192$, $K=4$, $C=10$).
* **Main Performance Sweep (Homogeneous Batching):**
  - Comparing Expert Ceiling, Uniform, Linear Router, QWS, L3-Linear, PFSR+MBH, and PFAB.
  - PFAB matches or exceeds prior SOTA, achieving 76.30% Joint Mean.
* **Stream Robustness Audit (Heterogeneous Batching):**
  - Verification of "Heterogeneity Collapse" in unregularized/regularized parametric routers (accuracy dropping to ~30-35%).
  - Showing how PFAB maintains 76.30% Joint Mean accuracy, outperforming PFSR+MBH (73.30%) due to fine-grained sample-wise resolution.
* **Systems-Level Latency and Scalability Profile:**
  - Graphing and tabulating latency as a function of active tasks $G \in \{1, 2, 3, 4\}$.
  - Showing flat latency of PFAB (5.87 ms) vs. linear scaling of PFSR+MBH (up to 8.29 ms, a 1.41x speedup at $G=4$).
* **Ablations and Discussion:** Why Unit-Norm Calibration (UNC) is vital for neutralizing representation-scale drift.

## 6. Section 5: Conclusion
* Recap of contributions.
* Core takeaway: Simple, elegant sample-wise activation-space blending completely bypasses heavy data-engineering solutions in serving infrastructures.
* Call to action for minimalist machine learning design.
