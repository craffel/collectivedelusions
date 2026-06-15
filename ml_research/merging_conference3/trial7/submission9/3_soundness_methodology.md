# Soundness and Methodology

## Technical Soundness of SABLE
The methodology proposed in SABLE (Sample-wise Activation Blending of Low-Rank Experts) is technically sound, mathematically rigorous, and exceptionally well-formulated. The paper translates high-level systems-engineering challenges (batch-level ensembling collapse) into a clean, network-level algebraic solution.

### 1. Mathematical Rigor & Formulation
The paper provides explicit, step-by-step mathematical formulations for:
- **Streaming Heterogeneity:** Modeling the batch-averaging constraint that forces weight-space merging into a static uniform average.
- **Subspace Cosine Projection:** Using normalized cosine similarity scores to represent coordinate alignment, which are invariant to feature norms and bounded within $[-1, 1]$.
- **Temperature-Scaled Softmax Routing:** Showing how routing scores are converted to dynamic blending coefficients with a control temperature $\tau$.
- **Dynamic Activation Blending:** Formulating the sample-wise ensembling of parallel low-rank expert adapters ($A_k B_k$) with the frozen base model projection.
- **Single-Pass Early-Layer Routing:** Bypassing 2-pass execution by propagating activations sequentially and computing routing coefficients on-the-fly, showing how this satisfies latency bounds.

### 2. Scalability and Bounded Complexity
SABLE anticipates realistic serving bottlenecks as the expert pool size $K$ grows:
- **Scalable Top-$M$ Expert Pruning:** Formulates the subset selection $\mathcal{E}_b$ containing only the top $M \ll K$ experts, re-normalizes their blending coefficients, and executes adapters strictly for those active experts. This caps complexity at $O(M)$ instead of $O(K)$, ensuring flatline serving latency.
- **Task-Agnostic Dynamic Head Blending:** Applies Top-$M$ pruning to the final classification heads, avoiding $O(K)$ head evaluation bottlenecks, and elegantly resolves **disjoint output spaces** (e.g., mismatched classification shapes) by falling back to hard head selection ($M=1$) strictly at the output layer while maintaining soft ensembling ($M \ge 2$) in the hidden layers.

### 3. Scientific Honesty and Explicit Boundaries
The authors are exceptionally careful and honest about the theoretical and architectural boundaries of their method:
- **Preservation of Equivalence across Non-linear Boundaries:** They explicitly state that SABLE's exact equivalence to parameter-space merging holds strictly inside linear layers prior to non-linear activations ($X(W_1 + W_2) = X W_1 + X W_2$). They show how SABLE naturally sidesteps representational drift by placing activation blending inside the linear projection interfaces of transformer blocks.
- **The Early-Feature Loss Trade-Off:** They acknowledge that SABLE's default Late Adaptation (Mid-Layer Routing) leaves early layers unadapted, meaning any early-layer specialized features learned during fine-tuning are discarded. They explain that SABLE is structurally limited to expert pools whose adaptation is concentrated in late-stage layers.
- **Theoretical Limitations of Zero-Data Centroids:** They discuss the **Dual-Space Mismatch** (projecting activation features onto parameters) and **Vector Cancellation** (averaged class-specific weights cancelling out), which explains the performance gap compared to using support data.
- **Representational Blurring Paradox and Input-Space Routing Boundaries:** They explain that Single-Pass Early-Routing is highly effective when inputs are starkly separable in raw feature space, but suffers from extreme noise under high-dimensional complex noise, requiring practitioners to navigate a clear trade-off between latency, model paradigm, and routing depth.
- **Storage Scalability and Memory Bandwidth Constraints:** They address the GPU memory bandwidth bottlenecks that occur when loading dozens of inactive adapters from HBM to SRAM. They discuss how Top-$M$ expert pruning bounds the loading footprint and suggest specialized multi-tenant serving frameworks (e.g., vLLM, Punica, S-LoRA) to eliminate CUDA kernel launch overhead in PyTorch.

### 4. Physical Validation Setup
The evaluation methodology is highly rigorous:
- **Analytical Coordinate Sandbox:** A synthetic 14-layer, 192-dimensional sandbox that simulates multi-task streams, enabling detailed layer-by-layer drift tracking and ablation sweeps of routing depth, rank, and hyperparameters.
- **High-Dimensional Foundation Feature Validation with ResNet-18:** Features are extracted using a pre-trained ImageNet ResNet-18 backbone and evaluated on MNIST and FashionMNIST, providing standard-setting correctness on standard image benchmarks.
- **Standard vs. Confounded Streams:** Evaluating both clean streams and challenging domain-confounded blended streams (50-50 overlaid images) under highly rigorous metrics (Top-2 Joint Recall), establishing SABLE's robustness under overlapping task domains.
- **Wall-Clock Serving Profiling:** Real-world hardware benchmarks (NVIDIA A100 GPU and Intel Xeon Platinum CPU) are used to record actual serving latency and peak memory usage, grounding theoretical FLOP savings in wall-clock reality.

## Conclusion on Soundness
The paper's technical claims are fully supported by both elegant mathematical derivations and extensive physical experiments. The authors' high level of scientific transparency regarding limitations, boundaries, and systems-level trade-offs demonstrates outstanding academic integrity and methodology of the highest caliber.
