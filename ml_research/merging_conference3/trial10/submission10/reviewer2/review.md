# Peer Review

## 1. Summary of the Paper
This paper addresses the critical and highly practical challenge of serving multi-task parameter-efficient fine-tuning (PEFT) experts (e.g., LoRA) on dynamic, non-i.i.d. edge-based sequential serving streams. In such environments, dynamic routers must route input tokens or samples through a network layer-by-layer, dynamically blending expert activations or parameters based on query similarity. This serving process is heavily corrupted by two orthogonal sources of representation noise: (1) **Intra-Sample Depth-Wise Noise (Routing Jitter)**, where ensembling coefficients oscillate wildly layer-to-layer during a forward pass, and (2) **Inter-Sample Temporal Noise**, where consecutive queries on a dynamic stream exhibit high-frequency variations. 

Applying Occam's razor to stateful serving, the authors deconstruct prior state-of-the-art stateful ensembling frameworks (such as non-equilibrium chemical reaction kinetics—ChemMerge, or learned state-space models—PAC-Kinetics) and prove that their noise-reduction capabilities stem primarily from local recursive filtering. Driven by this minimalist insight, the authors introduce **2D-STEM** (2D Spatio-Temporal Exponential Moving Average), a training-free, zero-parameter, and analytically simplex-preserving 2D bilinear recursive filter. 

Key elements of 2D-STEM include:
- **2D Bilinear Recurrence:** Integrates depth-wise (spatial) and sequence-wise (temporal) state propagation into a single, unified, single-line recursive update equation.
- **Analytical Simplex Preservation:** Proves that under a simple linear inequality constraint on the momentum coefficients ($\beta_{\text{depth}} + \beta_{\text{temp}, t} \le 1$), the ensembling coefficients are mathematically guaranteed to reside on the probability simplex $\Delta^{K-1}$ at all layers and steps without requiring any runtime projection or Softmax re-normalization operations.
- **Adaptive Temporal Gating (ATG) with Power-Law Sharpening (ATG-PL):** Measures stream homogeneity on-the-fly using representation coordinates at an early, frozen layer. To resolve the upward bias of cosine gating in non-negative spaces under overlapping manifolds, it scales temporal momentum dynamically via a power-law exponent ($\gamma = 3$) to instantly collapse temporal memory during task switches, eliminating transition lag.
- **Coordinate-Prior Spatial Boundary Condition:** Formulates the virtual boundary condition at the early frozen layer using task-coordinates normalized to the simplex, resolving spatial momentum cancellation at the entry layer without introducing task-agnostic accuracy drag.

The authors evaluate 2D-STEM in a 14-layer representation-space simulation environment—the Analytical Coordinate Sandbox (ACS)—across homogeneous and heterogeneous streams under orthogonal and overlapping task manifold layouts. To verify physical generalizability, they conduct an activation-space serving trajectory validation on a pre-trained Vision Transformer (`vit_tiny`) across 12 blocks and 4 overlapping visual domains, supported by formal paired t-tests, extensive hyperparameter sweeps, latency profiling, and hardware compilation roadmaps.

---

## 2. Overall Recommendation and Ratings

- **Overall Recommendation:** **6: Strong Accept**
  - *Rationale:* This is an exceptionally high-quality, technically flawless, and beautifully written paper. It provides an elegant, training-free, zero-parameter, and zero-overhead solution to a pressing industry problem (multi-task expert serving at the edge). The deconstruction of complex prior frameworks under Occam's razor is scientifically refreshing, and the proposed 2D-STEM filter is mathematically rigorous, highly deployable, and outperforms complex baselines. The paper is outstanding in its evaluation, completeness, and reproducibility.
- **Soundness:** **Excellent**
  - *Rationale:* The mathematical formulations are clean, the inductive proof of Theorem 1 is correct, and the experimental evaluation is highly thorough, incorporating controlled sandbox simulations, physical pre-trained ViT validation, paired t-tests showing extreme statistical significance ($p < 10^{-7}$), and comprehensive hyperparameter/ablation sweeps.
- **Presentation:** **Excellent**
  - *Rationale:* The paper is written with outstanding clarity, professional precision, and a highly cohesive logical flow. Visual trajectory plots are exceptionally clean and designed for grayscale readability. The inclusion of a complete, self-contained PyTorch implementation in the appendix makes the method immediately reproducible and ready for deployment.
- **Significance:** **Excellent**
  - *Rationale:* The proposed method addresses a major systems bottleneck in dynamic edge serving. By replacing computationally prohibitive biochemical ODE solvers or learned state-space models with a single-line arithmetic recurrence, 2D-STEM delivers a **49.5% reduction in serving latency** on standard backends. Furthermore, by suppressing routing oscillations by up to $5.23\times$, it stabilizes hardware activations, prevents cache thrashing and DRAM transfers, and drastically improves overall systems-level and energy efficiency.
- **Originality:** **Excellent**
  - *Rationale:* The creative combination of spatial-temporal EMA for deep model-serving, the analytical simplex preservation proof under a simple linear constraint, the formulation of Power-Law Gating (ATG-PL) to resolve the mathematical bias of non-negative cosine coordinates, and the Coordinate-Prior boundary condition represent highly original and impactful contributions.

---

## 3. Strengths (Practitioner and Systems Perspective)

1. **Outstanding Practical Utility and Ease of Deployment:**
   - **Zero Parameter and Training Overhead:** Unlike learned state-space models (such as PAC-Kinetics) that require offline training and online backpropagation, 2D-STEM is completely training-free and introduces zero additional parameter storage compared to stateless serving, requiring only a microscopic active runtime state (240 bytes) to track temporal history.
   - **No Runtime Projections:** The analytical proof of simplex preservation under a simple inequality constraint is a massive system advantage. It completely eliminates the need for expensive Softmax re-normalization or Euclidean projection steps, which typically act as major execution bottlenecks in deep learning pipelines.
   - **Substantial Latency Reductions:** Latency profiling on standard CPU backends proves that 2D-STEM executes in only $1,436.20\,\mu\text{s}$ per step, representing a **49.5% reduction in serving-time execution latency** compared to the continuous-time ChemMerge (Dynamic ODE) baseline ($2,845.48\,\mu\text{s}$), with negligible overhead relative to stateless SABLE ($1,156.31\,\mu\text{s}$).
   - **Hardware/Compiler Integration:** The zero-parameter, projection-free, and branch-free nature of 2D-STEM makes it highly compatible with modern inference compilation toolchains (such as ONNX, TensorRT, and vLLM/Punica). The entire routing and ensembling graph can be compiled into a single, fused CUDA/NPU kernel, eliminating GPU roundtrip latency.
   - **Extreme Data Efficiency:** The low-pass filtering properties of 2D-STEM make it highly robust to centroid calibration noise, allowing the system to maintain near-oracle ensembling performance even when the offline calibration set is reduced to an ultra-scarce regime ($N_{\text{cal}} = 5$ samples per task), with only a $0.11\%$ accuracy drop.

2. **Elegant Scientific and Mathematical Design:**
   - **Minimalist Deconstruction:** The paper applies Occam's razor beautifully, demonstrating that the noise-reduction of highly parameterized dynamical merging models (like ODEs or complex learned state spaces) is driven primarily by the mathematics of local recursive filtering, allowing them to prune unnecessary biochemical and learning complexity.
   - **Power-Law ATG-PL Gating:** The authors identify a fundamental mathematical bias of linear similarity gating under overlapping manifolds (where non-negative Softmax coordinates never collapse to zero during transitions, creating transition lag). The cubic sharpening exponent ($\gamma = 3$) elegantly squashes this transition similarity noise floor, allowing immediate transition responsiveness while keeping strong temporal smoothing active on stable blocks.
   - **Decoupled Transition Detection:** By measuring stream similarity ($Sim_t$) at an early, frozen layer, 2D-STEM decouples transition detection from the downstream noise corrupting deeper layer representations. This successfully avoids the catastrophic failure mode of the ChemMerge (Dynamic ODE) baseline, which misinterprets deep representation noise as task transitions, spiking its temperature, collapsing its temporal memory, and elevating its homogeneous routing jitter to $0.0283$ (worse than stateless SABLE's $0.0187$). 2D-STEM preserves highly stable smoothing, achieving a homogeneous jitter of $0.0068$ (a $4.16\times$ noise reduction).
   - **Coordinate-Prior Boundary Condition:** The formulation of the virtual boundary state at layer $L_{\text{frozen}}$ using task coordinates normalized to the simplex is a highly elegant boundary condition. It successfully resolves first-layer spatial momentum cancellation without introducing the task-agnostic accuracy drag characteristic of uniform boundaries.

3. **Thorough Scaling and Robustness Extensions:**
   - **Top-$k$ Coordinate Masking:** Mathematically guarantees $O(1)$ scaling complexity under extremely large expert pools ($K \ge 50$), sparsifying the coordinate space on-the-fly and forcing transition similarity to collapse exactly to zero.
   - **MLP Coordinate Mapper:** A low-overhead 2-layer MLP coordinate-prior mapper extension that trains in under 3 seconds on the tiny calibration split, resolving potential early-layer representation overlaps under extremely fine-grained domain boundaries.
   - **OOD Fallback Policy:** Explicit fallback policies (such as Uniform fallback or Temporal State Bypass) handle out-of-distribution queries and severe covariate shifts safely, preventing OOD noise from contaminating sequence history.

---

## 4. Weaknesses and Areas for Improvement

While the paper is exceptionally strong, the following constructive suggestions would further elevate its scientific and industrial impact:

1. **Physical Edge-Hardware Profiling:**
   - *Detail:* Although the CPU execution latency profiling ($1,436.20\,\mu\text{s}$ per step) is highly informative, measuring physical execution latencies, DRAM bandwidth usage, and L1/L2 cache miss rates on actual resource-constrained edge hardware (e.g., NVIDIA Jetson Nano/Orin, Raspberry Pi, or Google Edge TPU) would further validate the systems-level utility claims. Stabilizing routing weights reduces the need to load/unload specialized LoRA weights from DRAM to local cache, which is the primary driver of edge energy consumption and latency. Quantifying actual DRAM-to-cache data transfer reductions would provide a powerful concrete systems-level metric.
2. **Evaluation Modality Generalization:**
   - *Detail:* The empirical evaluations are currently focused on image classification and visual domains on a Vision Transformer backbone. Expanding the validation of 2D bilinear recursive filtering to natural language processing (NLP) sequence-level tasks or autoregressive token-level Mixture-of-Experts (MoE) would demonstrate the generalizability of 2D-STEM across other modalities. The authors briefly discuss this promising direction in the conclusion and appendix; providing even a preliminary proof-of-concept on text data would strengthen the paper's significance.

---

## 5. Technical and Soundness Verification

- **Mathematical Correctness:** The mathematical formulation of 2D-STEM in Equation 8 is solid, and Theorem 1's inductive proof of analytical simplex preservation is correct, utilizing the convex combination property of the probability simplex. 
- **Experimental Completeness:** The authors evaluate 2D-STEM across two stream configurations (Homogeneous and Heterogeneous) and two manifold configurations (Orthogonal and Overlapping) in ACS. The physical validation on a pre-trained Vision Transformer with 4 overlapping visual domains matches the synthetic findings perfectly, confirming that 2D-STEM reduces routing jitter by over $5.2\times$ while matching stateless ensembling accuracy and maintaining zero transition lag.
- **Statistical Rigor:** The results are averaged over 5 independent random evaluation seeds, including standard deviations. The relative paired t-tests in Table 3 verify that 2D-STEM's improvements in accuracy and routing jitter are highly statistically significant ($p < 0.05$, and in many cases $p < 10^{-7}$).
- **Reproducibility:** Excellent. All hyperparameters are explicitly listed, and a self-contained PyTorch implementation of the 2D-STEM router is provided in Listing 1, making it trivial for researchers and practitioners to reproduce the results and deploy the method.

---

## 6. Context and Presentation

- **Writing Quality:** The paper is exceptionally well-written, with high technical density, professional tone, and clear conceptual explanations. It is structured logically, starting with a deconstruction of stateful serving, presenting the mathematical formulations and proofs, validating them empirically, and discussing physical compiler integration.
- **Figures and Tables:** Table 1 and Table 2 (ACS results) and Table 5 (ViT results) are clearly laid out. The visual trajectories in Figure 1 are highly informative and use both distinct colors and highly contrasting line styles (solid, dashed, dotted) to guarantee complete readability under grayscale compilation.
- **Contextualization:** The paper does a superb job of positioning itself in the context of prior PEFT, MoE, and model-merging literature, clearly detailing how it differs from SABLE, Momentum-Merge, PAC-Kinetics, and ChemMerge.

---

## 7. Conclusion

This is an outstanding paper that perfectly embodies the principle of Occam's razor. By stripping away unnecessary biochemical and learning-theoretic complexity, the authors deliver a mathematically elegant, training-free, and zero-parameter 2D bilinear filter (2D-STEM) that solves a pressing industrial edge-serving challenge. The paper is exceptionally strong in its mathematical foundations, systems-level utility, and empirical rigor. I strongly recommend a **Strong Accept (6)**, and urge practitioners in the field to immediately adopt and compile this framework for stable, highly efficient edge-based model serving.
