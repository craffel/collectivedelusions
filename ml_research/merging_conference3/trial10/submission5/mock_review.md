# Official Peer Review

**Paper Title:** Unitary Geodesic Routing: Softmax-Free Simplex Projection and Geodesic Flow for Stateful Test-Time Model Ensembling

---

## 1. Summary of the Paper

This paper introduces **Unitary Geodesic Routing (UGR)**, a novel geometric framework designed for stateful test-time model ensembling across heterogeneous, non-stationary task streams. 

### Core Problem
Test-time model ensembling blends specialized expert weights (e.g., PEFT adapters like LoRA) on-the-fly for incoming sequential queries. Under non-stationary streams, stateful routing is preferred to smooth out representation noise while remaining responsive to task transitions. However, existing stateful routers (e.g., *Momentum-Merge*, *ChemMerge*) perform updates in unconstrained flat Euclidean spaces and project onto the probability simplex post-hoc via Softmax normalization. The authors argue that this unconstrained-to-constrained mismatch introduces:
1. **Representational Lag (Hysteresis):** Accumulation of flat-space inertia under task transitions, requiring unconstrained states to "unwind" before the Softmax output reflects a switch.
2. **Geometric Distortion:** Activation scale and norm mismatches when Euclidean interpolations "wiggle off" the natural probability manifold, leading to feature scale compression.
3. **High-Frequency Jitter:** Rapid oscillations when flat routers attempt to increase agility using small inertia coefficients or high Softmax temperatures.

### Proposed Solution
UGR models the ensembling state directly on the curved $(K-1)$-dimensional unit hypersphere $\mathbb{S}^{K-1} \subset \mathbb{R}^K$.
- **Born Mapping:** Leverages the square-root homeomorphism from Information Geometry ($\alpha_k = s_k^2$) to map spherical coordinates to the probability simplex natively. This completely eliminates post-hoc Softmax normalization for the state representation.
- **Closed-Form Geodesic Updates:** Derives a Rodrigues-like spherical linear interpolation (Slerp) along the shortest great-circle path, bypassing costly matrix exponentials and virtual-time numerical ODE solvers.
- **Torque-Driven Adaptive Agility:** Dynamically scales the angular step size proportionally to the angular distance (representational torque) between the current state and incoming signals, resolving the stability-plasticity trade-off.
- **Spatial-Temporal Geodesic Coupling:** Propagates the ensembling state from the final layer of the previous query to the initial adapted layer of the current query to maintain cross-query temporal coherence.

### Evaluation and Key Results
- **Synthetic Analytical Coordinate Sandbox (ICS):** Across 10 independent synchronized seeds, UGR achieves a Joint Mean Accuracy of **75.08%** (outperforming ChemMerge by **+5.43%**) and slashes layer-to-layer routing jitter ($L \ge 5$) by **2.10$\times$**.
- **Real-World Multi-Task Text Classification (20newsgroups):** Across 5 synchronized seeds, UGR delivers **92.25%** Joint Mean Accuracy (outperforming ChemMerge by **+21.60%** and Coupled Momentum-Merge by **+4.13%**) while reducing routing jitter by **1.63$\times$**.
- **Timing & Throughput:** Standard UGR adds less than 0.07 ms of latency. The fully Softmax-free target variant achieves **0.436 ms/query** and **2295.3 QPS** on an Intel Xeon CPU.

---

## 2. Main Strengths

### Mathematical Elegance & Theoretical Rigor
The mathematical foundation is exceptionally solid and creative:
- The paper beautifully connects Born's rule from quantum mechanics with the standard information-geometric square-root mapping (Hellinger/Bhattacharyya mapping) on the probability simplex.
- The authors provide a rigorous proof of **Positive Orthant Persistence** (Appendix A.2) to show that the routing state remains strictly within $\mathbb{S}^{K-1}_+$, eliminating antipodal path ambiguities during great-circle geodesic rotations.
- The control-theoretic framing of Torque-Driven Agility as a **first-order non-linear dynamical system with non-linear damping** is elegant and explains why the trajectory avoids overshoot or kinetic momentum accumulation.

### Algorithmic Efficiency
By deriving a closed-form, Rodrigues-like formulation for Slerp, the framework achieves linear complexity $\mathcal{O}(K)$ with respect to the expert pool. This completely bypasses expensive matrix operations or virtual-time numerical integration steps (e.g., Euler/Heun solvers in ChemMerge), resolving a major serving latency bottleneck.

### Outstanding Scientific Hygiene and Baseline Auditing
The paper displays an exemplary level of empirical integrity:
- **Baseline Audit:** The authors conducted a code-level audit of Momentum-Merge and discovered a biased initialization discrepancy (where Momentum-Merge Reset had its boundary prior overwritten with layer 4's target, artificially suppressing transition shock). They corrected this discrepancy and re-evaluated all methods fairly.
- **Decomposed Jitter Analysis:** To address why overall routing jitter is higher under highly randomized streams, they decomposed jitter into *Intra-Task Jitter* (stability) and *Inter-Task Jitter* (agility) on a block-structured stream. This analysis elegantly showed that UGR is **5.5$\times$ more stable** than Momentum-Merge inside identical task sequences, while its overall jitter is driven by purposeful, correct rotations at task boundaries.

---

## 3. Key Weaknesses & Areas for Improvement

While the paper is technically solid and exceptionally well-written, a rigorous critique reveals several key weaknesses and conceptual overclaims that must be addressed to maximize the paper's impact:

### 1. The Scale Gap of the "Real-World" Benchmark
The authors position UGR as a highly scalable solution for serving-time ensembling of parameter-efficient fine-tuning (PEFT) expert adapters (e.g., LoRA) in Large Language Models (LLMs). However, there is a massive scale gap between this ultimate vision and the paper's empirical evaluation:
- The real-world evaluation is performed on the classical `20newsgroups` dataset, which is a shallow document classification task.
- The text representation is built on a simple TF-IDF vectorizer of 1024 features, and the "expert adapters" are represented by shallow, 2-layer Multi-Layer Perceptrons (MLPs) with 128 hidden units.
- A static document-level MLP is a poor proxy for token-by-token autoregressive decoding sequences with pre-trained transformers (such as LLaMA, Mistral, or BERT) where routing actually takes place. The lack of evaluation on actual deep pre-trained models or standard PEFT benchmarks (like GLUE subcategories, GSM8K, or MATH) under continuous streaming limits the practical significance of the empirical results.

### 2. The "Softmax-Free" Overclaim
The paper heavily emphasizes the "completely Softmax-free" nature of the proposed router (even in the abstract and title). However, a closer look reveals that standard UGR is not completely Softmax-free:
- To construct the bottom-up target vector $\mathbf{e}_t^{(l)}$, standard UGR relies on a localized Softmax over cosine similarities (Equation 10).
- While the authors propose a fully Softmax-free target alternative (Equation 11 using ReLU and $L_1$-normalization) as an ablation, this variant incurs a noticeable performance degradation on both benchmarks: Joint Accuracy drops from **75.08% to 72.73%** in the synthetic sandbox, and from **92.25% to 87.40%** on the real-world text classification task. 
- This performance drop indicates that standard UGR's peak performance is still partially dependent on a Softmax operation, making the "completely Softmax-free" claim a slight marketing overstretch. The authors should moderate this language to reflect that the *state representation*, *geodesic updates*, and *simplex projections* are Softmax-free, while target construction still benefits from Softmax gating.

### 3. Layer Mismatch in Spatial-Temporal Geodesic Coupling
In Section 3.6, the Spatial-Temporal Geodesic Coupling initializes the *first adapted layer* of the current query using the *final layer's* state of the previous query:
$$\mathbf{s}_t^{(L_{\text{frozen}})} = \mathbf{s}_{t-1}^{(L)}$$
In deep networks, different layers capture different levels of feature abstraction (lower layers capture syntactic/local features, while deeper layers capture highly specialized semantic features). 
- Carrying over a high-level semantic routing decision ($\mathbf{s}_{t-1}^{(L)}$) to initialize a low-level routing state ($\mathbf{s}_t^{(L_{\text{frozen}})}$) introduces a cross-layer representational mismatch.
- The authors do not provide any theoretical or empirical justification for why they choose this cross-layer boundary coupling instead of a standard layer-wise coupling ($\mathbf{s}_t^{(l)} = \mathbf{s}_{t-1}^{(l)}$) or propagating the state within corresponding layers. An ablation or discussion comparing layer-wise coupling vs. cross-layer coupling is highly necessary to clarify this choice.

---

## 4. Minor Suggestions & Terminological Nitpicks

### 1. Terminological Accuracy of "Unitary"
The term "Unitary" in Unitary Geodesic Routing represents a minor abuse of mathematical terminology. In linear algebra and quantum mechanics, a "unitary operator" or "unitary matrix" satisfies $U^* U = I$ and preserves inner products. The authors' routing state $\mathbf{s}_t^{(l)}$ is a "unit vector" or "unit-norm vector" residing on the unit hypersphere, not an operator. While the authors explicitly clarify this in Section 3.2, using the word "Unitary" to mean "unit-norm vector" is non-standard. The authors should consider changing the name to "Spherical Geodesic Routing" or "Unit-Norm Geodesic Routing", or at least acknowledge this naming choice as a stylistic metaphor rather than a literal mathematical definition.

### 2. Marginal Gains over Stateless Baselines in the Sandbox
In the synthetic Analytical Coordinate Sandbox (ICS), standard SABLE (Stateless, $\tau=0.005$) achieves an accuracy of **74.74%**, which is virtually identical to standard UGR's **75.08%** (an absolute gain of only **+0.34%**). However, SABLE is completely stateless and requires zero history tracking or spatial-temporal boundary coupling. In contrast, UGR's spatial-temporal coupling introduces a significant boundary shock at layer 4 (with a Jitter $L \ge 4$ of **1068.25** compared to SABLE's **889.52**). Under highly dynamic, fast-switching streams, the marginal +0.34% accuracy gain of UGR may not justify the added complexity of state tracking. The authors should discuss this complexity-performance trade-off more transparently.

---

## 5. Ratings and Recommendation

* **Soundness:** **Good (3/4)** — The mathematical derivations, proofs, and control-theoretic formulations are correct and highly rigorous. However, the cross-layer boundary initialization in Spatial-Temporal Coupling lacks theoretical justification or empirical comparison, and standard UGR's success still partially depends on Softmax.
* **Presentation:** **Excellent (4/4)** — Clear, precise, and highly scholarly writing. The narrative flow is logical, and the figures are of outstanding quality.
* **Significance:** **Good (3/4)** — The paper addresses an important, highly timely problem. However, the practical significance is heavily bottlenecked by the small scale of the "real-world" benchmark (shallow 2-layer MLPs on 20newsgroups TF-IDF features).
* **Originality:** **Excellent (4/4)** — Introduces a fundamentally new geometric paradigm for stateful test-time routing, rejecting flat-space assumptions in favor of curved-manifold geodesic flow.

### Overall Recommendation
**4: Weak Accept**

This is an exceptionally creative, mathematically elegant, and well-written paper that introduces a powerful geometric perspective to test-time model ensembling. The baseline auditing and decomposed jitter analysis represent exemplary scientific hygiene. However, the practical significance of the empirical results is limited by the small scale of the MLP/TF-IDF evaluation, the "completely Softmax-free" claim is slightly overstated, and the cross-layer spatial-temporal coupling lacks rigorous justification. If the authors can address the scale gap (e.g., by running a pilot experiment on actual pre-trained transformers with LoRA adapters) and moderate their claims, this paper has the potential to become a strong contribution.
