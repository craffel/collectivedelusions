# Peer Review of GraviMerge

## 1. Summary of the Paper
The paper introduces **GraviMerge**, a dynamic, test-time model ensembling framework designed for multi-task edge serving of parameter-efficient expert adapters (e.g., LoRA). The paper aims to resolve the "accuracy-stability dilemma," where layer-to-layer ensembling routing weights fluctuate rapidly, causing representational instability and disrupting activation propagation. 

To address this, GraviMerge models deep representation routing as a continuous, second-order physical dynamical system on a spherical manifold. It maps intermediate activations to a virtual spacecraft coordinate probe on a unit hypersphere ($\mathbb{S}^{D-1}$), and pre-trained expert centroids to high-mass stationary stellar attractors. It introduces:
1. **Arrhenius Mass Activation (AMA):** An early, zero-shot alignment mechanism that dynamically sets expert masses at test-time based on similarity.
2. **Geodesic Trajectory Integration (GTI):** A stateful trajectory tracking scheme that projects acceleration and velocity onto tangent planes, integrates the trajectory via the spherical Exponential Map, and parallel transports the velocity state vector across layer steps to preserve geometric invariants under second-order physical inertia and viscous drag.
3. **Gravitational Influence Blending (GIB):** Translating localized gravitational forces into continuous, differentiable ensembling weights at each layer.

The method is evaluated on a synthetic "Projected Digit Representation Space (RDS) Proxy" benchmark derived from scikit-learn's `load_digits` dataset, where it is reported to achieve a serving accuracy of 88.69% and a 6.01$\times$ reduction in routing jitter compared to a first-order kinetics baseline (ChemMerge).

---

## 2. Strengths
- **Geometric and Mathematical Rigor:** The mathematical formulation is highly detailed and geometrically consistent. The projection of acceleration and velocity onto local tangent spaces, the exact closed-form parallel transport, and the geodesic integration via the Exponential Map are technically sound and prevent coordinate drift under finite-precision floating-point arithmetic.
- **Thorough Appendix and Ablations:** The authors conduct comprehensive parameter sweeps investigating key physical hyperparameters (the gravitational constant $G$, viscous drag $\gamma_{\text{drag}}$, and softening factor $\epsilon$), providing a clear picture of the physical regimes and phase transitions.
- **Polished Presentation:** The writing is professional, the equations are formally presented, and the qualitative visualizations (weight trajectories and accuracy-stability Pareto frontier) are clean and clear.

---

## 3. Weaknesses

### A. Extreme Over-Engineering and Conceptual Complexity
The core design of GraviMerge is characterized by a severe lack of simplicity and excessive complexity. Introducing a second-order classical mechanics simulation—featuring virtual spacecraft, stationary celestial attractors, viscous medium drag, softened gravitational constants, local tangent space projections, geodesic integrations, and closed-form parallel transport of velocity vectors across every single adapted layer—simply to calculate a scalar ensembling weight is an extreme case of over-engineering. 

In real-world multi-task serving, engineers prioritize simplicity, interpretability, low latency, and ease of debugging. Implementing and maintaining a complex, stateful, second-order physical manifold integrator running in parallel with deep model inference is highly impractical and introduces significant conceptual and engineering overhead for no substantial reason.

### B. The "Decoupling Illusion" in Default Mode
In the default "Decoupled GraviMerge" configuration (used for the main results in Table 1), the spacecraft's trajectory is completely independent of the actual propagating neural activations $\mathbf{h}^{(l)}$ in the deeper layers ($l \ge 4$). 
The dynamic masses $M_k$ of the expert stars and the initial spacecraft coordinates are computed once at Layer 3. Throughout layers 4 to 14, the spacecraft moves deterministically in this static gravity field, completely decoupled from any downstream representation drift. 

This is a major conceptual and methodology flaw: **there is no active, closed-loop feedback or dynamic adaptation during the forward pass.** The spacecraft's motion is purely pre-determined by the initial early activation at Layer 3. Consequently, GraviMerge is not actually a "dynamic, stateful router" that adapts to running representations; it is simply a complex, deterministic interpolation function. The entire layer-wise physical simulation is a computational illusion—the sequence of ensembling weights $\alpha_k^{(4)} \dots \alpha_k^{(14)}$ can be pre-computed immediately at Layer 3. 

While the authors mention "Coupled GraviMerge" (Equation 10) as a closed-loop alternative, it is not the default, adds further complexity, and introduces yet another hyperparameter to calibrate.

### C. Negligible Practical Gain Over Simple, Zero-Overhead Baselines
A close examination of the empirical results in Table 1 shows that GraviMerge provides virtually zero practical benefit over simpler, more elegant baselines. Specifically, let us compare GraviMerge with **SPS-ZCA** (Single-Pass Model Merging with Zero-Shot Centroid Alignment, Zhang et al., 2025):
- **SPS-ZCA:** An incredibly simple method that aligns inputs with centroids once in early layers and uses static routing weights throughout. It has **exactly zero (0.00000) layer-to-layer weight jitter by design**, requires zero stateful tracking, zero extra parameters, and zero runtime computational overhead.
- **Performance Comparison:**
  - **SPS-ZCA Serving Accuracy:** $88.51\% \pm 1.68\%$ with **0.00000** Jitter.
  - **GraviMerge Serving Accuracy:** $88.69\% \pm 1.68\%$ with **0.00190** Jitter.

The performance "delta" of GraviMerge over SPS-ZCA is an incredibly minuscule **0.18% absolute accuracy improvement**, while actually *introducing* weight jitter (from 0 to 0.00190 MAD). No system practitioner would deploy a massive, fragile, state-dependent orbital mechanics simulator across every layer of a neural network to gain 0.18% accuracy on a toy dataset while sacrificing perfect stability and introducing massive complexity.

### D. Highly Weak, Synthetic, and Toy Evaluation Space
The primary evaluation of this complex mathematical framework is performed on a highly simplified, synthetic coordinate simulation proxy:
1. **The Dataset:** The authors use scikit-learn's `load_digits` dataset, which consists of only 1,797 samples of 8x8 grayscale images of handwritten digits. This is a tiny, toy dataset from decades ago.
2. **The Simulation:** To simulate deep latent manifolds, they project these 8x8 pixel features (64 dimensions) to 192 dimensions using a random orthogonal projection matrix.
3. **No Real Deep Models:** There is no actual inference on a pretrained Transformer, no active LoRA adapters, and no real downstream generation. The backbone propagation and multi-expert ensembling are mathematically modeled as coordinate operations on these projected representation vectors.

Casting a simple coordinate projection study on 8x8 digits as a "Projected Digit Representation Space (RDS) Proxy benchmark" and using it to validate a complex physics-informed manifold integrator is a massive mismatch. Without evaluation on real-world, large-scale deep models (e.g., LLMs or ViTs) performing actual downstream tasks (e.g., GLUE, MMLU, GSM8k), the practical utility and scalability of GraviMerge remain completely unproven.

### E. Redundant and Misleading Empirical Presentation
In Table 1, the columns for "Homo. Acc.", "Hetero. Acc.", and "Real-Time Acc." report **exactly identical accuracies** for every single method (e.g., $88.69\% \pm 1.68\%$ for GraviMerge across all three). This is because each sample in the default execution model is processed independently, meaning the batching configuration has no mathematical effect on the individual sample's inference trajectory. 

Reporting the same mathematical results across three separate columns to claim "total multi-stream robustness" and "high resilience" is highly redundant and misleading. There is no actual dynamic streaming sequence or multi-stream interference being modeled in these main experiments.

### F. Fragility and Hyperparameter Calibration Overhead
GraviMerge introduces a laundry list of continuous hyperparameters: $G$ (gravitational constant), $\gamma_{\text{drag}}$ (viscous drag), $\Delta t$ (virtual step size), $\epsilon$ (softening factor), $\tau_{\text{grav}}$ (routing temperature), and potentially $\eta_{\text{feedback}}$ and $\lambda_{\text{temporal}}$. 
As demonstrated in the parameter sweeps, the system exhibits severe sensitivity and "phase transitions" (e.g., when $G \ge 0.05, \epsilon = 0.1$, force singularities trigger velocity spikes and explode jitter to $0.0226$). Such extreme sensitivity and high parameter calibration overhead make the framework highly fragile and impractical for real-world deployment.

---

## 4. Questions and Constructive Feedback for Authors
1. **Addressing the Decoupling Illusion:** In default Decoupled Mode, since the spacecraft's trajectory is completely independent of the propagating representations $\mathbf{h}^{(l)}$ for layers $l \ge 4$, why perform the physical integration layer-by-layer during inference? Why not pre-compute the entire sequence of ensembling weights $\alpha_k^{(l)}$ at Layer 3 using a simple feed-forward mapping, which would eliminate all layer-wise state tracking and sequential execution bottlenecks?
2. **Simpler Alternative Baselines:** If smoothing is required to suppress layer-to-layer weight jitter, why not evaluate standard, lightweight signal-processing smoothing techniques (e.g., a simple moving average, or a well-tuned EMA) directly on SABLE's routing weights? While the paper claims EMA suffers from "lag-induced control loop delays," did you perform a systematic hyperparameter sweep on the smoothing coefficient ($\beta$) to find an optimal balance, or is the lag-induced drop a consequence of the artificial proxy benchmark?
3. **Evaluating on Real Downstream Tasks:** To prove that this complex formulation has any practical value, do you plan to evaluate GraviMerge on actual, pretrained Large Language Models (such as LLaMA or Mistral) adapted with active LoRA experts on real language benchmarks (e.g., MMLU, GSM8k, or GLUE)? Showing a substantial performance gain on real tasks is essential to justify this level of complexity.
4. **Simplifying the Hyperparameter Space:** Can the framework be simplified to reduce the number of tunable parameters? For instance, can the virtual step $\Delta t$, drag $\gamma_{\text{drag}}$, and gravitational constant $G$ be analytically folded into a single, cohesive inertia parameter to make calibration more robust?
5. **Evaluating Genuine Streaming Workloads:** For the temporal carryover analysis, can you provide a genuine streaming experiment where sequential queries are temporally correlated, rather than reporting identical accuracies in Table 1 across three independent batching configurations?

---

## 5. Ratings & Overall Recommendation

- **Soundness:** **Fair**  
  *Justification:* The mathematical and geometric derivations on curved manifolds are technically sound, but the default "Decoupled" mode contains a severe conceptual flaw where the stateful simulation is completely independent of the running activations, rendering the layer-by-layer physical simulation redundant. Additionally, the method is highly sensitive and fragile to hyperparameter tuning.

- **Presentation:** **Fair**  
  *Justification:* While the paper is professionally written and formatted, the dense astrophysical analogies (spacecraft, stellar attractors, viscous medium drag) heavily obfuscate what are essentially simple interpolation and filtering operations. Furthermore, Table 1 contains highly redundant, identical accuracy columns to claim multi-stream resilience, which is misleading.

- **Significance:** **Poor**  
  *Justification:* The proposed method is highly over-engineered and impractical for real-world edge deployment. Crucially, the extremely simple, zero-overhead baseline SPS-ZCA achieves virtually identical accuracy (+0.18% difference) with perfect zero layer jitter and zero extra state/parameters, leaving no practical incentive for any practitioner to implement or deploy GraviMerge.

- **Originality:** **Good**  
  *Justification:* Mapping representation space routing to second-order Newtonian orbital mechanics on spherical manifolds is a highly unique, creative, and mathematically novel metaphor, even if it is practically over-complicated.

- **Overall Recommendation: 2 (Reject)**  
  *Justification:* Under standard ML design and engineering principles, complexity must only be introduced when absolutely necessary and justified by substantial gains. GraviMerge represents the absolute antithesis of this principle: it introduces an enormously complex, stateful, second-order physical manifold integrator simply to calculate routing weights, achieving a negligible 0.18% accuracy gain over an incredibly simple, zero-overhead, static-weight baseline (SPS-ZCA) that has perfect zero jitter by design. Evaluated only on a tiny, 1,797-sample toy dataset of projected 8x8 digits and suffering from high hyperparameter fragility, the paper represents a severe case of over-engineering that offers no practical value to the machine learning community. I strongly recommend rejection and encourage the authors to explore simpler, more elegant, and practically viable smoothing frameworks evaluated on real-world deep models.
