# Intermediate Review Phase 4: Experimental Evaluation Check

## 1. Quality and Scope of Experimental Setup
The empirical evaluation is highly structured and covers a wide array of configurations (seed sweeps, task entanglement, expert count scaling up to $K=16$, and frozen depth boundaries). However, a rigorous critique reveals that **the entire empirical foundation of this paper is built upon highly simplified, simulated "toy" environments**, which severely limits its practical significance.

### A. The Analytical Coordinate Sandbox (ICS)
- The primary ensembling performance results (representing MNIST, Fashion-MNIST, CIFAR-10, SVHN) are **entirely simulated**.
- No actual neural networks or adapters are trained or evaluated on image pixels.
- Instead, the authors construct synthetic orthogonal coordinates with hand-calibrated Gaussian representation noise scales ($\sigma = [0.05, 0.15, 0.40, 1.20]$) and synthetic projection logit functions to generate predictions.
- While the sandbox is highly transparently disclosed, synthetic orthogonal blocks are a very weak proxy for real multi-task representation manifolds, which are complex, non-orthogonal, highly non-linear, and shift dynamically.

### B. Real-World Vision Transformer (ViT-B/16) Validation
- This validation is explicitly a **routing-only simulation** conducted on offline, frozen activations.
- No actual task-specific expert adapters (such as LoRAs) are trained, loaded, or physically blended, and no parallel ensembled forward execution passes (CAB) are executed.
- The evaluation task is synthetic shape classification (PIL-generated Circles, Squares, Triangles, Crosses). This is an incredibly trivial task for a large pre-trained ViT-B/16 model (which easily achieves 93% accuracy).
- Thus, the paper contains **zero demonstration that ChemMerge's Catalytic Activation Blending (CAB) actually works for real model merging and ensembling on standard, challenging multi-task benchmarks** (e.g., VTAB, DomainNet, GLUE) with real trained adapters.

---

## 2. Baseline and Evaluation Suite
The paper evaluates ChemMerge against a rich and representative suite of baselines:
- **Oracle / Expert Ceiling:** Standalone execution of the correct expert.
- **Uniform Merging:** Static weight-space parameter averaging.
- **Linear Router:** Parametric classifier trained on calibration samples.
- **QWS-Merge SOTA:** Wavefunction superposition dynamic merging.
- **PFSR + MBH SOTA:** Parameter-Free Subspace Routing wrapped with a Micro-Batch Homogenization scheduling queue.
- **SABLE SOTA:** Stateless activation-space blending using raw cosine similarities.
- **SPS-ZCA SOTA:** Stateless early nearest-centroid routing.
- **Static EMA Routing:** Static Exponential Moving Average low-pass filters.

This represents an exceptionally complete baseline selection. However, the comparison to "Static EMA Routing" is a bit limited: was the static EMA hyperparameter $\beta$ optimized on a per-task basis?

---

## 3. Key Findings & Empirical Strengths
Despite the toy environments, the empirical sweeps do demonstrate several robust properties of ChemMerge's kinetics:
- **Immunity to Collapses:** Under vectorized serving ($B=1$), unregularized parametric routers (Linear Router, QWS-Merge) experience catastrophic Vectorization Collapse. ChemMerge maintains stable, optimal accuracy across all serving batch sizes without needing scheduling buffers.
- **Task Entanglement Resilience ($\rho$):** When task centroids are entangled ($\rho \in [0.0, 0.5]$), stateless nearest-centroid routing (SPS-ZCA) collapses immediately at $\rho = 0.1$ due to routing oscillations. ChemMerge degrades gracefully, confirming that depth-wise kinetics act as an exceptional noise filter.
- **Scale-Free Hyperparameter Transfer:** The continuous parameters ($\Delta t = 1.5, k_{\text{decay}} = 0.3$) transfer directly from the synthetic sandbox to the pre-trained ViT-B/16, confirming they represent physical, dimensionless rates of ensembling state transitions relative to layer depth.
- **Massive Jitter Reduction:** On real pre-trained ViT-B/16 activations, ChemMerge reduces layer-to-layer ensembling weight routing jitter by **9.9$\times$** compared to SPS-ZCA and over **2.15$\times$** compared to SABLE (at equivalent routing sensitivities).
- **Exponential Integrator Stability:** The derived analytical Exponential Integrator maintains a completely stable accuracy of $77.70\%$ even under extreme virtual step sizes ($\Delta t = 10.0$), outperforming standard Explicit Euler which degrades due to overshooting.

---

## 4. Lack of Physical Edge Hardware Benchmarks
The authors claim that ChemMerge is "hardware-friendly" and suitable for battery-powered edge devices due to its constant $O(1)$ edge serving latency.
- However, they only present CPU-bound, vectorized NumPy routing latencies (e.g., 19.9ms at $K=16$).
- Running 11 sequential Euler steps (or Exponential Integrator steps) at each layer block during the forward pass requires sequential computations of ensembling weights across depth, which cannot be parallelized.
- Implementing these steps in real PyTorch/TensorFlow serving pipelines would introduce Python-interpreter loop overhead or custom operator overhead at each layer block.
- An actual, wall-clock serving latency and energy-consumption evaluation on physical edge hardware (e.g., Raspberry Pi, NVIDIA Jetson, mobile phone NPUs) with real adapter loading (CAB) is missing, which weakens the systems-level claims.
