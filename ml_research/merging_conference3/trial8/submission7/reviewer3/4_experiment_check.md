# Evaluation Task 4: Experiment Check

## Critical Evaluation of the Experimental Setup
The paper utilizes two distinct experimental evaluation environments:

### 1. Analytical Coordinate Sandbox (ICS)
* **Design:** An isolated, closed-loop simulation of a 14-layer deep network with intermediate dimension $D=192$ and $K=4$ task manifolds representing MNIST, Fashion-MNIST, CIFAR-10, and SVHN.
* **Noise Calibration:** The authors mathematically model representation noise scales calibrated as $\sigma = [0.05, 0.15, 0.40, 1.20]$ to represent the actual relative empirical difficulties of these datasets in the literature.
* **Verdict:** While this is a simulated sandbox rather than an image-pixel execution of a real model, the sandbox is mathematically rigorous and highly appropriate for isolating representation and routing dynamics. It removes confounding optimization variables (such as optimizer learning rate, data augmentation, or random initialization noise), allowing the authors to focus on the ensembling mechanics with extreme scientific transparency.

### 2. Pre-trained Vision Transformer (ViT-B/16) routing-only simulation
* **Design:** Activations are extracted from the 12 encoder layers of a real pre-trained ViT-B/16 model on geometric shape streams (Circles, Squares, Triangles, Crosses) generated via PIL.
* **Separability Validation:** The authors explicitly validate feature discriminativeness across layers, showing that nearest-centroid classification accuracy rises from $26.00\%$ at Layer 1 to $93.00\%$ at Layer 12.
* **Verdict:** This setup successfully tests the method's Non-Equilibrium Kinetic Routing (NEKR) on real, high-dimensional, non-orthogonal activation manifolds. The choice of geometric shape streams is a reasonable proxy for checking manifold separability and routing stability, though standard multi-task benchmarks (such as VTAB-1k) remain a critical next step for demonstrating broad real-world adaptation impact.

---

## Baselines
The paper includes a highly comprehensive set of seven baselines that span weight-space merging, test-time ensembling, and scheduling:
1. **Expert Ceiling (Oracle):** An upper-bound reference executing the correct expert standalone.
2. **Uniform Merging:** Standard parameter averaging, evaluating static performance under parameter interference.
3. **Linear Router:** A parametric linear classifier trained on 64 calibration samples, evaluating standard test-time routing.
4. **QWS-Merge:** Quantum-inspired wavefunction superposition merging (recent state-of-the-art weight-space merging).
5. **PFSR + MBH:** Parameter-Free Subspace Routing wrapped with a Micro-Batch Homogenization scheduling queue (state-of-the-art scheduling baseline).
6. **SABLE (TMLR 2024):** Stateless, sample-wise activation-blending using raw cosine similarities.
7. **SPS-ZCA (JAIR 2025):** Early-layer nearest-centroid routing with unit-norm and dispersion calibration but without temporal state tracking.

---

## Critical Check of Results and Claims

### Claim 1: ChemMerge resolves the accuracy-latency trade-off of scheduling wrappers.
* **Claims:** Matches or exceeds the accuracy of PFSR+MBH SOTA while restoring constant $O(1)$ edge latency.
* **Evidence:** Table 1 shows PFSR+MBH achieves $77.52\%$ in heterogeneous sandbox with a $4\times$ latency penalty. ChemMerge achieves $78.06\%$ with $1\times$ latency.
* **Verdict:** Fully supported. ChemMerge completely eliminates the sequential backbone passes required by scheduling queues.

### Claim 2: ChemMerge is immune to both Heterogeneity and Vectorization Collapses.
* **Claims:** Robustness under mixed-task streams ($B=256$) and vectorized sample-by-sample serving ($B=1$).
* **Evidence:** Table 1 and Figure 3(b) show ChemMerge maintains flat, stable accuracy ($78.06\%$) across both $B=256$ and $B=1$. Unregularized parametric routers (Linear Router and QWS-Merge) perform well in homogeneous batches but collapse catastrophically under heterogeneous serving (QWS-Merge drops to $34.58\%$).
* **Verdict:** Fully supported. ChemMerge operates on sample-independent ODE kinetics, requiring no batch-level statistics or stateful buffering.

### Claim 3: ChemMerge dramatically reduces layer-to-layer ensembling weight routing jitter.
* **Claims:** Massive reduction in routing weight oscillations across the network depth compared to stateless methods.
* **Evidence:** On pre-trained ViT-B/16 activation manifolds (Table 3), ChemMerge achieves a routing accuracy of $93.20\%$ with a routing jitter of **0.0156**. SPS-ZCA achieves $92.80\%$ accuracy but has a high routing jitter of **0.1541** (9.9$\times$ higher than ChemMerge). At equivalent routing temperature ($\tau=0.01$), SABLE's routing jitter is **0.0336** (2.15$\times$ higher than ChemMerge's).
* **Verdict:** Fully supported. The ODE kinetics successfully act as a continuous-depth low-pass filter, smoothing the ensembling weight trajectory.

### Claim 4: ChemMerge's state-dependent kinetics outperform static EMA filters.
* **Claims:** Overcomes the lag-accuracy trade-off of standard static low-pass filters.
* **Evidence:** The authors sweep static EMA filters ($\beta \in [0.1, 0.9]$). At $\beta=0.5$ (comparable jitter to ChemMerge), the static EMA's routing accuracy drops to $91.80\%$ (a $-1.4\%$ drop). At $\beta=0.1$ (ultra-smoothed), it experiences a severe lag penalty, collapsing accuracy to $89.00\%$ (a $-4.2\%$ absolute drop).
* **Verdict:** Fully supported. ChemMerge's state-dependent adaptive kinetics scale the local smoothing factor dynamically, adapting quickly when similarity is high and decaying smoothly when it is low.

### Claim 5: The exact Exponential Integrator ensures absolute numerical stability.
* **Claims:** Exponential Integrator is stable across extreme step sizes without relying on heuristic projection clipping.
* **Evidence:** Figure 6 compares Explicit Euler (with clipping) and the Exponential Integrator across virtual step sizes $\Delta t \in [0.1, 10.0]$. While Explicit Euler experiences slight performance decay at very large step sizes (dropping to $77.40\%$ at $\Delta t = 10.0$), the Exponential Integrator maintains a perfectly stable, flat accuracy of $77.70\%$.
* **Verdict:** Fully supported. The convex combination mathematics mathematically guarantee containment within $[0, 1]$ for any step size.
