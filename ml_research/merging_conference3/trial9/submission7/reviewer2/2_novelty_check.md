# 2. Novelty Check

## Assessment of Key Novel Aspects and the 'Delta' from Prior Work
The core contribution of this paper is positioned as an extension of **ChemMerge** (a continuous-depth model ensembling framework based on discretized reaction-decay ODEs). The key "delta" or additions introduced in this paper over ChemMerge are:
1. **Dynamic, Closed-Loop Feedback:** Instead of a constant, heuristic active representation coupling rate ($\eta$), this paper proposes a closed-loop controller that adjusts $\eta^{(l)}$ on-the-fly based on a local Dissipation Guard derived from a candidate Lyapunov function.
2. **Entropy-Gated Concentration Gating (ECG-Reset):** A threshold-based rule ($H^{(l)} > 0.95 \implies \Delta t = 0.0$) that freezes the ODE kinetics state under high routing uncertainty.
3. **Representation-Agreement State Correction (RASC):** A rule that overrides corrupted feedforward routing outputs with coordinate-space similarity tracking when the router's prediction disagrees with representation-space nearest centroids.
4. **Entropy-Triggered Lyapunov Gating (ET-L-ARC):** A conditional optimization that bypasses the Dissipation Guard calculation when routing is highly confident or failed ($H^{(l)} < 0.15$ or $H^{(l)} > 0.95$).

---

## Characterization of Novelty: Significant vs. Incremental
While the mathematical formulation is highly elaborate—drawing on discrete-time Lyapunov stability, Taylor remainders, and Lipschitz error bounds—the underlying conceptual novelty is **incremental**. The paper is a straightforward extension of ChemMerge (2025). 

From a critical perspective, the "control-theoretic" framing wraps simple, classic threshold heuristics in complex mathematical jargon:
* **ECG-Reset** is a simple conditional check on routing entropy: if the Shannon entropy of the routing weights exceeds a static threshold ($0.95$), the integration step size is set to zero. This is a basic "sample-and-hold" or conditional gating heuristic, commonly used in signal processing, but here it is dressed up as a "control-theoretic kinetics-space shield."
* **RASC** is a standard fallback mechanism: if the feedforward router and representation-space nearest-neighbor classifications disagree, override the router weights with the representation similarities. While framed as a "dual-loop closed-loop self-correction," it is a highly intuitive sanity check that has been widely used in routing and cascading classifier architectures.
* **ET-L-ARC** is a basic conditional skip statement to avoid computing dot products when routing weights are already highly confident.

Furthermore, the paper's main theoretical asset—the Lyapunov-stable closed-loop feedback controller—is shown to be **practically redundant** under clean, standard serving workloads. The authors' own evaluations show that Decay-ChemMerge (which applies a simple linear decay to $\eta$ over depth) performs almost identically to full L-ARC in clean settings ($74.38\% \pm 0.31\%$ vs. $74.38\% \pm 0.30\%$). This suggests that the actual practical utility of the complex, on-the-fly Lyapunov dissipation calculation is negligible compared to simple depth-wise heuristics.

Thus, while the theoretical derivation of the dissipation coefficient and the second derivative remainder bounds are elegant, they represent an over-engineered mathematical wrapping around a set of relatively simple, intuitive routing heuristics built on top of ChemMerge. The novelty is largely incremental, and the primary accuracy gains in noisy settings are driven by the simple gating rules rather than the core Lyapunov stability feedback loop.
