# 3. Soundness and Methodology

## Evaluation of Theoretical Soundness and Mathematical Foundations
While L-ARC is mathematically elegant and borrows heavily from classical control-theoretic principles, a close inspection of its mathematical foundations reveals several highly fragile assumptions and logical gaps:

### 1. Fragility of the Layer-Identity Approximation
The central step in simplifying the discrete Lyapunov difference $\Delta V^{(l)}$ (Eq. 16) is the **Layer-Identity Approximation**:
$$S(h^{(l-2) \text{ warped}}, \mu_k^{(l-1)}) \approx S(h^{(l-1)}, \mu_k^{(l)})$$
This assumption implies that adjacent layers in a deep neural network act as an identity mapping, modified only by an infinitesimally small residual update. In actual deep learning models, layers are designed to execute highly transformative, non-linear feature abstractions. 
* **The Error Bound is Loose:** Theorem 3.2 attempts to bound this approximation error by $e^{(l)} \le 2 \| r^{(l-1)} \|_2 + \| \mu_k^{(l)} - \mu_k^{(l-1)} \|_2$. However, in modern transformer blocks—especially in the middle layers or when utilizing highly non-linear FFN/MLP blocks—the residual update scale $\|r^{(l-1)}\|_2$ can be very large. If $\|r^{(l-1)}\|_2$ is large, this bound becomes extremely loose and practically meaningless, violating the core dissipative guarantees of the controller.
* **Empirical Confirmation of Fragility:** The authors' own empirical sweeps over the residual block scale $\gamma$ (Section 4.3, "Empirical Validation of the Layer-Identity Error Bound") provide a striking confirmation of this fragility. When the residual scale is very small ($\gamma = 0.1$, i.e., layers are practically doing nothing), L-ARC achieves a statistically significant gain. However, when the residual scale is increased to $\gamma \ge 0.5$ (closer to real-world active layers), the accuracy gain of L-ARC over ChemMerge drops to just **0.02% - 0.03%** and becomes completely statistically insignificant ($p > 0.10$). This demonstrates that the entire mathematical foundation of the closed-loop controller collapses under realistic residual update scales.

### 2. Over-Optimistic Taylor Linearization Assumptions
The closed-loop control law is derived by linearizing the warped similarity around $\eta = 0$. However, in practice, the controller operates at finite, non-trivial step sizes up to $\eta_{\max} = 0.15$. 
* To justify this, Theorem 3.4 derives a second-order Taylor remainder bound $|R_1(\eta)| \le 0.0169$. However, this proof relies on highly optimistic assumptions: $\|w\|_2 \le 1.0$, $h \cdot \bar{\mu} \ge 0.5$, and $n(\xi) \ge 0.95$. 
* Under severe routing confusion or transient unaligned states (such as during task transitions or under systematic routing bias), these alignment assumptions are violated. The displacement vector $\|w\|_2^2$ can grow up to $2.0$, causing the second derivative bound to blow up to $|g''(\xi)| \approx 10.0$ and the remainder error to jump to $0.1125$. While the authors claim the controller remains stable, this represents a massive rise in the approximation error that severely undermines the theoretical guarantees of dissipative control.

### 3. Untested Theoretical Extensions
* **Mid-Network Recalibration (MNR):** The authors theoretically propose MNR (Eq. 10) to mitigate representational drift in extremely deep models (32-to-72 layers). However, they provide **zero empirical validation** for this mechanism. It remains a purely speculative equation with no simulation or real-world results, which is a major methodological omission.
* **Online Centroid Adaptation:** Similarly, Section 3.10 derives a slowly-varying Lyapunov theory for online centroid adaptation under non-stationary environments. This is a highly complex theoretical formulation, but it is **never evaluated** in the experimental section.

---

## Evaluation of Reproducibility and Experimental Methodology
* **Synthetic, Stylized Testbed:** The primary evaluation is conducted in the "Analytical Coordinate Sandbox (ICS)"—a custom, 14-layer toy simulation where activations reside in a 192-dimensional space. There are no real transformer layers, no real attention blocks, and no real-world datasets (like actual text or image inputs to a neural network). Representing complex ML serving on edge hardware via a stylized coordinate noise simulator is a massive threat to generalizability. A custom simulation is notoriously easy to over-parameterize and tune to produce any desired result.
* **Extremely Limited Real-World Validation:** While the authors report a "Small-Scale Real-World Pilot Study on LLaMA-3-8B," this study is evaluated on only **100 test queries** using only **16 calibration samples** per task. There is no code provided, no repository linked, and no detailed description of the datasets (SST-2, AG-News, GSM8K) or how the LoRAs were trained, making this pilot study completely irreproducible and scientifically weak.
* **Missing Details on Centroids:** The paper states that task centroids are extracted once at Layer 3 and used across all subsequent layers. It is unclear why Layer 3 was chosen, and whether the performance of SABLE and SPS-ZCA is highly sensitive to this specific choice of anchor layer.
