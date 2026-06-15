# Evaluation Checklist: Novelty, Literature Placement, and Originality

## 1. Contextualization in Prior Literature
The paper places itself elegantly at the intersection of three major paradigms in deep learning and cognitive science:
1. **Dynamic Model Merging & Gating:** Traditional routing methods (e.g., Mixture-of-Experts routing, SABLE [sable2024]) operate *statelessly* on a per-query basis. They process inputs in isolation, ignoring temporal structure. While they adapt instantly to task switches, they suffer from extreme high-frequency ensembling weight oscillation (jitter) under representation noise.
2. **Stateful Filtering & Biochemical Kinetics:** Recent frameworks like ChemMerge [chemmerge2025] and Momentum-Merge [momentummerge2025] introduce continuous statefulness to routing using biochemical ordinary differential equations (ODEs) or simple exponential moving averages (EMAs) of weights. While this acts as a low-pass filter to reject noise, it introduces massive representational lag (inertial drag) at task switch boundaries because these filters accumulate history rigidly and cannot distinguish noise from real transitions.
3. **Active Inference & Theoretical Neuroscience:** The Free Energy Principle (FEP) [friston2006free, friston2010free] posits that biological brains minimize Variational Free Energy to model the world and take action. This work is the *first* to translate this theoretical cognitive framework into a concrete, high-performance, and mathematically exact routing layer for deep learning serving streams.

---

## 2. Assessment of Originality & Conceptual Breakthroughs
The paper introduces several highly original, non-trivial breakthroughs:
- **Redefining Gating as Perceptual Action:** The conceptual formulation of a multi-expert routing layer as an *active, self-organizing cognitive agent* is a profound departure from traditional passive mathematical heuristics. It transforms model merging from static ensembling into an online active perception loop.
- **Analytical Derivation of Free Energy for Serving Streams:** The authors derive the Variational Free Energy specifically for a state-space formulation of modular serving streams. They show that this objective naturally simplifies to a combination of:
  1. *Sensory Prediction Error:* Downweights sensory noise based on learned observational precisions.
  2. *Prior Prediction Error:* Enforces sequential smoothness based on temporal prior transitions.
- **Closed-Form Active Inference Perception:** Active inference typically relies on slow, iterative, or unrolled optimization loops (e.g., gradient descent or backpropagation). In this paper, the authors prove that under a Gaussian state-space model, the variational free energy is strictly convex and quadratic. This enables solving for the optimal belief update $\mathbf{\mu}_t^*$ in a **single exact, closed-form analytical step** ($\mathbf{H}\mathbf{\mu}_t^* = \mathbf{b}_t$). This represents a major conceptual and systems-level breakthrough that completely eliminates the need for iterative unrolled step-size hyperparameters.
- **Control-Theoretic Equivalence:** The paper establishes a beautiful mathematical bridge, proving that under static variational covariance, this exact closed-form belief update is mathematically equivalent to a classical *linear Kalman filter / state observer*. This provides a rigorous control-theoretic derivation of linear state filters from first-principles variational active inference.

---

## 3. Novelty of Experimental and Scaling Results
The novelty is further cemented by the advanced scaling and robustness checks:
- **The AIR (Diagonal) Parameter-Efficient Variant:** To resolve the quadratic $\mathcal{O}(K^2)$ parameter scaling of the generative mapping $\mathbf{W}$ under larger registries, the authors introduce a diagonal structural constraint on $\mathbf{W}$. This compresses the calibration space to linear $\mathcal{O}(K)$ and enables high-performance calibration on tiny sequence lengths ($T_{\text{cal}} = 32$). This parameter-efficient routing variant represents a highly original development for large-scale Mixture-of-Experts systems.
- **Cross-Sequence Calibration Robustness:** The study of whether a routing agent's learned precision parameters generalize across extremely distinct environments (stable homogeneous vs. rapid heterogeneous) is highly original. It reveals that the Free Energy objective naturally converges to sequence-invariant precision parameters that generalize flawlessly across arbitrary test workloads without sequence-slicing overfitting.
- **High-Dimensional Nonlinear Manifold Stress Test:** By testing the linear-Gaussian approximation under a sinusoidal-quadratic warping and heavy-tailed Student's $t$ noise, the authors evaluate the limits of model mismatch. They show that AIR's routing stability prevents downstream categorical classification collapses (retaining $98.83\%$ vs. SABLE's $93.99\%$ collapse). This establishes a vital, previously unrecognized link between *sequential routing jitter* and *downstream representation alignment stability*.
- **PCA Dimension Sweep & Non-linear Projections:** The analysis of PCA projection dimension $d$ as a spatial low-pass filter, alongside evaluations of non-linear Contractive Autoencoders (CAEs) under Appendix N, represents a thorough and original exploration of representation subspace regularizations.

---

## 4. Distinction from Closest Works
The paper clearly distinguishes its contributions from related stateful and stateless baselines:
- **vs. SABLE (Stateless):** SABLE has no memory of past states. AIR introduces statefulness via top-down temporal priors, achieving up to 2.49$\times$ noise reduction on homogeneous streams, and up to 3.6$\times$ noise reduction under non-linear stress tests.
- **vs. ChemMerge & Momentum-Merge (Stateful):** These methods smooth trajectories rigidly, resulting in severe lag and accuracy collapse on heterogeneous streams (retaining only $\approx 53.4\%$ accuracy). AIR's precision-weighted prediction errors act as an adaptive switch: when context boundaries occur, prediction errors instantly spike and overcome the temporal prior, achieving near-instantaneous tracking (1-2 steps) and near-oracle accuracy ($66.23\%$).
- **vs. PAC-Kinetics (Recurrent/Learnable):** PAC-Kinetics relies on parameterized recurrent structures and unrolled training. AIR's exact closed-form solver is parameter-free during test-time, requires a tiny calibration sequence, possesses 100% numerical stability, and achieves superior robustness under non-linear model mismatch.

**Conclusion on Novelty:** The paper is exceptionally original. It represents a substantial conceptual leap that elegantly imports theoretical cognitive science and systems engineering into modern PEFT and Mixture-of-Experts serving systems, fully backed by rigorous, novel scaling and robustness investigations.
