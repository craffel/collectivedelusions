# 2_novelty_check.md: Novelty and Delta Check

## Characterization of Novelty
We characterize the novelty of this paper as **highly significant and methodologically refreshing**. Rather than proposing an increasingly convoluted, hyper-parameter-heavy online optimization algorithm that adds to test-time complexity, the paper conducts a rigorous "reality-check" audit. It demonstrates that the prior consensus on unsupervised test-time adaptive model merging is built on fragile, un-realistic assumptions. 

The paper's proposed alternative, **Offline Few-Shot Validation Tuning (OFS-Tune)**, is elegant in its simplicity. By combining continuous low-degree polynomial trajectory constraints with a tiny, stratified labeled validation set, it completely shifts optimization offline. This represents a major paradigm shift back to a highly practical, safe, and zero-test-time-compute calibration protocol.

---

## Detailed "Delta" from Prior Work

### 1. The Identification of "Task Suite Bias"
* **Prior Work (AdaMerging, PolyMerge, RegCalMerge):** These works validated their algorithms almost exclusively on a single, fixed combination of four datasets (MNIST, FashionMNIST, CIFAR-10, SVHN). The claimed SOTA superiority was accepted without questioning whether this specific task combination was a representative benchmark.
* **This Paper's Delta:** Exposes this single-suite benchmark as a major confounding variable. By systematically partitioning these four tasks into five distinct suites (representing different axes of domain distance and representational conflict), the authors reveal that performance rankings are highly sensitive to task relationships. Unconstrained online methods are shown to fail catastrophically in high-conflict suites (Suite B: CIFAR-SVHN), a critical weakness previously hidden by the standard multi-suite control benchmark (Suite E).

### 2. Modeling Realistic Transductive Stream Noise
* **Prior Simulators:** Abstracted noise during test-time adaptation as independent, zero-mean, i.i.d. noise per gradient update.
* **This Paper's Delta:** Formulates and models **transductive stream-level selection bias**. In physical environments, streaming batches are highly correlated. The authors model this by sampling a transductive stream noise offset ($\epsilon_{\text{stream}}$) exactly once per adaptation session and adding it directly to the optimal targets. This exposes how high-dimensional, unconstrained online TTA (AdaMerging) overfits to localized stream noise and collapses, which was obscured in prior idealized, zero-mean noise simulations.

### 3. Structural Dimensional Trajectory Constraints as Analytical Low-Pass Filters
* **Prior Work (PolyMerge):** Restricted layer-wise merging coefficients to a global polynomial function of depth to shrink the parameter search space. However, they still optimized this polynomial trajectory online over unlabeled streams via unsupervised entropy minimization.
* **This Paper's Delta:** Decouples the structural trajectory constraint from online adaptation, moving it to an offline, supervised few-shot validation tuning framework (OFS-Tune). Crucially, the authors introduce the **OFS-Unconstrained** baseline as an ablation. This isolates the regularizing effect of the polynomial trajectory constraint from the effect of having labeled few-shot validation data. The delta shows that having few-shot data alone is insufficient, as unconstrained offline optimization overfits to validation sampling noise; the continuous low-degree polynomial trajectory is a vital, mathematically necessary low-pass filter for robust model merging.

### 4. Alternative Localized Trajectory Formulations
* **Prior Work:** Relied strictly on global polynomials across network depth, assuming smooth sensitivity curves.
* **This Paper's Delta:** Proactively addresses potential circularity in global polynomial assumptions by formulating and evaluating two localized, non-smooth alternative parameterizations: **Piecewise Linear Splines** and **Block-wise Parameter Sharing** (attention vs. MLP grouping). Under highly challenging non-smooth "zig-zag" optimal trajectories, the authors prove that localized parameterizations successfully capture block-specific sensitivity spikes (such as MHA block precision) while maintaining low dimensionality, neutralizing circularity concerns and demonstrating scalability to physical Transformer architectures.

### 5. Deconstruction of the Optimization and Budget Asymmetry
* **Prior Work:** Did not analyze how different solvers and optimization horizons interact with the online/offline objectives.
* **This Paper's Delta:** Systematically evaluates symmetrical optimization baselines (such as OFS-Tune optimized via simple, restricted 100-step first-order Adam and online AdaMerging optimized via converged, second-order L-BFGS-B). The authors uncover a key scientific finding: online unconstrained prediction entropy under stream noise is a heavily misaligned surrogate objective. Forcing a high-capacity second-order optimizer (L-BFGS-B) to converge on this surface actually *degrades* performance by driving the parameters to overfit even more deeply to local stream bias, confirming that online TTA's failures are structural rather than optimizer-bound.

### 6. Physical Weight-Space Validation and the Role of Task Routing
* **Prior Work:** Standard evaluation protocols adapted on sequential single-task streams, ignoring the "privilege trap" of multi-task deployment on interleaved mixed streams.
* **This Paper's Delta:** Performs physical weight-space validation on CPU, examining both scratch-trained (independent) and pre-trained (mode-connected) initializations. They evaluate online TTA under both *Unsupervised* (mixed stream joint optimization) and *Privileged* (oracle task label routing) settings. The results show that:
   - On mixed unlabeled streams, unsupervised TTA collapses due to joint entropy minimization.
   - Even under privileged routing, online TTA actively degrades pre-trained weights by chasing stream noise.
   - Offline OFS-Tune operates successfully without any privileged routing assumptions at test time, and in scratch-trained regimes behaves as a "safe-by-default" algorithm, preserving the performance of a single expert rather than collapsing both.
   - Standard temporal parameter smoothing (EMA) helps online methods slightly in connected basins but fails to rescue them in high-conflict/independent basins, and still incurs test-time compute costs.
