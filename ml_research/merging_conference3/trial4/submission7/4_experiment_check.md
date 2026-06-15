# 4. Experimental Validation and Empirical Quality

## Strengths of the Experimental Design
1. **Statistical Rigor:** All simulated results are evaluated and averaged over **30 independent random seeds**, providing solid statistical confidence intervals and standard deviations. This represents a high standard of empirical reporting.
2. **Comprehensive Suite Partitioning:** Decomposing the standard 4-task pool into five distinct evaluation suites (Suites A to E) along axes of domain distance and representational conflict is highly effective for isolating the "Task Suite Bias".
3. **Symmetrical Optimization Baselines:** The authors address potential optimization budget and capability asymmetries by introducing three new baselines: `OFS-Tune (Adam)`, `OFS-Uncon (Adam)`, and `AdaMerging (L-BFGS-B)`. This deconstruction mathematically confirms that OFS-Tune's superior generalizability is driven by its structural trajectory constraints rather than optimizer budget.
4. **Validation Set Sensitivity Analysis:** In Appendix C, the authors perform a highly detailed sensitivity analysis over the validation set size $M \in \{1, 5, 10, 20, 50\}$, data draw variance, and class-imbalance risk. This establishes $M=10$ as a robust, highly practical "golden threshold".
5. **Validation Class-Imbalance Analysis:** The authors identify that random uniform draws of size $M=10$ from a 10-class dataset have a 99.96% probability of omitting at least one class entirely. To address this risk, they implement and document **stratified sampling** (ensuring exactly 1 sample per class is drawn), which represents outstanding methodological rigor.
6. **Physical Weight-Space Validation:** Implementing a physical weight-space model merging protocol on actual neural network weights (evaluating scratch-trained vs. pre-trained experts, and auditing temporal EMA smoothing) provides vital empirical verification, validating the "privilege trap" and confirming simulated predictions.
7. **Temporal Smoothing Audit:** The authors evaluate whether online TTA can be rescued through standard temporal Exponential Moving Average (EMA) parameter smoothing. This empirical addition shows excellent completeness.

---

## Weaknesses and Empirical Limitations

### 1. Small Scale of the Physical Validation
- **Analysis:** While the simulator is calibrated against a 12-layer Vision Transformer (ViT-B/32) backbone, the actual physical weight-space validation in Section 4.5 is performed on a **very small-scale 5-layer Convolutional Neural Network** on simple grayscale datasets (MNIST and FashionMNIST) trained on CPU.
- **Implication:** The paper's key claims regarding physical weight-space performance (e.g., "OFS-Tune successfully outperforms online PolyMerge by up to 3.70%") are based entirely on this toy CNN. It remains unproven whether these exact numerical advantages generalize to physical weight-space model merging in massive parameter spaces (like ViTs, LLMs, or VLMs).
- **Resolution:** The authors have prominently and transparently qualified this limitation in Section 4.5, explicitly stating that validating these dynamics on larger foundation models is a necessary step to establish the absolute scale of their generalizability advantages.

### 2. "Strawman" Nature of Regime A in Physical Validation
- **Analysis:** In Regime A (scratch-trained experts), the authors train experts starting from completely independent random initializations (different seeds) and show that linear model merging collapses completely to random guessing (~12% accuracy).
- **Implication:** While this is mathematically and physically true, it is a well-established consensus in the deep learning and model-merging literature that **linear weight-space interpolation fundamentally relies on shared pre-trained initializations (linear mode connectivity)**. 
- **Resolution:** The authors have clarified that this result serves as a standard *sanity check* to confirm standard weight-space properties (linear mode connectivity) rather than a major empirical discovery. This maintains an objective and peer-reviewed scholarly tone.

### 3. Abstraction of the Unsupervised Entropy Surface in Simulation
- **Analysis:** In the simulation study, the online TTA optimization objective $\mathcal{L}_{\text{TTA}}$ directly tracks the ground-truth optimal parameter profiles perturbed by noise. In reality, physical online TTA must optimize a highly non-convex, rugged unsupervised prediction entropy surface without any access to the true parameter target curves.
- **Implication:** This surrogate mismatch over-estimates the optimization capability of online TTA in the simulation, creating a noticeable simulated-to-physical gap (which the authors acknowledge in Section 4.5, where online TTA actually degrades the physical Uniform baseline, whereas it improves over it in the simulator).
- **Resolution:** The authors have added an extensive and highly insightful discussion in Section 4.2 detailing this simulated-to-physical gap and laying out design guidelines for future model-merging simulators to close this gap.
