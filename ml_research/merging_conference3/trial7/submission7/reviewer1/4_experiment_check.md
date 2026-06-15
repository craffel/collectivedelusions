# Evaluation: Experimental Validation

## 1. Experimental Setup and Baselines
The experimental validation is highly rigorous, extensive, and thorough:
- **Sandbox Environment:** Evaluates the core representation dynamics across 10 independent seeds using a high-fidelity 14-layer residual sandbox.
- **Physical Transformers:** Extends validation to physical, pre-trained architectures—a Vision Transformer (ViT-Tiny) on ImageNet-normalized pixels and a causal GPT-2 model on textual sequences—proving that the findings generalize to physical networks.
- **Strong Baselines:** Symmetrically compares ELATI against:
  1. *Expert Ceiling (Oracle)*: The absolute upper bound.
  2. *Static Uniform Merging*: Standard average merging.
  3. *DARE-Merging & TIES-Merging*: State-of-the-art static merging baselines.
  4. *Linear Routers (Reg & Unreg)*: Parameterized, supervised classifiers.
  5. *PFSR + MBH*: Penultimate-layer dynamic routing (the primary competitor).

---

## 2. Evaluation of Claims
The empirical results provide exceptional and robust support for all central claims:

### A. Accuracy and Conflict Resolution
- ELATI achieves a robust Joint Mean accuracy of **56.89% $\pm$ 1.66%**, representing a substantial **+8.62%** absolute gain over static Uniform Merging (**48.27%**). This proves that routing activations early at Layer 2 and soft-merging downstream weights successfully resolves representation conflicts.
- It vastly outperforms state-of-the-art static merging methods—**DARE-Merging** (**32.56% $\pm$ 2.66%**) and **TIES-Merging** (**37.39% $\pm$ 3.03%**)—proving that dynamic, on-the-fly ensembling is far superior to collapsing updates into a single static model.
- It retains a highly competitive accuracy profile relative to the deep penultimate PFSR (**58.25%**), losing only $1.36\%$ absolute accuracy while completely eliminating the throw-away first pass.

### B. Physical Systems Latency and Complexity Savings
- Symmetrical CPU execution benchmarks show a genuine **1.40$\times$ physical end-to-end CPU speedup** (reducing latency from 36.90 ms to 26.43 ms), validating the avoidance of 11 deep redundant layers in Pass 1.
- Vectorized projection micro-benchmarks show an outstanding **3.33$\times$ speedup** (0.39 ms vs 1.31 ms) due to reducing theoretical operations from $O(B \cdot K \cdot C \cdot D)$ to $O(B \cdot K \cdot D)$, which completely eliminates the class-head bottleneck.
- PyTorch CUDA-event profiling simulation on parallel GPU execution demonstrates a **5.36$\times$ speedup** on Pass 1 (0.1293 ms vs 0.6930 ms) by bypassing redundant kernel launches and HBM-to-register memory traffic.

### C. Robustness and Scaling Sweeps
- **Subspace Entanglement Sweep (Figure 8):** ELATI consistently maintains a positive utility margin over static Uniform Merging across the entire entanglement spectrum ($\eta \in [0.0, 0.8]$).
- **Calibration Split Size Sweep (Figure 11):** Shows that ELATI's unsupervised centroids converge rapidly with as few as 16 samples, and even 1 sample per task outperforms Uniform Merging.
- **OOD Noise Sweep (Figure 9):** Confirms that ELATI's non-parametric geometric centroids heavily outperform trained linear classifiers under severe out-of-distribution noise, demonstrating superior generalizability.
- **Active Expert Pruning Sweep (Figure 10):** Shows that applying a moderate pruning threshold ($\epsilon_{\text{prune}} \in [0.1, 0.3]$) successfully mitigates negative transfer from conflicting experts.

---

## 3. Critical Analytical Insights (Perspective of Simplicity)
From the perspective of a senior systems engineer who champions simple, robust designs:
- **Figure 9 (OOD Robustness)** is the most compelling result in the paper. It shows that the trained Regularized Linear Router—despite having a tiny 0.67% accuracy advantage under clean conditions—catastrophically collapses under test-time noise due to overfitting. Meanwhile, ELATI's simple, unoptimized cosine centroids degrade gracefully. This is a massive victory for non-parametric simplicity over parametric complexity.
- **Figure 11 (Calibration Size)** further reinforces this. The unsupervised centroids require zero optimization and stabilize almost instantly, whereas a parametric router would require much larger datasets to avoid overfitting.
- These results prove that the core, simplest formulation of ELATI is already highly optimal, rendering the complex "online centroid adaptation" (Section 3.2) completely superfluous.
