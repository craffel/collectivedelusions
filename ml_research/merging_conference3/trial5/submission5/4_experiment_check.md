# Experimental Quality and Validation Audit

## 1. Experimental Rigor Rating
**Rating**: **Excellent**

The empirical validation in this paper is exhaustive, thoroughly executing its multi-tiered research agenda. Rather than relying on a single experimental dataset, the authors validate their claims on both a highly controlled **Isolating Coordinate Sandbox** and a real-world **Vision-Language CLIP-ViT-B/16** scale pilot. 

---

## 2. Quantitative Results & Key Visualizations

The paper contains several high-quality tables and figures that collectively deconstruct wave-based routing and establish classical alternatives:

### A. Main Sandbox Multi-Task Generalization (Table 1)
Under rigorous evaluation on a separate test split (1,000 samples, 250 per task):
* **Expert Ceiling**: 79.80% (Joint Mean)
* **Uniform Merging (Static)**: 43.40%
* **QWS-Merge SOTA**: **36.10%** (Catastrophic collapse, performing worse than uniform average; only **2.00%** on out-of-distribution SVHN).
* **L3-Linear (Reg)**: **63.10%** (+27.00% absolute increase over QWS-Merge).
* **Linear Router (Global)**: **67.20%** (The absolute best performing dynamic model, exposing a major baseline blind spot in literature).

### B. Deployment Stream Audit under Batch Mixedness (Table 2)
Subjecting routers to different batch stream configurations exposes **"heterogeneity collapse"**:
* **Homogeneous stream ($B=256$)**: Linear Router achieves **67.20%**; QWS-Merge achieves **36.10%**.
* **Heterogeneous stream ($B=256$, mixed-task)**: Linear Router drops to **51.10%** (-16.10%); QWS-Merge collapses to **10.80%** (-25.30%).
* **L3-Softmax**: Drops from **54.40%** to **50.30%** (a minor **-4.10%** drop). However, as deconstructed in Appendix C, its absolute accuracy is consistently inferior to the Linear Router under all streams, demonstrating that its apparent robustness is an artifact of simplex compression forcing coefficients toward mediocrity.

### C. Real-Scale CLIP-ViT-B/16 Pilot (Section 4.5)
To prove that sandbox insights scale to commercial-scale models (86M parameters), the authors fine-tune and merge three CLIP models on MNIST, FashionMNIST, and CIFAR-10:
* **Expert Ceiling**: 92.80% (Joint Mean)
* **Static Uniform Merging (Task Arithmetic)**: 72.40%
* **QWS-Merge SOTA**: **41.20%** (Catastrophic collapse on real weight manifolds).
* **L3-Linear Router**: **84.80%** (Stable generalization, outperforming QWS-Merge by **+43.60%** absolute margin).
* **Global Classical Linear Router**: **88.60%** (Maintains strong baseline capacity on real weight manifolds).

---

## 3. Comprehensive Ablations & Sensitivity Studies

The appendices provide an impressive suite of additional evaluations to check robustness across multiple dimensions:

1. **Optimization Sensitivity Sweep (Appendix B)**: Verifies that QWS-Merge’s collapse is not an artifact of bad hyperparameters, sweeping learning rates $\eta \in [10^{-4}, 10^{-2}]$. Peak performance is indeed at $10^{-2}$, and lowering it results in total collapse to random chance.
2. **Multi-Seed Robustness Audit (Appendix D)**: Sweeping 5 independent seeds confirms that the collapse of wave routing is a statistically robust property ($\text{Seed Mean} = 33.34\% \pm 9.51\%$), while the Linear Router is highly stable ($69.68\% \pm 1.11\%$).
3. **Task Correlation and Overlap Sweep (Appendix E)**: Sweeping task-correlation $\rho \in \{0.0, 0.25, 0.50, 0.75\}$ refutes the hypothesis that orthogonal boundaries artificially favor linear routing, with classical linear models dominating at every correlation level.
4. **True Layer-by-Layer Merging Audit (Appendix F)**: Under a 14-layer deep expert merging setup with no coefficient averaging, QWS-Merge collapses catastrophically to **10.60%**, whereas the global Linear Router achieves the peak Joint Mean of **35.50%** and L3-Softmax achieves **23.90%**.
5. **Sensitivity to Projection Dimension $d$ (Appendix G)**: Sweeps $d \in \{2, 4, 8\}$ to demonstrate the information bottleneck under $d=2$ and the curse of dimensionality/overfitting under $d=8$. It reveals that the Softmax simplex normalization acts as a barrier that prevents overfitting under higher dimensions, allowing the model to leverage expanded capacity.

---

## 4. Strengths & Minor Suggestions for Improvement

### Strengths:
* Highly rigorous, multi-tiered experimental setup that scales from a toy sandbox to a real CLIP-ViT-B/16 vision-language manifold.
* Excellent statistical hygiene with multi-seed audits, correlation sweeps, optimization sweeps, and dimensionality sweeps.
* Clear, insightful analysis of failure modes (heterogeneity collapse, wave-routing instability, and layer-averaging collapse).

### Suggestions for Improvement (Minor):
* While the CLIP-ViT-B/16 pilot is a major step forward, the authors could briefly state in the future work section how they plan to extend their scale verification to even larger-scale generative models (like LLaMA-3-8B or Mistral-7B) using the detailed compiler-level roadmap provided in Appendix A.2.
* In Table 3 (Table of Deployment Audits), it would be helpful to include L3-Linear (L2 Reg) alongside L3-Softmax and the Linear Router, to show how unconstrained layer-wise classical models behave under mixed-task batching.
