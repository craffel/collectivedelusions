# Experimental Evaluation and Claims Verification

## Experimental Setup and Datasets
The authors build a highly controlled representation sandbox to evaluate dynamic routing mechanisms. 
- **Backbone Simulation:** They simulate a Vision Transformer (ViT-Tiny, `vit_tiny_patch16_224`) representation space with $L=14$ layer groups.
- **Task Construction:** Synthetic 192-dimensional hidden representations ($D=192$) are generated for four disparate visual classification tasks: MNIST, FashionMNIST, CIFAR-10, and SVHN. Class-specific prototypes are placed in orthogonal task subspaces, with noise added to represent task-specific difficulty (e.g., the SVHN expert ceiling is 32.00% due to simulated noise, reflecting realistic out-of-distribution difficulty).
- **Data Regime:** To simulate test-time adaptation and data scarcity, a tiny calibration split of exactly 64 samples (16 samples per task) is used to optimize the routers. A test split of 1000 samples (250 per task) is used for generalization testing.

This setup is highly appropriate for isolating routing-related errors from weight-space coordinate-alignment conflicts.

## Baselines
The paper includes a highly complete and rigorous set of baselines:
1. **Expert Ceiling:** The individual specialized expert accuracies (MNIST: 100%, F-MNIST: 96.8%, CIFAR: 90.4%, SVHN: 32.0%; Joint Mean: 79.8%).
2. **Static Uniform Merging:** Parameter averaging with static, uniform layer-wise weights.
3. **Global Classical Linear Router:** An unconstrained, single-layer global classical mapping that bypasses both low-dimensional projections and layer-wise specialized parameters.
4. **QWS-Merge SOTA:** The wave-inspired, quantum superposition model-merging formulation.
5. **Proposed L3-Routers:** Three classical alternative routing channels (Linear, Tanh, Softmax) evaluated in both unregularized and $L_2$ regularized configurations.

## Do the Results Support the Claims?
Yes, the experimental results provide overwhelming, unambiguous support for all of the paper's central claims:

1. **Catastrophic Collapse of QWS-Merge:** 
   * **Claim:** Quantum wave-like cosine formulations are highly unstable and collapse under sandbox validation.
   * **Evidence:** Table 2 shows QWS-Merge achieves only **36.10%** Joint Mean accuracy, performing worse than simple static Uniform Merging (**43.40%**). It collapses to a near-random **2.00%** on OOD SVHN images.
2. **Stability and Superiority of Classical Linear Routing:**
   * **Claim:** Simple classical linear projections project stable routing weights and avoid collapse.
   * **Evidence:** Table 2 shows **L3-Linear** achieving **63.10%** Joint Mean (+27.00% over QWS-Merge).
3. **The Ultimate Baseline Confounder:**
   * **Claim:** A simple global single-layer classical Linear Router baseline outperforms complex, layer-wise specialized routing.
   * **Evidence:** Table 2 reveals that the unregularized, global **Linear Router** baseline achieves the highest Joint Mean accuracy of **67.20%**.
4. **Deployment Stream Audit & Heterogeneity Collapse:**
   * **Claim:** Mixed-task batches cause severe heterogeneity collapse due to batch-averaging, but L3-Linear provides superior robustness.
   * **Evidence:** Table 3 shows the Linear Router's accuracy drops by **16.10%** (to 51.10%) and QWS-Merge drops by **25.30%** (to 10.80%) under mixed batches. **L3-Linear (L2 Reg)** achieves **52.30%** (the highest absolute accuracy under heterogeneous streams).
5. **Exposure of the Robustness-Accuracy Illusion:**
   * **Claim:** Simplex-normalization constraints (like Softmax) create an illusion of relative robustness that masks mediocre absolute capacity.
   * **Evidence:** Table 3 shows that while **L3-Softmax** is highly stable (dropping by only **4.10%** under shift), its absolute accuracy (**50.30%**) is inferior to the unconstrained Linear Router (**51.10%**) under both homogeneous and heterogeneous streams.
6. **Real-Scale Generalization:**
   * **Claim:** Sandbox insights generalize to real weight-space manifolds.
   * **Evidence:** Section 4.5 presents a real CLIP-ViT-B/16 model-merging pilot. On 86M parameter visual encoders, QWS-Merge collapses to **41.20%** Joint Mean, while L3-Linear achieves **84.80%** (+43.60% over QWS-Merge) and the global Linear Router achieves **88.60%**, matching the sandbox trends.
7. **Robustness to Objections:**
   * The claims are further validated across sweeps of task correlation $\rho$ (Table 4), multi-seed robustness audits (Appendix H), learning rate sweeps (Table 3, Appendix E), and true layer-by-layer weight-merging audits (Table 5, Appendix I).

The empirical coverage is exceptionally complete, rigorous, and completely consistent with the theoretical and conceptual claims of the paper.
