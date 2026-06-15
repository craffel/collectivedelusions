# Paper Summary

**Title**: Demystifying Quantum-Inspired Model Merging: Layer-Wise Low-Dimensional Classical Routing Beats "Quantum" Wavefunction Collapse

## 1. Executive Summary
This paper presents a rigorous methodological and empirical deconstruction of Quantum Wavefunction Superposition Merging (QWS-Merge), a recent state-of-the-art "quantum-inspired" test-time model-merging framework. QWS-Merge claims SOTA multi-task performance by modeling task merging coefficients as wave-like phase interference inside a parameter Hilbert space. 

The authors approach these claims with skepticism, identifying two critical flaws in the original evaluation of QWS-Merge:
1. **Crippled Baselines**: The classical comparison "Linear Router" baseline was intentionally restricted to be global, bypassing layer-wise specialized parameter capacity.
2. **Omission of Basic Regularization**: The failure of the classical linear router under severe data scarcity (64 calibration samples) was attributed to classical representation limits rather than a simple lack of standard $L_2$ regularization (weight decay).

To investigate, the authors propose a transparent, classical alternative called the **Layer-wise Low-dimensional Classical Router (L3-Router)** that matches the parameter footprint (in fact, reduces it by 16.7%) and input-space constraints of QWS-Merge. Under a controlled multi-task representation sandbox (and verified via a real-world CLIP-ViT-B/16 scale pilot), the authors empirically show that:
* The wave-like formulation of QWS-Merge completely collapses, performing worse than simple static uniform merging.
* The classical L3-Linear router avoids this collapse, outperforming QWS-Merge by +27.00% absolute accuracy in the sandbox and +43.60% in the real-scale CLIP pilot.
* A simple global unregularized classical **Linear Router** baseline outperforms all multi-layer models, demonstrating that layer-wise routing is an over-engineered mechanism when a single classification head is merged.
* When evaluated under heterogeneous mixed-task batches, all dynamic routers suffer from **"heterogeneity collapse"** due to batch-averaging. Although a Softmax-constrained variant (L3-Softmax) seems robust, the authors expose this as a **"Robustness-Accuracy Illusion"**, as its absolute accuracy is inferior to the Linear Router.

---

## 2. Core Methodological Contributions

### A. The L3-Router Family
The authors formulate three classical low-dimensional routers that act on the exact same unit-projected PCA representation space $\psi(x)_b$ as QWS-Merge, but replace the wave cosine formulation with standard linear projections:
1. **L3-Linear**: $\alpha_{k, b} = \langle \psi(x)_b, W_{l, k} \rangle + B_{l, k}$
2. **L3-Tanh**: $\alpha_{k, b} = \tanh(\langle \psi(x)_b, W_{l, k} \rangle + B_{l, k})$
3. **L3-Softmax**: $\boldsymbol{\alpha}_{:, b} = \text{Softmax}(\mathbf{W}^{(l)} \psi(x)_b + \mathbf{B}^{(l)})$

### B. Mathematical Proof of Layer-Averaging Collapse
The authors present a closed-form mathematical proof demonstrating that when layer-wise routing coefficients are averaged to merge a unified classification head:
$$\bar{\alpha}_k = \frac{1}{L} \sum_{l=1}^L \alpha_{l, k} = \langle \psi(x)_b, W_{eff, k} \rangle + B_{eff, k}$$
where $W_{eff, k}$ and $B_{eff, k}$ represent the average weights and biases across all layers. This proves that any multi-layer specialized complexity collapses back to a single-layer routing space, explaining why a global, single-layer Linear Router baseline systematically outperforms more complex layer-wise alternatives.

### C. The Isolating Coordinate Sandbox
To decouple routing dynamics from weight space misalignment, the authors construct a synthetic sandbox with orthogonal task subspaces. This permits precise evaluation of routing equations ($\text{Error}_{routing}$) in isolation, setting $\text{Error}_{alignment} \approx 0$. 

---

## 3. Main Empirical Findings

1. **Catastrophic Collapse of Wave-Based Routing**: In the sandbox, QWS-Merge collapses to **36.10%** Joint Mean accuracy (worse than Uniform's **43.40%**), and drops to **2.00%** on the out-of-distribution SVHN task.
2. **Decisive Superiority of Classical Routing**: L3-Linear achieves **63.10%** (+27.00% absolute improvement).
3. **Global Linear Router Dominance**: The global classical Linear Router achieves **67.20%** Joint Mean, beating all multi-layer models and highlighting a major baseline confounder in model-merging literature.
4. **Real-Scale CLIP Validation**: Merging 3 actual task-specific CLIP-ViT-B/16 models verifies the sandbox insights: L3-Linear achieves **84.80%** (outperforming QWS-Merge's **41.20%** by a massive **+43.60%** absolute margin), while the global classical Linear Router leads with **88.60%**.
5. **Robustness-Accuracy Illusion**: Under heterogeneous mixed-task streams, L3-Softmax drops by only **4.10%** (vs. Linear Router's **16.10%** drop), but its absolute accuracy remains inferior in all configurations, demonstrating that its stability is an artifact of the Softmax constraint forcing coefficients toward a mediocre uniform average.
