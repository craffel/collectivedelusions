# Intermediate Evaluation 4: Experimental Setup and Empirical Validation

## 1. Evaluation of the Experimental Setup, Datasets, and Baselines
- **Analytical Coordinate Sandbox (ICS):** The ICS environment is a highly controlled, synthetic 14-layer deep neural network simulation with standard ViT-Tiny configurations. It represents task manifolds using orthogonal coordinate blocks under cascading isotropic Gaussian noise. While synthetic, this environment is **highly appropriate** for isolating and diagnosing the physical mechanisms of representation noise and routing jitter in dynamic model serving without being confounded by training-time training instability.
- **Task Manifolds:** MNIST, Fashion-MNIST, CIFAR-10, and SVHN are modeled using different calibrated task-specific representation noise scales ($\sigma = [0.05, 0.15, 0.40, 1.20]$) and isotropic Gaussian noise ($\sigma_{\text{layer}} = 0.015$). This correctly models task difficulty and noise heterogeneity.
- **Baselines:** The choice of baselines (Oracle Ceiling, Uniform Static Merging, Stateless SABLE, and Stateful SOTA ChemMerge) is highly robust and representative. 
- **Statistical Hygiene:** The authors evaluate all methods across 10 independent random seeds under perfectly synchronized random streams. A pairwise seed-by-seed t-test is provided (SABLE vs. Momentum-Merge $p \approx 0.0212 < 0.05$; ChemMerge vs. Momentum-Merge $p \approx 0.0061 < 0.01$), which is exemplary scientific hygiene.

---

## 2. Empirical Support for the Core Claims
The empirical findings strongly support the paper's core claims:
1. **Occam's Razor Confirmed:** The basic Momentum-Merge matches or exceeds SOTA ChemMerge (Joint Accuracy **74.85%** vs. **74.71%**, Routing Jitter **0.0128** vs. **0.0153**) without any system-biochemistry or virtual-time solver overhead. This confirms that the biochemical ODE machinery is empirically redundant.
2. **The Accuracy-Stability Trade-off Mapped:** Evaluated under Layer-wise Centroid Calibration, SABLE + Layer Centroids achieves the highest joint accuracy (**77.24%**) but exhibits massive routing jitter (**0.0285**). Adding stateful temporal smoothing (Momentum-Merge Advanced) acts as a low-pass filter, trading 2.26% absolute classification accuracy to collapse routing jitter by **76.2$\times$** (down to a near-zero **0.000374**).
3. **Physical Interpretability of $\beta$:** Sweeping $\beta \in [0, 1]$ reveals a beautiful Pareto frontier (Figure 2) peaking at $\beta = 0.60$. This shows that $\beta$ acts as a physical controller balancing expert-routing responsiveness (plasticity) and representation-smoothing inertia.

---

## 3. Explaining Empirical Results with Mathematical Intuition (From Appendix)
As a theory-minded reviewer, we must highlight two outstanding investigations in the Appendix that map the exact physical boundaries of the proposed method:

### A. "Recurrence Trapping" under Scarce Calibration Data (Appendix D.3)
In Table 4, the authors sweep the size of the validation calibration subset $|\mathcal{C}_k|$. They expose a major architectural vulnerability in Momentum-Merge (Advanced) under data scarcity ($|\mathcal{C}_k| \le 16$), which they term **Recurrence Trapping**:
- When calibration data is scarce ($|\mathcal{C}_k| = 8$), the computed layer centroids are highly noisy, making the initial boundary weight highly inaccurate.
- Because Momentum-Merge has stateful temporal memory, this initial boundary error propagates through network depth, trapping the ensembling coefficients in highly sub-optimal states throughout the forward pass and collapsing joint accuracy to **71.20%** (a 4.80% absolute degradation compared to stateless SABLE + LC, which achieves **76.00%**).
- **Mathematical Intuition:** Stateless routing evaluates activations independently at each layer, making errors localized and allowing the network to recover in subsequent layers. Momentum-Merge (Advanced) initializes its stateful recurrence using Raw Boundary Initialization (Eq. 7). This couples the entire ensembling trajectory directly to the first layer's routing weight. Under noisy centroids, the initial weight acts as a biased prior, and the low-pass filter (high $\beta$) prevents the recurrence from adjusting quickly, "trapping" the ensembling weights.
- **Significance:** This is a brilliant theoretical and empirical finding. It establishes that stateful smoothing is highly stable but requires a minimum calibration subset size ($|\mathcal{C}_k| \ge 16$) to avoid recurrence trapping of its initial state.

### B. Task-Asymmetric Noise Regimes: Constant vs. Dynamic Inertia (Appendix D.4)
In Table 6, the authors stress-test the constant-inertia assumption of Momentum-Merge under task-asymmetric noise scales:
- Under extreme noise asymmetry (Scenario D: MNIST clean, SVHN extremely noisy), ChemMerge's state-dependent reaction kinetics offer a minor joint accuracy buffer (**68.55%** vs. **68.40%** for Momentum-Merge Advanced).
- **Mathematical Intuition:** Since ChemMerge dynamically scales its back-reaction and forward reaction rates according to local similarity, it can apply highly localized, task-specific damping to the extremely noisy task while allowing rapid, low-inertia task-switching on standard, quiet tasks. 
- **Significance:** This maps the exact theoretical boundaries of the constant-inertia assumption. It proves that while task-asymmetric kinetics can offer minor localized accuracy benefits (+0.15% absolute) under extreme asymmetry, Momentum-Merge remains the superior engineering choice because ChemMerge's dynamic kinetics cause routing jitter to surge catastrophically (**0.0260** vs. **0.0029** for Momentum-Merge).

---

## 4. Critical Weaknesses and Limitations
1. **Ecological Validity:** The main evaluations are conducted within the synthetic ICS sandbox. In physical networks (like LLaMA-7B or Mistral-7B), task representations are not orthogonal, and representation spaces are highly non-linear, meaning that inter-task similarity overlap is much higher. While the authors propose concrete scaling modules in Appendix B, actual empirical results on massive pre-trained Transformer architectures are not provided in the paper.
2. **Missing Theoretical Analysis of Jitter:** The main text describes "routing jitter" reduction as an empirical finding, but lacks a formal mathematical framework explaining how the momentum parameter $\beta$ scales noise variance and jitter under a noise-propagation model (as derived in Intermediate Evaluation 3).
