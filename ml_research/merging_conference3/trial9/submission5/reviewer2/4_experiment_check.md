# 4. Experiment Check

## Critical Evaluation of the Experimental Setup
The experimental evaluation is structured across two distinct environments: a high-fidelity synthetic coordinate-based sandbox (ICS) and a pre-trained BERT-Tiny model on natural language classification tasks.

### 1. Synthetic Sandbox (ICS) Setup
- **Strengths:** Allows for systematic, independent sweeps of critical variables: calibration sample size $N_{\text{cal}}$, representation anisotropy (via Toeplitz covariance injection $\rho \in [0.0, 0.5]$), and optimization hyperparameters. The calibration of expert accuracies to mimic real-world benchmarks (MNIST, Fashion-MNIST, CIFAR-10, SVHN) is a rigorous design choice.
- **Weaknesses:** Highly idealized. The attraction dynamics are strictly linear, lacking the non-linear activation functions and complex attention routing mechanisms characteristic of real foundation models.

### 2. Pre-trained BERT-Tiny Setup
- **Strengths:** Validates the core claims on actual neural network weights and real natural language datasets (SST-2 vs. QQP).
- **Weaknesses:** **Extremely limited scale.** BERT-Tiny is a toy model (4 layers, hidden dimension 128) and the adapters are under-fitted (standalone accuracies of only $58.80\%$ on SST-2 and $65.60\%$ on QQP). While the authors acknowledge this scale limitation, it fails to capture the high-dimensional activation manifolds and hardware constraints of modern multi-billion parameter large language models (LLMs) or vision transformers (ViTs).

---

## Evaluation of Baselines
The paper compares its proposed regularized parametric routers against an exceptionally comprehensive and appropriate suite of baselines:
- **Static Uniform Merging:** An unbiased control representing a baseline of maximum entropy.
- **SABLE:** A prominent training-free, stateless nearest-centroid activation router.
- **ChemMerge:** A state-of-the-art stateful continuous-time ODE kinetics router.
- **Unregularized and Randomly-Initialized Routers:** Directly isolating the effects of the authors' proposed initialization and regularization modifications.

---

## Critical Analysis: Do the Results Actually Support the Claims?

### Claim 1: Classical Routers "Catastrophically Fail" in Low-Data Regimes Due to Overfitting
- **Data Support:** Table 1 ($N_{\text{cal}} = 64$) shows that unregularized Softmax ($68.00\%$) and unregularized Sigmoid ($63.52\%$) lag far behind stateless SABLE ($73.76\%$) and stateful ChemMerge ($76.90\%$). This supports the overfitting hypothesis under under-determined optimization (768 parameters from 64 samples).
- **Inconsistency/Catch:** The authors' proposed "Proper L2 Regularized Router" ($\lambda=10^{-2}$) achieves only $67.34\% \pm 0.58\%$ accuracy in this regime, which is **mathematically worse** than the unregularized Softmax ($68.00\%$) or weakly regularized Softmax ($\lambda=10^{-4}$ at $68.14\%$). 
- Therefore, **regularization does NOT close the performance gap** to training-free methods under small-sample constraints. The parametric router remains significantly inferior to SABLE and ChemMerge in low-data regimes. The authors frame this as a "deconstruction" that justifies SABLE/ChemMerge as highly necessary inductive geometric priors, which is a fair and honest interpretation, but it means their proposed regularized router is *not* a viable low-data replacement.

### Claim 2: Complete Generalization Recovery in the Large-Sample Regime
- **Data Support:** Table 2 ($N_{\text{cal}} = 4000$) shows that the unregularized Softmax router achieves $76.22\% \pm 0.78\%$ accuracy, outperforming SABLE ($73.76\% \pm 0.72\%$) by $+2.46\%$ absolute (verified with a highly significant paired t-test $p < 0.01$). This strongly supports the claim that adequate data calibration resolves representational bottlenecks.
- **Inconsistency/Catch:** Even in this large-sample regime, **ChemMerge ($76.90\% \pm 0.68\%$) still outperforms all parametric configurations**, including the unregularized Softmax ($76.22\%$) and the proposed regularized router ($\lambda=10^{-4}$ at $75.70\%$). 
- While the classical router is "highly competitive," ChemMerge retains a statistical premium. The authors explain this elegantly via ChemMerge's closed-loop temporal low-pass filter (stateful inertia) that stabilizes trajectories under heavy activation noise. Thus, ChemMerge is not entirely redundant; its continuous kinetics provide a genuine feedback-stabilization advantage.

### Claim 3: The "Jitter Myth" is Debunked
- **Data Support:** Table 3 (layer-wise classical router ablation) shows that a true layer-wise classical router scales the parameter count to 8,448 but exhibits extremely low routing jitter ($0.0068 - 0.0458$), which is comparable to or lower than ChemMerge's feedback kinetics ($0.0368$). This strongly supports the claim that parametric routers do not inherently suffer from wild layer-to-layer weight oscillations.

### Claim 4: Real-World Task Separability Bypasses the Overfitting Bottleneck
- **Data Support:** Table 5 (BERT-Tiny validation) shows that under extreme small-sample constraints ($N_{\text{cal}} = 32$), the unregularized Softmax router achieves $61.90\%$, outperforming SABLE ($60.00\%$) and ChemMerge ($60.00\%$). Under this extreme constraint, there is **no overfitting bottleneck**.
- The authors' explanation is highly sound: because sentiment analysis (SST-2) and duplicate question detection (QQP) are semantically disjoint, their pre-trained token representations map to highly separated subspaces. A linear router with only 256 parameters can easily locate a stable separating hyperplane using only 16 samples per task. This is an excellent, nuanced finding that reveals the overfitting bottleneck is fundamentally task-dependent rather than a universal property of parametric gating.
