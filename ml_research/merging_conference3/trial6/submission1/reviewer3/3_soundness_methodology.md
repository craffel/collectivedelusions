# Evaluation Task 3: Soundness and Methodology

## 1. Clarity of Description
The technical description of EHPB is **exemplary in its clarity and rigor**. The mathematical formulation is complete, unambiguous, and systematically laid out:
- The decomposition of task vectors ($V_k$) from pre-trained base weights ($W_{\text{base}}$).
- The generation of 2D pseudo-orthogonal random bipolar carrier keys ($K_k = r_k c_k^T$).
- The process of element-wise Hadamard binding ($V_k \odot K_k$) and holographic superposition ($W_{\text{holo}}$).
- The sample-wise unbinding operator ($U_b = \sum \alpha_{k,b} K_k$) and vectorized demodulation ($W_b$).
- Figures 1 and 2 provide clean, helpful conceptual schemas of EHPB and the Post-Hoc Model Ensembling Trilemma, respectively.

---

## 2. Appropriateness of Methods
- **Hyperdimensional Representation:** Extending VSA binding and unbinding operators to weight matrices is a mathematically sound way to superimpose tensors. The proof in Theorem 1 rigorously establishes that demodulation recovers the target linear combination up to a zero-mean cross-talk noise term.
- **Routing Calibration:** Utilizing a lightweight linear routing network trained on a 64-sample multi-task calibration set with AdamW and $L_2$ weight decay represents a standard, appropriate post-hoc gating setup.
- **Vectorized Demodulation:** Implementing the dynamic unbinding step via PyTorch's vectorized map ($\mathtt{torch.vmap}$) is appropriate for batched, parallel execution in eager mode.

---

## 3. Critical Technical Flaws, Limitations, and Empirical Objections (Empiricist Perspective)

From an empiricist's viewpoint, several critical technical limitations and methodological gaps must be raised:

### A. Lack of Statistical Rigor (No Confidence Intervals or Seed Variations)
A major empirical shortcoming of the submission is the **complete absence of statistical error bars, standard deviations, or confidence intervals** across all quantitative tables (Tables 1, 2, 4, 5, 6). 
- Because EHPB relies on randomly sampled bipolar carrier keys ($r_k \in \{-1,1\}^R$, $c_k \in \{-1,1\}^C$) and the routing network is calibrated on a tiny 64-sample stochastic set, the final accuracies and MSE values are bound to have high variance.
- Evaluating the model over multiple random seeds is an absolute requirement to confirm that the reported performance metrics (especially the small deltas in Tables 5 & 6) are statistically significant and not artifacts of a single lucky run.

### B. The Hadamard Dominance Paradox
The most glaring limitation of the proposed EHPB is its **extremely poor absolute performance** in practice.
- In Table 1, EHPB achieves a Joint Mean accuracy of **25.4%** under homogeneous batch conditions.
- By contrast, a simple static **Uniform Merging** baseline achieves **52.3%** accuracy—dominating EHPB by a massive **+26.9% absolute margin**.
- Uniform Merging requires:
  - Zero parameter overhead (no carrier keys to store).
  - Zero dynamic routing networks.
  - Zero execution latency overhead (no test-time demodulation).
  - Exact $O(P)$ active memory scaling (identical to EHPB).
- Why should a practitioner deploy a highly complex, mathematically noisy holographic system when a simple, zero-overhead static average is more than twice as accurate?
- Even when using **Residual-EHPB** with a 5% uncompressed coordinate ratio, the rescued Joint Mean of **33.7%** remains substantially below the static average (52.3%). This "Hadamard Dominance Paradox" indicates that the reconstruction noise penalty of element-wise binding in deep, non-linear networks is exceptionally destructive and difficult to overcome post-hoc.

### C. The Multi-Layer Non-Linearity Confounder
The authors provide an elegant mathematical derivation in Section 3.7 demonstrating why zero-mean reconstruction noise does not propagate safely through deep, non-linear architectures:
1. **ReLU Positive Bias Rectification:** Passing zero-mean noise through ReLU clips negative coordinates, converting the noise into a strictly positive bias vector $B^{(l)} \ge 0$ that systematically distorts representations.
2. **LayerNorm Exponential Signal Attenuation:** Weight-reconstruction noise increases pre-activation variance, prompting Layer Normalization to scale down the active semantic signal. This attenuation compounds exponentially across layers (attenuation factor $\eta \approx 0.50$, leading to $\eta^{14} \approx 6 \times 10^{-5}$ across 14 layers), completely extinguishing the semantic signal.
- This represents a fundamental architectural incompatibility: hyperdimensional holographic superposition is inherently incompatible with the coordinate-sensitive, non-linear propagation of deep neural networks.

### D. The Low-Rank Key Confounder
To keep carrier key storage compact ($O(K(R+C))$ parameters), the 2D keys are constrained to rank-1 outer products. This restriction forces the cross-talk noise to be highly structured and low-rank, preventing downstream layers and spatial token pooling from filtering it out via central limit averaging. Although raising the key rank to $r=8$ improves accuracy to 34.0%, it remains far below the static average baseline.

### E. Practical Edge Latency Disadvantage
The physical CPU-bound latency profiling (Section 4.5) reveals that EHPB's optimized fused demodulation takes **39.454 ms** per forward pass. Naive eager-mode sequential materialization takes only **16.004 ms**, and the Direct Router ($\mathtt{vmap}$-Linear-Router) takes **24.979 ms**. This demonstrates that EHPB is computationally slower than naive baselines on edge CPUs, undermining its dynamic on-device utility unless compiled onto specialized, high-concurrency GPU architectures.

---

## 4. Reproducibility
The paper provides sufficient detail to reproduce the experiments:
- Backbone model: pre-trained Vision Transformer ($\mathtt{vit\_tiny\_patch16\_224}$), $L=14$ layers, feature dimension $D=192$.
- Tasks and datasets: MNIST, FashionMNIST, CIFAR-10, SVHN.
- Calibration set: 64 samples (16 per task).
- Routing Optimizer: AdamW, learning rate $10^{-3}$, weight decay $10^{-3}$, 50 steps.
- Mitigations: Residual-EHPB coordinate budget ($p=5\%$), CCN samples ($N_{\text{ccn}}=400$), and ReLU Bias learnable scale/shift calibration (16 samples).
- Mathematical proof sketches are provided in the main text, with full derivations referenced in the appendices.
