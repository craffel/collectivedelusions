# Evaluation Task 4: Experimental Evaluation and Claims Support

An empiricist reviewer evaluates research based on the rigor, realism, and transparency of the experimental design, and whether the empirical evidence genuinely supports the central scientific claims. Here is a detailed audit of the submission's experimental setup.

---

## 1. Quality and Realism of the Experimental Setup

### A. Synthetic and Simulated Environment (Toy Sandbox)
The major weakness of the experimental setup is its **artificial, synthetic nature**:
- The paper relies on a "Controlled Representation Sandbox" where representations are extracted from a pre-trained Vision Transformer (ViT-Tiny) and evaluated on synthetic class prototypes.
- The task vectors ($V_k$) representing the specialized expert weights are generated using **independent Gaussian parameters** ($\mathcal{N}(0, I)$). 
- While this toy setup allows the authors to perform rapid mathematical deconstructions, it does not represent real-world model merging. In practice, task-specific expert weights are fine-tuned from a shared initialization, meaning they are highly correlated and reside on low-dimensional manifolds.
- The authors do not evaluate EHPB on any real-world model-merging benchmarks (such as merging actual fine-tuned checkpoints of Large Language Models on GLUE/MMLU, or actual Vision Transformers on VTAB). Consequently, the empirical findings represent a simulation study rather than a validated machine learning system.

### B. The High-Noise SVHN Sandbox and Floor Effect
- For the SVHN dataset, the "Expert Ceiling" is artificially throttled to a low **16.8%** (by setting its simulation noise scale to a high $\sigma = 0.90$).
- As the authors honestly admit, this low signal-to-noise ratio (SNR) configuration creates a **floor effect**. 
- Because the ceiling is so low, all ensembling models are compressed into a narrow band between 9.6% and 14.4% accuracy. This floor effect masks the true severity of EHPB's performance collapse. On CIFAR-10 (a cleaner dataset), EHPB collapses from an 81.6% ceiling to 12.0% (-69.6% drop), whereas on SVHN the drop is only -7.2% (from 16.8% to 9.6%).

---

## 2. Rigor and Selection of Baselines
The paper compares EHPB against an extensive and highly appropriate set of baselines:
1. **Static Uniform Merging:** The standard linear average of task vectors.
2. **Global Linear Router:** Gating applied globally.
3. **vmap-Linear-Router:** A newly introduced baseline representing direct sample-by-sample additive ensembling implemented in PyTorch using vectorized map ($\mathtt{torch.vmap}$) without hyperdimensional binding overhead.
4. **QWS-Merge SOTA:** A wave phase-interference ensembling method.
5. **L3-Routers (Linear, Softmax, Tanh with/without Regularization):** Low-dimensional layer-wise routers proposed in prior work.

By including the `vmap-Linear-Router` and static `Uniform Merging`, the authors avoid evaluating EHPB against a "strawman" baseline.

---

## 3. Evaluation of Claims vs. Empirical Results

### Claim 1: EHPB is completely immune to task heterogeneity collapse.
- **Empirical Support:** **Fully supported.** The deployment audit ($B=256$) confirms that EHPB maintains a Delta of exactly 0.0% in classification accuracy between homogeneous and mixed-task batches. This is because EHPB executes weight demodulation sample-by-sample via parallel element-wise multiplication.
- **Empiricist Caveat:** While the "immunity" claim holds mathematically, it is **practically irrelevant** because EHPB's absolute performance is extremely poor. EHPB's Joint Mean accuracy is **25.4%**, whereas the standard baselines (even when subjected to heterogeneity collapse) perform vastly better. For example, QWS-Merge collapses from 58.7% by only -2.7% (retaining ~56% accuracy), and the Direct Router collapses from 51.0% by only a small fraction, maintaining around 48–50% accuracy. Thus, standard dynamic routers under heterogeneity collapse still outperform EHPB by over **+20.0% absolute accuracy**.

### Claim 2: Relative reconstruction error is scale-invariant across hidden dimensions under Hadamard binding, but decays as $O(1/\sqrt{D})$ under circular convolution.
- **Empirical Support:** **Fully supported.** 
  - Table 2 shows that EHPB's relative reconstruction error remains flat at approximately 170%–179% for dimensions $D \in [64, 2048]$, confirming the scale-invariance of element-wise Hadamard binding (the Coordinate Isolation Confounder).
  - Figure 4 shows that for circular convolution, the maximum cross-talk similarity with incorrect templates decays as $O(1/\sqrt{D})$ (from 12.02% at $D=128$ to 1.53% at $D=8192$), creating a wide, error-free decision margin for associative template retrieval.

### Claim 3: Increasing carrier key rank $r$ acts as a tunable knob to trade off structured weight noise against key-storage overhead.
- **Empirical Support:** **Fully supported.** The systematic rank sweep ($r \in [1, 192]$) in Section 4.2 reveals that raising the rank of the carrier keys from $r=1$ to $r=8$ breaks the low-rank structured sign correlation. This shifts the cross-talk noise to high-entropy isotropic noise that spatial pooling filters out more effectively, boosting the Joint Mean accuracy from 28.4% (at $r=1$) to 34.0% (at $r=8$).

### Claim 4: Residual-EHPB and Structured Row-wise Residual-EHPB can rescue representation propagation.
- **Empirical Support:** **Fully supported.** 
  - Designating just 5% of the most critical coordinates to bypass superposition improves EHPB's Joint Mean from 28.4% to 33.7% (+5.3% absolute).
  - The structured row-wise residual pathway (which is hardware-friendly as it runs via dense GEMMs) achieves a relative weight reconstruction error of 168.35% compared to 160.58% for the unstructured version—an exceptionally small error penalty of only 7.77% absolute increase.

### Claim 5: Continuous Cleanup Networks (CCN) and ReLU Bias Correction stabilize representation propagation.
- **Empirical Support:** **Supported with candid, intellectually honest caveats.**
  - CCN reduces Layer 3 activation MSE by 8.1$\times$ and rescues MNIST accuracy from 61.2% to 81.2% (+20.0% absolute). However, the joint mean accuracy (27.9%) does not improve because the linear mapping introduces a projection distortion that degrades low-SNR tasks (FashionMNIST drops from 26.8% to 8.0%). The bottleneck MLP layout (Non-Linear CCN) mitigates this, raising the joint mean to 28.20%.
  - Learnable scale and shift calibration on a 16-sample set reduces 5-layer propagation MSE by 31.4% (from 0.3835 to 0.2630) and increases cosine similarity to 0.9492, proving that lightweight post-hoc calibration is highly effective at absorbing rectification bias.
