# 4. Experiment Check

## 1. Analysis of the Analytical Coordinate Sandbox (ACS)
The authors evaluate their methods inside a simulated, synthetic environment called the **Analytical Coordinate Sandbox (ACS)**. 
*   **Purpose:** To isolate the geometric and learning-theoretic properties of weight-space merging trajectories from hardware and optimizer noise.
*   **Setup:** Modeled as a purely linear dynamical system of coordinate recurrence:
    $$h_l = h_{l-1} + \gamma_l \left( \sum_k \alpha_k v_k - h_{l-1} \right)$$
    It simulates a 4-task visual stream mapping distributions along task-specific direction vectors. Calibration uses a tiny budget of only 10 samples per task (40 samples total).
*   **Assessment of ACS:** While the sandbox is highly controlled, it is a **toy model** lacking essential neural network properties (non-linear activations like ReLU/GELU, attention maps, or convolutions). The authors explicitly disclose this limitation, which is commendable. However, the reliance on such a simplified model as the *primary* quantitative testbed is a potential weakness.

---

## 2. The "Static Uniform Dominance Paradox" and Anisotropic Shearing Pathology
A major and highly surprising finding in the sandbox experiments is that the zero-tuning **Static Uniform** baseline consistently and significantly outperforms all tuned and adaptive methods, achieving $85.10\%$ on CNN and $83.75\%$ on CLIP, compared to the best proposed trajectory method at $70.70\%$ and $72.70\%$, respectively.

### Why does this paradox occur?
*   The authors explain that the ACS assumes **perfect coordinate alignment**, structural symmetry, and a lack of layer-wise capacity imbalances among task experts.
*   In this perfectly symmetric linear coordinate space, any deviation from uniform ensembling ($1/K = 0.25$) introduces **anisotropic representation shearing**, distorting the geometric topology of the shared representation and degrading the final categorical test predictions.
*   **Assessment of Explanation:** The authors' explanation of this "paradox" is mathematically and geometrically elegant. They frame the sandbox as a highly controlled "worst-case" scenario for adaptive merging. However, it still highlights a major limitation of the sandbox: it is too symmetric and idealized to reflect the actual utility of adaptive merging. 

To resolve this, the authors conduct **Coordinate Rotation Misalignment Sweeps** ($\eta \in [0.0, 0.6]$). This sweep shows that as representational misalignment increases, the performance of Static Uniform begins to degrade (dropping to $82.05\%$ at $\eta=0.6$), and trajectory-based adaptive methods provide much-needed stability, with the proposed **RB-DCTM (Ours, F=1)** achieving $74.60\%$ at $\max$ alignment shift.

---

## 3. Proof-of-Concept Validation on Actual Vision Transformers (ViT-B/16)
To confirm their theoretical insights on real weights, the authors perform a real-world merging experiment using two actual ViT-B/16 checkpoints fine-tuned on CIFAR-10 and CIFAR-100.

### A. Dual-Dataset Footprint for ZipIt! Alignment
*   To align hidden coordinate permutations, the ZipIt! baseline requires first- and second-order activation covariance statistics of size $D \times D$ ($768 \times 768$).
*   Estimating these high-dimensional covariance matrices using only the tiny 10-shot calibration set (20 samples total) would yield severely rank-deficient, noisy matrices.
*   **Methodological Solution:** The authors use a separate, unlabeled calibration footprint of **100 samples per task** strictly to compute stable covariance statistics for ZipIt! alignment. Once aligned, the trajectory parameters themselves are optimized strictly on the tiny 10-shot calibration set, maintaining the realistic few-shot adaptive learning regime.
*   **Assessment of Soundness:** This is an outstandingly honest and mathematically sound disclosure. It solves a crucial practical rank-deficiency issue while preserving the integrity of the few-shot optimization protocol.

### B. Real-World Quantitative Results
Unlike the sandbox, the real-world results demonstrate the true utility of the proposed spectral regularizer:
*   **Static Uniform (ZipIt! Aligned):** $71.30\%$ joint average accuracy (showing significant representational interference and collapse in actual non-linear weight spaces).
*   **RBPM (Polynomial, $d=2$):** $70.70\%$ (degrading below Static Uniform due to boundary runaway/Runge's oscillations).
*   **RB-DCTM (Ours, F=2):** **$74.90\%$** ($+3.60\%$ over Static Uniform, $+4.20\%$ over polynomial merging, and $+5.10\%$ over unconstrained optimization).

These results are highly convincing. They demonstrate that:
1.  In actual heterogeneous and non-linear weight spaces, uniform merging is sub-optimal due to representational collapse.
2.  Low-frequency spectral trajectories (particularly the non-periodic RB-DCTM) provide the necessary adaptive capacity to trace non-linear loss valleys while successfully avoiding transductive overfitting and boundary runaway.

---

## 4. Analysis of Hyperparameter Sweeps

### A. Regularization Strength ($\gamma$)
*   The ablation study shows that a moderate regularization strength of $\gamma \approx 0.01$ serves as the optimal sweet spot.
*   *Reviewer Critique:* How does one choose this optimal $\gamma$ in practice when the calibration dataset is extremely small (10-shot)? Tuning $\gamma$ on the training/calibration split can lead to overfitting, and a separate validation split is often not available in few-shot scenarios. The paper should discuss data-driven, cross-validation-like strategies or automated adaptive heuristic selection of $\gamma$.

### B. Spectral Cutoff Frequency ($F$)
*   The authors sweep $F \in \{1, 2, 3, 4, 5\}$ and observe a classic bias-variance curve. On CNN, as $F$ increases, accuracy drops monotonically due to high-frequency overfitting and representation shearing. On CLIP, DCT (RB-DCTM) peaks at $F=3$ before deteriorating.
*   This empirical observation matches the logarithmic scaling predicted by Theorems 3.1 and 3.4.
*   **Automated Frequency Selection:** The authors propose an elegant automated frequency selection mechanism at the end of Section 4.5. By initializing with $F_{\max}$ and applying $L_1$ Spectral Lasso, the optimizer shrinks redundant high-frequency coefficients to zero. This is a very practical and elegant solution to eliminate manual sweeps of $F$.

---

## Empirical Evaluation Conclusion
The empirical evaluation is **good but has scope limitations**. The synthetic sandbox is mathematically clean but highly idealized, causing the Static Uniform baseline to act as a dominant upper bound. The real-world Vision Transformer validation successfully addresses this limitation, demonstrating a clear $+3.60\%$ absolute accuracy improvement over Static Uniform and $+4.20\%$ over the polynomial competitor. However, the real-world validation is small-scale (2 tasks, ViT-B/16 only). Expanding these real-world experiments to larger benchmarks (such as multiple visual tasks or large language models) would significantly strengthen the paper's empirical weight.
