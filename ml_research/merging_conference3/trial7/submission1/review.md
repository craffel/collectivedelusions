# Peer Review Report

## 1. Overall Recommendation
**Recommendation:** **5: Accept**  
*Reviewer Verdict:* This is a technically solid, exceptionally written, and conceptually profound paper. It adopts a highly refreshing and self-critical **Methodologist** perspective to deconstruct an influential claim in weight-space model merging ("Layer-Averaging Collapse" or rank-1 collapse). By combining rigorous Singular Value Decomposition (SVD) spectral analysis, hardware-grounded systems latency audits, and deep mathematical reasoning, the paper exposes key boundaries of dynamic model merging. It achieves perfect data synchronization with its codebase, respects all style and page limits, and compiles beautifully. It is an outstanding candidate for acceptance.

---

## 2. Strengths and Key Contributions
1. **Exceptional Intellectual and Scientific Honesty:** In a literature often dominated by selective hyperparameter tuning and over-hyped SOTA claims, this submission stands out for its absolute transparency and critical deconstruction. The authors openly report and analyze key failure modes and limitations of dynamic model merging:
   * **The Batch-Averaged Multi-Task Inference Paradox:** A brilliant, bulletproof conceptual formulation showing that dynamic model merging either collapses to static merging (under mixed batches due to batch-averaging) or becomes logically redundant/performance-degraded compared to direct expert execution (under homogeneous batches).
   * **The Capacity-Variance Trade-off:** The paper honestly reports that on TinyCNN-4, the minimal static baseline **OFS-Tune** consistently outclasses the Layer-wise Router under tight calibration splits. It provides an elegant explanation: the spatial redundancy of convolutional networks makes a global compromise highly effective, whereas the router's larger parameter count introduces optimization noise.
   * **The MLP Representational Collapse:** The paper acknowledges that under Cross-Domain task conflict on DeepMLP-12, all merged models collapse to near-random guessing ($11\%$--$16\%$), and uses this to expose a fundamental limitation of full-parameter dense linear interpolation.
2. **Deconstruction of rank-1 Collapse:** The paper successfully deconstructs the theoretical claim that layer-wise routing trajectories must collapse to a collinear rank-1 subspace. It proves that under cross-domain conflict on physical deep networks (TinyCNN-4 and DeepMLP-12), the Collinearity Ratio drops significantly ($0.49$--$0.56$), and pairwise cosine similarity maps show distinct, depth-specialized block structures aligned with hierarchical feature extraction.
3. **Rigorous Systems and Theoretical Grounding:**
   * **Memory-Bandwidth Latency Audit:** Appendix F details a formal memory-bandwidth transfer audit showing that dynamic weight-space merging on a 7B LLM requires transferring $70\text{GB}$ of parameters from HBM on every batch, adding $\ge 21\text{ms}$ latency on an H100. This mathematically justifies why dynamic merging must be restricted to low-rank PEFT/LoRA modules.
   * **Johnson-Lindenstrauss Projection Regularization:** The authors provide a highly novel theoretical and empirical justification for freezing the input projection matrix. Using the **JL Lemma**, they show that random projection acts as a crucial non-parametric regularizer, preventing the extreme calibration overfitting and generalization collapse that occurs when the projection is learnable.
   * **Softmax vs. BSigmoid Baseline:** The authors ran a hard head-to-head empirical comparison (Appendix A) showing that the proposed decoupled BSigmoid router outperforms standard competitive Softmax by massive margins ($+20\%$--$25\%$) on TinyCNN-4 convolutional layers.
4. **Pristine Formatting and Layout Compliance:** The paper is typographically flawless. It fits the entire main body within **exactly 8 pages**, with the bibliography commencing on Page 9 and the Appendix spanning Pages 10–17. Overfull horizontal boxes have been entirely resolved, and running header suppressions have been successfully fixed.
5. **Data Synchronization:** There is 100% perfect synchronization between the metrics serialized in the codebase (`results/metrics.json`) and the LaTeX tables in the paper (Tables 1, 2, 3, and Appendix Table 1).

---

## 3. Ratings
* **Soundness:** **Excellent** (The math, systems latency analysis, JL Lemma grounding, and batch-averaged paradox are theoretically bulletproof and highly rigorous).
* **Presentation:** **Excellent** (The manuscript is engaging, beautifully structured, and features high-resolution diagnostic plots and heatmaps of outstanding quality).
* **Significance:** **Excellent** (Re-opens the design space for layer-wise spatial routing while establishing clear systems and representational boundaries of weight-space merging).
* **Originality:** **Excellent** (Exposes fundamental boundaries and conceptual paradoxes of the field, replacing superficial performance chasing with high-signal scientific auditing).

---

## 4. Minor Suggestions and Constructive Feedback
The paper is solid and publication-ready. For future extensions or post-conference camera-ready revisions, the authors might consider exploring the following directions:

1. **Continuous-Time Dynamic Routing (Neural ODEs):**
   * *Idea:* In deep architectures, layer-wise routing coefficients can exhibit high variance across successive layers, which could exacerbate activation drift. 
   * *Suggestion:* Future work could investigate parameterizing the layer-wise routing coefficients as a continuous-time trajectory governed by a **Neural Ordinary Differential Equation (Neural ODE)**: $d\lambda(z)/dz = f_{\theta}(\lambda(z), \psi(x))$. Integrating this trajectory across the network depth ($z \in [0, 1]$) would enforce smooth, spatially regularized routing weights, potentially mitigating optimizer variance on scarce budgets.
2. **Alternative Non-Linear Fusion Operators:**
   * *Idea:* The authors deconstruct the convolutional Oracle gap by showing that linear weight blending acts as a low-pass filter on local spatial kernels, destroying high-frequency edge responses.
   * *Suggestion:* To bridge the $47\%$ Oracle gap on TinyCNN-4, research could look beyond linear blending. For instance, exploring **non-linear parameter blending operators** (such as spline-based interpolations or coordinate-based MLP weight generation) could preserve high-frequency filter profiles without requiring full-parameter ensembling.
3. **PEFT-Level Physical Scale-Up:**
   * *Idea:* The authors hypothesize in Appendix Section F and Appendix Section G that dynamic routing over low-rank PEFT/LoRA adapters would exhibit an even deeper spatial specialization (lower SVD Collinearity Ratio) than full parameters.
   * *Suggestion:* A physical verification of this PEFT adapter collapse hypothesis on Vision Transformers (ViT-B/16 CLIP) using standard task suites (e.g., Stanford Cars, Oxford Flowers, CUB-200) would represent an extremely high-impact, direct extension of this work.
