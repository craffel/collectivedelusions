# Conference Peer Review

## 1. Summary of the Paper
This paper addresses a critical and previously undocumented vulnerability in **dynamic model merging**: catastrophic low-data overfitting of lightweight input routing layers during post-hoc calibration. Dynamic model merging utilizes light gating layers to predict sample-specific weight-merging coefficients on the fly, but calibration data is typically extremely scarce in practical settings (e.g., $B_{cal} \le 64$ samples across tasks). Under such constraints, unconstrained routing parameters overfit to local noise, causing representation-space collapse and a severe failure to generalize.

To resolve this optimization bottleneck, the authors propose **Task-Space Anchor Regularization (TSAR)**, a simple, geometrically grounded classical regularizer. TSAR pre-computes task-specific feature centroids (anchors) from the pre-trained expert representations and incorporates a quadratic distance penalty to keep layer-wise routing weights anchored to these centroids. To prevent multi-task gradient cross-talk where hard, noisy tasks dominate calibration, the authors integrate **Projecting Conflicting Gradients (PCGrad)** into the optimization loop. 

Furthermore, the paper exposes and characterizes **heterogeneity collapse**—direct mathematical coefficient cancellation under mixed-task deployment streams—and resolves it using a **scaled Sigmoid activation** bounded at $[0, 1.5]$ with zero runtime latency or serving overhead. 

Through exhaustive, 5-seed evaluations in a 14-layer representation-space sandbox and on actual Vision Transformers (ViT-Tiny), the authors demonstrate that TSAR + PCGrad outperforms standard $L_2$-regularized routers by **+12.34%** and Static Uniform Merging by **+5.20%** on the sandbox, and by **+13.90%** (and **+23.60%** on raw natural images) on physical weight-space classification head merging.

---

## 2. Main Strengths

* **Outstanding Practical Utility and Engineering Relevance:**
  The paper targets a real-world bottleneck—low-data overfitting of dynamic model-merging routers—and provides a highly actionable, easily deployable solution. The spatial anchor regularizer requires only a few lines of code to implement, adding zero computational or memory overhead during deployment.
  
* **Exceptional Systems Efficiency & Simplification:**
  * **97.4% Gating Parameter Reduction:** By projecting high-dimensional features into a compact $K$-dimensional coordinate space, the single-layer global TSAR router requires only **20 parameters** compared to standard Mixture-of-Experts gating layers which require **768 parameters**, while maintaining competitive accuracy and superior noise robustness.
  * **Exposing Layer-Averaging Collapse:** The authors mathematically prove and empirically verify (Equations 5-7) that under linear batch-averaged routing, multi-layer routers collapse to a single global router at deployment. This is a vital result, proving that practitioners can completely bypass multi-layer routing complexity and reduce parameter footprint by **92.8%** (from 280 parameters to 20) with no loss in accuracy.

* **In-Depth Streaming Audits:**
  The characterization of **heterogeneity collapse** under mixed-task deployment streams is a highly significant contribution for distributed production servers. The proposed non-negative **scaled Sigmoid activation** resolves this collapse with mathematically elegant, absolute **zero runtime latency or memory overhead**, making dynamic routing a production-viable alternative to static parameter averaging.

* **Exceptional Scientific Honesty and Transparency:**
  The authors demonstrate exemplary scientific integrity:
  * They explicitly detail and prove that head-level classification merging is mathematically equivalent to output-level logit ensembling.
  * They highlight that their first physical ViT experiment is restricted to classification heads on top of frozen backbones and uses synthetic image manifolds to isolate representation noise.
  * They openly discuss the $O(K)$ computational complexity scaling bottleneck of PCGrad.
  * They proactively address every single one of these limitations with rigorous, multi-seed appendix experiments (realistic expert sweeps, raw natural image evaluations, and stochastic PCGrad sampling).

* **Empirical and Statistical Rigor:**
  Evaluating all experiments across **5 independent random seeds** (reporting both mean and standard deviation) rules out coordinate luck, providing highly reliable and robust empirical results.

---

## 3. Main Weaknesses & Areas for Improvement

* **Validation on Deep Internal Parameter Fusion:**
  As the authors honestly acknowledge, head-level classification merging is mathematically identical to output-level logit ensembling. To elevate this work to a major milestone in deep parameter fusion, TSAR must be validated on **deep internal weight-space merging** (e.g., intermediate self-attention projections and MLP weight matrices) where parameter-level fusion and logit ensembling diverge fundamentally due to intervening non-linear activations. The unique challenges of internal merging (permutation-routing coupling and non-linear coordinate coupling) are discussed in Appendix A, but this critical limitation and future direction should be given more prominence in the main text.

* **Main Text Integration of Compelling Appendix Findings:**
  Several of the most compelling and practically valuable findings are currently relegated to the Appendix:
  * **Random Gaussian Projections (Appendix B):** The finding that data-independent Random Gaussian projection consistently and substantially outperforms PCA under extreme scarcity ($B_{cal} \le 128$) due to zero-variance distance preservation (Johnson-Lindenstrauss Lemma) is highly impactful.
  * **Standard MoE Gating Comparison (Appendix D):** The 97.4% parameter reduction over standard MoE gating layers.
  * **Natural Image Evaluations (Appendix C.1):** The spectacular **+23.60%** absolute gain over Static Uniform Merging on actual natural images.
  * *Recommendation:* Integrating high-level summaries of these three findings directly into the main text's Section 4 would significantly strengthen the paper's core presentation.

---

## 4. Detailed Evaluation Ratings

### Soundness: Excellent
The mathematical proofs (such as the layer-averaging collapse in Equations 5-7) are flawless and highly rigorous. The coordinate approximations and uncentered projections are physically and empirically justified, and the authors are incredibly transparent about the boundary conditions and limitations of both simulated and physical experiments. 

### Presentation: Excellent
The paper is exceptionally well-written, clearly structured, and easy to follow. The transition from the simulated sandbox to physical ViT validation is natural and well-contextualized. Figures 1, 2, 3, and 4 are highly legible, with clear trends, error bars, and descriptive captions.

### Significance: Excellent
By making dynamic model merging stable under as few as 16 calibration samples and requiring only 20 gating parameters, TSAR significantly advances the feasibility of lightweight on-device customization and streaming adaptation. The scaled Sigmoid router completely bypasses serving bottlenecks under heterogeneous deployment, presenting a highly significant contribution to high-throughput production servers.

### Originality: Excellent
While distance-based centroids are a classic concept, integrating them as a geometrically grounded spatial anchor regularizer within the model-fusion routing framework is a highly original, refreshing, and systems-efficient contribution. The mathematical proofs of layer-averaging collapse and the formalization of heterogeneity collapse are both highly original and valuable to the community.

---

## 5. Overall Recommendation

**Score: 5: Accept**

**Justification:**
This is an incredibly complete, rigorous, and highly practical paper that addresses a major open bottleneck in dynamic model merging. The authors provide a flawless mathematical proof of layer-averaging collapse, formalize and resolve a critical production deployment failure (heterogeneity collapse), and achieve spectacular empirical gains over standard baselines. The paper sets an outstanding standard for scientific honesty, detailing its own limitations and proactively resolving them with thorough, multi-seed appendix evaluations (including actual physical Vision Transformers on raw, uncurated natural images). The systems-level efficiency benefits (97.4% parameter footprint reduction and zero-overhead streaming) make this paper highly significant and an easy accept.

---

## 6. Detailed Comments, Queries, and Suggestions

1. **Deeper Intermediate Layer Merging & Adaptive Regularization:**
   When scaling TSAR to deep internal layers, as discussed in Appendix A, the intermediate features are non-linearly coupled. Have the authors considered whether the spatial regularizer strength ($\lambda_{anchor}$) should be adaptively scaled based on the layer's depth? For instance, should early layers (which capture general low-level features) have stronger anchoring than deeper layers (which represent highly specialized multi-task features)?
   
2. **Extending Scaled Sigmoid to Text-Prompt LLM Anchors:**
   The zero-shot, text-prompt task anchors proposed in Appendix A represent an elegant path toward completely data-free dynamic merging. Under this data-free regime, how would the scaled Sigmoid activation router behave when sequence length vary significantly in LLM deployments? Would average-pooling sequence-level representations prior to projection fully prevent sequence-length-dependent coordinate distortion under uncentered projections?

3. **EMA Momentum Optimization under Non-Stationary Streaming:**
   The online EMA anchor-tracking scheme under linear coordinate drift (Appendix E) is highly compelling. What is the sensitivity of the optimal momentum factor ($\beta=0.20$) to the speed of the linear drift? If the environment transitions rapidly (high drift velocity), does the EMA tracking lag significantly, and is there an analytical guideline to dynamically adapt $\beta$ on-the-fly?
