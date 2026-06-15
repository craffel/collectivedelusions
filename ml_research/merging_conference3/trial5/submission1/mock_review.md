# Mock Review: Riemannian Curvature-Regularized Test-Time Model Merging

---

## 1. Summary of the Paper

This paper addresses a critical challenge in adaptive test-time model merging (TTA-MM), which the author terms the **Overfitting-Optimizer Paradox**. During online deployment, when merging coefficients are optimized online via gradient-based unsupervised objectives (such as minimizing the Shannon entropy of predictions) to adapt to local data streams, the optimization trajectory is highly vulnerable to local transductive noise. This results in high-frequency spatial oscillations of coefficients across the network depth, deforming intermediate representations and causing severe performance collapse.

To resolve this paradox, the authors introduce **Riemannian Curvature-Regularized Test-Time Model Merging (RCR-Merge)**. The framework models the parameter space as a Riemannian manifold where distance is locally scaled by the trace of the pre-trained base model's diagonal Fisher Information Matrix (FIM), computed offline on a minuscule calibration set ($|D_{\text{cal}}| = 64$). RCR-Merge implements a novel spatial regularizer, **Riemannian Curvature-Weighted Total Variation (RCR-TV)**, which penalizes coefficient variations between adjacent layers, scaling the penalty dynamically by the geometric mean of their base curvatures ($\sqrt{c_l c_{l-1}}$). This creates an analytical coordinate barrier that shields highly sensitive bottleneck layers (early and late blocks) from representation-destroying fluctuations while permitting adaptive specialization in flat, robust middle layers. 

To handle unsupervised hyperparameter tuning, the authors propose **Gradient Norm Balancing (GNB)**, which dynamically scales the spatial and absolute anchoring penalties at step 0 by evaluating gradient norms on a worst-case spectral perturbation (derived from the highest-frequency eigenvector of the 1-D graph Laplacian).

RCR-Merge is evaluated on a rigorous 30-seed simulation study over synthetic Coupled Model II and Stage-wise Modular Transition Landscapes, and validated on full-scale real-world pilot studies using `bert-base-uncased` (NLP) and `vit-base-patch16-224` (CV).

---

## 2. Main Strengths

1. **High Conceptual Originality:** The paper is highly innovative. While model merging is an active area of research, most methods apply static uniform scales or rigid global projection subspaces (like PolyMerge). AdaMerging was the first to adapt merging coefficients online using test-time entropy minimization but ignored spatial oscillation or noise overfitting. RCR-Merge is the first framework to identify the Overfitting-Optimizer Paradox and propose a second-order Riemannian solution. Repurposing the pre-trained base model FIM as a spatial coordinate weighting factor for online test-time optimization of layer-wise coefficients is highly creative.
2. **Exceptional Mathematical and Theoretical Rigor:** The mathematical foundations are exceptionally mature and detailed. The authors provide elegant derivations of the pullback metric tensor diagonal approximation, the Taylor error bound of the static metric, and a beautiful spectral graph theory analysis (modeling the spatial TV penalty as a Laplacian smoothing low-pass filter).
3. **Rigorous and Complete Mathematical Proofs:** The paper includes full proofs for both **Lemma 3.3 (Coordinate-Level Spatial Barrier)** and **Theorem 3.4 (Representation Drift Bounding)**. Connecting coordinate-level variations directly to intermediate activation representation drift provides a physically grounded justification for the spatial regularizer.
4. **Circularity-Breaking Experimental Design:** To address potential circularity concerns (since the simulator's standard covariance inverse $\boldsymbol{\Sigma}^{-1}$ has a tridiagonal graph Laplacian structure similar to the regularizer), the authors evaluate all methods on a completely decoupled **Decoupled Isotropic Euclidean Metric** ($\boldsymbol{\Sigma} = \mathbf{I}$). This level of scientific rigor is highly commendable and rarely seen.
5. **Exposing Polynomial Baseline Failure Modes:** Evaluating PolyMerge on a **Stage-wise Modular Transition Landscape** with discrete stage boundaries successfully demonstrates the severe limitations of rigid global polynomial constraints, which suffer from global curve deformation (Runge's phenomenon).
6. **Multimodal Real-World Pilot Studies:** Rather than relying purely on simulations, the authors execute pilot studies on full-scale real-world networks—`bert-base-uncased` and `vit-base-patch16-224`—using functional autograd (`torch.func`). These pilots empirically confirm representation collapse under unconstrained AdaMerging and successful stabilization via RCR-Merge.

---

## 3. Weaknesses and Constructive Critique

1. **Scale and Modality of Real-World Evaluation (Gap):**
   - While the BERT-Base and ViT-B/16 pilot studies are excellent additions that show functional autograd working on 100M-scale models, they remain **small-scale proof-of-concept pilot studies** evaluated on relatively small simulated test streams.
   - To fully confirm real-world utility and meet the highest standards of top-tier machine learning conferences (such as ICML, NeurIPS, or ICLR), RCR-Merge should be evaluated on standardized, high-dimensional streaming benchmarks, such as **ImageNet-C** and **ImageNet-R** for vision, or **GLUE/MMLU streaming domain shifts** for language.
2. **Lack of Baselines in Real-World Pilots:**
   - In the real-world pilot studies, the authors only compare **Unconstrained AdaMerging** against **RCR-Merge**.
   - They do not implement or evaluate other baselines, such as PolyMerge or flat TV-regularized AdaMerging, on the actual BERT or ViT weights.
   - Verifying whether PolyMerge actually suffers from the predicted curve deformation on real physical networks is a critical gap.
3. **Presentation and Compilation Error (Missing Figure LaTeX Block):**
   - In `04_experiments.tex` (L101 and L104), the text explicitly references `Figure~\ref{fig:visualizations}` (left and right) for qualitative visualizations of coefficient trajectories and sensitivity sweeps over $\beta$.
   - However, **there is no `\begin{figure}` block with label `fig:visualizations` in the entire LaTeX source code!**
   - The image files `rcr_beta_sensitivity.png` and `rcr_merge_trajectory.png` exist in the directory but are completely omitted from the compiled document, resulting in undefined reference warnings and a major presentation gap for readers.
4. **Conceptual Oversimplifications in Theorem and Model Assumptions:**
   - **Theorem 3.4 output drift bound:** The bound depends on the factor $\Lambda^{L-l}$ (Eq. 29). For deep neural networks (e.g., $L=12$ in BERT, $L=32$ in LLaMA), even a mild Lipschitz constant $\Lambda > 1$ causes $\Lambda^L$ to explode exponentially. This makes the global output representation drift bound extremely loose (practically vacuous) for deep networks. Presenting it without this transparency can mislead readers into thinking it is a tight quantitative guarantee, which impairs scientific clarity.
   - **Isotropic Block-Diagonal Trace Simplification:** The block-diagonal trace scalar assumption ($c_l$) groups attention and MLP blocks under a single scalar. Although justified empirically in BERT, K-FAC remains purely theoretical and is not implemented or evaluated in either the simulator or the real-world pilots, leaving the anisotropic claims purely theoretical.

---

## 4. Questions and Minor Suggestions for the Authors

1. **LaTeX Compilation Warning:** Please add the missing `\begin{figure}` block for `fig:visualizations` in `04_experiments.tex` to include the existing image files `rcr_beta_sensitivity.png` and `rcr_merge_trajectory.png` so that readers can view the qualitative trajectories and sensitivity sweeps.
2. **Loose Bounds Discussion:** Could the authors add a brief discussion in Section 3.4 acknowledging that the global output representation drift bound in Theorem 3.4 is primarily of **qualitative and conceptual value** (demonstrating the causal mechanism of curvature scaling) rather than a tight quantitative bound, due to the exponential growth of Lipschitz constants with depth?
3. **Pilot Gaps:** Have the authors considered running PolyMerge and flat TV-regularized baselines on the real-world BERT-Base or ViT-B/16 pilot architectures? Showing that PolyMerge actually struggles on BERT-Base or ViT-B/16 as predicted by the Modular simulator experiments would greatly strengthen the paper's claims.
4. **Long-Term Adapation Drift:** In Section 4.4, the proposed Threshold-Triggered Curvature Re-estimation is evaluated on the simulator over 2,000 steps. Could this triggering mechanism be tested or discussed in the context of real-world networks during prolonged streaming scenarios? Does the real BERT-Base FIM drift significantly past 50 steps of TTA?

---

## 5. Final Ratings

* **Soundness:** **Excellent (Good to Excellent)**
  The mathematics is highly mature, correct, and extensively detailed. The proofs are mathematically solid. The block-diagonal trace and static evaluation approximations are standard, well-justified simplifications.
* **Presentation:** **Good**
  The paper is exceptionally well-written, with high mathematical maturity, structured flow, and elegant explanations. However, the rating is dragged down by a critical presentation error (the missing figure LaTeX block).
* **Significance:** **Excellent**
  Model merging is a rapidly growing area in deep learning. Online test-time adaptation of merging coefficients is the natural next frontier. Grounding test-time optimization trajectories in pre-trained second-order geometry is a highly elegant, novel, and impactful contribution.
* **Originality:** **Excellent**
  RCR-Merge is the first framework to identify the Overfitting-Optimizer Paradox and propose a second-order Riemannian solution. The GNB unsupervised self-scaling mechanism is a highly creative contribution.

### **Overall Recommendation: 5 (Accept)**
This is an exceptionally strong, mathematically rigorous, and scientifically complete paper. It introduces a novel second-order geometric framework (RCR-Merge) to resolve the Overfitting-Optimizer Paradox in adaptive model merging, backed by complete theoretical proofs and rigorous experimental validation (including circularity-breaking decoupled evaluations and real-world pilots). While there are minor gaps in the scale of real-world benchmarks and a minor LaTeX compiling error, the overall contributions easily clear the acceptance bar for top-tier machine learning conferences. I strongly recommend **Accept** and encourage the authors to address the minor suggestions and presentation errors to make the manuscript flawless.
