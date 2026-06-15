# Peer Review of the Conference Submission

---

## 1. Summary of the Paper
This paper presents a critical, deconstructive audit of test-time adaptive model merging, exploring the physical and mathematical boundaries of parameter-frugal adaptation. While recent test-time model-merging literature has trended toward high-capacity parameterizations—such as FoldMerge's 2.6M parameter normalizing flows, or AdaMerging's layer-wise scaling parameters—this work steps back and maps the boundaries of parameter frugality.

To isolate these factors, the authors introduce **Barycentric Proximity-Anchored Merging (BPAM)**, a minimalist framework that optimizes only $K$ global task-wise scalars (representing exactly 8 parameters for the standard 8-task image classification benchmark). BPAM constrains the coefficients to a scale-preserving **Convex Barycentric Simplex** via ray-scaling projection, and regularizes adaptation using a closed-form **Mean-Field Proximity Penalty** that anchors the parameters to a uniform centroid. 

Through three distinct spatial and head-adaptation configurations, the authors demonstrate that:
1. **Localized bottleneck adaptation** (merging only the visual projection layer, keeping the rest of the encoder static) collapses performance to **51.38%**, proving that whole-model blending is necessary.
2. Under frozen classification heads, **BPAM-Static** (8 encoder parameters) achieves **69.21%** average accuracy, matching linear **Task Arithmetic** (69.10%), but underperforming static **TIES-Merging** (72.90%). This maps the performance threshold where layer-wise scaling (as in AdaMerging) is indispensable to resolve weight conflicts.
3. Concurrently adapting classification heads (**BPAM-Full**) yields **75.22%** average accuracy. Cross-comparisons with head-tuned static baselines show that **TIES-Merging + Head Tuning** (78.50%) is strictly superior, revealing that joint weight-head adaptation is highly bottlenecked under extreme parameter limits.
4. An ablation study demonstrates that low-parameter adaptation is structurally immune to transductive overfitting under standard calibration streams (calibration vs. unseen split accuracies are nearly identical), rendering the proximity penalty empirically redundant for BPAM-Static, although it serves as a critical geometric anchor under extreme low-data constraints (5 samples per class).
5. Centered Kernel Alignment (CKA) representation similarity checks resolve how SVHN and MNIST expert classifiers perform highly even when their explicit parameter contribution converges to exactly $0.0000$, confirming profound **representation sharing** within the compact, fine-tuned expert loss basin.

---

## 2. Strengths and Weaknesses

### Strengths
- **Exemplary Scientific Honesty and Rigor:** The paper is highly refreshing in its scientific transparency. The authors do not attempt to "oversell" BPAM as a state-of-the-art performance champion. Instead, they frame it as a **boundary probe baseline** designed to map performance thresholds. They openly report and analyze configurations where their constrained model underperforms static baselines, offering deep, deconstructive insights rather than selective, inflated numbers.
- **Deep Literature Situating and Contextualization:** The paper is deeply and broadly aware of the surrounding literature, properly situating itself within the historical context of model merging (e.g., Task Arithmetic, TIES-Merging, DARE, ZipIt!, Model Soups, RegMean) and test-time adaptation (e.g., AdaMerging, SyMerge, FoldMerge, Tent, SHOT, MEMO). It leverages findings from recent critical preprints (e.g., the SAIM audit and the layer-wise model merging sanity check) to motivate its minimalist philosophy.
- **Exhaustive and High-Signal Ablations:** The authors isolate and validate every design choice:
  - Bottlenecking vs. whole-model blending is evaluated via BPAM-Restricted.
  - Simplex projection constraints are evaluated via Unconstrained Scaling.
  - Generalization is split-tested via Calibration (20%) and Unseen (80%) splits.
  - Overfitting risks are tested under extreme low-data constraints (5 samples per class) to show the necessity of the Proximity Regularizer.
  - Sensitivity analysis of the regularization strength ($\beta$) is evaluated across multiple orders of magnitude.
- **Theological Rigor for Empirical Findings:** Rather than leaving the high performance of "0-weight" SVHN and MNIST experts as an unexplained mystery, the authors conduct a rigorous Centered Kernel Alignment (CKA) similarity analysis, proving that the merged model successfully reconstructs specialized representation sub-spaces (such as digit-like shapes from the GTSRB sign expert) due to the high connectivity of the fine-tuning basin.

### Weaknesses
- **Discussion on Optimization Asymmetry can be Formally Expanded:** In BPAM-Full, the 8 weight coefficients and the 388,096 classification head parameters are updated concurrently using a uniform learning rate ($\eta = 10^{-3}$). While the authors mention they extended their framework to support separate learning rates for classification heads (`--head-lr` CLI option), the main text would benefit from a more formal analysis of this optimization asymmetry. A brief discussion of how scaling down $\eta_{head}$ affects multi-task coordinate convergence would elevate the paper's optimization section.
- **Architectural Scope is Restricted to CLIP ViT-B/32:** While CLIP ViT-B/32 is the universal standard in the model-merging literature—making the results directly comparable and compatible with published baselines—the paper would be stronger if the authors discussed how their deconstructive findings generalize to diverse model families (e.g., convolutional backbones like ConvNeXt, or small language models).
- **Simplex Scaling Under Large-Scale Ensembles:** As noted in their limitations section, extending BPAM to extremely large-scale multi-task scenarios (e.g., dozens of models) is an open question, because uniform prior weight $\frac{1}{K+1}$ scales down as $K$ increases, which might overly penalize deviation. Discussing potential solutions (like grouped or hierarchical sub-simplices) in the future work section would be a great addition.

---

## 3. Soundness
* **Rating:** **Excellent**
* **Justification:** The mathematical formulation of the convex barycentric combination is precise and correct. The triangle inequality proof showing that the Frobenius norm is bounded is mathematically sound and provides a strong theoretical justification for preventing activation scale distortion. The defense of ray-scaling projection over standard Euclidean orthogonal projection (PGD) is highly insightful (orthogonal projection has a sparsification effect that can discard expert representations, whereas ray-scaling preserves directional ratios). The split-test generalization verification and extreme low-data checks are highly rigorous.

---

## 4. Presentation
* **Rating:** **Excellent**
* **Justification:** The paper is beautifully written, logical, and exceptionally structured. The introduction and related work sections are highly coherent and build the deconstructive narrative beautifully. Table 1 is meticulously laid out, separating frozen-head (Part A) and active-head (Part B) configurations to ensure a fair and symmetric comparison. The mathematical equations are precise and all terms are clearly defined.

---

## 5. Significance
* **Rating:** **Excellent**
* **Justification:** The paper provides an essential **sanity check** and **educational lesson** for the model-merging community. It alerts researchers to the fact that much of the performance gains in test-time adaptive merging are driven by classification head adaptation, rather than pure weight-space alignment. It demonstrates that applying simple decision-boundary tuning on top of a strong static conflict-resolved model (such as TIES) can easily outperform joint weight-head optimization under low-parameter constraints. By exposing empirical redundancies and mapping exact performance thresholds, this paper will help steer the community toward more principled, mathematically-sound, and parameter-frugal designs, preventing unnecessary architectural overengineering.

---

## 6. Originality
* **Rating:** **Good**
* **Justification:** The algorithm itself (BPAM) is conceptually incremental, consisting of standard components (convex combinations, simplex projection, L2 penalty). However, the paper's *use* of this minimalist model as a boundary-probe baseline and its *deconstructive analysis* (identifying head adaptation dominance, spatial bottleneck limits, and representation sharing) are highly original and insightful. The authors' high level of intellectual honesty and their deep, CKA-guided exploration of representation sharing elevate the work beyond a simple incremental method.

---

## 7. Overall Recommendation
* **Recommendation:** **5: Accept**
* **Synthesis:** This is a technically solid, exceptionally well-written, and scientifically rigorous paper. It presents a vital, deconstructive audit of test-time adaptive model merging, mapping the exact boundary where layer-wise degrees of freedom become necessary and revealing the dominating role of downstream classification head adaptation. It is deeply situated in the literature, properly attributes ideas, and provides a nuanced understanding of the historical context. The paper functions as a crucial educational sanity check that will steer the model-merging community toward more principled, parameter-frugal, and mathematically sound optimization designs. I recommend a strong **Accept (5)**.
