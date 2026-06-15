# Peer Review Report

## 1. Summary of the Paper
This paper presents a rigorous, methodology-focused sanity check and representational analysis of the layer-wise model merging paradigm. Modern test-time adaptive model merging frameworks (such as AdaMerging and SyMerge) optimize merging coefficients layer-by-layer under the foundational assumption that layer-wise parameter configurations are critical to navigate weight-space interference and capture localized, task-specific representational contributions. 

The authors stress-test this assumption on a pre-trained CLIP ViT-B/32 backbone across four diverse vision classification tasks (MNIST, FashionMNIST, CIFAR-10, SVHN) across three independent random seeds. They evaluate two distinct optimization regimes: a zero-order derivative-free Adaptive 1+1 Evolution Strategy (1+1 ES) and a first-order backpropagation-based Adam Gradient Descent (Adam GD). To examine the physical reality of the learned coefficients, they implement three diagnostic treatments: (1) **Intra-Task Layer Shuffling**, (2) **Task-Wise Spatial Averaging** (collapsing the 13 layer-wise coefficients per task to their mean, reducing parameters by 92.3%), and (3) **Norm-Bounded Noise Perturbations**. Additionally, they employ linear Centered Kernel Alignment (CKA) to evaluate activation-level similarities between the merged models and original task experts.

Their empirical analysis uncovers the **Overfitting-Optimizer Paradox**:
- **Under zero-order search (1+1 ES):** Layer-specificity is shown to be a high-frequency optimization noise artifact. Replacing the optimized layer-wise coefficients with their flat, spatial average (Mean Treatment) acts as a powerful spatial regularizer, actually improving average test accuracy from $85.07 \pm 0.47\%$ to $85.21 \pm 0.11\%$.
- **Under first-order search (Adam GD):** The optimizer discovers a highly precise and delicate layer configuration that is extremely sensitive to shuffling or averaging. However, this delicate structure is shown to be a **transductive overfitting artifact** rather than a generalizable representation. It fails to outperform the unoptimized Task Arithmetic baseline ($84.44 \pm 0.37\%$) on unseen test data while multiplying seed-to-seed variance by 4x.
- **Landscape Flatness:** Both optimizers operate in an exceptionally flat loss basin, tolerating up to 50% relative Gaussian noise on coefficients with negligible decay.
- **CKA vs. Accuracy Decoupling:** While spatial averaging improves average activation similarity, high-level linear CKA alignment can decouple from weight-space decision boundary integrity, with CIFAR-10 accuracy collapsing by over 10% under spatial averaging despite maintaining a high activation correlation ($>0.95$ CKA).
- **Joint Entropy Task-Bias:** Standard joint entropy objectives are shown to have an unformulated optimization bias that sacrifices performance on complex, high-entropy tasks (like SVHN) to minimize joint loss on simpler tasks.
- **Explicit Regularization Solution:** The authors propose explicit Proximity Regularization in the calibration loss, restricting coefficients from drifting excessively from their uniform baseline, which stabilizes performance and prevents transductive drift.

---

## 2. Strengths of the Paper

### Soundness (Excellent)
- **High Statistical Rigor:** The authors run their entire pipeline across 3 independent random trials using distinct seeds, reporting the exact mean and standard deviation of accuracies, which is a commendable standard of empirical science.
- **Rigorous Experimental Control Treatments:** Introducing Shuffling, Spatial Mean, and relative Gaussian noise perturbations as control treatments provides an exceptionally clean and rigorous analytical framework to evaluate learned parameters.
- **Isolated Optimizer Design:** Disentangling zero-order vs. first-order optimization characteristics from the physical properties of the weight manifold is crucial and executed flawlessly.

### Presentation (Excellent)
- **Coherent and Engaging Narrative:** The paper is beautifully written, structuring complex findings into clear, conceptual terms like the "Overfitting-Optimizer Paradox" and the "SVHN Rescue vs. CIFAR-10 Collapse Trade-off".
- **Visual Clarity:** The figures (such as Figure 1 and Figure 2) and tables are clean, complete with standard error bars/deviations, and integrated seamlessly into the text.

### Significance (Good)
- **Essential Community Course-Correction:** This work provides a timely warning to the model merging community, showing that what appeared to be "layer-specific representational contributions" was actually optimization noise or transductive overfitting on small calibration sets.
- **Exposing Objectives Flaws:** Exposing the joint entropy minimization task-bias helps researchers understand why multi-task models can catastrophically fail on harder tasks and guides the development of weighted or scale-normalized objectives.

### Originality (Excellent)
- **Novel Critical Perspective:** While most papers focus on designing increasingly complex and parameter-rich merging schemes, this paper takes a highly original meta-analytical approach, stress-testing foundational assumptions using rigorous control diagnostics.
- **Linking Merging with Test-Time Adaptation Literature:** The paper brilliantly places model merging within the historical context of Test-Time Adaptation (TTA) and transductive overfitting, highlighting a critical connection that has been ignored by prior publications.

---

## 3. Weaknesses of the Paper

### Soundness & Experimental Scope
- **Dataset and Model Scale:** The experiments are conducted on relatively saturated, low-resolution datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) and a relatively small pre-trained backbone (CLIP ViT-B/32, $L=13$ parameter groups). While this is highly appropriate for initial diagnostic studies, it remains an open question whether these findings generalize to modern, massive autoregressive decoder LLMs (e.g., 7B+ parameter networks) or high-resolution complex domains where representational hierarchies are highly distinct and task vectors reside much further from the pre-trained weights in the parameter manifold.
- **Lack of direct global scalar search baseline:** While the "Spatial Mean" serves as an optimized task-wise single-scalar baseline, comparing against a directly optimized global scalar baseline (i.e., directly searching for a single global scalar per task on the calibration split rather than averaging the optimized layer coefficients) would make the baseline comparison more complete.

---

## 4. Contextualization and Related Literature Analysis
This paper excels at situating itself within the broader scientific literature, acknowledging and differentiating from relevant prior works:
1. **Model Merging Foundations:** It builds on linear mode connectivity [Mirzadeh et al., 2020; Frankle et al., 2020], Model Soups [Wortsman et al., 2022], and Task Arithmetic [Ilharco et al., 2022]. It provides proper attribution of ideas to these pioneering works.
2. **Layer-wise Adaptive Merging:** It accurately describes the paradigms of AdaMerging [Yang et al., 2024] and SyMerge [Jung et al., 2025], identifying the specific optimization objectives they use.
3. **Test-Time Adaptation (TTA) and Overfitting:** The paper links model merging directly to TTA literature (such as Tent [Wang et al., 2021], EATA [Niu et al., 2022], and TTA surveys [Liang et al., 2023]), and properly attributes the risk of transductive overfitting to Ash et al. (2020).
4. **Representational Similarity:** It correctly contextualizes the use of linear Centered Kernel Alignment (CKA) [Kornblith et al., 2019] relative to traditional metrics like CCA [Morcos et al., 2018] and SVCCA [Raghu et al., 2017].

From a scholarly perspective, the paper's discussion of the optimization landscape flatness and representational decoupling could be further enriched by citing:
- Classic optimization literature on flat minima and generalization (e.g., Hochreiter & Schmidhuber, 1997, "Flat Minima" or Keskar et al., 2017, "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima") to ground their flatness findings in established theory.
- Studies examining the limitations of activation-level similarity metrics in predicting downstream task performance (e.g., work analyzing the failure modes of CKA or representation-functional decoupling) to strengthen the CKA-Accuracy discrepancy section.

---

## 5. Detailed Suggestions for Authors

### 1. Correct Bibliography Syntax Error
There is a minor syntax error in the bibliography file (`references.bib`) that must be corrected to prevent compilation/indexing failures in standard BibTeX parsers:
- In `@inproceedings{yang2023dataless}`, the author field is currently written as:
  `author={Yang={Enneng} and Shen, Li and others}`
- This contains an extra equals sign and brackets. It should be corrected to:
  `author={Yang, Enneng and Shen, Li and others}`

### 2. Discussion on Calibration Split Size
The calibration set size of 256 images (64 images per task) is highly appropriate to simulate the data-scarce TTA regime. However, the authors should explicitly discuss or model how scaling the calibration dataset size affects this overfitting threshold. For instance, would unconstrained Adam GD's "delicate layer-specificity" generalize better if calibrated on 2048 or 4096 images? Adding a short paragraph discussing this relationship would enrich the methodology section.

### 3. Discussing Sigmoid Parameterization vs. Clamping
In Adam GD, the coefficients are clamped to $[0, 1]$ after each update. Clamping can lead to gradient stagnation or "dead" parameters if the coefficients are driven to the boundaries. Did the authors observe any saturation issues at the boundaries during optimization? Discussing this, or proposing a sigmoid transformation to parameterize the coefficients smoothly within $(0, 1)$, would be a methodologically valuable addition.

### 4. Broaden Flatness and Representation Citations
As suggested in Section 4, grounding the landscape flatness results in classic flat-minima theory and citing studies on representational decoupling would strengthen the intellectual positioning of the paper.

---

## 6. Rating Metrics

* **Soundness:** Excellent
* **Presentation:** Excellent
* **Significance:** Good
* **Originality:** Excellent

## 7. Overall Recommendation
**Recommendation:** **5: Accept**

**Justification:** 
This is a technically solid, exceptionally well-written, and timely paper that provides a much-needed scientific course-correction for the model merging community. By designing and evaluating rigorous control treatments (Shuffling, Spatial Mean, and relative Gaussian noise perturbations) across multiple seeds and optimizers, the authors expose the **Overfitting-Optimizer Paradox** and show that what appeared to be "layer-specific representational contributions" was actually high-frequency optimization noise or transductive overfitting on small calibration splits. The paper's empirical results are thorough and statistically sound, its mathematical formulations are precise, and its connection to Test-Time Adaptation (TTA) and transductive learning literature is highly insightful. Although the evaluation is limited to low-resolution datasets and a relatively small backbone (CLIP ViT-B/32), the authors honestly discuss these limitations and outline a clear blueprint for future research in modern autoregressive decoder LLMs. This paper represents a highly valuable contribution that will influence how researchers evaluate and construct future adaptive model merging frameworks.
