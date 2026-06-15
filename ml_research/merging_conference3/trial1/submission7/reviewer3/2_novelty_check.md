# 2. Novelty Check

## Key Novel Aspects
The primary novelty of this submission is **meta-analytical and diagnostic**. Rather than proposing a new, more complex model-merging algorithm, the paper takes a highly critical, methodology-focused approach to deconstruct the core assumptions of existing state-of-the-art (SOTA) layer-wise model merging methods like AdaMerging and SyMerge.

The key novel insights include:
1. **The Overfitting-Optimizer Paradox:** The conceptualization that the "reality" of layer-specificity depends on the optimizer. Under zero-order search (1+1 ES), layer-specific variation is shown to be high-frequency optimization noise. Under first-order search (Adam GD), layer-specific variation is shown to be a delicate transductive overfitting configuration.
2. **First Application of Representational Similarity (CKA) to Model Merging:** The use of linear CKA to investigate the representation space of merged models. This yields the highly novel insight that activation alignment and downstream classification performance can decouple in fine-grained model merging due to weight-space coordinate shifts disrupting decision boundaries despite high activation correlation.
3. **The Multi-Task Entropy Bias:** A formal and empirical characterization of how joint entropy minimization objectives inherently sacrifice high-entropy complex tasks (like SVHN) in favor of low-entropy simple tasks (like MNIST).
4. **Proximity-Based Regularization vs. Weight Decay:** Distinguishing between standard weight decay (which pulls coefficients to 0.0 and collapses task experts) and proximity-based regularization (which pulls coefficients to a stable functional baseline and prevents overfitting).

---

## The "Delta" from Prior Work
Existing model merging literature (e.g., *Task Arithmetic*, *TIES-Merging*, *AdaMerging*, *SyMerge*) is largely constructive, aiming to show that finer-grained scaling parameters improve performance on downstream test sets. However, these papers typically compare their methods against a fixed, uniform Task Arithmetic baseline (usually with an uncalibrated scale like $\lambda=0.3$) and do not perform any diagnostic controls.

The "delta" of this paper is the introduction of a **rigorous control and diagnostic framework** to stress-test these methods. Specifically:
* **Shuffling and Averaging Controls:** No prior work has validated layer-wise coefficients by shuffling them or replacing them with their spatial average.
* **Dual Optimizer Control:** Prior work usually evaluates a single optimization regime (either first-order or zero-order) without analyzing how the optimizer choice interacts with parameter overfitting.
* **Proximity Regularization:** While $L_2$ regularization is standard, the specific design of pulling coefficients toward a functional uniform baseline ($\lambda=0.3$) rather than $0.0$ represents a simple but physically grounded adaptation for model merging.

---

## Characterization of Novelty
We characterize the novelty of this paper as **significant and highly valuable for the community**. 

While the individual tools used (such as 1+1 ES, Adam, CKA, and $L_2$ regularization) are standard in the deep learning literature, their combination and application as diagnostic probes to expose fundamental flaws in SOTA model merging assumptions is highly creative and insightful. 

The paper acts as a critical **course-correction**. In a sub-field that has rapidly moved toward increasingly overparameterized, complex, and unregularized test-time adaptation schemes, this paper provides a robust, empirically backed warning that much of the reported progress is an artifact of transductive overfitting and learning-rate calibration. It forces researchers to re-evaluate their baselines and methodologies.
