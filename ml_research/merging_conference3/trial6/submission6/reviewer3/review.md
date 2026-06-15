# Peer Review for Conference Submission

## Summary of the Paper
The submission addresses weight-space model merging for neural networks, specifically targeting the "Overfitting-Optimizer Paradox"—a phenomenon where ensembling coefficients optimized on very small calibration datasets (e.g., $M = 10$ samples per task) overfit to transductive noise, causing generalization collapse on test data. To address this, the authors introduce **PAC-Bayes Merge**. The method parameterizes layer-wise merging coefficients using a low-degree polynomial trajectory across network depth, reducing the parameter search space. They model the trajectory coefficients as a randomized Gaussian posterior centered at learnable parameters, and use a spherical Gaussian prior centered at the uniform ensembling baseline. The authors theoretically derive that minimizing Alquier's linear PAC-Bayesian bound yields a quadratic $L_2$ Consensus-Pulling penalty centered at the uniform consensus baseline. They also present a non-isotropic variant weighted by the empirical diagonal Fisher Information Matrix (FIM).

---

## Strengths and Weaknesses

### Strengths
1. **Mathematical Cleanliness of Derivations:** The paper's mathematical derivations in Section 3 are logically coherent. Specifically, the closed-form Kullback-Leibler divergence derivation for isotropic Gaussians and the resulting analytical simplification to a quadratic penalty are mathematically sound.
2. **Clear Conceptual Structure:** The paper's layout is clean, separating the trajectory projection, the PAC-Bayesian bound formulation, and the empirical ablation studies.
3. **Honest Discussion of Theoretical Limitations:** The authors are transparent in the text about the limitations of Theorem 3.1 (the SWA equivalence), acknowledging that distinct classification tasks reside in separate basins of attraction, which violates the single-basin assumption of SWA.

### Weaknesses
1. **Unnecessary Mathematical Obfuscation for a Simple Heuristic:**
   The paper introduces a massive, highly complex theoretical machinery—including McAllester and Alquier generalization bounds, randomized Gaussian posteriors, information-theoretic KL-divergence, and diagonal Fisher Information Matrices. Yet, this extensive mathematical overhead is used solely to justify a highly intuitive, standard heuristic: **a quadratic $L_2$ regularization (weight decay) pulling ensembling coefficients toward a uniform baseline**. Elevating a standard heuristic to a complex learning-theoretic "proof of necessity" represents a severe case of mathematical over-engineering.

2. **The Proposed Method Underperforms the Unregularized Baseline:**
   The entire motivation of the paper is that unregularized optimization under extreme scarcity suffers from severe transductive overfitting ("Overfitting-Optimizer Paradox"), and that their PAC-Bayesian regularizer is a vital defense. However, looking at the actual empirical results in Table 1 and `results.json`:
   * **Offline Unconstrained** (completely unregularized layer-wise tuning) achieves **35.51 $\pm$ 2.63%** Joint Mean accuracy.
   * **Ours (Deterministic Compiled)** achieves **35.37 $\pm$ 2.81%**.
   * **Ours (Randomized Ensemble)** achieves **35.24 $\pm$ 2.85%**.
   In other words, their proposed "PAC-Bayes Merge" actually *underperforms* the unregularized offline baseline.
   Furthermore, in Section 4.3.4, the authors state that a paired t-test shows the difference between their method and the unregularized baseline is "statistically indistinguishable ($p \approx 0.41$)." If the proposed complex regularizer is statistically indistinguishable from, and numerically worse than, a simple unconstrained baseline, then the "Overfitting-Optimizer Paradox" is practically non-existent here, and their proposed complexity offers zero empirical utility.

3. **Failure in Extreme Scarcity ($M=2$):**
   In the extreme scarcity regime ($M=2$), where regularization should theoretically provide the most benefit:
   * **Offline Unconstrained** achieves **34.16 $\pm$ 3.13%** Joint Mean.
   * **PAC-Bayes Merge** achieves **33.86 $\pm$ 3.36%**.
   * **PAC-Bayes-FIM Merge** achieves **33.43 $\pm$ 3.40%**.
   Again, the unregularized baseline outperforms their proposed regularizers. The authors attempt to explain this by pointing to "implicit regularization from optimization dynamics," but this highlights a fundamental redundancy: if simple early-stopped unconstrained tuning naturally and elegantly avoids overfitting, then the massive mathematical overhead of PAC-Bayes Merge is completely unnecessary.

4. **Severe Reporting Discrepancies and Contradictions:**
   There is a critical and unacceptable mismatch between the claims made in the text and the actual data presented in the paper's tables and raw results:
   * **Abstract/Conclusion Claims:** The authors claim that their advanced PAC-Bayes-FIM Merge achieves **36.13%** Joint Mean accuracy, outperforming the unconstrained baseline (**36.09%**).
   * **Actual Table 1 / JSON Data:** FIM Deterministic Compiled achieves **35.37%**, which actually **underperforms** the unconstrained baseline (**35.51%**).
   This is a serious scientific reporting error. The text in the abstract and conclusion reports inflated, fabricated numbers to support a false claim of superiority that is directly contradicted by the data in their own tables and JSON results.

5. **Internal Table Inconsistencies:**
   For the *exact same default parameters* ($\lambda_{\text{PAC}} = 0.010, \sigma = 0.05$):
   * Table 1 reports a Joint Mean accuracy of **35.37%** (Ours Deterministic Compiled).
   * Table 2 (Ablation) reports a Joint Mean accuracy of **36.09%** (with completely different sub-task accuracies, e.g., MNIST is 61.63% in Table 2 vs. 59.72% in Table 1).
   Reporting two completely different sets of results for the exact same method under identical default settings severely damages the empirical reliability and integrity of the paper.

6. **Unnecessary Practical Complexity with Zero Benefit:**
   The paper introduces a complex "Randomized Ensemble" testing mode which draws Monte Carlo samples from the posterior at test time and averages their softmax predictions. This incurs a heavy **$5\times$ forward-pass latency overhead**. However, the Randomized Ensemble mode actually performs *worse* than the simple Deterministic Compiled model (35.24% vs. 35.37%). Introducing high-overhead, randomized ensembling that degrades performance is the antithesis of elegant machine learning engineering.

7. **Highly Artificial, Non-Standard Sandbox:**
   The paper evaluates model merging in a highly customized, artificial setup: projecting MNIST, FashionMNIST, CIFAR-10, and SVHN using random Johnson-Lindenstrauss projection matrices into 192 features, and then feeding them into a custom 14-layer deep residual MLP.
   Because of this aggressive feature projection, the final classification accuracies are extremely poor (e.g., **12.89%** on CIFAR-10 and **15.71%** on SVHN, where random guessing is 10.0%). Drawing conclusions about deep neural network generalization and model merging from a network that barely performs better than random guessing is highly questionable, and holds no relevance to practical model merging on standard pre-trained architectures (like ViTs or LLMs).

---

## Dimensional Ratings

### Soundness
* **Rating:** Poor
* **Justification:** The proposed method is empirically ineffective, actually underperforming the completely unregularized offline baseline in all scarce regimes ($M=2, 5, 10$). The core theoretical motivation of the paper is refuted by their own experiments. Additionally, the SWA equivalence theorem rests on highly unrealistic assumptions (as acknowledged by the authors) and provides no practical scientific utility.

### Presentation
* **Rating:** Poor
* **Justification:** The paper contains severe, unacceptable writing errors and data discrepancies. The abstract and conclusion claim that the proposed method achieves 36.13% and outperforms unconstrained tuning (36.09%), whereas the results table and JSON files show it achieves 35.37% and underperforms unconstrained tuning (35.51%). Furthermore, there are major internal contradictions between the default results reported in Table 1 and Table 2.

### Significance
* **Rating:** Poor
* **Justification:** Practitioners in model merging deal with large, standard models like Vision Transformers (ViTs) and Large Language Models (LLMs). The proposed method is evaluated on an obscure, toy sandbox consisting of JL-projected MNIST/CIFAR-10 images in a custom residual MLP. The resulting accuracies are close to random guessing. No practitioner would adopt a complex trajectory optimizer that adds 5x test-time latency to achieve a performance drop compared to simple unconstrained tuning.

### Originality
* **Rating:** Fair
* **Justification:** The concept of continuous polynomial trajectory parameterization and consensus-pulling was already introduced in Rademacher-Bounded Polynomial Merging (RBPM). The "novelty" of this work is merely transitioning the regularizer from $L_1$ to $L_2$ and wrapping it in dense PAC-Bayesian bound terminology. This is a very incremental contribution dressed up as a major theoretical breakthrough.

---

## Overall Recommendation

* **Recommendation:** 2: Reject
* **Detailed Justification:** 
  This submission suffers from severe technical, empirical, and reporting flaws that make it unsuitable for publication.
  
  First, the paper introduces a highly complex, mathematically dense PAC-Bayesian framework solely to justify a standard, simple $L_2$ regularizer centered at a baseline. This massive theoretical complexity does not translate to any practical or empirical benefit: the proposed regularized method actually *underperforms* (or is at best statistically indistinguishable from) a completely unregularized, early-stopped offline baseline across all calibration data splits ($M=2, 5, 10$).
  
  Second, there are egregious reporting discrepancies that misrepresent the scientific findings. The abstract and conclusion explicitly claim that the proposed method outperforms the unconstrained baseline (36.13% vs. 36.09%), whereas the paper's own Table 1 and raw data show that it underperforms the baseline (35.37% vs. 35.51%). Additionally, Table 1 and Table 2 report completely different Joint Mean accuracies for the exact same default configuration. 
  
  Finally, the evaluation is performed on a highly non-standard, custom toy sandbox (JL-projected low-dimensional manifolds in a 14-layer MLP) where classification accuracies on CIFAR-10 and SVHN are barely above random guessing. The results are highly noisy, and have no relevance to modern model-merging applications.
  
  Given the combination of false claims, internal data contradictions, lack of empirical improvement, and highly artificial evaluation, a rejection is strongly recommended. The authors are encouraged to focus on simpler, more elegant, and effective methods, to evaluate them on standard real-world architectures, and to ensure rigorous scientific reporting where all written text matches the empirical data.
