# Empirical and Experimental Evaluation: PAC-Bayes Merge

This section provides a rigorous, empiricist evaluation of the paper's experimental setup, baselines, and results. We identify several critical weaknesses, major data reporting discrepancies, and evidence that contradicts the core claims of the paper.

## 1. Toy Experimental Sandbox & Weak Experts
The entire empirical evaluation is conducted in a highly non-standard "Real-World Projected Representation Sandbox":
- **Johnson-Lindenstrauss (JL) Projections:** Flattened images from MNIST, FashionMNIST, CIFAR-10, and SVHN are projected down to a low-dimensional space of $D_{\text{feat}} = 192$ features using a random matrix, and then passed through a very small, 14-layer deep MLP with hidden width 64. 
- **Extremely Weak Base Experts:** Because of this heavily simplified toy setup, the task-specific experts themselves are exceptionally weak:
  - **MNIST Expert Ceiling:** **78.37%** (a standard MLP on raw MNIST pixels easily gets >98%).
  - **FashionMNIST Expert Ceiling:** **72.27%** (standard models easily achieve >90%).
  - **CIFAR-10 Expert Ceiling:** **25.07%** (barely above random guessing of 10%).
  - **SVHN Expert Ceiling:** **17.57%** (extremely poor, virtually random guessing).
- **Questionable Generalization Claims:** The expert models are barely trained and are highly dysfunctional (especially on CIFAR-10 and SVHN). Merging models that have not learned robust representation spaces and drawing deep theoretical conclusions from their merging behaviors is highly questionable. It is unclear if these findings would scale to any realistic deep learning setup where experts are actually functional (e.g., standard ResNets or Vision Transformers on raw image pixels).

## 2. Major Discrepancy in Results Reporting (Data Inconsistency)
There is a severe, systemic discrepancy between the results presented in the tables (which are supported by the underlying `results.json` data) and the narrative claims in the text (Abstract, Introduction, and Conclusion):
- **Discrepancy in the Main Text:** The Abstract, Section 2.1 (L21), and Section 6 (L6) state that their advanced Fisher-guided non-isotropic PAC-Bayes-FIM Merge achieves a Joint Mean accuracy of **36.13%**, outperforming Static Uniform (**33.57%**), Ties-Merge (**29.68%**), DARE-Merge (**33.24%**), and unconstrained layer-wise tuning (**36.09%**).
- **Actual Main Results (Table 1):** In Table 1 of Section 5 (which matches the raw data in `results.json` exactly), the Joint Mean accuracies are actually:
  - **Static Uniform:** **33.35 $\pm$ 2.46%** (vs. 33.57% in the text)
  - **Ties-Merge:** **29.59 $\pm$ 2.68%** (vs. 29.68% in the text)
  - **DARE-Merge:** **32.76 $\pm$ 2.68%** (vs. 33.24% in the text)
  - **Offline Unconstrained:** **35.51 $\pm$ 2.63%** (vs. 36.09% in the text)
  - **Ours (FIM Deterministic Compiled):** **35.37 $\pm$ 2.84%** (vs. 36.13% in the text)
  - **Ours (FIM Randomized Ensemble):** **35.33 $\pm$ 2.89%**
- **Discrepancy in the Ablation Study (Table 2):**
  Table 2 reports ablation results for different hyperparameters. It states that the default model ($\lambda_{\text{PAC}} = 0.010, \sigma = 0.05$) yields a Joint Mean of **36.09 $\pm$ 2.23%**. However:
  - Table 1 reports the Joint Mean for this exact same model configuration (*Ours (Randomized Ensemble)*) as **35.24 $\pm$ 2.85%**.
  - In `results.json`, the actual data keys for the ablation sweep correspond to Joint Mean averages that are entirely different from Table 2:
    - `ablation_0.001`: **35.37%** (vs. 36.15% in Table 2)
    - `ablation_0.5`: **33.95%** (vs. 34.34% in Table 2)
    - `ablation_sigma_0.01`: **35.36%** (vs. 36.22% in Table 2)
    - `ablation_sigma_0.15`: **35.38%** (vs. 36.08% in Table 2)
- **Conclusion on Data Integrity:** Almost all numbers written in the main text of the paper and in Table 2 are completely mismatched/fabricated compared to the actual experimental data recorded in `results.json`. This is a severe violation of statistical integrity and rigor.

## 3. The Proposed Method Does NOT Beat the Baselines
The central premise of the paper is that unconstrained layer-wise optimization (*Offline Unconstrained*) suffers from transductive overfitting on scarce calibration data, and that PAC-Bayes Merge resolves this, leading to superior generalization. 

However, the empirical evidence completely contradicts this claim:
- **Main Evaluation ($M = 10$):**
  According to Table 1, the unregularized *Offline Unconstrained* baseline achieves a Joint Mean of **35.51 $\pm$ 2.63%**. 
  Meanwhile, the proposed *Ours (Deterministic Compiled)* achieves **35.37 $\pm$ 2.81%**, and *Ours (FIM Deterministic Compiled)* achieves **35.37 $\pm$ 2.84%**. 
  **The unregularized baseline actually outperforms all of the proposed regularized models!**
  The randomized expected model *Ours (Randomized Ensemble)* (**35.24 $\pm$ 2.85%**) is even worse, underperforming the unregularized baseline by **-0.27%** absolute.
- **Extreme Scarcity Sweep ($M = 2$):**
  In the scarcity sweep (`scarcity_results.json` and Figure 3), under extreme data scarcity of $M = 2$ samples per task:
  - *Offline Unconstrained:* **34.16 $\pm$ 3.13%**
  - *Ours (PAC-Bayes):* **33.86 $\pm$ 3.36%**
  - *Ours (PAC-Bayes-FIM):* **33.43 $\pm$ 3.40%**
  Again, the unregularized, unconstrained model outperforms the proposed regularized models by **+0.30%** and **+0.73%** absolute, respectively!
- **Lack of Empirical Justification:**
  An explicit regularizer is supposed to improve generalization over the unregularized baseline. In this paper, across all data scarcity regimes ($M = 2, 5, 10, 20$), the unregularized unconstrained baseline is virtually identical to or better than the proposed regularized method.
  This means that the massive mathematical machinery of PAC-Bayes Merge—including Alquier's bound, KL-divergence derivations, $L_2$ Consensus-Pulling, and FIM prior scaling—provides **zero empirical benefit**, and actually slightly degrades performance. The claims that PAC-Bayes Merge "successfully circumvents transductive overfitting" and "outperforms unconstrained tuning" are entirely false and unsupported by the actual results.
