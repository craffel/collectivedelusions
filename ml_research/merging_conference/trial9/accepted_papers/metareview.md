# Meta-Review Summary and Decisions

This document provides a comprehensive summary of the meta-review process and final decisions for the 10 submissions evaluated for the Test-Time Model Merging (TTMM) conference.

---

## 1. Executive Summary of Decisions

Out of 10 submissions, exactly **three (3)** have been selected for acceptance. The selection process was based on a rigorous analysis of each peer review (`review.md`), considering the overall recommendation score, mathematical soundness, experimental thoroughness, practical utility, and originality.

*   **Accepted Submissions:**
    1.  **Submission 2:** *Sparsity-Aware Hybrid Routing with Decisive Test-Time Model Merging under Severe Noise*
    2.  **Submission 4:** *BK-AHR: Bayesian Kronecker-Preconditioned Adaptive Hybrid Routing*
    3.  **Submission 10:** *SAM-TTMM: Sharpness-Aware Test-Time Model Merging for Robust Non-Stationary and Noisy Data Streams*

*   **Acceptance Rate:** 30% (3 out of 10)

---

## 2. Overview of All Submissions

The following table summarizes the evaluations and final decisions for all 10 submissions:

| Submission | Overall Recommendation | Soundness | Presentation | Significance | Originality | Decision |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S1** | 5 (Accept) | Excellent | Excellent | Good | Excellent | *Rejected* (Borderline) |
| **S2** | 5 (Accept) | Excellent | Excellent | Excellent | Excellent | **ACCEPTED** |
| **S3** | 5 (Accept) | Excellent | Excellent | Good | Excellent | *Rejected* (Borderline) |
| **S4** | 5 (Accept) | Excellent | Excellent | Good | Excellent | **ACCEPTED** |
| **S5** | 3 (Weak Reject) | Fair | Excellent | Fair | Good | *Rejected* |
| **S6** | 5 (Accept) | Excellent | Excellent | Good | Good | *Rejected* (Borderline) |
| **S7** | 3 (Weak Reject) | Fair | Fair | Good | Good | *Rejected* |
| **S8** | 3 (Weak Reject) | Fair | Good | Fair | Good | *Rejected* |
| **S9** | 5 (Accept) | Excellent | Excellent | Good | Good | *Rejected* (Borderline) |
| **S10** | 5 (Accept) | Excellent | Excellent | Good | Excellent | **ACCEPTED** |

---

## 3. Detailed Assessment of Accepted Submissions

### 1. Submission 2: Sparsity-Aware Hybrid Routing with Decisive Test-Time Model Merging under Severe Noise
*   **Recommendation Score:** 5 (Accept)
*   **Key Strengths & Contributions:**
    *   **Identifies a Fundamental Trade-Off:** Correctly identifies a critical scientific limitation in spherical metric learning (like CP-AM): spherical normalization ($f / \|f\|_2$) exponentially amplifies input noise in sparse backgrounds, causing representational collapse.
    *   **Strong Mathematical Rigor:** Includes a rigorous mathematical derivation of the "noise-overfitting" failure mode of standard test-time optimization loops (e.g., CL W-Fisher) that minimize predictive entropy under severe noise.
    *   **Innovative, Cohesive Solutions:** Proposes **AHR-SATS-DUN**, which integrates Adaptive Hybrid Routing (AHR) using Hoyer sparsity to select optimal routing metrics, Decisive Under Noise (DUN) temperature scaling, and Entropy-Adaptive Learning Rate (EALR).
    *   **High Practical Utility:** Proposes **Method G**, a closed-form unsupervised calibration heuristic that achieves **64.75%** accuracy out-of-the-box without post-hoc hyperparameter tuning, making it highly deployable for edge IoT.
*   **Meta-Reviewer Rationale:** This submission represents the highest overall quality among all papers. It received a clean sweep of **"Excellent"** across all four dimensions (Soundness, Presentation, Significance, Originality). Its deep mathematical rigor combined with outstanding practical validation makes it an indisputable top choice.

### 2. Submission 4: BK-AHR: Bayesian Kronecker-Preconditioned Adaptive Hybrid Routing
*   **Recommendation Score:** 5 (Accept)
*   **Key Strengths & Contributions:**
    *   **Fully Data-Free and Unsupervised:** Successfully eliminates the dependency on offline calibration data by computing parameter sensitivities on-the-fly using PyTorch's stateless `functional_call` and prediction entropy gradients.
    *   **Theoretically Grounded BN Statistic Fusion:** Section 3.5 provides a formal, elegant mathematical proof showing that the soft Batch Normalization fusion formula is equivalent to exact moment matching under a Mixture-of-Gaussians (MoG) formulation.
    *   **Significant Performance Gains:** Outperforms competitive baselines by substantial margins (+27% absolute gain over CLW-Fisher on Noisy FashionMNIST) while successfully protecting novel domains from feedback loops without requiring any offline calibration data.
    *   **High Empirical Reproducibility:** Re-running the evaluation script yielded performance numbers that match the reported tables exactly.
*   **Meta-Reviewer Rationale:** By resolving the data accessibility and privacy concerns inherent in previous calibration-dependent TTMM approaches, Submission 4 bridges a major gap for edge deployment. The mathematical proof of exact MoG moment-matching for BN statistics fusion is highly elegant and addresses a core weight-merging bottleneck.

### 3. Submission 10: SAM-TTMM: Sharpness-Aware Test-Time Model Merging for Robust Non-Stationary and Noisy Data Streams
*   **Recommendation Score:** 5 (Accept)
*   **Key Strengths & Contributions:**
    *   **Principled Sharpness-Aware Minimization:** Applies SAM directly to weight interpolation parameters to guide optimization toward flatter loss regions and completely bypass the "feedback trap".
    *   **Rigorous Curvature Link:** Mathematically derives a watertight spectral bound showing that minimizing parameter-space sharpness directly bounds input noise sensitivity, providing a theoretical guarantee of noise resilience.
    *   **Discovery of the Preconditioning Stability Trap:** Diagnoses a critical numerical vulnerability in trace-preconditioned TTMM systems (such as BK-CoMerge) under severe noise and resolves it with a robust stability floor ($\epsilon_{\text{stab}} = 0.1$).
    *   **Exceptional Compute Efficiency:** Proves that a single adaptation step ($N_{\text{step}}=1$) of sharpness-aware optimization achieves peak accuracy (54.63%) while running over **$2\times$ faster** than standard 5-step BK-CoMerge.
*   **Meta-Reviewer Rationale:** Submission 10 is an exceptional blend of elegant theory and outstanding engineering diagnostics. Identifying the "Preconditioning Stability Trap" solves a hidden numerical instability that caused previous methods to collapse under noise. Its 1-step optimization provides a highly significant, highly practical solution for real-world edge hardware with tight computational budgets.

---

## 4. Assessment of Non-Selected Submissions

### Borderline Accepted Submissions (Score: 5)
While these submissions were rated 5 (Accept) by reviewers, they were placed below the top 3 due to comparative limitations in significance, originality, or absolute performance:

*   **Submission 1 (VAKP-BC):** Proposes a variance-aware preconditioning framework. While theoretically solid, the evaluation is heavily restricted to a two-expert case, exhibits high hyperparameter sensitivity on the variance weight $\gamma_v$, and evaluates primarily on toy-scale datasets.
*   **Submission 3 (BAR-ACR):** Commendable for an extremely honest self-audit identifying PyTorch's `load_state_dict` autograd breakage and resolving it using `torch.func.functional_call`. However, its performance on novel tasks is low (KMNIST is only 6.25%), and it exhibits higher computational overhead compared to the 1-step efficiency of Submission 10.
*   **Submission 6 (Hyper-TTMM):** Highly original amortized inference approach using hypernetworks to bypass backpropagation entirely (achieving 8.2x speedup). However, it missed concurrent literature on amortized model merging (such as *LoRA.rar* and *MAP*), was restricted to simple architectures, and was sensitive to distance thresholds.
*   **Submission 9 (EGA-BK-CoMerge):** Provides exceptional empirical rigor and thorough mathematical proofs of MoG moment matching and online batch statistics variance. However, the absolute gain of Entropy-Gated Adaptation over standard BK-CoMerge is very modest (from 63.16% to 63.62% overall), putting it below the more transformative contributions of Submissions 2, 4, and 10.

### Rejected Submissions (Score: 3)
These submissions were rejected due to critical technical flaws, empirical contradictions, or overclaimed capabilities identified during review:

*   **Submission 5 (WB-CoMerge):** Suffers from a fundamental scientific disconnect. The central claim is that MoG BN statistic fusion suffers from a "fundamental flaw" of variance inflation causing representational collapse. However, the paper's own empirical results show that the MoG baseline consistently outperforms the proposed Wasserstein-Barycenter method. Furthermore, its claims of noise robustness are heavily exaggerated as it performs at random-guess level (~10%) on noisy segments.
*   **Submission 7 (AdaSim-CoMerge):** Possesses major gaps between the conceptual narrative and actual implementation. The "Adaptive Temperature Scaling" is mathematically a no-op due to a massive base stability floor. The claim of feature-sparsity bimodality to justify the trigger mechanism is factually incorrect (empirically, both MNIST and FashionMNIST have high sparsity due to ReLUs, so the trigger is a no-op). Denoising thresholds also lack scale invariance.
*   **Submission 8 (Self-Supervised Contrastive Prototype Alignment):** Suffers from a severely limited evaluation scale (3-layer SimpleCNN on 28x28 grayscale images), overclaims "Open-World" capability (collapses to near-random 8.12% on novel KMNIST), and relies on a highly simplistic, misleading gradient normalization caricature as its "BK-CoMerge (Approx)" baseline.

---

## 5. Conclusion

The selection of **Submissions 2, 4, and 10** represents a highly robust, technically sound, and complementary set of accepted papers for the conference. Together, they address:
1.  **Metric Space Geometry and Routing Noise:** Resolving background sparsity and spherical metrics under noise (Submission 2).
2.  **Unsupervised On-the-Fly Adaptability:** Eliminating offline calibration data dependencies with exact moment-matching (Submission 4).
3.  **Optimization Stability and Efficiency:** Flat-loss optimization to resolve the feedback trap, while diagnosing preconditioning numerical failures and offering 2x faster, 1-step edge-friendly adaptation (Submission 10).

This collection represents the absolute frontier of Test-Time Model Merging research.
