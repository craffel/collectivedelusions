# Meta-Review Summary and Decisions

## 1. Overview of the Meta-Review Process
We conducted a comprehensive and rigorous meta-review process for the 10 submissions to the Test-Time Model Merging (TTMM) track of the merging conference.
Each submission was evaluated based on the peer reviews (`review.md`) in its respective subdirectory (`submission1` through `submission10`).
The reviews were evaluated on:
1. **Soundness:** Technical correctness, mathematical rigor, experimental design, and support for claims.
2. **Presentation:** Writing quality, clarity, structure, and adherence to style and template guidelines.
3. **Significance:** Real-world impact, practical utility, scalability, and relevance to the model merging and test-time adaptation communities.
4. **Originality:** Novelty of concepts, creative synthesis of methods, and uniqueness of insights.
5. **Score/Recommendation:** Overall recommendation (ranging from 3: Weak Reject to 6: Strong Accept).

Our goal was to select the **three most positive and high-quality submissions** for acceptance into the conference, while ensuring the decisions were backed by both final scores and thorough review contents.

---

## 2. Summary of Submissions and Recommendations

| Submission | Title / Short Description | Recommendation | Soundness | Presentation | Significance | Originality | Decisions |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Submission 1** | MD-OPA with Dynamic Centering | 3: Weak Reject | Good | Excellent | Good | Good | Rejected |
| **Submission 2** | L-GMM: Directional vMF Mixture Models | 5: Accept | Excellent | Excellent | Good | Excellent | Not Selected |
| **Submission 3** | Rob-OW: Robust Open-World TTMM | 5: Accept | Excellent | Excellent | Good | Good | Not Selected |
| **Submission 4** | TMC-Fisher Coefficient & Fisher Adaptation | 4: Weak Accept | Excellent | Excellent | Good | Good | Rejected |
| **Submission 5** | DF-OW-TTMM: Data-Free Open-World TTMM | 5: Accept | Excellent | Excellent | Good | Excellent | Not Selected |
| **Submission 6** | D-TT-Fisher-DFC: Dynamic Fisher & DFC | 5: Accept | Excellent | Excellent | Good | Excellent | Not Selected |
| **Submission 7** | HAT-Merge: Heterogeneous Streams | 4: Weak Accept | Good | Excellent | Good | Good | Rejected |
| **Submission 8** | **CLW-Fisher: Co-acting Layer-Wise Fisher** | **6: Strong Accept** | **Excellent** | **Excellent** | **Excellent** | **Excellent** | **ACCEPTED** |
| **Submission 9** | **KT-Fisher: Kronecker Trace-Based Fisher** | **5: Accept** | **Excellent** | **Excellent** | **Good** | **Excellent** | **ACCEPTED** |
| **Submission 10**| **DF-Bayes-TTMM: Bayesian TTMM with Soft BN** | **5: Accept** | **Excellent** | **Excellent** | **Excellent**| **Excellent** | **ACCEPTED** |

---

## 3. Rationale for Accepted Submissions

We selected **Submissions 8, 10, and 9** as the three accepted papers. The reasons for their selection are detailed below:

### 1. Submission 8: "CLW-Fisher: Co-acting Layer-Wise Fisher Adaptation for Test-Time Model Merging"
*   **Recommendation:** 6: Strong Accept (Highest Score in the batch)
*   **Ratings:** Soundness: Excellent, Presentation: Excellent, Significance: Excellent, Originality: Excellent
*   **Key Contributions:**
    *   **Self-Calibrated Temperature Scaling (SCTS):** A parameter-free, scale-invariant routing temperature formulation that naturally uniformizes routing priors on novel domains and stabilizes them on known tasks.
    *   **Prior-Guided Parameter Initialization (PG-Init):** Translates the routing prior directly into initial merging parameters to accelerate optimization convergence.
    *   **Co-acting Layer-Wise Fisher Adaptation (CLW-Fisher):** Parameterizes each layer's merging coefficient as a combination of a global consensus logit and layer-specific offsets, using Consensus Coherence Regularization to prevent layer-wise representational misalignment under noise.
*   **Why Accepted:** This paper is technically flawless and highly innovative. It solves three critical bottlenecks of TTMM (routing temperature flatlining under noise, optimization resets to midpoints, and layer-wise representational drift) with elegant mathematical formulations. Its empirical validation includes four thorough sensitivity sweeps (batch size, prior regularization strength, SCTS sensitivity, coherence regularization strength), and it provides a rigorous multi-billion parameter transformer scaling analysis in the Appendix. It represents an exemplary "Strong Accept."

### 2. Submission 10: "Data-Free Bayesian Test-Time Model Merging with Soft Batch Normalization Buffer Fusion" (DF-Bayes-TTMM)
*   **Recommendation:** 5: Accept
*   **Ratings:** Soundness: Excellent, Presentation: Excellent, Significance: Excellent, Originality: Excellent (Only paper with all-Excellent ratings among the 5s)
*   **Key Contributions:**
    *   **Soft BN Buffer Fusion:** Interpolates Batch Normalization running statistics using posterior merging coefficients, backed by a rigorous mathematical proof (Proposition 3.2) showing that it perfectly reconstructs the first and second moments of the underlying mixture distribution.
    *   **Bayesian MoE Routing:** Solves the "feedback trap" inherent in unconstrained entropy minimization using a dynamic Bayesian mixture-of-experts formulation to compute continuous soft-routing posteriors based on predictive confidence.
    *   **Data-Free TTA:** Eliminates offline calibration data dependencies by computing "Test-Time Fisher" (TT-Fisher) sensitivities dynamically on target test streams.
*   **Why Accepted:** In addition to its theoretical elegance and rigorous moment-matching proof, this paper stands out as the only submission that proactively addresses scalability concerns by evaluating on a large-scale ResNet-18 model using CIFAR-10 streams (high-dimensional color images). This multi-scale empirical validation, combined with its data privacy and adaptation stability benefits, makes it a highly significant and robust contribution deserving of acceptance.

### 3. Submission 9: "KT-Fisher: Kronecker Trace-Based Test-Time Fisher Preconditioning for Robust Open-World Model Merging"
*   **Recommendation:** 5: Accept
*   **Ratings:** Soundness: Excellent, Presentation: Excellent, Significance: Good, Originality: Excellent
*   **Key Contributions:**
    *   **Kronecker Trace Sensitivity Approximation:** Proposes a mathematically elegant trace-based approximation ($\bar{F}_w \approx \frac{\operatorname{Tr}(G_l) \operatorname{Tr}(A_{l-1})}{|w_l|}$) to collapse high-fidelity Kronecker-factored curvature (KFAC) factors into scalar layer-wise preconditioning terms, avoiding computationally prohibitive matrix inversions.
    *   **Data-Free Sensitivity Estimation:** Estimates sensitivity on-the-fly during a single forward-backward pass using only target streaming activations and gradient norms.
    *   **Periodic Preconditioning:** Implements and sweeps over periodic update intervals to eliminate up to 29 out of 30 backpropagation steps, reducing CPU latency by over 14% while maintaining identical classification accuracy.
*   **Why Accepted:** This paper bridges a critical gap in weight-space TTA by making structured second-order preconditioning computationally viable at test-time. The Kronecker trace derivation (Lemma 5.1) is exceptionally elegant, and the author's proactive resolution of the test-time backpropagation cost via Periodic Preconditioning was highly praised by the reviewer as enhancing the completeness and rigor of the manuscript. It has fewer unaddressed weaknesses compared to other 5-rated submissions.

---

## 4. Analysis and Comparison of Non-Selected Submissions

While several other submissions received "5: Accept" ratings, they had more notable unaddressed weaknesses or were less complete compared to the accepted trio:

*   **Submission 2 (L-GMM):**
    *   *Strengths:* Beautifully characterizes and resolves the "Euclidean Bug" in deep representations using directional von Mises-Fisher (vMF) mixture modeling.
    *   *Unaddressed Weaknesses:* It relies on a critical "startup in-distribution assumption" for dynamic calibration (assuming the first batch is always in-distribution), which fails if the stream starts with an OOD domain. It is also sensitive to the concentration parameter $\sigma^2$ and is evaluated only on simple grayscale MNIST-like datasets.
*   **Submission 3 (Rob-OW):**
    *   *Strengths:* Rigorous proof of activation mismatch bounds (Proposition 1) and massive overall accuracy improvements (+21% absolute).
    *   *Unaddressed Weaknesses:* It has more combinatorial/incremental conceptual novelty (synthesizing existing methods like DR-Fisher, IGGS-OW, and AdaBN). It also suffers from a performance floor on completely unseen domains (~10-13% accuracy) and relies on several highly sensitive, uncalibrated hyperparameters.
*   **Submission 5 (DF-OW-TTMM):**
    *   *Strengths:* Proposes dynamic TT-Fisher with pseudo-labels and differentiable BN buffer merging with autograd-detached coefficient weights, backed by a convergence proof.
    *   *Unaddressed Weaknesses:* The scale of evaluation is limited to 2D CNNs on simple grayscale datasets. It also suffers from applicability questions for LayerNorm-based transformer architectures, offline threshold calibration dependencies, and minor hyperparameter inconsistencies in the manuscript (default $\beta = 1.0$ vs. optimal $\beta = 0.30$).
*   **Submission 6 (D-TT-Fisher-DFC):**
    *   *Strengths:* Elegant mathematical proof of scale invariance for Dynamic Feature Centering (DFC) and a highly honest discovery of the "Routing Overshadowing Trap."
    *   *Unaddressed Weaknesses:* DFC is highly dependent on batch size; very small batch sizes (e.g., $B \le 4$) will cause extremely high feature variance and degrade performance. It is also vulnerable to class imbalance (which biases the batch mean) and non-scaling/non-linear physical corruptions.
*   **Submission 7 (HAT-Merge):** (Score: 4: Weak Accept)
    *   *Unaddressed Weaknesses:* Extremely low default novelty detection rate (0.56% default), near-random accuracy on the novel task (~17% default), and significant inference overhead due to multiple forward passes required by dynamic sub-batch partitioning.
*   **Submission 4 (TMC-Fisher):** (Score: 4: Weak Accept)
    *   *Unaddressed Weaknesses:* Extremely low absolute accuracy under Contrast Shift (~16-21%), and its dynamic Fisher preconditioning has zero effect under Contrast Shift because large learning rates cause coefficients to saturate at the simplex boundaries.
*   **Submission 1 (MD-OPA with Dynamic Centering):** (Score: 3: Weak Reject)
    *   *Unaddressed Weaknesses:* MD-OPA consistently underperforms existing state-of-the-art baselines in classification accuracy (by up to 7%). Robust novelty detection is entirely driven by Dynamic Centering, which could easily be integrated into any other baseline, making MD-OPA redundant. Evaluated only on short toy streams and requires manual, corruption-specific novelty thresholds.

---

## 5. Conclusion
The selection of **Submissions 8, 10, and 9** represents a balanced and highly principled combination of:
1.  **Extreme Innovation and Theoretical Rigor** (CLW-Fisher / SCTS in Sub 8).
2.  **Practical Scalability and Moment-Matching Precision** (Bayesian Soft BN in Sub 10, evaluated on ResNet-18).
3.  **Optimization/Curvature Efficiency** (Kronecker Trace Preconditioning in Sub 9).

This ensures that the merging conference accepts work that is mathematically robust, empirically validated, and highly useful for real-world Test-Time Model Merging deployments.
