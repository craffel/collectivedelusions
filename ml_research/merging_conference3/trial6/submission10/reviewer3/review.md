# Peer Review of Submission 10

## 1. Summary of the Paper
The paper tackles the challenge of calibrating dynamic routing heads for multi-task model merging in low-data regimes. Standard dynamic routing heads (such as those using Softmax) suffer from catastrophic overfitting or representational collapse on high-conflict datasets during low-data calibration. 

To address this, the authors propose two main contributions:
1. **Bounded Sigmoidal Router (BSigmoid-Router)**: A Softmax-free routing head that utilizes independent, decoupled sigmoid functions to compute input-dependent merging coefficients. This eliminates the zero-sum competitive constraint inherent in Softmax.
2. **Task-Correlation Prior Regularization (TCPR)**: A regularization technique designed to guide low-data routing head calibration using pre-computed cross-task similarity priors (either parameter-space or representation-space cosine similarities), using off-diagonal centering and signature projection normalization.

The proposed methods are evaluated on a 4-task Vision Transformer benchmark (MNIST, FashionMNIST, CIFAR-10, SVHN) using sub-optimal, computationally-constrained experts. The authors compare their method against a rigorous set of seven baselines. While the decoupled BSigmoid-Router achieves the best overall performance (25.50% joint mean), the proposed TCPR regularizer fails to provide any empirical benefit, either matching the unregularized baseline (by being mathematically dead at small scales) or severely degrading performance at larger scales.

---

## 2. Strengths and Weaknesses

### Strengths
- **Elegant Architectural Simplicity**: The introduction of the Bounded Sigmoidal Router (BSigmoid-Router) is highly elegant and effective. By simply replacing the Softmax normalization with independent, decoupled sigmoids, the authors eliminate the competitive zero-sum bottleneck of dynamic routing. It achieves state-of-the-art performance among routing methods (25.50% joint mean vs. QWS-Merge's 21.80% and BL-Router's 19.10%) without any complex math, extra parameters, or optimization overhead. This is a brilliant example of achieving more with less.
- **Rigorous Evaluation Baseline**: The paper compares its approach against a comprehensive set of seven baselines, including static methods (Task Arithmetic), classical routers (BL-Router with/without regularization), and recent complex SOTA wave-interference methods (QWS-Merge). Evaluation is conducted under strict initialization seed control ($\mathtt{seed=42}$).
- **Intellectual Honesty and Analytical Rigor (Section 4.4)**: The "Empirical Inquest" is a high-quality scientific deconstruction of why Task-Correlation Prior Regularization (TCPR) fails. The authors do not hide the failure of their proposed regularizer; instead, they analyze it deeply and attribute it to three precise physical/mathematical phenomena (Scale Mismatch, Alignment-Interference Paradox, Static-Dynamic Conflict).
- **Exhaustive Hyperparameter Sweeps**: The paper includes a complete logarithmic sweep of $\beta \in [10^{-6}, 10^{2}]$, clearly demonstrating the transition from an inactive regularizer to a destructive one.

### Weaknesses
- **Severe Narrative Alignment Issues (Logical Self-Contradiction)**:
  - There is a jarring contradiction between the front of the paper and the back. 
  - The Title, Abstract, Intro, and Methodology present TCPR as a "simple yet highly effective approach" that "consistently prevents high-conflict task collapse" and "bridges the performance gap to specialist experts."
  - In stark contrast, the Results, Section 4.4, and Conclusion show that TCPR is either mathematically inactive ($\beta \le 10^{-6}$) where it slightly degrades performance (25.20% vs. 25.50%), or actively destructive ($\beta \ge 1.0$) where it collapses performance.
  - It is scientifically and structurally incoherent to write a paper that proposes a method in the abstract as a successful solution, while simultaneously proving in the results section that the method is a failure and should not be used.
- **Highly Exaggerated and Misleading Claims**:
  - The abstract claims that TCPR "bridges the performance gap to specialist experts."
  - In reality, the specialist experts achieve a joint mean of 62.40% while the proposed method gets 25.20% (or 25.50% unregularized), leaving a massive **37.20% absolute gap**. On SVHN, the proposed method gets 10.40% (which is barely above 10.00% random guessing) compared to the specialist's 23.20%. Claiming that this "bridges the performance gap" is highly exaggerated and misleading.
- **Validation on High-Quality Experts**:
  - The base expert models are extremely weak (MNIST 73.20%, SVHN 23.20%).
  - While the authors state that this is intentional to simulate resource-constrained environments, it remains a major limitation. The parameter noise of sub-optimal models may have caused the failure of the prior regularization. The paper fails to investigate whether TCPR would work if the experts were properly converged.

---

## 3. Detailed Dimension Ratings

### Soundness: Fair
The experimental execution, seed control, and baseline selection are technically sound. The mathematical derivations are precise. The analysis of regularizer failures in Section 4.4 is intellectually honest and rigorous. However, the central claims of the paper (that TCPR is a highly effective regularizer that prevents collapse and bridges the gap to specialist experts) are **not adequately supported with evidence and are directly refuted by the authors' own experiments**. Because the central claims are contradicted by the results, the soundness is rated as **Fair**.

### Presentation: Fair
The writing style is professional, and the equations are presented clearly. However, the overall narrative is highly incoherent. Presenting a method as a primary contribution in the abstract, introduction, and title, only to spend the latter half of the paper deconstructing why it fails and is actually a "scientific warning," makes the paper extremely confusing and structurally disjointed. 

### Significance: Good
The paper's significance is high, but not for the reasons the abstract claims. The significance lies in two areas:
1. It demonstrates that the complex "wave-interference" metaphors of SOTA methods like QWS-Merge can be outperformed simply by removing the Softmax competitive bottleneck and using independent sigmoids (BSigmoid-Router). This is a highly valuable, minimalist result for the model-merging community.
2. The deconstruction of why static prior regularizations fail is a highly valuable lesson that prevents future researchers from wasting effort on static weight-similarity constraints and redirects future effort towards dynamic, input-adaptive regularizers.

### Originality: Good
The Bounded Sigmoidal Router (BSigmoid-Router) represents a simple, highly elegant adaptation of sigmoid activations to decouple dynamic model-merging pathways. The analysis in Section 4.4 of the Scale Mismatch, the Alignment-Interference Paradox, and the Static-Dynamic Conflict is original and highly insightful.

---

## 4. Overall Recommendation
**Recommendation: 3 (Weak Reject)**

**Justification**:
This paper has clear, high-quality merits: the **BSigmoid-Router** is an exceptionally elegant, simple, and effective method that achieves state-of-the-art joint multi-task performance among dynamic routing methods. Furthermore, the analysis in Section 4.4 deconstructing the failure of the proposed TCPR regularizer is honest, deep, and scientifically valuable. 

However, the paper's weaknesses currently outweigh its merits due to the **severe logical contradiction and narrative mismatch** between the claims of success in the abstract/introduction and the proven failure in the results/conclusions. The paper cannot be accepted in its current self-refuting state. It requires a major revision to realign the narrative before it can be published and built upon by others.

---

## 5. Constructive Suggestions for the Authors (Actionable Feedback)

1. **Re-frame the Paper's Narrative (Critical)**:
   - Make the **Bounded Sigmoidal Router (BSigmoid-Router)** the **primary proposed contribution** of the paper. Frame it as an elegant, Softmax-free, minimalist architecture that outperforms complex SOTA wave-interference methods (such as QWS-Merge) through pure simplicity.
   - Present the **Task-Correlation Prior Regularization (TCPR)** as a **thorough exploratory investigation / negative result / case study**. Re-frame Section 3.3 and Section 4.4 to show that while incorporating task relationship priors is intuitive, static pre-computed priors are fundamentally incompatible with dynamic sample-level routing because of the Scale Mismatch, the Alignment-Interference Paradox, and the Static-Dynamic Conflict.
   - Realignment Example: Change the title to something like *"Decoupling Routing Pathways in Dynamic Model Merging: Bounded Sigmoidal Routing and the Failures of Static Priors"* and rewrite the abstract to reflect the actual findings.
2. **Tone Down Exaggerated Claims**:
   - Remove or tone down the claim that the proposed methods "bridge the performance gap to specialist experts." Acknowledge that in this challenging, sub-optimal expert regime, a massive 37.20% absolute gap remains, and discuss how future work can attempt to close this gap.
3. **Incorporate Experiments on Converged Experts (or Discuss Limitations)**:
   - Run the same set of evaluations using fully trained, high-quality converged expert models (e.g., MNIST expert >99%, SVHN expert >90%). This will determine if the failure of TCPR was merely due to the parameter noise of sub-optimal experts or if it is indeed a fundamental limitation of static prior regularization.
   - If running these experiments is computationally infeasible, add a dedicated "Limitations" section discussing the sub-optimal expert regime and how parameter noise might influence the regularizer's behavior.
4. **Clarify Figure 1 / The Hyperparameter Sweep**:
   - The discussion of the hyperparameter sweep in Section 4.5 is excellent, but ensure that Figure 1 is clearly labeled and directly references the exact numerical values of the collapsars ($\beta \ge 1.0$) discussed in the text.
