# Peer Review of RegCalMerge: Calibrated \& Regularized Test-Time Model Merging

## Summary of the Paper
This paper presents a rigorous empirical deconstruction of test-time adaptive model merging, specifically focusing on the state-of-the-art framework **AdaMerging** (ICLR 2024). The authors identify and analyze two critical, under-reported failure modes in existing adaptive test-time model merging:
1. **The Overfitting-Optimizer Paradox (Transductive Overfitting)**: Fine-grained, layer-wise merging coefficients optimized on small test-time calibration streams overfit to the local statistics of those small batches rather than capturing genuine localized representational interactions.
2. **Sacrificial Task Bias**: Under standard joint entropy minimization, the optimizer systematically degrades or "sacrifices" complex, high-entropy tasks (e.g., SVHN) to prioritize easier, low-entropy domains.

To resolve these limitations, the paper introduces **RegCalMerge**, a unified framework comprising:
- A core **Calibration Engine (CalMerge)** combining **Class-Capacity Normalization (CCN)** and **Scale-Normalized Entropy Weighting (SNEW)** to balance the optimization landscape across heterogeneous domains.
- An optional structural stabilizer called **Elastic Spatial Regularization (ESR)**, which applies a dual **Proximity Penalty ($\beta$)** and **Spatial Deviation Penalty ($\gamma$)** to smooth layer-wise coefficients and constrain parameter drift.

Evaluating these components across 4 visual classification tasks (MNIST, FashionMNIST, CIFAR-10, SVHN) using a CLIP ViT-B/32 backbone, the authors demonstrate that their unregularized calibration engine, CalMerge, achieves a state-of-the-art Joint Mean accuracy of **61.82\%** and elevates SVHN accuracy to **32.03\%**, successfully resolving the sacrificial task bias.

---

## Strengths

1. **Outstanding Analytical and Diagnostic Insights (Major Strength)**:
   The introduction of the **spatial shuffling diagnostic** is a highly original, elegant, and powerful contribution. By showing that randomly shuffling optimized layer-wise coefficients across the network's layers preserves nearly 95% of the Joint Mean performance (60.94% vs. 61.62%), the authors provide direct, irrefutable proof of the Overfitting-Optimizer Paradox. Exposing that fine-grained layer-wise test-time adaptation is largely a transductive overfitting illusion of parameter-drift is a brilliant contribution that demystifies and simplifies our understanding of adaptive model merging.

2. **Simple, Elegant, and Hyperparameter-Free Calibration**:
   The introduction of **Scale-Normalized Entropy Weighting (SNEW)** is exceptionally elegant. By simply scaling each task's entropy by its baseline uniform entropy at step 0 (initialization), the authors successfully balance the gradient contribution of different domains without introducing any training overhead or new hyperparameters to tune. This is a model of effective, simple, and direct problem-solving.

3. **Exceptional Empirical Rigor and Scientific Transparency**:
   The authors have executed their empirical evaluation with a very high level of discipline. Running multiple random seeds, conducting dense 2D hyperparameter sweeps ($\beta \times \gamma$), evaluating derivative-free optimizers (1+1 ES), and explicitly demonstrating the deterministic path convergence of GD shows high scientific integrity.

4. **Honesty and Transparency in Limitation Analysis**:
   The paper includes a highly commendable and self-critical discussion of its own limitations, such as the homogeneous class counts in standard benchmarks (remedied via a simulated heterogeneous experiment) and the "Hierarchical Representational Conflict" inherent in spatial regularizers.

---

## Weaknesses

1. **Unnecessary Complexity and Practical Failure of Elastic Spatial Regularization (ESR) (Major Weakness)**:
   The proposed ESR stabilizer introduces substantial mathematical formulation and two new continuous hyperparameters ($\beta$ and $\gamma$) that must be tuned. However, the empirical results show that ESR is of **no practical utility**:
   - As shown in the ablation study (Table 2), any positive values of $\beta$ or $\gamma$ strictly degrade Joint Mean performance monotonically.
   - At standard settings ($\beta=1, \gamma=1$, Method 7 in Table 1), RegCalMerge achieves a Joint Mean accuracy of **60.26\%**, which is **worse than the completely naive, zero-optimization, zero-hyperparameter Task Arithmetic baseline (60.35\%)**.
   - Introducing backpropagation, test-time optimization, and two new hyperparameters only to deliver performance inferior to the most basic static baseline represents an over-engineered solution of questionable value.

2. **Hierarchical Representational Conflict**:
   As the authors note, ESR's Spatial Deviation Penalty ($\gamma$) forces merging coefficients to be homogeneous across layers. This directly contradicts deep learning representation theory, which states that different layers represent different hierarchical levels of abstraction (early layers extract generic low-level features, whereas deep layers construct highly task-specific representations). Forcing spatial smoothness restricts the network's adaptive representational capacity, explaining why ESR consistently hurts performance. Introducing a complex mathematical regularizer that is fundamentally at odds with the network's hierarchical structure is a significant conceptual flaw.

3. **The Layer-wise Framework is Over-engineered Compared to Spatially Averaged Baselines**:
   The authors argue that "fine-grained layer-wise parameter flexibility is indeed necessary" because CalMerge (61.82% with 52 parameters) outperforms the Calibrated Spatial Mean (Cal-Mean) baseline (61.13% with 4 parameters).
   However, Cal-Mean achieves a highly robust 61.13% Joint Mean accuracy while reducing the optimization search space by a massive **13$\times$** (from 52 parameters to 4 parameters). Cal-Mean is completely immune to the Overfitting-Optimizer Paradox by construction, requires no complex regularizers (ESR), and requires no hyperparameter tuning. The extremely marginal gain of 0.69% from CalMerge does not justify the massive risk of transductive overfitting and the complexity of layer-wise optimization.

4. **Toy-Scale Experimental Setup**:
   The primary evaluation is restricted to simple visual datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) using a small ViT-B/32 backbone. Furthermore, both the calibration stream (16 samples per task) and test split (256 samples per task) are extremely small. Evaluating on more realistic, high-dimensional datasets (e.g., ImageNet, EuroSAT, DTD) or LLM merging tasks would significantly enhance the generalizability of the findings.

---

## Quality of Presentation
- **Presentation Rating**: Excellent
- **Justification**: The paper is exceptionally well-structured, clear, and easy to follow. The mathematical notation is elegant and precise, and the narrative deconstructing adaptive model merging is compelling and highly informative.

---

## Soundness, Significance, and Originality
- **Soundness Rating**: Good
  - *Justification*: The diagnostics, SNEW, and CCN are technically sound and well-reasoned. However, the proposed ESR regularizer is conceptually flawed due to its representational conflict and fails to outperform the most basic naive baseline, which slightly lowers the overall soundness of the proposed *methodological* framework.
- **Originality Rating**: Good
  - *Justification*: The spatial shuffling diagnostic and SNEW are highly original and creative solutions. The ESR regularization is more incremental but is applied in a novel test-time merging context.
- **Significance Rating**: Good
  - *Justification*: The analytical findings regarding the Overfitting-Optimizer Paradox and Sacrificial Task Bias are highly significant and will serve as a vital warning for future researchers in model merging and test-time adaptation. However, the significance of the proposed RegCalMerge method is limited because simpler, more elegant baselines (like Calibrated Spatial Mean) are practically superior.

---

## Overall Recommendation
- **Overall Recommendation**: 4: Weak Accept
- **Justification**: 
  The paper has clear merits. The deconstruction of the test-time model merging paradigm via the **spatial shuffling diagnostic** is an outstanding, simple, and elegant contribution that exposes the transductive overfitting of prior layer-wise optimization methods. Additionally, **SNEW** provides a simple, hyperparameter-free, and highly effective calibration mechanism that completely resolves sacrificial task bias.
  
  However, the proposed **Elastic Spatial Regularization (ESR)** is an over-engineered and unnecessary complexity. It introduces dual-penalty equations and new hyperparameters to tune, but physically degrades the model's accuracy below naive, zero-optimization Task Arithmetic and contradicts established deep learning representation theory.
  
  Overall, the paper's analytical insights and simple calibration engine (SNEW) are valuable enough to justify publication, as they will influence how future merging methods are evaluated. However, the authors are strongly encouraged to reframe their manuscript, de-emphasizing the over-engineered ESR stabilizer and highlighting the simpler, more elegant **Calibrated Spatial Mean** baseline as the superior practical solution for robust, overfit-free test-time model merging.

---

## Questions and Suggestions for the Authors

1. **Why prioritize ESR over the much simpler Calibrated Spatial Mean?**
   Since Calibrated Spatial Mean (Cal-Mean) achieves 61.13% with only 4 parameters, is completely immune to the Overfitting-Optimizer Paradox, requires no ESR, and has zero hyperparameters, why should a practitioner ever use the 52-parameter RegCalMerge (60.26% with ESR active)? Please consider reframing the paper to present Cal-Mean as a highly elegant, robust, and primary alternative to complex layer-wise optimization.

2. **Can SNEW and SNEW-based Cal-Mean be evaluated on more complex datasets?**
   While your simulated heterogeneous experiment (Section 4.3.3) provides elegant proof of SNEW's validity in imbalanced setups, evaluating on standard benchmarks with diverse domain shifts (e.g., ImageNet, EuroSAT, DTD) would greatly strengthen the generalizability of SNEW.

3. **Can you discuss the compute/time overhead of test-time optimization?**
   Test-time optimization via first-order gradient descent requires backpropagation through the model. For massive architectures like LLMs, this can be extremely slow and compute-intensive. Naive Task Arithmetic and Spatially Averaged methods have zero or minimal optimization overhead. Discussing this practical trade-off would add highly valuable context to the paper.
