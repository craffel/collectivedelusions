# Intermediate Evaluation 5: Presentation and Impact

## 1. Major Strengths
- **Relevance of the Topic:** Addressing test-time model serving of Parameter-Efficient Fine-Tuning (PEFT) experts under data scarcity is an important and highly practical problem in edge-computing and multi-tenant systems.
- **Writing Structure:** The paper is well-structured, following a standard academic layout. The introduction and related work are clearly written and situate the work well.
- **Inclusion of Physical Benchmarks:** The attempt to scale the experiments to physical pre-trained BERT backbones is commendable, even if the results are ultimately unfavorable to the proposed method.

## 2. Areas for Improvement
- **Eliminate "Mathiness" and Speculative Bloat:** Sections 5.1 and 5.3 (quantization, martingale streaming) are completely unvalidated and highly speculative. They should be removed or moved to an appendix, and the paper should focus on theories that are actually evaluated.
- **Mathematically Define Key Baselines:** The authors must formally define the "diversity penalty" of the PEM-Div variant in the Methodology section to ensure reproducibility and scientific integrity.
- **Address Real-World Underperformance Sincerely:** Instead of dismissing the BERT results as a "local ceiling" and hand-waving the superiority of simpler baselines, the authors must sincerely discuss the limitations of their method on physical networks.
- **Resolve Theory-Practice Gap:** Derive PAC-Bayesian bounds that are mathematically exact for *continuous activation blending* (which is what is evaluated), or transition the empirical evaluations to *Stochastic Expert Routing* to match the derived bounds.
- **Evaluate Obvious Regularized Baselines:** Include L2/L1 regularized log-temperature baselines to prove that the complex PAC-Bayesian bound is practically superior to standard regularization techniques.

## 3. Overall Presentation Quality
The presentation is highly polished, but it relies on an adversarial defensive style. It uses overly dense mathematical notation and complex terminology (e.g., "Stoic robustness", "watertight generalization under stochastic expert routing", "first-principles derivation of representation interference") to distract from critical empirical and theoretical shortcomings, such as the underperformance on physical BERT backbones and the theory-experiment mismatch.

## 4. Potential Impact and Significance
The potential impact of this paper is **Low to Moderate**.
- For researchers, the theoretical formulation of simplex-constrained PAC-Bayes bounds is an interesting mathematical exercise.
- However, for practitioners, the method offers **no practical utility**. On real-world models (BERT), simply averaging expert weights (Uniform Merging) or using static, uncalibrated routing (SABLE Norm) is not only computationally cheaper (zero serving-time adaptation latency) but also achieves **superior classification performance** across almost all scales. 
- Given that the proposed method introduces high mathematical complexity, requires 120 ms CPU optimization, and results in worse classification accuracy on real models, it is highly unlikely to be adopted in real-world edge serving deployments.
