# 5. Impact and Presentation

## Major Strengths
1. **Exceptional Empirical Honesty and Analysis:**
   - Although the paper starts by pitching TCPR as a successful method, the authors do not attempt to obfuscate its empirical failure in the results section.
   - Section 4.4 and Section 5 provide an outstanding, highly insightful scientific deconstruction of *why* the proposed static prior regularization fails (the Scale Mismatch, the Alignment-Interference Paradox, and the Static-Dynamic Conflict). This detailed post-mortem is the most valuable and rigorous part of the paper.
2. **Identification of the Softmax Competitive Bottleneck:**
   - The paper makes a compelling argument against the standard Softmax dynamic routing head, demonstrating that its zero-sum competitive constraint is harmful in multi-task settings.
   - The proposed **BSigmoid-Router** is a simple, elegant, and effective alternative that achieves a significant performance jump (25.50% joint mean accuracy vs. 19.10% for Softmax BL-Router) by decoupling the activation pathways.
3. **Rigorous Seed-Controlled Sweeps:**
   - The logarithmic hyperparameter sweeps over $\beta \in [10^{-6}, 10^2]$ are exhaustive and clearly demonstrate the transition from inactive to destructive regularization.

## Areas for Improvement
1. **Critical Narrative Overhaul (Fix the Self-Contradiction):**
   - The paper suffers from a severe structural disconnect. The Abstract, Introduction, and Contributions section pitch TCPR as a "simple yet highly effective approach" that "consistently prevents high-conflict task collapse." Yet, the Experiments and Conclusion sections show that TCPR is ineffective/harmful.
   - The paper must be completely rewritten to align its narrative with its empirical findings. It should be reframed as: *"An Empirical Study of Dynamic Router Calibration: The Power of Sigmoidal Routing and the Failure of Static Prior Regularization."* It should openly state from the beginning that static prior regularization is counterproductive, and focus on deconstructing this failure.
2. **Lack of Statistical Rigor (Multi-Seed Runs):**
   - Calibrating on a tiny set of 64 images makes the optimization extremely sensitive to random variance.
   - The authors must report means, standard deviations, and statistical significance tests across multiple random seeds (e.g., 5 or 10 runs with different initializations and calibration splits). Reporting results for a single seed ($\mathtt{seed=42}$) is statistically insufficient.
3. **Unrealistic and Under-Trained Specialists:**
   - The specialist models are poorly trained (e.g., SVHN expert at 23.20% and MNIST at 73.20%).
   - The authors should train these models to reasonable convergence so that they are merging actual task experts, not parameter noise.
4. **Scale and Modern Benchmarks:**
   - Evaluating on a 5.7M parameter ViT-Tiny and toy datasets limits the paper's impact. The authors should evaluate their independent sigmoidal router on modern model-merging benchmarks, such as merging LLMs (e.g., Llama-3 or Mistral) or Vision-Language Models (e.g., CLIP) on diverse downstream tasks.

## Overall Presentation Quality
- **Excellent.** The paper is exceptionally well-written, mathematically precise, and easy to read. The transitions between sections are smooth, and the vocabulary is professional.
- The equations are clearly typeset and defined. The figures and tables are clean and informative.

## Potential Impact and Significance
- **In Its Current Form (Weak):** Because the paper is framed around a "successful" method (TCPR) that is empirically proven to be a failure, it is highly confusing and lacks practical utility.
- **If Rewritten as a Negative-Result/Analysis Paper (High):** If reframed to focus on the unregularized `BSigmoid-Router` as a strong baseline and a rigorous analysis of why static prior regularizations fail, this paper could have a significant impact on the model-merging community. It would prevent other researchers from wasting time on static prior alignment and redirect research focus toward sample-dependent, dynamic structural regularizations.
