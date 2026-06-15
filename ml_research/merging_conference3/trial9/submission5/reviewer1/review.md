# Comprehensive Peer Review

## 1. Summary of the Paper
This paper presents a rigorous methodological audit and experimental deconstruction of dynamic activation-space model merging. Dynamic model merging has emerged as an important serving-time paradigm for ensembling task-specific expert adapters (e.g., LoRA) on the fly within a single parallel forward pass. Recently, several state-of-the-art (SOTA) works—such as SABLE (nearest-centroid routing) and ChemMerge (using continuous-time chemical kinetics via ODEs)—have claimed that classical parametric routers (e.g., linear gating heads optimized via gradient descent) catastrophically fail in low-data calibration regimes, justifying the need for highly complex mathematical or physical routing architectures.

The authors critically examine this consensus. They hypothesize that the reported failures of classical parametric routers in prior work are confounding artifacts of weak experimental methodology, specifically: (1) initializing parametric heads randomly without structural priors, and (2) failing to apply proper L2 weight regularization on tiny calibration splits. 

To test this, they introduce a properly regularized, maximum-entropy zero-initialized classical linear router and audit it inside a high-fidelity 14-layer synthetic Analytical Coordinate Sandbox (ICS) under varying representation entanglement (anisotropy) levels ($\rho \in [0.0, 0.5]$) and dual data regimes: a small-sample constraint regime ($N_{\text{cal}} = 64$) and a large-sample generalization regime ($N_{\text{cal}} = 4000$). Finally, they validate their insights on a pre-trained BERT-Tiny foundation model on actual GLUE datasets (SST-2 vs. QQP).

### Key Empirical Findings:
1. **The Small-Sample Overfitting Bottleneck:** Under extreme data scarcity ($N_{\text{cal}} = 64$), classical routers overfit because learning 768 parameters from 64 samples is under-determined. In this regime, training-free priors like SABLE and ChemMerge perform well because their cosine-based formulations act as highly effective inductive geometric priors.
2. **Spectacular Large-Sample Recovery:** When provided with adequate calibration budgets ($N_{\text{cal}} = 4000$), classical parametric routers recover completely. The unregularized Softmax router achieves $76.22\% \pm 0.78\%$ accuracy, outperforming stateless SABLE ($73.76\% \pm 0.72\%$) by $+2.46\%$ absolute (a highly statistically significant margin, $p = 0.0062$) and closely approaching ChemMerge ($76.90\% \pm 0.68\%$).
3. **The Bias-Variance Trade-off in Gating Regularization:** Strong L2 regularization ($\lambda = 10^{-2}$) is essential under small-sample constraints, but introduces an unnecessary constraint bias under data abundance, limiting accuracy. The authors show that scaling down to $\lambda = 10^{-4}$ under large-sample abundance resolves this constraint bias, reaching a near-optimal $75.70\% \pm 0.95\%$ accuracy.
4. **Control-Theoretic Demystification of ChemMerge:** ChemMerge’s continuous-time kinetics introduce a severe "representational lag" (intermediate representation quality of $0.912$ at Layer 14 vs. $0.992$ for the classical router). Under a control-theoretic lens, this lag acts as a beneficial temporal low-pass filter (closed-loop stateful inertia) that stabilizes ensembling trajectories under heavy activation noise, explaining its superior performance ceiling.
5. **Debunking the Jitter Myth:** A layer-wise classical router with separate heads exhibits exceptionally smooth routing trajectories (Jitter: $0.0068 - 0.0458$), comparable to or lower than ChemMerge ($0.0368$), debunking the claim that parametric routers are inherently unstable.
6. **Open-Loop vs. Closed-Loop Smoothing (EMA-SABLE):** Applying a simple Exponential Moving Average (EMA) to SABLE (EMA-SABLE) boosts accuracy by $+1.00\%$ absolute. However, ChemMerge's superior performance ceiling proves its ODE kinetics act as a true closed-loop feedback controller.
7. **Real-World BERT-Tiny Validation:** On a real pre-trained model, parametric routers outperform SABLE and ChemMerge under sufficient calibration data ($N_{\text{cal}} = 500$). Under extreme data scarcity ($N_{\text{cal}} = 32$), the unregularized router does not collapse but achieves the highest accuracy ($61.90\%$) because the underlying semantic spaces are highly disjoint and easily separated.

---

## 2. Strengths and Weaknesses

### Strengths:
* **Exceptional Methodological Rigor:** The paper is a superb application of Occam's razor to deep learning. It systematically deconstructs complex metaphorical SOTA architectures (continuous-time ODE kinetics) to reveal that a properly initialized and regularized classical linear gating head is highly competitive and often superior.
* **Deep Mechanistic Explanations:** The authors go far beyond simple accuracy comparisons. They analyze the underlying mechanics by:
  - Tracking layer-wise prototype similarities to expose representational lag.
  - Formulating EMA-SABLE to isolate open-loop smoothing from closed-loop feedback correction.
  - Designing a layer-wise classical router to debunk the "jitter myth".
  - Mapping the precise sample complexity crossover points where learning-based gating overtakes training-free priors.
* **Outstanding Practical Value:** 
  - Table 7 (Complexity Analysis) provides a direct comparison of Parameters, FLOPs, Gating Evaluation, and Sequential Serving-Time Overhead across architectures. This is exactly what a systems engineer needs to make informed edge deployment decisions.
  - Demystifying the serving-time "clamping hack" in ChemMerge exposes the hidden engineering duct-tape of metaphorical architectures, helping practitioners see through the marketing of continuous-time kinetics.
  - The proposed Zero-Initialized Softmax router with L2 regularization is exceptionally lightweight, simple to implement, and computationally efficient, executing sharp, instantaneous ensembling decisions without serving-time ODE overhead.
* **Excellent Writing Quality:** The narrative is clean, engaging, and uses a very strong technical vocabulary. It guides the reader through a logical progression from synthetic coordinate simulation to real-world pre-trained weights, maintaining a highly objective and skeptical tone.

### Weaknesses:
* **Scale of Foundation Model Validation:** The primary weakness is the scale of the real-world validation. Evaluating on a toy **BERT-Tiny** model (4 encoder layers, hidden size 128) with under-fitted expert adapters does not fully reflect the complex activation manifolds, representation cones, and massive hardware constraints of deploying multi-billion parameter foundation models (e.g., LLaMA-70B, Mistral, or ViT-H). Validating these findings on a larger, standard pre-trained model (e.g., LLaMA-1B/3B, RoBERTa-Base, or ViT-B/16) with fully converged experts would make the practical utility of the work unquestionable.
* **Generative Workloads:** The evaluation is entirely focused on multi-task classification. Modern edge serving of foundation models heavily features generative tasks (e.g., text summarization, code generation, image synthesis). In generative settings, task embeddings exhibit much denser geometric overlap due to shared syntactic structures. Sweeping these dynamics under generative, instruction-tuned workloads represents a crucial direction for future work to confirm if classical parametric routers survive without training-free priors.
* **Addressing Architectural Asymmetry:** In the BERT-Tiny experiments, the classical parametric router is evaluated as a stateless, embedding-level gating model (at Layer 0) while SABLE and ChemMerge compute routing decisions dynamically layer-by-layer. Implementing and evaluating a layer-wise classical router on BERT-Tiny would eliminate this structural asymmetry and provide a cleaner comparison.
* **BERT-Tiny Classifier Logit Blending Constraint:** As the authors note, the joint multi-task serving model in the BERT-Tiny validation computes predictions by taking a weighted sum of the task-specific classifiers' logit outputs. This design assumes that all task-specific classifiers output logits of the exact same dimensionality. If we were to serve tasks with mismatched label spaces, this direct blending equation would crash immediately. For a general-purpose serving framework, routing must instead occur directly to task-specific classifier heads or employ dynamic label space adapters.

---

## 3. Detailed Evaluations

### Soundness: Excellent
The methodology is exceptionally rigorous, sound, and thoroughly analyzed. It contains synthetic sweeps, hyperparameter sensitivity audits, mathematical deconstructions, statistical significance tests, and real-world pre-trained model validation. The mathematical definitions are flawless, and the paper is transparent about limitations (such as the sandbox's unbalanced noise favoring stateful models).

### Presentation: Excellent
The paper is beautifully written, extremely clear, and organized logically. The mathematical derivations and descriptions are precise. Visualizations and tables are professional and self-contained, and the narrative flow is highly engaging and clean.

### Significance: Good
The paper has high significance, especially for edge-serving engineers and practitioners. By simplifying dynamic model ensembling to a standard linear gating layer with weight decay and zero-initialization, it eliminates the need for serving-time ODE solvers and numerical clamping hacks. While the scale of validation is limited to BERT-Tiny, the conceptual contribution is vital to prevent over-engineering in the field.

### Originality: Good
While primarily an audit and deconstruction paper, it introduces several original baselines (Maximum-Entropy Zero-Initialization, EMA-SABLE, layer-wise classical routers) and provides a unique control-theoretic and sample complexity analysis of existing methods. This is highly original for an audit work and provides immense educational value.

---

## 4. Overall Recommendation
**Score: 5 - Accept**
*Justification:* This is a technically solid, exceptionally well-written, and methodologically rigorous paper that addresses an important and timely problem. It represents a vital corrective to the over-engineering trend in model merging, demonstrating that a simple, lightweight linear gating head is highly competitive and often superior to complex ODE-based architectures when a modest calibration budget is available. Although the scale of the real-world validation (BERT-Tiny) is small, the insights are profoundly practical and of high interest to the machine learning community.

---

## 5. Questions and Suggestions for the Authors

1. **Scale of Validation:** Do you plan to scale your real-world pre-trained model validation to larger architectures (e.g., LLaMA-1B/3B, RoBERTa-Base, or ViT-B/16) with fully converged experts? Showing that the classical router maintains its superiority on these standard, highly-parameterized models would make your practical guidelines significantly stronger.
2. **Generative Workloads:** In generative settings, task embeddings exhibit much larger geometric overlap due to shared syntactic structures. Have you considered evaluating your classical router under generative workloads (e.g., text summarization vs. code generation on large LLaMA-based adapters)? How do you hypothesize the overfitting bottleneck will behave under such dense representational overlap?
3. **Architectural Asymmetry:** In the BERT-Tiny experiments, SABLE and ChemMerge operate layer-wise while the classical router operates at Layer 0 (embedding level). To ensure a fairer, structurally symmetric comparison, have you considered training and evaluating a layer-wise classical router on BERT-Tiny?
4. **Addressing Label Space Mismatches:** How would you suggest adapting your real-world BERT-Tiny joint serving model to handle tasks with mismatched label spaces (e.g., combining binary classification with multi-class classification) where direct classifier logit blending is mathematically impossible?
