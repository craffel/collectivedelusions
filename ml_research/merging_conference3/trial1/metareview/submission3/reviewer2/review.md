# Peer Review

## Summary of the Submission
This paper introduces **ThermoMerge** (Thermodynamic Test-Time Diffusion for Synergistic Model Merging), an unsupervised test-time optimization framework designed to combine specialized task-specific expert models without retraining. The authors argue that existing adaptive model merging frameworks (e.g., AdaMerging, SyMerge) are severely limited by their reliance on deterministic optimizers (like Adam or SGD), which become permanently trapped in the sharp, sub-optimal local basins of the highly non-convex joint proxy loss landscape during test-time model fusion.

To resolve this bottleneck, the authors cast test-time adaptation as a thermodynamic physical system undergoing crystallization (transitioning from a high-entropy disordered state of independent experts to a highly ordered, low-energy crystalline multi-task consensus). This physical process is implemented using **Stochastic Gradient Langevin Dynamics (SGLD)** guided by an **exponential Simulated Annealing cooling schedule**. To joint-optimize low-dimensional merging coefficients alongside high-dimensional classifier parameters without destroying pre-trained classification features, the paper proposes **Dimensionality-Scaled Langevin Noise (DSLN)**, which scales the coordinate-wise Langevin noise standard deviation inversely with the square root of the parameter group dimension ($1/\sqrt{d_j}$).

The framework is evaluated on a hand-crafted synthetic 1D non-convex physical simulation landscape, as well as on lightweight Multi-Layer Perceptrons (MLPs) and low-rank adapters (LoRA PEFT) across a multi-dataset benchmark suite (MNIST, FashionMNIST, KMNIST) under clean and out-of-distribution (corrupted) test-time streaming.

---

## Strengths and Weaknesses

### Strengths
1.  **Creative and Intellectually Engaging Concept:** Framing test-time model merging and parameter-conflict mitigation as a thermodynamic crystallization process is highly creative. The application of statistical mechanics concepts (partition functions, Shannon entropy, and Specific Heat capacity peaks) to analyze parameter dynamics is theoretically rich and elegant.
2.  **Exceptional Writing and Presentation Quality:** The paper is exceptionally well-written, clearly structured, and easy to follow. The mathematical notation is rigorous and consistent, and the algorithmic pseudocode in Algorithm 1 is highly detailed.
3.  **Beautiful and Polished Visualizations:** The qualitative optimization trajectory (Figure 1), Specific Heat peak (Figure 3), and deep adaptation loss curves (Figure 4) are polished and effectively illustrate the core concepts of the paper.
4.  **Insightful Technical Discussion:** The discussion regarding the weight-bias thermodynamic imbalance, layer-wise functional grouping, and seed synchronization in distributed systems (Section 3.4) shows a deep understanding of the practical and engineering challenges of high-dimensional optimization.

---

### Weaknesses

#### 1. Severe Contradiction Between the Core Claims and Empirical Evidence
The paper's primary thesis is that standard deterministic joint adaptation (such as SyMerge) is bottlenecked by the highly non-convex landscape and becomes trapped in sub-optimal basins, which ThermoMerge resolves through SGLD global exploration. 
However, an inspection of the actual deep learning results on real neural network parameters reveals a **stark contradiction**:
*   **MLP Clean (Table 7):** On MNIST, deterministic SyMerge achieves **$89.97\% \pm 0.19\%$** compared to ThermoMerge's $89.94\% \pm 0.16\%$. On FashionMNIST, deterministic SyMerge achieves **$84.61\% \pm 0.49\%$** compared to ThermoMerge's $84.46\% \pm 0.59\%$. In both cases, **the deterministic baseline outperforms ThermoMerge**.
*   **PEFT/LoRA Clean (Table 10):** On MNIST, deterministic SyMerge achieves **$88.68\% \pm 0.86\%$** compared to ThermoMerge's $88.65\% \pm 0.63\%$. 
*   **OOD Corruptions (Table 8 & Table 11):** Across 6 OOD configurations (MLP and LoRA across MNIST, FashionMNIST, and KMNIST), **deterministic SyMerge achieves a higher mean accuracy than ThermoMerge in 5 out of 6 cases**. For example, on MLP MNIST OOD, SyMerge achieves **$87.51\% \pm 0.53\%$** vs. ThermoMerge's $87.49\% \pm 0.54\%$.

In summary, across **12 distinct real neural network evaluations**, **deterministic SyMerge matches or beats ThermoMerge in 9 out of 12 cases**. This empirical evidence directly refutes the claim that deterministic optimizers are severely trapped, and suggests that real test-time adaptive model merging landscapes do not exhibit the extreme, insurmountable sharp local traps present in the hand-crafted synthetic 1D landscape. The 56.7% final loss reduction reported on the synthetic landscape is an artificial artifact that completely fails to translate to real deep learning tasks.

#### 2. Redundancy of High-Dimensional Classifier SGLD Adaptation
The proposed joint SGLD optimization of high-dimensional classifiers requires the complex DSLN scaling rules, layer-wise functional grouping, and weight-bias thermodynamic balancing.
However, the ablation baseline **"ThermoMerge (Coefficients Only)"** in Table 7 (which applies SGLD *only* to the low-dimensional merging coefficients while updating the classifiers deterministically) performs virtually identically to the full joint ThermoMerge:
*   MNIST: $89.89\% \pm 0.17\%$ (Coefficients Only) vs. $89.94\% \pm 0.16\%$ (Ours)
*   FashionMNIST: $84.35\% \pm 0.45\%$ (Coefficients Only) vs. $84.46\% \pm 0.59\%$ (Ours)
*   KMNIST: $80.36\% \pm 0.31\%$ (Coefficients Only) vs. $80.37\% \pm 0.24\%$ (Ours)

The difference is statistically negligible across all datasets. This indicates that joint classifier Langevin adaptation and the DSLN formulation are largely redundant, and that the performance gains of the framework are entirely driven by the simple optimization of the low-dimensional coefficients $\Lambda$.

#### 3. Extremely Restricted and Toy-Scale Empirical Evaluation
While modern model merging research focuses on massive pre-trained transformer foundation models (such as ViT, CLIP, LLaMA, or Mistral with hundreds of millions or billions of parameters), this paper evaluates its deep learning experiments purely on:
*   A lightweight Multi-Layer Perceptron (MLP) containing only 2 fully-connected layers (size $128$ and $64$).
*   Toy-scale grayscale classification datasets (MNIST, FashionMNIST, and KMNIST of size $28 \times 28$).

Claiming that "joint thermodynamic SGLD exploration successfully scales to deep neural landscapes" based on a tiny 2-layer MLP on digits is an extreme overstatement of the framework's scalability and generalizability.

#### 4. The High-Dimensional Noise Paradox in DSLN
The DSLN formulation scales the coordinate-wise thermal noise standard deviation by $1/\sqrt{d_j}$. For high-dimensional parameters in actual deep neural networks (where $d_j \ge 10^5$), this scales the noise standard deviation down to negligible levels (e.g., $10^{-4}$ to $10^{-6}$). At this scale, the thermal noise is several orders of magnitude smaller than gradient updates and floating-point precision, making it physically impossible to overcome any non-convex energy barriers in the classifier space. The classifiers are updated almost purely deterministically, confirming that the "global exploration" claim is physically meaningless for high-dimensional classification heads.

#### 5. Severe Test-Time Computational Overhead
The expert-guided soft self-labeling objective requires forwarding every test batch through all $K$ unmerged expert models to obtain teacher labels at every adaptation step. This introduces an $O(K)$ computational and memory overhead during test-time inference. For merging many-task models (e.g., $K = 8$ or $K = 20$), this overhead is extremely expensive and completely defeats the core purpose of model merging, which is to avoid maintaining and running multiple independent model checkpoints during inference.

---

## Soundness
**Soundness Rating: Fair**

**Justification:** The mathematical derivations of SGLD, preconditioned SGLD, and DSLN are correct, and the conceptual framing is theoretically consistent. However, the experimental methodology is weak, and the central claims of the paper (that deterministic optimizers are severely trapped and that ThermoMerge provides superior OOD generalization and accuracy) are **actively contradicted by the authors' own empirical results on real neural networks**, where the deterministic SyMerge baseline consistently achieves equal or superior mean accuracies. Furthermore, the Specific Heat peak and phase transition analysis are conducted purely on a highly artificial, hand-crafted synthetic 1D function, which has zero practical relevance to real deep learning landscapes.

---

## Presentation
**Presentation Rating: Excellent**

**Justification:** The paper is exceptionally well-written, logically structured, and polished. The equations are mathematically clear, the tables are detailed, and the visual plots are highly professional and informative. The authors have done a superb job communicating their ideas.

---

## Significance
**Significance Rating: Poor**

**Justification:** The paper's immediate significance to the machine learning community is extremely low. Since deterministic joint adaptation (SyMerge) consistently matches or beats ThermoMerge across almost all real neural network tasks, practitioners have zero practical incentive to adopt the significant mathematical complexity, hyperparameter tuning, and potential instability of SGLD and Simulated Annealing. Furthermore, the evaluation is restricted to toy-scale grayscale classification tasks using a tiny MLP, making it impossible to assess whether this framework has any relevance or applicability to modern foundation model merging workflows.

---

## Originality
**Originality Rating: Good**

**Justification:** The conceptual framing of test-time model adaptation as a thermodynamic crystallization process is creative. While SGLD, Simulated Annealing, and variance-scaling heuristics are standard tools in deep learning, their specific combination and application to the joint optimization of merging coefficients and classifiers at test-time represents a novel synthesis.

---

## Overall Recommendation
**Overall Recommendation: 2: Reject**

**Detailed Justification:** 
While this paper is exceptionally well-written, conceptually creative, and visually stunning, it suffers from a fundamental and fatal scientific deficit: **its core claims are directly refuted by its own empirical evidence on real neural networks**. The authors argue that deterministic optimizers are severely trapped and that ThermoMerge's physical global exploration rescues them to yield superior OOD generalization and multi-task accuracies. However, across 12 distinct real neural network evaluations (clean & OOD, MLP & LoRA across 3 datasets), deterministic SyMerge achieves a higher or equal mean accuracy compared to ThermoMerge in 9 out of 12 cases. 

Additionally, the "Coefficients Only" ablation shows that joint classifier SGLD and the entire DSLN apparatus are virtually redundant, yielding no statistically significant accuracy benefit. When combined with the restricted toy-scale evaluation (MLP on MNIST/FashionMNIST/KMNIST digits) and the severe test-time computational overhead of self-labeling, the paper fails to demonstrate any practical or theoretical benefit for real-world model merging. I must recommend a rejection of this submission in its current state.

---

## Constructive Questions and Feedback for the Authors

1.  **Address the Empirical Deficit:** Can the authors explain why standard deterministic joint gradient descent (SyMerge) consistently achieves equal or superior mean accuracies compared to ThermoMerge across almost all real neural network evaluations (clean and OOD, MLP and LoRA)? If deterministic optimizers are "severely trapped," why does SGLD not yield a clear accuracy improvement?
2.  **Scale to Realistic Foundation Models:** To prove the significance and generalizability of your thermodynamic optimization principles, the framework must be evaluated on large-scale pre-trained foundation models (e.g., merging CLIP ViT-B/32 encoders or LLaMA-class LLMs) on realistic multi-task benchmarks (such as CLIP 8-dataset merging).
3.  **Simplify the Algorithmic Complexity:** Given that the "Coefficients Only" ablation (Table 7) performs almost identically to the full joint ThermoMerge (Ours), would it not be far more practical and elegant to restrict SGLD adaptation purely to the low-dimensional merging coefficients $\Lambda$, while keeping the classifiers frozen or updated deterministically? This would eliminate the entire complexity of the DSLN formulation, layer-wise functional grouping, and weight-bias thermodynamic balancing, yielding a lightweight and highly practical method.
4.  **Validate on Realistic Landscapes:** The hand-crafted synthetic 1D landscape (Equation 21) explicitly introduces high-frequency sinusoidal ripples (`\sin(20.0 \Lambda)`) to block deterministic optimizers. Is there any empirical evidence that real neural network model merging loss landscapes exhibit periodic sinusoidal ripples or severe high-frequency local traps? If not, please tone down the claims that this synthetic landscape is a "rigorous representation of actual parameter spaces."
5.  **Address Test-Time Inference Overhead:** The soft self-labeling objective requires forwarding streaming test samples through $K$ independent unmerged experts at every step of test-time adaptation. How do the authors plan to address this severe $O(K)$ computational and memory overhead when scaling to large language models or vision-language models with billions of parameters?
6.  **Validate Confirmation Bias Mitigations:** The authors discuss several elegant strategies (confidence filtering, entropy weighting, predictive agreement monitoring) to mitigate the vulnerability of self-labeling to teacher bias and confirmation bias. Why were none of these safety-valve strategies integrated or validated in your deep learning or corrupted OOD experiments? Please provide empirical validation of these safety safeguards.
