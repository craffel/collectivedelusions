# Intermediate Evaluation 4: Experimental Evaluation Check

## Evaluation of Experimental Setup and Baselines
The experimental setup is exceptionally thorough and rigorously designed. It evaluates:
1. **Diverse Data Regimes**: From extreme low-data calibration ($N_{\text{cal}} \in \{32, 64\}$) to high-data generalization ($N_{\text{cal}} \in \{500, 4000\}$).
2. **Anisotropy Sweeps**: Systematically varying $\rho \in [0.0, 0.5]$ to simulate the high-dimensional representation cones of foundation models.
3. **Comprehensive Baselines**: Uniform Merging, Stateless SABLE, Stateful ChemMerge, Unregularized Softmax, Regularized Softmax ($\lambda \in \{10^{-4}, 10^{-2}\}$), and Sigmoid.
4. **Controlled Ablations**: 
   - Layer-invariant vs. Layer-wise classical routers.
   - Zero-initialization vs. Random initialization.
   - Open-loop (EMA-SABLE) vs. Closed-loop (ChemMerge) trajectory smoothing.

All quantitative experiments are run across 5 independent seeds, reporting both means and standard deviations, ensuring statistical reliability.

## Supporting Evidence for Primary Claims
The empirical results provide compelling support for most of the authors' claims:
- **Small-Sample Bottleneck**: Table 1 clearly shows that under $N_{\text{cal}}=64$, Unregularized Softmax achieves $68.00\%$ accuracy, lagging behind SABLE ($73.76\%$) and ChemMerge ($76.90\%$). This confirms that training-free methods act as powerful inductive priors when parameters cannot be reliably optimized.
- **Large-Sample Recovery**: Table 2 confirms that under $N_{\text{cal}}=4000$, Unregularized Softmax achieves $76.22\%$, outperforming SABLE ($73.76\%$, $p = 0.0062$) and approaching ChemMerge ($76.90\%$). This validates the core thesis that classical routers are highly capable representationally when the data bottleneck is resolved.
- **Bias-Variance Trade-off**: The regularized Softmax ($\lambda=10^{-2}$) achieves $74.10\%$ under $N_{\text{cal}}=4000$, whereas the weakly regularized variant ($\lambda=10^{-4}$) reaches $75.70\%$, illustrating how excess weight decay introduces a constraint bias under data abundance.
- **Debunking the Jitter Myth**: Table 3 demonstrates that layer-wise classical routers achieve very stable trajectories (Jitter: $0.0068 - 0.0458$), indicating that the high-frequency oscillations reported in prior literature are not an inherent defect of parametric gating.

## Critical Evaluation and Flaws (Theorist Perspective)

Despite the extensive experiments, several empirical limitations must be highlighted:

### 1. The Real-World Validation is Highly Constrained
To validate generalizability, the authors use **BERT-Tiny** on SST-2 and QQP. This choice exhibits severe limitations:
- **Toy Model Scale**: BERT-Tiny has only 4 encoder layers and a hidden dimension of 128. This does not represent the high-dimensional latent spaces (typically $D \ge 4096$) or the complex representation cones of state-of-the-art multi-billion parameter foundation models (e.g., LLaMA, Mistral).
- **Under-fitted Experts**: The custom LoRA experts are significantly under-fitted, achieving standalone test accuracies of $58.80\%$ on SST-2 and $65.60\%$ on QQP. In a realistic serving scenario, practitioners deploy fully converged, high-performing experts. Under-fitted experts generate noisier, less stable activations, which might distort ensembling dynamics.

### 2. Architectural Limitation of Classifier Logit Blending
In the BERT-Tiny validation, the joint serving model blends task-specific predictions using a weighted sum of logits:
$$\text{logits} = \alpha_0 \cdot \text{classifier}_0(\text{pooled}) + \alpha_1 \cdot \text{classifier}_1(\text{pooled})$$
This formulation assumes both classifiers share the exact same label space dimensionality (2 output classes). In a realistic, heterogeneous multi-task serving registry (e.g., combining 2-class sentiment analysis with a 10-class image classification task), this equation would fail immediately due to shape mismatches. Thus, the validation setup is highly specialized and lacks generalizability to arbitrary multi-task registries.

### 3. Empirical Contradiction on Small-Sample Collapse
In Table 5 (BERT-Tiny results), under $N_{\text{cal}}=32$, the Unregularized Softmax Router achieves **$61.90\%$** accuracy, outperforming SABLE ($60.00\%$), ChemMerge ($60.00\%$), and the Proposed Zero-Init Router ($61.70\%$). 
This directly contradicts the core thesis modeled in the sandbox, where unregularized classical routers collapsed catastrophically under small-sample constraints. 
The authors explain this by pointing out that SST-2 and QQP are semantically disjoint tasks that map to highly separated subspaces in BERT-Tiny, meaning the router can easily locate a stable separating hyperplane with only 16 samples per task. While this is a highly insightful geometric observation, it reveals that the "catastrophic overfitting collapse" of classical routers is **not a universal property** under low-data budgets, but is highly task-dependent. In real-world settings with disjoint tasks, the classical unregularized router is actually the *most* robust and high-performing option, undermining the necessity of both training-free priors and proper L2 regularization.
