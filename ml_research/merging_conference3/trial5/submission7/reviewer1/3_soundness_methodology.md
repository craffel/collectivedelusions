# 3. Soundness and Methodology Evaluation

This section provides a rigorous critique of the technical soundness, clarity of description, appropriateness of methods, and reproducibility of the proposed PG-Merge framework.

## 1. Semantic Contradiction: Is PG-Merge Truly "Training-Free"?
The paper repeatedly describes PG-Merge as a **"training-free"** method (e.g., in the Abstract, Introduction, Methodology, and Conclusion). This characterization is semantically deceptive:
- **True training-free methods** (such as Task Arithmetic, Model Soups, and TIES-Merging) combine expert parameters algebraically in a single offline step, requiring **zero** optimization, zero backward passes, and zero parameter updates at test time.
- **PG-Merge** requires running **100 gradient steps** of backpropagation usingprediction entropy minimization on an unlabeled calibration set during test-time.

By definition, executing 100 steps of gradient descent via backpropagation to optimize merging coefficients is an active learning/adaptation process (often called Test-Time Training or Test-Time Adaptation). Labeling a gradient-descent-based adaptation loop as "training-free" is highly misleading and unfairly positions it against actual zero-shot, zero-compute static merging methods.

## 2. Technical Contradiction: The Adam vs. SGD Dilemma
In Appendix A ("Optimizer State Mismatch and SGD Compatibility"), the authors provide a very insightful theoretical critique of using PG-Merge with adaptive optimizers like Adam:
- When a coefficient is masked out (gradient is zeroed), Adam's historical momentum buffers ($\mathbf{m}$ and $\mathbf{v}$) continue to decay.
- This creates an "internal state mismatch" that artificially dampens the parameter's updates when it is eventually re-selected by the dynamic mask.
- To resolve this, the authors strongly advocate for pairing PG-Merge with **standard SGD without momentum**, stating that standard SGD "completely bypasses momentum leakage, rendering the post-update parameter projection... entirely redundant" and makes the pipeline "mathematically self-consistent with zero state management overhead."

However, this theoretical contribution is completely contradicted by their empirical implementation:
- In Section 4.1 (Experimental Setup), the authors state that they run the active model merging adaptation over 100 steps **using the Adam optimizer** with a learning rate of $10^{-3}$.
- The quantitative results in Table 1 (Scoreboard) and Table 2 (Ablation) are thus generated using **Adam**, meaning they are directly subject to the exact "state mismatch and momentum decay" flaws analyzed in Appendix A.
- The authors provide **no empirical results** for the SGD-based version of PG-Merge. 

This is a severe technical gap. If the SGD formulation is mathematically superior, clean, and eliminates the need for the complex post-update projection (Equation 13), the authors must provide empirical results using SGD. Evaluating the entire paper on Adam while theoretically advocating for SGD in the appendix represents an unresolved contradiction.

## 3. Under-Stated Hyperparameter Dependency
The authors argue that PG-Merge is "non-parametric" and criticize prior methods (like RegCalMerge) for their "delicate hyperparameter tuning." However, PG-Merge's performance is highly dependent on its own key hyperparameter: the **sparsity ratio $p$**.
As shown in the ablation study (Table 2):
- At $p=0.05$, the Joint Mean accuracy is $62.70\%$.
- Increasing $p$ slightly to $0.30$ drops the accuracy to $61.33\%$.
- At $p=1.00$ (unconstrained), the accuracy is $61.08\%$.

The performance of PG-Merge is highly sensitive to the choice of $p$, with a very narrow "sweet spot" at $p=0.05$. Claiming that PG-Merge is "hyperparameter-lean" while its success depends entirely on the precise tuning of this sparsity percentile is a significant overstatement. 

## 4. Reproducibility Concerns
While the mathematical formulation of the sparse gradient masking (Equations 8-13) is clearly written, there is **no code repository provided** (even an anonymous one), which is standard practice for modern machine learning submissions. Given that the paper uses a highly custom, low-capacity experimental pipeline (`vit_tiny` fine-tuned on custom subsets of 1,024 images, with a specialized 64-image calibration stream), releasing code is essential to verify that these results are reproducible and not the product of specific random seeds or overfitting to the toy calibration set itself.
