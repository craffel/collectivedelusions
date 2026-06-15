# Intermediate Evaluation: Impact and Presentation

## Major Strengths
1. **Conceptual Simplicity and Elegance:** The paper is guided by the philosophy of Occam's razor, challenging the recent trend of highly complex, over-parameterized regularizers (like quantum analogies or polynomial trajectories) in active model merging.
2. **Clear Mathematical Formulation:** The steps of PG-Merge, including gradient masking and the post-update parameter projection to prevent momentum leakage, are mathematically precise and well-articulated.
3. **Good Visualizations and Trajectory Analysis:** Figure 3 clearly illustrates the "Overfitting-Optimizer Paradox" in action—where prediction entropy is successfully minimized by unconstrained AdaMerging while the joint test accuracy decays. It provides a helpful visualization of the core problem.
4. **Computational Efficiency:** The proposed method requires zero training, zero auxiliary parameters, and operates on standard backpropagation gradients, making it computationally lean and extremely easy to implement.

## Areas for Improvement
1. **Scale up the Experiments:** Transition from the outdated, toy setup (`vit_tiny` on MNIST, FashionMNIST, CIFAR-10, and SVHN) to realistic modern settings. Evaluating on larger models (such as LLaMA-2-7B, Mistral-7B, or CLIP-ViT-B) on diverse, high-dimensional downstream tasks is crucial to prove that the findings generalize.
2. **Perform Rigorous Statistical Validation:** Report means and standard deviations across at least 5 different random seeds (with varying calibration and test subsets). Currently, the performance gap between PG-Merge and the static Uniform Merging baseline ($0.54\%$) is well within the statistical margin of error for the 512-image test sets.
3. **Resolve the SGD/Adam Discrepancy:** The authors strongly advocate for pairing PG-Merge with standard SGD in Appendix A to avoid momentum decay and make the post-update projection redundant. However, they only present empirical results using the Adam optimizer. They must include a comparative study showing both Adam and SGD results.
4. **Address the Classification Head Semantic Conflict:** Explain how the classification heads of different tasks (with disjoint label spaces) are merged and how semantic conflicts are resolved in the final output head.
5. **Re-evaluate and Calibrate Baselines:** Fix and re-optimize the PolyMerge baseline. PolyMerge should not collapse to near-random performance if properly configured, since the uniform baseline (which performs much better) is a restricted subset of PolyMerge's search space.
6. **Compare to a Properly Tuned AdaMerging Baseline:** Show whether the overfitting of unconstrained AdaMerging can be prevented simply by reducing the learning rate or applying early stopping, which would be even simpler than gradient pruning.

## Overall Presentation Quality
The overall presentation quality is **good**. The paper is logically structured, the writing style is professional, and the equations are clear. The authors do a commendable job of contextualizing their work within the existing literature of model merging and test-time adaptation. However, the tone is occasionally overly enthusiastic given the very minor and statistically questionable empirical improvements.

## Potential Impact and Significance
In its current form, the potential impact of this paper is **low**. While the philosophy of simplifying regularizers is highly admirable, the empirical support is extremely weak due to the toy scale of the experiments and the statistically marginal improvements over a zero-overhead static average. If the authors scale their evaluation to modern LLMs/VLMs and demonstrate robust, statistically significant improvements, the method (PG-Merge) could have **moderate-to-high impact** as a highly practical, computationally efficient, and elegant blueprint for real-world test-time model fusion.
