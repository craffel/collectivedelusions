# 5. Impact and Presentation Quality

## Major Strengths
1. **Exceptional Clarity and Structure:** The paper is exceptionally well-written, logically structured, and easy to follow. The introduction clearly motivates the problem, and the methodology provides clean, consistent mathematical notation.
2. **Move Towards Realism (Physical Sequential Merging):** The authors deserve high praise for going beyond a simple "virtual layer ensembling" sandbox and implementing a physical sequential weight-space merging framework. This exposes the realistic challenges of sequential propagation that are often brushed aside in the model-merging literature.
3. **Rigorous Empirical Mapping:** The authors conduct a large-scale grid sweep over 1,280 experiment runs across 5 independent seeds, providing solid statistical reporting (means and standard deviations) for all primary tables.
4. **Intellectual Honesty and Deep Analysis:** The discussions in the appendix (especially regarding Softmax's sandbox superiority and the optimization sluggishness of Sigmoid) are refreshingly honest, insightful, and constructive for practitioners.
5. **Practical Parameter Efficiency:** Demonstrating that tying routing weights across layers can reduce the trainable routing parameter footprint by up to 91.7% (with little to no loss in sandbox performance) is a highly practical finding for deployment-constrained scenarios.

## Areas for Improvement
1. **Rigor of Theoretical Derivations:** The core mathematical model of "Expected Ruggedness" contains a glaring algebraic omission in Equation 10, omitting the mean difference term $\left( \mathbb{E}[\bar{\alpha}_k^{(g+1)}] - \mathbb{E}[\bar{\alpha}_k^{(g)}] \right)^2$. This must be corrected, and the assumption of equal expected routing coefficients across blocks must be explicitly justified and discussed.
2. **Overclaiming on Intermediate Blocks:** The authors make strong arguments about intermediate block-sharing sizes representing the optimal architectural "sweet spot," but they only test $M=1$ (unshared) and $M=3$ (fully global) in their physical sequential experiments. They must temper these claims or evaluate on a deeper model (e.g., a 12-layer backbone) to test intermediate block sizes ($M=2, 3, 4, 6$) in the physical sequential setup.
3. **Fragility of Physical sequential Merging:** The physical model-merging experiments exhibit extremely high variance (standard deviations of over 22%) and poor absolute performance compared to expert ceilings. This fragility demonstrates that BWS-Router is an empirical heuristic stabilizer rather than a robust, theoretically grounded solution. A deeper theoretical analysis of cascading representation drift is needed to resolve this issue fundamentally.
4. **Toy Experimental Settings:** Both the "Task-Conflict Sandbox" and the 3-layer MLP physical experts are highly simplified, low-dimensional toy systems. To make the findings truly convincing, the authors must validate BWS-Router on standard, large-scale deep architectures (e.g., Vision Transformers or LLMs) on standard downstream datasets.

## Overall Presentation Quality
The overall presentation quality is **Excellent**.
- The figures are informative and clear.
- The equations are laid out elegantly.
- The paper successfully balances high-level conceptual exposition with detailed empirical documentation.
- The writing style is professional, engaging, and direct.

## Potential Impact and Significance
- **Practical Impact:** High. The block-wise sharing scheme provides a simple, direct, and effective method for compressing routing parameters, which is valuable for running resource-constrained dynamic ensembling.
- **Scientific/Theoretical Significance:** Modest. The paper's contribution is primarily an empirical deconstruction and an application of standard weight-tying heuristics. The theoretical modeling is incremental and contains algebraic omissions, meaning it does not fundamentally advance our theoretical understanding of why and how representation spaces align in merged models.
- **Future Directions:** The paper successfully highlights "sequential representation drift" and "high seed-wise variance in physical weight-space ensembling" as critical open challenges, which could inspire valuable future research in stabilizing physical sequential deep model merging.
