# Evaluation Part 3: Soundness and Methodology

## Clarity of the Description
The methodology is described with a high level of mathematical clarity. Equations 2, 3, and 4 define the convex barycentric simplex formulation. Section 3.4 outlines both the symmetric and asymmetric Mean-Field Proximity Penalty. The step-by-step optimization procedure is logically laid out, including the ray-scaling projection.

## Appropriateness of Methods
1. **Barycentric Formulation:** A convex combination of weights is a sound way to bound the Frobenius norm of the merged model, successfully keeping it within the convex hull of the base and expert weights.
2. **Ray-Scaling Projection:** Using $\lambda_k \leftarrow \lambda_k / \sum \lambda_j$ to project back to the simplex boundary is computationally simple and preserves relative directional ratios.
3. **Mean-Field Proximity Penalty:** Drawing inspiration from mean-field theory, regularizing towards a uniform prior ($\frac{1}{K+1}$) is a reasonable prior when no task information is known beforehand.

## Potential Technical Flaws and Conceptual Inconsistencies
1. **Contradiction on Sparsification:** The authors explicitly argue in Section 3.5 (Step 4) that standard Euclidean orthogonal projection onto the simplex is undesirable because of its "strong sparsification effect, which tends to push multiple coordinates to exactly zero." They claim that their ray-scaling projection avoids this hard sparsification. However, in Section 4.5, they report that the converged coefficients for both SVHN ($\lambda_5$) and MNIST ($\lambda_6$) are **exactly 0.0000**. This empirical outcome directly contradicts their theoretical motivation for using ray-scaling, as hard sparsification still occurs.
2. **Unconstrained Scaling Outperforming Constrained BPAM:** In Table 1, "Unconstrained Scaling" outperforms BPAM-Static by +2.30% absolute (71.51% vs 69.21%), and "Unconstrained + Head Tuning" outperforms BPAM-Full by +1.90% absolute (77.12% vs 75.22%). The authors argue that the constraints are "critical, parameter-free structural safeguards" against activation collapse and parameter explosion. However, they provide no empirical evidence showing that unconstrained scaling actually experiences these failures under any tested condition. Without such evidence, the practical utility of the proposed constraints is highly questionable given their clear performance cost.

## Reproducibility
The LaTeX source provides extensive details regarding hyperparameters (Adam, learning rate $10^{-3}$, 200 epochs, batch size 32, $\beta = 10^{-2}$), evaluation datasets (8 image classification benchmarks), and converged parameters. The optimization loop is self-contained and mathematically explicit, which indicates high reproducibility. However, the exact code repository is not provided in the text.
