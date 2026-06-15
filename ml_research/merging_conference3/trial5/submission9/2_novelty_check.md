# Novelty Check

## Originality and Concept Novelty
- **Conceptual Shift:** Rather than treating model parameters as independent scalars and relying on discrete coordinate-wise heuristics (such as TIES-Merging and Sparse Task Arithmetic), the paper shifts the paradigm to **continuous, joint multi-task spectral filtering**. It views the combined task updates as a unified parameter manifold.
- **Post-Hoc Geometric Subspace Consensus:** While low-rank structures (e.g., LoRA) are commonly enforced *during* training to constrain model updates, GSC-Merge is original in showing how to extract a low-rank consensus subspace *post-hoc* from multiple fully fine-tuned, dense neural networks. It maps these updates onto a shared Grassmannian manifold $\mathbf{Gr}(r, d_{out})$.
- **Addressing the Overfitting-Optimizer Paradox:** The paper introduces a highly original perspective on Offline Few-Shot Validation Tuning (OFS-Tune). It exposes a major vulnerability: unconstrained learning of merging coefficients on tiny calibration datasets (e.g., 16 samples per task) suffers from transductive overfitting to validation noise. The paper shows that projecting updates onto a low-rank consensus subspace acts as an implicit spectral regularizer.

## Theoretical Novelty
- **Eckart-Young-Mirsky Connection:** The paper is the first to directly connect the Eckart-Young-Mirsky Theorem to multi-task model merging, providing a strict upper bound on representational drift under low-rank projection.
- **Implicit Regularization Proof:** The paper provides a mathematical proof that the Grassmannian projection operator is a non-strict contraction under both spectral and Frobenius norms. This formally bounds the optimization space, explaining why GSC-Merge reduces variance and prevents optimization collapse.
- **Projection Direction Study (Appendix C):** Formulates and analyzes three SVD projection schemes (Left/Output-Space vs. Right/Input-Space vs. Bilateral), proving mathematically and representationally why projecting onto output coordinate spaces is optimal for maintaining activation alignment across sequential layers.

## Relationship to Prior Work
- **TIES-Merging / STA:** Differentiated by moving away from hard pruning and sign voting. Instead, GSC-Merge performs a continuous SVD projection, preserving structural correlations and providing closed-form guarantees.
- **AdaMerging / OFS-Tune:** Differentiated by identifying the transductive overfitting of unconstrained coefficient search and offering a spectral solution.
- **LoRA / PEFT:** Differentiated by being a post-hoc merging framework operating on fully dense weights, though Appendix C.3 provides an elegant bridge on how GSC-Merge can be applied to merge LoRA adapters efficiently.

## Conclusion on Novelty
The paper represents a **highly original, mathematically elegant, and timely** contribution. It provides a much-needed theoretical foundation to the largely heuristic field of weight-space model merging.
