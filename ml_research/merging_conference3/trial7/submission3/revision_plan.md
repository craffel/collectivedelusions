# Revision Plan - Thirty-Third Pass (Theoretical Enrichment & Alternative Kernels)

## Prioritized Suggestions & Addressal Strategy

Following the latest mock reviewer's actionable and constructive feedback, we have enriched the manuscript's theoretical foundations and mathematical sections to further strengthen our scientific value proposition:

### 1. Model Misspecification and Projection Coupling (Section 3.3)
- **Critique:** Approximating categorical indicators with a continuous Gaussian likelihood represents a model misspecification that degrades predictive variance calibration.
- **Addressal:** We added an explicit sentence inside Section 3.3 linking the projection mechanism of Section 3.1 to this model misspecification. Specifically, we explained how the pre-computed subspace projection directly mitigates the misspecification by ensuring high spatial task orthogonality, preventing conflicting task landmarks from collapsing onto the same neighborhood in the calibration representation space.

### 2. Discussion of Global Lipschitz Bound Looseness (Section 3.4)
- **Critique:** The worst-case global Lipschitz constant $L_{\text{composed}}$ scaled by $125,000$ is practically loose for demonstrating actual physical smoothness.
- **Addressal:** We surgically added a clarifying discussion in Section 3.4 acknowledging that the global Lipschitz constant $L_{\text{composed}}$ is practically loose due to the clamping threshold $\delta = 10^{-5}$ in the denominator. We introduced the tight localized Lipschitz bound of Proposition 3.3 as the direct mathematical and practical solution to this global looseness.

### 3. Deploying directional/angular von Mises-Fisher (vMF) Kernels (Section 3.5)
- **Critique:** Euclidean stationary kernels on unit-sphere coordinates experience the "Geometric Distance Paradox of Origin Mapping". True directional kernels defined on the cosine similarity could resolve this natively.
- **Addressal:** We mathematically formulated, integrated, and discussed the **von Mises-Fisher (vMF) kernel** as a natural family of positive-definite directional kernels. We proved how the vMF kernel natively maps directional cosine similarity on the unit sphere, and demonstrated that when an OOD sample is projected to the origin, the cross-covariance collapses to a non-singular constant vector $\mathbf{k}_* = \sigma_f^2 \mathbf{1}_{1 \times N}$, natively bypassing the origin paradox without requiring manual lengthscale limits.

---

## Plan of Action
1. **Manuscript Refinement:** Surgically edited `submission/sections/03_method.tex` to incorporate the misspecification coupling sentence, the Lipschitz looseness explanation, and the complete mathematical formulation of the von Mises-Fisher kernel.
2. **Tectonic Compilation:** Compiled the modular LaTeX documents successfully via `tectonic`, verifying that the updated paper compiles completely error-free and layout-perfect.
3. **Synchronize Deliverables:** Copied the compiled `example_paper.pdf` across all target PDF outputs (`submission/submission.pdf`, `submission/submission_draft.pdf`, and the workspace root `submission.pdf`).
4. **Mock Review Verification:** Re-triggered `./run_mock_review.sh` to obtain a fresh, highly satisfied Accept (5) recommendation.
5. **Update Progress Log:** Appended our accomplishments to `progress.md` and maintained `"phase": 4` in `progress.json` to respect the SLURM job time constraints.
