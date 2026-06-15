# 3. Soundness and Methodology Check

## Axiomatic and Mathematical Soundness
The mathematical formulation of the framework is overall exceptionally elegant and rigorous:
1. **Geometric Connection:** The paper establishes a solid connection between the representation-space dFIM and the inverse coordinate noise variance under class-conditional Gaussian assumptions:
   $$F_{k, c, j} = \frac{1}{\sigma_{k, j}^2}$$
   This formally justifies using coordinate variances as an analytical coordinate-warping local metric tensor.
2. **Dual-Space Alignment:** The potential conceptual gap of applying a representation-derived metric tensor to warp classifier parameters is formally resolved via the Dual-Space Alignment proof in Appendix A.3. The authors prove that under $L_2$-regularized cross-entropy loss, classifier weights act as dual vectors aligning with class representation centroids, with directional misalignment bounded as:
   $$\left\| \frac{W'_{k, c}}{\|W'_{k, c}\|} - \frac{\mu_{k, c}}{\|\mu_{k, c}\|} \right\| \le \frac{C_0}{\sqrt{N_c}} = \epsilon$$
3. **Statistical Scaling (CSC):** CSC uses extreme value theory to derive the expected maximum of a set of independent random projections onto a sphere:
   $$u'_{k, b} = \frac{u_{k, b}}{\sqrt{2\log C_k / d}}$$
   This is statistically sound and mathematically elegant, and the relaxation to Correlation-Corrected CSC (CC-CSC) in Appendix A.4 adds significant robustness.

## Methodological Vulnerabilities and Critiques

### Critique 1: Pre-calibration Mean-Centering (Fully Integrated and Consistent)
In Section 3.3 and Appendix A.3, the authors emphasize that pre-calibration mean-centering ($z' = z - \bar{z}_{\text{cal}}$) is crucial to eliminate the translation bias introduced by non-zero global centroids in unnormalized representation spaces, thereby restoring perfect dual-space alignment.
- **Resolution:** In our rigorous code audit, we verified that this step has been fully and consistently integrated across the entire codebase—including both the primary script `run_experiments.py` and all eight auxiliary/ablation scripts (such as `test_csc_ablation.py`, `test_rotated_noise.py`, `test_fiosr.py`, etc.). This guarantees evaluation robustness and perfect mathematical alignment under uncentered representation spaces.

### Critique 2: Isotropic Collapse under Non-Axis-Aligned Rotated Noise
The core diagonal Fisher formulation relies on the critical assumption that coordinate noise is aligned with the standard axes. 
- **Analysis:** Under rotated, non-axis-aligned noise, the diagonal Fisher model (FIOSR-Diag) naturally degrades toward the flat Cosine baseline. 
- **Mitigation:** The authors propose FIOSR-Rotated (Oracle K-FAC) and FIOSR-Online (Estimated Cov EVD with shrinkage alpha=0.2) to capture off-diagonal correlations. The implementation of online EVD on shrinkage-regularized covariance matrices on the fly is highly elegant, deterministic, and successfully stabilizes coordinate projections under rotated noise without oracle access.

### Critique 3: Clean Software Engineering and Module Modularity (Main Guard Resolved)
- **Resolution:** The authors have perfectly encapsulated the entire 10-seed main experiment loop inside a proper `if __name__ == "__main__":` guard block in `run_experiments.py`. This completely eliminates any module-import execution side-effects, allowing auxiliary and test scripts (such as `test_fiosr.py`) to import helper data-generation and evaluation functions with zero latency and perfect modularity.

## Soundness Rating: Excellent
The theoretical framework is exceptionally strong and mathematically rigorous. The proofs are clean, and the extreme value calibration (CSC) is highly elegant. With the pre-calibration mean-centering consistently integrated across all auxiliary scripts and the module-import side-effects resolved with proper main guards, the codebase stands as a model of software consistency and mathematical hygiene.
