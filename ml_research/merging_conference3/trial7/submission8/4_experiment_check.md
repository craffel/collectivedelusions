# 4_experiment_check.md - Experimental Evaluation

As an empirically driven reviewer, I prioritize rigorous, large-scale, and realistic validation over toy mathematical constructs. Evaluated through this lens, the experimental validation in this submission is extremely weak and fails to support the paper's broad claims.

## 1. Lack of Real-World Evaluation and Sandbox Reliance
The most critical empirical weakness is that **every single experiment in the paper is executed on a synthetic, 1-layer toy simulation (the "Isolating Coordinate Sandbox")**. 
- The paper contains zero experiments on real neural network backbones (e.g., Transformers, ResNets, ViTs).
- The paper contains zero experiments on real-world multi-adapter merging setups (e.g., merging LoRA adapters for GLUE, DomainNet, or instruction-following datasets).
- For a paper that markets itself as a "rigorous, large-scale empirical investigation" designed to "restore robustness" for "real-world production streams" and provide a "deployment-ready framework," relying entirely on a 1-layer toy coordinate simulator is a massive and unacceptable empirical gap.

## 2. Artificial and Handicapped Baselines
The authors' evaluation of static model-merging baselines is highly artificial:
- In the "Isolating Coordinate Sandbox," experts reside in strictly disjoint, non-overlapping coordinate blocks.
- Under this coordinate-isolated setup, advanced static merging methods like Task Arithmetic, TIES-Merging, and DARE mathematically reduce to simple Uniform Merging because there are no parameter conflicts across different experts.
- Consequently, the comparison in Table 1 shows dynamic routing beating "Uniform Merging." This comparison is highly misleading because the sandbox is constructed to completely eliminate the core benefits of TIES-Merging and DARE (which are designed precisely to resolve conflicts in highly overlapping weight spaces). In a real weight-merging environment, these advanced baselines would perform far better, and the authors' dynamic routers might struggle with representational interference.

## 3. Discrepancies Between Empirical Results and Written Claims

### A. The SVD Subspace Projection Claim is Empty
The text claims that SVD-Projected Global PFSR "successfully filters out the out-of-subspace noise... successfully bridging the gap to the clean Local PFSR baseline" under overlapping spaces. 
- However, looking at the actual data in Table 3 (and verified via the reproduction script), SVD projection only improves joint classification accuracy by a tiny **+0.10%** (from **70.10%** to **70.20%**) over standard Global PFSR.
- This marginal gain does not "bridge the gap" (the gap to Local PFSR is 1.4%, and SVD projection bridges only 0.1% of it).
- Furthermore, the rank sweep shows that the SVD projection requires a rank $r \ge d = 48$ (the full intrinsic dimension) to work, completely defeating any claim of low-rank compression.

### B. Soft-Confidence Fallback Homogenization is Practically Broken
The text claims that Soft-Confidence Fallback Homogenization ($\beta=0.5$) "completely eliminates this catastrophic dip, maintaining a highly stable and robust accuracy of 64.14%."
- However, Table 4 shows that using this fallback collapses accuracy under the error-free (0% routing error) regime from **73.54%** to **64.72%** (a massive **8.82% absolute accuracy collapse**).
- Gaining 1.80% in accuracy under a very high 30% routing error rate by sacrificing 8.82% in the standard, error-free regime is a terrible trade-off. The fallback mechanism performs barely better than zero-shot Uniform Merging (63.10%), which requires no routing at all.

### C. Hierarchical MBH is Strictly Worse and Collapses Catastrophically
The authors claim that Hierarchical MBH "performs exceptionally well, maintaining 72.28% accuracy at 0.0% error ... shielding the ensembling pipeline from the catastrophic effects of minor routing errors."
- In reality, Table 4 shows that Hierarchical MBH is strictly worse than standard MBH in the low-error regime: at 0% error, it is **1.26% worse** (72.28% vs 73.54%), and at 5% error, it is **0.28% worse** (70.62% vs 70.90%).
- Under high routing errors, Hierarchical MBH collapses catastrophically: at 75% error, it achieves only **44.44%** accuracy, which is **19.44% absolute worse** than standard MBH (63.88%) and **18.66% absolute worse** than simple Uniform Merging (63.10%).
- The authors' written claims are directly contradicted by their own empirical data. H-MBH does not shield the pipeline; it amplifies errors.

## 4. Evaluation of Calibration Sample Complexity $N$
The authors sweep $N \in \{16, 32, 64, 128, 256, 512\}$. While this is a wide sweep, the overfitting of the parametric router is artificially severe in their sandbox. Because the parametric router must map the global $192$-dimensional representation to $4$ tasks, it has to learn to ignore the random Gaussian noise in the other $3$ non-active blocks ($3 \times 48 = 144$ dimensions of pure noise). Under small $N$, the router inevitably overfits to this high-dimensional noise. In real neural networks, representations are overlapping and lower-dimensional, meaning that standard regularizers would likely perform much better than shown in Figure 2.
