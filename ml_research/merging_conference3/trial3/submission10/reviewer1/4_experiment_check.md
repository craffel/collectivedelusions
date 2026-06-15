# 4. Experimental Check

## Strengths of the Experimental Setup
1. **Diverse Baselines:** The paper includes a thorough selection of baselines, including static uniform Task Arithmetic, unconstrained AdaMerging, explicitly regularized (Total Variation and L2) AdaMerging, and the monomial subspace method PolyMerge.
2. **Mathematically Controlled Environments:** The two simulated environments are highly detailed. Model II is particularly impressive as it incorporates layer sensitivity scaling, non-diagonal covariance (inter-layer couplings), highly non-convex Rastrigin-type loss surfaces, and multi-scale transductive noise (alternating, white Gaussian, and Brownian drift).
3. **Robustness Over Multiple Seeds:** Averaging results across 30 independent random seeds ensures statistical significance and reduces the likelihood of reporting cherry-picked results.
4. **Learning Rate Sensitivity Sweeps:** Systematically sweeping the base learning rate is an excellent way to evaluate the claimed optimization stability benefits, directly linking the theoretical condition number analysis to practical robustness.

## Real-World Physical Validation Check
The authors deserve praise for evaluating ChebyMerge on a physical model: a pre-trained CLIP ViT-B/32, merging task vectors from actual MNIST and SVHN experts. However, a critical inspection of Table 4 ("Real-World Physical CLIP ViT-B/32 Merging Results") reveals a highly significant finding that limits the practical impact of the proposed test-time adaptation:

* **TTA Fails to Outperform Static Baseline:** Every single adaptive merging method—including ChebyMerge and ChebyMerge-CSD—performs **worse** than the non-adaptive Task Arithmetic baseline (which achieves **81.50%** accuracy). 
  - AdaMerging (Unconstrained) drops to **78.00%**.
  - PolyMerge ($d=2$) collapses to **70.50%**.
  - ChebyMerge ($d=2$) drops to **74.00%**.
  - ChebyMerge-CSD ($d=2$) drops to **75.50%**.
* **Implications:** While ChebyMerge-CSD ($d=2$) successfully mitigates the severe collapse of PolyMerge (achieving 75.50% vs 70.50%, a +5.00% absolute improvement), it still suffers from a **-6.00% absolute regression** compared to static uniform Task Arithmetic. This indicates that under realistic conditions (100 images in the TTA stream), unsupervised TTA based on prediction entropy minimization consistently degrades generalization accuracy due to the Overfitting-Optimizer Paradox. Thus, the practical utility of doing unsupervised test-time adaptation for model merging under these constraints is questionable, as static merging remains superior.

## Performance Gains over PolyMerge are Marginal (in Stable Regimes)
In the simulated environments and under stable learning rate regimes, the performance differences between ChebyMerge and PolyMerge are extremely small:
- **Model I (Table 2):** At quadratic degree $d=2$, PolyMerge achieves **87.70%** and ChebyMerge achieves **87.71%**. They are statistically identical.
- **Model II (Table 3):** At $d=2$, PolyMerge achieves **85.39%**, standard ChebyMerge achieves **85.25%**, and ChebyMerge-CSD achieves **85.48%** (an improvement of only **+0.09%** over PolyMerge).
- **Learning Rate Sweep (Table 5):** Under a low learning rate ($\eta = 10^{-4}$), PolyMerge achieves **81.00%** and ChebyMerge-CSD achieves **81.50%**. 

**Conclusion:** The main empirical advantage of ChebyMerge is **optimization robustness and safety** (preventing catastrophic divergence under high learning rates, as seen in Table 5 where PolyMerge collapses to 66.00% at $\eta = 2 \cdot 10^{-2}$ while ChebyMerge-CSD maintains 70.00%) rather than direct performance improvements. In well-tuned or low learning rate regimes, the standard monomial basis (PolyMerge) performs virtually identically to the Chebyshev basis, since its ill-conditioning acts as an accidental regularizer.
