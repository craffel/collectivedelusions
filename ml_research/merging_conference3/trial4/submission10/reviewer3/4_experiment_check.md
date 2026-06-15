# 4. Experiment Check

## Experimental Setup & Datasets
The experimental setup is exceptionally well-designed for testing high-conflict model merging. Selecting a diverse multi-task benchmark spanning MNIST, FashionMNIST, CIFAR-10, and SVHN ensures a highly challenging, heterogeneous set of visual distributions. Furthermore, the use of a highly compact Vision Transformer backbone (`vit_tiny_patch16_224`, 5.7M parameters) provides the restricted parameter capacity necessary to trigger and analyze "catastrophic representational collapse". The authors also successfully addressed potential "fake expert" concerns by training each expert to a high convergence ceiling (average $70.52\%$).

## Baselines
The paper compares against an appropriate suite of baselines, including static uniform merging (Task Arithmetic), unsupervised test-time adaptation (AdaMerging), supervised static optimization (OFS-Tune), and a classical dynamic baseline (Linear Router). 

However, as highlighted in the Soundness evaluation, the comparison against the Linear Router baseline has a critical flaw: the Linear Router is global (applying identical coefficients across all layers), whereas QWS-Merge is layer-wise. This mismatch prevents a clean ablation of the wave-like cosine formulation.

## Do the Results Support the Claims?
Overall, the empirical results strongly support the paper's core claims, with minor trade-offs:
1. **Resolution of Representational Collapse:** The results in Table 1 support the claim that QWS-Merge resolves catastrophic representational collapse on compact backbones, elevating joint mean accuracy from $49.35\%$ (Uniform Merging) to $59.32\%$.
2. **Wave-Like Subspace Regularization:** The claim that QWS-Merge's cosine phase projections provide robust regularization under extreme task conflict is strongly validated by the SVHN results. Under extreme domain shift (SVHN), the unconstrained Linear Router catastrophically collapses to $15.30\%$ (near-random), while QWS-Merge preserves $91.5\%$ of the specialized expert capacity at **31.60\%** (outperforming the Linear Router by $+16.30\%$ absolute). This is a compelling and significant empirical validation of the proposed method's robustness.
3. **The Capacity-Regularization Trade-Off:** The results also expose a notable trade-off that should be more prominently discussed. On low-conflict tasks (MNIST, FashionMNIST, and CIFAR-10), the unconstrained Linear Router actually outperforms QWS-Merge by significant margins (MNIST: $91.20\%$ vs $77.60\%$; FashionMNIST: $67.00\%$ vs $63.50\%$; CIFAR-10: $71.40\%$ vs $64.60\%$). This illustrates that QWS-Merge's heavy wave-like regularization acts as a double-edged sword, sacrificing capacity on simple tasks in exchange for robust prevention of collapse under extreme conflict.
4. **Heterogeneity Collapse:** The results in Table 2 and Figure 2 support the claims regarding batch size and task heterogeneity. The performance of both dynamic methods drops severely at $B=256$ due to batch-level task mixing (averaging out coefficients), with QWS-Merge maintaining a slight edge ($48.70\%$ vs $47.70\%$). This is an incredibly honest and high-value scientific finding.
