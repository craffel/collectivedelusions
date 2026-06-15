# 4. Experiment Check

## Critical Evaluation of the Experimental Setup
The experimental setup is exceptionally rigorous, well-designed, and transparent:
* **Models and Datasets:** The authors employ a standard compact Vision Transformer (`vit_tiny_patch16_224`) fine-tuned across four distinct classification domains: MNIST, FashionMNIST, CIFAR-10, and SVHN. This setup directly mirrors the benchmark environment of prior work, enabling a direct and fair deconstruction of their claims.
* **Baselines:** The paper compares RLR against a comprehensive list of baselines:
  1. *Individual Experts (Ceiling):* Represents the empirical upper bound.
  2. *Uniform Merging (Task Arithmetic):* A static baseline.
  3. *OFS-Tune:* A state-of-the-art supervised static merging method.
  4. *AdaMerging:* A state-of-the-art unsupervised test-time adaptation method.
  5. *Linear Router (Classical, unregularized):* The basic classical gating baseline.
  6. *QWS-Merge (Reported & Local):* Both the reported numbers from Vance et al. (2025) and a locally re-implemented baseline evaluated on the exact same expert weights to control for checkpoint-induced performance discrepancies. This represents the gold standard of comparative empirical evaluation.

## Supporting Evidence for Central Claims
The empirical results provide overwhelming support for the paper's main claims:
1. **Deconstructing SVHN Collapse (Refuting QWS-Merge):** The central claim of QWS-Merge was that classical linear routing is structurally limited and collapses on SVHN ($15.30\%$). Table 1 and Table 2 completely debunk this. With a parsimonious configuration, the classical unregularized router achieves $94.87\%$ SVHN and $95.46\%$ Joint Mean, significantly outperforming the local QWS-Merge baseline ($88.40\%$ SVHN, $90.03\%$ Joint Mean). 
2. **Diagnosing Prior Work's Collapse:** Table 2 identifies the specific sub-optimal configuration choices in prior work that triggered the collapse (deep task-warped representation routing, excessive learning rates, and over-optimization). This is a highly valuable, constructive diagnostic.
3. **Statistical Robustness (Multi-Seed Sweep):** Section 4.3 sweep over 5 random seeds demonstrates that both classical linear routing and RLR consistently achieve stable, high-performance convergence across all seeds (average SVHN accuracy is $91.20\% \pm 1.85\%$ and $91.20\% \pm 1.84\%$ respectively, with no collapse observed). This statistically confirms that simple classical routing is robust.
4. **Resilience to Heterogeneity Collapse:** Table 3 and Figure 2 show that RLR consistently maintains an accuracy buffer over the unregularized Linear Router as the evaluation batch size increases ($+1.37\%$ at $B=16$ and $+1.88\%$ at $B=256$), validating the stabilizing effect of weight decay and temperature scaling in mixed-task streams.
5. **Robustness to Hyperparameters and Routing Layer:** Table 4 (ablation of routing source layer) and Figure 3 (2D sensitivity heatmap) show that RLR is highly insensitive to hyperparameters ($\alpha$ and $T$) and converges successfully regardless of whether the routing signal is extracted from early, middle, or late layers.

The experimental section is exemplary. The authors do not cherry-pick results, they transparently report average performances across seeds, they highlight that RLR and unregularized routing are statistically indistinguishable in homogeneous settings, and they clearly lay out the structural trade-off under heterogeneous mixed-task evaluation streams. This extreme honesty and thoroughness is a major asset of the paper.
