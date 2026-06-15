# Experimental Completeness and Baseline Evaluation

The experimental evaluation in this paper is outstandingly thorough and robust. It includes a comprehensive set of baselines, extensive parameter sweeps, and systematic stress tests that validate every major claim in the paper.

### 1. Strengths of the Experimental Design
- **Comprehensive Baseline Comparison**: The authors compare their method against a wide range of relevant baselines:
  - **Static Baselines**: Static Uniform Merging, AdaMerging, and static output-level logit ensembling.
  - **Dynamic Baselines**: Global Linear Router, standard unconstrained L3-Linear (Unregularized and $L_2$-regularized), L3-Softmax (regularized and unregularized), and the quantum wave-superposition SOTA (QWS-Merge).
  - **Training-Free Baselines**: Training-Free Centroid Router, proving that gradient-based calibration under TSAR is necessary to learn task inhibition.
  - **Classic MoE Baselines**: Appendix N includes comparisons with Sparsely-Gated MoE, Switch Transformer gating, and standard softmax-gating, showing that TSAR outpaces them under data-sparse splits.
- **Multi-Seed Evaluation**: Every accuracy figure is averaged over **5 independent random seeds** with standard deviations reported. This is a vital control, especially in the low-data regime ($B_{cal}=64$) where seed variance is typically high.
- **Robustness to Overlapping Manifolds**: The **Subspace Leakage Sweep** ($\eta \in [0.0, 0.4]$) systematically tests the router under non-orthogonal, overlapping feature distributions, showing that TSAR + PCGrad's performance premium remains highly stable and robust.
- **Deployment Stream Audit**: The authors don't just evaluate their method under ideal batching. They simulate three real-world deployment stream configurations (including heterogeneous, mixed-task batches) to identify and successfully resolve the "heterogeneity collapse" phenomenon.
- **Real-World Vision Transformer Validation**: The paper validates TSAR on a real Vision Transformer (`vit_tiny_patch16_224`) by fine-tuning and merging classification heads across 4 tasks, demonstrating a dramatic **+13.90%** absolute accuracy margin improvement over Static Uniform Merging.

### 2. Deep Analysis of Experimental Results
- **TSAR Sensitivity Sweep ($\lambda_{anchor}$)**: Table 2 shows that TSAR is highly robust to hyperparameter tuning. Any penalty value between $0.01$ and $1.0$ yields peak, stable performance (hovering around 54.1%), showing that the method does not require tedious parameter searches.
- **The $B_{cal}=128$ Collapse & PCGrad**: The authors identify a counter-intuitive performance drop at $B_{cal}=128$ for Standard TSAR (dropping to 47.70% from 54.08% at $B_{cal}=64$). They diagnose this as multi-task gradient dominance (where SVHN/CIFAR-10 gradients drag easy-task parameters away from their anchors over long training). Integrating PCGrad successfully resolves this collapse (raising Joint Mean to 49.86% and keeping easy tasks stable), proving the necessity of gradient balancing.
- **Overcoming Heterogeneity Collapse**: Table 4 demonstrates that unconstrained TSAR collapses to 43.10% under mixed-task batching due to coefficient cancellation. Replacing unconstrained linear activation with a **scaled Sigmoid (1.5 headroom)** fully resolves this collapse, achieving **50.80%** under heterogeneous streaming with zero serving-time latency or parameter overhead.
- **Unsupervised PCA vs. Random Gaussian Projection**: Appendix D reveals that under extreme data scarcity ($B_{cal} \le 32$), PCA projection matrices suffer from high sampling noise. Employing a data-independent **Random Gaussian projection** (grounded in the Johnson-Lindenstrauss Lemma) yields a massive **+5.26%** absolute accuracy boost at $B_{cal}=16$ while cutting seed variance by more than half, showing outstanding engineering depth.

### Missing Baselines or Gaps:
There are **no major missing baselines or experimental gaps**. The authors have preemptively addressed every standard review critique:
- **No MoE Baselines?** Addressed in Appendix N.
- **SVHN Experts are under-trained/broken?** Addressed in Appendix O by validating under a highly accurate SVHN expert (91.45% accuracy), showing that relative gains remain structurally identical.
- **Real-world temporal non-stationarity?** Addressed in Appendix I via an online EMA anchor tracking sweep under coordinate drift.
- **PCGrad is too slow for many tasks?** Addressed in Appendix K with a 20-task scalability audit evaluating Stochastic PCGrad ($M=2$) and Task Grouping ($G=4$).

However, a key analytical gap is the lack of discussion regarding the **SVHN performance drop under PCGrad**. In Table 1, SVHN accuracy drops from **15.52%** (standard TSAR) to **13.36%** (TSAR + PCGrad). While PCGrad is highly effective at boosting the joint mean, it does so by suppressing the noisy gradients of the hardest task (SVHN) to protect other tasks from interference, which systematically degrades SVHN's individual performance. Identifying and discussing this trade-off would improve the scientific depth of the optimization analysis.

### Experimental Check Rating: Excellent
The empirical execution of this paper is incredibly thorough. The authors have systematically isolated variables, sweeps, and streaming behaviors, and have preemptively resolved potential objections with dedicated appendix evaluations.
