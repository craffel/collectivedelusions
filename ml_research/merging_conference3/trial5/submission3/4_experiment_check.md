# 4. Experimental Setup and Evaluation Check

## Validity of the Experimental Setup
The experimental setup designed in this paper is highly valid, controlled, and directly addresses the core research questions:
- **Consistent Model Backbone:** Using the unified compact Vision Transformer ($\mathtt{vit\_tiny\_patch16\_224}$) with 5.7M parameters provides a clean, well-established, and standard model architecture for parameter fusion and model merging benchmarks.
- **Diverse Multitask Domains:** Combining MNIST, FashionMNIST, CIFAR-10, and SVHN forms a standard and highly challenging multitask vision benchmark. SVHN, in particular, represents a highly challenging out-of-distribution task with high domain variance, making it an excellent task for evaluating representation collapse and task conflict.
- **Rigorous and Extensive Baselines:** The authors evaluate RLR against an outstanding suite of baselines, including both static and dynamic model merging paradigms (Uniform Merging, OFS-Tune, AdaMerging, and the classical unregularized Linear Router). 

## High Rigor in Baseline Re-implementation
The inclusion of a **local re-implementation of QWS-Merge** is a hallmark of rigorous empirical science. Often, papers compare proposed methods against published, cross-paper numbers from other works, which can lead to unfair comparisons due to differing expert checkpoints, training configurations, or data splits. By locally training the QWS-Merge baseline on the exact same expert weights and under identical conditions as their classical router and RLR, the authors ensure a perfectly unified and fair comparison. 

The empirical findings are highly striking:
- QWS-Merge (reported in Vance et al., 2025) achieved a mere $59.32\%$ Joint Mean and $31.60\%$ SVHN.
- QWS-Merge (Local Baseline) improves to $90.03\%$ Joint Mean and $88.40\%$ SVHN under identical conditions.
- However, the classical unregularized Linear Router achieves **$95.46\%$ Joint Mean and $94.87\%$ SVHN** on seed 42.
- Robust Linear Routing (RLR) achieves **$94.68\%$ Joint Mean and $94.36\%$ SVHN** on seed 42.

This local evaluation definitively deconstructs QWS-Merge's core thesis. It proves that its wavefunction metaphors and complex projection operators do not offer any empirical performance benefit, and that classical linear routing is inherently superior and highly robust when properly configured.

## Diagnostic and Statistical Rigor
The paper exhibits exceptional scientific rigor through two major additions:
1. **Baseline Configuration Diagnostic Table (Table 2):** To guide future researchers and diagnose why Vance et al. (2025) reported a catastrophic collapse of classical routing on SVHN ($15.30\%$), the authors provide a systematic diagnostic comparison. They identify that extracting representations from deep task-warped layers, using excessive learning rates ($>0.1$), and over-optimizing for too many steps ($>1000$ steps) in the absence of regularization are the primary causes of the collapse. Under their stable parsimonious configuration, classical linear routing achieves outstanding, stable convergence.
2. **Multi-Seed Statistical Sweep:** Rather than reporting results on a single, cherry-picked calibration seed, the authors execute a multi-seed sweep over 5 random calibration seeds $\{42, 101, 202, 303, 404\}$. Across all random draws, the classical unregularized router achieves an average Joint Mean accuracy of $91.53\% \pm 0.41\%$ and an SVHN accuracy of $91.20\% \pm 1.85\%$ (with no instance of collapse). RLR achieves a statistically identical average performance of $91.46\% \pm 0.42\%$ Joint Mean and $91.20\% \pm 1.84\%$ SVHN. This multi-seed check provides statistical proof that classical linear routing is highly robust on average.

## Mixed heterogeneous streams and Trade-offs
Evaluating the methods under mixed heterogeneous test streams across batch sizes $B \in \{1, 16, 256\}$ represents a highly realistic deployment scenario:
- **Resilience to Heterogeneity Collapse:** As the batch size increases, dynamic methods suffer from batch-level coefficient averaging (heterogeneity collapse). RLR demonstrates superior resilience to this collapse compared to the unregularized Linear Router, consistently maintaining an accuracy buffer (e.g., at $B=256$, RLR achieves $75.03\%$ compared to $73.15\%$, a $+1.88\%$ absolute benefit). This directly confirms the stabilizing and smoothing effects of $L_2$ weight regularization and temperature scaling under representation shift.
- **Intellectually Honest Trade-offs:** The authors are highly transparent in noting that static supervised techniques like OFS-Tune remain unaffected by batch size and maintain a robust $86.23\%$ across all evaluations, outperforming both RLR and the unregularized router at larger batch sizes ($B=16$ and $B=256$). This level of intellectual honesty is highly commendable, as it provides clear, honest, and actionable deployment guidelines for practitioners, raising the overall scientific utility of the paper.

## Ablation and Sensitivity Analysis
The paper is highly complete with:
- **Ablation Study on Representation Source Layer (Table 4):** Systematically compares extracting representations from early, middle, and late layers. Shows that routing from any layer converges successfully under their stable, parsimonious training loop, with deeper layers yielding slightly higher Joint Mean accuracies ($95.41\%$ at Late block 11) because deeper blocks capture richer semantic representations.
- **2D Hyperparameter Sensitivity Sweep (Figure 4):** A comprehensive grid search over the regularization coefficient $\alpha \in [0.0, 0.02]$ and softmax temperature $T \in [1.0, 5.0]$, showing that the performance is highly stable and insensitive to wide sweeps. This confirms that RLR's performance is not a result of sensitive hyperparameter tuning.
