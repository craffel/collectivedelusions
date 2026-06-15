# 4_experiment_check.md: Critical Evaluation of Empirical Results and Methodology

## Experimental Setup and Datasets
The submission evaluates the proposed Dirichlet-PAC framework in two distinct environments:
1. **The Analytical Coordinate Sandbox (ICS):** A mathematically controlled 14-layer simulation with hidden dimension $D=192$ and $K=4$ task experts. The ICS enables precise sweeps over task-manifold entanglement ($\rho \in [0.0, 0.5]$), non-stationary noise scales ($\boldsymbol{\sigma}$), and representation interference ($\eta$).
2. **Physical Pre-Trained BERT Backbones:** A physical multi-scale evaluation spanning four BERT variants (\texttt{bert-tiny}, \texttt{bert-mini}, \texttt{bert-medium}, \texttt{bert-base-uncased}) with Multi-LoRA adapters fine-tuned on three distinct text classification tasks (Sentiment Analysis, Topic Classification, and Sentence Type Classification).

*Critical Assessment:* The experimental design is exceptionally thorough. By combining a highly controlled synthetic sandbox (which isolates structural variables like entanglement) with physical evaluations on pre-trained transformer backbones, the authors successfully validate both the theoretical properties and systems-level feasibility of their model.

## Comprehensive Baselines
The authors evaluate Dirichlet-PAC against a comprehensive set of nine baselines, representing the state-of-the-art in both static weight-space consolidation and dynamic activation-blending:
- **Weight-Space Consolidation:** Uniform Merging (Task Arithmetic), DARE-Merging, and TIES-Merging.
- **Static Dynamic Routing:** SABLE (Raw Coords), SABLE (SEP-Block), and SABLE (SEP-Block) Norm.
- **Learned Dynamic Routing:** Temp-Only ERM (unregularized) and PAC-ZCA (Gaussian PAC-Bayes over log-temperatures).

*Critical Assessment:* The baselines are highly appropriate. Comparing against prominent weight-space methods (TIES, DARE) and learning-theoretic routers (PAC-ZCA) ensures that the empirical advantages of Dirichlet-PAC are properly contextualized.

## Do the Results Support the Claims?
Yes, the empirical results strongly support the paper's core claims, with several key observations:

1. **Efficacy of Simplex-Constrained Complexity Control:** In the synthetic sandbox (Table 1), Dirichlet-PAC achieves $77.88\%$ (orthogonal) and $76.32\%$ (overlapping) accuracy, outperforming Temp-Only ERM and PAC-ZCA. While the absolute accuracy gain in the sandbox is modest, the major advantage is **optimization stability**: Dirichlet-PAC reduces the standard deviation across seeds ($\pm 1.19\%$ vs. $\pm 1.86\%$ for ERM), confirming that the Dirichlet KL penalty stabilizes temperature calibration.
2. **Catastrophic Overfitting in Physical Networks:** The real-world BERT experiments (Table 3) provide a striking validation of the framework. Under extreme data scarcity ($N_{\text{opt}} = 8$ per task), unregularized routers (Temp-Only ERM and PAC-ZCA) collapse catastrophically, averaging only $67\%$ to $74\%$ accuracy. In contrast, Dirichlet-PAC and PEM-Div maintain outstanding accuracies ($92.00\%$ on BERT-Base and $99.33\%$ on BERT-Mini). This demonstrates that physical transformer layers suffer from severe *representation corruption* when temperatures collapse, and that Dirichlet-PAC's complexity penalty is practically essential.
3. **The Victory of Unsupervised PEM-Div:** The unsupervised PEM-Div variant achieves outstanding performance, scoring $79.43\%$ in the sandbox (outperforming supervised Dirichlet-PAC at $77.88\%$) and $91.33\%$ on BERT-Base. Minimizing individual query entropy while forcing batch-wide routing diversity acts as a powerful transductive semi-supervised regularizer over unlabeled streaming queries.
4. **Vulnerability of Weight-Space Merging:** Static consolidation methods (TIES, DARE) collapse to near-random accuracies ($\approx 45\%$) under representation interference, and TIES-Merging collapses further (down to $36.82\%$) under overlapping manifolds due to sign-filtering. Dirichlet-PAC's dynamic activation-space blending completely bypasses this destruction.

## Areas of Critical Critique and Empirical Gaps

1. **The "SABLE (Raw Coords)" Baseline Ceiling:**
   On orthogonal manifolds ($\rho = 0.0$), the static SABLE (Raw Coords) baseline achieves $79.02\% \pm 0.98\%$ accuracy, slightly outperforming supervised Dirichlet-PAC ($77.88\% \pm 1.19\%$). SABLE uses supervised centroid-based coordinates (which require ground-truth task labels to find centroids), making it inapplicable for label-free edge deployments. While the authors point out that their unsupervised PEM-Div variant successfully bridges this gap ($79.43\% \pm 1.05\%$), it is worth noting that under clean, orthogonal conditions, a simple static prior (SABLE) remains a highly competitive baseline that avoids serving-time optimization altogether.
2. **The "Representation Corruption" Phenomenon in Clean Settings:**
   In Table 2 and Section 4.5, the authors explain that under zero-interference ($\eta = 0.0$), Uniform Merging acts as a distance-preserving "no-op" that achieves a high ceiling ($85.03\%$), whereas dynamic routers perform worse due to representation corruption on noisy queries. The authors show that Dirichlet-PAC's energy-normalization acts as a safety valve, but the fact remains that in highly stationary, noise-free, and orthogonal settings, dynamic routing can introduce unnecessary representation distortion compared to simple static averaging. The authors honestly address this in their "Practitioner's Deployment Guide", which is a highly transparent and helpful discussion.
3. **Low Gains in Sandboxed Overlapping Manifolds:**
   In the synthetic sandbox under overlapping manifolds ($\rho=0.33$), Dirichlet-PAC ($76.32\%$) only outperforms Temp-Only ERM ($75.67\%$) by $+0.65\%$. This small gain indicates that in low-dimensional synthetic spaces, the benefit of the PAC bound over unregularized optimization is minimal. However, this is completely redeemed by the BERT base scale-up, where the gap increases to $+20.67\%$ absolute ($92.00\%$ vs. $71.33\%$), showing that synthetic sandboxes can understate the severity of transductive overfitting compared to physical multi-layer networks.
