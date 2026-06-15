# 3. Soundness and Methodology Evaluation

## Clarity of the Description
The methodology of EdgeMerge is exceptionally well-written, clear, and mathematically rigorous. The paper details the formulation of each stage clearly:
- **Forward-Only Activation Sampling (FOAS)** (Equations 1-4)
- **Scale-Normalized Delta Activation Salience (SNDAS)** (Equations 5-7)
- **Channel-Wise Softmax Gating (CWSG)** (Equation 8)
- **Weight Reconstruction** (Equation 9)
- **Decoupled Scale Routing (DSR)** (Equations 10-12)

The text explains the motivations behind each component clearly, resolving potential logical contradictions (e.g., explaining why base model representations can be reused for experts, and outlining the offline developer workflow).

## Appropriateness of Methods
The proposed methods are highly appropriate for the resource-constrained edge computing setting. 
- **Reusing $X_k^{base}$:** Reusing the base model's internal activations across all experts is a brilliant shortcut. It reduces the visual encoder's deep forward passes to exactly $1\times$, restricting the GPU memory footprint to a single model's size (~100 MB) instead of loading $K$ independent checkpoints. This makes the method mathematically and logistically viable for edge-calibration.
- **SNDAS Frobenius Normalization:** Normalizing activation shifts by the Frobenius norm is an appropriate way to ensure that experts with larger natural activation scales do not disproportionately dominate the channel-routing coefficients.
- **DSR Decoupling:** Decoupling the projection scaling factor ($\lambda_{proj}$) from the static scaling factor ($\lambda_{static}$) is a mathematically elegant solution to resolve the dampening effect caused by softmax normalization.

## Potential Technical Flaws and Critical Insights
Despite the clarity and conceptual elegance of the methods, a deep scientific analysis of the ablation studies (Section 5.3.4) reveals a major technical vulnerability regarding the core contribution of the paper:
1. **Redundancy of Channel-Wise Routing (CWSG):** In the ablation studies, the authors compare the optimal DSR configuration ($\lambda_{static}=0.25, \lambda_{proj}=0.20, \tau=0.10$) against versions that completely disable its core components:
   - **No SNDAS (Frobenius scale normalization):** Achieves **69.58%** average accuracy.
   - **Layer-wise Gating (LWG) (collapsing channel routing into a single layer-wise scalar):** Achieves **69.59%** average accuracy.
   - **Uniform Gating (fixing routing coefficients uniformly to $\alpha_k = 1/K$):** Achieves **69.58%** average accuracy.
   - These ablations reveal that once the scale discrepancy is resolved by **Decoupled Scale Routing (DSR)**, the fine-grained, channel-wise routing weights ($\alpha_k[j]$) computed via activation shifts provide *no empirical benefit* over uniform blending or simple layer-wise scale selection. 
   - Consequently, the performance boost (+0.84% over Task Arithmetic) is entirely driven by DSR (setting $\lambda_{proj} = 0.20$ as a regularizer at the bottleneck while setting $\lambda_{static} = 0.25$ for the rest of the network), rather than the proposed channel-wise activation saliency algorithm.
   - This suggests that while CWSG is conceptually elegant, its mathematical complexity is practically redundant compared to a simple decoupled global baseline (Decoupled Task Arithmetic). The authors deserve immense credit for their intellectual honesty in reporting these ablations, but as reviewers, we must highlight that the core "activation-guided channel routing" mechanism does not deliver the claimed performance-boosting utility.
2. **Coupled Scaling Non-Monotonicity:** In the standard coupled grid sweep (Table 4), the paper notes a sharp drop in performance at intermediate routing temperatures (e.g., $\tau=1.00$ dropping to 51.49% accuracy) which is only resolved by over-scaling or using DSR. This coupled-scaling fragility underscores that the channel routing coefficients are mathematically sensitive and require careful hyperparameter stabilization to avoid degrading representational flow.

## Reproducibility
The reproducibility of the submission is **excellent**:
- **Detailed Experimental Configurations:** Appendix A (Table 1) provides exhaustive details of the visual projection bottleneck dimensions, calibration batch sizes ($B=32$), temperature sweep ranges, optimal hyperparameter values, and hardware/software frameworks.
- **Formulas and Algorithms:** Algorithm 1 clearly outlines the step-by-step pseudo-code for the implementation.
- **Statistical Significance:** The paper provides a rigorous statistical standard error analysis ($SE_{\text{avg}} \approx 0.51\%$ for the 1024-subset multi-task average), confirming that the reported differences are highly stable and representative of the full datasets.
- **Practical Selection Heuristics:** Section 3.7 provides clear heuristics and decision trees to help engineers select strategic choke-point layers on non-CLIP architectures, further supporting empirical transferability.
