# 5. Impact and Presentation

## Presentation Quality
The presentation quality is outstanding. The paper is exceptionally well-written, clearly structured, and easy to follow.
* **Cohesive Narrative:** The introduction clearly lays out the three unstudied assumptions, setting a solid foundation for the subsequent chapters. The transitions between the methodology, empirical results, and constructive recommendations are logical and seamless.
* **Mathematical Precision:** All formulas—from dynamic PTQ operators to Adam momentum updates, expectation-based randomized smoothing, and the Hybrid Optimization algorithm—are formalized with meticulous detail.
* **Exemplary Data Visualization:** Tables are clean, professional, and contain complete statistical details (averages, standard deviations, task-wise performance). Appendix A and Algorithm 1 are beautifully integrated.

## Major Strengths
1. **Exceptional Conceptual Novelty:** Exposing the **Cross-Schema Generalization Gap** and the **Low-Capacity Generalization Illusion** challenges the prevailing "state-of-the-art parameter-chasing" narrative in weight-space consolidation, forcing the community to acknowledge realistic hardware deployment constraints.
2. **Exhaustive Empirical Rigor:** Dissecting Q-Merge across four distinct axes (calibration size, cross-schema matrix, optimizer/regularization, and stream distortions) backed by statistical validation over three random seeds.
3. **Flawless Baseline Isolation:** Incorporating the **Quantized AdaMerging** baseline and the **Supervised Calibration** baseline is an exceptional methodological choice. These baselines successfully isolate and prove that direct low-bit STE optimization is unnecessary, and that prediction entropy minimization is structurally fragile.
4. **Constructive Scientific Outlook:** Rather than being purely critical, the paper provides a highly constructive pathway forward. The mathematical formalization of the **Hybrid Optimization Pipeline (Algorithm 1)** and the suggestion of smoothing the landscape prior to discretization using parameter filtering techniques (TIES-Merging/DARE) are invaluable contributions to the community.
5. **Architectural and Subspace Generalizability:** Proactively extending the audit to CNNs (ResNet-18) and low-rank SVD projections provides a highly complete picture of how receptive field structures and ensembling spaces moderate quantization-operator overfitting.

## Areas for Improvement
While the paper is of outstanding quality, the following areas could be explored to make the work even more comprehensive:
1. **Natively-Trained PEFT/LoRA Experts:** The authors use global, post-hoc SVD projection to compress task vectors into a rank-4 subspace as a PEFT proxy. They transparently show that this results in severe capacity degradation (collapsing performance to $13.00\%$), which they term the "Low-Capacity Generalization Illusion". Evaluating natively-trained LoRA experts (which preserve model capacity) remains a critical future step to verify if actual PEFT experts remain resilient to schema shifts.
2. **Joint Weight-Activation Quantization:** The audit is conducted under weight-only quantization (W4). Real-world edge deployments (e.g., on NPUs/DSPs) often mandate joint weight-activation quantization (e.g., W4A8 or W4A4). Evaluating how dynamic activation outliers propagate noise through softmax attention maps and dynamic scale clipping would further enhance deployment realism.
3. **Activation Outlier Mitigation:** Evaluating the integration of outlier-aware activation smoothers (such as SmoothQuant) into the merging pipeline prior to discretization would provide a valuable empirical defense against joint quantization errors.

## Potential Impact and Significance
The potential impact of this paper is highly significant. It will likely trigger a major shift in the model-merging and test-time adaptation literature:
* **Heterogeneous Deployment Standards:** Practitioners will move away from evaluating models under simulated, matched operators, establishing cross-operator validation as a mandatory standard before deploying models on physical hardware (e.g., edge TPUs vs DSPs).
* **Landscape Smoothing Focus:** Researchers will focus on pre-discretization landscape smoothing (e.g., filtering parameter sign conflicts or sign-magnitude updates) rather than relying on unconstrained first-order STE gradient search.
* **Hybrid Optimizer Adaptation:** The proposed Hybrid Optimization Pipeline is highly generalizable and could be widely adopted in other post-training quantization tasks, such as quantized fine-tuning (QLoRA) or low-bit prompt tuning.
This paper serves as an exemplary blueprint for deep learning research auditing.
