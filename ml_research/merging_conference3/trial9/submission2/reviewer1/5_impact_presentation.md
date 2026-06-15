# 5. Impact and Presentation

## Major Strengths
1. **Rigorous Theoretical Grounding:** Unlike typical empirical ensembling papers, this work provides a solid mathematical foundation for "activation dilution" using covariance modeling, proving that dynamic thresholding acts as a hard-thresholding covariance regularizer.
2. **Exhaustive Systems-ML Co-Design:** The paper demonstrates deep awareness of edge hardware constraints, incorporating a detailed Roofline Model analysis to prove that expert serving is strictly memory-bandwidth-bound ($\text{OI}_{\text{expert}} = 0.5$ FLOPs/byte) and analyzing LPDDR5 queue saturation under massive expert pools.
3. **Microsecond-Scale closed-loop Control:** The closed-form control functions ($M(C_{\text{budget}})$, $\theta(C_{\text{budget}})$) are training-free and arithmetic-only, enabling instant adaptation to hardware interrupts (e.g., thermal throttle warnings, battery saver events) without virtual memory paging or graph re-compilation.
4. **Exceptional Writing and Presentation:** The paper is beautifully written, with precise terminology, professional data-flow diagrams, comprehensive tables, and thorough sensitivity sweeps on routing temperature and pruning thresholds.

## Areas for Improvement
1. **Coupled Covariance Modeling:** The theoretical proof in Appendix A.1 should be corrected to account for non-zero cross-covariance terms of secondary experts caused by the shared input noise perturbation ($\epsilon_l$). Addressing this coupling strengthens, rather than weakens, the theoretical case for dynamic pruning.
2. **Smooth Control-Loop Alternatives:** The authors should analyze or propose continuous, smooth alternatives to the step-function $M(C_{\text{budget}})$ (such as a sigmoid-gated top-$M$ selection) to eliminate transient latency and capacity jitter in real-time OS control loops.
3. **Quantization Noise Analysis:** Expand the discussion on how low-precision quantization (INT8/INT4 in Q-SPS) shifts early representation coordinates, and explain how Intra-Task Dispersion Calibration (IDC) mathematically mitigates this coordinate shift to preserve routing precision.
4. **HMD-GMM Level-1 Theoretical Bounds:** Provide a theoretical bound or proof on the Level-1 OOD misclassification rate under HMD-GMM when macro-domain manifolds exhibit partial overlap.

## Overall Presentation Quality
The presentation quality is **excellent**. The paper is highly polished, clearly organized, and adheres to strict academic formatting conventions. The inclusion of a detailed notational glossary, system-level OS driver roadmaps, and bare-metal physical profiling designs in the appendix makes the submission exceptionally robust and publication-ready.

## Potential Impact and Significance
The paper has **high potential significance** for the edge AI and TinyML community. As specialized PEFT models continue to proliferate, serving multiple downstream tasks on resource-constrained devices without suffering from either heterogeneity collapse (static merging) or memory-bus choking (dynamic ensembling) is a critical bottleneck. RB-TopM provides a highly practical, training-free, and mathematically rigorous solution that can be compiled and deployed on physical edge hardware immediately. The combination of dynamic hardware-governed control loops, GMM-based early OOD filtering, and robust ZCA routing establishes a new, sustainable paradigm for low-power edge model serving.
