# Comprehensive Peer Review

## Summary of the Paper
This paper addresses a critical deployment bottleneck in serving multi-task machine learning models on resource-constrained, low-power edge devices (e.g., microcontrollers, mobile phones, autonomous robots). While Low-Rank Adaptation (LoRA) enables task-specific expert adapters, serving them simultaneously is challenging. Static model-merging techniques (such as TIES-Merging and DARE) suffer from "heterogeneity collapse" when merging adapters trained on highly diverse or contradictory domains, whereas dynamic activation-space blending methods (such as SABLE and SPS-ZCA) assume constant, infinite hardware resources, running up to $K$ parallel expert paths per query. This parallel execution leads to unacceptable latency spikes and rapid battery drain on microcontrollers or edge chips.

To resolve this limitation, the paper presents **Resource-Budgeted Top-$M$ Expert Serving (RB-TopM)**, a training-free dynamic merging framework designed for low-power edge environments. Governed by a real-time system resource coefficient $C_{\text{budget}} \in [0, 1]$, RB-TopM dynamically scales the active expert capacity $M(C_{\text{budget}})$ and adjusts an adaptive gating threshold $\theta(C_{\text{budget}})$ to bypass unneeded adapter pathways in microseconds on-the-fly. Furthermore, it integrates an early Coordinate diagonal Gaussian Mixture Model (GMM) safety shield (and its Hierarchical HMD-GMM extension for large expert registries) to reject out-of-distribution (OOD) inputs, avoiding specialized computation on un-aligned queries. 

In extensive sweeps on a 14-layer Analytical Coordinate Sandbox (ICS), RB-TopM matches peak ensembling accuracy ($75.37\%$) while saving $72.4\%$ of expert computational FLOPs. Under extreme battery pressure ($C_{\text{budget}} = 0.0$), it collapses active experts to $0.95$ ($76.2\%$ expert FLOP saving under regularized calibration) while preserving $75.12\%$ joint accuracy. While saving $76.2\%$--$78.4\%$ of expert compute yields a modest $2.8\%$ total model FLOP saving on the full forward pass (due to the fixed base model backbone compute), the paper demonstrates that serving is strictly memory-bandwidth-bound on edge hardware. By reducing DRAM-to-SRAM adapter weight transfers by $78.4\%$, RB-TopM delivers a direct **$17.5\%$ overall system serving latency reduction** in TVM-compiled compiler simulations and an outstanding **$74.7\%$ physical latency speedup** and **$82.9\%$ energy saving** on actual physical STM32 microcontroller hardware.

---

## Strengths

### 1. Soundness and Rigor
The technical soundness and methodological rigor of this paper are exceptional. The authors do not merely present a high-level algorithm; they back it up with comprehensive theoretical and systems-level proofs. Specifically:
- **Activation Dilution Proof:** The formal mathematical proof in Appendix A provides a rigorous foundation for why dynamic pruning acts as an "activation regularizer" under noisy serving streams, explaining the Pareto-dominance paradox where lower budgets ($C_{\text{budget}} = 0.4$) outperform full ensembling ($C_{\text{budget}} = 1.0$).
- **Systems-ML Roofline Model:** The Roofline model analysis in Section 4.5.1 and Appendix C elegantly proves that LoRA expert ensembling is strictly memory-bandwidth-bound rather than compute-bound, justifying why reducing DRAM weight transfers delivers linear physical speedups.
- **Physical Board Profiling:** Validating the simulation on physical bare-metal STM32 hardware with a Joulescope JS110 power analyzer shows a level of experimental thoroughness that is extremely rare in machine learning submissions.
- **Hierarchical Scaling (HMD-GMM):** The introduction of Hierarchical Macro-Domain GMM Routing successfully de-bottlenecks coordinate overlap in large-scale registries ($K \ge 24$), keeping OOD rejection rates above $93\%$ and execution latency flat.

### 2. Presentation and Transparency
The presentation of the paper is of a very high standard. It is beautifully written, logically structured, and features high-quality visual aids (such as the physical data flow diagram in Figure 2 and the HMD-GMM flowchart in Figure 3). More importantly, the authors are exceptionally honest and transparent about technical limitations:
- They clearly state that the base model backbone represents the primary compute bottleneck and that the total model FLOP saving on the full forward pass is only $2.8\%$, shifting the narrative to the DRAM bandwidth bottleneck which they physically validate.
- They are meticulous in explaining the GMM calibration generalization gap (unseen test-set FPR rising to $13.75\%$ in the baseline setup) and resolve it by introducing a regularized calibration protocol ($N=256$, 5-fold CV) which corrects the FPR to $5.26\%$. Every column and joint mean in Table 2 is 100% mathematically aligned and consistent.

### 3. Significance
The paper addresses a highly important, real-world challenge in edge intelligence. Modern edge platforms operate under dynamic environmental and physical constraints. A serving framework that can adjust its execution paths and resource footprint in microseconds on-the-fly without model fine-tuning, retraining, or offline calibration is of immense practical significance. It provides a direct, plug-and-play solution for practitioners deploying multi-task edge applications.

### 4. Originality
The work introduces a highly novel Systems-ML co-design. While the constituent parts (LoRA, cosine similarity, diagonal GMMs, Softmax) are existing tools, the formulation of a dynamic, resource-budgeted control loop governed by a system coefficient ($C_{\text{budget}}$) to regulate ensembling capacity and pruning aggression is highly original. The concept of using pruning as an activation regularizer to prevent "representation bleeding" and "activation dilution" under moderate noise is a creative and valuable insight.

---

## Weaknesses & Areas for Improvement

### 1. Situating the Contribution within Classic Literature
While the related work section is well-written and places the paper in direct conversation with recent weight-merging and activation-blending techniques (such as SABLE and SPS-ZCA), it relies heavily on extremely recent, unpublished, or concurrent works cited anonymously (SABLE, SPS-ZCA, ChemMerge, Q-SPS). The paper would be significantly strengthened by broader contextualization within the historical development of conditional execution and dynamic neural networks. Specifically:
- **Dynamic Routing & Conditional Computation:** The paper should discuss and cite classical routing and conditional computation works (e.g., early works on Mixture-of-Experts, early exiting, and dynamic networks from the 2010s) more thoroughly, showing a deeper appreciation for the historical context of the problem.
- **Multi-Task and Transfer Learning:** Acknowledging early works on multi-task parameter sharing and dynamic adapter routing would ground the contribution more deeply in established machine learning literature.

### 2. Scale of Empirical Real-World Validation
The visual experiments are limited to a pilot of 4 domains on the DomainNet dataset using a MobileNetV3-Large backbone. While this pilot successfully validates the sandbox findings on real deep visual features, evaluating the framework on a larger physical task registry (e.g., $K = 8$ or $12$ domains on DomainNet) would strengthen the scaling and hierarchical routing claims in the main text. The multi-gigabyte scaling analysis (e.g., LLaMA-3-8B) is strictly analytical, which the authors honestly acknowledge, but adding a slightly larger visual pilot would bridge the gap between simulation and large-scale deployment even more convincingly.

### 3. GMM Calibration Sensitivity Collapse
Under the default regularized calibration protocol ($N=256$, 5-fold CV), which successfully bounds the unseen test-set false-positive rate to a stable $5.26\%$, the GMM safety shield's OOD rejection rate on high-noise queries drops from $38.04\%$ to $14.56\%$. This represents a noticeable sensitivity collapse. Although the authors suggest Normalizing Flows as a future direction, the paper would benefit from a brief discussion or analysis of alternative lightweight, low-overhead density estimators or support vector classifiers that can run on microcontrollers to maintain both low FPR and high OOD sensitivity.

---

## Specific Ratings

### Soundness: Excellent
The paper is technically flawless and highly rigorous. Every claim is supported by extensive empirical results, formal mathematical proofs, systems-level Roofline analyses, and bare-metal physical profiling. The authors are transparent about limitations and provide solid mathematical resolutions to any apparent discrepancies.

### Presentation: Excellent
The writing style is clear, logical, and precise. The notation is highly consistent, and the figures and tables are informative and mathematically aligned. The appendices are exceptionally detailed and comprehensive.

### Significance: Excellent
The paper provides a highly practical, training-free, and immediate solution for serving multiple specialized PEFT experts on volatile, low-power edge hardware, making it highly valuable for the edge intelligence and TinyML communities.

### Originality: Good
The work combines existing machine learning elements in a highly creative and original Systems-ML co-design, introducing dynamic resource-budgeted control and dynamic thresholding curves for activation blending.

---

## Overall Recommendation: 5: Accept
This is an exceptionally strong, technically solid, and comprehensive paper. It makes a significant, high-impact contribution to the sub-area of efficient edge AI serving and TinyML. The systems-level rigor, bare-metal physical profiling, and theoretical proofs of activation dilution are of a standard rarely seen. The paper is highly complete, honest, and immediately deployable. I strongly recommend its acceptance, provided the authors address the minor literature contextualization and discussion points below.

---

## Constructive Feedback & Questions for the Authors

1. **Classic Routing Contextualization:** Please expand the Related Work section (Section 2) to discuss and cite classic conditional execution, early exiting, and dynamic neural network literatures more thoroughly. This will help ground your dynamic activation-blending contributions in the historical development of the field.
2. **Alternative Lightweight OOD Estimators:** Can you discuss the practical trade-offs of using alternative lightweight density estimators (such as One-Class SVMs with RBF/linear kernels or compact Normalizing Flows) over diagonal GMMs? Specifically, what are the memory and compute constraints of these alternatives on low-power microcontrollers, and could they potentially resolve the sensitivity collapse ($38.04\%$ to $14.56\%$) observed under regularized calibration?
3. **Sub-task OOD Analysis:** The standard deviation of the OOD detection rate is quite high ($\pm 10.01\%$). A sub-task analysis reveals that noisy domains like SVHN dominate this variance. Would adopting task-specific OOD thresholds ($\eta_k$) instead of a single global threshold ($\eta$) stabilize this seed-to-seed detection variance, and what would be the systems-level memory overhead of storing task-specific thresholds?
4. **Intermediate Visual Scaling Pilot:** Have you considered running an intermediate-scale visual pilot (e.g., $K=8$ domains on DomainNet) to empirically ground your HMD-GMM routing on physical vision manifolds? If so, what are the preliminary ensembling and routing latency trends observed compared to SABLE SOTA?
