# Peer Review of Conference Submission

## Strengths and Weaknesses

### Strengths
1. **Exceptional Intellectual Honesty and Transparency:** The authors deserve immense credit for their rare level of scientific candor. Rather than attempting to obscure EHPB's low performance, they dedicate substantial portions of the paper to naming and diagnosing the **Hadamard Dominance Paradox**, documenting that their proposed model is heavily outperformed by a simple static average (Uniform Merging) by an absolute margin of **+26.9%**. They also candidly lay out the severe non-linear propagation limits of their method and the "floor effect" of their SVHN setup.
2. **Highly Original Interdisciplinary Synthesis:** Establishing a conceptual bridge between Vector Symbolic Architectures (VSA) and deep neural network model merging is a highly creative and original direction. Reframing parameter space as a holographic associative memory where expert weights are modulated onto pseudo-orthogonal random keys is a fascinating paradigm shift from standard linear model-averaging.
3. **Rigorous Theoretical Diagnostics:** The theoretical deconstructions in the paper are outstanding. The **Post-Hoc Model Ensembling Trilemma** (Dynamic Adaptability, Resource Efficiency, Weight Integrity) provides a valuable taxonomical framework to classify merging methods. Furthermore, the mathematical modeling of ReLU Positive Bias Rectification and LayerNorm Exponential Signal Attenuation provides highly rigorous and elegant physics-based explanations of noise propagation in deep non-linear networks.
4. **Clever, Low-Overhead Mitigations:** The paper formulates and evaluates several novel mitigations (Residual-EHPB, Continuous Cleanup Networks, and ReLU Bias Correction) which succeed in rescuing significant representation fidelity and mitigating positive rectification bias.
5. **Clear Future Roadmap:** By proving that circular convolution weight operators are required to restore the $O(1/\sqrt{D})$ noise-decay in continuous parameter spaces, the paper provides a vital and actionable roadmap for future hyperdimensional weight-space ensembling.

### Weaknesses
1. **Complete Lack of Statistical Rigor (No Error Bars or Multiple Seeds):** For a paper exploring the empirical performance limits of a highly stochastic framework—relying on randomly sampled bipolar carrier keys, a tiny 64-sample router calibration set, and 16-sample ReLU bias calibrations—the **complete absence of standard deviations, confidence intervals, or indications of multiple random seeds** is a major scientific oversight. It is impossible to determine if the reported accuracies and MSE values are statistically stable or artifacts of single lucky runs.
2. **Highly Artificial, Simulated Evaluation (Toy Sandbox):** The entire empirical evaluation is conducted inside a "Controlled Representation Sandbox" utilizing a pre-trained ViT-Tiny backbone and synthetic task vectors generated as independent Gaussian matrices. In real-world model merging, specialized expert weights are fine-tuned from a shared initialization, meaning they are highly correlated and reside on low-dimensional manifolds. Generating task vectors as completely independent Gaussian parameters represents a highly pessimistic toy setup. The lack of any evaluation on actual fine-tuned checkpoints of real models (e.g., LLMs on GLUE or vision models on VTAB) significantly limits the practical empirical validation of this work.
3. **Core Method is Practically Uncompetitive:** Despite the elegant mitigations, EHPB's Joint Mean accuracy under homogeneous streaming remains extremely low (**25.4%**), whereas simple static Uniform Merging achieves **52.3%** accuracy—more than double EHPB's score. Uniform Merging requires zero parameter storage for keys, zero dynamic routing, zero dynamic unbinding latency, and has an identical $O(P)$ active memory scaling. In its current form, the EHPB method is more of an academic diagnostic study of element-wise Hadamard binding limitations than a viable, competitive deep learning tool.
4. **Practical Edge Latency Overhead:** The CPU-bound latency profiling reveals that EHPB's demodulation takes **39.454 ms** per forward pass, whereas naive eager sequential execution (which EHPB seeks to replace) takes only **16.004 ms**. This demonstrates that on commodity edge hardware, EHPB is computationally slower, restricting its efficiency benefits to specialized compiled Triton GPU pipelines.

---

## Soundness
**Rating: Fair**

**Justification:**  
While the theoretical formulations and the mathematical proof sketches in the paper are exceptionally rigorous and correct, the **empirical methodology exhibits significant gaps** that fall short of excellent scientific standards. First, there is a total lack of statistical error bars, confidence intervals, or multi-seed trials across all tables. Given the stochastic nature of random carrier keys and lightweight calibration subsets, reporting standard deviations over multiple seeds is a crucial prerequisite to validate the robustness of the results. Second, the entire experimental setup is synthetic, utilizing generated Gaussian task vectors rather than actual fine-tuned network checkpoints on real multi-task benchmarks. Third, the core method underperforms so severely (25.4% vs 52.3% for Uniform Merging) that the primary claims regarding the practical utility of holographic superposition are not supported by the data, making EHPB practically unviable without further development of the circular-convolution roadmap.

---

## Presentation
**Rating: Excellent**

**Justification:**  
The submission is beautifully written, exceptionally well-structured, and easy to follow. The notation is precise and consistent throughout the mathematical expansions. Figures 1 and 2 are clear, highly informative, and aid the reader's understanding of complex concepts. The related work is thorough, and the authors are remarkably candid about their framework's performance limits and the synthetic nature of their sandbox. Constructive suggestions are addressed thoroughly.

---

## Significance
**Rating: Fair**

**Justification:**  
The conceptual significance of this paper is quite high; the Post-Hoc Model Ensembling Trilemma and the application of VSA principles to model weight spaces are highly creative and could influence future academic directions. However, its **practical significance in its current state is very low**. Due to the Hadamard Dominance Paradox, deep learning practitioners and deployment engineers are highly unlikely to use EHPB, since a simple, zero-overhead static average performs twice as well. The paper serves primarily as an diagnostic post-mortem of element-wise Hadamard binding in weight space rather than a ready-to-deploy, high-impact ensembling library.

---

## Originality
**Rating: Excellent**

**Justification:**  
The paper is highly original. The interdisciplinary fusion of Vector Symbolic Architectures (VSA) and deep network model merging is a novel combination of existing fields. Reframing weight layers as holographic associative memories, defining low-rank outer-product bipolar keys, and deriving the exact impact of unbinding cross-talk noise represent highly original contributions that differ substantially from closely related literature.

---

## Overall Recommendation
**Rating: 3: Weak reject**

**Justification:**  
This paper represents a highly commendable and intellectually honest attempt to introduce hyperdimensional computing to model merging. It possesses outstanding presentation quality, conceptual originality, and deep theoretical diagnostics. However, from an empirical perspective, the weaknesses outweigh the merits in its current form. The complete absence of statistical error bars (standard deviations, seeds, or confidence intervals) is a major flaw for a framework heavily dependent on stochastic components. Furthermore, the evaluation is entirely confined to a synthetic sandbox with simulated Gaussian weight updates. Finally, the core EHPB method achieves only 25.4% accuracy, which is vastly dominated by a simple static average (52.3%) that requires no routing, keys, or dynamic latency. To be suitable for acceptance, the paper requires a revision that introduces statistical error bars over multiple seeds, validates EHPB on actual fine-tuned network checkpoints on real benchmarks, and shows a more competitive empirical performance profile.
