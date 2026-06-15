# Peer Review

**Paper Title**: ThermoMerge: Thermodynamic Test-Time Diffusion for Synergistic Model Merging

---

## 1. Summary of the Paper
The paper introduces **ThermoMerge**, a physics-inspired test-time adaptation framework for model merging. It addresses the limitation of existing test-time adaptive merging methods (such as AdaMerging and SyMerge) which rely on deterministic optimizers (like SGD or Adam) and are easily trapped in sharp, sub-optimal local basins of the highly non-convex multi-task loss landscape.

To overcome this, the authors model test-time merging as a thermodynamic physical system transitioning from a disordered, high-entropy state (chaotic independent experts) to a highly ordered crystalline state (synergistic multi-task fusion). Specifically, the framework utilizes **Stochastic Gradient Langevin Dynamics (SGLD)** guided by an **exponential Simulated Annealing cooling schedule** to explore the parameter space during an early "hot phase" and crystallize in flatter global minima during a "cold phase." To handle the dimensionality mismatch between low-dimensional merging coefficients and high-dimensional classification heads, the authors propose **Dimensionality-Scaled Langevin Noise (DSLN)**, which scales noise coordinate-wise inversely with the parameter group's dimension to prevent representation destruction.

The framework is evaluated on a synthetic 1D non-convex landscape and lightweight Multi-Layer Perceptrons (MLPs) and LoRA adapters on MNIST, FashionMNIST, and KMNIST.

---

## 2. Strengths
* **Highly Creative Conceptual Framing**: Framing joint parameter adaptation at test-time as a thermodynamic physical crystallization process is an engaging and original perspective. The use of a physical vocabulary to analyze optimization landscapes represents a beautiful conceptual contribution.
* **Mathematical Rigor and Detail-Oriented Design**:
  * The formulation of preconditioned Langevin dynamics (Adam-SGLD) is mathematically sound and follows statistical mechanics principles.
  * **Dimensionality-Scaled Langevin Noise (DSLN)** represents a rigorous, well-reasoned solution to scale isotropic noise across heterogeneous dimensions, successfully preventing high-dimensional noise catastrophe.
  * **Layer-wise Functional Parameter-Group Scaling** (grouping weight-bias pairs of the same layer to avoid thermodynamic imbalance) is a brilliant, highly thoughtful detail that demonstrates a deep understanding of neural network optimization.
* **Outstanding Intellectual Honesty and Transparency**: The authors are exceptionally upfront about the limitations of their work, explicitly pointing out that the empirical gains on real deep networks are subtle compared to the synthetic 1D landscape. This transparent self-critique is refreshing and highly commendable.
* **Rigorous System-Level Thinking**: The paper contains a comprehensive discussion of practical engineering constraints (such as distributed seed synchronization under tensor, pipeline, and data parallelism) and exact computational/latency profiling (F-Evals and G-Evals), proving the computational efficiency of the framework.

---

## 3. Weaknesses
* **Lack of Empirical Support for Central Claims on Deep Networks**:
  The central claim of the paper is that deterministic joint adaptation (SyMerge) is trapped in sub-optimal basins, whereas ThermoMerge's physical global exploration consistently escapes these traps to find superior flat global minima. However, a close inspection of the empirical tables on actual neural networks (Table 6, 7, 8, and 9) reveals that **this claim is completely unsupported by the deep learning experiments**:
  * In **8 out of 12 evaluations**, deterministic joint adaptation (SyMerge) actually achieves a higher mean multi-task accuracy than ThermoMerge (Ours).
  * On KMNIST, the results are virtually identical between the two methods.
  * On FashionMNIST (LoRA clean and OOD), ThermoMerge achieves a slightly higher mean accuracy (by $+0.99\%$ and $+1.11\%$, respectively), but the standard deviations overlap heavily, making this performance gain statistically fragile.
  * Thus, the claimed "56.7% reduction in final proxy loss" is **strictly restricted to the toy 1D simulation landscape** (which was specifically engineered to trap deterministic gradient descent). On real parameter landscapes, the benefits of the proposed global exploration are subtle or non-existent, and standard deterministic optimization remains highly competitive or superior.
* **Toy-Scale Evaluation and Missing Scale Validation**:
  * The neural network evaluation is restricted to tiny, 2-hidden-layer MLPs and LoRA adapters on MNIST, FashionMNIST, and KMNIST.
  * Modern model merging research focuses on massive foundation models (e.g., CLIP vision encoders, LLaMA-scale LLMs) on complex multi-task benchmarks (e.g., ImageNet, GLUE, MMLU).
  * While the authors transparently explain their cluster node and download limitations, the lack of scale evaluation is a major weakness. It remains unproven whether the proposed SGLD updates and DSLN scaling rule can scale and deliver practical value on real, modern foundation models.
* **Lack of Rigorous Statistical Significance Testing**:
  * The deep learning experiments are run over only 5 independent random seeds.
  * Given that the mean differences are tiny (often $0.05\% - 0.2\%$) and the standard deviations are relatively large, the paper is missing any rigorous statistical significance testing (such as t-tests) to confirm if the small gains on FashionMNIST are meaningful or merely noise.
* **Unvalidated Conceptual Recommendations**:
  * In Section 3.2, the authors suggest three elegant strategies to mitigate teacher-bias (Confidence-Based Filtering, Entropy-Based Weighting, and Predictive Agreement Monitoring).
  * However, these are presented as speculative recommendations and are not systematically evaluated with empirical ablation tables in Section 4, leaving their practical effectiveness unproven.

---

## 4. Detailed Feedback and Questions for the Authors
1. **Empirical Performance on Deep Networks**: Can you explain why deterministic SyMerge frequently outperforms ThermoMerge (Ours) in mean accuracy across MNIST and FashionMNIST/KMNIST clean and OOD MLP evaluations? If deterministic optimizers are indeed stuck in "sub-optimal sharp local basins," why does SGLD's global exploration yield lower or identical accuracies on actual parameters?
2. **Missing Statistical Significance**: Given the overlapping standard deviations, please provide p-values or t-test results comparing SyMerge and ThermoMerge across all MLP and LoRA configurations to demonstrate whether any of the deep learning improvements are statistically significant.
3. **Ablation of Heuristics**: Please provide a systematic ablation study on the MLP tasks to show the isolated impact of "Confidence-Based Filtering" and "Entropy-Based Weighting." Are these strategies strictly necessary to achieve stable performance under severe domain shift?
4. **Baselines**: Have you evaluated whether running simple randomized restarts (Multi-Start) of SyMerge on the MLP tasks can match or exceed the performance of ThermoMerge? Since the MLP tasks are small, this would serve as a crucial baseline to isolate the benefits of SGLD.

---

## 5. Overall Recommendation
**Rating**: 3: Weak Reject

**Justification for Rating**:
The paper has clear and substantial merits. The thermodynamic framing is elegant and engaging, the mathematical derivations (DSLN, preconditioned Langevin preconditioning, and functional weight-bias parameter-group scaling) are highly rigorous, and the authors demonstrate outstanding transparency regarding their work's limitations.

However, the primary weaknesses lie in the empirical evaluation. As a scientific work, the paper's central claims (that deterministic optimizers are severely trapped and that ThermoMerge's thermodynamic exploration successfully rescues the model to find superior minima on real networks) are **not supported by the deep learning experiments**. Across the majority of neural network evaluations, ThermoMerge actually underperforms or matches standard deterministic joint adaptation, with overlapping standard deviations. Furthermore, the evaluation scale is extremely outdated, focusing only on toy-scale MLPs and MNIST-level datasets. SGLD adds optimization complexity (new hyperparameters like $T_0, \gamma$) with virtually zero practical performance gain on actual deep learning workloads.

To be suitable for a top-tier machine learning conference, the paper requires a revision to scale up its evaluations to modern foundation models (e.g., CLIP or LLMs) where task conflicts are severe, and to demonstrate statistically significant performance gains that justify the complexity of thermodynamic Langevin diffusion.
