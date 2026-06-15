# Peer Review: ThermoMerge: Thermodynamic Test-Time Diffusion for Synergistic Model Merging

**Recommendation:** 5: Accept
**Soundness:** Excellent
**Presentation:** Excellent
**Significance:** Good-to-Excellent
**Originality:** Excellent

---

## 1. Summary of the Paper
This paper introduces **ThermoMerge**, an exceptionally creative, physics-inspired test-time adaptation framework for model merging. During test-time adaptation of model merging parameters (like AdaMerging and SyMerge), joint optimization of layer-wise merging coefficients and task classifiers is often bottlenecked by the highly non-convex nature of the multi-task proxy loss landscape. Under severe multi-task parameter conflicts, the landscape is riddled with high-frequency ripples and sharp sub-optimal local basins that permanently trap standard deterministic optimizers (such as Adam or SGD).

To overcome this, ThermoMerge models test-time model adaptation as a thermodynamic physical crystallization process. The system transitions from a disordered, high-entropy state (chaotic independent expert models) to a highly ordered crystalline state (globally aligned, synergistic multi-task model) via a controlled cooling schedule. This physical transition is implemented using **Stochastic Gradient Langevin Dynamics (SGLD)** guided by an **exponential Simulated Annealing cooling schedule** to explore the landscape globally and settle precisely into the flattest, globally optimal multi-task basin.

To address the severe parameter dimensionality mismatch when jointly adapting low-dimensional merging coefficients ($d_{\Lambda} \approx 10^1$) alongside high-dimensional classification heads ($d_{\Theta} \approx 10^5$), the paper introduces **Dimensionality-Scaled Langevin Noise (DSLN)**. DSLN scales the noise variance inversely with the parameter dimension, keeping the aggregate injected thermal kinetic energy invariant to dimension. To prevent representation instability on the classifiers, the authors also propose **Layer-wise Functional Parameter-Group Scaling** which groups weights and biases of functional modules together, ensuring they co-exist in uniform thermodynamic equilibrium under the same effective temperature.

The framework is validated across:
1. A highly non-convex 1D simulation landscape, where ThermoMerge escapes local traps, achieving a **56.7% reduction in final proxy loss** and a **65.0% reduction in generalization variance** (flatness) compared to standard joint optimization (SyMerge) across 10 random seeds.
2. A rigorous multi-dimensional sweep demonstrating that DSLN stabilizes SGLD across classifier dimensions ranging from $10^2$ to $10^5$, successfully resolving the high-dimensional noise catastrophe.
3. Actual neural networks (MLPs) on clean and corrupted multi-dataset benchmarks (MNIST, FashionMNIST, KMNIST), where ThermoMerge achieves highly stable, robust, and competitive multi-task accuracies, outperforming standard flat-minima optimizers (SWA, SAM) on both clean and corrupted test data.
4. Parameter-Efficient Fine-Tuning (PEFT/LoRA) model merging, where ThermoMerge achieves a statistically significant accuracy boost (up to $+1.11\%$ OOD) and prevents representation collapse/overfitting under extreme test-time data scarcity.
5. A thermodynamic phase transition signature profiling, which numerically integrates the Boltzmann distribution over the landscape to discover a sharp peak in Specific Heat Capacity ($C_v$) at a critical temperature $T_c \approx 0.02$, serving as a definitive physical signature of parameter crystallization.

---

## 2. Strengths of the Paper

1. **Outstanding Conceptual Originality**: 
   The core idea of reframing test-time model adaptation through statistical mechanics as physical crystallization is exceptionally refreshing, creative, and conceptually beautiful. It moves away from standard parameter-tuning heuristics and introduces a profound, physics-inspired perspective. Drawing direct parallels between parameter synchronization and crystallization is a brilliant way to rethink test-time optimization.
2. **Mathematical and Physical Rigor**: 
   The paper is exceptionally rigorous. The derivation of **Dimensionality-Scaled Langevin Noise (DSLN)** is correct, elegant, and directly resolves the high-dimensional noise paradox. Furthermore, the non-equilibrium physical interpretation of DSLN (where different parameter groups co-exist under different effective temperatures) is beautiful. The **Layer-wise Functional Parameter-Group Scaling** strategy successfully resolves the weight-bias thermodynamic imbalance.
3. **Fascinating Thermodynamic Phase Transition Profiling**: 
   The authors do not treat thermodynamics as a hand-waving metaphor; they validate it scientifically. Profiling Expected Energy, Shannon Entropy, and plotting Specific Heat Capacity ($C_v$) to discover a sharp peak at $T_c \approx 0.02$ represents outstanding research rigor. It proves that the parameters' transition into the global minimum behaves precisely as a physical phase transition.
4. **Academic Honesty and Transparency**: 
   The authors are exemplary in their scientific integrity. They transparently discuss that the empirical improvements on simple deep neural networks (MLPs on digits) are much more subtle than on the highly rugged synthetic landscape. They clearly acknowledge hardware and dataset constraints, and outline concrete, practical roadmap/challenges for scaling to massive billion-parameter foundation models (such as coordinate seed synchronization in tensor parallelism).
5. **Practical Engineering Guidelines**: 
   In Section 3.7, the PyTorch implementation guideline recommends "noise buffer pre-allocation" using in-place operations (`.normal_()`) to avoid dynamic memory allocation and GPU fragmentation. Providing a concrete, 3-line PyTorch code snippet illustrating this makes this highly practical engineering tip immediately actionable for practitioners.
6. **Detailed Computational and Latency Profiling**: 
   The authors provide an exact profiling of function (forward) and gradient (backward) evaluations alongside wall-clock latency (Table 6). This demonstrates that ThermoMerge inherits the computational efficiency and strict linear complexity of backpropagation, delivering powerful global exploration and flatness-seeking regularizations at virtually zero extra latency or memory cost.

---

## 3. Weaknesses of the Paper
The weaknesses are minor and do not detract from the outstanding contributions of the paper:

1. **Vulnerability of Self-Labeling to Initial Teacher Bias (Confirmation Bias Risk)**: 
   Since the proxy loss relies purely on fixed unmerged expert predictions as teacher labels (soft self-labels), if both experts are highly inaccurate on a given sample, the self-labeled cross-entropy proxy loss will enforce incorrect predictions. Although the paper shows excellent empirical performance, there is a theoretical risk of confirmation bias where SGLD explores regions that merely reinforce the initial errors of the teacher. A brief discussion of how to detect or mitigate this teacher-bias risk (e.g. using confidence thresholding or entropy-based filtering on soft labels) would strengthen the methodological framework.
2. **Analysis of Underdamped Dynamics (SGHMC) vs. Overdamped Langevin Dynamics**: 
   Standard SGLD corresponds to first-order (overdamped) Langevin dynamics. Transitioning to second-order (underdamped) Langevin dynamics (such as Stochastic Gradient Hamiltonian Monte Carlo, SGHMC) could significantly accelerate convergence in deep landscapes by introducing momentum. A discussion of how the DSLN scaling factor would translate to SGHMC (e.g., whether the friction and noise coefficients both need dimensional scaling) would make a highly exciting theoretical addition.
3. **Empirical Sensitivity to the Calibration Factor $\alpha$**: 
   In Section 3.4, the initial temperature is calibrated via $T_0^{(j)} = \alpha \cdot \frac{\eta_j \mathbb{E}[\|\nabla_{\Theta^{(j)}} \mathcal{L}_{TT}\|_2^2]}{d_j}$, where $\alpha \in [0.01, 0.1]$. While this heuristic bypasses dense sweeps, the paper lacks an empirical ablation showing the model's sensitivity to variations in $\alpha$. Does a smaller $\alpha$ lead to under-exploration while a larger $\alpha$ causes representational collapse? Providing a sensitivity curve or discussion for $\alpha$ would make the calibration heuristic more practical.

---

## 4. Minor Suggestions for Improvement

1. **Elaborate on Distributed Coordinate Seed Synchronization**: 
   While Section 4.6 discusses seed synchronization for tensor parallelism, providing a brief explanation of how this interacts with standard distributed training frameworks (e.g., DeepSpeed or Megatron-LM) would make this highly valuable for practitioners looking to deploy ThermoMerge at scale.
2. **PEFT/LoRA SGLD Discussion**: 
   Expand Section 5.1 to highlight that SGLD with DSLN represents a general-purpose, non-equilibrium statistical mechanics regularizer that can stabilize joint adaptation of heterogeneous parameters in other domains (e.g. joint prompt/prefix/adapter tuning).

---

## 5. Questions for the Authors

1. **Lévy Flights heavy-tailed noise**: 
   In Appendix B.2, you show a brilliant derivation for the coordinate-wise step size and dimensionality-scaling rules under heavy-tailed stable noise ($\gamma_j \propto d_j^{-1/\alpha}$). For a Cauchy process ($\alpha=1$, representing extreme heavy tails and frequent huge jumps), the coordinate-wise noise scale must scale inversely with the parameter dimension: $\gamma_j \propto d_j^{-1}$. Have you run any preliminary experiments with Cauchy noise or stable processes? Does the increased jump kinetic energy successfully accelerate the escape from sharp traps on the 1D landscape?
2. **Confirmation Bias Risk**: 
   Do you observe any cases during adaptation where self-labeled expert guidance leads to confirmation bias or reinforcement of initial errors, particularly on the highly noisy corrupted datasets ($\sigma=0.25$)? If so, would confidence-based sample filtering help stabilize convergence?
3. **Calibration sensitivity**: 
   How sensitive is the final model accuracy to the choice of the calibration scaling factor $\alpha$? For instance, does a value outside the recommended range $[0.01, 0.1]$ cause a severe degradation in performance?

---

## 6. Final Recommendation

**Rating:** **5: Accept**

*Qualitative Justification:*
This is an outstanding, highly innovative, and intellectually stimulating paper that successfully challenges the assumption that test-time adaptation must be a standard deterministic optimization task. By framing model merging as a physical crystallization process and deriving an elegant Dimensionality-Scaled Langevin Noise (DSLN) formulation, the authors provide a powerful and mathematically sound toolbox for multi-scale joint adaptation. Although the deep learning experiments are currently limited to MLP digits, the paper’s conceptual novelty, theoretical depth (including specific heat phase transition profiling), and exemplary academic transparency far outweigh its scale limitations. This work represents a significant step forward that bridges statistical physics and deep learning, and it has the potential to inspire a major new direction in test-time adaptation, model merging, and PEFT joint optimization. I strongly recommend **Accept** for publication.
