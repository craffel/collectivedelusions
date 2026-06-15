# Official Peer Review

## Summary of the Paper
The paper introduces **ThermoMerge** (Thermodynamic Test-Time Diffusion for Synergistic Model Merging), a physics-inspired test-time adaptation (TTA) framework for model merging. Traditional test-time adaptation methods (like AdaMerging and SyMerge) rely on deterministic, gradient-based optimizers to tune merging coefficients and classification heads on unlabeled downstream data. The authors argue that the joint multi-task loss landscape is highly non-convex due to parameter conflicts and task interference, trapping deterministic optimizers in sub-optimal local basins.

To escape these traps, ThermoMerge casts test-time model merging as a thermodynamic physical crystallization process. The framework replaces deterministic gradient descent with **Stochastic Gradient Langevin Dynamics (SGLD)** guided by an **exponential Simulated Annealing cooling schedule** to transition parameters from high-entropy, global exploration to stable, low-entropy convergence. To address the dimensional mismatch between low-dimensional merging coefficients and high-dimensional classification heads, the authors propose **Dimensionality-Scaled Langevin Noise (DSLN)**, which scales the coordinate-wise noise variance inversely with each parameter group's dimension.

The authors evaluate ThermoMerge on a synthetic, non-convex 1D physical landscape, as well as on lightweight Multi-Layer Perceptrons (MLPs) and LoRA adapters on MNIST, FashionMNIST, and KMNIST task splits.

---

## Ratings

* **Soundness:** **Fair**
* **Presentation:** **Excellent**
* **Significance:** **Fair**
* **Originality:** **Fair**

---

## Strengths and Weaknesses

### Strengths:
1. **Outstanding Clarity and Presentation:** The paper is exceptionally well-written, clearly structured, and easy to follow. Complex physical analogies (crystallization, Boltzmann distributions, partition functions, specific heat capacity) are explained beautifully and linked logically to machine learning optimization.
2. **Transparent and Commendable Scientific Candor:** The authors demonstrate excellent scientific integrity. They are completely honest about their evaluation scope,Cluster hardware constraints, the fact that their physical phase transition profiling is non-verifiable on high-dimensional neural networks, and the highly modest nature of their empirical gains on MLPs.
3. **Rigorous Detailing of Engineering Heuristics:** The paper does not skip over subtle implementation challenges. The derivation of **Layer-wise Functional Parameter-Group Scaling** (grouping weights and biases to prevent bias over-perturbation) and the theoretical analysis of seed synchronization across distributed GPU ranks (data, tensor, and ZeRO-3 parallelism) are highly rigorous and practically valuable.

### Weaknesses:
1. **Severe Evaluation Scale Limitations (The "Toy" Benchmark):**
   All actual neural network experiments are conducted on **MNIST, FashionMNIST, and KMNIST** splits using a **2-layer MLP (128 and 64 units)** with a rank $r=4$ LoRA adapter. In the modern model merging and test-time adaptation literature, standard benchmarks require evaluating **large-scale foundation models** (such as CLIP ViT-B/16 or LLMs like LLaMA-7B/13B) on real-world datasets. MLPs on MNIST digits represent highly contrived, low-dimensional settings. Results on these datasets do not reliably generalize to modern architectures or industrial use cases, leaving a massive gap in validating the practical utility of the proposed method.

2. **Extremely Marginal or Negative Empirical Performance Gains:**
   A critical review of the empirical results in Tables 6, 7, 8, and 9 reveals that ThermoMerge's active adaptation machinery fails to yield meaningful advantages over standard deterministic joint adaptation (SyMerge) or simple static averaging:
   * **MLP Merging (Table 6):** ThermoMerge fails to outperform SyMerge on clean data (e.g., MNIST: $89.94\%$ vs. $89.97\%$; FashionMNIST: $84.46\%$ vs. $84.61\%$). The only improvement is an insignificant $0.02\%$ on KMNIST.
   * **MLP Merging Under OOD Corruption (Table 8):** The full ThermoMerge method **consistently underperforms** deterministic SyMerge across all three datasets under Gaussian noise corruption.
   * **LoRA PEFT Merging (Table 7 & 9):** On clean and corrupted LoRA merging, ThermoMerge underperforms or is virtually identical to SyMerge in 4 out of 6 configurations. The only noticeable gains are on FashionMNIST (clean: $+0.99\%$; corrupted: $+1.11\%$).
   * **Negative Adaptation Compared to Static Baselines:** In Table 7 (LoRA clean), static Task Arithmetic ($\lambda=0.5$) achieves **$89.85\%$** on MNIST, whereas ThermoMerge drops to **$88.65\% \pm 0.63\%$** (and SyMerge to $88.68\%$). This indicates that the active test-time adaptation process is actually degrading the model's performance compared to a training-free static average.

3. **Theoretical Contradiction of Dimensionality-Scaled Langevin Noise (DSLN):**
   The authors argue that scaling the coordinate-wise Langevin noise standard deviation by $1/\sqrt{d_j}$ prevents high-dimensional feature degradation. However, from a statistical physics perspective, scaling noise variance by $1/d_j$ is equivalent to assigning an effective temperature of $T_j = T_t / d_j$ to that layer. For a classification head ($d_j = 640$) or larger layers, this forces the effective temperature to be extremely cold. Consequently, individual parameters are perturbed by an infinitesimally small amount of noise ($\approx 10^{-3}$).
   This means that **the high-dimensional weights are adapted almost purely deterministically** via standard gradient descent. The claimed "thermodynamic global search" and "escaping local traps" are almost entirely confined to the extremely low-dimensional merging coefficients $\Lambda$. This undermines the paper's core conceptual claim of a unified, physical crystallization process for the entire network.

4. **Engineering Complexity and Over-Engineering:**
   To make SGLD stable and prevent "thermal vaporization" of features, the authors introduce multiple nested, post-hoc heuristics:
   * Confidence-based filtering ($\tau = 0.85$)
   * Entropy-based weighting
   * Rolling dynamic calibration (EMA of gradient norms)
   * Early-stage predictive agreement and entropy safeguards (Emergency Quenching and parameter resets)
   
   This accumulation of thresholds, safeguards, and feedback loops transforms a simple optimization algorithm (SGLD) into a highly complex, brittle engineering system. For practitioners, deploying a stochastic system with automated emergency rollbacks is highly unappealing, especially when standard deterministic optimizers or static task arithmetic perform similarly or better with zero tuning.

---

## Overall Recommendation

**3: Weak Reject**

### Justification:
The paper has clear merits, including its outstanding writing quality, rigorous detailing of practical engineering heuristics (like functional weight-bias grouping), complete candor about results, and a highly elegant physical framing. 

However, the weaknesses significantly outweigh these merits:
1. The evaluation is restricted entirely to toy-scale MLPs and MNIST digits, which is far below the standards of modern foundation model merging research.
2. The empirical performance gains of ThermoMerge are statistically weak, negligible, or negative compared to simple deterministic optimizers and static task arithmetic in 10 out of 12 evaluated configurations.
3. The proposed DSLN formulation conceptually reduces the high-dimensional weight updates to de facto deterministic optimization, contradicting the claim of a global thermodynamic crystallization.
4. The system is heavily over-engineered with nested thresholds and emergency quenching rollbacks, making it highly impractical for real-world deployments. 

Therefore, significant revisions—most notably, evaluating the method on standard foundation models (CLIP/LLMs) and demonstrating clear, simplified, and substantial performance gains—are required before this work can be considered ready for publication.

---

## Questions and Constructive Feedback for the Authors

1. **Evaluation on Modern Foundation Models:** Can you evaluate ThermoMerge on standard model merging benchmarks, such as merging CLIP (ViT-B/16) encoders on downstream image classification tasks, or merging LLaMA-7B/13B LoRA adapters? Evaluating on actual foundation models is crucial to demonstrate the real-world utility of your framework.
2. **Simplification of Safeguards:** Under clean and controlled test-time adaptation settings, are the complex predictive safeguards and emergency quenching mechanisms actually necessary? Can you show a simplified version of ThermoMerge that does not require these nested post-hoc thresholds, and evaluate its stability?
3. **Addressing Negative Adaptation:** Why does the test-time adaptation process (both SyMerge and ThermoMerge) degrade performance on MNIST LoRA compared to static Task Arithmetic by over $1.2\%$? What steps can be taken to prevent active adaptation from harming the baseline merged model under data scarcity?
4. **Effective Temperature Analysis:** Given that DSLN scales down coordinate-wise noise to near-zero levels on high-dimensional weights (effectively making them deterministic), is it fair to claim that the entire network undergoes a global thermodynamic exploration? Have you considered applying SGLD *only* to the merging coefficients while updating the classification head purely deterministically from the start, and comparing the performance?
