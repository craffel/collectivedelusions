# Peer Review Report

## Summary of the Paper
This paper proposes **ThermoMerge (Thermodynamic Model Merging)**, a framework that reformulates test-time adaptive model merging through the lens of statistical mechanics and thermodynamics. Rather than linearly interpolating network parameters in a static Euclidean space, the authors propose to treat classification logits as negative microstate energies within a finite-temperature canonical Boltzmann ensemble. 

To optimize layer-wise merging coefficients on unlabeled sequential data streams during inference, the framework uses a **Thermodynamic Annealing Schedule (TAS)** (simulated cooling) to help the optimizer navigate non-convex boundaries. It minimizes a **Helmholtz Free Energy Discrepancy (F-Min)** objective, which the authors prove is mathematically equivalent to the temperature-scaled Kullback-Leibler (KL) divergence. Under a sequential streaming test-time adaptation setting on pre-trained ResNet-18 (across MNIST, FashionMNIST, CIFAR-10, and SVHN), ThermoMerge achieves a multi-task average accuracy of 29.05%, which the authors claim is a state-of-the-art result that mitigates the "Overfitting-Optimizer Paradox" and bypasses representation collapse on color datasets.

---

## Strengths and Weaknesses

### Strengths
* **Clear Structure and Mathematical Flow:** The paper is well-organized, mathematically coherent, and written with a high level of academic fluency.
* **Detailed Derivations:** The appendix contains a complete, step-by-step algebraic expansion proving the equivalence between the temperature-scaled KL divergence and the variational free energy terms, which is appreciated for mathematical completeness.
* **Creative Metaphor:** The analogy of mapping classifier outputs to canonical Boltzmann ensembles is intellectually creative and thoroughly worked out in the text.

### Weaknesses
* **Catastrophically Low Absolute Performance:** An average multi-task accuracy of 29.05% across toy datasets like MNIST, FashionMNIST, CIFAR-10, and SVHN is practically non-functional. For example, the merged model achieves only **20.00%** on MNIST (random guessing is 10%). Standard joint training or multi-task learning easily achieves $>90\%$ average accuracy on this suite. A model that performs at near-random levels is completely unusable in practice.
* **Critical Baseline Omission:** The paper completely fails to report the standalone performance of the fine-tuned experts prior to merging. Without these baseline accuracies, it is impossible to evaluate how much representation collapse occurred or if the experts themselves were poorly trained from the start. 
* **Over-theorization and Academic Spin:** The "thermodynamic" formulation is purely an analogical overlay on top of standard machine learning concepts. "Canonical Ensemble Mapping" is temperature-scaled Softmax; "Helmholtz Free Energy" is the log-sum-exp of logits scaled by $-T$; and "Free Energy Discrepancy Minimization" is standard temperature-scaled soft-label Knowledge Distillation (KL divergence). Presenting standard, well-known techniques under exotic physical names without explicitly clarifying their functional equivalence is highly misleading.
* **Unrealistic Test-Time Data Requirements:** The test-time adaptation protocol requires **12,800 unlabeled images** from the target domains over 100 optimization steps. For datasets like MNIST, FashionMNIST, CIFAR-10, and SVHN, the entire test sets are only 10,000 images each. Requiring a calibration dataset larger than the test set itself violates the core constraints of test-time adaptation, which should operate in data-scarce or online settings.
* **Marginal Improvement over Static Baselines:** ThermoMerge improves performance from the zero-overhead static Task Arithmetic baseline (27.25%) to 29.05% (a marginal $+1.80\%$ gain). This tiny improvement comes at the cost of **100 backpropagation steps during inference** and running $K=4$ parallel forward passes through frozen expert models at each step. This massive latency and compute overhead are practically unjustifiable.
* **Failure on SimpleCNN Backbone:** On the from-scratch SimpleCNN backbone, ThermoMerge completely collapses to near-random chance on CIFAR-10 (11.40%) and SVHN (16.20%), proving that the method is fragile and entirely reliant on pre-trained ImageNet representations to function.

---

## Detailed Evaluation

### Soundness: Poor
The mathematical derivations are algebraically correct, but the methodological assumptions are deeply flawed:
1. **The Physical Metaphor is Over-claimed:** There is no physical heat bath, no real thermodynamic system, and no physical equilibrium. The "cooling schedule" is simply a loss function hyperparameter decay. Claiming a "rigorous connection" to physical thermodynamics is conceptually misleading.
2. **Trainable Task Temperatures ($\tau_k$):** As stated in Section 3.5, the final evaluation accuracy is mathematically invariant to $\tau_k$ (since division of logits by a positive scalar does not change the rank-ordering of predictions during $\arg\max$). Therefore, $\tau_k$ acts solely as a training-phase gradient scaler. The authors have to clamp $\tau_k \in [0.2, 5.0]$ to prevent division-by-zero or gradient explosion, indicating that this parameter is highly sensitive and numerically unstable.
3. **Stand-alone Experts:** The absence of stand-alone expert performance results makes the soundness of the entire empirical evaluation highly questionable.

### Presentation: Fair
While the writing is fluent and clear, the presentation is severely compromised by "academic spin." By wrapping standard machine learning techniques—specifically Softmax, Log-Sum-Exp, KL divergence, simulated annealing, and temperature scaling—in dense, dramatic thermodynamic prose, the authors have obscured the true nature of their contribution. A superbly written paper should aim for simplicity and transparency rather than dressing up standard methods in complex metaphors to manufacture novelty.

### Significance: Poor
The significance of this work is exceptionally low:
1. **No Practical Utility:** A multi-task model with an average accuracy of $29\%$ is practically useless.
2. **High Latency Bottleneck:** Requiring backpropagation and parallel forward passes through multiple frozen models during test-time adaptation introduces severe latency that defeats the purpose of model merging as a fast, zero-training alternative.
3. **Inefficient Data Usage:** Requiring 12,800 calibration images is highly inefficient and unrealistic for test-time adaptation.

### Originality: Poor
The core technical contribution is highly incremental. Using soft-label Knowledge Distillation (KL divergence) to align a merged student model with task-specific expert teachers on unlabeled data is a straightforward application of distillation. Using simulated annealing to decay the distillation temperature is also well-known. The originality is confined to the vocabulary used to describe these operations rather than the operations themselves.

---

## Overall Recommendation

**Score: 2 (Reject)**

**Justification:**  
This paper is technically sound in its mathematical identities but fails on almost every empirical and practical front. The absolute accuracies achieved by the merged model (e.g., 20.00% on MNIST and 30.60% on SVHN) are catastrophically low, making the model completely non-functional. The omission of standalone expert performance is a critical scientific gap. Furthermore, the framework introduces massive computational, latency, and data overhead (12,800 calibration images and 100 backpropagation steps during inference) to achieve a negligible $+1.80\%$ accuracy improvement over static Task Arithmetic. Finally, the paper suffers from severe over-theorization, dressing up standard temperature-scaled Softmax and Knowledge Distillation in complex thermodynamic jargon to create an illusion of novelty. For these reasons, the paper falls far below the standard for acceptance and is a clear reject.

---

## Questions and Constructive Suggestions for the Authors

1. **Why are the absolute accuracies so low?** Please explain why a pre-trained ResNet-18 model gets only 20% on MNIST and 33% on CIFAR-10. Even under merging, these values are catastrophically low. What are the standalone accuracies of the fine-tuned experts? These must be included in Table 1.
2. **Acknowledge Technical Equivalence:** Please explicitly state in the methodology section that "Canonical Ensemble Mapping" is temperature-scaled Softmax, "Helmholtz Free Energy" is the scaled log-sum-exp of logits, and "F-Min" is temperature-scaled KL divergence (Knowledge Distillation). Transparency is critical for scientific integrity; dressing up standard deep learning operations in physical metaphors without clarifying their equivalence is misleading.
3. **Test under realistic data constraints:** Test-time adaptation must operate in data-scarce environments. Please evaluate ThermoMerge's performance when the number of calibration images is restricted to realistic scales, such as 32, 64, or 128 images total, rather than 12,800. How does the method perform under these scarce regimes?
4. **Evaluate computational latency:** Please provide a detailed table comparing the computational latency and memory footprint of ThermoMerge against static methods (Task Arithmetic, TIES-Merging) and other adaptive methods (SyMerge, AdaMerging) during test-time inference.
5. **Scale to modern architectures:** Model merging is primarily used to merge massive foundation models (e.g., LLMs or ViTs) where joint training is computationally prohibitive. Please evaluate your method on standard benchmarks using modern architectures (such as CLIP ViT-B or LLaMA-7B) to demonstrate that your thermodynamic principles generalize and offer practical utility.
