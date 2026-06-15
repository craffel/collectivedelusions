# 4. Experimental Evaluation Check

## Experimental Setup and Datasets
The empirical evaluation on actual neural networks is **exceptionally weak and restricted to a toy-scale benchmark**. 
*   **Toy-Scale Datasets:** The authors evaluate their deep learning experiments purely on MNIST, FashionMNIST, and KMNIST. These are extremely simple, grayscale, $28 \times 28$ resolution handwritten digit or clothing classification datasets.
*   **Toy-Scale Model:** The neural networks adapted are simple Multi-Layer Perceptrons (MLPs) containing only two fully-connected layers (size $128$ and $64$).
*   **Atypical Baseline Configurations:** Modern model merging research is overwhelmingly performed on massive pre-trained transformer architectures (such as ViT, CLIP, LLaMA, or Mistral) with hundreds of millions or billions of parameters, operating on highly non-convex vision-language or text distributions. Evaluating a tiny MLP on digit classification and claiming that "joint thermodynamic SGLD exploration successfully scales to deep neural landscapes" is an extreme overstatement of the framework's scalability and generalizability.

---

## Weak Baselines and Missing Sanity Checks
While the paper compares ThermoMerge against training-free baselines (Task Arithmetic, Ties-Merging, DARE) and simple adaptive baselines (AdaMerging, SyMerge), several critical baselines are missing:
1. **Deterministic Classifier Freezing:** A baseline where *only* the low-dimensional merging coefficients $\Lambda$ are optimized deterministically (using SGD/Adam), while keeping the classifiers frozen. If this simple, training-free classifier baseline performs comparably, it would render the entire high-dimensional classifier adaptation redundant.
2. **Standard Stochastic Optimization Alternatives:** Standard SGD with high momentum, learning rate decay, or simple random restarts (running SyMerge 5 times from different initializations and selecting the best based on proxy loss).
3. **Noisy Gradient Descent (SGD Noise):** A baseline where isotropic Gaussian noise is injected with a constant temperature (without Simulated Annealing), to isolate the benefits of the cooling schedule.

---

## Do the Results Support the Claims?
**Absolutely not.** This is the most critical and damning flaw of the submission.

The authors' primary thesis is that deterministic optimizers (like those used in AdaMerging and SyMerge) get permanently trapped in sub-optimal, sharp local basins under severe task interference, resulting in poor multi-task trade-offs and bad generalization. They present ThermoMerge as a necessary "rescue" that escapes these traps to find superior flat global minima.

However, an inspection of the actual deep learning results on real neural networks (MLP and LoRA) reveals a **stark contradiction between the claims and the empirical evidence**:

### 1. MLP Clean Accuracies (Table 7)
In standard, clean evaluations on actual neural network parameters:
*   **MNIST:** Deterministic SyMerge achieves **$89.97\% \pm 0.19\%$**, while ThermoMerge (Ours) achieves $89.94\% \pm 0.16\%$. **The deterministic optimizer has a higher mean accuracy.**
*   **FashionMNIST:** Deterministic SyMerge achieves **$84.61\% \pm 0.49\%$**, while ThermoMerge (Ours) achieves $84.46\% \pm 0.59\%$. **The deterministic optimizer has a higher mean accuracy.**
*   **KMNIST:** Deterministic SyMerge achieves $80.35\% \pm 0.19\%$, while ThermoMerge (Ours) achieves $80.37\% \pm 0.24\%$. **The difference is an insignificant $0.02\%$ with heavily overlapping standard deviations.**

### 2. LoRA Clean Accuracies (Table 10)
In the Parameter-Efficient Fine-Tuning (PEFT) regime, where the authors claim SGLD is particularly vital because of constrained low-rank spaces:
*   **MNIST:** Deterministic SyMerge achieves **$88.68\% \pm 0.86\%$**, while ThermoMerge (Ours) achieves $88.65\% \pm 0.63\%$. **The deterministic optimizer has a higher mean accuracy.**
*   **KMNIST:** Deterministic SyMerge achieves $76.57\% \pm 0.18\%$, while ThermoMerge (Ours) achieves $76.62\% \pm 0.28\%$. **The difference is an insignificant $0.05\%$.**
*   **FashionMNIST:** ThermoMerge achieves $78.41\% \pm 1.67\%$ compared to SyMerge's $77.42\% \pm 1.52\%$. While this is a $\approx 1\%$ improvement, the standard deviations overlap heavily ($75.9\% - 80.1\%$ vs. $76.7\% - 80.0\%$), indicating statistical marginality.

### 3. Out-of-Distribution (OOD) Accuracies Under Corruption (Table 8 & Table 11)
The authors claim that ThermoMerge's physical exploration settles in flatter, wider basins that provide superior OOD robustness. However, under severe Gaussian noise corruption ($\sigma = 0.25$):
*   **MLP MNIST OOD:** Deterministic SyMerge achieves **$87.51\% \pm 0.53\%$**, while ThermoMerge (Ours) achieves $87.49\% \pm 0.54\%$. **SyMerge has a higher mean.**
*   **MLP FashionMNIST OOD:** Deterministic SyMerge achieves **$76.91\% \pm 0.83\%$**, while ThermoMerge (Ours) achieves $76.83\% \pm 0.84\%$. **SyMerge has a higher mean.**
*   **MLP KMNIST OOD:** Deterministic SyMerge achieves **$72.25\% \pm 0.37\%$**, while ThermoMerge (Ours) achieves $72.20\% \pm 0.19\%$. **SyMerge has a higher mean.**
*   **LoRA MNIST OOD:** Deterministic SyMerge achieves **$81.62\% \pm 1.72\%$**, while ThermoMerge (Ours) achieves $81.27\% \pm 1.57\%$. **SyMerge has a higher mean.**
*   **LoRA KMNIST OOD:** Deterministic SyMerge achieves **$67.43\% \pm 0.34\%$**, while ThermoMerge (Ours) achieves $67.41\% \pm 0.47\%$. **SyMerge has a higher mean.**

### Summary of the Empirical Deficit
Across **12 distinct real neural network configurations** (clean & OOD, MLP & LoRA, across 3 datasets), **deterministic SyMerge actually achieves an equal or superior mean accuracy compared to ThermoMerge in 9 out of 12 cases!**

This empirical reality completely invalidates the authors' core claims. It indicates that:
1. Real test-time adaptive model merging loss landscapes **do not contain the insurmountable sharp local basins** that trap deterministic optimizers in practice.
2. Standard deterministic gradient descent (SyMerge) is highly effective and consistently outperforms or matches SGLD in final accuracy and OOD generalization.
3. The hand-crafted 1D loss landscape (which contains artificial, high-frequency sinusoidal ripples explicitly designed to block gradient descent) is a highly unrealistic representation of actual deep learning parameter spaces. The massive 56.7% loss reduction achieved on this toy landscape is an artificial artifact that completely fails to translate to real deep learning tasks.
