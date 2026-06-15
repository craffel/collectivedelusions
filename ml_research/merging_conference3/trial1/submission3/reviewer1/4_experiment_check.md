# Evaluation Task 4: Experimental Setup and Results Check

## Datasets and Model Scale (The "Toy" Limitation)
The most prominent weakness of the paper's empirical evaluation is the choice of datasets and model scale:
1. **Toy Datasets:** All actual neural network experiments are conducted on **MNIST, FashionMNIST, and KMNIST** task splits. These are small-scale, grayscale, $28 \times 28$ image classification benchmarks.
2. **Toy Architecture:** The model used is a **2-layer Multi-Layer Perceptron (MLP)** with 128 and 64 hidden units, totaling around $10^5$ parameters.
3. **PEFT Subspace:** The LoRA experiments use a rank $r=4$ adapter on this tiny MLP.

### Critique from a Practical Standpoint:
In the modern model merging and test-time adaptation literature, the standard benchmarks involve **large-scale foundation models** (e.g., CLIP ViT-B/16, LLaMA-7B/13B, Mistral-7B) evaluated on complex real-world datasets (e.g., ImageNet, Cars, PACS, MMLU, GLUE, GSM8k). Evaluating an optimization algorithm solely on 2-layer MLPs and MNIST digits fails to demonstrate that the method works or scales in real-world scenarios. A practitioner looking to merge task-specific CLIP experts or LLM adapters will find these results unconvincing and irrelevant to actual deployment challenges.

---

## Detailed Accuracy Performance Analysis
A close examination of the results in Tables 6, 7, 8, and 9 reveals that the empirical gains of ThermoMerge over standard deterministic joint adaptation (SyMerge) are **extremely marginal, statistically weak, or non-existent**:

### 1. Joint MLP Merging (Table 6 - Clean Accuracies):
* **MNIST:** Deterministic (SyMerge) achieves **$89.97\% \pm 0.19\%$** vs. ThermoMerge **$89.94\% \pm 0.16\%$** (ThermoMerge is slightly worse).
* **FashionMNIST:** SyMerge achieves **$84.61\% \pm 0.49\%$** vs. ThermoMerge **$84.46\% \pm 0.59\%$** (ThermoMerge is slightly worse).
* **KMNIST:** SyMerge achieves $80.35\% \pm 0.19\%$ vs. ThermoMerge **$80.37\% \pm 0.24\%$** (ThermoMerge is $0.02\%$ better, completely within standard deviation).

*On standard MLP merging, ThermoMerge fails to outperform its deterministic baseline (SyMerge) on 2 out of 3 tasks, and the only improvement is a statistically insignificant $0.02\%$.*

### 2. Joint MLP Merging Under OOD Gaussian Corruption (Table 8 - Robustness):
* **MNIST:** SyMerge achieves $87.51\% \pm 0.53\%$ vs. ThermoMerge **$87.49\% \pm 0.54\%$** (ThermoMerge is slightly worse).
* **FashionMNIST:** SyMerge achieves **$76.91\% \pm 0.83\%$** vs. ThermoMerge **$76.83\% \pm 0.84\%$** (ThermoMerge is slightly worse).
* **KMNIST:** SyMerge achieves **$72.25\% \pm 0.37\%$** vs. ThermoMerge **$72.20\% \pm 0.19\%$** (ThermoMerge is slightly worse, though it has lower variance).

*Under image corruption, the full ThermoMerge (Ours) consistently underperforms deterministic SyMerge across all three datasets.*

### 3. LoRA PEFT Merging (Table 7 - Clean Accuracies):
* **MNIST:** SyMerge achieves $88.68\% \pm 0.86\%$ vs. ThermoMerge **$88.65\% \pm 0.63\%$** (ThermoMerge is slightly worse).
* **FashionMNIST:** SyMerge achieves $77.42\% \pm 1.52\%$ vs. ThermoMerge **$78.41\% \pm 1.67\%$** (ThermoMerge is $0.99\%$ better, but standard deviations overlap heavily).
* **KMNIST:** SyMerge achieves $76.57\% \pm 0.18\%$ vs. ThermoMerge **$76.62\% \pm 0.28\%$** (ThermoMerge is $0.05\%$ better, within standard deviation).

*Once again, on 2 out of 3 clean tasks, ThermoMerge is identical or worse than SyMerge.*

### 4. LoRA PEFT Merging Under OOD Gaussian Corruption (Table 9 - Robustness):
* **MNIST:** SyMerge achieves $81.62\% \pm 1.72\%$ vs. ThermoMerge **$81.27\% \pm 1.57\%$** (ThermoMerge is worse).
* **FashionMNIST:** SyMerge achieves $64.57\% \pm 1.38\%$ vs. ThermoMerge **$65.68\% \pm 2.14\%$** (ThermoMerge is $1.11\%$ better, but standard deviations overlap).
* **KMNIST:** SyMerge achieves **$67.43\% \pm 0.34\%$** vs. ThermoMerge **$67.41\% \pm 0.47\%$** (ThermoMerge is slightly worse).

*Under LoRA OOD adaptation, ThermoMerge underperforms SyMerge on 2 out of 3 tasks.*

### Summary of Performance:
Across **12 evaluated deep neural network configurations** (clean/corrupted $\times$ full/PEFT $\times$ 3 datasets), the full ThermoMerge (Ours) method:
* **Underperforms** deterministic SyMerge in **8 out of 12 configurations**.
* **Is virtually identical** (within standard deviation) in **2 out of 12 configurations** (MNIST LoRA clean, KMNIST LoRA clean).
* **Shows minor, overlapping improvements** in **only 2 configurations** (FashionMNIST LoRA clean $+0.99\%$, FashionMNIST LoRA corrupted $+1.11\%$).

---

## Performance Compared to Static Baselines (Negative Adaptation)
A particularly concerning finding appears in Table 7 (LoRA clean):
* **Static Task Arithmetic** achieves an accuracy of **$89.85\%$** on MNIST.
* **Deterministic SyMerge** achieves **$88.68\% \pm 0.86\%$**.
* **ThermoMerge (Ours)** achieves **$88.65\% \pm 0.63\%$**.

This means that both test-time adaptation methods actually **degrade** the performance of the merged model by over **$1.15\%$ to $1.20\%$** compared to a simple, training-free, static model average! This "negative adaptation" is a known hazard of test-time optimization under data scarcity and label bias, and the fact that ThermoMerge cannot outperform static Task Arithmetic on MNIST LoRA clean undermines the necessity of its complex stochastic machinery.

---

## Over-Reliance on Synthetic 1D Simulation
The paper devotes a substantial portion of its text, figures, and tables (Tables 1, 2, 3, 4, 5, Figures 1, 2) to a synthetic, hand-crafted 1D loss landscape ($\mathcal{L}_{TT}(\Lambda)$). While this 1D visualization is highly effective at illustrating how SGLD can escape a narrow local trap and find a flat global minimum, it is a highly contrived, low-dimensional environment. 
* The elegant physical analysis (e.g., calculating partition functions $Z(T)$, average energy $\langle E \rangle$, Shannon entropy, and detecting a Specific Heat capacity $C_v$ peak at $T_c \approx 0.02$) is **strictly illustrative**.
* As the authors transparently admit, computing these thermodynamic quantities is mathematically and computationally intractable for real deep neural networks. 
* Thus, the theoretical core of the paper (the thermodynamic crystallization phase transition) is entirely non-verifiable on real neural networks. For actual architectures, the thermodynamic framing acts merely as a loose metaphor.

---

## Computational Cost and Engineering Overheads
The authors argue that SGLD adds virtually zero computational complexity ($O(d)$ complexity and a negligible $\approx 1.5\%$ wall-clock latency overhead in Table 5) compared to SyMerge. However:
1. This is evaluated on a tiny 2-layer MLP with only $10^5$ parameters.
2. In billion-scale models (e.g., LLaMA-7B), sampling a random Gaussian vector of size equal to the adapted parameters at *every* single step introduces non-trivial system overhead (memory allocation, random number generation latency, seed synchronization across distributed GPU ranks).
3. Implementing the recommended **unsupervised safeguards** (Predictive Agreement, Entropy Monitors, Emergency Quenching, and Dynamic Rolling Calibration) dramatically increases the code's complexity, requiring a massive engineering effort to deploy and maintain safely in real-world, distributed systems.
