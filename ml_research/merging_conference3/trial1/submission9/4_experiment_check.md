# Empirical and Experimental Review

## 1. Experimental Setup and Dataset Choices
The experimental design is highly structured, comprehensive, and scientifically rigorous.
* **Simulating Uncoordinated Downstream Schedules:** To mimic a realistic scenario where models are fine-tuned independently and exhibit mismatched parameter updates, the authors fine-tune the experts on MNIST (3 epochs, lr=1e-3), FashionMNIST (2 epochs, lr=3e-3), and KMNIST (1 epoch, lr=2e-3). This is a well-designed mechanism to induce realistic scale mismatches across tasks (for example, FashionMNIST's parameter updates have standard deviations up to 1.9x larger than the other tasks).
* **Separate Validation and Test Splits (Zero Target Leakage):** Crucially, the authors avoid oracle target leakage (tuning hyperparameters on the test set) by separating their datasets into disjoint validation and test subsets. All hyperparameter grid searches (such as the global scaling coefficient $\lambda \in [0.3, 1.5]$ and Ties-Merging's pruning ratio $p \in [0.2, 0.4, 0.6]$) are performed strictly on the validation sets, and final performance is evaluated on held-out test sets.
* **Statistical Rigor:** All main results on the SimpleCNN benchmark are aggregated over **3 independent random seeds** and reported as mean $\pm$ standard deviation.

---

## 2. Baseline Selections and Correctness
The authors choose a robust suite of model merging baselines:
1. **Task Arithmetic (Ilharco et al., 2022)**
2. **Ties-Merging (Yadav et al., 2023)**
3. **DARE (Yu et al., 2024)**
4. **AdaMerging (Yang et al., 2024b):** The authors identify and correct a silent technical bug in AdaMerging where `load_state_dict()` severed PyTorch's autograd connection, ensuring a fully functional and optimized AdaMerging baseline via `torch.func.functional_call`.
5. **SVD Isotropic Merging (SAIM-like, 2025)**

---

## 3. Key Experimental Findings
* **Tuned SD-Scale (73.23%) and RMS-Scale (73.22%)** slightly outperform SVD Isotropic Merging (73.13%), Task Arithmetic (72.50%), and Ties-Merging (71.77%) on average, while significantly boosting individual weaker tasks like MNIST (+4.37%) and KMNIST (+3.94%) compared to Task Arithmetic.
* **AdaMerging fails** (62.79% accuracy) on this heterogeneous multi-domain setup due to the optimization landscape of test-time entropy minimization becoming ill-conditioned under severe scale mismatches.
* **Parameter-Free RMS-Scale (PF-RMS)** achieves an outstanding **72.23%** test average accuracy out-of-the-box. This successfully outclasses default un-tuned Task Arithmetic (71.68%) and Ties-Merging (71.81%) without requiring a disjoint validation set or post-hoc tuning.

---

## 4. High-Dimensional CLIP ViT-B/32 Evaluation
To address the evaluation scale gap of the SimpleCNN, the authors execute a physical merging evaluation on **36 real-world projection weight matrices from OpenAI's pretrained CLIP ViT-B/32 visual encoder**:
* Both RMS-Scale and PF-RMS achieve **exactly the same average cosine alignment (57.74%)** and isotropic balance (0.15% std) as the SVD Isotropic baseline. This provides outstanding empirical proof for their Frobenius Equivalence proof directly on real-world foundation weights.
* Crucially, SVD Isotropic takes **571.92 milliseconds per layer** due to cubic SVD scaling, while RMS-Scale takes only **5.67 milliseconds** and PF-RMS takes **6.50 milliseconds**—representing a massive **100x wall-clock speedup**! This confirms the supreme scalability of the proposed method to multi-billion parameter models.

---

## 5. Alternative Scale Estimators and Sensitivity Analyses
* **Ablation of Scale Estimators:** Under PF-RMS, the authors compare four scale estimators across their seeds:
  - *Harmonic Mean:* **72.63 $\pm$ 2.18%** average accuracy (best performer).
  - *Geometric Mean:* **72.52 $\pm$ 2.29%** average accuracy.
  - *Arithmetic Mean:* **72.23 $\pm$ 2.25%** average accuracy.
  - *Maximum Scale:* **67.81 $\pm$ 2.68%** average accuracy (severe degradation due to outlier dominance).
  This provides deep scientific insights, showing that the Harmonic mean is highly effective at damping extreme outlier updates.
* **Sensitivity to stability constant $\epsilon$:** The model's performance remains completely insensitive to varying $\epsilon \in \{10^{-12}, 10^{-8}, 10^{-5}, 10^{-4}\}$ (consistently 73.22% average), proving that it operates purely as a passive safeguard.
* **Sensitivity to clipping threshold $\gamma$:** Capping $\gamma \le 1.0$ suppresses scale restoration (yielding 19.23%), while setting $\gamma \in [1.5, 3.0]$ provides optimal dynamic scale restoration (72.23%) and safeguards against extreme orthogonal conflicts (where $\alpha^l \to 0$).

---

## 6. Weaknesses and Missing Elements in the Evaluation
Despite the exceptionally high quality of the experimental section, we identify a minor weakness:
1. **Evaluation Scale Gap for Classification Accuracy:**
   While the activation-space alignment analysis is conducted on real-world CLIP ViT-B/32 weights, the end-to-end multi-task classification accuracy evaluation remains restricted to the SimpleCNN benchmark on grayscale MNIST datasets. Grayscale MNIST-like datasets are extremely low-resolution (28x28) and are considered toy tasks in modern deep learning (2026). The paper would be stronger if they reported final end-to-end multi-task classification accuracy of a merged CLIP model on standard downstream vision benchmarks (e.g., Stanford Cars, DTD, EuroSAT, SUN397, etc.).
   
   *Note:* The authors have explicitly acknowledged this evaluation scale gap under the Limitations section and positioned it as a primary direction for future library releases. While the activation-space alignment is a strong proxy, it does not guarantee that downstream classification accuracy is preserved.

---

## 7. Empirical Rating
**Excellent / Good.** The experiments are comprehensive, highly detailed, statistical, and address the evaluation scale gap using real-world foundation model weight layers. The baseline implementations are correct, and target leakage has been meticulously avoided. The addition of alternative scale estimators and sensitivity analyses demonstrates exemplary scientific rigor.
