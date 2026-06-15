# 3. Soundness and Methodology Evaluation

## Clarity and Mathematical Structure
The methodology is written with commendable clarity. The forward DCT-II and inverse DCT-III equations are formally correct and well-motivated. The authors' discussion of energy compaction and boundary-flattening effects (guaranteeing flat spatial derivatives at virtual boundaries) is mathematically sound. The extension to architectural heterogeneity via Block-wise Spectral Merging is also clearly explained and logically structured.

However, a highly critical examination of the methodology reveals several significant technical concerns, hidden assumptions, and methodological flaws.

## Methodological Flaws and Critiques

### 1. The Inactive Layer Optimization Paradox (Self-Inflicted Discontinuity)
In Section 4.6 (pre-trained ResNet-18 CIFAR-10 checkpoints), the authors fine-tuned only the final classification head and the deep convolutional layer block `layer4` (equivalent to layers 13 to 17), leaving layers 0 to 12 completely frozen. 

* **The Problem:** Because layers 0 to 12 are frozen and identical between the experts, the task vectors for these layers are exactly zero ($V_k^{(l)} = 0$ for $l \le 12$). Consequently, any merging coefficients $\alpha_k(l)$ for these layers have **absolutely zero physical effect** on the merged model weights. 
* **The Flaw:** Despite this, the authors attempt to optimize the merging coefficients across **all 18 physical layers**. Because they apply a global DCT-II across the entire 18-layer sequence, the coefficients across all layers become linearly coupled in the spectral coordinates. To adjust the active layers (13–17), the optimizer must change the spectral parameters, which inevitably introduces artificial oscillations and values in the inactive layers (0–12).
* **The Self-Inflicted Discontinuity:** This coupling creates a sharp "step-function discontinuity" in parameter sensitivity across depth. The authors frame this as a fundamental challenge ("PEFT-Induced Step-Function Discontinuity") and use it to explain why the hard-cutoff SpectralMerge-LP ($F=3$) collapses to $29.00\%$ accuracy, while SpectralMerge-Reg survives ($54.00\%$) by softly regularizing the high frequencies.
* **The Critical Insight:** This entire "discontinuity" is an artifact of a poorly designed optimization setup. If the authors had simply restricted the merging coefficient optimization to the **5 active layers** that actually changed, the search space would have been tiny (10 dimensions total), there would be no step-function discontinuity, and standard unconstrained spatial search would have worked perfectly without any global spectral coupling. The authors have introduced an artificial optimization bottleneck by optimizing inactive layers, and then claimed their soft-spectral regularization is a major breakthrough for "resolving" it.

### 2. Extremely Weak and Toy Physical Baselines
The paper's physical neural network experiments are conducted in highly artificial, toy regimes:
* **Toy MLP:** The MLP has only 12 layers and is evaluated on a synthetic classification dataset.
* **Toy ResNet-18 on CIFAR-10:** The experts are trained on only 120 samples per task, and are limited to simple binary classification tasks (Expert 0: Vehicles, Expert 1: Animals). 
* **Poor Absolute Performance:** Expert 1 achieves an extremely low accuracy of $65.00\%$ on a binary task (barely above a random coin toss of $50\%$). The final merged model SpectralMerge-Reg achieves only $54.00\%$ multi-task accuracy. While the authors spin $54.00\%$ as a "blowout performance" (since the overfitted spatial and polynomial baselines collapsed to $29.00\%$), in absolute terms, $54.00\%$ is extremely poor for a ResNet-18 on CIFAR-10 tasks. A model trained jointly or fine-tuned properly would easily achieve $>90\%$ accuracy. The extreme data scarcity ($M=15$) and weak expert models make it highly questionable whether these findings generalize to real-world, high-performance model merging scenarios.

### 3. Simplicity of DC-Only Baseline Under TTA
Under Online Test-Time Adaptation (Table 1), the **Online Global Task-Wise (DC-Only)** baseline (which optimizes a single scalar per task and shares it globally across all layers) achieves an accuracy of **85.91%**, which actually **outperforms** both Online SpectralMerge-LP (85.32%) and Online SpectralMerge-Reg (85.17%). 
* This suggests that in online TTA regimes, allowing any layer-wise variations (even low-pass ones) is unnecessary and harmful compared to a simple, highly constrained task-level scalar. The authors fail to adequately address why a practitioner should accept the mathematical complexity and hyperparameter tuning of SpectralMerge when a simple global scalar performs better in this regime.

## Reproducibility
The authors provide mathematical formulations, model hyperparameters, and experimental protocols. However, because the primary evaluation relies on a simplified, calibrated mathematical simulation (Model II) and a toy physical ResNet-18 dataset split, replicating these results exactly will depend heavily on the specific calibration constants and synthetic data generation seeds. The lack of standard, large-scale deep learning benchmarks (like GLUE or full VTAB) limits the reproducibility of the method on realistic, standard model merging pipelines.
