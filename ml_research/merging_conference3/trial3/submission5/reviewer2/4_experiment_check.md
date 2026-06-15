# Experimental Evaluation and Setup Check: Q-PolyMerge

## 1. Critical Evaluation of the Experimental Setup
Through a rigorous empirical lens, the experimental setup exhibits several severe limitations in terms of scale, data diversity, and statistical strength:

### A. Toy-Scale Architecture and Datasets
- **Backbone Network:** The authors utilize **ViT-Tiny** (approximately 5.7M parameters). While suitable for low-power edge microcontrollers, this is an extremely small, lightweight model. In modern model merging and test-time adaptation research (as of 2025/2026), the standard evaluation involves large foundation models (e.g., LLaMA-7B/70B, CLIP-ViT-B/16, or large VLMs). Evaluating exclusively on a 5.7M parameter toy backbone limits the empirical strength and generalizability of the findings.
- **Datasets:** The evaluation is conducted on **MNIST, FashionMNIST, CIFAR-10, and SVHN**. These are highly saturated, classic toy vision datasets. Furthermore, MNIST and FashionMNIST are 28x28 grayscale images, while CIFAR-10 and SVHN are 32x32 color images. Since the `vit_tiny_patch16_224` backbone requires 224x224 inputs, these tiny images must be upsampled by a factor of 7x, which is a highly artificial setup. Evaluating on upsampled toy datasets does not accurately reflect the complexity, clutter, and high-frequency features of real-world edge deployment scenarios.

### B. Statistical Limitations and Weakness
- **Sample Size:** The authors use an extremely downscaled evaluation pipeline:
  - **512 training samples** per task (to train the experts).
  - **16 unlabeled calibration samples** (for test-time adaptation).
  - **2000 test samples** per task.
- **Critique of Low-Data Regime:** While the authors argue that this downscaled setup mimics extreme data scarcity in edge environments, it severely limits the statistical power of the results. 
  - An adaptation set of only 16 unlabeled images across 4 tasks means the model receives only **4 images per task**. Optimizing a multi-task objective on just 4 images per domain is highly unstable and noise-prone.
  - The number of random trials is restricted to **3 seeds** (42, 100, 2026). While standard, 3 trials is a small sample size for drawing robust statistical conclusions, especially given the massive standard deviations reported for individual tasks.
- **Extremely High Variance:** Looking at the individual task results, the standard deviations across the 3 seeds are extraordinarily high:
  - Under 8-Bit Q-PolyMerge (ES): MNIST std dev is **11.75%**, FashionMNIST is **9.11%**, CIFAR-10 is **18.95%**, and SVHN is **9.07%**.
  - Under 8-Bit Q-Merge (ES): MNIST std dev is **21.12%**, CIFAR-10 is **19.66%**, and SVHN is **12.66%**.
  - *Analysis:* A standard deviation of **18.95%** or **21.12%** on a task indicates that the optimization is highly unstable and sensitive to the specific 16 calibration images sampled in that seed. With such massive overlapping confidence intervals, drawing definitive conclusions about the superiority of one method over another is statistically weak.

## 2. Baseline Comparison and Fairness
- **Tuning of Baselines:** The paper does not specify if the learning rates, step sizes, or optimization schedules of the baselines (AdaMerging, Q-Merge) were independently tuned and optimized for this specific low-data (16 images) and toy-scale setup. If the baselines were evaluated using default hyperparameters while Q-PolyMerge was highly tuned, the comparison is inherently unfair.
- **Omission of the Task-Wise Baseline:** As discussed in the methodology review, the failure to compare against an optimized **Task-Wise Q-Merge** (4 parameters) is a major empirical gap. Without this baseline, the claim that layer-wise polynomial variation is beneficial remains unproven.

## 3. Support for Claims
- **Does the data support the "Overfitting-Optimizer Paradox"?** 
  - Yes, to some extent. The unconstrained Q-Merge (Adam STE) exhibits higher variance across seeds (2.36% average std dev vs 1.22% for Q-PolyMerge) and learns highly jagged trajectories. Bounding the search space to a polynomial curve stabilizes the variance.
- **Does the data support on-device feasibility?**
  - **The Gap:** The systems-level SRAM metrics (Table 3) and hardware latency/energy metrics (Table 4) are **modeled and theoretical projections** rather than physical on-chip measurements. 
  - **Critique:** The paper claims to enable "on-device adaptation... on physical edge microcontrollers for the first time." However, there is **zero physical hardware-in-the-loop evaluation** on actual microcontrollers (e.g., ARM Cortex-M7 or RISC-V GAP8) to measure actual clock cycles, SRAM memory fragmentation, or battery drain. Relying purely on theoretical mathematical derivations of activation caching and modeled latency/energy profiles does not satisfy the empirical standard required to claim physical, on-device viability.
- **Inconsistent Performance Gains:** Under 8-bit PTQ, Q-PolyMerge (Adam) achieves **59.76%** average accuracy, which is actually *worse* than the unconstrained Q-Merge (Adam) at **60.03%**. The authors justify this by highlighting the reduction in variance and the +29% gain on SVHN, but on the other three tasks, the unconstrained optimizer is superior. Under 4-bit PTQ, Q-PolyMerge (Adam) outperforms Q-Merge (Adam) on average, but is still **-6.54% worse** on CIFAR-10. This inconsistent task-wise performance indicates that the polynomial prior is a double-edged sword, causing severe underfitting on less sensitive tasks.
