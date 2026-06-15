# 4. Experimental Check

## Physical Empirical Setup & Scale-Up Limitations
The empirical setup is designed to isolate and study routing trajectories with high precision, and the authors have expanded their experiments to natural images:
- **Architectures:** The paper evaluates a deep multi-layer fully connected network (**DeepMLP-12** with 12 layers) and a convolutional network (**TinyCNN-4** with 4 blocks).
- **Primary Dataset:** Split-MNIST subsets (Task 0: digits 0-1, Task 1: 2-3, Task 2: 4-5, Task 3: 6-7).
- **Task Suites:** Designed to scale semantic conflict: Low-Conflict ($K=2$), High-Conflict ($K=2$), and Cross-Domain ($K=4$).
- **Stochastic Control:** All physical results are averaged over 5 independent random seeds and reported as Mean $\pm$ Standard Deviation.

### Physical Natural-Image Experiment:
In response to previous feedback, the authors have added a physical natural-image weight-space merging experiment on CIFAR-10 (natural objects) and SVHN (street digits) using a 4-layer 3-channel Convolutional Neural Network backbone (`NaturalCNN-4`) in Section 4.5. Under standard few-shot calibration (128 samples per task), the BSigmoid Layer-wise Router achieves $20.20 \pm 1.71\%$ accuracy, outperforming the L1-Global Router baseline ($20.05\%$) and Static Uniform merging ($15.10\%$). The SVD Collinearity Ratio registers as $0.9167$, showing a structured routing trajectory.
- **Remaining Limitation:** While this is a highly commendable addition, `NaturalCNN-4` is still a very small toy convolutional network, and the resulting joint accuracy of $20.20\%$ is extremely low (even though it is above the $5\%$ random guessing barrier for 20 classes). The paper still lacks empirical scale-up verification on standard high-capacity architectures (such as ViT-B/16 or ResNet-50) on natural images, relegating these to theoretical discussions or preliminary simulations in the appendix.

## Baseline Comparisons & The Utility Question
The paper compares its Layer-wise Router against five baselines:
1. **Static Uniform Merging:** Fixed equal blending across all experts.
2. **OFS-Tune (Static):** Optimizes a single global blending weight vector across the depth.
3. **L1-Global Router:** Replicates the same dynamic routing decision across all layers.
4. **Layer-wise (No Reg):** Ablates the default $L_2$ weight decay regularization.
5. **Oracle Ceiling:** Separate forward passes through specialized experts.

The author provides a sound, mathematically correct justification for omitting advanced static alignment baselines (such as ZipIt! or TIES-Merging), noting that since all experts are fine-tuned from a **shared base model initialization**, they reside in the same local loss basin. Under this prerequisite, permutation alignment and sign-conflict pruning collapse back to standard arithmetic interpolation and provide no additional representational benefits, making Uniform and OFS-Tune the most direct and correct baselines.

### The Utility Question & Calibration Budget Scaling Analysis:
On TinyCNN-4 under few-shot calibration (128 samples per task), the static baseline **OFS-Tune** outclasses the proposed dynamic Layer-wise Router across all three task-conflict suites (e.g., $53.40\% \pm 7.16\%$ vs. $52.52\% \pm 5.95\%$ on Cross-Domain). 
To address this, the authors have added a calibration budget scaling analysis in Figure 4 and Section 4.5:
- **Crossover Point:** They demonstrate that under scarce data ($B = 64$ samples), OFS-Tune outperforms the dynamic router due to near-zero parameter variance. However, as the calibration budget scales (crossing over at $B \ge 256$ samples per task), the dynamic router's high capacity is unlocked, reaching $54.50\% \pm 8.64\%$ at $B = 1024$ samples.
- **Remaining Gap:** While this scaling plot is an excellent addition that successfully addresses the "Practical Utility" critique, it reveals a sobering reality: even at $B = 1024$ samples, the dynamic router's accuracy of $54.50\%$ is only marginally superior to the static OFS-Tune ($53.40\%$) and remains far below the Oracle ceiling ($99.30\% \pm 0.23\%$). This demonstrates that even under larger calibration budgets, the representational damage of linearly blending spatial convolutional filters acts as a massive bottleneck, and the practical advantage of high-capacity dynamic routing over simpler static compromises is extremely small.

## Scientific Integrity and Self-Critical Audits
The paper is highly commendable for its outstanding scientific honesty, openly reporting and critically deconstructing the limitations and failure modes of its own proposed methods:

1. **The Parameter-Variance Constraint (OFS-Tune Superiority):** The author openly highlights and analyzes the superiority of OFS-Tune on TinyCNN-4 as a *Variance-Capacity Trade-off*, demonstrating that weight sharing and translation invariance in CNNs cushion the weight-space under parameter shifts.
2. **Representational Damage & The Random Guessing Barrier in MLPs:** On DeepMLP-12 under Cross-Domain task conflict, the Layer-wise Router achieves a statistically significant improvement over other merging models ($16.15\% \pm 5.60\%$ vs Uniform's $11.80\%$), but the author explicitly acknowledges that this is barely above the random guessing barrier of $12.5\%$. The paper openly states: *"full-parameter linear interpolation of deep, fully connected layers under multi-task conflict is fundamentally a failed paradigm."*
3. **The Massive Oracle Gap on CNNs:** The author audits the substantial $47\%$ performance gap between the Layer-wise Router ($52.52\%$) and the Oracle ceiling ($99.30\%$) on TinyCNN-4 Cross-Domain, explaining that linearly blending spatial convolutional filters acts as a low-pass filter, destroying high-frequency edge-detection capabilities.
4. **Learnable Projections vs. Random Projections:** Making the projection learnable increases parameters by over $4,300\%$. Under tight few-shot budgets, this leads to extreme training memorization and a catastrophic generalization collapse on the test set, with accuracy dropping from $52.52\%$ down to $26.12\%$ on TinyCNN-4 Cross-Domain.

## Spectral and Diagnostic Stability
- **Collinearity Drop:** SVD results show a sharp drop in the Collinearity Ratio to $0.4987 \pm 0.08$ (DeepMLP-12) and $0.5673 \pm 0.03$ (TinyCNN-4) under Cross-Domain task conflict. This proves multi-dimensional, depth-specialized routing.
- **Robustness to Projection Seeds:** In Table 9 of the Appendix, the SVD Collinearity Ratio is evaluated across 5 random projection seeds, exhibiting an extremely tiny standard deviation of $\pm 0.003$ on both architectures. This confirms that the spectral audit is highly robust to projection matrix initialization and is a stable diagnostic.

Overall, the empirical evaluation has been significantly strengthened with the natural image experiments and budget scaling analysis, though the findings still highlight the severe performance limits of physical full-parameter weight blending compared to the Oracle ceiling.
