# 4. Experimental Evaluation and Critical Checks

## Critique of Experimental Setup and Datasets
1. **Contrived and Toy-Scale Academic Datasets:**
   The paper evaluates its method on a 4-task model merging benchmark consisting of **MNIST, FashionMNIST, CIFAR-10, and SVHN**. 
   * These are small, low-resolution datasets (28x28 and 32x32 pixels) representing extremely simple academic toy classification tasks.
   * Fine-tuning and merging experts for such widely orthogonal and simple domains is a highly artificial setup. In real-world applications, model merging is applied to large foundation models (such as LLaMA, Mistral, or CLIP) fine-tuned on complex, overlapping domains (e.g., code generation, mathematics, translation, or specialized medical/legal text). 
   * Evaluating *only* in this simulated toy "sandbox" fails to prove that CAM-Router is scalable or useful in practical, industry-standard deployment scenarios.
2. **Inappropriate Model/Image Configuration:**
   The backbone model used is `vit_tiny_patch16_224`, which is designed for 224x224 high-resolution images. Upscaling or padding 28x28 MNIST images to 224x224 to run them through a Vision Transformer backbone is computationally inefficient and represents a contrived experimental setting.

---

## Critical Data Inconsistencies and Contradictions
A close, rigorous audit of the numerical results in the paper reveals multiple major contradictions and errors that raise serious concerns about the reliability of the empirical findings:

### 1. Abstract vs. Main Body/Table 1 Discrepancies
* **Joint Mean Accuracy:** The Abstract claims a Joint Mean Accuracy of **57.07%** ($+15.10\%$ over Static Uniform). However, the Introduction and Table 1 (Section 4.2) state the Joint Mean Accuracy is **53.07%** ($+11.10\%$ over Static Uniform). It appears the authors mistakenly copied the SVHN accuracy of **57.07%** from Table 1 and reported it as the overall Joint Mean Accuracy in the abstract, using it to inflate their claimed performance improvements.
* **Spatial Occlusion Accuracy:** The Abstract claims a stable accuracy of **53.63%** under up to 80% patch masking. However, Table 3 (Sweep 2) shows the accuracy at 0.8 mask ratio is **50.57%**, and at 0.6 mask ratio is **55.37%**. The number **53.63%** does not appear anywhere in Table 3.
* **Batch Heterogeneity Accuracy:** The Abstract claims a stable accuracy of **55.47%** at batch size $B=256$. However, Table 4 (Sweep 3) shows the accuracy at $B=256$ is **54.30%**. The number **55.47%** does not appear in Table 4.

### 2. Deep Contradiction Between Table 1 and Table 4 (The $B=1$ Paradox)
Section 3.3 specifies that at inference time, absolute determinism is guaranteed by operating on a single-sample basis ($B=1$), which is the default operating mode. Thus, the $B=1$ row in Table 4 should logically match the main results in Table 1.
However, looking at the $B=1$ results in Table 4:
* **CAM-Router** achieves **50.00%** Joint Mean Accuracy (whereas Table 1 reports **53.07%**).
* **BSigmoid-Router** (the primary classical baseline) achieves **58.33%** Joint Mean Accuracy (whereas Table 1 reports **28.70%**).

**This is a fatal empirical contradiction:** 
In Table 4 at $B=1$, **BSigmoid-Router actually outperforms CAM-Router by a solid +8.33% absolute margin (58.33% vs. 50.00%)!**
If the baseline BSigmoid-Router is actually superior to CAM-Router in the single-sample inference mode, the central claim and entire motivation of the paper are invalidated. The authors do not acknowledge or explain this massive discrepancy.

---

## Fairness of Baselines
The reported performance of all dynamic routing baselines in Table 1 is suspiciously low:
* **QWS-Merge SOTA** (24.90%), **BSigmoid-Router** (28.70%), and **L3-Router** (28.77%) all perform significantly *worse* than the simple **Static Uniform** baseline (41.97%).
* A dynamic router trained via gradient descent on 800 calibration samples should, at the very least, be able to converge to the uniform merging weights (e.g., setting all weights to 0.3), which would yield 41.97% accuracy.
* The fact that all baselines perform nearly 13%--17% worse than Static Uniform suggests that either the baselines were not optimized properly, their hyperparameters were poorly tuned, or the training of baselines was deliberately crippled to make the proposed CAM-Router appear superior. This severely undermines the fairness of the comparison.
