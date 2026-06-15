# 4. Experimental Setup and Evaluation Check

## Evaluation of Experimental Setup
The authors evaluate CAM-Router on a multi-task setup consisting of four disparate image recognition datasets: MNIST, FashionMNIST, CIFAR-10, and SVHN. They use a "high-fidelity token-level sandbox representing a 14-layer compact Vision Transformer backbone." 
While this setup covers diverse domains, several critical issues emerge upon close examination:

## Critical Empirical Weaknesses and Contradictions

### 1. Inexplicable Statistical Contradictions (Table 4 vs Table 6)
There is a massive, unexplained contradiction between the main results in Table 4 and the batch-size sweeps in Table 6:
- **BSigmoid-Router anomaly:** In Table 4, the Joint Mean Accuracy of the `BSigmoid-Router` baseline is reported as **28.70%** (with individual accuracies around 26.67% to 29.87%). However, in Table 6 (under the Batch Size and Heterogeneity Resilience sweep), at $B=1$ (which corresponds to single-sample inference, equivalent to Table 4's evaluation condition), the BSigmoid-Router accuracy is reported as **58.33%**! This is more than *double* its reported main performance. How can the same baseline under identical evaluation conditions (batch size 1) exhibit such wildly different accuracies?
- **CAM-Router anomaly:** Similarly, in Table 4, CAM-Router achieves a Joint Mean Accuracy of **53.07%**. But in Table 6, at $B=1$, its accuracy is reported as only **50.00%**. Even more bizarrely, its accuracy *increases* to **62.50%** at $B=8$ and **58.33%** at $B=32$. If single-sample inference is the cleanest, most deterministic mode that avoids "heterogeneity collapse", why does CAM-Router perform *worse* at $B=1$ than at $B=8$ or $B=32$?
These contradictions suggest severe errors in the experimental logging, or a highly unstable evaluation pipeline, throwing the validity of all reported metrics into doubt.

### 2. Underperforming and Poorly Tuned Baselines
- **Worse than Static:** All five dynamic routing baselines (Unreg. Global Linear, Reg. Global Linear, QWS-Merge SOTA, BSigmoid-Router, and L3-Router) achieve Joint Mean Accuracies between 24.90% and 28.77%, which is **significantly worse** than the simple `Static Uniform` baseline (41.97%).
- This is highly atypical. A dynamic router, which has access to sample-specific features and trainable weights, should easily learn to at least mimic a uniform average (by setting routing weights to be roughly equal) and achieve at least 41.97%. The fact that all dynamic baselines collapse to around 25-28% suggests they were not properly tuned, under-trained, or implemented incorrectly in the custom "sandbox".
- The paper lacks standard, highly relevant baselines, such as **Task Arithmetic** (with tuned static coefficients), **Ties-Merging** (with tuned static coefficients), and **activation-routing** (where the input is forwarded to the single best expert, which serves as the upper bound of 85.85% joint accuracy).

### 3. Lack of Real-World Scale and Validation
- The entire evaluation is conducted on a "14-layer compact Vision Transformer coordinate sandbox" with toy datasets (MNIST, FashionMNIST, CIFAR-10, SVHN).
- In modern model merging literature (e.g., at ICML), it is standard to evaluate on large-scale models and datasets, such as merging CLIP models (ViT-B or ViT-L) on ImageNet and its variants (ImageNet-A, ImageNet-R, ImageNet-Sketch) or merging large language models (LLMs) like LLaMA-7B across instructions/reasoning tasks.
- Relying entirely on a "simulated coordinate sandbox" on toy datasets makes the empirical findings extremely weak and limits their applicability to real-world foundation models.

### 4. Abysmally Low Absolute Accuracies
- While CAM-Router is reported to outperform the (apparently broken) dynamic baselines, its absolute performance is extremely poor:
  - **MNIST:** 65.47% (vs. 97.26% for the individual expert)
  - **FashionMNIST:** 58.67% (vs. 87.43% for the individual expert)
  - **CIFAR-10:** 31.07% (vs. 73.71% for the individual expert)
  - **SVHN:** 57.07% (vs. 85.00% for the individual expert)
- A merged model that gets only 31.07% accuracy on CIFAR-10 (where random guessing is 10%) and 65.47% on MNIST is practically useless. The severe performance degradation (a drop of ~32.8% in Joint Mean Accuracy compared to the individual experts at 85.85%) indicates that weight-space dynamic merging on these compact backbones causes catastrophic representational interference that CAM-Router only marginally mitigates.
