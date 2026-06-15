# 4. Experimental Evaluation & Results Check

## Critical Evaluation of the Experimental Setup

### 1. Scale of Evaluation (Toy-Scale Benchmarks)
The experimental evaluation is restricted to a very small, toy-scale setting:
* **Datasets:** MNIST, FashionMNIST, CIFAR-10, and SVHN are small, standard academic image datasets. MNIST is heavily saturated and too simple to be representative of real-world multi-task learning challenges.
* **Model Backbone:** The backbone used is **ViT-Tiny** (`vit_tiny_patch16_224`), which has only **5.7M parameters** and a 192-dimensional representation space. This is extremely small compared to modern neural network architectures (such as ViT-Base/Large, ResNet-50, or modern LLMs like LLaMA with billions of parameters).
* **Layer Constraints:** The authors only fine-tune and merge the last two blocks (Blocks 10 and 11) of the model. In standard model merging and task arithmetic, all layers of the model are typically merged. Restricting the merge to only two blocks is a highly synthetic, artificial setting.

### 2. Under-tuned Expert (SVHN Performance Bottleneck)
The SVHN expert is fine-tuned for only 5 epochs, reaching a test accuracy of only **64.60%**.
* This under-tuned, weak expert acts as a severe performance bottleneck for all merging protocols. In Tables 4.4 and 4.5, the accuracy on SVHN for all merged models is between **17.00% and 30.00%**.
* Since SVHN is a 10-class dataset, 17.00% is barely above random guessing (10.00%).
* Evaluating merging protocols on experts that perform at near-random levels makes the results highly noisy and of questionable scientific value.

---

## Analysis of Baselines and Claims vs. Evidence

A critical reading of the empirical results reveals major contradictions that challenge the paper's core claims:

### 1. Simple L2 Regularization Outperforms the Proposed Method (Table 4.1)
The paper heavily criticizes heuristic methods and standard regularization, proposing CFR as a superior learning-theoretic alternative. However, the main results in Table 4.1 directly contradict this superiority:
* **Standard L2 Reg L3-Router** achieves:
  * **66.88%** on Homogeneous Streams (compared to R2D-Merge's **65.62%** — a **-1.26%** degradation for the proposed method).
  * **66.88%** on Sample-wise Heterogeneous Streams (compared to R2D-Merge's **65.62%** — a **-1.26%** degradation).
  * **65.88%** on Collapsed Heterogeneous Streams (compared to R2D-Merge's **65.62%** — a **-0.26%** degradation).
* Standard L2 decay is strictly superior to the proposed CFR method across all evaluation stream configurations! 
* From a practitioner's perspective, this is a fatal flaw: why would an engineer implement a complex offline covariance profiling pipeline, allocate disk storage for auxiliary covariance matrices, and handle specialized loading workflows, when they can simply use a standard L2 weight decay penalty and achieve **strictly higher classification accuracy**?

### 2. Contradiction between Motivation and Empirical Requirements (Table 4.3)
The paper's introduction and methodology strongly emphasize the need for robustness under **extremely sparse calibration data** (e.g., $N=64$ or fewer samples):
* In Table 4.3, the authors ablate the calibration sample size $N$:
  * For **$N=16$**, standard L2 decay outperforms CFR by **+1.76%** (59.88% vs 58.12%).
  * For **$N=32$**, standard L2 decay outperforms CFR by **+0.62%** (63.12% vs 62.50%).
* The authors acknowledge that under data constraints, the empirical covariance matrices are highly noisy estimators, and they explicitly state:
  *"under extreme data scarcity (where $N \le 32$ total samples), practitioners should default to standard isotropic L2 decay."*
* This creates a major logical conflict. The paper's primary motivation is to solve dynamic merging under extremely sparse calibration data, but the proposed method (CFR) actually **performs worse** than standard L2 decay in this exact low-data regime! CFR only begins to outperform standard L2 decay at $N \geq 128$ (+0.12%) or $N=256$ (+0.24%), which directly violates the paper's low-data calibration premise.

### 3. Verification of "Absolute Resilience" Claim
The authors claim that R2D-Merge is "absolutely resilient" to heterogeneity collapse (0.00% drop). However, as analyzed in Section 3, this is a trivial result:
* In Table 4.1, the **Static Layer-Wise (Optimized)** baseline also achieves **0.00%** drop and has the **exact same** average accuracy of **65.62%**.
* This confirms that the proposed method achieves resilience not through a superior dynamic routing manifold, but simply by suppressing all dynamic routing capacity and behaving identically to a static, layer-wise optimized model. Thus, the "evidence" for resilience is actually evidence of dynamic routing collapse.
