# Soundness and Methodology Evaluation: SABLE

## 1. Clarity of the Description
The methodology of SABLE is described with high clarity and mathematical precision. The equations for the Subspace Cosine Projection, Temperature-Scaled Softmax Routing, Activation Blending Layer, and Top-$M$ Expert Pruning are well-structured, self-contained, and easy to follow. The inclusion of Figure 1 (Architectural Schematic) and Algorithm 1 (Adaptive Dynamic Thresholding) greatly aids in understanding the sequential flow and runtime execution of the proposed system.

## 2. Appropriateness of Methods
Shifting the ensembling step from parameter space to activation space via the distributive law is a theoretically sound and elegant way to enable sample-specific routing coefficients within a batch. Leveraging parameter-efficient LoRA adapters to limit the computational and memory-bandwidth overhead of parallel passes is highly appropriate, as is the use of Top-$M$ expert pruning to bound the active inference complexity.

## 3. Potential Technical and Methodological Flaws
From a rigorous empirical and scientific perspective, several critical technical and methodological concerns must be highlighted:

### A. Lack of Statistical Soundness (No Random Seeds, No Confidence Intervals)
Across all quantitative results in the paper (Tables 1, 3, 5, 6, 7, 8), the authors report single point-estimates for joint mean accuracies. There are **no error bars, standard deviations, or confidence intervals**, and there is no mention of running experiments over multiple random seeds or initializations. 
- In Table 5 (Sandbox results), SABLE Late Adaptation (68.10%) is compared against PFSR+MBH (67.20%). The margin is only **0.90%**. Without reporting statistical variance over multiple seeds, it is impossible to verify whether SABLE actually outperforms PFSR+MBH or if this difference is within the range of random noise.
- In Table 3 (ResNet-18 features), SABLE Hybrid at $r=2$ (62.10%) is compared against SABLE Strict at $r=2$ (57.20%). While the +4.90% margin is larger, the lack of confidence intervals still undermines the scientific rigor of the claim.

### B. Speculative "Low-Rank Regularization Paradox"
In Table 3, the authors observe a non-monotonic performance trend for SABLE Hybrid: accuracy is 62.10% at $r=2$, drops to 58.90% at $r=4$, and then rises to 66.30% at $r=8$. The authors label this a "Low-Rank Regularization Paradox" and claim that constraining the intermediate hidden layer to $r=2$ acts as a "powerful regularizer" that filters representation noise.
- This claim is highly speculative and is not backed by any solid empirical evidence (such as training/validation loss curves, overfitting metrics, or evaluations across different architectures and larger datasets).
- A non-monotonic trend of this nature is often an artifact of suboptimal hyperparameter selection (e.g., learning rates, training epochs) or seed-specific variance rather than a reproducible mathematical phenomenon. Without further proof, this explanation is unconvincing.

### C. Weakness in Baseline Tuning and Evaluation
- **Linear Router (Unreg):** This parametric baseline is trained on only 64 calibration samples, resulting in 55.50% joint accuracy. Training a parametric router on such a tiny calibration set is almost guaranteed to lead to overfitting and poor generalization. The paper does not specify if any regularization (e.g., L2 decay, dropout) was applied, nor does it explore training the router on a larger, more realistic subset of the training data. This makes it an artificially weak baseline.
- **Tuning of Hyperparameters:** It is unclear if SABLE's hyperparameters ($\tau = 0.05$ and $\gamma_{\text{OOD}} = 0.2$) were tuned on a separate validation set or selected post-hoc to optimize the test performance. Furthermore, Table 7 shows that $\tau = 0.01$ achieves 67.50% in the sandbox compared to the default $\tau = 0.05$ (66.60%). The authors justify using $\tau = 0.05$ because lower temperatures collapse to hard routing, which is detrimental under confounded inputs. However, this trade-off is not systematically evaluated using a validation set, raising concerns of test-set overfitting.

### D. Theoretical Weakness of Task-Averaged Zero-Data Centroids
The Zero-Data Centroids construction method takes the row-wise mean of classification weights: $c_{\text{zero}, k} = \frac{1}{C}\sum_c W_{\text{expert}, k}[c, :]$. 
- Classification weights are optimized to project feature representations into class-specific logit spaces. Averaging these class vectors to form a single task centroid is theoretically questionable. In a multi-class task, different classes have highly distinct (often orthogonal or opposing) weight vectors. Averaging them washes out class-specific details.
- For a single test sample belonging to a specific class (e.g., digit '1' in MNIST), its penultimate representation $z_b$ is optimized to align with the weight vector for class '1'. It may project poorly or even negatively onto the average task centroid of all classes (especially if some class weights are anti-correlated). The authors' proposed weight L2-normalization (Refined Zero-Data Centroids) helps preserve orientation, but does not resolve the fundamental issue that averaging class-specific vectors degrades the specificity of the task-level representation.

### E. Scaling to Deep Hidden Layers and Cumulative Non-Linear Drift
Although the authors theoretically discuss "cumulative non-linear drift" (Section 4.4) and track cosine similarity over a 4-layer MLP (Table 9), a 4-layer MLP is a toy-scale setting.
- In deeper networks (such as 12-layer ViTs or 32-layer LLMs), the compounding effect of multiple sequential non-linear layers (ReLU, GeLU, LayerNorm) will cause SABLE's blended activations to drift significantly from the true expert manifolds.
- SABLE's default Late Adaptation configuration (which leaves the first 12 out of 14 layers unadapted) is used to avoid this drift. However, this highlights a critical limitation: SABLE is unable to perform effective multi-layer ensembling across deep structures without risking catastrophic activation divergence, forcing it to fall back to a single-layer or final-block ensembling mechanism.

## 4. Reproducibility
The authors provide highly detailed experimental setup notes (learning rates, epochs, samples, model sizes), which suggests a high degree of reproducibility. However, the lack of a provided codebase, random seeds, or standard dataset loaders means that reproducing the exact numbers would require rebuilding the coordinate sandbox from scratch. Additionally, because the sandbox is synthetic and its code is not fully disclosed, reproducing the sandbox-specific results (Table 5) is highly challenging.
