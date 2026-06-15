# 1. Summary of the Paper

## Main Topic
The paper addresses the challenge of **parameter-space model merging** on resource-constrained edge devices. Standard adaptive merging methods (like AdaMerging or SyMerge) rely on test-time gradient-based optimization, which is computationally expensive, memory-heavy, and prone to overfitting on small test batches. To overcome these deployment bottlenecks, the paper proposes **EdgeMerge**, a training-free, forward-only adaptive model merging framework that extracts fine-grained, channel-wise merging coefficients in a single forward pass without backpropagation.

## Proposed Approach
EdgeMerge operates by computing localized, channel-wise merging coefficients in closed-form using a small, unlabeled calibration dataset. It introduces three primary components:
1. **Forward-Only Activation Sampling (FOAS):** Samples activation features from a single forward pass of a tiny calibration batch (e.g., $B = 32$ samples per task) through the base model and expert models. To optimize memory and latency, the input features $X_k$ are extracted exclusively from the pre-trained base model's encoder and reused for all experts.
2. **Scale-Normalized Delta Activation Salience (SNDAS):** Calculates the functional importance of each channel by computing the Frobenius-norm normalized difference between the expert's activations and the base model's activations.
3. **Channel-Wise Softmax Gating (CWSG):** Normalizes the channel-wise salience across tasks using a softmax function with a temperature parameter $\tau$, yielding localized routing coefficients $\alpha_k[j]$ for the visual projection bottleneck layer (`model.visual.proj`).
4. **Decoupled Scale Routing (DSR):** Separates the global scaling factor of the statically merged layers ($\lambda_{static}$) from the scaling of the gated visual projection layer ($\lambda_{proj}$) to resolve a mathematical scale discrepancy caused by softmax normalization.

## Key Findings
- **High Computational Efficiency:** EdgeMerge reduces preparation latency from 10 minutes (for gradient-based SyMerge) to just **11.95 seconds** (a $50\times$ speedup) while keeping training GPU memory overhead restricted to approximately 100 MB.
- **Performance Trade-offs:** EdgeMerge achieves a peak multi-task average accuracy of **68.69%** under coupled scaling (nearly identical to a fully optimized standard Task Arithmetic baseline of 68.74%). When coupled with **Decoupled Scale Routing (DSR)**, it achieves a global peak accuracy of **69.58%** (outperforming standard Task Arithmetic by +0.84% absolute points). However, this forward-only approach incurs a substantial performance gap of **21.05%** compared to server-grade, gradient-based optimization like SyMerge (89.74%).
- **Plateau Preservation & Hyperparameter Stability:** Standard Task Arithmetic is highly fragile with a narrow performance peak at $\lambda = 0.20$. In contrast, EdgeMerge exhibits a broad, stable performance plateau across a wide range of global scaling factors ($\lambda \in [0.20, 0.35]$) and temperatures $\tau \in [0.01, 2.00]$.
- **Robustness to Calibration Data:** Standard deviation across multiple random seeds is exactly $0.000\%$. Surprisingly, calibration using synthetic Gaussian noise or pure zeros yields the exact same average accuracy of 68.69% due to the pre-conditioning of inputs by the pre-trained CLIP encoder.

## Claimed Contributions and Evidence
1. **Training-Free, Forward-Only Adaptive Merging Framework:** EdgeMerge is proposed as the first framework to perform fine-grained channel-wise merging in a single forward pass. This is supported by runtime measurements (11.95s on a single node) and low memory footprint (~100 MB).
2. **Decoupled Scale Routing (DSR):** The paper identifies and addresses a scaling mismatch caused by softmax normalization, proving that decoupling $\lambda_{proj}$ from $\lambda_{static}$ can improve accuracy to 69.58%.
3. **Hyperparameter Stabilization:** The paper demonstrates that EdgeMerge opens up a broad, stable plateau of high performance, protecting deployment pipelines from the fragile, unguided scaling of standard Task Arithmetic. This is evidenced by the detailed grid sweeps and sensitivity plots.
4. **Seed-invariant and Data-Free Calibration:** EdgeMerge demonstrates that channel salience can be estimated robustly using synthetic data (noise or zeros) with zero standard deviation, backed by Cosine Similarity and Spearman Rank Correlation analyses between physical and synthetic saliency vectors.
