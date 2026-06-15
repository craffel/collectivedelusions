# Paper Summary

## Main Topic and Goal
The paper addresses the challenge of **parameter-space model merging**—composing multiple task-specific expert models (fine-tuned from a common pre-trained base model) into a single, multi-task unified model without joint training. 

The primary goal is to develop a **training-free, forward-only adaptive model merging framework** named **EdgeMerge**. This framework is specifically designed to overcome the severe compute, latency, and memory constraints of modern test-time adaptation or gradient-based merging methods (such as SyMerge or AdaMerging) during edge deployment, enabling efficient, near-instantaneous model composition.

---

## Proposed Approach
EdgeMerge operates by calculating localized, channel-wise merging coefficients in closed-form on a target "choke-point" bottleneck layer using a tiny, unlabeled calibration dataset (e.g., 32 samples per task) in a single forward pass. 

The approach is divided into three main stages, localized exclusively to the visual projection bottleneck layer (`model.visual.proj`) representing $<0.5\%$ of the model parameters:
1. **Forward-Only Activation Sampling (FOAS):** Samples internal feature activations from the base model and task experts on a tiny calibration batch. To save memory and latency, input representations are extracted solely from the base model's visual encoder and shared across all expert projections.
2. **Scale-Normalized Delta Activation Salience (SNDAS):** Computes the shift in activations between task experts and the base model, and normalizes them using a Frobenius norm to prevent experts with larger natural activation scales from dominating the merge. It then computes a channel-wise salience score.
3. **Channel-Wise Softmax Gating (CWSG):** Normalizes the saliency scores across tasks using a softmax function with a temperature hyperparameter $\tau$. These gated weights are then used to reconstruct the merged weight matrix.

To resolve a mathematical scaling discrepancy introduced by softmax normalization in the gated bottleneck layer versus static summation in the rest of the network, the paper proposes **Decoupled Scale Routing (DSR)**. DSR decouples the scaling coefficient of the gated projection layer ($\lambda_{proj}$) from that of the static layers ($\lambda_{static}$).

---

## Key Findings
1. **Pragmatic Efficiency:** EdgeMerge completely bypasses backpropagation and gradient tracking. On the 8-task Vision-Language CLIP ViT-B/32 benchmark, EdgeMerge reduces preparation latency from 10 minutes (for SyMerge) to just **11.95 seconds** (a $50\times$ speedup), with a training memory overhead of only $\sim$100 MB.
2. **Performance Trade-Off:** Standard EdgeMerge (coupled scaling) achieves **68.69%** average accuracy on the 8-task benchmark, which matches the peak optimized Task Arithmetic (TA) baseline (**68.74%**) but suffers a substantial performance drop of **21.05%** absolute points compared to server-grade gradient-based SyMerge (**89.74%**).
3. **Decoupled Scale Routing (DSR) Performance:** When applying DSR, EdgeMerge achieves **68.82%** in the Scale-Aligned routing regime ($\lambda_{static}=0.20, \lambda_{proj}=1.80$), outperforming standard Task Arithmetic. In the High-Scale Composition with Bottleneck Regularization regime ($\lambda_{static}=0.25, \lambda_{proj}=0.20$), it achieves a peak average accuracy of **69.58%** (+0.84% over peak TA).
4. **Hyperparameter Robustness:** While Task Arithmetic exhibits a highly fragile, narrow performance peak around $\lambda=0.20$ and degrades rapidly elsewhere, EdgeMerge opens up a broad, stable plateau of high performance, making it much safer to deploy in practice.
5. **Ablation Insight:** The ablation studies reveal that the elaborate activation-guided channel gating machinery (SNDAS, FOAS, CWSG) does not actually outperform a simple, uniform gating configuration ($\alpha_k = 1/K$) or global layer-wise gating, both of which achieve virtually identical performance (69.58% and 69.59% respectively) under the DSR framework. The performance boost is primarily driven by Decoupled Scale Routing (DSR) itself.

---

## Explicitly Claimed Contributions and Evidence
The paper explicitly claims several key contributions:
- **Training-Free, Forward-Only Adaptive Merging:** Fully described mathematically and implemented as a pipeline that takes only 11.95 seconds and $\sim$100 MB memory on a single forward pass over 32 samples per task. (Validated by resource profiling in Table 7).
- **Decoupled Scale Routing (DSR):** Outlines the mathematical scale discrepancy in coupled merging and proposes DSR. (Supported by performance improvements from 68.69% to 68.82% and 69.58% in Section 4.3.3).
- **Strategic Choke-Point Bottleneck Selection:** Localizes gating to the single visual projection layer, which minimizes calibration overhead and acts as a visual router. (Justified in Section 3.7 and supported by the CLIP ViT-B/32 experiments).
- **Hyperparameter Robustness (Plateau Preservation):** Shows that EdgeMerge mitigates the risk of static weight averaging by opening a wide, stable scaling plateau. (Evidence provided in Section 4.3.5 and Figure 3).
- **Representational Invariance of CLIP Latent Spaces:** Demonstrates that running calibration with the base model encoder features vs. expert encoders yields identical results. (Proven quantitatively in Table 6).
