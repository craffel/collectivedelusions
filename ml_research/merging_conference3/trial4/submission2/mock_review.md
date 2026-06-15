# Mock Review: OmniMerge (Multi-Schema Stochastic Co-Optimization)

## 1. Paper Summary
This paper addresses a highly practical and critical bottleneck in edge-AI deployment: the post-training quantization (PTQ) schema mismatch that occurs when deploying merged models across heterogeneous hardware fleets. Weight-space model merging (e.g., Task Arithmetic, Model Soups) is a popular, zero-overhead paradigm for on-device multi-task ensembling. However, different edge compilers and ASICs (e.g., TPUs, DSPs, Apple Neural Engine, TensorRT) utilize highly heterogeneous, incompatible PTQ standards (ranging from symmetric per-channel to asymmetric per-tensor).

The paper demonstrates that prior quantization-aware model merging methods (like Q-Merge) optimize merging coefficients strictly under a single simulated operator. This triggers **cross-schema performance degradation**, where the learned coefficients overfit to the exactRounding boundaries of the training schema and fail when deployed on mismatched target compilers.

To resolve this, the authors introduce **OmniMerge**, a training-free, multi-schema stochastic co-optimization framework. OmniMerge operates during test-time calibration and requires zero hardware metadata, zero training, and adds zero latency or memory overhead at inference time. It achieves robustness via two core techniques:
1. **Stochastic Operator Sampling (SOS):** Stochastically selects an active quantization operator from a discrete pool of four hardware-relevant schemas at each optimization step. This acts as parameter-space data augmentation to prevent boundary-overfitting.
2. **Scale and Zero-Point Noise Perturbation (SZNP):** Perturbs scale factors and zero-point offsets dynamically with multiplicative and additive Gaussian noise, smoothing out the rugged, discontinuous rounding grid of standard post-training quantization.
3. **Task-Consensus Regularization (TCR):** An unsupervised ensembling penalty that prevents task-specific blending coefficients from drifting too far from their layer-wise average consensus.

The framework is evaluated on a Vision Transformer backbone (`ViT-Tiny`) across four image classification datasets: MNIST, FashionMNIST, CIFAR-10, and SVHN. The results demonstrate that OmniMerge out-performs standard baselines (Naive Merge-then-Quantize, Quantized AdaMerging, and Q-Merge) across five mismatched target post-training quantization schemas under robust 8-bit quantization, achieving an average accuracy of up to **50.78%** and delivering a worst-case ensembling gain of **+11.43%** relative to the FP16 Task Arithmetic baseline.

---

## 2. Strengths and Weaknesses

### Strengths
1. **High Practical Relevance:** The paper addresses a highly realistic and pressing edge-deployment challenge: the hardware-quantization mismatch in heterogeneous edge fleets. It moves beyond the common academic simplification of assuming a static, homogeneous target environment.
2. **Zero Inference Overhead:** Since OmniMerge is a test-time adaptation method that runs only once during calibration, it introduces absolutely zero memory, latency, or compute overhead during inference, making it highly viable for edge deployment.
3. **Comprehensive Baselines:** The paper compares OmniMerge against a highly rigorous and fair set of baselines, including Naive Merge-then-Quantize, Quantized AdaMerging (optimized in FP16 and then quantized), and Q-Merge (optimized under a single source schema).
4. **Validation on Unseen Target Schema:** The authors evaluate OmniMerge on Double Quantization, which was excluded from the stochastic training pool during calibration. Its strong performance there provides convincing empirical proof that the learned coefficients have achieved genuine, schema-invariant robustness.
5. **Excellent Writing and Structure:** The manuscript is exceptionally well-written, clearly structured, and easy to follow. Figure 1 provides a very clear visual intuition of the proposed problem and solution.

### Weaknesses
1. **Major Ablation Study Contradiction (SOS Performance Regression):** In Table 2 (Ablation Study), the configuration with only SZNP (`Baseline + TCR + SZNP`) achieves **50.45%** average accuracy, which actually *outperforms* the full proposed `OmniMerge` framework (`Full OmniMerge (Ours)`, which combines SOS and SZNP) at **50.33%** average accuracy. The authors completely gloss over this sub-additive result and fail to discuss why removing Stochastic Operator Sampling (SOS) actually leads to a performance improvement.
2. **Overclaiming Statistical Significance on "Discrete Weight Denoising":** The authors claim that an absolute improvement of **+0.39%** (50.78% vs 50.39%) of the quantized OmniMerge model over its unquantized FP16 counterpart is "definitive empirical proof" that weight-space discretization via rounding acts as a beneficial "discrete weight denoising" filter. However, given that the validation set consists of only 1024 total images, a difference of 0.39% represents just **4 images**, which is highly statistically insignificant and well within the standard error of the binomial distribution ($\approx 1.56\%$).
3. **Artificial Benchmark and Weak Task Experts:** The task-specific experts are extremely weak (e.g., the SVHN expert gets only 28.91% validation accuracy due to training on only 256 images). Merging 4 completely unrelated datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) into a tiny, 5.7M parameter `ViT-Tiny` backbone is highly artificial, leading to severe weight-space interference and a very low performance ceiling (38.67% Task Arithmetic ceiling vs ~80% average expert accuracy).
4. **Impracticality of Per-Channel Double Quantization:** The authors apply Double Quantization per-channel to scale factors. However, for a small backbone like `ViT-Tiny`, per-channel scale storage is already negligible ($< 1\%$ of the weight size). Compressing scales further via 8-bit symmetric quantization saves less than 0.5% of model size while introducing unnecessary compiler and dequantization overhead on hardware, contradicting the focus on practical utility.
5. **Methodological Ambiguity on Autograd Graph:** The paper does not specify whether the gradients are propagated through the scale factor and zero-point computation (which involve non-differentiable min/max operations across tensors) or if scale and zero-point factors are detached from the computation graph during backpropagation. This is critical for the reproducibility and technical clarity of the proposed STE optimization.

---

## 3. Ratings

- **Soundness:** **Fair** (3/4)
  While the mathematical derivation of quantization is precise, the math behind continuous Gaussian zero-point noise introduces a training-test mismatch. Furthermore, the lack of clarity on detaching scale/zero-point factors and the severe overclaims on statistical significance drag down the soundness rating.
- **Presentation:** **Excellent** (4/4)
  The paper is exceptionally clear, beautifully structured, and professional. The visual quality of the plots and mathematical equations is outstanding.
- **Significance:** **Good** (3/4)
  The paper addresses a highly important practical problem in edge deployment. However, its significance is currently limited by the toy evaluation setup (ViT-Tiny on image classification) and the lack of empirical validation on modern Large Language Models (LLMs), where model merging is most actively used.
- **Originality:** **Good** (3/4)
  Combining stochastic operator sampling and scale/zero-point noise to solve cross-schema model merging robustness is a creative and highly novel application of stochastic optimization.

---

## 4. Overall Recommendation

**Rating:** **3: Weak Reject** (A paper with clear merits, but also some weaknesses, which overall outweigh the merits. Papers in this category require revisions before they can be meaningfully built upon by others.)

### Justification of Rating
The paper identifies a crucial practical problem in edge-AI deployment and proposes a creative, training-free solution (OmniMerge) that achieves solid results. However, the paper suffers from three critical flaws that prevent a recommendation to accept in its current state: (1) a major contradiction in the ablation study where the full model underperforms an ablated sub-component, which is completely ignored in the text; (2) overclaiming statistical significance for a 4-image accuracy difference (+0.39%) on a tiny validation split; and (3) a highly artificial, weak vision-classification ensembling setup with significant methodological ambiguities regarding autograd gradient flow. Addressing these flaws is essential to make the work scientifically rigorous and ready for publication.

---

## 5. Critical Flaws (Detailed Critique)

### Flaw 1: Ablation Study Performance Regression (SOS Underperforms SZNP Alone)
In the ablation study (Table 2), the authors report the following cross-schema validation accuracies:
- **Configuration (4) Baseline + TCR + SZNP (No SOS):** **50.45%**
- **Configuration (5) Full OmniMerge (SOS + SZNP + TCR):** **50.33%**

This data shows that the ablated configuration using *only* Scale/Zero-Point Noise Perturbation (SZNP) under a static Symmetric Per-Channel operator actually **outperforms** the full proposed OmniMerge framework (which adds Stochastic Operator Sampling - SOS) by **0.12%**. 

This indicates that adding SOS to the SZNP baseline causes a **performance regression**, suggesting that the two core contributions are sub-additive or that SOS introduces excess gradient variance during test-time adaptation that slightly degrades the final learned coefficients.

Despite this performance drop, the authors write: *"Finally, the full OmniMerge framework (SOS + SZNP + TCR) maintains a highly robust and balanced average accuracy of 50.33% across all five operators, validating our unified formulation."*

This completely ignores the empirical contradiction in Table 2. If SOS actually hurts performance slightly compared to just using SZNP, then the central claim that SOS is a necessary and beneficial component of the co-optimization framework is severely weakened. The authors must provide an honest, detailed discussion explaining why removing SOS improves performance, or show scenarios where SOS is unambiguously beneficial.

### Flaw 2: Overclaiming Statistical Significance on the "Discrete Weight Denoising" Hypothesis
In Section 4.4, the authors propose a novel "discrete weight denoising" hypothesis, claiming that weight discretization via rounding acts as a beneficial non-linear noise filter that filters out small task-arithmetic interference in weight-space:
- They report that quantized OmniMerge achieves **50.78%** accuracy under the Symmetric Per-Channel operator, while its unquantized FP16 counterpart achieves **50.39%** accuracy.
- They state: *"This direct control experiment provides definitive empirical proof that weight-space discretization via rounding acts as a beneficial non-linear noise filter..."*

Given that the evaluation stream consists of exactly $N_{\text{eval}} = 1024$ total images, let's look at the absolute numbers:
- 50.39% of 1024 = **516.0** correct predictions.
- 50.78% of 1024 = **520.0** correct predictions.

The difference is exactly **4 images** out of 1024!

A difference of 4 correct predictions is completely statistically insignificant and well within the standard error of a binomial distribution for $N=1024$ and $p \approx 0.5$ (which is $\sqrt{p(1-p)/N} \approx 1.56\%$). Claiming that a 0.39% accuracy change on a tiny validation sample is "definitive empirical proof" of a fundamental mathematical ensembling phenomenon is a massive overclaim. Without statistical error bars, standard deviations across multiple random seeds, or a formal statistical significance test (e.g., McNemar's test), this claim lacks scientific validity and should be re-framed as an interesting, speculative observation rather than a proven fact.

### Flaw 3: Methodological Ambiguity on Autograd Graph and Toy Benchmark
- **STE Gradient Flow Ambiguity:** 
  The paper relies on the Straight-Through Estimator (STE) to propagate gradients through the rounding operation. However, the paper is completely silent on whether the scale factor $s$ and zero-point $z$ are **detached** from the autograd graph during backpropagation. If $s$ and $z$ are not detached, PyTorch propagates gradients through the non-differentiable min/max operations, which yields extremely sparse and noisy gradients (non-zero only at the single argmax/argmin coordinates). Treating $s$ and $z$ as detached constant parameters is a standard, essential step for PTQ stability. The authors must clarify this crucial implementation detail.
- **Artificial and Weak Benchmark:**
  The task-specific experts are extremely weak (e.g., the SVHN expert gets only 28.91% validation accuracy due to training on only 256 images). Merging 4 completely unrelated datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) into a tiny, 5.7M parameter `ViT-Tiny` backbone is highly artificial, leading to severe weight-space interference and a very low performance ceiling (38.67% Task Arithmetic ceiling vs ~80% average expert accuracy). In practice, model merging is performed on related tasks or similar domains (e.g., multi-lingual translation, instruction following) using much larger backbones.

---

## 6. Questions and Suggestions for the Authors

1. **Clarify the Ablation Study Drop:** Why does Configuration (4) `Baseline + TCR + SZNP` outperform the full model `Full OmniMerge` in Table 2? Please provide an analysis or additional experiments to clarify the role and necessity of SOS when combined with SZNP.
2. **Clarify the Autograd Graph:** Did you detach the scale factors ($s$) and zero-points ($z$) from the PyTorch computation graph during the backward pass? If so, please state this explicitly in Section 3.3. If not, did you observe gradient instability due to backpropagation through the min/max operations?
3. **Re-frame "Discrete Weight Denoising" Claim:** Please tone down the "definitive empirical proof" language in Section 4.4 regarding the +0.39% accuracy increase under quantization. Re-frame it as a speculative observation, or perform a rigorous statistical significance test (e.g., McNemar's test) over multiple random seeds to prove that this difference is not just statistical noise.
4. **Discuss Continuous Zero-Point Noise:** In Equation 6, adding continuous Gaussian noise $\epsilon_z$ to $z_{\text{asym}}$ during calibration creates fractional zero-point offsets. This introduces a training-test mismatch, as the final deployment model must use rounded integer zero-points. Have you considered using integer-valued or discrete uniform noise for the zero-point perturbation instead?
5. **Double Quantization per-channel practicality:** For a tiny model like `ViT-Tiny`, applying Double Quantization (DQ) per-channel to scale factors saves less than 0.5% of the model's weight size. Since the storage overhead of per-channel scales is already negligible ($< 1\%$), please discuss or justify why per-channel Double Quantization is practically useful, or re-target DQ to a block-wise configuration where the scale overhead is actually significant.
6. **Baselines and Epoch Budgets:** Standard AdaMerging and Q-Merge are optimized with $\eta = 10^{-2}$, while OmniMerge uses $\eta = 2 \times 10^{-2}$. Since all methods are restricted to 15 steps, is it possible that standard baselines simply need more steps to converge at their lower learning rate? Please provide a convergence plot or evaluate all methods at full convergence to ensure absolute fairness.
