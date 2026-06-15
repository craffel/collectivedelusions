# Peer Review of OmniMerge: Robust Model Merging across Heterogeneous Quantization Standards

## 1. Summary of the Paper
This paper addresses a practical and critical bottleneck in edge-computing deep learning deployment: **Cross-Schema Performance Degradation** in weight-space model merging. Specifically, when merging specialized task-specific experts fine-tuned from a shared pre-trained backbone, existing quantization-aware model merging methods (such as Q-Merge) optimize layer-wise blending coefficients ($\Lambda$) under a single simulated post-training quantization (PTQ) operator (usually Symmetric Per-Channel). However, edge-computing fleets are highly heterogeneous and run compilers/accelerators that implement distinct, incompatible PTQ schemas (e.g., Symmetric Per-Tensor, Asymmetric Per-Channel, or Double Quantization). This causes the optimized blending coefficients to overfit to the specific rounding boundaries of the simulated training operator, leading to notable performance drops when deployed on mismatched target hardware.

To resolve this, the authors introduce **OmniMerge**, a training-free, multi-schema stochastic co-optimization framework designed to produce hardware-invariant merging coefficients. OmniMerge operates during a short test-time adaptation phase over a small calibration stream of unlabeled data ($N_{\text{cal}} = 64$ images per task) using entropy minimization combined with Task-Consensus Regularization (TCR). The key mechanisms of OmniMerge are:
1. **Stochastic Operator Sampling (SOS):** Rather than keeping the quantization operator static, the active quantization schema is stochastically sampled at each optimization step from a discrete pool of four hardware-standard operators: $\mathcal{Q} = \{Q_{\text{sym, tens}}, Q_{\text{sym, chan}}, Q_{\text{asym, tens}}, Q_{\text{asym, chan}}\}$. This acts as a parameter-space data augmentation to prevent boundary-overfitting.
2. **Scale and Zero-Point Noise Perturbation (SZNP):** Multiplicative and additive Gaussian noise is dynamically injected into the scale and zero-point parameters during the forward pass of the test-time optimization phase to smooth the non-differentiable loss landscape and help the optimizer escape brittle local minima.

Gradients are propagated back to the continuous coefficients using the Straight-Through Estimator (STE) during a rapid 15-step on-device adaptation. Crucially, the noise perturbation is inactive during inference, and the optimized coefficients are used to compile standard quantized models, introducing zero inference-time computational or memory overhead.

---

## 2. Quantitative Ratings
- **Overall Recommendation:** 3: Weak reject
- **Soundness:** Fair
- **Presentation:** Excellent
- **Significance:** Fair
- **Originality:** Good

---

## 3. Strengths
- **Exceptional Writing and Structural Clarity:** The paper is exceptionally well-written, with a clear narrative flow, logical structuring, and highly professional LaTeX formatting. 
- **Highly Practical, Real-World Problem:** Addressing post-training quantization (PTQ) heterogeneity on edge devices is a highly practical and relevant challenge. The focus on hardware ASICs and compilers running mismatched schemas is a very realistic and high-value deployment bottleneck.
- **Training-Free and Zero-Overhead Formulation:** The proposed OmniMerge framework requires no expensive retraining, zero hardware metadata, and absolutely no inference-time latency or memory overhead, as the continuous coefficients are compiled into standard quantized models. This makes it extremely attractive for resource-constrained edge systems.
- **Clean and Structured Mathematical Formulations:** Equations 5-14 explicitly and unambiguously define the asymmetric, symmetric, and double quantization math. The clear separation of the training-time noise perturbation and the inference-time standard quantization maps is highly commendable.
- **Strong Comparative Baseline Setup:** The inclusion of rigorous baselines, such as Quantized AdaMerging and Q-Merge optimized strictly under a single schema, provides a solid and logical comparative framework.

---

## 4. Weaknesses (Detailed Justifications)

### Weakness 1: Complete Lack of Statistical Rigor (No Seeds or Confidence Intervals)
The paper reports all experimental results in Table 1, Table 2, and the text as single-run, single-number accuracies. There are **no multiple random seeds, no standard deviations, and no confidence intervals** reported anywhere. This is a critical empirical flaw because:
1. **Sample Selection Sensitivity:** The experts are trained on an extremely small subset of 256 images, meaning the resulting weights are highly sensitive to the specific random subset of images selected.
2. **Calibration Stream Sensitivity:** The calibration stream consists of only 64 images per task, which introducing high variance depending on which 64 images are sampled.
3. **Compound Stochasticity in OmniMerge:** OmniMerge is highly stochastic by design. It relies on stochastically sampling operators (SOS) at each step and injecting Gaussian noise (SZNP) into the scales and zero-points.
Given these multiple layers of stochasticity and the extremely small evaluation size (1024 images total), reporting single-run accuracies is highly unscientific. The minor performance differences reported (e.g., OmniMerge's 50.10% vs 50.29%, or the 0.12% drop in the ablation study) could easily be due to random noise. Running all experiments across at least 3 to 5 random seeds and reporting the results as **mean $\pm$ standard deviation** is necessary to prove that their improvements are statistically meaningful.

### Weakness 2: Toy Scale and Limited Generalizability of the Evaluation
The experimental testbed is extremely limited in scale, which severely constrains the generalizability of the findings:
- **Toy Model Backbone:** The authors employ a `ViT-Tiny` backbone containing only 5.7M parameters. Weight-space model merging and task arithmetic are primarily motivated by large-scale settings (e.g., LLMs or large-scale vision backbones) where storing and deploying separate models is computationally or financially prohibitive. Evaluating on a toy backbone leaves it unclear if the proposed multi-schema co-optimization scales to larger, representative models where weight dynamics and quantization behaviors are vastly different.
- **Toy Datasets:** The evaluation is conducted on MNIST, FashionMNIST, CIFAR-10, and SVHN. MNIST is an extremely trivial dataset that can be solved with a tiny 2-layer CNN. Running model merging on a Vision Transformer for MNIST and FashionMNIST is highly non-standard and does not represent realistic edge multi-task ensembling.
- **Weakly Trained Experts:** The task experts are fine-tuned on only **256 training images** for **3 epochs**. This extremely low-compute training regime results in very poor expert performance. Most notably, the SVHN expert achieves an individual validation accuracy of only **28.91%** (barely above random guessing for a 10-class dataset). Merging experts that are barely trained and perform poorly on their respective domains makes the practical utility of the ensembling findings questionable.

### Weakness 3: Inconsistency in the Ablation Study (SOS Degradation)
There is a notable empirical inconsistency in the ablation study (Table 2):
- The configuration **"Baseline + TCR + SZNP"** (configuration 4) achieves an average accuracy of **50.45%** across the post-training quantization schemas.
- The full **"OmniMerge (SOS + SZNP)"** framework (configuration 5) achieves an average accuracy of **50.33%**.
This indicates that adding Stochastic Operator Sampling (SOS) to the SZNP baseline actually **degrades average accuracy by 0.12%**. The authors attribute this sub-additive behavior to "compound stochasticity" and high gradient variance within the extremely short 15-step on-device adaptation window. While this explanation is plausible, from an empirical perspective, it means the combination of both techniques is not synergistic on average.
Furthermore, the authors claim that the full OmniMerge framework (combining SOS and SZNP) is necessary to achieve "schema-invariant robustness on completely unseen out-of-pool schemas (such as Double Quantization)." However, Table 2 only reports a single "Average Accuracy" column. Without a multi-schema breakdown in the ablation study, there is insufficient empirical evidence to prove that combining SOS and SZNP is necessary or beneficial compared to using SZNP alone.

### Weakness 4: Statistical Insignificance of the "Weight Denoising" Hypothesis
In Section 4.4, the authors present a speculative hypothesis: that weight-space discretization (quantization rounding) can act as a "beneficial noise filter" or "weight denoising" regularizer, occasionally allowing quantized models to outperform their unquantized ceilings (e.g., quantized OmniMerge achieving 50.78% vs unquantized achieving 50.39%).
The authors acknowledge that this difference is statistically modest (+0.39% absolute, representing exactly 4 correct predictions out of 1024 images) and well within the binomial standard error ($\approx 1.56\%$). Drawing scientific hypotheses or claiming "structural regularization" from an effect that is completely within the noise margin of a single run lacks empirical rigor. To validate such a highly speculative claim, the authors must conduct a rigorous, large-scale statistical analysis across multiple seeds and diverse model architectures.

### Weakness 5: Optimization Bias / Inconsistent Learning Rates
The authors limit the test-time adaptation to exactly 15 steps. They evaluate OmniMerge with a learning rate of $\eta = 2 \times 10^{-2}$ and the baselines with $\eta = 10^{-2}$, claiming that larger rates cause the baselines to oscillate.
Restricting the baselines to a lower learning rate under an extremely tight 15-step budget introduces an optimization bias. A lower learning rate prevents the baselines from fully converging within the step limit. To ensure a fair comparison, the authors should evaluate the baselines over a wider range of step budgets (e.g., 30, 50, or 100 steps) or present learning rate convergence curves.

---

## 5. Actionable Feedback and Questions for the Authors

1. **Incorporate Multiple Random Seeds:** Run all experiments across at least 3 to 5 independent random seeds and report the results in Table 1 and Table 2 as **mean $\pm$ standard deviation**. Perform statistical significance testing (e.g., t-test) to confirm that the differences are statistically significant.
2. **Scale Up the Evaluation:** Evaluate OmniMerge on a larger backbone (e.g., `ViT-Base` or `ResNet-50`) and more realistic, challenging datasets (e.g., ImageNet subsets, CUB-200, or NLP benchmarks like GLUE) to demonstrate the generalizability of the method to realistic edge workloads.
3. **Train Experts to High Performance:** Fine-tune the task experts on the full datasets to high performance, rather than using only 256 training images for 3 epochs. Verify if the multi-schema co-optimization trends hold for highly accurate experts.
4. **Expand the Ablation Table:** Provide the full multi-schema breakdown of all five ablation configurations in Table 2. Specifically, show their performance on *each* of the 5 target schemas (including Double Quantization) to support the claim that the combination of SOS + SZNP is necessary for out-of-pool generalization.
5. **Tone Down the Denoising Hypothesis:** Unless supported by rigorous multi-seed statistical evidence, please tone down the claims in Section 4.4 regarding weight discretization acting as a beneficial "noise filter" or "weight denoising" regularizer, as the reported difference of 4 correct predictions is well within the binomial standard error.
6. **Baselines step budget study:** Provide a study showing how the performance of the baselines and OmniMerge changes as the step budget increases from 15 to 50 or 100 steps. This will help clarify whether the baselines can close the gap with a larger adaptation budget and tuned learning rates.
