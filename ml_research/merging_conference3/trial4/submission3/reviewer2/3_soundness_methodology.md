# Soundness and Methodology Evaluation

This document critically evaluates the methodology proposed in the paper, identifying several major technical flaws, questionable assumptions, and limitations in reproducibility.

## 1. Naive and Sub-optimal Quantizer Implementation
The authors report a massive "Double Quantization Noise" reconstruction error in **Table 1**. For example, they show that converting the base model weights from 4-bit NormalFloat (NF4) to INT8 Symmetric Per-Channel results in a relative Frobenius reconstruction error of **17.21%** on `vit_tiny` and **30.40%** on `vit_base`.

*   **Critical Technical Flaw:**
    This extremely high reconstruction error (30.4% error when mapping a 4-bit format to an 8-bit format) is highly anomalous and reveals a major flaw in their quantization pipeline. Since 4-bit NF4 has only 16 discrete values, an 8-bit uniform quantizer (which has 256 bins) should theoretically be able to represent it with near-zero error. 
    The only reason the error is so high is that the authors used an extremely naive uniform symmetric quantizer that computes the step size $s$ solely based on the absolute maximum value ($\max(|W|)$) of the entire tensor or channel, without any outlier clipping or standard post-training calibration (e.g., percentile clipping, grid search for optimal clipping, or AWQ/SmoothQuant-style scaling). A single outlier in the weights "squashes" the linear grid, forcing the dense normal distribution around zero into a handful of bins. 
    By evaluating against an unclipped, naive symmetric quantizer, the authors are presenting a "strawman" failure mode. If they had utilized standard industry-grade PTQ pipelines (such as GPTQ or basic percentile clipping), this double quantization format-shift error would be virtually non-existent in INT8.

## 2. Inherent Contradiction and Overcomplication of SAWS
The formulation of Scale-Adaptive Weight Shifting (SAWS) introduces a layer-wise adaptation norm ratio $\gamma^l$ to boost the adapters, followed by a weight alignment factor $c^l$:
$$c^l = \frac{\langle W^l_{\text{merged}}(\Lambda), \tilde{W}^l_{\text{final, saws}}(\Lambda) \rangle}{\|\tilde{W}^l_{\text{final, saws}}(\Lambda)\|^2_F}$$

*   **Critical Methodology Flaw:**
    1.  **Overcomplicated Mathematical Distraction:** The authors admit in Section 3.3.1 that $c^l \approx 1.0$ (typically $\approx 0.99$ in practice) because the weight tensors are dominated by the base weights. If $c^l \approx 1.0$, the entire derivation of the "elegant, closed-form alignment factor" is mathematically redundant and does nothing to restore the scale.
    2.  **No True Scale Preservation:** The authors prove in their "Representation Scale Preservation Dilemma" that true scale preservation (applying $1/\gamma^l$ to the output) collapses the pre-trained base model representations. Therefore, they do not apply it.
    3.  **Functional Equivalence to Heuristic Tuning:** Since $c^l \approx 1.0$ and $1/\gamma^l$ is not applied, SAWS is functionally just scaling up the adapter task vectors by $\gamma^l$ (which ranges from 10 to 100). Scaling task vectors during merging is a standard hyperparameter (e.g., the scaling factor in Task Arithmetic). The mathematical framework of SAWS is an overcomplicated way of describing a simple heuristic: "multiply the adapter weights by a large scalar." 

## 3. Failure of SAWS in the Only Failure Regime (Per-Tensor)
The "Re-Quantization Silence" is shown to be nearly lossless in standard per-channel configurations (Naive-RQ drops only 0.30% in INT8 and 1.80% in INT4). The catastrophic collapse only occurs in the aggressive INT4 Symmetric Per-Tensor configuration (Table 5), where Naive-RQ drops to 56.75%.

*   **Critical Logical Flaw:**
    In the only configuration where there is actually a catastrophic collapse (INT4 Symmetric Per-Tensor), the proposed data-free method, **SAWS, actually worsens performance**, achieving **56.40%** mean accuracy (compared to Naive-RQ's 56.75%). 
    Under the standard per-channel configurations where SAWS does improve performance (e.g., 67.80% in Table 3 vs. 64.85% Naive-RQ), there is no "catastrophic collapse" to begin with (Naive-RQ only drops 1.80% from the 66.65% FP16 ceiling). This means SAWS is ineffective where it is actually needed, and its improvements in per-channel configurations are simply due to tuning the adapter scaling factors, which would be better done in full precision.

## 4. Underperformance of QA-ACS Against Simple Baselines
QA-ACS is proposed as a novel test-time optimization that propagates gradients through the quantization operator using STE.

*   **Critical Methodological Flaw:**
    In **all four** evaluated configurations, QA-ACS is outperformed by the existing **AdaMerging (PH-Q)** baseline, which simply optimizes the coefficients in FP16 and then performs post-hoc quantization without any STE or quantization-aware optimization:
    - INT8 Symmetric Per-Channel: AdaMerging PH-Q (**70.10%**) > QA-ACS (69.35%)
    - INT4 Symmetric Per-Channel: AdaMerging PH-Q (**68.80%**) > QA-ACS (68.00%)
    - INT4 Asymmetric Per-Channel: AdaMerging PH-Q (**68.25%**) > QA-ACS (64.75%)
    - INT4 Symmetric Per-Tensor: AdaMerging PH-Q (**57.25%**) > QA-ACS (57.00%)
    This indicates that the entire complexity of the STE backward pass, the Adam optimizer tuning, and quantization-aware test-time search is mathematically and empirically unjustified, as it performs strictly worse than the simpler, existing baseline.

## 5. Toy-Scale Experimental Evaluation and Confounding Interference
The empirical results are based entirely on `vit_tiny` (5.7M parameters) evaluated on four toy classification datasets (MNIST, FashionMNIST, CIFAR-10, SVHN).

*   **Critical Evaluation of the Experimental Design:**
    1.  **Extreme Baseline Collapse:** The full-precision merged model (Naive FP16 Merge) already suffers from catastrophic task interference, achieving only **66.65%** mean accuracy compared to the **93.85%** unmerged expert ceiling. On MNIST, the full-precision merged model gets only **45.40%** (a drop of 52.8% in FP16!).
    2.  **Confounding Noise:** Evaluating post-training quantization on top of a model that is already severely collapsed in full precision introduces massive confounding noise. The severe task interference in weight space means the representations are highly unstable, and any slight discretization noise can cause unpredictable shifts. Drawing broad methodological conclusions about post-training quantization behavior from such a degraded toy setup is highly unreliable.
    3.  **Lack of Generalizability:** Modern model merging and PEFT are predominantly applied to LLMs (7B+ parameters) or large diffusion models. A 5.7M parameter ViT on MNIST has fundamentally different representation dynamics, layer shapes, and quantization sensitivity than modern LLMs.

## 6. Reproducibility Concerns
The authors do not provide any link to a code repository, nor do they detail the specific hyperparameters of the training process for the unmerged experts, making exact reproduction highly challenging. The optimization of QA-ACS (using Adam with STE on 16 samples for 40 steps) is highly sensitive to the learning rate, initialization, and specific calibration samples, none of which are characterized or ablated.
