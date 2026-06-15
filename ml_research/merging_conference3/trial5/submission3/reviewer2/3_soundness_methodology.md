# Soundness and Methodology Evaluation

## Clarity of the Description
The methodology is described with commendable clarity and mathematical precision. Section 3 outlines:
1. **Dynamic Parameter Blending:** The mathematical framework of mapping representations $x \in \mathbb{R}^d$ to raw logits $z \in \mathbb{R}^N$ via linear projection, converting them to task-blending coefficients $a_k$ via temperature-scaled softmax, and linearly combining task-specific weights $\theta_{k,l}$ to construct dynamically merged weights $\theta_l(x)$.
2. **The Mechanism of Gating Collapse:** The hypothesis that "deep task-warped representation shift" produces extreme logits $z_k$ that saturate the softmax output and lead to deterministic single-expert routing, causing representational collapse under OOD shifts.
3. **Proposed Regularization (RLR):** The application of standard $L_2$ weight decay and Softmax Temperature scaling.
4. **Calibration Optimization:** The unweighted cross-entropy loss formulation and the training hyperparameters (100 steps of Adam with $\eta = 0.01$ on 64 calibration samples).

The formulation is clean, simple, and easy to reproduce.

## Appropriateness of Methods
The methods employed are highly appropriate for the problem. Applying classical regularizations to a lightweight linear gating layer is a direct, elegant application of Occam's razor to deconstruct over-engineered dynamic parameter fusion architectures. Using functional calling APIs (`torch.func.functional_call`) in PyTorch is standard and efficient.

## Internal Soundness and Logical Inconsistencies (Critical Scholar Critique)
While the paper's experiments are rigorous, a closer analysis reveals a subtle logical tension and internal inconsistency regarding the core hypothesis of "deep task-warped representation shift" triggering gating collapse:

1. **The Core Hypothesis vs. Empirical Reality of Unregularized Routing:**
   - In Section 3.2, the authors argue that when routing representations are extracted from deep layers (e.g., Block 11), they are "highly specialized" and warped, which produces high-variance inputs to the router, extreme logits, and catastrophic collapse for the unregularized router.
   - However, in Table 1 (and confirmed in Table 4), the **unregularized classical Linear Router** is evaluated using **Late (Block 11)** representations. Under Seed 42, it achieves a stellar **$94.87\%$** accuracy on SVHN and a **$95.46\%$** Joint Mean accuracy, outperforming RLR ($94.36\%$ SVHN, $94.68\%$ Joint Mean).
   - In Table 4 (Ablation Study), routing from Block 11 (Late) for the classical router yields **$95.41\%$** Joint Mean and **$94.83\%$** SVHN accuracy, with **no collapse whatsoever**.
   - If deep representation routing inherently causes catastrophic collapse due to "deep task-warped representation shift," why does the unregularized classical router perform so exceptionally well using Block 11 representations in Table 1 and Table 4?

2. **The True Drivers of Gating Collapse:**
   - This empirical finding strongly suggests that the "deep representation source" itself is **not** the primary driver of gating collapse.
   - Instead, as Table 2 (Diagnostic Configuration) indicates, the collapse reported in Vance et al. (2025) was primarily triggered by a combination of **excessive learning rates ($\eta > 0.1$)** and **massive over-optimization ($>1000$ steps)** on a tiny calibration set.
   - When training is parsimonious (100 steps, $\eta = 0.01$), even the unregularized classical router converges stably to a near-optimal solution regardless of the layer source (Early: $90.65\%$, Middle: $94.25\%$, Late: $95.41\%$).
   - The authors should refine their text to clarify that deep task-warped representations are only problematic when combined with reckless optimization (high learning rate and over-training), rather than being a standalone structural cause of collapse.

3. **Inconsistent Defaults Between Sections:**
   - In Section 3.1, the authors state: "In our main model configuration, we default to extracting representations from the deep layers (globally average-pooled representation of Block 11)..." (which is used in Table 1).
   - But in Table 2, "Our Stable Configuration" is defined as using "First Patch Embedding (Early)" as the routing input source. This configuration is used for the multi-seed sweep in Section 4.3 (which reports $91.53\% \pm 0.41\%$ Joint Mean).
   - This shifting default (Late Block 11 for Table 1 and Table 4, but Early Patch Embed for Table 2 and Section 4.3) is confusing and should be homogenized or more transparently documented.

## Reproducibility
The reproducibility of the submission is **excellent**. The authors provide precise hyperparameters (Adam, learning rate 0.01, 100 steps, $T=2.0$, $\alpha=0.005$, calibration seed 42 with 16 samples per task, ViT-Tiny backbone). The simplicity of the gating layer (a single linear layer of 768 weights and 4 biases) and the detailed diagnostic analysis in Table 2 make it straightforward for any researcher to reproduce the findings.
