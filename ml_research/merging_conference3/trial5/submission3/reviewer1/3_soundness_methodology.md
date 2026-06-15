# 3. Soundness and Methodology

## Clarity of Description and Mathematical Formulation
The methodology in Section 3 is described with exceptional clarity and mathematical transparency. The paper avoids unnecessary mathematical obfuscation, formulating **Robust Linear Routing (RLR)** with basic, elegant primitives:
* **Gating Layer:** A standard classical linear projection mapping a globally average-pooled transformer representation $x \in \mathbb{R}^d$ to raw logits $z \in \mathbb{R}^N$:
  $$z = Wx + b$$
  This uses a mere 772 trainable parameters ($4 \times 192 + 4$) for a 4-task vision benchmark.
* **Continuous Stabilizers:** Rather than using complex multi-stage operators, RLR stabilizes dynamic parameter blending with:
  1. Standard $L_2$ weight regularization (Frobenius norm penalty on $W$).
  2. Softmax Temperature scaling ($T \ge 1$) to soften blending outputs:
     $$a_k = \frac{\exp(z_k / T)}{\sum_j \exp(z_j / T)}$$
* **Calibration Loss:** A straightforward, unweighted uniform cross-entropy calibration loss on a tiny 64-sample dataset.

## Appropriateness of Methods and Potential Technical Flaws
The methods chosen are highly appropriate. Using standard, well-understood machine learning techniques to stabilize a lightweight gating layer is a major strength.
* **Potential Flaws Evaluated:**
  * **Representation Source Layer:** While routing from deeper blocks yields higher peak performance (as shown in Table 4), representations in deep layers are highly task-warped, which can trigger high-variance routing logits and cause overconfident softmax gating in the absence of regularization. The authors acknowledge this mechanism in Section 3.2. 
  * **Dynamic vs. Static Trade-offs:** The authors are highly transparent about the inherent limitation of dynamic routing in mixed heterogeneous test streams (heterogeneity collapse at large batch sizes like $B=256$, where performance drops to $75.03\%$). They honestly compare this to static methods (like OFS-Tune, which maintains a stable $86.23\%$) and provide practical design guidelines (e.g., utilizing a pre-sorting layer to group queries into homogeneous mini-batches). This high level of honesty and scientific rigor is commendable.
  * **Peak Performance in Homogeneous Settings:** The authors are extremely honest about the fact that under homogeneous settings, the regularized (RLR) and unregularized classical linear routers are statistically indistinguishable in mean performance across multiple seeds (Section 4.3). Rather than overselling RLR as a tool to boost peak in-distribution performance, they clarify that RLR's value lies in acting as a robust shield that trades off a negligible margin of peak homogeneous performance to secure superior resilience to out-of-distribution shifts and heterogeneous test streams.

## Reproducibility
Reproducibility is excellent:
* The training process is parsimonious (100 steps of Adam with a learning rate of 0.01).
* The calibration dataset is extremely small (16 samples per task, 64 total images under seed 42) and deterministic.
* The authors conduct a 5-seed random calibration draw study to ensure that their findings are not seed-dependent, demonstrating high statistical rigor.
* The paper utilizes a standard Vision Transformer (`vit_tiny_patch16_224`), and the authors locally re-implemented the QWS-Merge baseline under identical conditions on the exact same expert weights to ensure a fair, rigorous, and controlled comparison.
* The entire RLR framework is exceptionally simple and can be implemented in under 100 lines of standard PyTorch code.
