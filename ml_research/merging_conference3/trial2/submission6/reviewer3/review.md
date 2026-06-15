# Peer Review of Conference Submission

## Paper Summary
The paper addresses a key edge-deployment challenge: serving multiple task-specific expert neural networks under strict hardware memory and storage constraints. While weight-space **model merging** (e.g., Task Arithmetic) fuses specialized expert models into a single multi-task network with zero inference-overhead, practical deployment also requires **Post-Training Quantization (PTQ)** to INT8 or INT4 formats. Unfortunately, merging models in uncompressed space followed by post-hoc quantization degrades performance due to rounding noise, whereas quantizing experts first and then merging breaks linear mode connectivity due to grid misalignment.

To solve this, the authors propose **Quantization-Aware Model Merging (Q-Merge)**. Q-Merge is a lightweight, calibration-free test-time adaptation framework that optimizes a compact set of layer-wise task-merging coefficients ($\Lambda$) directly under the non-differentiable quantization operator. By minimizing the Shannon entropy of predictions over a tiny, unlabeled calibration stream (64 images in total), Q-Merge shifts the continuous parameter representations to find optimal coordinate structures that neutralize rounding noise. The authors evaluate two optimization paradigms: zero-order (1+1 ES) and first-order (Adam GD with Straight-Through Estimator). 

Using a ViT-Tiny backbone on a four-task classification benchmark, 8-bit Q-Merge with Adam GD + STE achieves an average accuracy of **74.30%**, outperforming the unquantized uniform FP16 baseline (**71.88%**) and standard unquantized AdaMerging (**73.21%**), while recovering $99.9\%$ of the unquantized Adam-optimized ceiling (**74.38%**). Under aggressive 4-bit quantization, combining standard per-channel (channel-wise) weight quantization with Q-Merge (Adam GD + STE) achieves **63.36%** average accuracy, outperforming the naive post-merge quantization baseline (**56.66%**) by **6.70%** absolute.

---

## Strengths and Weaknesses

### Strengths
1. **Conceptual Simplicity and Elegance:** Rather than over-engineering the solution with complex, multi-stage training schedules, dense reinforcement learning loops, or heavy architectural changes, the authors solve a difficult problem (weight-space model merging under quantization) in the most direct way possible. Optimizing just 56 blending parameters directly under the quantization operator using the Straight-Through Estimator (STE) is a highly elegant, low-overhead blueprint.
2. **Exceptional Scientific and Intellectual Honesty:** The authors avoid standard machine learning hyperbole by carefully isolating potential confounding factors in their experimental matrix. For instance, they implement a fully differentiable, unquantized *AdaMerging (Adam GD)* baseline to isolate the "optimizer confounding factor" (showing that the apparent "boost" over standard AdaMerging in 8-bit is primarily due to the transition to a superior first-order optimizer, which Q-Merge's STE elegantly enables). They also provide a clean quantitative breakdown separating the *Merging Penalty* (task-interference) from the *Quantization Penalty* (rounding noise), which is highly informative.
3. **Rigorous and Transparent Evaluation:** All experiments are conducted across **three independent random seeds** with reported standard deviations. The paper includes extensive ablations, including:
   - Sensitivity analyses to the calibration set size (showing stable performance down to only 8 images per task).
   - Practical edge execution wall-clock latency (showing CPU adaptation of just 2.43 seconds, and GPU adaptation of 80 milliseconds).
   - On-device input stream imbalance/skew analyses with a highly practical systems-level FIFO stratification heuristic.
   - Validation of a fully integer-quantized weight pipeline by post-hoc quantizing classification heads to 8-bit with virtually zero accuracy loss ($<0.01\%$).
4. **Strong Analytical Foundations:** Section 3.4.2 provides a beautifully clear and rigorous mathematical derivation of the dual-path gradient flow through PyTorch Autograd. It shows how gradients flow through both direct coordinates and the dynamic scaling factors, demystifying the mechanics of STE-based optimization in quantized space.

### Weaknesses
1. **Experimental Scale and Parameter Drift Limitations:** The main limitation of the work (explicitly and transparently acknowledged by the authors in Section 5.2) is its scale. The evaluation is restricted to a pre-trained timm ViT-Tiny (5.7M parameters) on four standard classification tasks (MNIST, FashionMNIST, CIFAR-10, SVHN). Furthermore, training experts on extremely compact disjoint sets (512 images per task) represents a **low parameter drift** regime where expert weights remain close to the base checkpoint. In large-scale enterprise settings, experts are fully fine-tuned on millions of tokens/images, resulting in significant parameter drift where weights diverge far from the base, challenging linear mode connectivity more severely.
2. **Lack of Activation Quantization:** The paper focuses on weight-only quantization (W8A16 and W4A16), leaving intermediate activations in full precision (FP16/FP32). While weight-only compression is highly effective for reducing storage and bandwidth bottlenecks, deploying on extreme low-power hardware often requires integers-only execution (W8A8 or W4A4).

---

## Detailed Evaluation Ratings

### Soundness: Excellent
The paper’s methodology is rigorous and mathematically sound. The authors decouple multiple potential confounding factors, evaluate the method across multiple random seeds, and are highly transparent about their assumptions. The mathematical derivation in Section 3.4.2 is exceptionally clear, and the claims are fully backed by rigorous empirical evidence.

### Presentation: Excellent
The presentation quality is outstanding. The paper is logically structured, clearly written, and completely free of unnecessary mathematical obfuscation. It establishes a perfect narrative flow from identifying the core deployment bottleneck (M-then-Q vs. Q-then-M) to detailing the mathematical formulation and presenting extensive quantitative ablations.

### Significance: Good to Excellent
Deploying multi-task merged networks to resource-constrained edge devices has been heavily bottlenecked by quantization noise. By demonstrating that 8-bit quantization is nearly lossless, and that extreme 4-bit model merging is highly viable and practical when combining per-channel quantization with quantization-aware test-time adaptation, this work provides immediate, practical utility to edge systems practitioners and will likely influence future deployment pipelines.

### Originality: Good
While the paper combines existing building blocks—the Straight-Through Estimator (STE), layer-wise task arithmetic, and test-time prediction entropy minimization—its synthesis is highly novel. Q-Merge is the first framework to formulate and solve model-merging adaptive blending directly under the quantization operator, and its conceptual "delta" from unquantized AdaMerging and post-hoc advanced PTQ (like AdaRound) is well-articulated and thoroughly validated.

---

## Overall Recommendation

**Rating: 5 (Accept)**

**Justification:** Q-Merge is a technically solid, exceptionally well-structured, and highly practical paper. By resisting unnecessary complexity and avoiding convoluted architectural modifications, the authors present a remarkably simple and direct solution that effectively resolves the conflict between weight-space model merging and post-training network quantization. The statistical and scientific rigor of the empirical evaluation is exemplary, and the authors' honesty regarding the scale and parameter drift limitations of their work is highly commendable. While scaling to multi-billion parameter LLMs under high parameter drift is an important future research path, this work provides a watertight, reproducible, and highly elegant blueprint that is fully ready for conference publication.

---

## Questions and Constructive Feedback for the Authors
1. **Scaling to LLMs and VLMs:** In Section 5.2, you discuss the theoretical scaling of Q-Merge to large-scale generative architectures (such as 7B parameter LLMs). Have you conducted any preliminary experiments on LLM task vectors (e.g., LLaMA or Mistral fine-tuned experts)? How does the optimization time scale as the number of layers increases?
2. **Fine-Grained Parameterization Trade-offs:** If the merging parameter search space ($\Lambda$) is expanded from layer-wise grouping to head-wise or weight-matrix-wise grouping to handle more discordant tasks, do you observe an increased risk of overfitting or "class collapse" when optimizing on extremely small (e.g., 8 images per task) calibration streams?
3. **Fully Integer Execution (W8A8):** Given that Q-Merge is highly effective at finding coordinate-alignment under weight quantization, do you expect the framework (or its joint prediction entropy loss) to remain stable if activation quantization (e.g., INT8 activations) is also introduced during the test-time adaptation pass?
