# Final Peer Review

## 1. Summary of the Submission
This paper introduces **Quantization-Aware Model Merging (Q-Merge)**, a zero-shot, test-time adaptation framework designed to deploy merged multi-task networks under strict edge-hardware constraints. While weight-space model merging is highly efficient for multi-task fusion, subsequent post-training quantization (PTQ) to low-bit formats (such as INT8 or INT4) degrades accuracy due to rounding noise. Conversely, pre-quantizing experts before merging breaks linear mode connectivity. 

To resolve this bottleneck, Q-Merge optimizes layer-wise merging coefficients directly under the non-differentiable quantization operator on a small, unlabeled calibration stream (64 images total) via prediction entropy minimization. The authors evaluate two optimization paradigms: zero-order 1+1 Evolution Strategy (1+1 ES) and first-order Adam gradient descent with a Straight-Through Estimator (Adam GD with STE). Evaluated on a pre-trained ViT-Tiny backbone across a four-task classification benchmark (MNIST, FashionMNIST, CIFAR-10, SVHN), 8-bit Q-Merge with STE achieves an average multi-task accuracy of **74.30%**, outperforming unquantized uniform merging and recovering 99.9% of the unquantized Adam ceiling. Under aggressive 4-bit per-channel quantization, Q-Merge with STE successfully prevents model collapse and achieves **63.36%** accuracy, outperforming the naive post-merge baseline by **56.66%** by **6.70%** absolute.

---

## 2. Strengths
* **High Practical Significance:** The paper addresses an extremely important, real-world bottleneck in edge deployment. By enabling high-fidelity 8-bit model merging and highly viable 4-bit model merging, this framework directly slashes the off-chip memory transfer bandwidth and storage footprint of serving multi-task models.
* **Deep Scientific and Baseline Rigor:** The experimental design is exceptionally thorough. The authors meticulously isolate the "optimizer confounding factor" by introducing a differentiable, unquantized FP16 Adam-optimized AdaMerging baseline. They also compare against advanced PTQ algorithms like AdaRound and demonstrate a sequential joint pipeline (Q-Merge + AdaRound) that achieves state-of-the-art results (64.46% average accuracy under 4-bit).
* **High Statistical Rigor:** All main results report means and standard deviations across **three independent random seeds (42, 100, 2026)**, ensuring that the reported improvements are statistically significant and robust.
* **Comprehensive Systems Validation:** To demonstrate the method's edge-deployment viability, the authors conduct extensive, high-signal systems evaluations:
  1. *Fully Quantized Weights:* Show that post-hoc 8-bit quantization of the task classification heads yields virtually zero degradation ($0.00\%$ drop under 8-bit, $0.01\%$ drop under 4-bit).
  2. *Scale Discretization:* Analyze scale factor quantization under integer-only MCU constraints (Appendix C), showing high robustness.
  3. *Online Balancing:* Propose and validate Confidence-Based FIFO Stratification to manage non-stationary, unbalanced test-time calibration streams (Appendix B & Section 4.10).
  4. *Latency and Data Efficiency:* Show that adaptation completes in under 2.5s on a CPU and 80ms on a GPU, using only 8 images per task.
* **Exceptional Writing and Presentation:** The paper is extremely well-written, structured logically, and mathematically precise (specifically the explicit formulation of dual-path gradient flow through per-channel scales).

---

## 3. Weaknesses and Areas for Improvement
* **Toy-Scale Model and Datasets (Scale of Evaluation):** The primary limitation is the scale of the experimental evaluation. The authors utilize a toy-scale **ViT-Tiny** backbone (5.7M parameters) and simple classification datasets (MNIST, FashionMNIST, CIFAR-10, SVHN). Modern weight-space model merging and post-training quantization are heavily utilized in Large Language Models (LLMs) and Vision-Language Models (VLMs) containing billions of parameters. While the authors discuss scaling, activation caching, and forward-mode AD in detail, demonstrating the method on large-scale models remains an open future direction.
* **Low Parameter-Drift Regime:** Because downstream experts are fine-tuned on extremely scarce datasets (512 images per task), they remain structurally very close to the pre-trained base model. This low-drift regime represents a simplified setting where linear mode connectivity is highly stable. In real-world enterprise applications, expert models are fully converged on massive datasets, causing significant weight divergence. It is unclear how well Q-Merge's linear blending performs in high parameter-drift regimes.
* **Non-Converged Expert Baselines:** Due to the scarce expert training data, the SVHN unmerged expert is extremely weak (41.34% average accuracy). In real-world multi-task deployments, practitioners typically merge highly accurate, fully converged experts.

---

## 4. Minor Suggestions & Questions for the Authors
1. **Did you observe any overfitting to the calibration stream?** While Table 5 shows excellent stability across calibration sizes (8 to 64 images), was there any sign of overfitting if the adaptation budget (iterations) was increased significantly?
2. **How does the computation cost of backpropagation at test-time compare to the forward-only 1+1 ES on low-power edge platforms?** The paper notes that forward-mode AD or gradient checkpointing can reduce activation memory, but a brief discussion of the actual runtime memory footprint of Adam GD with STE versus 1+1 ES in Appendix E would be a valuable addition for systems engineers.
3. **Typo / Formatting:** In Equation (4), ensure that the clipping boundaries are mathematically aligned with the standard signed integer ranges (e.g., $[-2^{b-1}, 2^{b-1}-1]$). The text is correct, but the brackets are slightly truncated in some renderings.

---

## 5. Overall Ratings & Recommendation

* **Soundness:** **Excellent**
  * The mathematical formulations are flawless, the baselines are rigorously controlled, and the statistical evaluation is exemplary.
* **Presentation:** **Excellent**
  * The paper is clearly written, beautifully structured, and positions itself outstandingly well relative to prior work.
* **Significance:** **Good**
  * The work addresses a highly important practical bottleneck in edge AI, though its overall significance is slightly constrained by the toy-scale evaluation (ViT-Tiny and classification benchmarks).
* **Originality:** **Good**
  * Optimizing continuous blending coefficients directly under the quantization operator using the Straight-Through Estimator and demonstrating dual-path gradient flow represents a highly novel and creative combination of established techniques.

* **Overall Recommendation:** **5: Accept**
  * This is a technically solid, exceptionally complete, and highly practical paper. The authors have done an outstanding job of validating their method under realistic systems-level constraints (e.g., fixed-point scales, stream unbalancing, fully integer weight pipelines) with meticulous control baselines. While scaling to multi-billion parameter LLMs remains an open future direction, the empirical and theoretical foundations laid in this work are extremely strong, making it highly ready for publication.
