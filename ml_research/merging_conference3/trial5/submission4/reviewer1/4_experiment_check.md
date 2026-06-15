# Intermediate Review Evaluation: Experimental Rigor and Validation

## 1. Experimental Setup and Backbone Choice
The paper employs a highly rigorous, controlled, and appropriate experimental setup:
* **Capacity-Constrained Backbone:** The choice of a compact Vision Transformer (`vit_tiny_patch16_224`, 5.7M parameters, $L=14, D=192$) is a highly effective, deliberate scientific decision. In larger models, representation capacity is abundant, which can mask the destructive interference of model merging. A smaller, capacity-constrained backbone is highly sensitive to parameter-space conflicts, making it a perfect, high-sensitivity playground to isolate and deconstruct model-merging routing mechanisms.
* **Diversified Dataset Suite:** The evaluation spans MNIST (grayscale digits), FashionMNIST (grayscale apparel), CIFAR-10 (natural object classification), and SVHN (street numbers), which represent high-conflict task gradients in weight space. 
* **Statistical Significance:** All trainable routing experiments are reported as Mean $\pm$ Standard Deviation evaluated over 3 random calibration-sampling seeds. This provides crucial visibility into the stability and robustness of each method.

---

## 2. Baseline Comparisons
The paper does not cut corners on baseline comparisons. It includes:
* **Static Merging Baselines:** Uniform Merge (Task Arithmetic) and OFS-Tune (supervised static parameter-space tuning).
* **Online Adaptor Baselines:** AdaMerging (unsupervised Test-Time Adaptation).
* **Classical Routing Baselines:** The standard, unregularized Linear Router gating model.
* **State-of-the-Art Baselines:** QWS-Merge (SOTA Quantum Wavefunction Superposition Merging).
* **Targeted Surgical Baselines (Ours):** BL-Router (with and without Reg), GLS-Router (with and without Reg), and BSigmoid-Router (with and without Reg).

This is an exceptionally complete and honest suite of baselines that covers static, online TTA, classical, and SOTA dynamic routing paradigms.

---

## 3. Support for Claims and Major Findings
The empirical results provide outstanding, unambiguous support for the paper's core claims:
* **Support for Claim 1 (Classical routing can succeed when regularized):** In Table 1, the unregularized Linear Router suffers from a high-conflict SVHN collapse ($74.00 \pm 16.14\%$). Applying standard L2 regularization (weight decay $\gamma = 1 \times 10^{-4}$) to the classical routing head completely resolves the collapse, achieving **$91.73 \pm 3.71\%$** SVHN accuracy. This is a massive **+12.00%** improvement over QWS-Merge ($79.73 \pm 0.75\%$) and directly validates the hypothesis that classical routing is highly competitive when properly regularized.
* **Support for Claim 2 (Deconstructing the Softmax zero-sum competitive bottleneck):** Table 1 shows that unregularized BL-Router catastrophically collapses on SVHN ($31.73 \pm 16.03\%$), and regularized BL-Router remains severely degraded ($43.20 \pm 8.02\%$) due to the Softmax-induced under-scaling bottleneck (sum of coefficients capped at 0.3). By replacing Softmax with independent Sigmoids, the **BSigmoid-Router (Ours - Sigmoidal)** completely resolves this, achieving **$75.33 \pm 3.21\%$** SVHN accuracy and an outstanding **$83.73 \pm 1.93\%$** joint homogeneous mean, matching QWS-Merge while maintaining a parameter footprint of only 772 parameters.
* **Support for Claim 3 (QWS-Merge as a structural regularizer):** Table 1 reveals that unregularized GLS-Router exhibits extreme sensitivity across seeds on SVHN (standard deviation of **$24.30\%$**) and collapses on FashionMNIST ($64.80 \pm 3.53\%$). In contrast, QWS-Merge, which also optimizes layer-specific dynamic coefficients, achieves an extremely tight and stable standard deviation of only **0.53%** ($83.97 \pm 0.53\%$) on the joint mean. This provides elegant empirical proof that the complex wave projection equations in QWS-Merge function as an effective structural regularizer that stabilizes optimization in low-data calibration regimes.
* **Support for Claim 4 (Batch-Averaging Bottleneck):** Table 2 shows that as the streaming batch size scales to $B=256$, the performance of all dynamic routing methods converges closely to the static Uniform Merge ($85.10\%$). Appendix Figure 3 and Appendix Section B.1 mathematically and visually illustrate the flattening of routing coefficients due to the Central Limit Theorem, fully validating the batch-averaging bottleneck.

---

## 4. Methodological Transparency and Honest Critique
The authors do not shy away from reporting critical limitations of dynamic routing, which enhances the credibility of the work:
* **Generalist-Specialist Tradeoff:** Section 4.3 contains an incredibly honest and refreshing discussion on the "practical utility paradox" of dynamic weight routing. The authors explicitly point out that while regularized Linear Router achieves a peak SVHN accuracy of 91.73%, it does so by sacrificing performance on other tasks (MNIST, FashionMNIST, CIFAR-10 drops by up to 10%), leading to a lower overall joint mean than a simple static Uniform Merge (82.80% vs 85.10%). This level of scientific honesty is rarely seen in the literature and represents a massive contribution to the community's realistic understanding of model merging.
* **Inference Latency vs. PyTorch Memory Overhead:** The latency profiling in Appendix B.2 is outstandingly thorough. It shows that BSigmoid-Router is 25x faster than AdaMerging (18.5 ms vs 482 ms), and honestly reveals that 80% of BSigmoid-Router's latency is consumed by high-level PyTorch tensor management (cloning/duplicating parameter buffers) rather than the routing head itself.
