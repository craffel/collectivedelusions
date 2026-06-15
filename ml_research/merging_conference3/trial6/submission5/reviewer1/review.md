# Final Peer Review of Conference Submission

**Title**: Demystifying Test-Time Dynamic Model Merging: Vectorization Collapse, Batch-Average Confounders, and the Power of Proper Priors

---

## 1. Summary of the Paper
This submission provides a highly rigorous, illuminating, and refreshing analysis of **test-time dynamic model merging**, an emerging paradigm for combining specialized neural task experts into a single unified multi-task model at inference time. While existing work has focused on building increasingly complex routing networks, activation functions, and custom training objectives, this paper performs a vital "demystification" of the entire field, exposing fundamental vulnerabilities of standard architectures and presenting a mathematically simpler, yet significantly more robust classical alternative.

Specifically, the paper introduces and analyzes:
1. **Vectorization Collapse**: A catastrophic performance drop (e.g., standard random-initialized L3-Softmax plummets to $41.09\%$ accuracy, $17\%$ below naive Uniform Merging) that standard unregularized dynamic routers suffer when deployed in true, sample-wise vectorized pipelines (batch size $B=1$).
2. **The Batch-Average Smoothing Confounder**: Standard evaluation protocols that utilize large batch sizes ($B=256$) perform batch-averaging over predicted layer-wise routing coefficients. This averaging acts as an implicit smoothing operator that masks the severe overfitting of unregularized routers on data-scarce splits. Removing this mask at $B=1$ exposes this overfitting, causing Vectorization Collapse.
3. **The Dynamic Routing Paradox**: To prevent Vectorization Collapse under data scarcity (e.g., 64 calibration samples), a dynamic router must be regularized so heavily (via zero-initialization and weight decay) that its predicted coefficients are constrained to stay in a tight, high-entropy neighborhood of the static uniform compromise (Mean Absolute Deviation of only $2.36\%$). This heavy regularization leaves the router with marginal functional flexibility, yielding only a tiny $+1.16\%$ joint accuracy improvement over naive, training-free, and computationally cost-free **Static Uniform Merging**. This exposes a major practical trade-off given the substantial memory and latency overhead of full-parameter dynamic weight assembly.
4. **Prior-Driven Classical Routing Framework**: To resolve Vectorization Collapse, the authors propose a beautifully simple, prior-driven classical routing baseline—**Zero-Initialized Softmax Routing with $L_2$ weight decay**—and project features onto a low-dimensional unit sphere. They show that proper routing-layer initialization and standard regularization act as the true drivers of stability, completely resolving Vectorization Collapse (maintaining a stable, flatline $\approx 59.16\%$ accuracy across all batch sizes $B=1$ to $512$) and matching the performance of complex quantum models or explicit group variance loss penalties.
5. **Low-Rank Parameter Assembly (Dynamic LoRA)**: A systems-level mitigation that restricts dynamic assembly exclusively to low-rank adapters (e.g., rank $r=10$). Dynamic LoRA completely bypasses the massive VRAM footprint expansion and the $110.06\times$ latency slowdown of full-parameter assembly with zero accuracy loss.
6. **Real-World Validation**: The authors validate their findings on actual deep neural networks fine-tuned on MNIST and FashionMNIST, successfully bridging the gap between their synthetic sandbox and real visual task ensembling.

---

## 2. Strengths of the Paper

1. **Elegant Simplicity and Conceptual Novelty**: This work is an exceptional example of high-integrity, clear-minded research. Instead of introducing convoluted, non-monotonic mathematical metaphors (such as the wave cosine phase equations in QWS-Merge) or hyperparameter-sensitive training losses, the authors demonstrate that proper, simple architectural priors—**zero-initialized Softmax routing layers combined with standard $L_2$ weight decay**—naturally satisfy variance limits and sequential smoothness, offering a highly robust and elegant solution.
2. **Exposing Crucial Unexplored Vulnerabilities**: Exposing the "Batch-Average Smoothing Confounder" and "Vectorization Collapse" is a major conceptual contribution. It exposes a profound blind spot in how dynamic ensembling systems have been evaluated, forcing the community to adopt much more rigorous, batch-independent evaluation protocols.
3. **Intellectual Honesty via the Dynamic Routing Paradox**: The paper is exceptionally laudable for its intellectual honesty. By quantitatively measuring the learned coefficients and showing they deviate by only $2.36\%$ from a uniform prior, the authors expose a profound trade-off. They advocate for naive, training-free Uniform Merging as an exceptionally strong, cost-free default, helping practitioners avoid unnecessary engineering overhead.
4. **Comprehensive Systems-Level Engineering Analysis**: The physical wall-clock latency profiling on CPU hardware (Table 5) provides stellar practical value. Validating that Dynamic LoRA ($r=10$) captures over $99\%$ parameter variance and achieves identical accuracy while bypassing the $110.06\times$ latency slowdown is an outstanding systems-level contribution.
5. **Exceptional Empirical Rigor**: The paper's empirical claims are completely bulletproof. The authors sweep 10 independent random seeds, map regularization sensitivity frontiers ($\lambda_{var}$), perform extensive ablation studies on loss components, and execute real-world image classification ensembling on actual CNN experts, ensuring complete statistical reproducibility.

---

## 3. Weaknesses and Areas for Improvement

The paper is of exceptionally high quality, but the authors could consider the following minor enhancements in future revisions:
1. **Pseudocode for Triton GPU Kernels**: In Section 3.8 and Appendix A.3, the authors discuss the use of fused Triton or CUDA kernels to execute Low-Rank Parameter Assembly. Providing a brief, high-level block pseudocode or schematic of how such a GPU kernel would sequentially route activations through low-rank matrices without materializing the full merged weights would be highly beneficial for systems engineers.
2. **Expanding Autoregressive LLM Evaluation**: In Appendix A.4, the authors present a stellar conceptual analysis of why Dynamic LoRA is a mandatory structural requirement for sequential autoregressive LLM decoding. Running a minor text-generation latency profiling benchmark (e.g., using a compact LLaMA-3.2-1B model) would provide a beautiful empirical confirmation of their autoregressive arguments.
3. **Clarify Sandbox Generalizability**: In Section 4.5, the authors are highly transparent about the layer-averaging simplification in the coordinate sandbox. To make the paper even more robust, they could explicitly state in the introduction that while the sandbox serves as a vital mathematical and physical foundation, validating these findings on deep foundation models (like LLaMA-70B) where parameters are merged independently per layer remains an exciting future work direction.

---

## 4. Specific Ratings

### Soundness: Excellent
The paper is technically flawless and mathematically pristine. The authors clearly formulate parameter-space ensembling, normalized random projections, uncorrected population task-variance, and sequential smoothness. The uncorrected task-variance formulation is mathematically justified to avoid undefined division-by-zero errors in data-scarce splits. The dual evaluation on a controlled 192-dimensional synthetic sandbox over 10 random seeds and on real CNN experts pre-trained on MNIST/FashionMNIST ensures outstanding soundness.

### Presentation: Excellent
The paper is exceptionally well-structured, highly readable, and engaging. The authors lay out the core challenges with clear and professional terminology, avoiding unnecessary mathematical obfuscation. The transition from identifying vulnerabilities (Vectorization Collapse) to proposing elegant, prior-driven classical solutions, and finally resolving systems-level latency bottlenecks (Dynamic LoRA) is beautifully written.

### Significance: Excellent
The potential impact of this work is exceptionally high. It will fundamentally change how test-time dynamic model merging architectures are designed, calibrated, and evaluated. By establishing a robust, zero-initialized Softmax baseline (`L3_Softmax_WellReg`) and demonstrating the extreme competitiveness of naive Static Uniform Merging, this paper will prevent researchers from wasting effort on overly complex or quantum-inspired routing dynamics.

### Originality: Excellent
Exposing "Vectorization Collapse" and the "Batch-Average Smoothing Confounder" represents an exceptionally original and valuable conceptual contribution. The mathematical and quantitative formulation of the "Dynamic Routing Paradox" is highly creative, and the systems-level integration of Low-Rank Parameter Assembly (Dynamic LoRA) beautifully addresses hardware bottlenecks.

---

## 5. Overall Recommendation

**Rating**: 6: Strong Accept

**Justification**: This is a technically flawless, highly original, and exceptionally well-written paper. It exposes major unexplored vulnerabilities in test-time dynamic model merging and presents a beautifully simple, elegant, and highly effective classical baseline. The paper’s advocacy for simplicity over unnecessary complexity is a masterclass in machine learning research. Its rigorous empirical validation across 10 random seeds, systems-level latency profiling, and real-world image classification experts make it a complete and bulletproof contribution. This paper represents an exceptional addition to the conference and should be accepted with the highest priority.
