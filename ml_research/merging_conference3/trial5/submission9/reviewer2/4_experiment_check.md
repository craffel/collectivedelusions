# Experimental Evaluation Check of GSC-Merge

## Critique of Experimental Setup
- **Backbone Model (ViT-Tiny):** The choice of `vit_tiny_patch16_224` (approximately 5.7M parameters) is a relatively lightweight backbone. While standard for proof-of-concept academic research, modern model merging evaluations frequently employ larger models such as `vit_base_patch16_224` (86M parameters) or Large Language Models (e.g., LLaMA-7B). Testing on a larger backbone would strengthen the generalizability claims.
- **Task Suite (MNIST, FashionMNIST, CIFAR-10, SVHN):** These tasks represent extremely disparate and conflicting visual domains. While these small-scale datasets are useful for simulating a highly adversarial "parameter interference" scenario, natural-image merging benchmarks (such as CIFAR-100, STL-10, or DomainNet) are more representative of real-world multi-task serving. However, as a "stress test" for model merging, this conflicting task suite is highly appropriate.
- **Tuning and Calibration:** The calibration dataset is tiny (16 samples per task, 64 total), which is a standard "few-shot" regime. The authors' execution over 5 independent random validation calibration splits (with mean and standard deviation reported) is highly commendable and ensures statistical rigor.

## Quality and Fair-mindedness of Baselines
The baselines chosen for comparison are excellent and represent the standard in the model merging literature:
1. **Uniform Merging** (simplest baseline).
2. **Task Arithmetic (TA)** (globally scaled, standard baseline).
3. **Sparse Task Arithmetic (STA)** (coordinate-wise magnitude pruning, strong baseline).
4. **TIES-Merging** (state-of-the-art coordinate-wise consensus baseline).
5. **Unconstrained OFS-Tune** (direct validation-tuned baseline, serves as the direct ablation for the SVD projection).

Crucially, the authors did not "under-tune" the baselines:
- For **Task Arithmetic**, they swept the global scale factor $\lambda \in [0.1, 1.0]$.
- For **STA** and **TIES-Merging**, they performed a grid sweep over the pruning threshold $\theta \in [0.1, 0.9]$ on *each independent split*. This guarantees that coordinate-wise baselines are evaluated at their optimal operating capacity, eliminating tuning bias.

## Do the Results Support the Claims?
Yes, the empirical results strongly support the authors' central claims, with minor nuances that are discussed transparently in the paper.

1. **Claim: GSC-Merge significantly outperforms coordinate-wise heuristics (TIES, STA).**
   - **Support:** In the task-conditional swapping setting (Table 1), TIES-Merging and Sparse Task Arithmetic (STA) achieve joint mean accuracies of only $12.91 \pm 1.38\%$ and $11.99 \pm 0.94\%$ respectively. Uniform Merging fails at $11.16 \pm 0.00\%$. In comparison, GSC-Merge with $\gamma=0.3$ achieves $42.13 \pm 2.76\%$, and with $\gamma=0.5$ achieves $43.88 \pm 4.07\%$. This is an absolute improvement of over **30% in accuracy**, showing that coordinate-wise heuristics completely fail to resolve multi-task parameter conflicts on highly disparate domains, whereas spectral consensus preserves deep structural alignment.
2. **Claim: Grassmannian projection acts as a robust spectral regularizer, reducing split-sensitivity variance.**
   - **Support:** In Table 1, unconstrained OFS-Tune achieves a joint mean accuracy of $44.08 \pm 4.31\%$. GSC-Merge with $\gamma=0.3$ achieves a joint mean accuracy of $42.13 \pm 2.76\%$. This represents a **36% reduction in standard deviation** across validation splits (from $\pm 4.31\%$ to $\pm 2.76\%$), validating that the Grassmannian projection successfully filters out validation noise and stabilizes optimization.
3. **Claim: GSC-Merge matches unconstrained validation tuning performance while restricting parameter search.**
   - **Support:** Under both Task-Conditional (Table 1) and Task-Agnostic (Table 2) settings, GSC-Merge with $\gamma=0.5$ matches the performance of unconstrained OFS-Tune within statistical variance ($43.88 \pm 4.07\%$ vs. $44.08 \pm 4.31\%$ in Table 1; $20.61 \pm 4.80\%$ vs. $20.86 \pm 4.81\%$ in Table 2). This demonstrates that GSC-Merge retains the maximum possible multi-task update energy in a highly compressed parameter space.

## Nuanced Performance Discussion & Scientific Integrity
The authors exhibit exceptional scientific integrity and transparency in analyzing their experimental findings:
- **No Overclaiming:** The authors openly admit that GSC-Merge slightly degrades performance on individual tasks compared to unconstrained OFS-Tune, framing this as a classic bias-variance trade-off where spectral projection introduces a minor representation bias in exchange for robust noise-filtering.
- **Acknowledgement of the Performance Gap:** The authors do not shy away from pointing out that a massive performance gap remains between GSC-Merge ($43.88\%$) and the task-specific expert ceiling ($74.96\%$). Under the task-agnostic setting, the gap is even wider ($20.61\%$ vs. $74.96\%$). Highlighting this as a critical open research challenge for the community is a major strength and adds substantial credibility to the work.
