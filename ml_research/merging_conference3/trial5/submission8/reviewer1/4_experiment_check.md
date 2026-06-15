# Experimental Evaluation Check

## 1. Experimental Setup and Datasets
The paper evaluates the proposed framework using:
* **Model Backbone:** `vit_tiny_patch16_224` (5.7M parameters). This is a reasonable choice for a proof-of-concept but is relatively small compared to modern vision and language foundation models where model merging is most actively deployed.
* **Datasets:** MNIST, FashionMNIST, CIFAR-10, and SVHN. These are standard, widely-used benchmarks but represent **toy classification tasks**. Grayscale datasets like MNIST and FashionMNIST have very low-dimensional representation manifolds, which may simplify the ensembling task and make it easier for the Epigenetic Reader Heads to learn task boundaries. Expanding the evaluation to more complex, modern vision datasets (e.g., ImageNet-C, DomainNet) or language benchmarks would significantly strengthen the empirical validity.
* **Stream Configurations:** Shuffled I.I.D., Bursty temporal task shifts, and Small Batch ($B=2$). These streams are well-designed and represent a highly realistic set of conditions to test the transductive robustness of dynamic merging models.

---

## 2. Baselines
The paper compares against an appropriate and highly representative set of baselines:
1. **Static Baseline (Uniform Merging / Task Arithmetic):** Represents the zero-shot static ensembling starting point.
2. **Supervised Static Baseline (OFS-Tune):** Represents the offline supervised static ensembling limit under the same calibration budget.
3. **Online Test-Time Adaptation (AdaMerging):** Represents the state-of-the-art unsupervised local-adaptation baseline.
4. **Classical Dynamic Baseline (Linear Router):** Represents global scalar-gating dynamic ensembling.
5. **Quantum-Inspired Dynamic Baseline (QWS-Merge):** Represents state-of-the-art batch-averaged dynamic routing.

Comparing against these baselines provides a comprehensive picture of the different ensembling paradigms and their relative trade-offs.

---

## 3. Support for Claims
The empirical results strongly support the paper's key claims:
* **Online TTA Fragility:** Validated. Table 1 shows that AdaMerging collapses severely across all streams (yielding only $\approx 12\%$), failing to surpass even Uniform Merging ($19.05\%$). This supports the claim that local unsupervised entropy minimization is highly fragile under non-stationary streams and small-batch noise.
* **Batch-Averaging Coupling Hazard:** Validated mathematically in Appendix A and empirically discussed.
* **Stream Consistency of EpiMerge:** Validated. Table 1 shows that EpiMerge's performance is mathematically identical across Shuffled, Bursty, and Small Batch streams ($39.30\%$ for Rank-2). This empirically proves that sample-wise independent inference guarantees perfect robustness to test-time stream distribution shifts.
* **Expressiveness of Coordinate-Wise Gating:** Validated. EpiMerge-Rank2 ($39.30\%$) significantly outperforms the Linear Router ($34.95\%$) and QWS-Merge ($34.85\%$) by +4.35% and +4.45% absolute, proving that fine-grained coordinate-wise parameter gating is far more expressive than global scalar routing.
* **The Gating Rank and Optimization Trade-Off:** Validated. The "Rank-4 Degradation Paradox" (dropping to $31.05\%$) is a highly revealing result that honestly validates how expanding the rank of coordinate gating complicates the optimization landscape, leading to underfitting when data is extremely scarce.
* **Overfitting Trajectory and Dataset Scaling:** Validated.
  * Ablation A (Table 2) clearly maps the transductive overfitting trajectory, showing that optimizing the gating heads on the 64-sample calibration dataset for too many steps (e.g., 1000 steps) degrades test performance (from 40.35% to 38.90%).
  * Ablation B (Table 3) beautifully resolves the "Supervised Static Paradox," demonstrating that when the calibration budget is scaled to 512 samples, EpiMerge's accuracy surges by **+23.85% absolute** to $61.45\%$, virtually closing the performance gap with the static OFS-Tune baseline ($61.92\%$). It also validates that applying a Cosine Annealing learning rate scheduler yields a clean **+1.20% absolute improvement** over a constant learning rate.
* **Systems and Overhead Profiling:** Validated. Table 5 provides high-quality profiling of Peak GPU Memory and Latency, proving that EpiMerge successfully operates in parallel at a predictable 3x latency and mild memory overhead (+144MB at $B=64$) compared to static models.
