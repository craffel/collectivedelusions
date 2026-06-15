# Experimental Validation Check: EpiMerge

## 1. Experimental Design and Benchmark Selection
The empirical evaluation of EpiMerge is well-structured and comprehensive:
*   **Backbone:** Vision Transformer (`vit_tiny_patch16_224`, 5.7M parameters) fine-tuned independently on 4 tasks.
*   **Tasks:** A diverse 4-task classification mixture consisting of MNIST, FashionMNIST, CIFAR-10, and SVHN. The independent expert upper bound (average ceiling of 94.85%) highlights the substantial representation conflicts and parameter interference that occur during merging, framing multi-task model synthesis as a highly challenging and realistic research area.
*   **Stream Configurations:** Shuffled I.I.D. stream, Bursty stream (temporal clustering by task), and Small Batch ($B=2$, representing edge deployment with local noise). These streams are well-designed to stress-test the models against temporal task drift and batch size dependencies.

## 2. In-Depth Results and Baseline Comparisons
*   **Fragility of online TTA:** Online AdaMerging collapses to $12.25\%$ (I.I.D.) and $11.85\%$ ($B=2$), failing to surpass even Uniform Merging ($19.05\%$). This confirms that unsupervised test-time optimization of layer-wise coefficients on small or noisy batches leads to transductive overfitting and representation collapse.
*   **Consistency across Streams:** Because EpiMerge maintains true sample-wise independent inference, its performance remains perfectly consistent across Shuffled I.I.D. ($39.30\%$), Bursty ($39.30\%$), and Small Batch ($39.28\%$) streams, unlike online test-time adaptation which collapses. This is a crucial validation of its robustness.

## 3. Detailed Critical Analysis of Ablation Studies
The paper includes two highly valuable ablation studies to investigate the high-dimensional optimization bottlenecks, but a rigorous analysis reveals several critical empirical and methodological limitations:

### Ablation Study A: Calibration Steps ($\tau$)
*   Sweeping $\tau \in \{100, 200, 500, 1000\}$ steps reveals a classic transductive overfitting trajectory. At 100 steps of calibration, EpiMerge-Rank2 achieves its peak generalization of 40.35% (on Shuffled I.I.D. stream, Seed 42).
*   As steps increase, accuracy steadily drops to 39.60% (200 steps), 39.65% (500 steps), and collapses to 38.90% (1000 steps). This demonstrates that over-optimizing the gating parameters for too many steps forces the ERHs to memorize the specific pixel alignments of the calibration samples, damaging their test-time adaptation and generalization on the unseen test stream.

### Ablation Study B: Calibration Dataset Size and Learning Rate Schedulers
The scaling trajectory presents a massive performance surge:
*   **Breakthrough Scaling:** As calibration size $|\mathcal{D}_{cal}|$ increases from 64 to 512 samples, EpiMerge's accuracy surges dramatically: $37.60\% \rightarrow 43.60\% \rightarrow 51.40\% \rightarrow 61.45\%$.
*   At 256 samples, EpiMerge-Rank2 achieves $51.40\%$, and at 512 samples, it reaches an outstanding **61.45%** accuracy, representing a monumental **+23.85% absolute leap** over its default performance at 64 samples.

However, several major empirical limitations must be highlighted:
*   **Empirical Limitation 1: Lack of Seed Variance and Standard Deviations in Ablations:** Unlike the main results in Table 1, which report means and standard deviations over 3 independent seeds, all ablation results in Tables 2 and 3 are reported for a single seed (seed 42) and lack standard deviations. Since model-merging performance is highly sensitive to the specific few-shot samples chosen, conducting these sweeps over multiple seeds is necessary to verify that the observed scaling trends are statistically robust and not seed-dependent artifacts.
*   **Empirical Limitation 2: The "Supervised Static Paradox" is Not Fully Resolved in Absolute Terms:** The authors claim that scaling the dataset to 512 samples "resolves the Supervised Static Paradox" by showing that EpiMerge (61.45%) almost completely recovers the static supervised ceiling. However, looking at Table 3, the static supervised baseline OFS-Tune *also* scales with data, achieving **61.80%** at 512 samples. Thus, in absolute terms, the simpler static baseline (which only optimizes 48 layer-wise scalars) **consistently outperforms** the dynamic coordinate gating of EpiMerge across all dataset sizes. The paradox is resolved in terms of closing the gap, but the simpler static baseline remains an extremely formidable and highly regularized competitor.
*   **Empirical Limitation 3: The Rank-4 Performance Degradation Paradox:** Why does EpiMerge-Rank4 (31.05%) perform significantly worse than EpiMerge-Rank1 (39.22%) and EpiMerge-Rank2 (39.30%) under the default 64-sample budget? While higher rank increases expressivity, it also increases the parameter space and optimization difficulty, which apparently exacerbates underfitting/saddle-point issues under the 64-sample budget. This optimization bottleneck is a major limitation of high-rank formulations under low-data constraints and should be analyzed more thoroughly.
*   **Empirical Limitation 4: Lack of Ablation on Layer Partitioning ($L_{early}$):** For the "Active-Early" variant, which slashes parameter memory and latency, the paper partitions the model into early static and deep dynamic stages. However, there is no sweep or ablation on the selection of $L_{early}$. How sensitive is the model's accuracy and inference latency to the choice of $L_{early}$ (e.g., $L_{early} \in \{2, 4, 6, 8\}$)?

## 4. Systems Profiling and Gating Dynamics
*   **Rigorous GPU Profiling:** The authors are highly commended for performing wall-clock latency and peak GPU memory profiling across batch sizes $B \in \{1, 8, 16, 32, 64\}$. This empirical profiling confirms that reconstructing weight tensors on-the-fly and running the duplicate sensory copy triples wall-clock latency (from 9.12ms to 27.34ms at $B=64$) and increases peak memory by +22.8% (+144.05MB at $B=64$). This is an extremely thorough systems-level characterization.
*   **Task-Specific Adaptation Verification:** Average learned routing weights of Linear Router and average row-gating intensities of EpiMerge (Table 4) confirm active gating. For instance, MNIST inputs heavily activate the MNIST expert (0.611) and the row-gating intensity for MNIST expert is 0.516 (vs. SVHN at 0.498), demonstrating that the model learns to align parameter scaling with the input task's representational characteristics.

## 5. Experimental Verdict
**Good-to-Excellent.** The experimental design, stream configurations, baselines, and systems-level profiling are exceptionally rigorous and comprehensive. However, the lack of multi-seed averages for ablation studies, the persistent absolute superiority of the static baseline (OFS-Tune), and the missing analysis of layer partitioning sensitivity represent important empirical gaps that the authors should address.
