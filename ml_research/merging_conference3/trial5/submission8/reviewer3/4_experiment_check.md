# 4. Experiment Check

## Critical Evaluation of the Experimental Setup
The experimental evaluation is designed to test the robustness of the model merging frameworks under different stream distributions (Shuffled, Bursty, and Small Batch). While this setup is theoretically interesting, the choice of datasets and the resulting absolute performance levels reveal significant weaknesses:

1. **Toy Dataset Setup:**
   The evaluation is restricted to very simple, low-resolution datasets: MNIST, FashionMNIST, CIFAR-10, and SVHN (10 classes each). These are standard toy benchmarks.
   
2. **Abysmally Low Absolute Performance:**
   The unmerged individual experts (Upper Bound) achieve an average accuracy of **94.85%** across these four tasks (with MNIST at 99.2%, FashionMNIST at 93.6%, CIFAR-10 at 91.5%, and SVHN at 95.1%). 
   However, the best-performing dynamic model (EpiMerge-Rank2) gets only **39.30%** average accuracy in Table 1, and the static supervised baseline (OFS-Tune) gets **41.48%** (or **53.23%** in Table 3).
   - This represents a catastrophic loss of accuracy (over **55% absolute drop** from the expert ceiling). 
   - On a task-conditioned 10-class problem, random guessing is 10%. Achieving only 39.30% on incredibly simple tasks like MNIST and FashionMNIST indicates that the merged model is practically useless for real-world applications. The authors gloss over this massive performance collapse and focus entirely on outperforming Uniform Merging (which gets 19.05%).

3. **The Unrealism of the Task-Conditioning Oracle:**
   The entire evaluation assumes a **Task-Conditioning Oracle** at test-time. This means the model is given the ground-truth task label of each test sample so it can select the correct classification head. 
   - In a realistic deployment, the model would not have access to these labels, making the task significantly harder (routing across all 40 classes, dealing with severe inter-class overlap and representation conflicts).
   - While the authors propose two "non-oracle pathways" in Section 4.8 (Integrated Task Classifier and Shared Unified Head), they **do not evaluate either of them empirically**. The entire experimental evaluation relies on this artificial, unrealistic oracle assumption, which severely undercuts the credibility of the results for actual production.

---

## Critical Evaluation of the Baselines and Results

1. **The Static Baseline Paradox (OFS-Tune vs. EpiMerge):**
   The authors proposed EpiMerge as a highly advanced, coordinate-wise dynamic merging framework. However, the simplest supervised static baseline, **OFS-Tune** (which simply optimizes 48 layer-wise ensembling scalars), **consistently outperforms EpiMerge across all data regimes**:
   - At 64 samples: OFS-Tune (53.23% $\pm$ 0.05%) vs. EpiMerge-Rank2 (37.60% $\pm$ 1.82%) $\rightarrow$ OFS-Tune is **+15.63% better**.
   - At 128 samples: OFS-Tune (57.98% $\pm$ 0.10%) vs. EpiMerge-Rank2 (43.60% $\pm$ 1.95%) $\rightarrow$ OFS-Tune is **+14.38% better**.
   - At 256 samples: OFS-Tune (60.05% $\pm$ 0.04%) vs. EpiMerge-Rank2 (51.40% $\pm$ 1.74%) $\rightarrow$ OFS-Tune is **+8.65% better**.
   - At 512 samples: OFS-Tune (61.92% $\pm$ 0.20%) vs. EpiMerge-Rank2 (61.45% $\pm$ 1.88%) $\rightarrow$ OFS-Tune is **+0.47% better**.
   
   Why would any practitioner implement EpiMerge—which doubles the parameter footprint (2.0x), triples wall-clock latency (3x), and requires complex `torch.einsum` operations—when a simple static compromise (OFS-Tune) gets **consistently higher accuracy** with **zero latency overhead and zero extra parameter footprint**? The empirical results show that the dynamic complexity of EpiMerge does not translate to superior accuracy over simpler static baselines.

2. **The Rank-4 Degradation Paradox:**
   In Table 1, scaling the rank from $R=2$ to $R=4$ collapses performance to **31.05%** (a drop of -8.25% absolute). The authors explain that the increased parameter count complicates the optimization landscape. But $R=4$ only introduces about 73k parameters across the entire ViT-Tiny network. If the model collapses so easily on such a minor increase in parameters, it indicates extreme training instability and high sensitivity to hyperparameters, casting doubt on the robustness of the entire framework.

3. **Discrepancies in the Control Group Results:**
   As highlighted in the methodology check, the results for the 64-sample calibration baseline are completely inconsistent:
   - Table 1: OFS-Tune (41.48% $\pm$ 3.18%), EpiMerge-Rank2 (39.30% $\pm$ 1.81%).
   - Table 3: OFS-Tune (53.23% $\pm$ 0.05%), EpiMerge-Rank2 (37.60% $\pm$ 1.82%).
   The performance of the *exact same configurations* on the *exact same budget* fluctuates by over **11.7% absolute** for OFS-Tune and **1.7% absolute** for EpiMerge, with the standard deviations changing dramatically (e.g., from 3.18% to 0.05% for OFS-Tune). This indicates a serious lack of experimental control or major bugs in their evaluation pipeline.
