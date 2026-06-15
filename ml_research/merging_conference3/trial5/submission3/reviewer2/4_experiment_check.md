# Experimental Setup and Results Evaluation

## Evaluation of Experimental Setup
The experimental setup is designed to directly address and deconstruct the baseline collapse reported in previous literature (QWS-Merge; Vance, 2025).
- **Backbone Model:** A compact Vision Transformer ($\mathtt{vit\_tiny\_patch16\_224}$) with 5.7M parameters. This is appropriate as a controlled, lightweight setting to perform thorough scientific analysis and multi-seed sweeps.
- **Tasks and Datasets:** A 4-task vision benchmark:
  1. **MNIST** (handwritten digits)
  2. **FashionMNIST** (clothing items)
  3. **CIFAR-10** (natural objects)
  4. **SVHN** (street view numbers)
  While these are standard, relatively low-resolution image classification datasets, they represent a diverse mix of domains (synthetic, natural, and real-world numbers) with varying difficulty.
- **Calibration Set:** 16 samples per task (64 total samples), drawn deterministically under seed 42. This is extremely few-shot, which is typical for data-free or low-resource model merging.

## Evaluation of Baselines
The paper compares RLR against a highly comprehensive set of baselines:
1. **Individual Experts (Ceiling):** Non-merged task-specific models.
2. **Uniform Merging:** Static baseline.
3. **OFS-Tune:** Supervised static coefficient search.
4. **AdaMerging:** Unsupervised test-time adaptation.
5. **Linear Router (Classical):** Unregularized dynamic routing.
6. **QWS-Merge (Reported & Local Baseline):** The state-of-the-art dynamic method being deconstructed.

The inclusion of a **local re-implementation of QWS-Merge** trained under identical conditions on the exact same expert weights is an outstanding strength. It ensures that the comparison is completely fair and isolates the architectural effects of wave phase-basis projections from checkpoints or training differences.

## Do the Results Support the Claims?
Yes, the empirical results strongly support the paper's primary and secondary claims:

1. **Deconstruction of SVHN Collapse (Supported):**
   - In Table 1, the classical unregularized Linear Router achieves **$94.87\%$** SVHN accuracy and **$95.46\%$** Joint Mean accuracy on seed 42.
   - Across 5 random calibration seeds (Section 4.3), the classical Linear Router consistently converges without a single instance of collapse, achieving **$91.20\% \pm 1.85\%$** SVHN accuracy and **$91.53\% \pm 0.41\%$** Joint Mean.
   - This directly and statistically debunks Vance et al.’s claim that classical gating collapses to $15.30\%$ on SVHN.

2. **Resilience to Heterogeneous Mixed-Task Streams (Supported):**
   - In Table 3, as the evaluation batch size increases from $B=1$ to $B=256$, all dynamic methods experience performance degradation due to batch-level coefficient averaging.
   - However, **RLR demonstrates superior resilience compared to the unregularized Linear Router**:
     - At $B=16$: RLR achieves **$76.85\%$** accuracy vs. the baseline's **$75.48\%$** ($+1.37\%$ absolute benefit).
     - At $B=256$: RLR achieves **$75.03\%$** accuracy vs. the baseline's **$73.15\%$** ($+1.88\%$ absolute benefit).
     - This validates the hypothesis that weight decay and temperature scaling act as stabilizers that prevent the gating weights from saturating and collapsing to hard single-expert decisions.

3. **Hyperparameter Insensitivity (Supported):**
   - The 2D sensitivity sweep (Figure 3) demonstrates that RLR converges stably across a wide range of values for $L_2$ regularization coefficient $\alpha \in [0.0, 0.02]$ and temperature $T \in [1.0, 5.0]$, indicating that the performance is not highly sensitive to hyperparameter tuning.

## Critiques and Areas for Improvement

1. **Scale of Evaluation:**
   - The experiments are restricted to a compact Vision Transformer (ViT-Tiny) and standard classification datasets. Although this was the exact setup used in Vance et al. (2025), a stronger paper would demonstrate the efficacy and scalability of RLR on modern large-scale architectures, such as Large Language Models (LLMs) or larger Vision-Language Models (e.g., CLIP-ViT-Base or Large) on more complex multitask benchmarks. The authors propose concrete scaling pathways to LLMs in Section 5, but do not provide any actual empirical results on them.
2. **Dynamic Routing Performance in Highly Heterogeneous Streams:**
   - Table 3 reveals that while RLR is more resilient to heterogeneity collapse than unregularized routing, its performance at large batch sizes ($B=256$, **$75.03\%$**) is significantly worse than supervised static methods like OFS-Tune (**$86.23\%$**). 
   - The authors include a commendable, honest discussion of this trade-off in Section 4.4, noting that static methods are superior for low-latency systems receiving randomized test streams. However, this highlights that dynamic model merging still has severe fundamental limitations when queries cannot be batched homogeneously.
