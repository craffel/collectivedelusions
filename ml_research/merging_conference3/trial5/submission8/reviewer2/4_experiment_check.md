# 4. Experiment Check

## Critical Evaluation of the Experimental Setup
1. **Toy Datasets & Tiny Backbone:** The authors evaluate their framework on a very small model, **Vision Transformer (ViT-Tiny)** with 5.7M parameters, across four simple, low-resolution classification tasks: **MNIST, FashionMNIST, CIFAR-10, and SVHN**. 
   - *Critique:* These are extremely small-scale, standard toy benchmarks in machine learning. Today, model merging is primarily motivated by and applied to large language models (LLMs) or large-scale multi-modal models (e.g., CLIP, ViT-Base/Large). Evaluating *only* on a ViT-Tiny model and toy datasets makes it unclear whether the proposed epigenetic gating mechanism and tensor contractions (`torch.einsum`) can scale to larger, more realistic models and diverse real-world tasks.
2. **Oracle Routing:** The evaluation relies on a "Task-Conditioning Oracle" which uses ground-truth labels to route test samples to task-specific heads. This avoids evaluating the model in a realistic multi-task setup where task identity is unknown or where a single unified classifier head is used, artificially inflating the reported accuracies.

## Evaluation of Baselines
1. **The AdaMerging "Strawman":** As discussed in Section 3, AdaMerging's performance (~12%) is barely above random guessing. Since AdaMerging is a peer-reviewed, highly successful test-time adaptation model, such a low score suggests that it was implemented with incorrect hyperparameters or not tuned at all.
2. **Missing Key Static Baselines:** The authors compare against Uniform Merging (Task Arithmetic) and OFS-Tune. However, they miss comparing against the most widely used and standard static model-merging baselines that resolve parameter conflicts, such as **TIES-Merging** and **DARE**. Since conflict resolution is a central theme of this paper, omitting TIES-Merging and DARE is a significant gap in the evaluation.
3. **No Baseline Tuning Sweeps:** The authors state that they use a constant scaling coefficient ($\lambda = 0.3$) for Uniform Merging. There is no evidence of a sweep over $\lambda$ to find the optimal static baseline performance, which is standard practice in model-merging evaluations.

## Data Inconsistencies and Contradictions (Crucial Empirical Concern)
A highly concerning aspect of the empirical results is the major discrepancy in baseline performance between Table 1 and Table 3:
- **OFS-Tune (64 samples):**
  - In **Table 1 (Main Results):** OFS-Tune is reported to achieve **41.48% $\pm$ 3.18%** accuracy on Shuffled I.I.D. using the 64-sample calibration dataset.
  - In **Table 3 (Ablation B1):** OFS-Tune under the "64 samples (16/task)" configuration is reported to achieve **53.23% $\pm$ 0.05%** accuracy!
  - *Critique:* This is a massive, unexplained discrepancy of **11.75% absolute** for the exact same static baseline trained on the exact same calibration size (64 samples).
- **EpiMerge-Rank2 (64 samples):**
  - In **Table 1:** EpiMerge-Rank2 is reported to achieve **39.30% $\pm$ 1.81%** accuracy.
  - In **Table 3:** EpiMerge (Rank-2) under the "64 samples (16/task)" configuration is reported to achieve **37.60% $\pm$ 1.82%** accuracy.
  - *Critique:* This is a discrepancy of **1.70% absolute**.
- **Implication:** Shifting baseline numbers between the main results and the ablation studies severely undermines the empirical rigor of the paper. It suggests a lack of standardization in the experimental pipeline, potential code leakage, or selective reporting of results.

## Do the Results Support the Claims?
- **Claim 1: "EpiMerge... outperforming uniform model-merging by +20.25% absolute and exceeding static supervised merging by +22.45% absolute."**
  - *Critique:* This claim is **partially false or highly misleading**. While EpiMerge outperforms Uniform Merging in Table 1, it does **not** exceed the static supervised baseline (OFS-Tune). In Table 1, OFS-Tune gets 41.48% and EpiMerge-Rank2 gets 39.30% (EpiMerge is *underperforming* by 2.18% absolute). In Table 3, OFS-Tune consistently outperforms EpiMerge across all calibration sizes (e.g., 53.23% vs 37.60% at 64 samples, and 61.92% vs 61.45% at 512 samples). The claim in the abstract that EpiMerge exceeds static supervised merging by +22.45% is contradicted by their own tables.
- **Claim 2: "EpiMerge's performance is mathematically guaranteed to remain perfectly consistent across Shuffled, Bursty, and Small Batch streams."**
  - *Critique:* This claim is technically true because EpiMerge uses sample-independent inference. However, this is **not** a unique property of EpiMerge. Static OFS-Tune and the Linear Router *also* produce mathematically identical results across all three streams in Table 1. Therefore, framing stream consistency as a novel, unique advantage of EpiMerge is highly misleading; it is simply a shared property of any sample-independent inference framework.
- **Claim 3: "EpiMerge-Active... reduces parameters to exactly 1.0x."**
  - *Critique:* While true that it reduces the parameter footprint to 1.0x, Table 1 reveals that EpiMerge-Active incurs a significant performance drop, falling to **36.70%** (a drop of 2.60% absolute compared to Rank-2, and 4.78% absolute compared to OFS-Tune). This substantial system-accuracy trade-off is not sufficiently highlighted as a major deployment limitation in the abstract.

## Missing Analyses
1. **Per-Task Accuracy Breakdown:** The paper only reports average multi-task accuracy. It is highly critical to see individual accuracies for MNIST, FashionMNIST, CIFAR-10, and SVHN. CIFAR-10 and SVHN are significantly more challenging than MNIST/FashionMNIST. Reporting only the average hides whether the model is completely failing on the harder datasets (which is likely, given the low average accuracy of ~39%).
2. **Sweep over Gating Latent Dimension $d$:** The authors set $d = K = 4$. They do not show any ablation sweeping $d \in \{2, 8, 16\}$ to verify how the latent bottleneck dimension affects optimization and accuracy.
3. **Statistical Significance Testing:** With standard deviations up to 3.22%, there are no t-tests or significance markers to verify if the differences between EpiMerge and other dynamic methods (like QWS-Merge or Linear Router) are statistically significant.
