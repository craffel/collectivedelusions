# 4. Experimental Evaluation and Claim Support

## Experimental Setup & Datasets
1. **Toy-Scale Benchmarks:**
   The entire empirical evaluation is conducted on four classic, low-resolution toy datasets: MNIST (28x28 grayscale, single-channel projected to three channels), FashionMNIST (28x28 grayscale), CIFAR-10 (32x32), and SVHN (32x32). This setup is highly unrepresentative of modern machine learning, where model merging is primarily utilized for large-scale vision-language models or large language models (LLMs) on high-resolution images or complex text benchmarks (e.g., ImageNet-1K, GLUE, MMLU).
2. **Sub-Scale Backbone:**
   The authors use a tiny, obsolete backbone: $\mathtt{vit\_tiny\_patch16\_224}$ with only 5.7M parameters. Modern model merging literature evaluates on models that are several orders of magnitude larger (e.g., ViT-Base/Large with 86M/307M parameters, or LLMs like LLaMA-2-7B/13B). Results on a 5.7M parameter model on toy vision datasets may not generalize to large-scale modern architectures, making the paper's speculative claims about "billion-parameter LLMs" completely unvalidated.
3. **Low-Data Split & High Variance:**
   The training and calibration sets are extremely small (2,000 fine-tuning samples and 64 calibration samples). This creates a highly volatile optimization landscape where small changes in sample selection could drastically alter the results. However, the authors report single deterministic numbers without any standard deviations over multiple seeds, casting doubt on the statistical robustness of the reported gains.

## Baseline Analysis
The paper includes a reasonable list of baselines (Task Arithmetic, AdaMerging, Linear Router, QWS-Merge, and OFS-Tune), but a critical look at the results reveals that **the proposed ChaosMerge method is substantially weaker than almost all optimized baselines**:

1. **Failure to Beat Supervised Static Tuning (OFS-Tune):**
   Under the Task-Averaged setting, ChaosMerge (G-CML) achieves **71.20%** average accuracy, which is worse than the simple supervised static baseline (OFS-Tune) at **73.55%**. In other words, a static set of shared layer-wise weights optimized on the 64-sample calibration set beats the complex, dynamic G-CML model by **+2.35% absolute** without any runtime feature projection, CML recurrence, or gating.
2. **Failure to Beat Unconstrained Dynamic Routers:**
   Under the Task-Specific setting, the Linear Router (77.10%) and QWS-Merge (77.05%) outperform standard ChaosMerge G-CML (73.80%) by **+3.30% absolute**. 
3. **Catastrophic Failure vs. Task-Specific OFS-Tune:**
   The newly introduced task-conditional static baseline, Task-Specific OFS-Tune, directly optimizes a separate set of static weights per task on the calibration set. Since G-CML's centroid formulation also resolves to a static task-specific weight at test-time, this is a highly appropriate symmetric baseline. This static task-conditional baseline achieves **82.90%** average accuracy, outperforming standard G-CML (73.80%) by a massive **+9.10% absolute**, and outperforming the proposed Annealed G-CML (78.12%) by **+4.78% absolute**. This shows that the heavy mathematical machinery of G-CML actually hinders representational performance compared to simple, unconstrained static optimization.

## Evaluation of Specific Claims
- **Claim 1: "ChaosMerge significantly stabilizes optimization... outperforming competitive static and dynamic baselines."**
  *Status: Refuted.* Standard ChaosMerge (73.80%) is significantly outperformed by the Linear Router (77.10%), QWS-Merge (77.05%), and Task-Specific OFS-Tune (82.90%). Only when the authors introduce a highly complex, hybrid "Annealed Chaos-to-Order" framework (Table 2) does it reach 78.12%, which barely beats the Linear Router by +1.02% at the cost of substantial algorithmic complexity.
- **Claim 2: "Enforcing a 384-parameter physical lattice successfully avoids the Overfitting-Optimizer Paradox."**
  *Status: Weakly Supported / Contradicted.* The authors claim that unconstrained dynamic routers (10k+ parameters) overfit to small calibration sets. However, the paper's own empirical results in Table 1 contradict this: on the tiny 64-sample calibration set, the Linear Router and QWS-Merge achieve 77.10% and 77.05% average accuracy, whereas ChaosMerge G-CML achieves only 73.80%. If the baselines were suffering from severe overfitting, their test accuracy would crash below ChaosMerge. Instead, they perform significantly better, suggesting that 10k parameters is still highly regularized enough and that ChaosMerge's 384 parameters represent an *under-parameterization* that underfits the data.
- **Claim 3: "Centroid-based dynamic routing resolves the batch-averaging contradiction... with virtually zero test-time overhead."**
  *Status: Partially True but Highly Restricted.* While the centroid formulation does avoid test-time swap latency, Section 3.4 demonstrates that deploying it task-agnostically via unsupervised clustering results in a catastrophic accuracy crash (**-29.69%** absolute drop). Thus, the method is practically restricted to task-aware environments where task boundaries are known, completely undermining the claim of a "task-agnostic deployment."
