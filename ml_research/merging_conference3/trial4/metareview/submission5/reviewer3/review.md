# Peer Review

## Summary of the Paper
This paper addresses the challenge of representational interference in weight-space model merging. When independent task-specific models (experts) fine-tuned from a shared pre-trained base are merged via linear addition (e.g., standard Task Arithmetic), they often suffer from severe representation collapse. To resolve this, the authors propose **Sparsity-Guided Task Arithmetic (SG-TA)**, a deterministic framework that applies magnitude-based binary masking to individual task-specific update vectors prior to merging. This masking is intended to filter out low-magnitude background noise and preserve only the high-magnitude parameter updates that store specialized task behavior.

The paper evaluates two masking scopes: **Global Quantile (GQ)**, which computes a single threshold globally across the entire model, and **Layer-wise Quantile (LQ)**, which enforces a homogeneous keep-ratio per layer. The merging scaling factor and keep-ratios are optimized via Offline Few-Shot Validation Tuning (OFS-Tune) on a tiny validation set of 10 samples per task. Evaluated on a 4-dataset classification benchmark (MNIST, FashionMNIST, CIFAR-10, SVHN) using a compact Vision Transformer backbone (ViT-Tiny), SG-TA (GQ) achieves a Joint Mean Accuracy of **61.40% $\pm$ 1.39%**, outperforming standard Task Arithmetic, unpruned layer-group scaling, and showing comparable performance to more complex baselines like TIES-Merging and DARE-Merging.

---

## Strengths and Weaknesses

### Strengths
1. **Outstanding Empirical Rigor and Comprehensive Ablations:** The empirical evaluation is exemplary. The authors perform exhaustive multi-axial grid sweeps over parameters, use 5 random calibration seeds to report statistical variance, and include excellent control baselines. The inclusion of Layer-Group Scaling (L-Scale) as a control experiment successfully isolates weight-space pruning as the primary driver of performance. The paper also includes useful analyses on task-vector magnitude normalization, continuous soft gating, non-uniform coordinate-search optimization, cosine similarities of expert task vectors (orthogonal noise hypothesis), and simulated layer specialization in transformer blocks.
2. **Exemplary Academic Honesty and Transparency:** The paper is written with a high level of scientific integrity. The authors explicitly state that their method's improvement over TIES-Merging is not statistically significant due to overlapping standard deviations. Furthermore, they dedicate a significant portion of their discussion to analyzing the "Absolute Performance Degradation Bottleneck" (the $34.51\%$ absolute performance gap between the merged model and the expert ceiling), critically examining the limitations of compact backbones.
3. **Exceptional Clarity and Structure:** The paper is incredibly well-written, logically organized, and mathematically precise. The notation is consistent, the methodology is easy to follow, and the formatting is professional. Sufficient details are provided for an expert reader to easily reproduce all findings.

### Weaknesses
1. **Low Conceptual Novelty (Incremental Contribution):** The core algorithmic idea is highly incremental. Applying magnitude-based pruning to task vectors is a well-known concept in model merging; indeed, standard **TIES-Merging** (Yadav et al., 2023) utilizes magnitude pruning as its very first step (the "Trim" phase). The "delta" here is simply omitting TIES-Merging's sign election and sign-compatible averaging phases. Stripping away components of an existing algorithm is a minor simplification rather than a bold conceptual leap. 
2. **Equivalence to Existing Baselines:** As the authors themselves acknowledge in Section 4.2, the Layer-wise Quantile (LQ) variant is mathematically equivalent to standard Decoupled Prune-then-Merge (P-then-M) under fully optimized conditions. This means the only remaining algorithmic distinction is Global Quantile (GQ) masking, which is a very straightforward extension of layer-wise pruning to a global threshold. While showing that global pruning outperforms layer-wise pruning is an interesting empirical result, it represents a minor engineering detail rather than a fundamentally original or paradigm-shifting mechanism.
3. **No Statistically Significant Improvement Over State-of-the-Art:** Under a fully optimized and fair calibration protocol, the proposed SG-TA (GQ) achieves $61.40\% \pm 1.39\%$, while the existing TIES-Merging baseline achieves $60.64\% \pm 1.30\%$. Because the difference is only $0.76\%$ and their standard deviations overlap, the performance improvement is not statistically significant. This raises doubts about whether the proposed framework represents a true advancement over existing work.
4. **Toy-Scale Experimental Setup:** The evaluation is restricted to a very small Vision Transformer (ViT-Tiny, 5.7M parameters) and basic, low-resolution image classification datasets (MNIST, FashionMNIST, CIFAR-10, SVHN). In practice, weight-space merging is primarily used to consolidate massive generative foundation models (such as LLMs with billions of parameters) where parameter redundancy, layer behavior, and representational dynamics are fundamentally different. It is unclear if these findings on toy-scale visual classification will generalize to large-scale generative settings.
5. **Vulnerability of Few-Shot Calibration to Small-Sample Noise:** Under Task Vector Normalization (TV-Norm), the calibration becomes highly volatile at $N_{\text{val}}=10$, with the standard deviation rising to $\pm 4.56\%$. While increasing the pool size to 20 or 100 stabilizes performance, this sensitivity demonstrates that the few-shot calibration paradigm is fragile and noise-sensitive under balanced update scales, which reduces its reliability in true low-data regimes.
6. **Unresolved Core Capacity Bottleneck:** The absolute performance of the merged model remains extremely low ($61.40\%$ Joint Mean) compared to both separate experts ($95.91\%$) and Joint MTL training ($95.55\%$). For simple datasets like MNIST, the merged model's performance collapses to $36.74\%$ (compared to the $99.05\%$ expert ceiling). This indicates that simple deterministic magnitude masking, while helpful in relative terms, does not solve the fundamental capacity constraints and representational bottlenecks of small-scale neural network consolidation.

---

## Evaluation of Criteria

### Soundness: Excellent
The paper is technically highly sound. The mathematical framework is precise, the experimental methodology is rigorous, and the authors are careful and honest in evaluating both the strengths and weaknesses of their work. The use of multiple random seeds and comprehensive control baselines (like L-Scale) ensures that the empirical findings are trustworthy.

### Presentation: Excellent
The submission is beautifully written, clear, and well-structured. The narrative is easy to follow, and the paper does a superb job of positioning itself relative to prior literature. All equations, figures, and tables are clean and informative.

### Significance: Fair
The practical significance of this paper is limited. Because the merged model experiences severe absolute performance degradation (e.g., MNIST collapsing to 36%) and the evaluation is confined to small-scale datasets, the resulting model is practically undeployable for high-stakes applications. However, the scientific insights regarding global budget flexibility over rigid layer-wise constraints, the use of continuous soft gating to reduce calibration variance, and the value of task-vector normalization provide useful reference points and actionable guidelines for researchers working on weight-space consolidation.

### Originality: Fair
The paper provides valuable empirical observations, but the conceptual originality is low. The proposed framework is a collection of minor engineering adjustments, simplifications, and parameter sweeps built directly on top of existing ideas (Task Arithmetic and TIES-Merging). It lacks a truly original, bold, or ambitious conceptual leap that could change how the community thinks about the problem of model merging.

---

## Overall Recommendation

### Rating: 3: Weak reject
**Justification:** 
This paper has several clear merits. The manuscript is exceptionally well-written, mathematically precise, and demonstrates an outstanding level of academic honesty and experimental rigor. The authors' thorough evaluations, detailed ablations, and candid discussions of limitations are highly commendable.

However, from the perspective of conceptual novelty and magnitude of contribution, the paper falls short of the standard required for a conference publication. The proposed SG-TA is a highly incremental framework that represents a minor simplification of TIES-Merging (pure magnitude pruning without sign consensus) or a straightforward global extension of Decoupled Prune-then-Merge. The resulting performance improvement over TIES-Merging ($+0.76\%$) is not statistically significant. Furthermore, the experiments are confined to toy-scale visual classification on a tiny 5.7M parameter model, leaving the generalizability of these findings to modern foundation models unproven. Ultimately, because the core contribution is highly incremental and lacks a fundamentally original or paradigm-shifting conceptual leap, the weaknesses outweigh the merits.
