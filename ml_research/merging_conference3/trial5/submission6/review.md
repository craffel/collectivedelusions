# Peer Review of SLD-Merge

## 1. Summary of the Paper
This paper addresses the deployment bottlenecks of weight-space dynamic model-merging methods (such as QWS-Merge and Linear Routers) under heterogeneous, multi-task inference streams. The authors show that traditional dynamic methods suffer from "heterogeneity collapse" and batch-dependency because they average routing coefficients across the batch dimension. This violates the standard I.I.D. assumption, making a single sample's prediction shift depending on other samples co-packaged in the same batch.

To resolve this, the authors propose **Sparse Low-Rank Dynamic Merging (SLD-Merge)**. The method consists of:
1. **Offline SVD Task-Vector Decomposition:** Compressing dense late-layer expert task vectors into post-hoc, low-rank adapters ($r=8$), which reduces weight storage overhead by 91%.
2. **Bounded Cosine-Similarity Router:** Restricting average-pooled activation representations onto a bounded spherical space to suppress high-frequency representation noise.
3. **Top-1 Sparse Gating (Hard Gating):** Routing activations sample-wise through only the single most relevant low-rank expert adapter path, achieving fully parallel, batch-independent inference.
4. **Activation-Space Mean Initialization:** Centering task bases at representative activation centroids on a tiny unlabeled calibration set to enable zero-shot routing.
5. **Autonomous Classification Head Selection:** Eliminating test-time task-oracle dependencies by dynamically selecting the classification head using layer-averaged cosine similarity scores.

Evaluating on a 4-dataset Vision Transformer benchmark (MNIST, FashionMNIST, CIFAR-10, SVHN) using a ViT-Tiny backbone, SLD-Merge is reported to maintain a stable joint accuracy of **63.87%** across all batch sizes, outperforming batch-dependent dynamic baselines by up to **+8.50%** while adding only 8.3% FLOPs.

---

## 2. Strengths
*   **Pragmatic Problem Motivation:** The paper addresses a highly relevant, real-world deployment challenge. Exposing that traditional dynamic merging methods suffer from "heterogeneity collapse" and batch-dependency—meaning that a sample's prediction changes depending on its batch-mates—is a critical, practical observation that directly impacts production systems.
*   **Parameter and Computational Efficiency:** Combining offline SVD for task-vector compression with Top-1 hard gating is a very clever way to design a resource-efficient dynamic model. Restricting additional storage to 4MB (a 91% reduction) and keeping inference compute overhead to only 8.3% extra FLOPs makes the method highly suitable for edge NPUs and memory-constrained microcontrollers.
*   **Excellent Writing and Visuals:** The paper is written with high clarity and structure. Figure 1 beautifully conceptualizes heterogeneity collapse, and the mathematical equations (1-12) are rigorous, clean, and easy to follow.

---

## 3. Weaknesses & Critical Flaws

### Critical Flaw 1: Discrepancy between Claims and Code regarding "Autonomous Classification Head Selection" (Oracle Leakage)
A core claim of this paper is that SLD-Merge operates in a "completely batch-independent, stateless" manner without relying on a "privileged oracle" at test time to select the classification head. Section 3.5 introduces "Autonomous Classification Head Selection" (Equations 13-14), claiming that the layer-averaged cosine similarity scores select the correct classification head with $\approx 100\%$ accuracy, achieving identical performance to oracle head selection.

However, **a detailed code audit of the evaluation script (`run_experiments.py`) reveals that no autonomous classification head selection is actually implemented or run during evaluation.**
1. In the evaluation loop (`evaluate_stream`), the ground-truth task labels (`task_idxs`) are passed directly from the data loader to `model.predict`:
   ```python
   logits = model.predict(imgs, task_idxs, mode=mode, rank=rank, batch_size_eval=batch_size)
   ```
2. Inside `predict` in `run_experiments.py`, the classification head is chosen **strictly using these ground-truth task labels**:
   ```python
   def predict(self, imgs, task_labels, mode='uniform', rank=8, batch_size_eval=16):
       features = self.forward_backbone_with_merging(imgs, mode=mode, rank=rank, batch_size_eval=batch_size_eval)
       all_logits = torch.stack([head(features) for head in self.heads]) # (K, B, 10)
       B_size = imgs.shape[0]
       selected_logits = all_logits[task_labels, torch.arange(B_size)] # Oracle Selection
       return selected_logits
   ```
This means that **every single method (baselines and SLD-Merge) was evaluated using a privileged task oracle at test time**. In a real-world streaming environment, there is no ground-truth oracle label. If the model relies on a task oracle to select the classification head, it is completely undeployable and cannot be run in a stateless, autonomous fashion. 

Furthermore, the comparative ablation study presented in Section 4.4 and Table 1/2 claiming to compare "Autonomous vs. Oracle Head Selection" is unsupported by any code or execution. The script simply reports the oracle accuracy twice under different names. This discrepancy represents a severe scientific soundness issue and compromises the integrity of the empirical results.

### Critical Flaw 2: Micro-Scale Subsampled Datasets (Toy Scale)
The evaluation is restricted to subsets of only **256 training, 128 calibration, and 256 test samples per dataset** across MNIST, FashionMNIST, CIFAR-10, and SVHN, resulting in a total test stream of just 1,024 samples. While the authors justify this as modeling a "low-shot streaming environment," this scale is far too small to establish generalizability. 
More importantly, the standalone expert ceiling on SVHN is exceptionally low (**29.30%**), which is barely above random guessing (10%) on a 10-class task. This indicates that the expert models themselves are severely under-trained (only 3 epochs on 256 samples with frozen blocks 0-8). Evaluating a weight-merging framework on near-random, under-trained experts severely weakens the validity of the merging conclusions.

### Critical Flaw 3: Absence of Statistical Variance Reporting
Given the micro-scale of the datasets (256 test samples per task), the performance is highly susceptible to data-subset bias and random seed variations. However, the paper reports single set of exact percentages (e.g., 63.87%) without standard deviations, confidence intervals, or evaluations across multiple random seeds and splits. For a practitioner trying to deploy such a system, knowing the stability, robustness, and variance of the performance is a crucial requirement.

---

## 4. Minor Comments & Suggestions
*   **Representation Shift in Calibration:** During `initialize_sld_basis`, the routing bases $\Phi_k$ are calibrated using the model's forward pass. Because the forward pass does not pass routing coefficients, the `MergedLinear` modules default to `'uniform'` mode. Thus, the activation centroids are computed from a model that is uniformly merged, rather than the base model or the individual expert models. It would be valuable to discuss this potential representation shift and its impact on routing quality.
*   **Routing Jitter Analysis:** It would be helpful to include quantitative statistics showing the frequency of "routing jitter" (where a sample gets routed to different experts across blocks 9, 10, and 11), particularly for ambiguous/boundary inputs.
*   **Limitations & Scalability:** Discuss how the method scales as the number of tasks $K$ grows very large (e.g., $K > 50$). Storing $K$ low-rank adapters, although compressed, will still scale linearly, and routing accuracy might degrade as the embedding space becomes crowded.

---

## 5. Overall Recommendation
**Score: 3 (Weak Reject)**

**Reasoning:**
The core concepts of SLD-Merge—offline SVD decomposition of expert task vectors, bounded cosine-similarity routing, and sample-level parallel low-rank forward passes—are excellent, highly pragmatic, and address a major, neglected bottleneck in dynamic model merging (batch-dependency and heterogeneity collapse). However, the complete mismatch between the paper's claims and the actual codebase regarding "Autonomous Classification Head Selection" represents a critical methodological and scientific integrity flaw. All evaluations are currently reliant on a privileged oracle to select the classification head, making the system undeployable in a real-world stateless stream in its current state. Additionally, the micro-scale of the datasets and the under-trained SVHN expert ceiling (29.30%) weaken the empirical claims. I encourage the authors to implement and evaluate the autonomous classification head selection in their codebase and run evaluations on a larger, more realistic dataset scale with proper statistical variance. Once these issues are addressed, this has the potential to be a strong, highly impactful paper.

---

## 6. Questions for the Authors
1. **How is the Autonomous Classification Head Selection implemented in your codebase?** Since the provided evaluation script `run_experiments.py` always passes the ground-truth `task_labels` directly to choose the active classification head, can you provide the exact code or script that generated the "Autonomous" results in Section 4.4?
2. **How does SLD-Merge perform on fully trained experts?** Since the SVHN standalone expert ceiling is only 29.30%, can you evaluate the framework using experts trained to convergence on the full datasets to confirm that SVD and cosine routing scale to high-quality representations?
3. **Can you report standard deviations across multiple random splits or seeds?** Given the small size of the test subsets (256 samples), reporting statistical variance is essential to confirm that the +8.50% improvement is statistically significant.
