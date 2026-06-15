# Peer Review

## Summary of the Paper
The paper proposes **EpiMerge** (**Epigenetic Weight Masking**), a biologically-inspired model-merging framework designed to combine multiple task-specific expert neural networks into a single multi-task model without retraining. To escape the parameter conflicts of static merging and the transductive batch coupling of existing dynamic routers, EpiMerge performs sample-specific weight reconstruction. 

Specifically, EpiMerge uses an unmodified copy of the pre-trained base model as a **Deep Semantic Sensory Extractor** to generate global representational vectors for each input. It feeds these vectors into **Epigenetic Reader Heads (ERHs)** to compute row-wise and column-wise gating masks, which are combined via a low-rank outer product ($G = \sum \mathbf{r} \otimes \mathbf{c}$) to dynamically scale expert task vectors. The forward pass is parallelized across batches using PyTorch's `torch.einsum` to ensure sample-wise independent inference. The authors also present **EpiMerge-Active**, a lightweight variant that extracts representations from early layers of the active model, reducing the footprint to 1.0x parameters. They evaluate their method using a Vision Transformer (ViT-Tiny) backbone across four tasks (MNIST, FashionMNIST, CIFAR-10, SVHN) under three test-time stream configurations.

---

## Strengths and Weaknesses

### Strengths
1. **Compelling Conceptual Narrative:** Framing parameter-space coordinate ensembling as "epigenetics" is highly creative, engaging, and biologically intuitive. The mapping of row/column scaling coordinates to molecular marks provides an elegant and coherent narrative.
2. **Mathematical and Structural Rigor:** The mathematical formulation of the low-rank row-column dual gating mechanism ($G = \sum \mathbf{r} \otimes \mathbf{c}$) and its integration into the PyTorch tensor contraction framework (`torch.einsum`) is exceptionally clear, precise, and formal. Stating the exact index notation is very helpful.
3. **True Sample-Wise Parallelization:** The paper addresses a major systems bottleneck of prior dynamic routers (batch-averaging and serialization) and provides a technically sound, concurrent forward pass formulation that guarantees sample-wise independent inference.
4. **Transparent Systems-Level Profiling:** The authors conduct thorough latency and GPU memory profiling across different batch sizes and configurations, providing a highly transparent and detailed mapping of the physical systems-level trade-offs (Table 4).
5. **Thorough Ablations on Overfitting and Scaling:** The sweeping of training steps (Table 2) and calibration dataset sizes (Table 3) provides useful scientific insights into the optimization limits and transductive overfitting dynamics of high-dimensional coordinate gating.

### Weaknesses (Critical Empirical & Methodological Concerns)
1. **Severe Empirical Inconsistency (Shifting Baseline Discrepancy):** 
   There is a major, unexplained discrepancy in baseline performance between the main results (Table 1) and the dataset scaling ablation studies (Table 3) for the *exact same model configurations*:
   - In **Table 1**, the static supervised baseline **OFS-Tune** is reported to achieve **41.48% $\pm$ 3.18%** accuracy on the Shuffled I.I.D. stream under the default 64-sample calibration dataset.
   - In **Table 3 (Ablation B1)**, the same **OFS-Tune** under the "64 samples (16/task)" configuration is reported to achieve **53.23% $\pm$ 0.05%** accuracy.
   - This represents a massive **11.75% absolute discrepancy** in accuracy for the exact same baseline under the exact same calibration size.
   - Similarly, **EpiMerge-Rank2** achieves **39.30% $\pm$ 1.81%** in Table 1 but is reported as **37.60% $\pm$ 1.82%** under the corresponding 64-sample setup in Table 3.
   - Shifting baseline and proposed method numbers across tables severely undermines the empirical rigor of the paper and suggests a lack of standardization in the experimental pipeline.
2. **The Dynamic Flatness Paradox (Under-optimized Gating):**
   Looking at Table 5 (Routing Dynamics Analysis), the learned row-gating intensities for EpiMerge are exceptionally flat, hovering tightly between **0.498** and **0.516** (a variation of only $\pm 0.01$ around 0.50).
   - This indicates that the "dynamic" model is practically static. The Epigenetic Reader Heads are failing to learn active, distinctive, and sample-sensitive gating boundaries, and have instead converged to a task-independent, flat compromise.
   - If the gating masks are virtually constant across all inputs, the entire machinery of sensory feature extraction, projection, and sample-specific weight reconstruction is redundant. A simple static merged model can achieve the same (or better) performance with zero latency and memory overhead. This strongly undermines the core claim of "true, sample-wise dynamic merging."
3. **Underperformance of proposed dynamic method vs. simpler static baseline:**
   The proposed dynamic coordinate gating model (EpiMerge-Rank2 at 39.30% in Table 1) is consistently outperformed by the simpler supervised static baseline OFS-Tune (41.48% in Table 1) under the default 64-sample calibration budget. Furthermore, Table 3 shows that even when the calibration dataset scales to 512 samples, the simpler static baseline OFS-Tune (61.92%) *still* outperforms the highly complex dynamic EpiMerge (61.45%).
   - In machine learning, a proposed complex dynamic method must justify its added systems complexity (which includes a 2.0x parameter footprint, dynamic activation memory, and a 3x increase in inference latency) by showing superior performance over simpler, standard baselines. Because EpiMerge consistently underperforms the static baseline OFS-Tune across *all* dataset sizes, there is no empirical justification for deploying the proposed dynamic method. The added architectural complexity does not translate to superior model accuracy.
4. **Baseline Tuning Fairness (The AdaMerging "Strawman"):**
   In Table 1, AdaMerging (Online TTA) achieves only **12.25%** accuracy on Shuffled I.I.D. and **12.15%** on Bursty streams.
   - In a 4-task classification setup where a "Task-Conditioning Oracle" routes samples to task-specific heads, each head has 10 classes, meaning a random guess gets 10% accuracy. AdaMerging, which is a highly competitive, peer-reviewed test-time adaptation framework, is performing barely better than a random guess (12.25%). This suggests that either the authors' implementation of AdaMerging has a severe integration bug, or the hyperparameters were completely un-tuned. Comparing a proposed method against a poorly implemented "strawman" baseline is methodologically unsound.
5. **Omission of Standard Conflict-Resolving Baselines:**
   The authors compare against Uniform Merging (Task Arithmetic) and OFS-Tune. However, they omit the most widely used and standard static model-merging baselines that resolve parameter conflicts, such as **TIES-Merging** and **DARE**. Since parameter conflict resolution is a central theme of this paper, omitting TIES-Merging and DARE is a significant gap in the evaluation.
6. **Lack of Per-Task Accuracy Breakdown:**
   The paper only reports average multi-task accuracy. It is critical to see individual accuracies for MNIST, FashionMNIST, CIFAR-10, and SVHN. CIFAR-10 and SVHN are significantly more challenging than MNIST/FashionMNIST. Reporting only the average hides whether the model is completely failing on the harder datasets (which is likely, given the low average accuracy of ~39%).
7. **Scale of Evaluation:**
   The evaluation is restricted to a very small model, **Vision Transformer (ViT-Tiny)** with 5.7M parameters, across four simple, low-resolution classification tasks (**MNIST, FashionMNIST, CIFAR-10, SVHN**). It is unclear whether this biologically-mimicking paradigm can scale to larger, more realistic architectures (e.g., ViT-Base or LLMs) and diverse real-world tasks.

---

## Ratings and Justifications

### Soundness
*   **Rating: Fair**
*   **Justification:** While the mathematical derivations and systems parallelization formulations are technically sound and elegant, the overall soundness of the empirical methodology is compromised by: (1) severe, unexplained data discrepancies in baseline performance between Table 1 and Table 3; (2) very flat routing intensities (Table 5) indicating that the gating mechanism is virtually static and under-optimized; (3) underperforming a simpler static baseline (OFS-Tune) across all Few-Shot calibration budgets, making the added systems complexity hard to justify; and (4) the potential under-tuning of key baselines (e.g., AdaMerging performing at near-random accuracy).

### Presentation
*   **Rating: Excellent**
*   **Justification:** The paper is exceptionally well-written, clearly structured, and easy to follow. The transition from molecular biology to weight-space ensembling is highly coherent. Figures and tables are well-placed, captioned thoroughly, and present rich quantitative data. The mathematical notations are precise, and standard ML terminology is adhered to.

### Significance
*   **Rating: Fair**
*   **Justification:** The practical significance of the proposed method is currently limited. The proposed dynamic method consistently underperforms a simpler static baseline (OFS-Tune) across all calibration sizes, while adding significant systems complexity (3x latency, 2x parameter memory, and $O(B \cdot D_{out} \cdot D_{in})$ weight reconstruction memory). Furthermore, the evaluation is restricted to toy classification datasets and a tiny backbone, making its applicability to modern production-scale foundational architectures unproven.

### Originality
*   **Rating: Good**
*   **Justification:** Framing parameter-space coordinate ensembling as "epigenetic weight masking" is highly creative. The low-rank row-column dual gating mechanism is a clever and parameter-efficient formulation of coordinate gating. However, mathematically, it is a specific low-rank coordinate ensembling instance built entirely from established deep learning primitives (Sigmoids, low-rank outer products, frozen base networks, and `torch.einsum`), representing an evolutionary rather than revolutionary step.

---

## Overall Recommendation
*   **Rating: 3: Weak Reject**
*   **Justification:** 
    The paper has clear merits: a compelling biologically-inspired conceptual narrative, high-quality writing, rigorous mathematical formulations of the low-rank outer product gates, and detailed systems-level profiling. 
    
    However, the empirical weaknesses currently outweigh the merits. Crucially, there is a major, unexplained discrepancy in the static baseline's accuracy between Table 1 (41.48%) and Table 3 (53.23%) for the exact same 64-sample setup, which must be resolved to ensure empirical integrity. Methodologically, the learned routing gates are virtually flat (Table 5), demonstrating that the complex dynamic ensembling pipeline is practically acting as a static model. This is reflected in the accuracy tables, where the proposed complex dynamic method is consistently outperformed by a simpler, standard static baseline (OFS-Tune) across all calibration sizes, despite adding a 2.0x parameter footprint and tripling the inference latency. Furthermore, the extremely poor performance of AdaMerging (~12%) suggests that baselines were not properly tuned.
    
    The paper requires revisions to resolve these data discrepancies, properly tune and expand the baselines (including TIES-Merging and DARE), provide a per-task accuracy breakdown, and address or mitigate the under-optimization of the routing gates before it can be accepted.

---

## Questions and Suggestions for the Authors
1. **Explain Shifting Baselines:** Please explain the massive 11.75% absolute accuracy discrepancy for OFS-Tune (64 samples) between Table 1 (41.48% $\pm$ 3.18%) and Table 3 (53.23% $\pm$ 0.05%). Why do the baseline numbers shift so drastically? Similarly, why does EpiMerge (64 samples) shift from 39.30% to 37.60%?
2. **Address Gating Flatness Paradox:** Why are the learned row-gating intensities in Table 5 so flat, ranging tightly between 0.498 and 0.516? If the dynamic gating masks are virtually constant across inputs, how do you justify the extra 3x latency and 2x parameter memory overhead of sensory extraction and weight reconstruction? Have you tried applying specialized regularizers or learning rate schedules to encourage more distinct, active routing gates?
3. **Justify Dynamic Complexity:** Given that the simpler static baseline (OFS-Tune) consistently outperforms EpiMerge across all calibration budgets (64 to 512 samples) and has zero parameter memory, zero dynamic weight memory, and runs 3 times faster, what is the practical, empirical justification for deploying EpiMerge?
4. **Tune AdaMerging Fairly:** Why does AdaMerging perform so poorly (~12%)? In a 10-class task routing oracle setup, 12% is barely above random guessing. Please verify your implementation and hyperparameters (e.g., learning rate, steps, entropy loss weight) for AdaMerging to ensure it is not a "strawman."
5. **Incorporate Standard Conflict-Resolving Baselines:** Please include standard static model-merging baselines that resolve parameter conflicts, such as **TIES-Merging** and **DARE**, to make the evaluation comprehensive.
6. **Provide Per-Task Accuracy Breakdown:** Please provide a table showing the individual accuracies on MNIST, FashionMNIST, CIFAR-10, and SVHN for the main methods in Table 1 to verify that the model is actually learning multi-task representations rather than collapsing on the harder datasets.
7. **Scale Up Evaluation:** To demonstrate the scalability of the epigenetic weight-masking paradigm, please provide a small-scale experiment on a larger backbone (e.g., ViT-Base) or a natural language model (e.g., a 1B LLM merged across tasks).
