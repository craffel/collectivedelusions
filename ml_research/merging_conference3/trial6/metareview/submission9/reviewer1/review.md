# Peer Review: A Simulation Analysis of Spatial Cross-Attention Routing for Dynamic Model Merging

## 1. Summary of the Paper
This paper proposes **Cross-Attention Multi-Expert Routing (CAM-Router)**, a dynamic weight-space model merging framework designed for Vision Transformers (ViTs). Weight-space model merging is a promising paradigm for combining specialized fine-tuned models into a single set of weights, thereby preserving task capabilities without parameter or inference compute overhead. While existing dynamic model merging methods rely on global average pooling of intermediate tokens before routing—which discards spatial information and makes them vulnerable to spatial occlusions—CAM-Router retains the full spatial token sequences. It introduces $K$ trainable task-expert query embeddings to dynamically attend to localized token regions via Multi-Head Cross-Attention (MHCA). Additionally, it replaces competitive Softmax constraints with independent Bounded Sigmoid activations, and proposes **Decoupled Historical Gating (DHG)** to mitigate "heterogeneity collapse" in large mixed-task batches. The authors evaluate CAM-Router on a custom 14-layer compact Vision Transformer "coordinate sandbox" across four toy datasets (MNIST, FashionMNIST, CIFAR-10, SVHN) and report that it outperforms average-pooling-based classical and quantum dynamic routers.

---

## 2. Strengths and Weaknesses

### Strengths
- **Insightful Problem Formulation:** The paper correctly identifies a critical flaw in existing dynamic routers: global average pooling collapses spatial resolution, making the routing decision vulnerable to spatial occlusions and mixed-task batching. Focusing on retaining spatial tokens is a strong and intuitive starting point.
- **Detailed Structure and Outline:** The paper is structurally well-organized, with clear mathematical formulations of the cross-attention gating and independent activation mechanisms.
- **Thorough Parameter Sweeps:** The authors include extensive multi-dimensional parameter sweeps (attention heads, occlusion ratios, batch sizes, query initializations, and regularization), which show how the system behaves under different simulated conditions.

### Weaknesses

#### A. Extreme Architectural & Inference Complexity (The "Minimalist" Critique)
The primary appeal of parameter-space model merging is its simplicity: it aims to deliver multi-task capabilities with **zero** additional parameters, **zero** inference latency, and **zero** modifications to the underlying backbone's forwarding pipeline. CAM-Router completely compromises this elegance by introducing a heavy, highly engineered routing system:
1. **Heavy Parameter Overhead:** It adds trainable task-expert queries, query/key/value/output projections, and linear routing classifiers (~0.15M parameters). While the authors describe this as "lightweight" (2.61% of a ViT-Tiny backbone), it violates the parameter-free nature of model merging.
2. **Inference Latency & Memory Bandwidth Bottlenecks:** To compile the model weights on-the-fly, the system must execute cross-attention over all spatial patches, calculate scaling coefficients, and then physically sum huge weight tensors ($W_{base} + \sum \alpha_k V_k$) across layers 2 to 14 *during the forward pass*. Naive eager implementation causes severe latency. The authors propose caching or custom Triton kernels, but these are merely "conceptual" and remain unimplemented.
3. **Ad-Hoc "First-Block Paradox" Solution:** Because the routing coefficients must be predicted before running the layers, the authors run the patch embedding and first transformer block using frozen, base weights. This hybrid, ad-hoc design breaks the clean, uniform structure of standard model merging and could lead to feature misalignment between the static block 1 and the dynamically merged layers 2-14.

#### B. Serious Methodological Flaw: Non-Deterministic, Stateful Inference (DHG)
To handle mixed-task batches without "heterogeneity collapse" (a problem created by sample-dependent routing), the authors introduce **Decoupled Historical Gating (DHG)**. DHG computes per-sample coefficients and maintains an Exponential Moving Average (EMA) of these coefficients over a sliding historical window. 
This is a **major technical flaw**: keeping a historical EMA of merging coefficients means that the weights of the model (and thus its predictions) are **stateful and depend directly on the sequence of preceding inputs**. This violates the standard independent and identically distributed (i.e.d.) assumption of classification inference. A single image $x$ will produce different classifications depending on what images were processed before it in the inference stream. This non-determinism is completely unacceptable for standard production deployments, safety-critical applications, or rigorous benchmarking.

#### C. Severe Numerical Discrepancies and Inconsistencies
The paper contains glaring, highly unprofessional discrepancies in its reported performance numbers. The Abstract reports significantly higher numbers than those actually presented in the main text and tables:
- **Joint Mean Accuracy:** The Abstract claims **57.07%** Joint Mean Accuracy (representing a **+15.10%** improvement over Static Uniform and outperforming QWS-Merge by **+32.17%**). However, Table 1 and Section 4.2 report **53.07%** Joint Mean Accuracy (representing a **+11.10%** improvement and outperforming QWS-Merge by **+28.17%**).
- **Spatial Occlusion Robustness:** The Abstract claims a stable accuracy of **53.63%** under up to 80% patch masking. However, Table 3 and Section 4.3 report **50.57%** accuracy at 80% masking.
- **Batch Heterogeneity Resilience:** The Abstract claims a stable accuracy of **55.47%** at batch size $B=256$. However, Table 4 and Section 4.3 report **54.30%** accuracy at $B=256$.
These numerical mismatches suggest a severe lack of thoroughness in writing or, worse, selective reporting from a different/fabricated run, which severely compromises the paper's scientific credibility.

#### D. Weak Experimental Evaluation and Scope
- **Toy simulated sandbox:** The experiments are not run on standard, large-scale benchmarks (such as CLIP or LLaMA) but are restricted to a custom "14-layer compact Vision Transformer coordinate sandbox" evaluated on toy image datasets (MNIST, FashionMNIST, CIFAR-10, SVHN). It is unclear how these simulated results translate to standard foundation models.
- **Missing Core Baselines:** Table 1 compares only against global pooling dynamic routers, but completely omits standard **static merging** baselines like Ties-Merging, Task Arithmetic, or DARE.
- **Overfitting on Tiny Calibration Set:** Training 148,996 parameters on a tiny calibration set of 800 samples makes the model highly vulnerable to overfitting. This is evident in Table 7 (Sweep 5), where applying standard $L_2$ weight decay ($\lambda_{wd} = 10^{-3}$) causes CAM-Router's performance to drop sharply from **53.07%** to **47.40%**. The model is highly unstable and fragile under standard regularization.

---

## 3. Questions and Suggestions for the Authors
1. **Explain the Numerical Discrepancies:** Please explain why the Abstract reports a Joint Mean Accuracy of 57.07%, while Table 1 reports 53.07% (and similar mismatches for occlusion and batch sizes). Which numbers are correct and how did these discrepancies occur?
2. **Address non-determinism in DHG:** How do you justify the stateful, non-deterministic nature of Decoupled Historical Gating in a standard classification pipeline? If a safety-critical system receives a sequence of benign images followed by an anomalous one, will the historical EMA bias the model weights and cause a misclassification? Why not use a stateless, single-sample batching setup for parallel batched inference?
3. **Compare against Static Merging:** Why are standard static merging baselines like Ties-Merging, Task Arithmetic, or DARE missing from the main results in Table 1?
4. **Scale to Real-World Benchmarks:** Can you evaluate CAM-Router on standard vision-language models (e.g., CLIP-ViT-B/16 on ImageNet/COCO) or large language models to demonstrate that this cross-attention mechanism scales beyond a simulated toy sandbox?
5. **Report Actual Physical Latency:** Please provide physical latency and memory throughput measurements (in milliseconds and GB/s) comparing CAM-Router (using eager PyTorch) against Static Uniform and standard single-model inference. Without implementing the custom Triton kernels, what is the actual computational slowdown?

---

## 4. Detailed Ratings

### Soundness: Poor
The paper has major conceptual and technical flaws. Decoupled Historical Gating (DHG) introduces stateful, non-deterministic inference where predictions depend on preceding inputs, breaking i.i.d. classification. Additionally, the ad-hoc hybrid architecture (static block 1, dynamic layers 2-L) and eager summation latency are major practical bottlenecks. The unregularized model shows extreme sensitivity to standard regularization, indicating severe overfitting on the tiny 800-sample calibration set. Finally, the experiments are restricted to a custom toy simulated sandbox.

### Presentation: Poor
While the paper is mathematically clear, the presence of glaring, systematic numerical discrepancies between the Abstract and the rest of the text/tables (e.g., reporting 57.07% vs 53.07% Joint Mean Accuracy) represents a critical presentation failure that severely compromises scholarly standards.

### Significance: Poor
Parameter-space model merging is valued for its zero-latency, stateless, parameter-free deployment. By introducing heavy cross-attention projection layers, on-the-fly model compilation, and non-deterministic historical gating, CAM-Router eliminates these core benefits. Because of these severe overheads and safety issues, ML practitioners are highly unlikely to adopt this method over static merging or standard task-adapters (LoRA). Its significance is further limited by its restriction to a toy simulated sandbox.

### Originality: Fair
The concept of using spatial cross-attention for weight routing is moderately interesting and novel in the narrow domain of dynamic model merging. However, the implementation is highly over-engineered, relying on standard deep learning blocks (MHCA, EMAs) to patch over issues created by the dynamic routing formulation itself.

---

## 5. Overall Recommendation
**Recommendation: 2: Reject**

**Justification:** While the paper addresses an interesting and real limitation of average-pooling in dynamic model merging, the proposed solution (CAM-Router) is mathematically over-engineered and introduces critical technical flaws:
1. **Glaring Numerical Discrepancies:** The abstract systematically overstates the model's accuracy, occlusion robustness, and batch resilience compared to the actual experimental results reported in the tables and text.
2. **Non-Deterministic, Stateful Inference:** The DHG mechanism relies on a historical EMA that makes the model's classification output dependent on preceding inputs, breaking standard i.i.d. assumptions and making the model unsafe for real-world deployment.
3. **Over-Engineering & High Latency:** The model introduces significant parameter and computational complexity (MHCA, on-the-fly weight summation, hybrid static-dynamic architecture) which defeats the primary "zero-overhead" advantage of weight-space model merging, without delivering performance close to the individual experts.

Due to these severe presentation, technical, and methodological flaws, this paper falls far short of the bar for acceptance and is recommended for rejection.
