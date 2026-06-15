# Peer Review of Conference Submission

**Title:** Sparse Low-Rank Dynamic Merging: Enabling Batch-Independent and Parameter-Efficient Multi-Task Inference

---

## 1. Summary of the Paper
The paper addresses a critical deployment bottleneck in dynamic weight-space model merging, termed **batch-dependency** and **heterogeneity collapse**. Traditional dynamic model merging methods average routing coefficients across the batch dimension to reconstruct a single set of merged weights. This violates the standard sample-level I.I.D. assumption (as a sample's prediction shifts depending on co-packaged samples) and degrades performance when processing heterogeneous, mixed-task inference streams.

To resolve this, the paper proposes **Sparse Low-Rank Dynamic Merging (SLD-Merge)**, which shifts dynamic adaptation from the parameter space to a sample-wise activation space. Offline, it computes specialized task vectors for experts ($V_k^{(l)} = W_k^{(l)} - W_{base}^{(l)}$) and performs Singular Value Decomposition (SVD) on them, truncating them to a low rank $r$ (e.g., $r=8$) to construct lightweight low-rank adapters ($B_k^{(l)}, A_k^{(l)}$). At inference time, a bounded cosine-similarity router extracts global representation vectors from input activations and uses Top-1 hard gating to route each sample completely independently through only its most relevant adapter path. 

The authors also introduce **Activation-Space Mean Initialization** to initialize the routing basis vectors to the empirical mean activations of each task on a tiny, unlabeled calibration set, achieving high-quality zero-shot routing without backpropagation calibration. Evaluating across MNIST, FashionMNIST, CIFAR-10, and SVHN streams on a Vision Transformer (ViT-Tiny) backbone, SLD-Merge maintains a stable joint accuracy of **$63.87\%$** across all batch sizes, outperforming weight-averaging dynamic baselines, and reducing storage overhead by **$92.5\%$** while adding only **$8.3\%$** FLOPs.

---

## 2. Strengths and Weaknesses

### Strengths
1. **Thorough and Comprehensive Ablations:** The paper features an exceptionally detailed ablation study exploring key operational aspects, including sensitivity to SVD rank, zero-shot vs. optimized routing via Straight-Through Estimators (STE), SVD truncation error isolation, autonomous vs. oracle classification head selection, routing jitter across layers, and statistical stream sequence variance.
2. **On-Device Hardware Profiling:** Unlike many model merging papers that rely solely on theoretical FLOP counts, the authors perform physical execution profiling on a Raspberry Pi 4 edge computer. They measure actual wall-clock latency (showing an $85.2\%$ latency reduction compared to weight-reconstruction baselines) and peak RAM utilization.
3. **Excellent Writing and Presentation:** The paper is highly polished, well-structured, and clearly written. The mathematical formulations are clean and correct, and the visualizations are of professional quality.

### Weaknesses
1. **Conceptual Mischaracterization (Not Model Merging):** In actual weight-space model merging, the objective is to fuse multiple specialized expert models into a *single set of unified weights* $W_{merged}$ that can process all inputs. Because the weights are physically merged, a single forward pass $X W_{merged}$ handles all inputs, avoiding the need to store separate expert models in memory. 
   SLD-Merge completely departs from this paradigm. It keeps $K$ separate sets of low-rank adapters in memory and uses a router to dynamically choose which of the separate adapters to execute. This is **not model merging**; it is a **multi-LoRA Mixture-of-Experts (MoE)** or a multi-adapter routing framework (similar to *LoRA-Hub*, *LoRA-MoE*, or *LLaVA-MoE*). Claiming SLD-Merge "resolves" weight-space merging bottlenecks is conceptually unfair, as it simply bypasses the core challenge by keeping the task parameter pathways separate.
2. **The PyTorch Parallel Forward Pass Complexity Bottleneck:** In Section 3.3, the parallel forward pass is implemented as:
   $$Y = X W_{base}^{(l)} + \sum_{k=1}^K \alpha_k \odot \left( (X A_k^{(l)}) B_k^{(l)} \right)$$
   This formulation reveals a critical scalability flaw: **all $K$ low-rank adapters are fully executed** for every batch, and the outputs of inactive adapters are simply multiplied by zero. Consequently, the computational complexity scales **linearly with the number of tasks $K$**. While the reported computational overhead is a low $+8.3\%$ FLOPs, this is only because the evaluation uses a tiny task suite of $K=4$. If scaled to $K=50$ or $K=100$ tasks, the computational overhead of running all adapter paths in parallel will be massive, completely defeating the on-device "computationally lean" deployment goals. In contrast, actual weight merging has a forward pass complexity completely independent of $K$.
3. **Unrealistic "Toy" Evaluation Setup (Disjoint Domains & Low-Shot Experts):** 
   - The paper evaluates on a joint stream of **MNIST, FashionMNIST, CIFAR-10, and SVHN**. These represent four completely disjoint and highly distinct visual domains (handwritten digits, clothing, natural objects, street numbers) whose representations are highly orthogonal in the latent space of a ViT. Distinguishing them with a simple cosine router is trivial, which artificially inflates routing accuracy ($93.26\%$ to $\approx 100\%$) and masks routing jitter.
   - The training set is restricted to a mere **256 samples per dataset**, resulting in severely under-trained experts (e.g., the SVHN expert's accuracy is only $29.30\%$). Merging under-trained and overfitted experts is highly non-standard and leads to unstable representation spaces that are not representative of real-world model merging.
4. **Suspicious "SVD Regularization" Claim:** The authors claim that rank-16 SLD-Merge outperforms the full-rank baseline by $+1.38\%$ because SVD acts as an "implicit regularizer" that filters out noise. This is highly suspicious and is likely a classic case of framing a training bug (under-training on 256 samples) as a feature. If the experts were properly trained to convergence on standard-scale datasets with proper regularization, SVD truncation would almost certainly *degrade* performance compared to full-rank. 
5. **Missing Multi-Adapter and MoE Baselines:** Because the framework is structurally a multi-adapter Mixture-of-Experts, the authors must compare SLD-Merge against standard multi-LoRA routing and MoE baselines (such as *LoRA-Hub*, *LoRA-MoE*, or *LLaVA-MoE*), rather than comparing only to weight-space model merging methods that are bound by completely different architectural constraints.

---

## 3. Detailed Evaluations

### Soundness: Fair
The framework contains critical technical and mathematical limitations that are overlooked:
- **Linear Scaling with $K$:** The vectorized parallel forward pass executes all $K$ paths, leading to $O(K)$ computational scaling.
- **Lack of Activation-Awareness in SVD:** Performing SVD on raw weight task vectors without considering the activation distribution is mathematically suboptimal. Modern post-training low-rank compression methods (e.g., *SVD-LLM*, *ASVD*) show that scaling weights by the activation covariance before SVD is required to preserve out-of-distribution performance.
- **Representation Shift during Calibration:** The routing bases $\Phi_k^{(l)}$ are calibrated on uniform weight activations but evaluated on sparse low-rank activations, introducing a representation shift that is not theoretically bounded or mathematically analyzed.

### Presentation: Good
The paper is exceptionally well-written, with high-quality tables, figures, and notation. However, the presentation is undermined by a significant conceptual misalignment, framing a multi-adapter routing framework as a "model merging" method.

### Significance: Fair
While the practical edge hardware profiling on a Raspberry Pi 4 is highly commendable, the overall significance of the work is limited by the artificial "toy-scale" setup (256-sample dataset, disjoint domains) and the severe unaddressed computational complexity scaling bottleneck with respect to $K$.

### Originality: Fair
The core components of the proposed method—SVD-based weight decomposition, centroid-based activation routing, and multi-LoRA routing—are all well-established techniques in the literature. Combining them in this manner is a pragmatic engineering effort, but the scientific originality is highly incremental and overstated due to the reframing of MoE as "model merging."

---

## 4. Overall Recommendation
**Recommendation: 3: Weak reject**

**Justification:** 
The paper has clear engineering merits, including excellent presentation, physical edge hardware execution profiling, and highly thorough ablation studies. However, the fundamental weaknesses overall outweigh these merits. 

Specifically:
1. The framework is conceptually misaligned; it is an activation-space Mixture-of-Experts/multi-LoRA model rather than a weight-space model merging method. Bypassing weight fusion by keeping $K$ separate adapter pathways in memory avoids "heterogeneity collapse" but defeats the purpose of parameter merging.
2. The PyTorch implementation of the parallel forward pass executes all $K$ adapters, creating a severe $O(K)$ computational bottleneck that prevents scaling to large numbers of tasks.
3. The empirical evaluation is conducted on a non-standard "toy" setup (256-sample datasets on completely disjoint domains with highly under-trained experts) which acts as a major confounding variable and inflates the router's performance.

Before this work can be built upon meaningfully by others, the authors must:
- Resolve the parallel execution bottleneck by implementing a truly sparse forward pass (e.g., via conditional execution/scatter-gather).
- Reframer the paper to correctly position it within the multi-LoRA and MoE literature, toning down the weight-merging claims.
- Validate their findings on standard-scale, fully converged experts to prove that the SVD regularization benefits and zero-shot routing hold in realistic deployment scenarios.

---

## 5. Questions and Constructive Feedback for the Authors

1. **On Parallel Forward Pass Complexity:** Your parallel PyTorch forward pass executes $(X A_k^{(l)}) B_k^{(l)}$ for all $K$ tasks and then multiplies by $\alpha_k$. For $K=4$, the FLOP overhead is $8.3\%$. What is the FLOP and latency overhead when $K=50$ or $K=100$? To make this truly scalable, have you considered implementing a sparse routing forward pass using PyTorch scatter/gather operations to group samples by their active expert, thereby running only a single adapter path per sample?
2. **On SVD Regularization and Converged Experts:** Your rank-16 model outperforms the full-rank model by $+1.38\%$, which you attribute to SVD filtering out noise. If you train the experts to full convergence on the entire MNIST, FashionMNIST, CIFAR-10, and SVHN datasets (where noise/overfitting is minimized), does this performance advantage persist, or does the SVD low-rank model perform strictly worse than or equal to the full-rank expert ceiling?
3. **On Baseline Comparisons:** Since SLD-Merge maintains separate low-rank adapter pathways for each expert and routes samples to them, why did you not compare against established multi-adapter and MoE routing methods like *LoRA-Hub*, *LoRA-MoE*, or *LLaVA-MoE*? Such comparisons are essential to establish what benefits your specific SVD-based and centroid-based routing design provides over standard multi-adapter frameworks.
4. **On Activation-Aware SVD:** Have you considered incorporating activation-covariance scaling into your offline SVD phase (similar to *ASVD* or *SVD-LLM*) to ensure that the low-rank projection is mathematically optimized for the features that actually flow through the network?
