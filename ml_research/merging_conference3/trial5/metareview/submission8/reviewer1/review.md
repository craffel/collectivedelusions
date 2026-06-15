# Peer Review of Conference Submission: EpiMerge

## Summary of the Paper
This paper addresses the fundamental challenge of **multi-task model merging**, where specialized, fine-tuned expert models sharing a common base initialization are synthesized into a single cohesive network without retraining. The authors target a major limitation in current methods: static merging (such as Task Arithmetic or TIES-Merging) forces a fixed set of weights for all test-time inputs, triggering catastrophic parameter interference; whereas existing dynamic merging approaches either suffer from test-time adaptation (TTA) fragility under temporal shifts (such as AdaMerging) or rely on batch-averaged dynamic routing shortcuts (such as QWS-Merge) that couple independent inferences inside a batch, leading to "heterogeneity collapse."

Inspired by cellular molecular biology, the authors propose **EpiMerge (Epigenetic Weight Masking)**. EpiMerge utilizes the input sample itself to dynamically modulate the expression of pre-trained expert parameters. For each expert, highly parameter-efficient **Epigenetic Reader Heads (ERHs)** generate row-wise and column-wise gating masks via low-rank outer products ($G = \sum_{r=1}^R \mathbf{r} \otimes \mathbf{c}$). These masks scale the coordinate-wise task vectors on-the-fly, reconstructing a unique, sample-specific weight matrix. Crucially, using vectorized tensor contractions (`torch.einsum`), EpiMerge executes true sample-specific parallel inference across a batch, bypassing batch-averaging shortcuts and preserving sample independence. To address resource overheads, the paper also introduces **EpiMerge-Active**, which partitions the active network so that early static layers serve as the sensory extractor, reducing parameters to exactly 1.0x.

The framework is evaluated using a Vision Transformer backbone across four classification benchmarks (MNIST, FashionMNIST, CIFAR-10, SVHN) under three test-time streams: Shuffled I.I.D., Bursty temporal clusters, and Small Batch Size ($B=2$). The results demonstrate that EpiMerge maintains perfect consistency across all streams due to its sample-wise independent inference. The authors also conduct extensive ablations on gating rank, calibration steps, and dataset size, and provide detailed profiling of GPU memory and wall-clock latency.

---

## Overall Recommendation
* **Rating:** 4 (Weak Accept)
* **Justification:** EpiMerge is a technically solid, creative, and highly original paper that introduces a novel, biologically-inspired paradigm for dynamic model merging. Its core strengths lie in its elegant formulation of low-rank coordinate gating and its vectorized parallel execution via tensor contractions, which mathematically guarantee sample-wise inference independence and stream robustness. The authors are highly commended for their exceptional scientific honesty, detailing optimization bottlenecks (the Rank-4 Degradation and the Supervised Static Paradox), resource overheads (3x latency), and evaluation limitations (the Task-Conditioning Oracle). 
However, the paper's immediate impact is somewhat limited by the toy scale of the vision benchmarks (e.g., MNIST, SVHN) and the fact that the dynamic coordinate gating is highly prone to underfitting under small data budgets, consistently underperforming the simpler, static supervised baseline (OFS-Tune). Furthermore, there are several gaps in situating the work within foundational and concurrent literature on parameter-space ensembling and PEFT routing. 
Overall, this is a highly promising work that the machine learning community is likely to build on (especially for scaling to large language models or complex domains), making it a valuable addition to the conference.

---

## Strengths
1. **Conceptual Originality:** The biological metaphor linking cellular epigenetics to weight-space multi-task model merging is a highly refreshing, engaging, and creative contribution.
2. **True Sample-Specific Parallel Inference:** By formulating the forward pass as a vectorized tensor contraction using PyTorch's `torch.einsum`, EpiMerge performs true, decoupled, sample-wise weight reconstruction and inference in parallel, completely bypassing the batch-averaged ensembling shortcuts of prior dynamic routers and preserving sample independence.
3. **Stream Robustness and Consistency:** Both mathematically and empirically, the paper demonstrates that sample-wise independent inference guarantees perfect robustness to non-I.I.D. temporal task drifts (Bursty stream) and extreme small-batch noise ($B=2$).
4. **Systems Practicality (EpiMerge-Active):** The introduction of the "Active-Early Sensory Extraction" variant is an elegant and appropriate architectural solution. It successfully partitions the active model to serve as its own sensory extractor, slashes the static parameter footprint to exactly 1.0x, and eliminates the second forward pass.
5. **Outstanding Transparency and Scientific Honesty:** The authors exhibit exemplary academic integrity by thoroughly documenting and analyzing their framework's limitations, including:
   - The **Rank-4 Degradation Paradox** (optimization challenges of high-rank gates on limited calibration data).
   - The **Supervised Static Paradox** (underperforming the static OFS-Tune baseline under constrained budgets).
   - Systems-level profiling of **GPU Latency and Memory** (documenting the 3x latency cost).
   - The **Task-Conditioning Oracle** limitation and providing concrete architectural pathways to transition to a non-oracle, fully autonomous deployment.

---

## Weaknesses

### 1. Gaps in Literature Contextualization and Situating
While the paper does a decent job citing general PEFT and static model merging, it misses critical connections to foundational weight-space ensembling and concurrent dynamic PEFT routing literature:
* **The Diagonal Fisher Merging Connection:** The authors propose low-rank dual gating to perform fine-grained coordinate-wise parameter gating. From a scholarly perspective, this is the dynamic, input-dependent analogue of **Fisher Merging** (*"Merging Models with Fisher Information"*, Matena & Raffel, 2022). Fisher Merging is a foundational static merging technique that uses diagonal Fisher information matrices of specialized experts to perform coordinate-wise importance-weighted parameter fusion. Synthesizing this connection would significantly elevate the paper's theoretical framework.
* **Citing Model Patching:** The paper omits **Model Patching** (*"Patching open-vocabulary models by interpolating weights"*, Ilharco et al., 2022), which is an important precursor to task arithmetic and model merging.
* **Citing Concurrent PEFT Adapter Routing:** While traditional token-level MoE is discussed, the authors should cite and contrast their work against recent frameworks that dynamically route or fuse specialized PEFT adapters, such as **LoRA Hub** (*"LoRA Hub: Efficient Cross-Task Generalization via Dynamic Adapter Fusion"*, Huang et al., 2023) and **ZipLoRA** (*"ZipLoRA: Any-Subject Zero-Shot Image Generation via Low-Rank Adapter Merging"*, Shah et al., 2023).

### 2. Underperforming Simpler, Zero-Overhead Static Baselines
A major weakness is that the highly complex, computationally heavy coordinate-gating of EpiMerge consistently underperforms the simpler static supervised baseline, **OFS-Tune** (Table 1: OFS-Tune at 41.48% vs. EpiMerge-Rank2 at 39.30%). Even when the calibration dataset size is scaled up to 512 samples in Ablation B (Table 3), the static OFS-Tune baseline still maintains an absolute performance advantage ($61.92\%$ vs. $61.45\%$), although the gap narrows. This reveals that the high-dimensional expressive capacity of coordinate gating introduces a difficult, non-convex optimization landscape that is very prone to underfitting when calibration data is scarce. In practice, a zero-overhead static model that only optimizes 48 layer-wise scalars remains a more robust and superior choice unless fine-grained sample-wise dynamic adaptability is strictly required.

### 3. Limited Evaluation Scale and Realism
The empirical evaluation is restricted to a ViT-Tiny backbone (5.7M parameters) on relatively simple, toy image classification benchmarks (MNIST, FashionMNIST, CIFAR-10, SVHN). Grayscale datasets like MNIST/FashionMNIST have low-dimensional representation manifolds, which may simplify the ensembling task. Modern model-merging research is heavily focused on much larger models (e.g., LLaMA, Mistral, Stable Diffusion) on complex generative, reasoning, or multi-modal tasks. While the authors outline a highly compelling "Dynamic LoRA-Style EpiMerge" formulation for LLMs, they do not provide empirical validation at this scale.

### 4. Systems Serialization Bottleneck
The proposal to utilize final-layer (Layer 12) semantic representations from the frozen sensory extractor to guide early-layer Epigenetic Reader Heads (ERHs) creates a severe **systems serialization bottleneck**. Because the early layers of the active model cannot begin calculation until the final layer of the sensory model completes, the entire forward pass of the sensory model must execute sequentially before the active model starts. This backward dependency prevents pipeline parallelism or concurrent block execution on the GPU, directly contributing to the **3x latency increase** (Table 5).

---

## Detailed Ratings

### Soundness: Good
The mathematical and architectural formulation of EpiMerge is highly rigorous and technically sound. The use of PyTorch's vectorized `torch.einsum` to perform true sample-wise dynamic weight reconstruction in parallel is a mathematically sound way to preserve both high GPU tensor core utilization and sample-wise inference independence. The proof in Appendix A is elegant and robustly supports the stream-consistency claims. 
However, the soundness rating is capped at "good" because the dynamic model underperforms the simpler static baseline (OFS-Tune) across all data scales, reflecting a critical optimization/underfitting bottleneck that is not fully solved, and because the evaluation relies on an unrealistic Task-Conditioning Oracle.

### Presentation: Excellent
The paper is exceptionally well-structured, engaging, and clear. Every equation and coordinate-wise tensor operation is precisely defined. The systems-level code snippets bridge the gap between abstract mathematics and actual code. Figure 1 is a beautiful, high-signal illustration of the epigenetic metaphor. Tables 1 through 6 are meticulously organized, presenting high-density quantitative findings. The writing style is professional, scholarly, and extremely transparent about its limitations and physical trade-offs.

### Significance: Good
By introducing true, decoupled, sample-wise parameter scaling in parallel, the paper addresses a critical transductive hazard in prior dynamic routers and challenges the static-compromise mindset of model merging. It paves an exciting path toward self-organizing and resilient neural networks.
However, the significance of the current results is somewhat constrained by the toy vision benchmarks and the 3x systems-level latency overhead, which might limit immediate adoption in latency-critical production environments unless the proposed LoRA-style or lightweight sensory variants are fully realized.

### Originality: Excellent
The conceptual originality of mapping biological epigenetic regulation mechanisms (such as DNA methylation and histone modifications) to deep weight spaces is outstanding. Furthermore, combining low-rank row-column dual gating with vectorized tensor contractions is a highly creative and methodologically novel way to achieve fine-grained, coordinate-wise dynamic gating in parallel without parameter explosion.

---

## Constructive Questions and Recommendations for the Authors

1. **Incorporate Foundational Literature:** Please explicitly discuss and draw connections to **Fisher Merging** (Matena & Raffel, 2022). Emphasize that diagonal Fisher Merging is the static, coordinate-wise importance-weighted analogue of EpiMerge's dynamic, input-dependent coordinate gating. Also, cite and discuss **Model Patching** (Ilharco et al., 2022) and dynamic PEFT fusion frameworks like **LoRA Hub** (Huang et al., 2023) and **ZipLoRA** (Shah et al., 2023) in the Related Work section.
2. **Empirical Validation of Non-Oracle Pathways:** Since the Task-Conditioning Oracle is an unrealistic assumption for production deployment, it would greatly strengthen the paper to implement and evaluate either the **Integrated Task Classifier** or the **Shared Unified Multi-Task Head** proposed in Section 4.5. Even a preliminary experiment evaluating these non-oracle pathways on the 64-sample dataset would significantly elevate the paper's practical significance.
3. **Address the Systems Serialization Bottleneck:** The backward dependency (using Layer 12 representation to guide Layer 1 ERHs) sequentializes the execution of the sensory and active backbones. Have you explored asynchronous gating (predicting gating masks based on early sensory layers) or dual-stream pipelining to mitigate the 3x latency overhead?
4. **Scale Beyond Toy Datasets:** While the ViT-Tiny vision experiments serve as a strong proof-of-concept, the value of model merging lies in scaling to large foundation models. In future iterations, we highly recommend empirically validating the proposed **Dynamic LoRA-Style EpiMerge** formulation on a medium-sized Transformer (e.g., LLaMA-1B or Mistral-7B) to demonstrate its systems-level feasibility and scalability.
