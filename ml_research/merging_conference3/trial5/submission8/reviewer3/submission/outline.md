# Paper Outline: EpiMerge: Epigenetic Weight Masking for True Sample-Wise Dynamic Model Merging

## 1. Title & Abstract
- **Title**: EpiMerge: Epigenetic Weight Masking for True Sample-Wise Dynamic Model Merging
- **Abstract**:
  - *Context*: Model merging is a powerful paradigm to combine multiple task-specific expert neural networks without retraining.
  - *The Limitation*: Current merging methods are either static (forcing a fixed weight compromise across all tasks) or dynamically route at the batch level (leading to "heterogeneity collapse" when a batch contains mixed-task samples, or failing under small-batch test-time streams).
  - *Our Paradigm Shift*: Inspired by cellular epigenetics—where the physical DNA (static weights) remains constant but gene expression is dynamically and reversibly modulated by row/column-wise epigenetic markers in response to the cellular environment (individual input samples).
  - *Methodology*: We present EpiMerge. It introduces highly parameter-efficient Epigenetic Reader Heads (ERHs) at each layer group. ERHs project a global, unit-sphere-normalized latent input representation into dynamic, coordinate-wise row and column scaling masks via low-rank outer products. This allows true parallel sample-wise weight contraction in a single parallelized `torch.einsum` forward pass, completely bypassing batch dependencies.
  - *Results*: Evaluated across four diverse vision tasks (MNIST, FashionMNIST, CIFAR-10, SVHN) using a ViT-Tiny backbone. EpiMerge exhibits total immunity to target stream non-I.I.D. temporal clustering (Bursty) and extreme small-batch noise ($B=2$), outperforming or matching dynamic baselines while resolving the representation conflict and "heterogeneity collapse" of prior works with zero test-time computational overhead.

## 2. Introduction
- **The Static Merging Compromise**: Explain how standard model merging (e.g., Task Arithmetic, TIES) attempts to find a static weight configuration. This is inherently a zero-sum compromise. If tasks have conflicting gradients or parameters, static average results in severe performance trade-offs.
- **The Failure of Dynamic Merging (Heterogeneity Collapse)**: Discuss recent quantum-inspired or routing-based dynamic merging (e.g., QWS-Merge). Highlight their fatal flaw: to process batches efficiently, they average routing coefficients across the batch dimension. This causes a "wavefunction collapse" or "heterogeneity collapse" when batches are heterogeneous (mixed tasks) or extremely small, reducing them to static compromises.
- **The Epigenetic Inspiration**: Introducing the core metaphor and philosophy of **The Visionary** persona. In biology, cell differentiation and environmental adaptation do not rewrite the underlying DNA. Instead, they utilize epigenetic chemical markers (methylation, histone modifications) to selectively scale or silence expression pathways of specific genes. We translate this molecular mechanism to deep neural network parameter space.
- **Contributions**:
  1. We propose EpiMerge, the first *true* sample-wise model merging framework that adapts weights dynamically and independently for every individual sample in parallel.
  2. We introduce the Low-Rank Row-Column Dual Gating mechanism to construct sample-specific weight matrices without parameter explosion.
  3. We design a parallel vectorized forward pass using tensor contraction (`torch.einsum`) to maintain standard I.I.D. batched inference.
  4. We demonstrate empirical robustness across multiple test stream distributions (Shuffled IID, Bursty temporal shift, and Small-batch noise), showing complete immunity to heterogeneity collapse.

## 3. Related Work
- **Static Model Merging**: Task Arithmetic, TIES-Merging, RegMean, AdaMerging (static optimization). Emphasize that these force a single static compromise.
- **Dynamic Model Merging & Mixture of Experts (MoE)**: Traditional MoE routes tokens to discrete experts, which is computationally expensive and hard to train. QWS-Merge and related work route at the batch level, causing collapse in mixed-task batches.
- **Parameter-Efficient Tuning & Weight Editing**: LoRA, AdaLoRA, etc. Contrast EpiMerge's sample-wise dynamic scaling of *existing* task vectors with static parameter-efficient tuning.
- **Biologically-Inspired Architectures**: Review how neural networks draw from biology (synapses, dendrites), and position epigenetics as the natural next frontier for parameter modulation.

## 4. Methodology (The EpiMerge Framework)
- **Problem Setting**: Merging $K$ expert networks $\{W_k^{(l)}\}$ with a shared pre-trained base model $W_{base}^{(l)}$.
- **Global Input Representation Extraction**:
  - Spatial pooling of patch tokens to get a global feature vector $z(x)_b$.
  - Random frozen projection $P$ onto a low-dimensional unit sphere to get $\psi(x)_b$.
- **Low-Rank Epigenetic Mask Generation**:
  - Row masks $\mathbf{r}_{k, b}^{(l)}(x) = \text{Sigmoid}( U_k^{(l)} \psi(x)_b )$ and column masks $\mathbf{c}_{k, b}^{(l)}(x) = \text{Sigmoid}( V_k^{(l)} \psi(x)_b )$. Explain how the outer product $\mathbf{r} \otimes \mathbf{c}$ forms a full weight-shaped gate matrix, preserving rich coordinate-wise control with negligible parameter overhead.
- **Sample-Specific Weight Reconstruction**:
  - $W_{merged, b}^{(l)}(x) = W_{base}^{(l)} + \sum_{k=1}^K (\mathbf{r}_{k, b}^{(l)}(x) \otimes \mathbf{c}_{k, b}^{(l)}(x)) \odot (W_k^{(l)} - W_{base}^{(l)})$.
- **Vectorized Parallel Forward Pass**:
  - The beautiful mathematical trick: standard linear layers process $X \in \mathbb{R}^{B \times N \times D_{in}}$ with a single shared $W$. EpiMerge uses `torch.einsum('bni,boi->bno', X, W_merged)` to run different weight matrices for each sample in the batch *simultaneously* and *in parallel* on the GPU, achieving true sample-wise merging without serialization.
- **Calibration Phase (Optimization)**:
  - We optimize the tiny reader matrices $\{U_k^{(l)}, V_k^{(l)}\}$ offline using a 64-sample stratified calibration dataset via Adam.

## 5. Experimental Evaluation
- **Experimental Setup**:
  - Models: ViT-Tiny (`vit_tiny_patch16_224`).
  - Datasets: MNIST, FashionMNIST, CIFAR-10, SVHN.
  - Calibration: 64 samples (16 per task) optimized for 100 steps.
- **Target Stream Configurations**:
  1. Shuffled IID Stream.
  2. Bursty Stream (temporal task shift).
  3. Small Batch Size Stream ($B=2$).
- **Baselines**:
  - Uniform Merging (Task Arithmetic)
  - AdaMerging (Online unsupervised TTA)
  - OFS-Tune (Supervised static layer-wise coefficients)
  - Linear Router (Classical dynamic task routing)
  - QWS-Merge (Quantum-inspired batch-averaged routing)
- **Main Quantitative Results Table**: Presentation of the classification accuracies.
- **Analysis and Discussion**:
  - *The Fragility of Online TTA*: Why AdaMerging collapses under Bursty and small batch streams (local batch-noise overfitting).
  - *The Batch-Averaging Compromise of Prior Dynamic Merging*: Show how prior methods collapse when mixed-task batches occur, whereas EpiMerge remains completely stable.
  - *Parametric and Computational Efficiency*: Discuss the minimal overhead of the ERH modules.

## 6. Conclusion & Future Outlook
- Recap of the epigenetic model-merging paradigm.
- Strong visionary perspective on future avenues: dynamic weight modulation for scaling to hundreds of tasks, brain-like lifelong learning, and open-ended evolutionary weight adaptation.
