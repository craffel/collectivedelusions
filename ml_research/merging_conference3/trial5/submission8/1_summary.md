# Paper Summary: EpiMerge

## 1. Overview and Core Proposal
"EpiMerge: Epigenetic Weight Masking for True Sample-Wise Dynamic Model Merging" proposes a biologically-inspired model-merging framework designed to synthesize multiple task-specific expert neural networks into a single multi-task model without retraining. 

Instead of searching for a static weight compromise (as in traditional static merging like Task Arithmetic or TIES-Merging) or employing batch-averaged dynamic routers (like QWS-Merge) that couple sample inferences, EpiMerge performs **true sample-wise dynamic merging**. It dynamically modulates pre-trained expert parameters for each individual input sample in parallel, drawing inspiration from cellular epigenetics (where gene expression is dynamically scaled in response to environmental stimuli without altering the underlying static DNA).

## 2. Key Components
The framework consists of the following key architectural and mathematical components:
*   **Semantic Sensory Extractor:** To obtain high-level semantic feedback, EpiMerge passes the input through a frozen, unmodified copy of the base model ($\mathcal{M}_{base}$). It average-pools the final-layer embeddings and projects them to a compact, low-dimensional latent space ($d = K$ tasks) via a frozen random projection matrix. An alternative **Active-Early Sensory Extraction** variant is proposed to eliminate the duplicate sensory model by extracting activations directly from the early layers of the active model.
*   **Epigenetic Reader Heads (ERHs):** Trainable modules introduced at each layer group. Given the global representational state, they generate coordinate-wise row gating masks ($\mathbf{r} \in \mathbb{R}^{D_{out}}$) and column gating masks ($\mathbf{c} \in \mathbb{R}^{D_{in}}$) via Sigmoid activations.
*   **Low-Rank Row-Column Dual Gating:** Gating is parameterized as a low-rank outer product of row and column masks (generalized to rank $R \ge 1$), which is highly parameter-efficient ($O(R \cdot d \cdot (D_{out} + D_{in}))$ compared to $O(D_{out} \times D_{in})$ for a full-rank mask).
*   **Parallel Vectorized Weight Reconstruction:** Utilizing PyTorch's `torch.einsum`, EpiMerge reconstructs and scales sample-specific weight matrices in parallel across a mixed batch. This completely eliminates batch-averaging shortcuts and maintains perfect sample-wise inference independence.
*   **Offline Calibration:** The tiny fraction of ERH parameters ($<0.1\%$) is calibrated offline on a compact 64-sample dataset (16 samples per task) for 100 steps using backpropagation, requiring zero online test-time adaptation.

## 3. Main Results and Insights
*   **Evaluation Setup:** The paper evaluates EpiMerge on a Vision Transformer backbone (`vit_tiny_patch16_224`) across four image classification tasks (MNIST, FashionMNIST, CIFAR-10, SVHN) under three target stream configurations: Shuffled I.I.D., Bursty temporal task shifts, and Small Batch ($B=2$).
*   **Empirical Performance:**
    *   EpiMerge-Rank1 ($39.22\%$) and EpiMerge-Rank2 ($39.30\%$) consistently outperform Uniform Merging ($19.05\%$) and coarse-grained dynamic routers (Linear Router at $34.95\%$ and QWS-Merge at $34.85\%$).
    *   Under the default 64-sample budget, static supervised merging (**OFS-Tune**) outperforms EpiMerge by $2.18\%$ absolute ($41.48\%$), exposing an expressivity-optimization trade-off in the high-dimensional search space of coordinate gating.
    *   However, when the calibration dataset size is expanded to **256 samples**, EpiMerge-Rank2 achieves **51.40%** accuracy, completely dominating OFS-Tune (+10% absolute). Scaling to **512 samples** surges EpiMerge to **61.45%** accuracy (+19.97% absolute over OFS-Tune).
    *   Because EpiMerge maintains sample-wise independent inference, its accuracy remains perfectly consistent across Shuffled, Bursty, and Small Batch streams, unlike online test-time adaptation (AdaMerging) which collapses to $11.85\%$ due to local transductive overfitting.
*   **Physical Trade-offs:** Latency and GPU memory profiling confirm that reconstructing weight tensors on-the-fly and running the duplicate sensory extractor triples wall-clock latency (from 9.12ms to 27.34ms at $B=64$) and increases peak memory by +22.8% (+144.05MB at $B=64$). A Dynamic LoRA-style EpiMerge is outlined to eliminate this memory bottleneck for large foundation models.
