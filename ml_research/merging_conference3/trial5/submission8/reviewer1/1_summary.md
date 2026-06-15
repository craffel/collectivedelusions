# Summary of EpiMerge

## 1. Main Topic
The paper addresses the challenge of **multi-task model merging**, where multiple task-specific expert neural networks (sharing a common pre-trained base initialization) are combined into a single cohesive model without undergoing expensive multi-task joint retraining. The paper targets the fundamental dichotomy of existing model-merging techniques:
- **Static merging methods** (such as Task Arithmetic, TIES-Merging, RegMean, DARE, etc.) force a single, fixed set of weights for all test-time inputs. This triggers catastrophic representation conflicts and representation collapse when experts have orthogonal or opposing gradients.
- **Dynamic merging methods** (like AdaMerging) adapt ensembling coefficients at test-time based on test inputs. However, they either rely on local test-time adaptation (TTA) via unsupervised entropy minimization—which collapses under temporal shifts or small-batch noise due to transductive overfitting—or they employ routing networks (like QWS-Merge) that average ensembling coefficients across the batch dimension to maintain GPU concurrency. This batch-averaged shortcut couples independent inferences inside a batch, introducing transductive dependencies and triggering "heterogeneity collapse" under mixed-task batches.

To resolve this, the paper proposes **EpiMerge (Epigenetic Weight Masking)**, a biologically-inspired framework that enables true sample-specific dynamic model merging in a single, parallel forward pass without batch-averaging shortcuts.

---

## 2. Approach
Inspired by cellular molecular biology, where the static DNA sequence (analogous to the static base model) is dynamically modulated by epigenetic row and column chemical markers in response to environmental stimuli, EpiMerge uses the input sample itself to dynamically modulate pre-trained expert parameters.

The core architectural and mathematical components of the EpiMerge framework are:
1. **Global Input Representation Extraction:** 
   An input sample $x$ is passed through a frozen, duplicate copy of the pre-trained base model, acting as a **Deep Semantic Sensory Extractor** ($\mathcal{M}_{base}$), to retrieve a highly contextualized and semantically rich representation $\mathbf{z}(x)_b \in \mathbb{R}^D$ from the final transformer block. This is projected to a compact latent space $\mathbf{h}(x)_b \in \mathbb{R}^d$ (with $d = K$ tasks) via a frozen random projection matrix $P$.
2. **Low-Rank Epigenetic Mask Generation:**
   For each task-specific expert $k$ at each layer $l$, the framework introduces small, trainable epigenetic reader tensors:
   $$U_k^{(l)} \in \mathbb{R}^{D_{out} \times R \times d} \quad \text{and} \quad V_k^{(l)} \in \mathbb{R}^{D_{in} \times R \times d}$$
   These tensors act as **Epigenetic Reader Heads (ERHs)**. Given the global representational state $\mathbf{h}(x)_b$, they generate row-gating masks $\mathbf{r}_{k, b, r}^{(l)}(x)$ and column-gating masks $\mathbf{c}_{k, b, r}^{(l)}(x)$ via element-wise Sigmoids. By taking the sum of the outer products of these masks across all rank components $r \in \{1, \dots, R\}$, they construct a coordinate-wise gating matrix $G_{k, b}^{(l)}(x)$:
   $$G_{k, b}^{(l)}(x) = \sum_{r=1}^R \mathbf{r}_{k, b, r}^{(l)}(x) \otimes \mathbf{c}_{k, b, r}^{(l)}(x)$$
3. **Sample-Specific Weight Reconstruction:**
   Using the generated row and column gating masks, the framework scales each expert's task vector $T_k^{(l)} = W_k^{(l)} - W_{base}^{(l)}$ and blends them with the pre-trained base weight $W_{base}^{(l)}$ to reconstruct a unique, sample-specific weight matrix:
   $$W_{merged, b}^{(l)}(x) = W_{base}^{(l)} + \sum_{k=1}^K \sum_{r=1}^R \left( \mathbf{r}_{k, b, r}^{(l)}(x) \otimes \mathbf{c}_{k, b, r}^{(l)}(x) \right) \odot T_k^{(l)}$$
4. **Vectorized Parallel Forward Pass:**
   To execute sample-specific weight reconstruction in parallel without losing GPU concurrency, EpiMerge formulates the forward pass as a vectorized tensor contraction using PyTorch's `torch.einsum`:
   $$\Delta W = \text{\texttt{torch.einsum}}('\text{kbor},\text{kbir},\text{koi}\rightarrow\text{boi}', R, C, T)$$
   $$W_{merged} = W_{base} + \Delta W$$
   $$Y = \text{\texttt{torch.einsum}}('\text{bni},\text{boi}\rightarrow\text{bno}', X, W_{merged}) + \mathbf{b}^{(l)}$$
5. **Active-Early Sensory Extraction (Lightweight Variant):**
   To bypass the $2.0\times$ static parameter footprint and the extra forward pass of the frozen duplicate base model, the paper proposes **EpiMerge-Active**. It partitions the $L$ blocks of the active model into an early static stage ($L_{early}=4$) and a deep dynamic stage. The input batch is passed through the early static blocks, average pooled to obtain $\mathbf{z}_{early}(x)_b$, and projected to obtain $\mathbf{h}(x)_b$. Only the deep stage blocks are dynamically modulated, reducing parameters to **exactly 1.0x** and running in a single forward pass.

---

## 3. Key Findings
- **Online TTA Fragility:** Unsupervised test-time adaptation (AdaMerging) collapses under temporal non-I.I.D. shifts (Bursty stream) and small-batch regimes ($B=2$), losing over 40% accuracy compared to its static ceiling, due to transductive overfitting on local batch noise.
- **Transductive Batch Coupling in Prior Routers:** Prior dynamic routers (like QWS-Merge) average ensembling coefficients across the batch, mathematically coupling independent inferences, which violates the I.I.D. assumption and constitutes a major transductive hazard.
- **EpiMerge Stream Consistency:** By maintaining true sample-wise independent inference, EpiMerge's performance is mathematically guaranteed to remain perfectly consistent across Shuffled I.I.D., Bursty temporal clustering, and Small-Batch streams ($39.30\%$ accuracy for Rank-2), avoiding transductive local batch dependencies.
- **Gating Rank Trade-off & Rank-4 Paradox:** Scaling the coordinate gating rank from $R=2$ to $R=4$ doubles the learnable ERH parameters, which drastically complicates the optimization landscape. Under a constrained 64-sample calibration budget, this leads to severe underfitting/degradation (collapsing from $39.30\%$ to $31.05\%$).
- **Supervised Static Paradox & Data Scaling:** At a tiny budget of 64 samples, the static supervised baseline OFS-Tune ($41.48\% \pm 3.18\%$) outperforms EpiMerge ($39.30\% \pm 1.81\%$) due to its low-dimensional search space acting as a strong regularizer. However, as the calibration dataset size scales to 512 samples, EpiMerge's accuracy surges by +23.85% absolute, reaching $61.45\%$ and virtually closing the gap with OFS-Tune ($61.92\%$).

---

## 4. Explicitly Claimed Contributions (with Evidence)
1. **Concept of Epigenetic Weight Masking (EpiMerge):** Reversible input-dependent scaling of pre-trained expert parameters mimicking molecular biological regulation.
   * *Evidence:* Formulated mathematically in Sections 3.3 and 3.4, and illustrated in Figure 1.
2. **Low-Rank Row-Column Dual Gating (with rank $R \ge 1$):** Highly parameter-efficient gating that controls individual coordinates of the task vectors.
   * *Evidence:* Formulation in Section 3.3 and analysis of gating rank trade-offs in Section 4.3 (Table 1, showing Rank-1, Rank-2, Rank-4 configurations).
3. **Parallel, Vectorized Forward Pass via `torch.einsum`:** True sample-wise weight reconstruction and inference in parallel across a mixed batch.
   * *Evidence:* Formulation in Section 3.5, mathematical proof of sample-wise independence in Appendix A, and GPU latency/memory profiling in Section 4.4 (Table 5).
4. **Lightweight Active-Early Sensory Extraction (EpiMerge-Active):** A parameter-free and single-pass variant that reduces parameters to exactly 1.0x and eliminates the second forward pass.
   * *Evidence:* Formulation in Section 3.7, evaluation in Section 4.2 (Table 1, showing $36.70\%$ accuracy), and sensitivity analysis of partition boundary $L_{early}$ in Section 4.4 (Table 4).
5. **Exhaustive Empirical Evaluation and Ablations:** Demonstrating consistency across streams, analyzing training steps, dataset scaling, and learning rate schedulers.
   * *Evidence:* Quantitative multi-task results in Table 1; Ablation Study A (Table 2); Ablation Study B (Table 3); Resource profiling in Table 5; Routing dynamics in Table 6.
