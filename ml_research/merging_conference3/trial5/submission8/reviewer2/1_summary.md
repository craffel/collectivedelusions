# 1. Summary of the Paper

## Main Topic and Motivation
The paper addresses the challenge of **model merging**, which aims to synthesize multiple specialized neural network expert models (fine-tuned from a shared pre-trained base initialization) into a single multi-task model without retraining. The authors identify a fundamental limitation in existing methods, dividing them into:
1. **Static Merging:** Creating a single, fixed compromise weight matrix (e.g., Task Arithmetic, TIES-Merging), which causes catastrophic parameter conflicts when tasks have divergent or orthogonal gradients.
2. **Dynamic Merging & Test-Time Adaptation:** Dynamically adjusting weights at inference time based on inputs. However, existing methods either overfit to local test streams (e.g., AdaMerging utilizing unsupervised test-time optimization) or perform batch-averaged ensembling (e.g., QWS-Merge), which mathematically couples unrelated samples within a batch and violates sample-wise inference independence.

To overcome these, the paper proposes **EpiMerge**, a biologically-inspired dynamic merging framework that achieves true, decoupled sample-wise weight modulation in parallel.

## Proposed Approach (EpiMerge)
EpiMerge uses the input sample itself as an environmental stimulus to scale the parameters of pre-trained task experts on-the-fly. The core components of the framework are:
1. **Semantic Sensory Extractor:** Projects the input sample $x$ through a frozen copy of the pre-trained base model $\mathcal{M}_{base}$ to extract a contextualized global latent representation $\mathbf{z}(x)_b \in \mathbb{R}^D$ ($D=192$ for ViT-Tiny), which is then projected to a compact latent space $\mathbf{h}(x)_b \in \mathbb{R}^d$.
2. **Low-Rank Epigenetic Reader Heads (ERHs):** Small trainable tensors $U_k$ and $V_k$ generate sample-specific, rank-specific row-wise and column-wise gating masks ($\mathbf{r}$ and $\mathbf{c}$) via Sigmoid activations. Taking the sum of outer products across rank components $r \in \{1,\dots,R\}$ constructs a coordinate-wise gating mask:
   $$G_{k, b}(x) = \sum_{r=1}^R \mathbf{r}_{k, b, r}(x) \otimes \mathbf{c}_{k, b, r}(x)$$
3. **Sample-Specific Weight Reconstruction:** The gating masks dynamically scale the task-specific expert vectors $T_k$, yielding a unique weight matrix for each sample:
   $$W_{merged, b} = W_{base} + \sum_{k=1}^K \sum_{r=1}^R \left( \mathbf{r}_{k, b, r}(x) \otimes \mathbf{c}_{k, b, r}(x) \right) \odot T_k$$
4. **Vectorized Parallel Forward Pass:** To maintain GPU concurrency, the forward pass is formulated as a vectorized tensor contraction using `torch.einsum` (e.g., `torch.einsum('kbor,kbir,koi->boi', R, C, T)`). This processes an entire batch of mixed tasks in parallel while ensuring each sample is transformed by its own independent, custom weight state.
5. **EpiMerge-Active (Lightweight Variant):** To avoid the 2.0x parameter memory overhead of keeping a frozen copy of $\mathcal{M}_{base}$ as a sensory extractor, this variant statically merges early layers of the active model and pools their intermediate representations to serve as the guiding signals, reducing the footprint to 1.0x parameters and 1.0x latency.

## Key Findings & Results
The authors evaluate EpiMerge against five baselines using a Vision Transformer (ViT-Tiny) backbone across four tasks (MNIST, FashionMNIST, CIFAR-10, SVHN) under three stream configurations (Shuffled I.I.D., Bursty, and Small Batch Size $B=2$).
- **AdaMerging Collapse:** Online test-time adaptation (AdaMerging) collapses to ~12% accuracy under temporal burstiness and small-batch noise due to unsupervised entropy minimization overfitting.
- **EpiMerge Performance:** EpiMerge-Rank2 achieves **39.30%** accuracy across all three target streams, showing absolute mathematical consistency because of sample-wise inference independence. This outperforms Uniform Merging (+20.25% absolute), QWS-Merge (+4.45% absolute), and Linear Router (+4.35% absolute).
- **The Supervised Static Paradox:** The proposed dynamic method (39.30%) is outperformed by a bug-free static supervised baseline, **OFS-Tune** (**41.48%**), under the default 64-sample calibration budget. The authors attribute this to the high-dimensional non-convex search space of coordinate gating under extremely low data constraints. When calibration data is scaled to 512 samples, EpiMerge accuracy rises to **61.45%** (nearing OFS-Tune's 61.92%).

## Explicitly Claimed Contributions
1. **Epigenetic Weight Masking:** A biologically-inspired model-merging framework enabling true sample-specific dynamic model merging.
2. **Low-Rank Row-Column Dual Gating:** Fine-grained coordinate-wise scaling of expert task vectors, generalized to arbitrary gating ranks $R \ge 1$.
3. **Vectorized Parallel Forward Pass:** A `torch.einsum`-based implementation for GPU concurrency and sample independence, along with the `EpiMerge-Active` 1.0x parameter footprint variant.
4. **Exhaustive Empirical Evaluation:** Demonstrating perfect stream consistency under non-I.I.D. shifts and small-batch noise, and characterizing optimization, dataset scaling, and system resource trade-offs.
