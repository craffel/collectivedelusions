# Paper Summary: Hybrid-Router

## 1. Objective and Problem Statement
Deploying multiple task-specific fine-tuned models in production is highly impractical due to linear scaling of storage, VRAM, and computation. **Parameter-space model merging** offers a zero-overhead alternative (e.g., Task Arithmetic, TIES-Merging, AdaMerging), but static merging suffers from catastrophic domain sacrifice and representation conflicts when combining highly divergent tasks (e.g., MNIST digits vs. CIFAR-10 natural objects).

To bypass this conflict, recent work has explored **dynamic, test-time model merging** (routing), which dynamically scales and reconstructs weights on-the-fly based on input batch styles. However, on-the-fly reconstruction of full model weights (which often consist of tens or hundreds of millions of parameters) introduces prohibitive memory-bandwidth and computational latency during inference, rendering it impractical for real-world deployments on edge devices or low-latency pipelines.

## 2. Proposed Method: Hybrid-Router
The paper presents **Hybrid-Router**, a low-latency hybrid dynamic model merging framework that partitions the model layer-wise to address the runtime assembly bottleneck:
1. **Early layers ($1 \dots L-k$)** are task-agnostic feature extractors (capturing edges, shapes, basic styles) and are **statically merged offline** (using uniform blending or AdaMerging), incurring absolutely zero test-time overhead.
2. **Late layers (final $k$ layers)** are task-specific and are **dynamically routed and merged on-the-fly** at test-time based on the input style.

### Key Components:
- **$H_0$ Style-based Routing:** Extracts features very early in the forward pass by average-pooling token embeddings directly from the initial Patch Embedding layer ($H_0$). This allows feature extraction and weight reconstruction to run in parallel with early-layer execution, avoiding GPU synchronization blocks.
- **Routing Heads:**
  - *Standard Softmax-Router:* Competitive, normalized zero-sum scaling ($\sum_k \alpha_k = 1$) for peak raw classification accuracy.
  - *BSigmoid-Router:* An exploratory, Softmax-free activation engine using independent Sigmoids capped at $\lambda_{\text{max}} = 0.3$ to study the dynamics of uncoupled, parallel task activations.
  - *BL-Router:* Bounded Softmax router (capped at $\lambda_{\text{max}} = 0.3$) acting as a bridge to isolate the impact of scaling bounds vs. activation functions.
- **Dynamic Batch Filtering (DBF):** To resolve "Batch Style Blur" under large, heterogeneous batch sizes (where batch-averaged coefficients collapse to uniform weights), DBF dynamically clusters incoming batches into style-homogeneous sub-batches, performs distinct weight reconstructions for each sub-batch, and passes them through.

## 3. Experimental Setup & Benchmarks
- **Parameter-Space Representation Sandbox:** A PyTorch-based emulator modeling a 14-layer ViT-Tiny ($L=14$ layer groups, $D=192$) across high-conflict domains (MNIST, FashionMNIST, CIFAR-10, SVHN).
- **Physical Validation:** End-to-end model merging experiments in PyTorch using a shallow convolutional neural network (SimpleCNN, ~25k parameters, 4 layer groups) trained and evaluated on real MNIST, FashionMNIST, CIFAR-10, and SVHN pixels.
- **Data Regime:** Calibration of routing parameters using a highly-constrained split of only **64 samples** (16 per task) over 3 independent seeds.

## 4. Main Results and Achievements
1. **Pareto Frontier and Resource Savings:**
   - In the ViT-Tiny sandbox, at $k=4$, Hybrid-Router achieves a joint mean accuracy of **76.75%** (a massive **+4.44%** improvement over SOTA static AdaMerging of 72.31%) while cutting weight reconstruction latency and VRAM storage overhead by **71.3%** and **71.4%** respectively.
2. **Structural Regularization (Overfitting-Optimizer Paradox):**
   - Freezing early layers restricts the routing optimizer's search space under scarce data regimes (64 samples). At $k=12$, this structural regularization yields **84.79%** joint mean accuracy, outperforming fully dynamic routing ($k=14$ at **84.57%**) by **+0.22%** while saving **14.3%** in latency.
3. **Resolving the Softmax-Sigmoid Gap:**
   - Under standard scaling constraints ($\lambda_{\text{max}} = 0.3$), BSigmoid-Router achieves **84.57%** (a ~1.2% gap behind BL-Router's **85.76%**). When scaled up to match the classical router's bound ($\lambda_{\text{max}} = 1.2$), BSigmoid-Router jumps to **94.93%**, proving that independent sigmoids are highly competitive with Softmax.
4. **Streaming Robustness & DBF Success:**
   - Under batch size $B=1$, BSigmoid-Router (Reg) maintains **84.55%** stream accuracy.
   - At larger, highly shuffled batch sizes (e.g., $B=256$), standard routing collapses due to Batch Style Blur. Activating **Dynamic Batch Filtering (DBF)** recovers performance: BSigmoid-Router climbs from **66.63%** to **83.18%** (+16.55% absolute gain), and Linear Router climbs from **63.54%** to **93.77%** (+30.23% absolute gain).
5. **Physical CNN Validation:**
   - Confirms that the entire ensembling framework is end-to-end differentiable and physically realizable. A SimpleCNN sweep over $k \in \{0, 1, 2, 3, 4\}$ shows a monotonically increasing Pareto curve, with $k=4$ reaching **76.67 $\pm$ 0.94%**. DBF is physically validated, showing massive absolute accuracy gains of **+27.59%** ($B=16$) and **+30.56%** ($B=64$) under shuffled streams.
