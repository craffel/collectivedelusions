# Idea Proposal: Low-Latency Hybrid Dynamic Merging (Hybrid-Router)

## 1. Persona Alignment
*The Pragmatist* focuses on real-world deployment constraints, inference latency, memory footprint, and robust applicability. While dynamic parameter-space ensembling (such as BSigmoid-Router) can achieve high joint accuracy, performing dynamic weight reconstruction (i.e., high-dimensional parameter-space linear combinations) across all layer groups before every forward pass introduces a severe computational and memory-bandwidth latency bottleneck. This makes dynamic ensembling highly impractical for real-time applications or resource-constrained edge devices.

The **Hybrid-Router** aligns perfectly with the Pragmatist. It recognizes that in deep neural networks (like ViTs), early layers function as task-agnostic feature extractors, whereas late layers capture task-specific specialized representations. Therefore, we partition the model:
1. We **statically merge** the first $L-k$ layers offline (using a standard zero-overhead Uniform Merge), completely eliminating the runtime weight-assembly latency for the majority of the network.
2. We **dynamically merge** only the final $k$ layers (including the classification head) at test-time based on input representations.
This hybrid approach preserves almost all task-routing capabilities while reducing the weight-reconstruction latency by up to 90%, proving that dynamic ensembling can be made highly efficient, lightweight, and deployable in the wild.

## 2. Core Techniques
*   **Layer-wise Partitioning:** Splitting the model of depth $L$ into a static partition (layers $1 \dots L-k$) and a dynamic partition (layers $L-k+1 \dots L$) parameterized by a threshold $k$.
*   **Static Partition Merging:** Off-line parameter-space blending of task vectors on the first $L-k$ layers using static Uniform Merging, which incurs absolutely zero computational or memory overhead during inference.
*   **Dynamic Partition Merging:** On-the-fly parameter ensembling of the final $k$ layers driven by a lightweight, classical linear projection routing head with independent Sigmoid activations (Softmax-free) to avoid calibration bottlenecks.
*   **Regularized Calibration:** Optimizing the tiny routing projection head (772 parameters) on a 64-sample calibration dataset with standard $L_2$ weight decay to prevent overfitting.
*   **Latency-vs-Accuracy Ablation Sweeps:** Exhaustively sweeping the partition threshold $k \in \{0, 1, 2, 4, 12, 14\}$ to map the exact Pareto frontier of joint multi-task accuracy versus actual GPU/CPU weight-assembly wall-clock time (in milliseconds).

## 3. Mathematical Formulation
Let $W_{base}^{(l)}$ represent the pre-trained weights of a base model at layer $l \in \{1, \dots, L\}$. Given $K$ task-specific experts fine-tuned from the same base initialization, we define the expert task vectors $V_k^{(l)}$ as:
\begin{equation}
    V_k^{(l)} = W_k^{(l)} - W_{base}^{(l)}
\end{equation}

For a given dynamic partition depth $k \in \{0, \dots, L\}$, we assemble the merged weights $W_{merged}^{(l)}$ for each layer $l$ as follows:

1. **Static Partition ($l \le L-k$):**
   We statically blend the task vectors offline using a uniform scaling coefficient $\lambda_{static} = 0.3$:
   \begin{equation}
       W_{merged}^{(l)} = W_{base}^{(l)} + \lambda_{static} \sum_{k=1}^K V_k^{(l)}
   \end{equation}
   These weights are computed once and never modified during inference.

2. **Dynamic Partition ($l > L-k$):**
   Let $z(x) \in \mathbb{R}^{B \times D}$ be the global representation of an input batch $x$ of size $B$, extracted by average-pooling the patch tokens from the Patch Embedding layer of the ViT backbone. 
   The global routing logits $o(x)_b \in \mathbb{R}^K$ are computed via a single linear routing projection:
   \begin{equation}
       o(x)_b = z(x)_b W_{route} + b_{route}
   \end{equation}
   where $W_{route} \in \mathbb{R}^{D \times K}$ and $b_{route} \in \mathbb{R}^K$.
   To avoid the zero-sum Softmax competitive bottleneck during mixed-batch calibration, we apply independent Sigmoid activations bounded by $\lambda_{max} = 0.3$:
   \begin{equation}
       \alpha_{k, b} = \lambda_{max} \times \text{Sigmoid}(o(x)_{b, k}) \quad \text{with } \lambda_{max} = 0.3
   \end{equation}
   We collapse the sample-level routing coefficients to batch-level ensembling weights:
   \begin{equation}
       \bar{\alpha}_k = \frac{1}{B} \sum_{b=1}^B \alpha_{k, b}
   \end{equation}
   The active parameters for the dynamic partition layers are dynamically assembled on-the-fly:
   \begin{equation}
       W_{merged}^{(l)}(x) = W_{base}^{(l)} + \sum_{k=1}^K \bar{\alpha}_k V_k^{(l)}
   \end{equation}

**Calibration Objective:**
The routing projection head parameters ($W_{route}, b_{route}$) are optimized on a tiny 64-sample calibration split $\mathcal{D}_{cal}$ using Adam with $L_2$ weight regularization:
\begin{equation}
    \mathcal{L} = \frac{1}{|\mathcal{D}_{cal}|} \sum_{(x, y) \in \mathcal{D}_{cal}} \mathcal{L}_{CE}(f(x; W_{merged}(x)), y) + \gamma \|W_{route}\|_2^2
\end{equation}
where $f(x; W_{merged}(x))$ represents the full forward pass of the model, and $\gamma = 1\times 10^{-4}$ is the weight decay hyperparameter.

## 4. Architecture Specifications
*   **Backbone Model:** Vision Transformer (`vit_tiny_patch16_224`) consisting of $L=14$ layer groups (Patch Embedding, 12 Transformer blocks, and a final Layer Normalization layer) with hidden dimension $D=192$.
*   **Routing Input:** Splayed global features $z(x) \in \mathbb{R}^{B \times 192}$ obtained by average-pooling the token embeddings from the patch embedding layer.
*   **Routing Projection Head:** A single linear layer mapping $\mathbb{R}^{192} \to \mathbb{R}^4$, consisting of exactly $192 \times 4 + 4 = 772$ parameters.
*   **Target Tasks ($K=4$):** MNIST, FashionMNIST, CIFAR-10, SVHN.
*   **Ablation Sweeps:** We explicitly evaluate dynamic partition depth $k \in \{0, 1, 2, 4, 12, 14\}$ layers.
    *   $k=0$ represents a pure static uniform merge (0% dynamic parameters).
    *   $k=14$ represents fully dynamic ensembling (100% dynamic parameters).
    *   $k \in \{1, 2, 4\}$ represent the proposed high-performance, low-overhead Hybrid-Router states.

## 5. Baselines
We compare Hybrid-Router against four foundational baselines:
1. **Static Uniform Merging ($k=0$):** Fuses all expert weights with static uniform weights ($\lambda = 0.3$). Has zero ensembling latency but represents the lower-bound on accuracy due to representation conflict.
2. **Standard Linear Router (Classical):** Unregularized Softmax-based global linear router. Highly prone to SVHN collapse and overfitting under low-data calibration.
3. **Fully Dynamic BSigmoid-Router ($k=14$):** Dynamically ensembles every layer group in the ViT backbone. This serves as the upper-bound on routing capacity but represents the worst-case ensembling latency.
4. **QWS-Merge (SOTA Cosine-Wave):** Quantum-inspired wave-interference superposition model merging, evaluating whether our simple classical hybrid routing matches or beats its capacity.

## 6. Step-by-Step Interaction
1. **Offline Preparation:**
   * For the static partition layers ($l \le L-k$), compute $W_{merged}^{(l)}$ offline via Uniform Merging and cache the weights.
   * For the dynamic partition layers ($l > L-k$), extract and store the task vectors $V_k^{(l)}$ in memory.
2. **Forward Inference Pass:**
   * Receive an input batch $x$ of size $B$.
   * Pass $x$ through the Patch Embedding layer of the ViT backbone to extract the initial patch tokens $H_0 \in \mathbb{R}^{B \times N \times D}$.
   * Spatially average $H_0$ across the token dimension to yield the global representation $z(x) \in \mathbb{R}^{B \times 192}$.
   * Pass $z(x)$ through the linear projection head of the router to obtain raw logits: $o(x)_b = z(x)_b W_{route} + b_{route}$.
   * Compute independent, bounded task coefficients: $\alpha_{k, b} = 0.3 \times \text{Sigmoid}(o(x)_{b, k})$.
   * Collapse these coefficients to obtain the batch-level merging coefficients: $\bar{\alpha}_k = \frac{1}{B} \sum_{b=1}^B \alpha_{k, b}$.
   * For the dynamic partition layers ($l > L-k$), dynamically reconstruct the weights: $W_{merged}^{(l)}(x) = W_{base}^{(l)} + \sum_{k=1}^K \bar{\alpha}_k V_k^{(l)}$.
   * Process the initial tokens $H_0$ sequentially through the $L=14$ layer groups, applying the statically pre-merged weights for $l \le L-k$, and applying the dynamically reconstructed weights for $l > L-k$.
   * Output the final classification predictions.
3. **Profiling and Evaluation:**
   * Measure the actual wall-clock execution time (in ms) spent performing the parameter ensembling operations ($\sum_{k=1}^K \bar{\alpha}_k V_k^{(l)}$) to empirically verify and plot the accuracy-latency trade-off curves.
