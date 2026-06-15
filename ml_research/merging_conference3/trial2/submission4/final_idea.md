# EdgeMerge (Forward-Only Adaptive Model Merging)

## 1. Persona Alignment
EdgeMerge is highly aligned with **The Pragmatist** persona. Current state-of-the-art adaptive merging methods (such as SyMerge and FoldMerge) require joint gradient-based optimization over 500 steps, which takes up to 10+ minutes on H100 GPUs, demands enormous memory for backpropagation, and requires holding multiple expert models in memory. 

EdgeMerge completely removes these barriers. It is a **one-shot, training-free, forward-only** adaptive merging framework. By using a small batch of unlabeled calibration inputs (e.g., $B = 32$) in a single forward pass, it extracts activation statistics to compute channel-wise (neuron-wise) merging coefficients in closed-form. This results in:
- **Zero Gradient Memory Overhead:** Runs safely on resource-constrained edge devices (IoT, mobile, CPUs) with no backpropagation.
- **Near-Zero Latency:** Computes merging weights in a fraction of a second instead of 10+ minutes of gradient-descent.
- **Immunity to Overfitting:** Bypasses gradient-descent entirely, avoiding the *Overfitting-Optimizer Paradox* where delicate test-time tuning degrades generalization on unseen test data.
- **Deployment Simplicity:** Easy to integrate into standard ML inference pipelines.

## 2. Core Techniques
EdgeMerge introduces three key techniques:
1. **Forward-Only Activation Sampling (FOAS):** Runs a single forward pass of a tiny, unlabeled calibration dataset through the base model and the task-specific experts to capture their functional behavior.
2. **Scale-Normalized Delta Activation Salience (SNDAS):** Calculates the relative change in internal activations between each expert and the base model, normalized across tasks to prevent experts with larger natural activation scales from dominating.
3. **Channel-Wise Softmax Gating (CWSG):** Normalizes the importance scores of each neuron/channel across the tasks using a softmax function, producing fine-grained, localized merging coefficients instead of a single rigid layer-wise coefficient.

## 3. Mathematical Formulation
Let $W_{base} \in \mathbb{R}^{d_{out} \times d_{in}}$ denote the target weight matrix of the pre-trained base model, and $W_k \in \mathbb{R}^{d_{out} \times d_{in}}$ denote the weights of the $k$-th task expert model ($k \in \{1, \dots, K\}$). 

Let $X \in \mathbb{R}^{B \times d_{in}}$ be a small calibration batch of $B$ unlabeled inputs (e.g., $B = 32$). The activations at the target layer are computed as:
$$H_{base} = X W_{base}^T \in \mathbb{R}^{B \times d_{out}}$$
$$H_k = X W_k^T \in \mathbb{R}^{B \times d_{out}}$$

For each task $k$, we define the activation delta relative to the base model:
$$\Delta H_k = H_k - H_{base} \in \mathbb{R}^{B \times d_{out}}$$

To prevent tasks with larger activation scales from disproportionately dominating the merge, we apply Frobenius norm-based scale normalization to each delta activation tensor:
$$\tilde{\Delta} H_k = \frac{\Delta H_k}{\|\Delta H_k\|_F}$$

We compute the scale-normalized channel-wise salience vector $S_k \in \mathbb{R}^{d_{out}}$, representing the functional importance of each output channel $j \in \{1, \dots, d_{out}\}$ for task $k$:
$$S_k[j] = \frac{1}{B} \sum_{i=1}^B |\tilde{\Delta} H_k[i, j]|$$

To resolve channel-wise interference, we normalize the channel-wise importance across the $K$ experts using a softmax function with a temperature hyperparameter $\tau > 0$:
$$\alpha_k[j] = \frac{\exp(S_k[j] / \tau)}{\sum_{l=1}^K \exp(S_l[j] / \tau)}$$

The final merged multi-task weight matrix $W_{MTL} \in \mathbb{R}^{d_{out} \times d_{in}}$ is reconstructed row-by-row (channel-by-channel) as:
$$W_{MTL}[j, :] = W_{base}[j, :] + \lambda \sum_{k=1}^K \alpha_k[j] \left(W_k[j, :] - W_{base}[j, :]\right)$$
where $\lambda > 0$ is a global scaling factor (default $\lambda = 0.5$ or optimized via a simple grid search on a validation set).

## 4. Architecture Specifications
- **Target Backbone:** ViT-B/32 Vision-Language model (CLIP).
- **Target Layer:** Visual projection layer (`model.visual.proj`) of the image encoder.
- **Dimensions:** Input dimension $d_{in} = 768$, output dimension $d_{out} = 512$.
- **Target Weights:** $W_{base}, W_k \in \mathbb{R}^{512 \times 768}$.
- **Input Representation:** $X \in \mathbb{R}^{B \times 768}$.
- **Intermediate Activation Representations:** $H_base, H_k \in \mathbb{R}^{B \times 512}$.
- **Saliency Scores and Coefficients:** $S_k, \alpha_k \in \mathbb{R}^{512}$ (a vector of 512 coefficients per task).
- **Final Output Weight Matrix:** $W_{MTL} \in \mathbb{R}^{512 \times 768}$.

## 5. Baselines
EdgeMerge will be evaluated against five diverse and competitive baselines on the 8-task benchmark:
1. **Task Arithmetic:** Simple linear weight-space combination with uniform, manual scaling.
2. **TIES-Merging:** Static weight-space sign consensus and pruning.
3. **AdaMerging:** Gradient-based layer-wise coefficient optimization at test-time.
4. **SyMerge:** Gradient-based low-rank adapter scaling optimized via test-time teacher predictions.
5. **FoldMerge:** Non-linear weight-coordinate warping using normalizing flows optimized via test-time teacher predictions.

EdgeMerge will be compared not only on multi-task classification accuracy, but also on **computational cost metrics**:
- Total optimization time (seconds).
- Peak GPU memory consumption (MB).
- Number of backward passes required.

## 6. Step-by-Step Interaction
1. **Data Feeding:** A small batch of $B$ unlabeled calibration images $X$ is fed into the target layer.
2. **Forward Activation Pass:** The base weights $W_{base}$ and task weights $W_k$ process the inputs to produce activations $H_{base}$ and $H_k$.
3. **Delta Extraction & Normalization:** The activations are subtracted to compute task-wise delta matrices $\Delta H_k$, which are subsequently scale-normalized using their Frobenius norm.
4. **Salience Computation:** Column-wise (channel-wise) absolute means are computed to produce task saliency vectors $S_k \in \mathbb{R}^{512}$.
5. **Softmax Gating:** Saliency vectors are passed through a softmax across the $K$ tasks for each channel, yielding channel-wise scaling coefficients $\alpha_k \in \mathbb{R}^{512}$.
6. **Weight Reconstruction:** The merged projection matrix $W_{MTL}$ is reconstructed by adding the scaled channel-wise expert task vectors to the base model weights.
7. **Model Loading:** The reconstructed weights are directly loaded into the visual projection layer of the model, which is immediately ready for deployment and high-speed inference.
