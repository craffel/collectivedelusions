# Idea Proposal: SABLE (Sample-wise Activation Blending of Low-Rank Experts)

## 1. Persona Alignment
This proposal is a relentless and direct application of **The Minimalist** research persona to the problem of test-time dynamic model merging under streaming heterogeneity and data constraints:
- **Occam's Razor over Complex Heuristics**: Prior state-of-the-art methods like PFSR+MBH (Trial 6, Submission 7) resolve "heterogeneity collapse" in mixed-task streams by introducing Micro-Batch Homogenization (MBH)—a complex, stateful dynamic buffering, stream-sorting, and partitioning pipeline. SABLE completely strips away this entire stateful runtime wrapper.
- **Stripping Unnecessary Components**: Instead of merging parameters in weight space (which forces the averaging of coefficients over a batch to avoid running separate forward passes), SABLE operates in **activation space**. It performs a shared base model forward pass and parallel, extremely lightweight low-rank adapter (LoRA) forward passes, combining their activations using sample-specific coefficients.
- **Extreme Cleanliness & No Trainable Bloat**: SABLE requires zero trainable parameters, zero calibration split data, zero optimization loops, and zero stateful buffering latency. It achieves perfect robustness to both vectorization collapse and heterogeneity collapse through simple, closed-form, feed-forward activation scaling.

## 2. Core Techniques
SABLE is built upon three core, lightweight mechanisms:
1. **Low-Rank Parameter-Efficient Experts (LoRA)**: Downstream task experts are parameterized as low-rank adapters $(A_k, B_k)$ fine-tuned from a shared pre-trained base model $W_{\text{base}}$, representing $<1\%$ of the total model capacity.
2. **Sample-wise Activation Blending (Activation-Space Merging)**: Leveraging the distributive property of matrix multiplication, we perform dynamic weight blending directly in activation space: $Y_b = X_b W_{\text{base}} + \sum_k \alpha_{k, b} (X_b A_k B_k)$ for each individual sample $b$ in the batch. This eliminates the need to average routing coefficients over the batch dimension, bypassing heterogeneity collapse natively.
3. **Non-parametric Cosine Subspace Projection (from PFSR \cite{pfsr})**: Routing coefficients $\alpha_{k, b}$ are derived directly on-the-fly by projecting penultimate layer features onto a frozen low-dimensional coordinate subspace using cosine similarity against pre-trained classification heads, requiring zero training or calibration data.

## 3. Mathematical Formulation
Let $z_b \in \mathbb{R}^D$ be the globally pooled penultimate layer representation of sample $b$ in a batch $B$. Let $w_k \in \mathbb{R}^D$ be the frozen classification weight vector (or centroid) of the $k$-th task expert, where $k \in \{1, \dots, K\}$.

### Subspace Cosine Projection
The similarity score $s_{k, b}$ between sample $b$ and expert $k$ is computed as:
$$s_{k, b} = \frac{z_b \cdot w_k}{\|z_b\|_2 \|w_k\|_2}$$

### Out-of-Distribution (OOD) Rejection Threshold
To guard the experts against OOD noise (e.g., SVHN samples when in-distribution tasks are MNIST/FashionMNIST/CIFAR-10), we apply a threshold $\gamma_{\text{OOD}} > 0$:
$$\text{If } \max_{k} s_{k, b} < \gamma_{\text{OOD}}, \text{ then } \alpha_{k, b} = 0 \quad \forall k \in \{1, \dots, K\}$$

### Temperature-Scaled Softmax Routing
For in-distribution samples, the dynamic routing coefficients $\alpha_{k, b}$ are obtained via temperature-scaled Softmax:
$$\alpha_{k, b} = \frac{\exp(s_{k, b} / \tau)}{\sum_{j=1}^K \exp(s_{j, b} / \tau)}$$
where $\tau > 0$ is a temperature hyperparameter.

### Dynamic Activation Blending Layer
For any model layer with a shared base weight matrix $W_{\text{base}} \in \mathbb{R}^{D_{\text{in}} \times D_{\text{out}}}$ and $K$ specialized LoRA experts parameterized by low-rank matrices $A_k \in \mathbb{R}^{D_{\text{in}} \times r}$ and $B_k \in \mathbb{R}^{r \times D_{\text{out}}}$ (where $r \ll \min(D_{\text{in}}, D_{\text{out}})$), the layer's output $Y_b \in \mathbb{R}^{D_{\text{out}}}$ for input representation $X_b \in \mathbb{R}^{D_{\text{in}}}$ of sample $b$ is computed as:
$$Y_b = X_b W_{\text{base}} + \sum_{k=1}^K \alpha_{k, b} \cdot \left( (X_b A_k) B_k \right)$$

This activation blending is mathematically identical to merging the weights $W_{\text{merged}, b} = W_{\text{base}} + \sum_k \alpha_{k, b} A_k B_k$ for a single-sample batch ($B=1$), but generalizes seamlessly to any batch size $B > 1$ without incurring weight averaging errors or OOD overfitting.

## 4. Architecture Specifications
- **Backbone Dimensions**: Input dimension $D_{\text{in}} = 192$, output dimension $D_{\text{out}} = 192$, representing a 14-layer deep network sandbox ($L=14$).
- **Expert Configuration**: $K=4$ task experts representing MNIST, FashionMNIST, CIFAR-10, and SVHN (the latter acts as an OOD task or in-distribution depending on the stream regime).
- **PEFT Parameterization**: LoRA adapter rank $r = 8$.
- **Hyperparameters**:
  - Routing Temperature: $\tau = 0.05$
  - OOD Rejection Threshold: $\gamma_{\text{OOD}} = 0.4$
- **Routing Source**: Penultimate representation layer $z_b \in \mathbb{R}^{192}$ is projected onto frozen task classifier heads of shape $10 \times 192$.

## 5. Baselines
We will evaluate SABLE against the following standard baselines:
1. **Static Uniform Merging**: The parameter-space average baseline ($W_{\text{merged}} = \frac{1}{K} \sum_k W_k$) with zero input adaptivity.
2. **L3-Softmax / VR-Router (Prior-Driven Classical Routing)**: Parametric, well-regularized layer-wise routers optimized on 64 calibration samples.
3. **Parameter-Free Subspace Routing (PFSR) + Micro-Batch Homogenization (MBH)**: The SOTA parameter-free routing method which requires stream buffering, sorting, and micro-batch grouping.

SABLE will be demonstrated to match PFSR+MBH in Joint Mean accuracy (75.00%+) while completely eliminating the stateful buffering latency, sorting complexity, and stream dependencies.

## 6. Step-by-Step Interaction
1. **Batch Reception**: The network receives a heterogeneous batch of representations $X \in \mathbb{R}^{B \times D_{\text{in}}}$.
2. **Penultimate Feature Extraction**: Features flow through the base backbone layers until the penultimate layer, yielding representation vectors $z_b \in \mathbb{R}^{192}$ for each sample $b$.
3. **Subspace Routing Coefficient Inference**:
   - Compute cosine similarities $s_{k, b}$ of $z_b$ against frozen classifier centroids $w_k$.
   - Perform OOD rejection check: if max similarity is below $\gamma_{\text{OOD}} = 0.4$, set $\alpha_{k, b} = 0$.
   - Otherwise, apply temperature-scaled Softmax ($\tau = 0.05$) to obtain the task blending coefficients $\alpha_{k, b}$ for sample $b$.
4. **Layer-wise Parallel Blending**: At each of the 14 layers, for each sample $b$ in the batch:
   - Compute base projection: $H_{\text{base}, b} = X_b W_{\text{base}}$.
   - For each active expert $k$ where $\alpha_{k, b} > 0$, compute low-rank adapter activations: $H_{k, b} = (X_b A_k) B_k$.
   - Blend activations: $Y_b = H_{\text{base}, b} + \sum_k \alpha_{k, b} H_{k, b}$.
5. **Prediction**: The output of the final layer is fed to the classifier heads to compute final multi-task classification logits.
