# Idea Proposal: Confidence-Gated Hybrid Routing (CGHR)

## 1. Persona Alignment
As **The Empiricist**, our core philosophy is that true progress in machine learning comes from exhaustive empirical validation and large-scale parallel sweeps across parameters, datasets, and seeds. **Confidence-Gated Hybrid Routing (CGHR)** is specifically designed to be an extremely rich sandbox for empirical exploration. Instead of presenting a purely theoretical framework, CGHR introduces a gating threshold parameter $\gamma_{\text{conf}}$ that allows us to sweep and map the complete continuous trade-off curve between parametric routing flexibility and non-parametric routing robustness. This design enables massive parallel parameter sweeps across:
1. The confidence threshold $\gamma_{\text{conf}} \in [0, 1]$ to empirically find the optimal boundary between parametric and parameter-free regimes.
2. The calibration set sample complexity $N \in \{16, 32, 64, 128, 256, 512\}$ to rigorously map when parametric routers overfit and how CGHR stabilizes learning.
3. Multiple seeds and batch sizes ($B=1$ to $B=512$) to empirically verify robustness against "vectorization collapse" and "heterogeneity collapse".
4. Different confidence metrics (entropy vs. margin vs. max probability) and parametric regularizations (L2 weight decay vs. TSAR) to isolate the true drivers of multi-task performance.

## 2. Core Techniques
CGHR introduces a hybrid, dual-pathway routing system:
- **Pathway A (Flexible Parametric Routing)**: A lightweight, trained linear layer (Softmax or Sigmoid activated) that learns task-specific routing coefficients using gradient descent on a small calibration set. While highly flexible for in-distribution (ID) data, it is vulnerable to overfitting and out-of-distribution (OOD) collapse.
- **Pathway B (Robust Parameter-Free Subspace Routing)**: A zero-shot, completely non-parametric router (PFSR) that projects penultimate representations onto pre-trained expert classification weight manifolds using cosine similarities. It has zero trainable parameters and is completely immune to overfitting.
- **Confidence Gating Mechanism**: At inference time, we calculate the prediction confidence of the parametric router. If the confidence is high ($\ge \gamma_{\text{conf}}$), we route using Pathway A. If the confidence is low ($< \gamma_{\text{conf}}$), indicating a potential OOD task or ambiguous sample, we dynamically fall back to the robust Pathway B (PFSR).
- **Micro-Batch Homogenization (MBH)**: Combined with CGHR to partition heterogeneous streams on the fly and prevent batch-averaging smoothing collapse.

## 3. Mathematical Formulation
Let $z_b \in \mathbb{R}^D$ be the globally pooled high-dimensional feature representation of sample $b$ from the penultimate layer of the backbone model.

### Parametric Gating (Pathway A)
The trained parametric router is parameterized by weights $W_{\text{param}} \in \mathbb{R}^{K \times D}$ and bias $b_{\text{param}} \in \mathbb{R}^K$. It outputs raw routing logits:
\begin{equation}
a_b = W_{\text{param}} z_b + b_{\text{param}}
\end{equation}
The parametric routing coefficients $\alpha^{\text{param}}_b \in \mathbb{R}^K$ are computed using a Softmax activation:
\begin{equation}
\alpha^{\text{param}}_{k, b} = \frac{\exp(a_{k, b})}{\sum_{j=1}^K \exp(a_{j, b})}
\end{equation}

### Parameter-Free Subspace Routing (Pathway B)
Let the weights of Expert $k$'s classification head be $W_k \in \mathbb{R}^{C_k \times d}$, where $d = D // K$. We compute the calibrated cosine-similarity task coordinates:
\begin{equation}
u'_{k, b} = \frac{\max_{c} \frac{W_{k, c} \cdot z_{k, b}}{\|W_{k, c}\|_2 \|z_{k, b}\|_2}}{\sqrt{2\log C_k / d}}
\end{equation}
The parameter-free coefficients $\alpha^{\text{pfsr}}_b \in \mathbb{R}^K$ are derived via temperature-scaled Softmax:
\begin{equation}
\alpha^{\text{pfsr}}_{k, b} = \frac{\exp(u'_{k, b} / \tau)}{\sum_{j=1}^K \exp(u'_{j, b} / \tau)}
\end{equation}

### Confidence Gating and Hybrid Coefficients
We define the confidence $\mathcal{C}(\alpha^{\text{param}}_b)$ of the parametric router's prediction. We evaluate three distinct formulations:
1. **Max Probability**:
   \begin{equation}
   \mathcal{C}_{\text{max}}(\alpha^{\text{param}}_b) = \max_{k \in \{1, \dots, K\}} \alpha^{\text{param}}_{k, b}
   \end{equation}
2. **Negative Entropy**:
   \begin{equation}
   \mathcal{C}_{\text{entropy}}(\alpha^{\text{param}}_b) = 1 - \frac{-\sum_{k=1}^K \alpha^{\text{param}}_{k, b} \log \alpha^{\text{param}}_{k, b}}{\log K}
   \end{equation}
3. **Margin**:
   \begin{equation}
   \mathcal{C}_{\text{margin}}(\alpha^{\text{param}}_b) = \alpha^{\text{param}}_{[1], b} - \alpha^{\text{param}}_{[2], b}
   \end{equation}
   where $[1]$ and $[2]$ denote the first and second largest coefficients in $\alpha^{\text{param}}_b$.

Given a static confidence threshold $\gamma_{\text{conf}} \in [0, 1]$, the final hybrid routing coefficients $\alpha^{\text{hybrid}}_b \in \mathbb{R}^K$ are defined sample-wise:
\begin{equation}
\alpha^{\text{hybrid}}_b = \begin{cases} 
\alpha^{\text{param}}_b & \text{if } \mathcal{C}(\alpha^{\text{param}}_b) \ge \gamma_{\text{conf}} \\
\alpha^{\text{pfsr}}_b & \text{if } \mathcal{C}(\alpha^{\text{param}}_b) < \gamma_{\text{conf}}
\end{cases}
\end{equation}

## 4. Architecture Specifications
- **Backbone**: Vision Transformer (ViT-Base or ViT-Tiny) or synthetic Sandbox backbone.
- **Parametric Router**: A single linear projection layer $D \to K$ trained using Cross-Entropy loss on the calibration task labels, optimized with AdamW and L2 regularization.
- **Classification Heads**: $K$ pre-trained frozen expert head matrices $W_k \in \mathbb{R}^{C_k \times d}$ used for zero-shot projection.
- **Hyperparameters**:
  - Routing Temperature: $\tau = 0.001$.
  - Confidence Threshold: $\gamma_{\text{conf}} \in [0, 1]$ (default $\gamma_{\text{conf}} = 0.85$, dynamically swept).
  - Feature Dimension: $D = 192$ (Sandbox) or $D = 768$ (ViT).
  - Number of Experts: $K = 4$.

## 5. Baselines
Our exhaustive empirical sweeps will compare CGHR against:
1. **Static Uniform Merging**: Standard baseline blending experts equally ($\alpha_k = 1/K$).
2. **Parametric Linear Router (Unregularized)**: Standard softmax-based linear router prone to vectorization collapse and OOD overfitting.
3. **Parametric Linear Router (Reg)**: Highly-tuned classical baseline regularized with L2 weight decay.
4. **VR-Router**: Modern state-of-the-art parametric router regularized with Task-Variance Regularization ($\mathcal{L}_{VR}$).
5. **Parameter-Free Subspace Routing (PFSR)**: Pure non-parametric zero-shot routing baseline.
6. **Expert Ceiling**: The standalone standalone accuracy of unmerged, specialized expert models.

## 6. Step-by-Step Interaction
The flow of data through the CGHR framework proceeds as follows:
1. **Feature Extraction**: An input batch $X = \{x_1, \dots, x_B\}$ is processed by the frozen model backbone to extract the penultimate representation $Z = \{z_1, \dots, z_B\} \in \mathbb{R}^{B \times D}$.
2. **Parametric Inference**: The intermediate representations $Z$ are passed through the trained parametric routing head to generate raw logits $a_b$, which are softmax-activated to obtain the parametric coefficients $\alpha^{\text{param}}_b$.
3. **Confidence Scoring**: For each sample $b$, we compute the confidence score $\mathcal{C}(\alpha^{\text{param}}_b)$.
4. **Subspace Projection (on demand)**: For any sample $b$ where $\mathcal{C}(\alpha^{\text{param}}_b) < \gamma_{\text{conf}}$, we compute the calibrated cosine similarities $u'_{k, b}$ against the classification prototypes of all experts $k$, deriving $\alpha^{\text{pfsr}}_b$.
5. **Hybrid Selection**: The final sample-wise coefficients $\alpha^{\text{hybrid}}_b$ are compiled based on the threshold $\gamma_{\text{conf}}$.
6. **Micro-Batch Homogenization (MBH)**: The batch is dynamically partitioned into homogeneous micro-batches $X^{(1)}, \dots, X^{(G)}$ on the fly.
7. **Parameter Fusion & Output Compilation**: For each micro-batch, coefficients are averaged and used to merge LoRA expert adapters dynamically. Predictions are run and re-sorted to match the original input batch ordering.
