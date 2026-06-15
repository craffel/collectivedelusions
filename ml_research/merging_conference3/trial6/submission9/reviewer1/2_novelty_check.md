# Novelty and Delta Check: CAM-Router

## Key Novel Aspects
- **Preservation of Spatial Token Coordinates:** Unlike existing dynamic routers which immediately apply global average pooling to the token sequence of the backbone, the proposed CAM-Router retains the full spatial dimensions.
- **Trainable Task-Expert Queries:** The introduction of learned queries ($Q$) representing specific task domains to execute cross-attention over intermediate patch tokens.
- **Decoupled Historical Gating (DHG):** A temporal exponential moving average (EMA) smoothing mechanism designed specifically to prevent sample-level task feature dilution in concurrent, mixed-task batches.

## The 'Delta' from Prior Work
- **From Static Merging (e.g., Task Arithmetic, Ties-Merging, DARE):** Prior methods calculate static merging weights that remain constant for all test inputs. CAM-Router dynamically adjusts merging coefficients on-the-fly depending on the input sample's content.
- **From Average-Pooling Dynamic Routers (e.g., QWS-Merge, BSigmoid-Router):** Existing dynamic routers use a flat, average-pooled feature vector to predict routing logits, making them highly vulnerable to spatial occlusions and mixed-task batch compositions. CAM-Router's spatial cross-attention and DHG represent a direct structural and algorithmic departure to preserve spatial and sample-specific routing signatures.

## Characterization of Novelty
The novelty of this work is **moderate to high** within the specialized subfield of dynamic model merging. While Multi-Head Cross-Attention (MHCA) and Exponential Moving Averages (EMA) are well-established deep learning building blocks, their specific application to weight-space dynamic routing is creative and addresses a genuine limitation of prior global pooling methods. 

However, from a **Minimalist** perspective, this novelty comes at a significant cost of simplicity and elegance. The "delta" in performance is achieved by grafting a heavy, stateful cross-attention mechanism and EMA-based historical tracking onto model-merging. This compromises the core promise of model-merging: being a lightweight, stateless, zero-overhead technique. The introduction of multiple projection matrices, query embeddings, and a stateful historical context tracker is an over-engineered way to solve a problem that might be addressed through much simpler, more direct mechanisms.
