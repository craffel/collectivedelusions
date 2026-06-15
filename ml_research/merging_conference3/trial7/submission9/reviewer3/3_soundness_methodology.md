# Evaluation Component 3: Soundness and Methodology Evaluation

## 1. Technical Soundness of SABLE
SABLE's core technical idea is mathematically straightforward. At the layer level, ensembling activations with sample-wise coefficients:
$$Y_{l, b} = X_b W_{\text{base}, l} + \sum_{k=1}^K \alpha_{k, b} \cdot \left( (X_b A_{k, l}) B_{k, l} \right)$$
is exactly mathematically equivalent to parameter-space ensembling with sample-specific weights:
$$W_{\text{merged}, b, l} = W_{\text{base}, l} + \sum_{k=1}^K \alpha_{k, b} A_{k, l} B_{k, l}$$
because the distributive property of linear algebra holds:
$$X_b (W_{\text{base}, l} + \sum_{k=1}^K \alpha_{k, b} A_{k, l} B_{k, l}) = X_b W_{\text{base}, l} + \sum_{k=1}^K \alpha_{k, b} X_b A_{k, l} B_{k, l}$$
Thus, for a single sample $b$, SABLE's activation blending is indeed exact. This is a sound and elegant algebraic observation that allows SABLE to process heterogeneous batches of size $B > 1$ while preserving sample-wise ensembling, which would otherwise require $B$ separate forward passes in parameter space (defeating parallel execution).

---

## 2. Methodology Appropriateness
The proposed routing, OOD rejection, and pruning methods are highly appropriate for low-latency, stateless deployment, but they introduce major structural trade-offs and theoretical limitations:

1. **Mid-Layer Routing (Late Adaptation) vs. Early-Feature Loss:**
   * *Method:* SABLE's default setup leaves layers $l < L_{\text{route}}$ unadapted and runs them strictly through the pre-trained base network.
   * *Critique:* This assumes that task-specific adaptation is concentrated entirely in the late-stage layers. If downstream experts have learned critical task-specific features in early layers (e.g., in low-level vision or specialized text embeddings), SABLE Late Adaptation is structurally unable to leverage them. 
2. **Early-Layer Routing vs. Representational Alignment Paradox:**
   * *Method:* Early-Layer Routing runs Layer 0 of the base model to compute routing coefficients.
   * *Critique:* Lower-layer features represent low-level sensory patterns (edges, local textures) that are not semantically aligned with final classification heads $w_k$, resulting in highly noisy routing coefficients. The paper's fallback of using an external routing backbone (e.g., MiniLM, MobileNet) to resolve this introduces extra model weights and parameters, violating SABLE's "zero-parameter" minimalist claim.

---

## 3. Potential Technical Flaws and Theoretical Gaps
From a rigorous, theory-minded perspective, SABLE has several critical theoretical gaps and unproven assumptions:

### A. Dual-Space Mismatch of Zero-Data Centroids
The "Zero-Data Centroids" heuristic projects activation representation vectors $z_b$ onto expert classification weight matrices $W_{\text{expert}, k}$ to compute cosine similarity:
$$s_{k, b} = \frac{z_b \cdot c_{\text{zero}, k}}{\|z_b\|_2 \|c_{\text{zero}, k}\|_2}$$
where $c_{\text{zero}, k}$ is derived from the parameters of the classification heads.
* **The Gap:** Weights and activations lie in fundamentally different mathematical spaces, manifolds, and scales. Weights are optimized directions in a parameter space designed to maximize logit margins; activations are feature distributions in a representational space. Measuring their cosine similarity directly constitutes a **dual-space mismatch**. There is no mathematical proof or guarantee that the parameter space maps isomorphically to the representation space, nor are there bounds on the routing error introduced by this mismatch.

### B. Vector Cancellation in Parameter-Based Centroids
Taking the row-wise mean of $W_{\text{expert}, k}$ to construct $c_{\text{zero}, k}$ assumes that the average of class vectors represents the overall task coordinate.
* **The Gap:** Since classification heads are trained discriminatively to maximize class margins, class-specific weight vectors often point in opposite or orthogonal directions. Taking their simple mean can lead to **partial vector cancellation**, producing a centroid vector with reduced norm or a skewed semantic orientation. While L2-normalizing the weights before averaging (Refined Zero-Data Centroids) empirically improves performance, it remains a heuristic adjustment lacking a formal, mathematically rigorous probabilistic proof.

### C. Cumulative Non-Linear Drift
For multi-layer networks, successive layers are separated by non-linear activation functions (e.g., ReLU). 
* **The Gap:** The distributive property of linear algebra does not hold across a sequence of multiple non-linear layers:
  $$\sigma\left( \sum_k \alpha_k (X W_k) \right) \neq \sum_k \alpha_k \sigma(X W_k)$$
  Although the paper shows high similarity ($>0.83$) in a shallow 4-layer MLP, there is no theoretical bound or stability guarantee showing that this cumulative non-linear drift does not diverge catastrophically in deeper architectures (e.g., 32-70 layer modern LLMs).

### D. Qualitative Explanations of Anomalies
The paper relies on post-hoc, qualitative explanations for interesting empirical phenomena rather than providing rigorous mathematical proofs:
* **The Low-Rank Regularization Paradox:** The paper claims that restricting the hidden layer rank to $r=2$ (under SABLE Hybrid) "acts as a powerful regularizer, pruning high-frequency representation noise." This is a speculative assertion without a mathematical definition of "high-frequency representation noise" or a formal proof of this regularizing effect.
* **Destructive Representational Interference:** The reversal where SABLE Hard ($M=1$) outperforms SABLE Soft ($M=2$) at higher ranks ($r=8$) under confounded streams is described qualitatively as "colliding in the intermediate layers, causing severe mutual cancellation." A formal mathematical framework defining this collision or bounding the manifold incompatibility is entirely absent.

---

## 4. Reproducibility
The algorithmic steps are presented clearly, and the paper includes precise equations, architectural schematics (Figure 1), and complete hyperparameter details ($\tau = 0.05$, $\gamma_{\text{OOD}} = 0.2$, $r=8$). 

However, the "Analytical Coordinate Sandbox" is a synthetic, custom-built 14-layer benchmark. Because this sandbox is entirely synthetic and custom-calibrated, reproducing its exact quantitative results depends entirely on the availability of the authors' custom coordinate generation and training code, which is not standard or publicly documented. The inclusion of physical CNN and ResNet-18 experiments on standard benchmarks (MNIST/FashionMNIST) improves reproducibility, though the lack of a publicly available repository URL or code appendix limits immediate verification.
