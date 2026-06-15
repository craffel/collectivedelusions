# Peer Review of Conference Submission: SPS-ZCA

## Summary of the Paper
The paper addresses the practical challenges of serving multiple task-specific Low-Rank Adaptation (LoRA) expert adapters concurrently from a shared pre-trained base model on resource-constrained edge hardware. To handle heterogeneous, mixed-task streaming workloads, the authors propose **SPS-ZCA** (Single-Pass Sample-Wise Routing with Zero-Shot Centroid Alignment), a training-free dynamic model-merging framework. 

SPS-ZCA operates through three primary components:
1. **Zero-Shot Centroid Pre-computation:** Centroids for each task expert are pre-computed offline from a tiny, 64-sample calibration split in the shared early representation space (Layer 3) of the pre-trained base model.
2. **Zero-Shot Centroid Alignment (ZCA) Routing:** Inputs are routed using the cosine similarity between their Layer 3 activations and the pre-computed centroids, bypassing noisy task-specific classification heads.
3. **Single-Pass Activation-Space Dynamic Blending (SPS):** Instead of splitting heterogeneous batches sequentially (which introduces linear latency overhead), SPS executes a single, parallel forward pass and blends expert activations layer-wise on-the-fly.

To bolster robustness, the authors introduce Unit-Norm Calibration (UNC), Intra-Task Dispersion Calibration (IDC), and a coordinate-space diagonal Gaussian Mixture Model (GMM) for out-of-distribution (OOD) task rejection. On a 4-task vision suite (MNIST, F-MNIST, CIFAR-10, SVHN) using a Vision Transformer (ViT-Tiny) backbone, the authors report that SPS-ZCA recovers 100% of the isolated Expert Ceiling (79.80% Joint Mean accuracy), outperforming prior non-parametric SOTA methods by +3.66%. The authors also show generalizability to text models (GPT-2) and profile actual physical CPU serving overheads in PyTorch, presenting a compiler-co-designed loop layout to bridge the "serving gap" at scale.

---

## Detailed Assessment: Strengths and Weaknesses

### Strengths
1. **Systems-ML Relevance and Motivation:** The paper addresses a highly critical bottleneck in edge deep learning deployment: the DRAM memory bandwidth and kernel launch latency overhead of sequentially executing multiple base-model forward passes.
2. **Transparency and Commendable Honesty:** The authors are exceptionally honest and transparent regarding the "serving gap" (Section 4.8.2). They openly profile their framework and show that in standard uncompiled PyTorch under large batch sizes ($B=256$), their single-pass activation blending actually experiences a minor physical wall-clock slowdown due to framework-level indexing and memory allocation overhead. This realistic profiling is rare and highly valuable for systems researchers.
3. **Thorough Empirical Ablations:** The experimental evaluation includes a wide range of targeted sweeps, validating the impact of scale imbalance (UNC), manifold variance (IDC), routing temperature ($\tau$), and OOD log-likelihood thresholds ($\eta$).
4. **Modality Generalizability:** The paper successfully extends the early-layer centroid routing paradigm to autoregressive text generation (using a pre-trained GPT-2 model across Legal, Medical, and Code domains) and formalizes essential text-modal systems elements like KV Cache Sharing and ROUGE-L quality.
5. **Excellent Clarity and Presentation:** The paper is extremely well-structured, clear, and easy to follow. The mathematical notation is clean, and figures/tables are highly informative.

### Weaknesses (Theory and Rigor Focus)
While the paper presents a highly engineered systems-ML framework with strong practical motivations, a rigorous evaluation reveals severe **gaps in theoretical soundness, lack of mathematical proofs, and reliance on unproven heuristics**:

1. **Linear Blending of Non-Linear Pathways (Lack of Mathematical Guarantees):**
   The core Single-Pass Activation Blending formulation (Equation 5) assumes that task-specific activations can be blended linearly at each layer $l$:
   $$h_b^{(l)} = h_b^{(l-1)} W_{\text{base}}^{(l)} + \sum_{k=1}^K \alpha_{k, b} \left( h_b^{(l-1)} A_k^{(l)} B_k^{(l)} \right)$$
   However, in deep neural network backbones, the output of layer $l$ is immediately passed through highly non-linear operators (such as GELU/ReLU activations, Multi-Head Attention softmax, and Layer Normalization) before reaching layer $l+1$. 
   Because these operators are non-linear, a linear combination of adapter outputs at layer $l$ does **not** mathematically correspond to a linear combination of downstream activations at layer $l+1$:
   $$\text{Block}^{(l+1)}(h_b^{(l)}) \neq \sum_{k=1}^K \alpha_{k,b} \text{Block}^{(l+1)}\left(h_b^{(l-1)} W_{\text{base}}^{(l)} + h_b^{(l-1)} A_k^{(l)} B_k^{(l)}\right)$$
   The paper provides **no theoretical analysis, stability proofs, or mathematical bounds on the error propagation** when blending activations across multiple sequential non-linear layers. Under what mathematical conditions is this linear blending approximation stable? If the routing coefficients $\alpha_{k,b}$ are non-binary (e.g., in fuzzy routing, or during adaptive temperature relaxation), does this linear blending steer intermediate features into out-of-distribution regions of the activation space, leading to chaotic representation drift? The paper relies entirely on empirical validation to bypass this fundamental theoretical question.

2. **Heuristic Nature of Centroid Routing and Lack of Mathematical Separability Guarantees:**
   ZCA routing pre-computes centroids via simple sample means and performs routing in the early representation space (Layer 3). While the authors empirically show high separability (Fisher Separability Criterion FSC = 47.50 at Layer 3), they provide **no theoretical framework or mathematical guarantees** explaining *why* early layers are guaranteed to be task-separable. Under what mathematical assumptions about the pre-trained base model or the data-generating distributions of downstream tasks is Layer 3 separable? Furthermore, how does this separability degrade as the number of tasks $K$ scales, or when tasks belong to highly fine-grained or overlapping domains (e.g., medical subtypes)? 
   The paper acknowledges that spatial overlap causes routing confusion and "activation bleeding," confirming that the training-free centroid-aligned routing is theoretically brittle and highly sensitive to representational geometry, yet fails to provide mathematical bounds on the acceptable task overlap.

3. **Overfitting Risks and High Variance in Coordinate-Space GMMs:**
   The coordinate-space diagonal GMM for OOD task rejection is fitted on a tiny calibration split consisting of only $|\mathcal{C}_k| = 64$ samples.
   In classical statistics, fitting a Gaussian Mixture Model (even with diagonal covariance) under extreme data scarcity is highly prone to severe overfitting, high estimation variance, and singular covariance matrices. Although the authors propose adding a diagonal ridge term $\gamma I$ ($\gamma = 10^{-4}$), this is a purely heuristic patch. There is no rigorous mathematical proof or risk bound demonstrating that this regularized density estimator generalizes under such low-sample regimes, particularly when the coordinate dimensions scale with a higher number of task experts $K$.

4. **Heuristic and Redundant Calibration Operators:**
   - **Unit-Norm Calibration (UNC)** is mathematically identical to standard cosine similarity. Presenting it as a separate "novel calibration technique" is theoretically redundant and over-claims novelty.
   - **Intra-Task Dispersion Calibration (IDC)** scales similarity coordinates by dividing them by their expected in-distribution mean. This is a heuristic normalization. The authors provide no probabilistic or statistical proof showing that this division preserves valid density properties or corresponds to a mathematically rigorous coordinate transformation on the unit hypersphere.

5. **Simplistic Visual Benchmarks and Sub-optimal Baselines:**
   - The primary visual evaluation relies on highly simplistic, low-resolution datasets (MNIST, Fashion-MNIST, CIFAR-10, SVHN). These represent toy classification tasks with extremely distinct geometric features, resulting in an artificially high FSC (47.50). The framework is not evaluated on realistic, complex, or fine-grained domains where representational overlap is severe.
   - The classification accuracy of the **SVHN Expert is exceptionally low (29.78% physical, 31.20% simulated)**. For a standard 10-class task, a well-optimized expert should easily exceed 90% accuracy. A baseline under 30% indicates severe sub-optimality, making the joint "Expert Ceiling" recovery claims less convincing.

6. **Lack of Physical Compiled Binary at Scale:**
   The projected 3.90$\times$ speedup relies on an analytical cost model and a conceptual loop layout (Appendix A) rather than an actual physical compiled implementation. In physical PyTorch, the framework actually suffers from a slowdown at larger batch sizes ($B=256$) due to list indexing and allocation overhead. The systems claims are therefore partially unverified on physical hardware at scale.

---

## Ratings on Key Dimensions

### Soundness: Fair
The systems motivation is strong, and the empirical ablations are thorough. However, from a theoretical perspective, the paper lacks mathematical rigor, formal proofs, or error propagation bounds for its core activation-blending and calibration operators. The visual benchmarks are toy-like, and the SVHN expert baseline is sub-optimal.

### Presentation: Excellent
The writing is exceptionally clear, logical, and highly readable. The authors are transparent about framework-level overheads ("serving gap") and provide comprehensive tables and figures.

### Significance: Fair
The practical utility of the framework for low-resource on-device serving is clear, and edge developers may find the vectorized scatter-gather (SPS-VSG) immediately useful. However, due to its heuristic nature and lack of theoretical foundations or mathematical guarantees, the paper is unlikely to influence future theoretical research in machine learning representations or multi-expert optimization.

### Originality: Fair
The delta from prior systems (MBH and PFSR) is clear. However, the conceptual components (sample means, cosine similarity, diagonal GMMs, Softmax) are standard statistical tools. The novelty lies in their practical engineering integration rather than new algorithmic or theoretical concepts.

---

## Overall Recommendation: 3: Weak Reject
The paper is a well-written and highly transparent systems-ML engineering paper with strong empirical results on toy tasks. However, it suffers from a critical lack of theoretical grounding, formal proofs of correctness, or mathematical guarantees. The core formulation of linear activation blending across sequential non-linear layers is presented heuristically without any error propagation bounds or stability analysis. The visual evaluation is restricted to simplistic datasets, and the SVHN baseline is highly sub-optimal. 

For these reasons, the theoretical and methodological weaknesses currently outweigh the systems-level merits. The paper requires revision to establish a rigorous mathematical framework, prove stability/error bounds for activation blending, and demonstrate scalability on realistic, fine-grained, or complex tasks before it can be meaningfully built upon by others.

---

## Detailed Questions and Feedback for the Authors
1. **Mathematical Stability of Activation Blending:** Please provide a formal theoretical analysis or mathematical proof of the error propagation and stability of linear activation blending (Equation 5) across sequential non-linear layers (such as GELU, Multi-Head Attention, and LayerNorm). Under what mathematical assumptions is the linear blending approximation of non-linear pathways guaranteed to have bounded error?
2. **Early-Layer Separability Guarantees:** Can you provide a formal theoretical justification or proof of why the early representation space (e.g., Layer 3) is guaranteed to be task-separable? Under what conditions or assumptions about the pre-trained base model does this separability hold?
3. **Generalization of Coordinate GMMs:** Please provide a rigorous statistical or probabilistic analysis of the coordinate-space GMM's generalization error. Given that the GMM is fitted on only $|\mathcal{C}_k| = 64$ samples, how can you mathematically guarantee that the density boundaries do not suffer from severe estimation variance or overfitting under mild covariate shifts?
4. **Validation on Fine-Grained/Overlapping Domains:** To demonstrate the robustness of ZCA routing beyond toy datasets with artificially high FSC, please provide empirical results on realistic, fine-grained, or highly overlapping domains (e.g., ImageNet subclasses, CUB-200 birds, Stanford Cars). How do your proposed Hierarchical Centroid Clustering or Supervised Head Fine-Tuning (SHFT) methods perform empirically on these tasks?
5. **SVHN Expert Optimization:** Why is the classification accuracy of your SVHN task-expert so low (29.78% physical, 31.20% simulated)? Please optimize or retrain this expert to reflect a realistic, high-accuracy baseline, and verify if your "Expert Ceiling" recovery claims still hold when all experts are highly specialized.
6. **Physical Compiled Speedups:** To fully support your systems speedup claims, can you compile your vectorized scatter-gather kernels (using a compiler like Apache TVM, ExecuTorch, or ONNX Runtime CustomOps) and report actual physical wall-clock speedups under large batch sizes ($B=256$) rather than relying purely on analytical projections?
