# Peer Review Report

**Paper Title:** EdgeMerge: Forward-Only Adaptive Model Merging
**Recommendation:** 2: Reject

---

## 1. Summary of the Paper
The paper introduces **EdgeMerge**, a training-free, forward-only adaptive model merging framework designed to resolve the computational and memory bottlenecks of test-time adaptation (TTA) in multi-task model composition. Standard state-of-the-art adaptive model merging methods (e.g., SyMerge, FoldMerge) rely on test-time backpropagation and multi-minute gradient-descent loops, making them impractical for resource-constrained edge hardware. 

EdgeMerge completely eliminates backpropagation. By running a single forward pass over a tiny unlabeled calibration dataset (e.g., 32 samples per task) through the base model and experts, the framework computes localized, channel-wise merging coefficients in closed-form on a targeted projection bottleneck layer. The paper evaluates EdgeMerge on an 8-task visual classification benchmark using the modular Vision-Language CLIP ViT-B/32 architecture, showing a $50\times$ speedup in preparation time (11.95 seconds vs. 10 minutes) and negligible GPU memory overhead (~100 MB) compared to gradient-based adaptive baselines.

---

## 2. Strengths of the Paper
1. **Clear Practical Focus:** The paper's emphasis on reducing the compute cost, preparation time, and memory footprint of model merging on resource-constrained devices addresses a highly important and realistic engineering challenge.
2. **Elegant Scale Normalization:** The inclusion of Scale-Normalized Delta Activation Salience (SNDAS) to normalize task activations via their Frobenius norm is a simple and effective way to prevent task experts with large activation scales from disproportionately dominating the channel-wise merging weights.
3. **Well-Written and Structured:** The paper is exceptionally well-structured, easy to follow, and has highly clear mathematical formulations.

---

## 3. Weaknesses of the Paper

### Soundness (Rating: Poor)
1. **Critical Representation Mismatch Flaw:** The mathematical formulation of EdgeMerge relies on comparing activations $H_{base, k} = X_k W_{base}^T$ and $H_k = X_k W_k^T$ to compute the activation delta $\Delta H_k = H_k - H_{base, k}$. In the paper, $X_k$ is defined as "the input features presented to the target layer for task $k$." Because the task experts are fine-tuned, their upstream encoder layers are different from those of the base model. Consequently, the actual input features presented to the target layer in the expert model during inference are $X_k^{expert}$, which differs significantly from the base model's intermediate representations $X_k^{base}$. In the implementation (verified via the project's source code), the authors extract $X_k$ *only* from the base model, and then feed this base representation into the expert weights to compute $H_k = X_k W_k^T$. This is a severe conceptual mismatch: the expert weights $W_k$ were trained to process $X_k^{expert}$, not $X_k^{base}$. Feeding mismatched features into fine-tuned expert weights invalidates the computed salience scores $S_k[j]$.
2. **Single-Layer Gating Disconnect:** The paper is framed as a general adaptive multi-task model merging framework. However, the channel gating is applied **exclusively to a single layer**: the visual projection layer (`model.visual.proj`), which accounts for less than 0.5% of the network's total parameters. All other layers (the entire Multi-Head Attention and MLP layers of the Transformer backbone) are merged using standard, non-adaptive Task Arithmetic. This represents a massive disconnect. If inter-task weight conflicts occur across the entire model, gating a single projection bottleneck layer cannot solve interference in prior layers. The authors provide zero explanation, layer ablations, or empirical results justifying this extreme limitation.

### Presentation (Rating: Fair)
1. **Misleading and Non-Transparent Abstract Framing:** The abstract highlights that EdgeMerge "matches" a thoroughly optimized Task Arithmetic baseline (68.69% vs. 68.74%) and achieves a $50\times$ speedup over SyMerge. However, the abstract **completely conceals** the catastrophic performance gap between EdgeMerge and SyMerge. SyMerge achieves **89.74%** accuracy, which is over **21% absolute accuracy points higher** than EdgeMerge (68.69%). Presenting a model with a 21% performance degradation as "competitive" in the abstract, without explicitly stating the massive accuracy tradeoff, is disingenuous and intellectually dishonest.

### Significance (Rating: Poor)
1. **Practical Utility Squeeze:** The paper frames its contributions from a "pragmatic" perspective. However, there is no realistic scenario where a practitioner would choose EdgeMerge:
   - *If a practitioner has no data or needs instant merging,* they will choose **Task Arithmetic**, which requires zero calibration data, runs in ~0.1 seconds (100x faster than EdgeMerge's 11.95s), and achieves higher peak accuracy (68.74% vs 68.69%).
   - *If a practitioner needs a highly accurate multi-task model,* they will choose **SyMerge** or **FoldMerge** (scoring ~90%). A 10-minute one-time offline optimization process on a server before deployment is an extremely reasonable cost to prevent a catastrophic 21% drop in model accuracy.
   Consequently, EdgeMerge is squeezed out on both sides of the performance-cost trade-off.

### Originality (Rating: Fair)
1. **Incremental Formulation:** The core concept of utilizing average absolute activation scales over a calibration batch to measure weight importance is a standard and heavily researched technique in the network pruning (e.g., channel pruning) and dynamic routing literature. Applying this post-hoc to model merging represents an incremental combination of established ideas rather than a fundamentally new concept.

---

## 4. Key Critical Flaws (Up to 3)

### 1. The Representational Mismatch Flaw (Methodology)
The methodology computes expert activations $H_k = X_k W_k^T$ using intermediate features $X_k$ extracted from the pre-trained base model instead of the fine-tuned task expert. Because the upstream layers of the expert model were modified during fine-tuning, its representations $X_k^{expert}$ differ from the base representations $X_k^{base}$. Feeding base representations into fine-tuned expert weights introduces a severe feature-weight mismatch that invalidates the computed activation deltas and channel saliencies.

### 2. Extreme Gating Restriction (Generalizability & Scope)
The dynamic channel gating is restricted exclusively to a single visual projection layer (`model.visual.proj`) representing less than 0.5% of the total parameters, while over 99.5% of the model is merged using standard, non-adaptive Task Arithmetic. The paper fails to provide any mathematical or empirical justification for why gating a single layer is sufficient, and presents no ablation studies showing what happens if gating is applied to other layers.

### 3. Catastrophic Performance Trade-off Hype (Experiments & Presentation)
The paper heavily oversells EdgeMerge as a practical alternative to state-of-the-art adaptive methods by hiding a massive **21.05% absolute performance drop** compared to SyMerge (68.69% vs. 89.74%). Simultaneously, it fails to outperform standard Task Arithmetic (68.74%), which requires zero calibration data and is 100x faster to prepare, making the practical utility of the proposed method virtually non-existent.

---

## 5. Recommendations for Improvement
1. **Correct the Representational Mismatch:** To make the activation analysis mathematically sound, the features $X_k$ must be extracted independently for the base model and each task expert by running the calibration batch through their respective encoder layers. That is, compute $H_{base, k} = X_k^{base} W_{base}^T$ and $H_k = X_k^{expert} W_k^T$.
2. **Expand Gating across the Entire Model:** Implement and evaluate the channel-wise gating mechanism across other layer types (such as Multi-Head Attention projections and MLP layers) to demonstrate a truly general, model-wide routing solution. Provide comprehensive layer ablation studies.
3. **Revise the Narrative with Intellectual Honesty:** Be transparent about the performance-accuracy tradeoffs. The abstract and introduction must explicitly state that the forward-only, training-free approach incurs a substantial (~21%) accuracy penalty compared to gradient-based TTA methods, framing the work as an extreme-efficiency exploration rather than claiming it "matches" state-of-the-art capabilities.
