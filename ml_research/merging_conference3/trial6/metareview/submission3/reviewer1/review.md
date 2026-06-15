# Peer Review

## Summary of the Paper
This paper deconstructs the capacity-generalization trade-off in dynamic weight-space model merging. Existing dynamic model merging techniques, such as the Layer-wise Low-dimensional Classical Router (L3-Router), employ independent, unshared routing networks for each of the $L$ layers. The authors identify that these unshared structures introduce high parameter overhead, cascade representation drift, and exhibit layer-to-layer weight blending fluctuations ("coefficient ruggedness") under scarce calibration splits. 

To address these limitations, the paper introduces the **Block-wise Weight-Sharing Router (BWS-Router)**. BWS-Router groups the $L$ layers into $G = L / M$ uniform blocks and shares routing weights within each block. By combining block sharing with an unsupervised PCA pre-projector and independent Sigmoidal gating, BWS-Router dramatically compresses the routing parameter footprint (by 66.7% to 91.7%) and stabilizes weight ensembling. 

The authors evaluate their method inside a simulated Task-Conflict Sandbox, on a physical sequential weight-space model-merging system (using 3-layer MLP experts), and via a CPU latency profiling pilot on a physical Vision Transformer backbone. The results demonstrate that BWS-Router matches or outperforms high-capacity unshared baselines while requiring only a fraction of the parameter budget, and drastically stabilizes sequential propagation under task-heterogeneous shifts.

---

## Strengths and Weaknesses

### Strengths:
1. **Strong Theoretical Motivation:** The paper's core motivation—that independent layer-wise routers are overparameterized and suffer from layer-to-layer optimization ruggedness and representation drift—is conceptually compelling and mathematically formalized. The attempt to model expected ruggedness using Depth-Dependent Variance Scales and Adjacent Layer Correlation is highly commendable.
2. **Exceptional Empirical Completeness:** The paper provides an incredibly thorough empirical validation. It sweeps over 1,280 grid configurations across 5 independent random seeds and includes comprehensive ablations on block size sensitivity, gating activations, learning rates, weight decays, task scaling ceilings ($\lambda_{max}$), gating bias initializations, PCA dimensions, projection kernels, residual links, sequential smoothing regularizations, sample complexity, and expert scaling (up to $K=10$ expert tasks).
3. **Intellectual Honesty and Transparency:** The authors provide a highly transparent discussion of their results. They openly discuss the high variance observed in physical sequential propagation, the optimization sluggishness of Sigmoidal gating, and why Softmax gating empirically outperforms Sigmoid in closed classification proxies while defending Sigmoid's superiority for open-world deployments.
4. **Transition to Physical Sequential weight blending:** Unlike typical model-merging works that evaluate purely simulated virtual ensembling (where coefficients are averaged), this work moves to a physical sequential weight-blending system on multi-layer MLP experts, evaluating how representations propagate sequentially.
5. **Highly Practical Bridge to Large Foundation Models:** The proposed "Bridge to Physical Model Merging" implementation recipe for deep ViTs (Section 5) and the quantitative parameter/computational savings analysis for CLIP and LLaMA-2 make this work exceptionally valuable for downstream practitioners.

### Weaknesses:
1. **Mathematical Oversimplification in Expected Ruggedness Expansion:** In Section 3.3, Equation 7 expands the expected value of the squared adjacent block-wise routing coefficient difference. The expansion *completely omits* the squared difference of the expectations, which is a key mathematical discrepancy. (See Soundness for a detailed mathematical analysis).
2. **Discrepancy in Zero-Difference Claim for Sequential Propagation:** The paper claims that under BWS-Router, adjacent layer routing coefficients inside the same block group have exactly zero difference. While true in the virtual sandbox, this is not true in physical sequential propagation, where representations are transformed layer-by-layer, changing the input to the router. (See Soundness for a detailed analysis).
3. **Absence of Formal Learning and Convergence Guarantees:** Despite modeling Expected Ruggedness, the paper lacks formal statistical learning guarantees (e.g., Rademacher complexity bounds showing that block-sharing reduces overfitting) or convergence rate guarantees under the sluggish, bounded Sigmoidal gating function.
4. **Main Performance Sweeps rely on Simulated Sandbox:** The vast majority of the quantitative analyses are conducted within a synthetic task-conflict representation sandbox with linear classification heads. While highly appropriate as a high-throughput proxy, verifying the classification accuracy of BWS-Router on actual physical foundation models (e.g., CLIP-ViT or LLaMA-2) fine-tuned on real vision/NLP datasets remains a critical next step.

---

## Soundness
**Rating: Good**

### Justification:
The methodology and empirical results are highly robust, and the paper is exceptionally thorough. However, from a theory-minded perspective, there are two major theoretical gaps and oversimplifications that prevent an "Excellent" rating:

#### 1. Mathematical Error in Expected Ruggedness Expansion (Equation 7)
The authors expand the expectation of adjacent block-wise routing coefficient differences as:
$$\mathbb{E}\left[ \left( \bar{\alpha}_k^{(g+1)} - \bar{\alpha}_k^{(g)} \right)^2 \right] = \sigma_{g+1}^2 + \sigma_g^2 - 2 \rho_g \sigma_g \sigma_{g+1}$$
However, the standard expansion of the expected value of a squared difference is:
$$\mathbb{E}[(X - Y)^2] = \operatorname{Var}(X) + \operatorname{Var}(Y) - 2 \operatorname{Cov}(X, Y) + \left( \mathbb{E}[X] - \mathbb{E}[Y] \right)^2$$
The expansion in Equation 7 **completely omits** the squared difference of the expectations: $\left( \mathbb{E}[\bar{\alpha}_k^{(g+1)}] - \mathbb{E}[\bar{\alpha}_k^{(g)}] \right)^2$.

This omission is mathematically correct if and only if the expected value of the routing coefficients is constant across successive block groups (i.e., $\mathbb{E}[\bar{\alpha}_k^{(g+1)}] = \mathbb{E}[\bar{\alpha}_k^{(g)}]$). However, this contradicts the authors' own discussion of "Coarse-to-Fine Functional Specialization" in Section 4.3, where they explain that shallow layers extract generic representations whereas deep layers specialize in semantic features, which naturally alters their expected routing weights. The authors must explicitly state and justify this unstated assumption, or correct the equation.

#### 2. Discrepancy in the Zero-Difference Claim inside Block Groups
The authors state that layers within the same block group $g$ share identical routing weights, making their coefficient difference exactly zero: $\bar{\alpha}_k^{(l+1)} - \bar{\alpha}_k^{(l)} = 0$ for all $l, l+1 \in \mathcal{G}_g$. 

While this is true in their Virtual Sandbox (where a single global representation $\psi(x)_b$ is fed to the router at all depths), it does **not** hold in the Physical Sequential Weight-Space Merging setup (Section 3.4). In physical propagation, the intermediate representation $h_b^{(l-1)}$ is transformed layer-by-layer:
$$h_b^{(l)} = \text{ReLU}\left( h_b^{(l-1)} W_{merged, b}^{(l) T} + B_{merged, b}^{(l)} \right)$$
Because the hidden representation changes layer-by-layer, the layer-specific input to the router $\psi_b^{(l)} = \text{PCA}^{(l)}(h_b^{(l-1)})$ also changes. Consequently, even when sharing the routing parameters $W_{group}^{(g)}$ within a block, the predicted coefficients are different:
$$\alpha_{k, b}^{(l)} = \text{gating}(\psi_b^{(l)})_k \neq \text{gating}(\psi_b^{(l+1)})_k = \alpha_{k, b}^{(l+1)}$$
Thus, in actual physical sequential propagation, the coefficient differences inside blocks are not zero. The authors' claim that BWS-Router "structurally clips layer-to-layer weight blending fluctuations to exactly zero" is a methodological artifact of the virtual sandbox, and does not hold strictly in physical deep deployments. This gap between the stylized theoretical model and physical reality must be clearly acknowledged.

---

## Presentation
**Rating: Excellent**

### Justification:
The paper is beautifully written, extremely well-structured, and highly technical. The authors maintain a highly professional and scholarly tone throughout. The architectural schematic (Figure 1) is exceptionally professional, and the tables and plots are clear, comprehensive, and informative. The Appendix is incredibly detailed, offering outstanding transparency and actionable engineering recipes.

---

## Significance
**Rating: Excellent**

### Justification:
Deploying multiple specialized foundation models in production leads to linear scaling of computational and memory overhead. Weight-space model ensembling is a zero-inference-latency solution to this problem. 

By proving that highly compressed, block-shared routing weights (slashing parameters and routing forward passes by over 94% to 96% for CLIP and LLaMA-2) match or exceed unshared performance while significantly stabilizing sequential propagation, this paper provides a much-needed foundation for scaling dynamic ensembling to massive modern architectures. The significance of this contribution to the model merging and PEFT communities is exceptionally high.

---

## Originality
**Rating: Good**

### Justification:
While block-wise parameter sharing and PCA pre-projection are established neural network and dimension reduction techniques, their synthesis and application to solve the unique problem of layer-to-layer coefficient ruggedness and cascading representation drift in dynamic weight merging is novel. The deconstructive, grounded analysis represents a highly original and valuable contribution that rejects speculative mathematical metaphors in favor of solid, rigorous engineering and empirical mapping.

---

## Questions for the Authors / Requested Clarifications

1. **Correction of Expected Ruggedness Expansion:** Please address the omission of the expectation-difference term $\left( \mathbb{E}[\bar{\alpha}_k^{(g+1)}] - \mathbb{E}[\bar{\alpha}_k^{(g)}] \right)^2$ in the derivation of Expected Ruggedness in Equation 7. If you assume this term is zero, please explicitly state and theoretically justify this assumption.
2. **Clarification on Coefficient Differences inside Blocks:** Please clarify that the layer-to-layer coefficient difference within a block group is exactly zero only in the virtual sandbox where inputs are static. In physical sequential deep propagation, because intermediate representations are transformed at each layer, the input to the router varies, leading to non-zero coefficient differences even when router weights are shared.
3. **QWS-Merge Instability Conditions:** In Section 4.2, you describe QWS-Merge as suffering from "frequent training collapse across random seeds" and "extreme non-convexity," yet Table 1 shows that QWS-Merge achieves a highly competitive Joint Mean of $78.07 \pm 1.06\%$, which is statistically stable. Please clarify under what specific conditions (e.g., higher learning rates, lack of regularization) QWS-Merge exhibits optimization collapse.
4. **Generalization on Foundation Models:** While the latency profiling on `vit_tiny` is highly valuable, do you have any preliminary or qualitative results on whether the classification accuracy of BWS-Router holds on real vision/NLP checkpoints (such as CLIP-ViT-B/16 or LLaMA-2) fine-tuned on real multi-task datasets, rather than relying on synthetic task-vectors?

---

## Overall Recommendation
**Rating: 5: Accept**

### Justification:
This is a technically solid, highly thorough, and exceptionally well-written paper that addresses a highly relevant and important problem in parameter-efficient multi-task adaptation. 

The proposal of Block-wise Weight-Sharing Router (BWS-Router) is highly practical, achieving up to a 91.7% reduction in parameters and routing forward passes with absolutely zero loss in performance, while significantly stabilizing deep sequential propagation under task-heterogeneous batch shifts. While there are a few minor mathematical oversimplifications and discrepancies between the stylized sandbox theory and physical sequential propagation, these do not invalidate the core empirical and conceptual contributions of the work. If the authors address the requested clarifications and correct/justify the mathematical derivations, this paper will represent an outstanding addition to the conference.
