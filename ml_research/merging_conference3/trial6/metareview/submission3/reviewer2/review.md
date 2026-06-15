# Peer Review for Conference Submission

## Summary of the Paper
This paper investigates the problem of multi-task dynamic model merging, where specialized, task-specific expert weights fine-tuned from a common base model are dynamically blended at runtime using input-conditioned routing coefficients. To address the parameter scaling excess and "layer-to-layer coefficient ruggedness" (divergent and fluctuating routing decisions across sequential layers) associated with unshared layer-wise routing networks, the authors introduce the **Block-wise Weight-Sharing Router (BWS-Router)**. BWS-Router groups the $L$ layers of the model into $G = L / M$ uniform blocks and shares routing weights within each block, providing substantial parameter compression. Input features are compressed using unsupervised PCA, normalized onto the unit sphere, and projected to logits activated via independent Sigmoidal gating.

The authors evaluate BWS-Router in a virtual-layer "Task-Conflict Sandbox" (where routing coefficients are averaged across layers) and a "Physical Sequential Weight-Space Merging Framework" across 3-layer MLP experts (where representation propagation is sequential and weights are physically blended). Through extensive grid sweeps across 5 independent random seeds, the authors demonstrate that block-sharing reduces parameter footprints by up to 91.7% with minimal loss in sandbox performance, stabilizes dynamic ensembling compared to static uniform baselines, and outperforms unshared baselines under heterogeneous, mixed-task batching in physical sequential weight-space merging.

---

## Strengths and Weaknesses

### Strengths
1. **Exceptional Clarity and Structure:** The paper is highly polished, logically organized, and written with exceptional clarity. The figures and equations are clear, and the overall narrative flows smoothly.
2. **Move Towards Realism (Physical Sequential Merging):** The authors make a commendable effort to evaluate their method under a physical sequential weight-space merging environment, which is a significant step forward from virtual-layer ensembling sandboxes that average routing weights and ignore the sequential dynamics of feature propagation.
3. **Rigorous Empirical and Statistical Reporting:** The inclusion of means and standard deviations across 5 independent random seeds and the large-scale grid search of over 1,280 configurations demonstrate strong empirical diligence.
4. **Insightful and Self-Aware Appendix:** The appendix provides a refreshingly honest and comprehensive analysis of the optimization difficulties of independent Sigmoidal routing, its learning-rate sensitivity, and its relative sandbox underperformance compared to Softmax gating.

### Weaknesses
1. **Critical Algebraic Omission in Expected Ruggedness Proof:** The core "rigorous mathematical model" of Expected Ruggedness (Equation 10) contains an algebraic error. The derivation omits the squared difference between the expected values (means) of adjacent block routing coefficients, representing a significant technical gap in the paper's theoretical foundation.
2. **Lack of Formal Optimization or Convergence Guarantees:** The paper's core design elements (unsupervised PCA pre-projection, independent Sigmoidal gating, uniform negative bias initialization, light weight decay) are introduced as empirical heuristics. There are no formal proofs of convergence or mathematical derivations showing that these choices are optimal from an optimization perspective.
3. **Overclaimed Benefits of "Intermediate Block-Sharing Sizes":** The authors argue extensively that intermediate block-sharing sizes (such as $M=3, 4$) represent the optimal architectural "sweet spot" by preserving "coarse-to-fine functional specialization" while mitigating sequential representation drift. However, in their physical sequential weight-merging experiments, they use a 3-layer MLP expert backbone ($L=3$), which only permits block sizes of $M=1$ (fully unshared) or $M=3$ (fully shared global). Thus, **the authors never empirically test or validate intermediate block-sharing sizes inside the physical framework**, making their claims on this front speculative.
4. **Fragility and High Variance of Physical Merging:** The physical sequential weight-merging experiments exhibit extremely high variance (standard deviations exceeding 22%) and poor absolute performance compared to expert ceilings, demonstrating that the proposed block-sharing heuristic is an empirical stabilizer rather than a robust, theoretically grounded solution to representation drift.
5. **Toy Experimental Settings:** Both the simulated "Task-Conflict Sandbox" and the 3-layer MLP physical experts are highly simplified, low-dimensional toy systems. It is unclear if these findings scale or generalize to massive non-linear deep neural networks (such as Vision Transformers or LLMs) on standard real-world benchmarks.

---

## Detailed Evaluation of Soundness and Theoretical Gaps

### 1. Mathematical Error in Expected Ruggedness Formulation
In Section 3.3, the authors define Expected Ruggedness $\mathbb{E}[R(\alpha_k)]$ and present the following derivation in Equation 10:
$$\mathbb{E}[R(\alpha_k)] = \frac{1}{L-1} \sum_{g=1}^{G-1} \left( \sigma_{g+1}^2 + \sigma_g^2 - 2 \rho_g \sigma_g \sigma_{g+1} \right)$$
where $\sigma_g^2 = \operatorname{Var}(\bar{\alpha}_k^{(g)})$, and $\rho_g$ is the adjacent block correlation.

Let us rigorously expand the expectation of the squared difference of two adjacent block routing random variables $X = \bar{\alpha}_k^{(g+1)}$ and $Y = \bar{\alpha}_k^{(g)}$:
$$\mathbb{E}[(X - Y)^2] = \mathbb{E}[X^2] + \mathbb{E}[Y^2] - 2\mathbb{E}[XY]$$
By using the standard relationships $\mathbb{E}[Z^2] = \operatorname{Var}(Z) + (\mathbb{E}[Z])^2$ and $\mathbb{E}[XY] = \operatorname{Cov}(X, Y) + \mathbb{E}[X]\mathbb{E}[Y]$:
$$\mathbb{E}[(X - Y)^2] = \operatorname{Var}(X) + (\mathbb{E}[X])^2 + \operatorname{Var}(Y) + (\mathbb{E}[Y])^2 - 2\left(\operatorname{Cov}(X, Y) + \mathbb{E}[X]\mathbb{E}[Y]\right)$$
$$\mathbb{E}[(X - Y)^2] = \operatorname{Var}(X) + \operatorname{Var}(Y) - 2\operatorname{Cov}(X, Y) + \left(\mathbb{E}[X] - \mathbb{E}[Y]\right)^2$$
Substituting the paper's notation where $\operatorname{Var}(X) = \sigma_{g+1}^2$, $\operatorname{Var}(Y) = \sigma_g^2$, and $\operatorname{Cov}(X, Y) = \rho_g \sigma_g \sigma_{g+1}$:
$$\mathbb{E}\left[\left( \bar{\alpha}_k^{(g+1)} - \bar{\alpha}_k^{(g)} \right)^2\right] = \sigma_{g+1}^2 + \sigma_g^2 - 2 \rho_g \sigma_g \sigma_{g+1} + \left( \mathbb{E}[\bar{\alpha}_k^{(g+1)}] - \mathbb{E}[\bar{\alpha}_k^{(g)}] \right)^2$$

In Equation 10, the authors have **entirely omitted** the last term: $\left( \mathbb{E}[\bar{\alpha}_k^{(g+1)}] - \mathbb{E}[\bar{\alpha}_k^{(g)}] \right)^2$, which is the squared difference between the expected values (means) of adjacent block routing coefficients. 
This omission is a significant technical flaw. It implicitly assumes that:
$$\mathbb{E}[\bar{\alpha}_k^{(g+1)}] = \mathbb{E}[\bar{\alpha}_k^{(g)}] \quad \forall g \in \{1, \dots, G-1\}$$
This assumption is theoretically and physically incorrect. In physical deep neural networks, early layers learn generic, task-agnostic features, while deeper layers capture specialized semantic features. Consequently, the routing behavior (and thus the expected values of the routing coefficients) must systematically differ across blocks (e.g., uniform routing in early blocks, sparse/decisive routing in deep blocks). Omitting this mean difference term oversimplifies the expected ruggedness and fails to capture the systematic shift in routing expectations across layer blocks, undermining the mathematical soundness of their "mitigation proof."

### 2. Lack of Formal Bounds on Sequential Representation Drift
The authors claim that BWS-Router "prevents cascading representation drift" and acts as a "low-pass filter on representation drift." However, these claims are supported only by qualitative analogies rather than mathematical proofs. To be theoretically convincing, the paper must provide formal bounds showing how tying routing weights across layers bounds the cumulative representation divergence or stabilizes the Lipschitz constant of the sequential transformations relative to an unshared baseline. Without this, the claim is merely a speculative empirical observation.

### 3. Heuristic Design Principles
The core methodology relies entirely on empirical heuristics that lack theoretical justification:
- **PCA Pre-Projection:** The choice of unsupervised PCA for feature compression is heuristic. There is no proof showing that the principal components of the representation space align with the optimal directions for task routing or expert selection.
- **Sigmoidal Gating Sluggishness:** The choice of Sigmoidal gating requires highly specific, tuned learning rates ($\eta = 0.05$) and weight decay ($\lambda_{wd} = 10^{-4}$) to prevent collapse, and is actually outperformed by Softmax in the virtual sandbox (80.56% vs. 79.50%). The authors' "open-world" defense of Sigmoidal gating is qualitative and lacks a rigorous optimization-based or information-theoretic foundation.

---

## Detailed Evaluation of Experiments and Evaluation

### 1. Inhomogeneous Superiority in Stable Physical Streams
In Table 4 (physical sequential merging), under the Homogeneous stream, the unshared physical router ($M=1$) actually **outperforms** the block-shared physical router ($M=3$) by 2.78% absolute accuracy (**48.04 $\pm$ 6.71%** vs. **45.26 $\pm$ 10.11%**). This result is highly critical because it directly contradicts the main thesis that block-sharing is universally superior. It demonstrates that when task streams are stable, unshared routing is preferred because it allows fine-grained layer-wise specialization, and block-sharing acts as a restrictive constraint that degrades performance.

### 2. Speculative Validation of Intermediate Block-Sharing Sizes
As noted, the authors' physical experiments are conducted using a 3-layer MLP expert backbone ($L=3$), which only permits block sizes of $M=1$ (unshared) or $M=3$ (fully global). Therefore, **the authors never empirically test or validate intermediate block-sharing sizes in physical sequential propagation.** Their arguments about intermediate block sizes representing the optimal "sweet spot" that preserves "coarse-to-fine functional specialization" while mitigating drift are entirely unvalidated by their physical experiments. To validate this claim, they must evaluate on a deeper backbone (e.g., $L \ge 12$) to compare intermediate sizes ($M=2, 3, 4, 6$) against the global baseline ($M=12$).

### 3. Severe Fragility and High Variance of Physical Merging
The physical sequential weight-merging results in Table 4 show extremely large standard deviations: **22.49%** for $M=3$ Heterogeneous and **21.28%** for $M=1$ Heterogeneous. Under certain seeds, the model collapses completely (e.g., CIFAR-10 accuracy dropping to 3.30%). Furthermore, the overall accuracies achieved (43%--48%) are extremely poor compared to the Expert Ceiling (~81%) or the virtual sandbox (~79%). This highlights that physical sequential weight-blending remains highly fragile and fundamentally unsolved. The proposed block-sharing heuristic does not provide a robust, mathematically guaranteed solution; it merely reduces variance in one stream relative to a very poor baseline, while still suffering from catastrophic seed-dependent fluctuations.

### 4. Sandbox Global Router Superiority
In the block-wise sensitivity sweep inside the sandbox (Table 3), the global router ($M=12$, using only 20 parameters) achieves the highest Joint Mean accuracy of **79.60 $\pm$ 1.15%**, outperforming the unshared baseline ($M=1$, 79.30%) and matching or slightly outperforming intermediate sizes ($M=3$, 79.57%). If a single global router achieves peak performance while using 75% fewer parameters than $M=3$, the complexity of implementing block-wise weight sharing is empirically unjustified within their primary sandbox setup.

---

## Rating and Overall Recommendation

### Overall Recommendation: 3: Weak Reject
*A paper with clear merits, but also some weaknesses, which overall outweigh the merits. Papers in this category require revisions before they can be meaningfully built upon by others.*

**Justification:** While the paper is beautifully written and introduces a highly practical parameter-compression scheme (block-wise weight sharing) alongside a valuable physical sequential merging framework, its weaknesses currently outweigh its merits from a rigorous theoretical perspective. The core mathematical derivation of Expected Ruggedness (Equation 10) contains a significant algebraic omission that assumes constant expected routing decisions across sequential layers—an assumption that is physically and theoretically incorrect. Furthermore, the claims regarding intermediate block-sharing sizes representing the optimal architectural sweet spot are completely unvalidated in the physical sequential experiments, where only $M=1$ and $M=3$ are tested on a toy 3-layer MLP. Finally, the massive variance (exceeding 22% standard deviation) and poor absolute performance of physical sequential weight merging demonstrate that BWS-Router is an empirical heuristic stabilizer rather than a robust, mathematically guaranteed solution. Correcting the mathematical proof, evaluating intermediate block sizes on a deeper physical model (e.g., $L \ge 12$), and providing formal bounds on representation drift are necessary revisions before this work is ready for publication.

### Ratings Breakdown
- **Soundness:** **Fair** (Clear methodology, but contains a mathematical error in Expected Ruggedness derivation, and lacks formal convergence proofs or representation drift bounds).
- **Presentation:** **Excellent** (Highly polished, clear figures, elegant formatting, and structurally sound).
- **Significance:** **Fair** (Practical parameter compression, but limited by toy low-dimensional experimental setups and highly unstable physical ensembling results).
- **Originality:** **Fair** (Tying weights across layers is a standard, established deep learning technique; its application to dynamic model merging routers is an incremental, intuitive extension rather than a major conceptual breakthrough).

---

## Questions and Constructive Feedback for the Authors

1. **Correction of Expected Ruggedness Proof:** Can you explain why the mean difference term $\left( \mathbb{E}[\bar{\alpha}_k^{(g+1)}] - \mathbb{E}[\bar{\alpha}_k^{(g)}] \right)^2$ was omitted from the Expected Ruggedness derivation in Equation 10? If we assume $\mathbb{E}[\bar{\alpha}_k^{(g+1)}] \neq \mathbb{E}[\bar{\alpha}_k^{(g)}]$ due to standard depth-dependent functional specialization, how does this extra term impact your theoretical claims about block sharing?
2. **Evaluation of Deeper Physical Backbones:** To validate your core claim that intermediate block sizes (e.g., $M=3$ or $M=4$) represent the optimal architectural "sweet spot," can you evaluate BWS-Router on a deeper sequential backbone (e.g., a 12-layer MLP or a standard Vision Transformer) where intermediate block configurations can actually be tested and compared against unshared ($M=1$) and global ($M=12$) physical routers?
3. **Formal Bounds on Cascading Representation Drift:** Rather than relying on qualitative low-pass filter analogies, can you provide a formal mathematical bound showing how tying routing parameters across layers restricts cumulative representation divergence or bounds the Lipschitz constant of sequential transformations under runtime physical weight-blending?
4. **Physical Experiments for the Global Router:** Why are there no physical sequential weight-space merging experiments for the global router baseline ($M=3$ is evaluated, but on a 3-layer MLP $M=3$ *is* the global router)? Comparing unshared $M=1$ against global $M=3$ does not show the behavior of intermediate block sizes.
5. **Real-world Generalization:** How does BWS-Router perform when scaled to real deep neural networks (e.g., ResNets, ViTs, or LLaMAs) on standard downstream datasets, rather than being restricted to a custom 192-dimensional simulated representation sandbox and 3-layer MLP experts?
