# Peer Review of Conference Submission

## Summary of the Paper
The paper introduces **Fisher-Information Optimal Subspace Routing (FIOSR)**, a training-free and parameter-free dynamic ensembling framework designed for test-time model merging. In test-time model merging, specialized domain-expert weights (such as LoRA adapters) sharing a common pre-trained backbone are dynamically ensembled to handle heterogeneous input streams without manual task boundaries. Existing dynamic routing methods either rely on parametric routers that suffer from severe overfitting under extreme calibration data scarcity ("The Dynamic Routing Paradox") or exhibit high instability under single-sample sequential streams ("Vectorization Collapse"). Prior parameter-free subspace routing (PFSR) methods resolve the optimization bottleneck by projecting representations onto frozen class prototypes, but their reliance on unweighted cosine similarity assumes a flat, isotropic weight space.

FIOSR models the parameter and representation spaces as Riemannian manifolds, where local geometries are defined by a smoothed and power-scaled diagonal empirical Fisher Information Matrix (dFIM). Derived over a microscopic calibration split, the dFIM warps the projection coordinate space during inference, suppressing noisy task-irrelevant feature dimensions (high variance/low Fisher) and magnifying highly informative task features (low variance/high Fisher). Additionally, the framework incorporates Class-Size Scaling Calibration (CSC) to correct for statistical maximum bias under asymmetric class vocabularies, and Micro-Batch Homogenization (MBH) to partition heterogeneous streams into homogeneous micro-batches to safeguard against "heterogeneity collapse."

The authors evaluate FIOSR in three settings:
1. A 192-dimensional synthetic "Analytical Coordinate Sandbox" modeling MNIST, FashionMNIST, CIFAR-10, and SVHN equivalents.
2. A 64-dimensional LoRA activation space simulation using realistic, anisotropic feature variances.
3. An end-to-end physical validation on a pre-trained ResNet-18 backbone evaluated on MNIST, FashionMNIST, and SVHN.

---

## Strengths and Weaknesses

### Strengths
1. **Elegant Theoretical Formulation:** The paper combines information geometry, Riemannian manifolds, and test-time model merging in a highly rigorous and beautiful mathematical framework. Connecting diagonal empirical Fisher coordinates directly to inverse coordinate-wise noise variances under conditional Gaussian assumptions is elegant, and the dual-space proof in Appendix A.3 formally bounding classifier weight and activation centroid alignment is a strong theoretical contribution.
2. **Optimization-Free Robustness:** By bypassing test-time parameter optimization entirely, the framework is completely immune to the overfitting and instability collapses that plague parametric routers. It maintains a stable performance flatline across all batching regimes ($B=1$ to $512$).
3. **Exceptional Transparency and Self-Awareness:** The authors must be highly commended for their thorough and honest discussion of seven key limitations, system-level trade-offs, and practical constraints in Appendix A.1. This level of rigorous self-critique is rare and highly valuable.
4. **Concrete Engineering Safeguards:** The addition of CSC (extreme-value calibration for asymmetric class vocabularies) and MBH (unsupervised batch partitioning for stream heterogeneity) directly targets realistic, practical deployment challenges rather than ignoring them.
5. **Outstanding Presentation Quality:** The paper is written with extreme clarity, logical flow, and exhaustive documentation. Hyperparameters, experimental setups, and mathematical proofs are meticulously detailed, ensuring exceptional reproducibility.

### Weaknesses
1. **Severe Systems-Level Latency Bottleneck in MBH:** Micro-Batch Homogenization (MBH) partitions a heterogeneous batch of size $B$ into $G \le K$ homogeneous micro-batches, executing $G$ separate sequential forward passes with dynamically assembled weights. In a highly diverse, non-stationary streaming environment, we encounter the worst-case scenario where $G = K$ micro-batches are formed. From a systems-engineering perspective, running $K$ sequential forward passes with $K$ distinct dynamically merged models is computationally equivalent (in terms of FLOPs) to simply executing the original, unmerged specialized expert models on their respective samples. This completely defeats the primary objective of test-time weight merging (which is to run a single merged model to save execution computation). Dynamic weight assembly, routing coefficient calculation, and scatter-gather operations on the fly further exacerbate latency and memory bandwidth bottlenecks. Bounding this via Top-1 expert gating ($M=1$) completely eliminates sequential MBH overhead, but collapses the framework back to a hard task-routing selection mechanism, losing the benefits of parameter-space ensembling for that sample. This represents a severe systems-level ceiling for real-world low-latency streaming applications in industry.
2. **Underwhelming Empirical Gains on Real-World Backbones:** While FIOSR achieves dramatic performance gains in the synthetic "Analytical Coordinate Sandbox" ($+8.56\%$ accuracy over flat Cosine), its performance on an actual physical ResNet-18 network (Section 4.8) is underwhelming and indicates a massive performance gap. Specifically, routing accuracy improves by only **+2.67%** (59.00% vs. 56.33%) and joint ensembling accuracy improves by a negligible **+1.33%** (52.00% vs. 50.67%). Furthermore, both methods fall **17.67%** short of the Direct Expert Routing Oracle (69.67%). This stark contrast suggests that under realistic, highly correlated, non-Gaussian representation spaces, diagonal Fisher-weighted coordinate warping provides very limited practical utility. The substantial computational and mathematical overhead of FIM estimation and smoothing yields negligible real-world returns.
3. **Calibration Dependency (Zero-Shot Limitation):** Although the method is parameter-free, it still requires a calibration split of $N_c \ge 8$ samples per task to stable-estimate the coordinate variances. As shown in Appendix B.3, if $N_c \le 4$, the variance estimation is mathematically underdetermined, causing FIOSR to overfit to calibration noise and perform significantly worse than flat Cosine (-9.48% absolute loss at $N_c=2$). This calibration dependency prevents immediate, zero-shot out-of-the-box deployment for new, uncalibrated tasks in highly dynamic environments.
4. **Lack of Empirical Validation for Large Vocabulary Scaling:** While the authors propose "Class-Grouped Pooling" and "Low-Rank FIM Factorization" to mitigate the storage and memory overhead of storing $K \times C \times d$ Fisher coefficients for large language models (LLMs) with massive vocabularies ($32\text{K}$ to $128\text{K}$ tokens), these compression strategies are purely theoretical and lack any empirical validation.

---

## Soundness
**Rating: Good**

**Justification:** The mathematical derivations are highly rigorous, and the authors go to great lengths to theoretically address non-Gaussianity/ReLU sparsity (Appendix A.2) and dual-space alignment bounding (Appendix A.3). However, the technical soundness is slightly limited by:
- The diagonal approximation of the Fisher Information Matrix, which assumes coordinate independence. Under rotated, non-axis-aligned noise (Section 4.6), diagonal Fisher collapses below the flat Cosine baseline (67.38% vs 67.50%), proving that diagonal coordinate warping is fragile and highly sensitive to coordinate correlation. Full covariance estimation (which is computationally expensive and does not scale) is required to restore the gains.
- The highly contrived nature of the synthetic "Analytical Coordinate Sandbox," where noise is deterministic, axis-aligned, and perfectly independent by design—representing a setup where diagonal Fisher weighting is guaranteed to excel.

---

## Presentation
**Rating: Excellent**

**Justification:** The writing is impeccable, highly professional, and logically structured. The mathematical notations are clean and clearly defined in a dedicated table (Appendix A.1). The inclusion of pseudocode and explicit details on training linear classifiers on ResNet-18 features ensures that the entire lifecycle of the proposed framework can be effortlessly reproduced. The authors' thorough and transparent discussion of limitations and system-level trade-offs is exemplary.

---

## Significance
**Rating: Fair**

**Justification:** While the theoretical concepts are highly elegant and will certainly influence future researchers to explore Riemannian geometries in modular deep learning, the practical utility of this work is heavily constrained:
- Micro-Batch Homogenization (MBH) introduces sequential forward pass overhead that defeats the core computational advantage of model merging under highly diverse streams.
- The empirical gains on actual physical networks are extremely modest ($+1.33\%$ joint accuracy), indicating that the practical performance improvement on real architectures may not justify the added operational complexity of on-the-fly FIM estimation and streaming synchronization.

---

## Originality
**Rating: Good**

**Justification:** The concept of using diagonal Fisher Information dynamically at test-time as a coordinate-warping metric tensor to warp representation spaces is highly creative and novel. Combining this information-geometric warping with CSC and MBH to address asymmetric class vocabularies and stream non-stationarity is an original and well-articulated synthesis of statistical and systems concepts.

---

## Overall Recommendation
**Score: 3: Weak reject**

**Justification:** 
The submission features a highly elegant mathematical framework, clear writing, and commendable empirical thoroughness. However, from a practical applications and real-world deployment perspective, the core weaknesses outweigh the merits:
1. **The MBH Systems Bottleneck:** Dynamic test-time model merging is motivated by the desire to run a single, merged model to handle multiple domains, saving computational FLOPs. In a heterogeneous stream, MBH partitions the batch and requires up to $K$ sequential forward passes. This is computationally equivalent to simply running the original, unmerged specialized experts on their respective samples, which completely negates the primary motivation of model merging.
2. **Real-World Performance Gap:** On actual physical ResNet-18 features, the joint ensembling accuracy gain is an insignificant **+1.33%**, which is extremely small compared to the **+8.56%** reported in the highly contrived synthetic sandbox. This indicates that diagonal coordinate warping fails to capture the complex, non-axis-aligned covariance structures of real-world deep neural network activations, rendering the mathematical overhead of FIM estimation of limited practical value.
3. **Calibration Constraint:** The dependency on a calibration split of $N_c \ge 8$ samples per task prevents true zero-shot deployment, adding operational complexity in highly dynamic streaming pipelines.

To be suitable for publication, the paper requires revisions that address these severe practical limitations. Specifically, the authors must empirically validate scalable alternatives to MBH that do not suffer from sequential forward pass overhead under high heterogeneity, and demonstrate more significant, robust gains on realistic, physical networks without assuming oracle rotation matrices.

---

## Questions and Actionable Feedback for the Authors

1. **Scalable Micro-Batching alternatives:** Can you propose or empirically evaluate an alternative to MBH that does not scale sequentially with the number of experts $K$ under highly heterogeneous streams? While you discuss Top-$M$ gating, setting $M=1$ completely collapses ensembling back to a hard task-routing mechanism. Is there a way to perform parallel multi-expert ensembling in a single forward pass without sequential partitioning or collapsing back to hard routing?
2. **Rotated and Correlated Noise scaling:** Your rotated noise experiment (Section 4.6) shows that diagonal Fisher collapses below the flat baseline unless full covariance estimation (FIOSR-Online / FIOSR-Rotated) is applied. However, performing eigenvalue decomposition of a full $d \times d$ covariance matrix is computationally prohibitive for high-dimensional hidden representations (e.g., $d \ge 1024$). How do you propose to scale covariance alignment to massive representation spaces without introducing prohibitive computational latency?
3. **Empirical Validation of LLM Compression:** Storing $K \times C \times d$ Fisher coefficients for LLMs is a significant memory bottleneck. Can you provide empirical results validating either "Class-Grouped Pooling" or "Low-Rank FIM Factorization" on a realistic, small-scale language model (e.g., a 125M parameter model) to prove that these compression strategies preserve routing stability?
4. **Physical Backbone Performance:** Why do you think the joint ensembling accuracy improvement on the physical ResNet-18 backbone is so small (+1.33%) compared to the synthetic sandbox (+8.56%)? Are there architectural normalizers (such as LayerNorm or GroupNorm) or ReLU-induced dead dimensions that are actively disrupting the diagonal Fisher coordinate warp? Providing a detailed empirical or qualitative analysis of this physical gap would significantly improve the practical value of the paper.
