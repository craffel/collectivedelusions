# Peer Review of Conference Submission

## 1. Summary of the Paper
The paper addresses the challenge of parameter-space model merging for compact networks in high-conflict, multi-task visual domains (MNIST, FashionMNIST, CIFAR-10, SVHN). Standard model merging methods seek a static consensus in weight-space, which the authors show leads to "catastrophic representational collapse" on a compact 5.7M-parameter Vision Transformer backbone. 

To resolve this, the authors propose **Quantum Wavefunction Superposition Merging (QWS-Merge)**, a quantum-inspired dynamic merging approach. QWS-Merge models fine-tuned expert weights as task eigenstates in a parameter Hilbert space. Merging is represented as a coherent quantum-like superposition that dynamically collapses to a localized classical weight configuration based on the wave-like phase-interference of incoming features. The input phase state is extracted via random projection and spherical normalization, and dynamic coefficients are calculated via layer-wise trainable phase-basis vectors, scaling amplitudes, and phase biases.

The method is highly parameter-efficient (336 parameters) and is calibrated on a tiny 64-sample offline validation set. The authors demonstrate that QWS-Merge outperforms uniform merging and unsupervised AdaMerging on homogeneous streams, and exhibits strong regularization on the highly conflicting SVHN dataset compared to an unregularized classical Linear Router. They also transparently document and analyze the "heterogeneity collapse" that dynamic routers suffer from at larger batch sizes under mixed-task streams.

---

## 2. Strengths and Weaknesses

### Strengths:
- **Creative Formulation**: Porting physical and quantum principles (superposition, wavefunction collapse, and phase-coherence) to parameter-space model merging is highly creative and original. It provides a unique non-monotonic cosine-similarity-based activation mechanism for routing.
- **Extreme Parameter and Sample Efficiency**: Calibrating a dynamic router using only 336 parameters on a tiny validation set of 64 samples in under 30 seconds is highly impressive, showcasing excellent feasibility for data-scarce and resource-constrained environments.
- **Outstanding Scientific Honesty**: The authors deserve strong praise for their rigorous and transparent investigation of the "heterogeneity collapse" under mixed-task streams. Instead of obfuscating this limitation, they dedicate Table 2 and Figure 2 to documenting and analyzing why dynamic routers degrade at larger batch sizes on mixed streams. They also transparently discuss the I.I.D. violation in their limitations section.
- **Strong Capacity Preservation on SVHN**: QWS-Merge maintains $31.60\%$ accuracy on SVHN, preserving $91.5\%$ of the expert ceiling, and dramatically outperforming the classical Linear Router baseline ($15.30\%$).

### Weaknesses:
- **Lack of Statistical Robustness**: The paper reports only single point estimates without standard deviations, error bars, or confidence intervals. Given that the calibration set contains only 64 samples, the optimization of both the Linear Router and QWS-Merge is highly susceptible to variance. Evaluating over multiple random seeds and calibration splits is critical to confirm the statistical soundness of the results.
- **Conflated Variables and "Strawman" Baseline**: The comparison between QWS-Merge and the classical Linear Router baseline is conflated:
  1. QWS-Merge has layer-wise routing, whereas the Linear Router is restricted to global routing. Its SVHN advantage could be due to this layer-wise flexibility rather than the "quantum wave-like projection."
  2. The Linear Router (772 parameters) is completely unregularized on the tiny 64-sample dataset, making it a "strawman" baseline. Comparing QWS-Merge against a regularized (e.g., L2/weight decay) or parameter-constrained Linear Router is necessary to validate the "wave-like subspace regularization" claim.
- **Suboptimal Expert Performance**: The reported SVHN specialized expert ceiling is only $34.50\%$. For a standard Vision Transformer, SVHN is typically a relatively easy task (often yielding $>90\%$ accuracy). A ceiling of $34.50\%$ suggests that the underlying experts were trained with highly suboptimal hyperparameters or failed to converge. This undermines the credibility of the downstream merging evaluation on SVHN.
- **Practical Utility and Collapse Under Heterogeneous Streams**: Table 2 reveals that in the only scenario where dynamic routing is practically required (mixed-task streams where task identities are unknown), QWS-Merge collapses to $48.70\%$ accuracy at $B=256$. This is significantly worse than simple static merging methods like AdaMerging ($57.20\%$) and OFS-Tune ($55.60\%$). Consequently, in its current formulation, the method does not offer practical advantages over existing static baselines for real-world mixed deployment.
- **Absence of Critical Ablations**: The paper lacks any ablation studies or sensitivity analyses on key hyperparameters (such as the wave frequency $\omega$, random projection dimension $d$, the scaling amplitude initialization, and the size of the calibration set).

---

## 3. Soundness
*Rating*: **Fair**

**Justification**:
While the mathematical framework is elegant, several empirical and methodological limitations affect the soundness of the paper:
1. **No Statistical Analysis**: The absence of confidence intervals or standard deviations over multiple calibration splits is a major concern. With only 16 samples per task, point estimates are highly unstable.
2. **Oracle Assumptions & Head Routing**: The paper does not clarify how the task-specific classification heads are handled during multi-task inference. If head selection is performed using oracle task labels, this must be explicitly stated and justified, as it limits the fully autonomous deployment of the model.
3. **The I.I.D. Violation**: The "wavefunction collapse" step averages coefficients across the batch dimension, meaning that the parameters used to process an individual image depend on the other images in the same batch. This violates the core independent-and-identically-distributed (I.I.D.) assumption of standard machine learning, making the model's behavior batch-dependent and unpredictable in production.
4. **Weak Baseline Comparison**: The Linear Router baseline is not regularized, making its failure on SVHN predictable and easily fixable with standard weight decay or dropout, rather than requiring quantum-inspired projections.

---

## 4. Presentation
*Rating*: **Excellent**

**Justification**:
The paper is exceptionally well-written and structured. The equations (1-9) are mathematically sound, standard, and clearly explained. The transitions from static weight-space compromises to dynamic Hilbert space projections are well-motivated. Crucially, the authors are exceptionally honest and transparent about their method's limitations. They do not hide the "heterogeneity collapse" or the I.I.D. violation; indeed, they dedicate a deep empirical analysis to understanding these phenomena. Both Figure 1 and Figure 2 are clean, highly professional, and communicate their respective findings beautifully.

---

## 5. Significance
*Rating*: **Fair**

**Justification**:
The theoretical significance is notable: the paper introduces a highly original perspective on model merging, showing how wave-inspired, non-monotonic projection functions can act as heavy regularizers to prevent parameter-space collapse. 
However, the practical significance is severely limited in its current form. On a homogeneous test stream (Table 1), there is no practical reason to use a dynamic merger because the task identity is known, meaning we could simply run the task-specific expert to achieve the specialized ceiling ($70.52\%$ vs QWS-Merge's $59.32\%$). On a heterogeneous mixed stream (Table 2), where dynamic routing is actually needed, QWS-Merge collapses and performs significantly worse than existing static merging methods (AdaMerging). Therefore, the method currently lacks immediate practical utility, though it provides a valuable benchmark and foundation for future research.

---

## 6. Originality
*Rating*: **Good**

**Justification**:
The originality of the conceptual framework is high. Porting wavefunction superposition and collapse to parameter space to build dynamic multi-task models is a highly creative approach. However, the actual technical implementation is a moderate/incremental modification of dynamic routing, leveraging layer-wise cosines on a unit sphere. The "quantum" terminology is metaphorical rather than representing a new technical paradigm, but the resulting non-monotonic cosine projection design is highly novel and distinct from standard softmax-based routers.

---

## 7. Overall Recommendation
*Score*: **3: Weak Reject**

**Key Reasons**:
The paper has clear merits, particularly its highly creative formulation, extreme parameter-efficiency, and outstanding scientific transparency in documenting and analyzing its own limitations. However, from an empirical perspective, the weaknesses currently outweigh the merits. The paper suffers from a lack of statistical analysis (no seeds or error bars on a tiny 64-sample calibration set), an unfair comparison against an unregularized global baseline (instead of a regularized, layer-wise baseline), suboptimal expert convergence on SVHN, and a practical performance collapse on mixed-task streams that underperforms simple static methods. 

Addressing these empirical gaps through proper baselines, standard deviations, and ablation studies is essential before the paper can be meaningfully built upon by the community.

---

## 8. Questions and Constructive Feedback for the Authors

1. **Statistical Robustness**: Please run your calibration and evaluation across at least 5 different random seeds with different splits of the 16-sample-per-task calibration set. Report the mean and standard deviation in Tables 1 and 2 to demonstrate that the results are statistically stable.
2. **Layer-wise Linear Router with Regularization**: To isolate the benefit of the wave-like cosine projection, please compare QWS-Merge against:
   - A **Layer-wise Linear Router** (where each layer has its own linear routing layer).
   - A Linear Router regularized with standard **L2 regularization (weight decay)** or **dropout**. 
   - Show whether a simple, regularized Linear Router still collapses on SVHN.
3. **SVHN Expert Convergence**: Why is the specialized expert performance on SVHN so low ($34.50\%$)? A standard CNN or Vision Transformer typically achieves $>90\%$ accuracy on SVHN. Please verify your training hyperparameters and convergence for the SVHN expert, as a poorly trained expert undermines the downstream merging results on this task.
4. **Classification Head Routing**: Please clarify how the multi-task classification heads are routed at inference time, especially in heterogeneous mixed-task batches.
5. **Ablation Studies**: To understand the contribution of each component of QWS-Merge, please provide ablation studies on:
   - The projection dimension $d$.
   - The wave frequency $\omega$ (currently fixed at $\pi$).
   - The initialization values of the scaling amplitudes $R_k^{(l)}$ and phase offsets $\phi_k^{(l)}$.
   - The size of the calibration set (e.g., 4, 8, 16, 32, 64 samples per task).
6. **I.I.D. Violation and Heterogeneity Collapse**: Have you considered using an Exponential Moving Average (EMA) or a rolling queue of sample-level coefficients to bypass the batch-averaging step at inference time? This would resolve the I.I.D. violation and potentially mitigate the "heterogeneity collapse" at larger batch sizes.
