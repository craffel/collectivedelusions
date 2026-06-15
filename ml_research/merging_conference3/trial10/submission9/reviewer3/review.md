# Peer Review

## Strengths and Weaknesses

### Strengths
- **Mathematical Elegance and Simplicity of the Solver:** The paper successfully reframes sequential expert ensembling as a stateful perceptual process. Crucially, the authors reduce what could be a computationally intensive, iterative active inference optimization into a simple, strictly convex quadratic objective. Solving this exactly using a precomputed Cholesky factorization of the Hessian is exceptionally elegant, numerically stable, and computationally instantaneous ($\mathcal{O}(K^2)$ substitution at test-time). It is refreshing to see a method that achieves high adaptability through clean, closed-form mathematical reduction rather than relying on unrolled gradient approximations or complex neural networks.
- **Dynamic Resolution of the Jitter-Lag Trade-Off:** The proposed framework beautifully addresses a major practical bottleneck. It achieves the stability of stateful methods under noise (reducing routing jitter by up to **2.49$\times$**) while preserving the near-instantaneous tracking speed and accuracy of stateless models during abrupt context switches (adapting in 1--2 steps).
- **The Elegant, Highly Parameter-Efficient Diagonal Variant:** The introduction of **AIR (Diagonal)** is a brilliant demonstration of elegant and lightweight design. By restricting the generative coordinate mapping to be diagonal, the authors create a model with linear $\mathcal{O}(K)$ parameter complexity ($5K$ parameters) that can be calibrated on a tiny stream of 32 samples. It achieves outstanding stability and accuracy, completely bypassing the overfitting and sequence-slicing risks of the dense model while maintaining maximum simplicity.
- **Rigorous Validation and Reproducibility:** The empirical evaluation on the Analytical Coordinate Sandbox (ACS) is robust, testing different noise profiles, manifold geometries, and adversarial non-linear manifold warpings. The complete disclosure of hyperparameters, detailed pseudocode, and public open-source code ensure absolute reproducibility.

### Weaknesses
- **Conceptual Over-Engineering and Jargon-Heavy Presentation:** The paper employs extensive neuroscientific and cognitive science terminology ("Active Inference", "Variational Free Energy", "self-organizing cognitive agent", "perceptual action", "inhibitory pathways", etc.). While this first-principles framing is interesting, once the variational covariance is assumed static and non-dependent terms are discarded, the model collapses *exactly* to a classic **linear state observer (Kalman filter)**. Couching a simple, elegant classical method in heavy, jargon-filled terminology adds substantial cognitive load and feels like conceptual over-marketing. The paper would be far stronger, more transparent, and more accessible if it focused on the simplicity of the resulting linear state observer from the outset, rather than cloaking it in extensive neuroscience terminology.
- **Speculative Over-Engineering in the Appendix:** The appendix is extremely large (20 sections) and details numerous highly complex, speculative mathematical extensions (e.g., non-static covariance models, non-negative Truncated Gaussian likelihoods, quadratic Laplace approximations, Contractive Autoencoders for non-linear projections). While mathematically sophisticated, these extensions are **completely un-evaluated** in the main paper. Introducing these untested, highly engineered constructs contradicts the elegant, closed-form simplicity of the core AIR model. The authors should simplify, prune these speculative sections, and focus on the elegant linear-Gaussian model.
- **Under-Emphasizing the Minimalist Diagonal Variant:** Because the **AIR (Diagonal)** model is incredibly simple, highly parameter-efficient, and extremely resistant to overfitting, it represents the ideal engineering solution. However, this elegant model is buried in the appendix under scaling and calibration studies. The authors should have featured this simple, elegant variant as a core contribution in the main text.

---

## Soundness
**Rating: Excellent**

**Justification:**
The mathematical derivation is highly rigorous, correct, and internally consistent. Under static variational covariance assumptions, the simplification of the free energy objective to a quadratic convex form is mathematically sound. The exact closed-form linear system solve ($\mathbf{H}\mathbf{\mu}_t^* = \mathbf{b}_t$) via precomputed Cholesky factorization is mathematically correct and highly efficient. The empirical evaluation is comprehensive, covering two distinct stream configurations (homogeneous and heterogeneous), multiple noise profiles, and a non-linear manifold stress test. The ablation study confirming the mechanistic role of active inhibition is highly convincing and methodologically sound.

---

## Presentation
**Rating: Good**

**Justification:**
The paper is well-structured, clearly written, and the narrative is easy to follow. The systems execution flowchart (Figure 1) is exceptionally helpful and beautifully illustrates the test-time serving loop. However, the rating is limited to "Good" due to the heavy use of theoretical neuroscience jargon. Framing a simple and elegant linear state observer (Kalman filter) through the complex lens of active inference adds unnecessary conceptual complexity and cognitive overhead for the reader. The presentation would be vastly improved by adopting a more direct, simpler narrative that celebrates the elegant control-theoretic simplicity of the core method.

---

## Significance
**Rating: Excellent**

**Justification:**
The paper addresses an important, highly relevant problem in dynamic model serving and Mixture-of-Experts architectures. The Jitter-Lag Trade-Off is a well-known systems-level bottleneck that causes hardware cache thrashing and representational instability. Demonstrating that an adaptive linear state observer can solve this severe bottleneck with microsecond-level serving overhead is highly significant and practically valuable. The outstanding performance and simplicity of the **AIR (Diagonal)** variant further enhance the practical utility and potential impact of this work for real-world deployments.

---

## Originality
**Rating: Good**

**Justification:**
The paper provides a novel combination of theoretical neuroscience and machine learning systems serving. While the underlying mechanism simplifies to a well-known linear filter, deriving dynamic ensembling from first-principles Variational Free Energy minimization is highly original and provides a robust, theoretically grounded basis for temporal gating. The verification of the necessity of active inhibition (allowing negative weights in the generative coordinate mapping to form negative feedback loops) is also a novel and insightful contribution to the understanding of dynamic ensembling.

---

## Overall Recommendation
**Rating: 5: Accept**

**Justification:**
This is a technically solid, highly thorough paper that addresses an important and practical problem in dynamic model serving. The core proposed method is exceptionally elegant, simple, and effective. By simplifying a complex active inference formulation into a quadratic convex objective, the authors achieve a mathematically exact, closed-form linear solver that runs in near-zero latency at test-time via Cholesky precomputation. The empirical performance is outstanding, successfully resolving the Jitter-Lag Trade-Off. The highly parameter-efficient **AIR (Diagonal)** variant further demonstrates the strength and beauty of simple design.

While the paper suffers from conceptual over-engineering, heavy cognitive science jargon, and speculative theoretical additions in the appendix, these are primarily presentation and scoping issues that do not detract from the technical soundness and practical elegance of the core method. The authors are strongly encouraged to simplify their narrative, tone down the neuroscience jargon, prune the speculative appendix sections, and highlight the elegant **AIR (Diagonal)** model as a central contribution. This will make their valuable work far more accessible and impactful for the broader machine learning community.
