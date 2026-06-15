# Peer Review of Conference Submission: Unitary Geodesic Routing (UGR)

## 1. Summary of the Paper
This paper introduces **Unitary Geodesic Routing (UGR)**, a non-Euclidean geometric framework for dynamic test-time model ensembling of specialized parameter-efficient fine-tuning (PEFT) expert adapters. In non-stationary, sequential serving streams, a stateful router is required to smooth out representation noise while remaining responsive to task transitions. Prior stateful ensembling routers (e.g., *Momentum-Merge*, *ChemMerge*) perform routing weight updates in unconstrained flat Euclidean spaces and project onto the probability simplex post-hoc via Softmax normalization. The authors argue that this Euclidean-to-simplex mismatch introduces representational lag (hysteresis), geometric scale distortion of hidden activations, and routing jitter.

To resolve these issues, UGR models the ensembling state directly on the curved unit hypersphere $\mathbb{S}^{K-1}$. It maps coordinates to the probability simplex via the square-root homeomorphism (Born's rule): $\alpha_{k,t} = (s_{k,t})^2$, representing closed-form Fisher-Rao geodesic flows on the simplex with zero geometric distortion or scale-mismatch artifacts. UGR derives an analytical, closed-form Rodrigues-like geodesic rotation operator (Slerp) along the shortest great-circle path, bypassing costly matrix exponentials or numerical ordinary differential equation (ODE) solvers. It introduces **Torque-Driven Adaptive Agility** (a first-order self-regulating control loop where angular torque scales the rotational step-size to accelerate on task boundaries and stabilize on stable streams) and **Spatial-Temporal Geodesic Coupling** (propagating converged states across sequential query boundaries).

UGR is evaluated on a 14-layer synthetic sandbox (ICS) across 10 random seeds (achieving 75.08% accuracy and 2.10$\times$ lower routing jitter than ChemMerge) and on a 20newsgroups text classification stream across 5 seeds (achieving 92.25% accuracy and 1.63$\times$ lower jitter than Coupled Momentum-Merge).

---

## 2. Main Strengths

* **Elegant and Principled Geometric Formulation:** Rejecting unconstrained flat updates in favor of modeling the stateful ensembling weights on the curved unit hypersphere $\mathbb{S}^{K-1}$ is conceptually elegant and mathematically rigorous. Mapping the spherical state to the probability simplex via Born's rule provides a clean, Softmax-free projection that natively respects probability constraints and preserves activation norms.
* **Closed-Form Rodrigues-like Spherical Rotations:** Deriving an analytical, closed-form spherical interpolation operator (Slerp) completely bypasses expensive matrix exponentials or virtual-time numerical ODE solvers. This keeps the algorithm extremely computationally efficient ($\mathcal{O}(KD)$ complexity per layer), adding $<0.07$ ms of latency per query in single-threaded CPU wall-clock benchmarks.
* **High Scientific Hygiene and Rigorous Baseline Auditing:** The authors demonstrate commendable scientific integrity. They audited previous reports of the Momentum-Merge baseline, identified a crucial initialization bias (target injection), corrected it, and enforced strict uniform initialization across all uncoupled models. Furthermore, they evaluated all stateful baselines under both "Reset" and "Coupled" configurations to isolate the gains of the curved geodesic flow itself from the cross-query memory propagation mechanism, ensuring complete experimental symmetry.
* **Robust Statistical Transparency and Detailed Jitter Analysis:** The evaluation is statistically sound, reporting averages and standard deviations across 10 independent seeds (synthetic) and 5 seeds (real-world NLP). The authors provide a brilliant decomposed jitter analysis that separates *intra-task stability* (where UGR is over 5.5$\times$ more stable than Momentum-Merge Advanced) from *inter-task agility* (where UGR rotates rapidly at task boundaries). This proves that the model's high overall jitter in randomized streams is due to purposeful, correct transitions rather than random noise.

---

## 3. Main Weaknesses

* **Severe Empirical Scale Gap / Outdated Benchmarks:** The most significant weakness is the massive disconnect between the visionary, modern claims of "PEFT expert adapters serving" and the highly outdated, toy experimental setup used for evaluation. 
  * The "real-world" benchmark is restricted to the classic **20newsgroups** dataset using **TF-IDF features** ($D=1024$) and a **2-layer MLP** (128 hidden units) with only $K=4$ experts.
  * In contemporary deep learning serving, Mixture-of-Experts routing and test-time ensembling are applied to Large Language Models (LLMs) and Vision Transformers (ViTs) on complex token-level or image-level sequences. A toy MLP on TF-IDF features fails to demonstrate UGR's viability on modern deep learning representation manifolds, where task shifts can happen mid-generation or during continuous multi-turn dialogue.
* **Lack of Expert Scaling-up Ablations ($K > 4$):** The evaluation is restricted to a very small expert pool of $K=4$. In practical MoE serving, the number of experts can be much larger ($K \ge 16$). The paper conceptually describes "local geodesic routing" in the Appendix to handle "measure concentration" and "centroid crowding" under larger pools, but provides **zero empirical results** demonstrating UGR's performance under larger expert pools. Proof of scalability is critical for an empirical serving framework.
* **Suboptimal Evaluation of the SOTA Baseline (ChemMerge):** ChemMerge is evaluated under a relatively large step size of $dt=1.5$, causing its virtual-time Euler-integration to become numerically unstable and oscillate violently. This suboptimal tuning likely explains why ChemMerge achieves a poor joint classification accuracy of only 70.67% on text classification (barely beating static uniform at 64.95%). The authors argue that smaller step sizes introduce "unacceptable latency," but they provide no empirical wall-clock latency curves to substantiate this. This raises questions about whether the baseline comparisons were completely fair and well-tuned.
* **Theoretical vs. Empirical Divergence in Target Mapping:** The flashy theoretical connection to the "exact, closed-form Fisher-Rao geodesic flow" is compromised in practice to achieve the best accuracy. Under the exact Born Target mapping, classification accuracy drops from 75.08% to 74.47% (synthetic) and 92.25% to 90.67% (NLP). Standard UGR relies on a heuristic quadratic projection distortion ($L_2$-normalization) to sharpen ensembling boundaries and maximize accuracy.

---

## 4. Evaluation of Dimensions

### Soundness: Good
The mathematical derivations are exceptionally rigorous, and the sign invariance and Positive Orthant Persistence are soundly proven. The authors audited and corrected previous baseline cheating (the Momentum-Merge initialization discrepancy) and isolated confounding factors through symmetric Reset/Coupled configurations. However, the soundness of the empirical claims is limited by the toy nature of the datasets (20newsgroups with TF-IDF) and the small expert pool size ($K=4$).

### Presentation: Excellent
The paper is beautifully written, mathematically detailed, and exceptionally clear. The pseudo-code in Algorithm 1 is highly structured, and the figures (Figures 1, 3, and 4) are extremely informative and high-signal, particularly the Accuracy-Stability Pareto frontier in Figure 4.

### Significance: Fair
The conceptual and mathematical contributions are highly significant and could establish a new geometric paradigm for test-time adaptive serving. However, because the evaluation is limited to a synthetic sandbox and a toy MLP, the immediate significance of the paper is limited. The paper feels like a beautiful mathematical exercise that requires scaling validation on real-world transformers to be fully accepted by the modern machine learning community.

### Originality: Good
Operating on the curved unit hypersphere $\mathbb{S}^{K-1}$ with Born's rule simplex projection, deriving a closed-form Rodrigues-like Slerp update, and designing a training-free torque-driven control loop represent highly creative and original combinations of information-geometric and physical concepts.

---

## 5. Overall Recommendation

**Rating: 4: Weak Accept**

**Justification:** UGR is a mathematically elegant and highly original framework that achieves state-of-the-art accuracy and stability in its evaluated settings. The paper demonstrates outstanding scientific hygiene (auditing and fixing baseline discrepancies, evaluating symmetric uncoupled/coupled controls, providing decomposed stability-plasticity jitter analyses). However, the severe empirical scale gap (evaluating on a 2-layer MLP on TF-IDF 20newsgroups) and the lack of empirical results for larger expert pools ($K \ge 16$) limit its immediate significance. The paper is technically very solid, and its high-quality presentation and rigorous control experiments make it worthy of acceptance, but the authors are strongly encouraged to address the empirical limitations in their discussion and future work.

---

## 6. Actionable Questions and Feedback for the Authors

1. **Expert Pool Scaling Sweep:** Can you provide empirical classification accuracy and routing jitter results for larger expert pools (e.g., $K=8, 16, 32$) on the text classification task? Demonstrating that the "local geodesic routing" strategy proposed in the Appendix prevents performance degradation due to measure concentration is crucial.
2. **ChemMerge Tuning and Latency Analysis:** Can you provide empirical wall-clock latency curves for ChemMerge under varying ODE integration step sizes (e.g., $dt = 0.1, 0.5, 1.0, 1.5$)? This would empirically support the claim that a stable ODE integration step size introduces an unacceptable latency bottleneck, justifying the necessity of evaluating ChemMerge under an unstable $dt=1.5$.
3. **Scaling to Autoregressive LLM Serving:** Autoregressive decoding in modern Transformers operates on sequential, token-level representations where task shifts can occur mid-generation. Can you discuss or provide a conceptual blueprint of how UGR's spatial-temporal coupling and torque-driven step-size would handle token-by-token generation, given that token-level hidden states are highly noisy and shift at high frequencies?
4. **Theoretical Gap:** Under the exact Born Target mapping (`UGR (Born Target)`), classification accuracy drops across both benchmarks, meaning standard UGR relies on a heuristic quadratic projection distortion ($L_2$-normalization) to maximize accuracy. Can you elaborate on why the exact, mathematically pure Fisher-Rao geodesic flow performs worse in practice than the heuristic sharpening distortion?
