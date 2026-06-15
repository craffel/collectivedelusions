# Impact, Presentation, and Limitations

## 1. Quality of Presentation
- **Writing Style:** The writing is exceptionally clear, precise, and grammatically flawless, maintaining a highly scholarly and sophisticated academic tone.
- **Structure:** The logical flow is seamless, starting from the mathematical motivation, to the closed-form derivations, the physics-inspired torque dynamics, the spatial-temporal coupling, and ending with extensive empirical results and ablations.
- **Mathematical Clarity:** All variables are clearly defined, and equations are beautifully formatted and annotated.
- **Visualization:** The figures are of very high quality, highly professional, and informative, directly supporting the core claims.

## 2. Significance and Potential Impact
- **Paradigm-Shifting Concept:** By rejecting unconstrained flat Euclidean space updates in favor of native curved manifold geodesic flow on the unit hypersphere, this paper introduces a powerful new perspective to Mixture-of-Experts and test-time model ensembling.
- **Practical Utility:** The closed-form Rodrigues-like geodesic updates provide a training-free, extremely fast, and highly stable alternative to existing SOTA biochemical kinetic approaches, making it highly viable for real-world low-latency serving pipelines.

## 3. Major Strengths
1. **Mathematical Rigor:** Flawless derivations of Slerp on $\mathbb{S}^{K-1}$ and proofs of positive orthant persistence.
2. **Scientific Hygiene:** Exemplary baseline auditing (uncovering a prior injection bug in Momentum-Merge) and decomposed jitter analysis (resolving the stability-plasticity trade-off).
3. **Hardware Profiling:** Practical CPU latency and throughput benchmarks.

## 4. Key Areas for Improvement
1. **Scale Gap:** The "real-world" benchmark (shallow 2-layer MLPs on 20newsgroups TF-IDF features) is a poor proxy for the claimed LLM/PEFT/LoRA serving vision. Evaluations on actual pre-trained transformers on multi-task sequential streams are missing.
2. **"Softmax-Free" Overclaim:** Standard UGR relies on a target Softmax to construct target vectors. The proposed fully Softmax-free target (ReLU + $L_1$-norm) is slower and suffers from a performance degradation, meaning the core success is still partially Softmax-dependent.
3. **Layer Mismatch in Spatial-Temporal Coupling:** Initializing the first adapted layer's router of a new query with the final layer's routing state of the previous query ($\mathbf{s}_t^{(L_{\text{frozen}})} = \mathbf{s}_{t-1}^{(L)}$) is structurally unjustified compared to simpler layer-wise coupling ($\mathbf{s}_t^{(l)} = \mathbf{s}_{t-1}^{(l)}$).
4. **Terminological Accuracy:** The word "Unitary" is used to refer to a "unit vector" rather than a unitary operator, which is terminologically non-standard in linear algebra and quantum mechanics.

## Presentation & Impact Conclusion
The presentation is of outstanding quality, and the underlying geometric concept is highly original and significant. If the scale gap is addressed by evaluating on pre-trained transformers with LoRA expert adapters, and the "Softmax-Free" and "Unitary" claims are toned down to be mathematically precise, this paper has the potential to become a seminal work in non-Euclidean representation routing for multi-task serving.
