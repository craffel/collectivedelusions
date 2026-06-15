# 2. Novelty Check

## Key Novel Aspects
The paper represents a **methodological audit and systematic deconstruction** rather than a proposal for a highly complex new architecture. Its primary novelty lies in exposing and explaining the flaws of existing state-of-the-art (SOTA) works in dynamic model merging (SABLE, ChemMerge) using Occam's razor. Specific novel elements include:

1. **Isolation of Confounding Variables:** Systematically isolating that the widely reported failure and collapse of classical parametric routers in prior work is not due to representational limitations, but is an artifact of weak baseline setups (random initialization and lack of regularization under data scarcity).
2. **Information-Theoretic Framing of Zero-Initialization:** Proposing **Maximum-Entropy Zero-Initialization** as a critical baseline design pattern. While zero-initialization is standard in deep learning, framing and evaluating it as an unbiased maximum-entropy prior that provides a guaranteed safety fallback (gracefully degrading to static Uniform Merging) is highly elegant and novel in this context.
3. **Control-Theoretic Demystification of Physical Metaphors:** Exposing that the continuous-time chemical kinetics of ChemMerge are functionally a closed-loop temporal low-pass filter (closed-loop stateful inertia) rather than a mystical physical representational panacea.
4. **The Jitter Myth Debunked:** Implementing a layer-wise classical routing baseline to empirically prove that classical parametric routers do not inherently suffer from severe layer-wise weight oscillations, contradicting prior claims.
5. **EMA-SABLE Baseline:** Introducing Exponential Moving Average SABLE (EMA-SABLE) to successfully isolate and compare the accuracy gains of open-loop trajectory smoothing versus ChemMerge's closed-loop feedback correction.

## 'Delta' from Prior Work
The "delta" of this paper is meta-scientific and highly practical:
* **SABLE & ChemMerge:** These prior SOTA methods claimed that classical parametric linear routers catastrophically fail in low-data regimes, requiring complex nearest-centroid projections or ODE solvers. The delta in this paper is proving that this failure is easily resolved with proper L2 regularization and zero-initialization.
* **Sample Complexity Crossover:** Prior work evaluated models on isolated, extremely small calibration datasets ($N \le 128$). This paper sweeps sample complexity up to $4,000$ samples, establishing the exact data crossover boundaries where learning-based routing becomes superior to training-free geometric priors.
* **Hyperparameter Sensitivity:** Prior work marketed training-free methods as plug-and-play. This paper sweeps the temperature parameter $\tau$, proving that SABLE is highly sensitive to manual tuning and degrades to Uniform Merging under sub-optimal parameters, whereas ChemMerge provides robust hyperparameter buffering.

## Characterization of Novelty
The novelty is **significant and highly valuable, particularly for practitioners**. 
In machine learning, there is a recurring tendency to introduce overly complex, metaphorical architectures (such as continuous-time ODEs mimicking chemical reaction kinetics) that introduce severe serving-time overhead. This paper acts as a critical corrective. By proving that a simple classical linear router—with proper L2 regularization and standard zero-initialization—can match or outperform these complex metaphorical models when a modest calibration budget is available, the paper simplifies deployment pipelines. This is a crucial practical contribution, shifting the dynamic model merging paradigm back toward robust, lightweight, and well-understood linear layers.
