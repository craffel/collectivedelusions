# Part 4: Experimental Setup and Verification

## Assessment of Experimental Design
The experimental evaluation is highly detailed and structurally sound, testing across multiple axes:
1. **Manifold Alignment**: Orthogonal vs. Overlapping Manifolds.
2. **Stream Characteristics**: Homogeneous (no task shifts) vs. Heterogeneous (frequent and sharp task shifts).
3. **Evaluation Metrics**: Mean joint representation alignment (higher is better) and high-frequency routing jitter (lower is better).

The baselines are comprehensive and cover:
* **Stateless**: SABLE, SPS-ZCA, PAC-ZCA (reproducing state-of-the-art test-time ensembling methods).
* **Stateful Heuristics**: ChemMerge (verifying if a simpler stateful design is sufficient).
* **Baselines with State**: Stateful ERM (verifying the contribution of the PAC-Bayesian bound and KL regularizer).

The results are highly convincing. Under heterogeneous streams, stateless methods have low representation alignment ($76\%$ to $81\%$) due to jitter-induced cascading representation collapse. The heuristic stateful method ChemMerge reduces jitter but fails on heterogeneous tasks, falling to $70.59\%$ accuracy because its static parameters cannot respond quickly to task switches (inertial drag). PAC-Kinetics achieves the best of both worlds: high alignment ($92.90\%$) and extremely low routing jitter ($0.0096$ vs. SABLE's $1.4202$), demonstrating that the Adaptive Online Kinetics successfully resolves the stability-responsiveness trade-off.

## Empirical Weaknesses & Missing Baselines

Despite the impressive empirical results, there are a few areas of potential concern:

* **Narrow Scope of Physical Validation**:
   * The physical validation on real neural networks is restricted to a relatively small multi-layer perceptron (MLP) trained on subsets of MNIST and Fashion-MNIST.
   * While this serves as an excellent proof-of-concept for representation blending, modern parameter-efficient fine-tuning (PEFT) and dynamic ensembling are primarily applied to large-scale autoregressive Transformer models (e.g., LLaMA, Mistral) with Billions of parameters. Demonstrating that PAC-Kinetics successfully prevents cascading representation collapse in a deep Transformer stack on NLP benchmarks (e.g., GLUE, GSM8K, or MMLU) would significantly elevate the paper's impact.

* **Missing Sequence Modeling Baselines**:
   * The authors present Stateful ERM as a baseline, which is a standard feedforward parameterization optimized without the PAC-Bayesian bound. However, they do not compare against standard sequence-modeling architectures such as LSTMs or GRUs.
   * One could argue that an LSTM or GRU could act as a stateful dynamic router by taking the projected coordinates as input and predicting the routing coefficients. While the authors discuss in the Appendix that such models overfit on short sequences and lack stability guarantees, including an explicit empirical baseline comparison against a GRU-based router would make the empirical validation much stronger.

* **Lack of GPU Multi-Batching Serving Benchmarks**:
   * The paper emphasizes "stable test-time model serving," which is a production systems concern. In real-world model serving frameworks (e.g., vLLM, S-LoRA, Punica), incoming queries are batched concurrently to maximize GPU utilization.
   * In a batched environment, the stateful routing state $s_t$ must be maintained independently per active sequence in memory. This could introduce memory bandwidth bottlenecks or kernel launch overheads. The paper provides CPU latency measurements in the Appendix but does not present benchmark results for concurrent GPU serving. Providing GPU wall-clock latencies under batch sizes $B \in \{16, 64, 128\}$ would confirm that tracking the stateful memory does not introduce memory bandwidth or CUDA kernel launch bottlenecks in production serving.
