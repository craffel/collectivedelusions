# 4. Experiment Check

## Critical Evaluation of the Experimental Setup

### 1. The Synthetic Simulation Sandbox (ICS)
The authors evaluate their model primarily in a **14-layer Analytical Coordinate Sandbox (ICS)**.
- **Strength:** The ICS allows for a highly controlled evaluation of representation dynamics across three manifold configurations: Orthogonal, Overlapping, and Composite. This allows the authors to study clean, mathematical properties of routing (e.g., non-monotonic task composition where the target task shifts at Layer 9).
- **Limitation:** The ICS is a synthetic simulation environment, not a real neural network. While it models representation drift and noise, its generalizability to actual deep learning workloads remains a question.

### 2. Physical Validation on ResNet-18
To bridge the "reality gap," the authors physically validate their model on a pre-trained ResNet-18 model using natural ImageNet-1K streams.
- **Strength:** This is a valuable addition that shows the method is capable of operating on high-dimensional natural manifolds with real parameter spaces. The expansion to 40 distinct ImageNet-1K classes (10 classes per task) over 200 query samples with test-time data augmentations provides a solid empirical foundation.
- **Limitation:** ResNet-18 is an 8-block convolutional neural network, which is a lightweight proxy for modern, massive autoregressive Large Language Models (LLMs) or dense Mixture-of-Experts (MoE) architectures where routing jitter is a critical bottleneck.
- **Surrogate Modulation:** Instead of fine-tuned PEFT adapters (such as task-specific LoRA weights), the authors use a training-free channel-wise modulation surrogate based on un-optimized activation signatures. While this surrogate is isomorphic to adapter ensembling, it introduces noise into the representation space.

### 3. The Oracle Sub-optimality and Signature Perturbation Effect
An intriguing result in Table 4 is that the `Oracle` baseline (which forces the ensembling weight to 1.0 for the true task) underperforms dynamic routers (including QPathMerge and ChemMerge) on homogeneous streams, and yields noisy performance on heterogeneous streams.
- The authors explain this as the **Signature Perturbation Effect**: since the signatures are extracted via few-shot calibration rather than joint end-to-end training, forcing 100% of a single signature acts as a localized destructive perturbation.
- **Critique:** This explanation exposes a weakness in the physical validation setup. In a standard production MoE or LoRA serving system, fine-tuned adapters naturally define the performance upper bound (and thus the Oracle would outperform any smoothed or uniform blend). The fact that the Oracle underperforms here indicates that the physical validation uses a somewhat noisy surrogate that may not fully reflect the behavior of true, mathematically optimized fine-tuned adapters.

### 4. Workload Bias in Baseline Comparison
The authors demonstrate a dramatic performance collapse of stateful models (ChemMerge, Momentum-Merge) under heterogeneous streams, where task switches occur frequently and abruptly.
- **Critique:** While this collapse is mathematically consistent and supports the claim of zero temporal lag, the evaluation is heavily biased toward stateless models. In a real-world serving setting, user queries are rarely completely independent and rapidly switching on a sample-by-sample basis. Instead, they exhibit high temporal correlation (e.g., a multi-turn conversation on coding or translation). 
- Under a temporally correlated stream, stateful models would benefit from historical temporal context to smooth out local activation noise, whereas a completely stateless model like QPathMerge would be blind to this history. The authors should discuss this trade-off and acknowledge that their evaluation focuses on the worst-case scenario for stateful models.

## Support for Claims
Despite these limitations, the experimental results strongly support the paper's core claims:
- **Jitter Reduction:** QPathMerge consistently achieves a $3\times$ to $5\times$ reduction in spatial layer-wise jitter compared to stateless SABLE-Dynamic across both synthetic and physical environments.
- **Zero Temporal Lag:** QPathMerge matches or exceeds SABLE's stateless agility under rapid task switches, whereas stateful models suffer from severe temporal hysteresis.
- **Pareto Control:** Sweeping the transition leakage $M$ maps out a clear accuracy-jitter Pareto frontier, proving that $M$ serves as a robust control parameter for system administrators.
