# Novelty and Literature Positioning Check

## Key Novel Aspects
The paper introduces two major contributions to resolve key failure modes of dynamic model merging:
1. **Parameter-Free Subspace Routing (PFSR):** The first zero-shot, completely non-parametric dynamic routing framework for model merging. Rather than training a parametric routing network on a calibration split (which is prone to overfitting and OOD collapse), PFSR projects penultimate layer representations onto a task coordinate subspace using the cosine similarity against the pre-trained expert classification weights themselves. This eliminates all trainable parameters and calibration data requirements.
2. **Micro-Batch Homogenization (MBH):** A novel data-stream orchestration layer that solves "heterogeneity collapse" at the batch level rather than the weight-parameter level. By dynamically partitioning heterogeneous mixed-task streams into homogeneous micro-batches on the fly, performing specialized inference, and re-assembling the outputs, it completely bypasses the batch-averaging degradation that plagues standard dynamic routers.

## The 'Delta' from Prior Work
The proposed framework is situated within several distinct lines of research:

### 1. Dynamic Routing in Model Merging
* **Prior Work:** Existing dynamic model merging frameworks (e.g., `PredecessorT4S6`, `PredecessorT4S10` [QWS-Merge], and `PredecessorT5S5` [L3-Router]) train lightweight parametric routing layers (e.g., linear layers, multi-layer networks, or wave wavefunction routers) on a small few-shot calibration split (e.g., $N_c = 64$ samples) using optimization algorithms like AdamW.
* **The Delta:** PFSR removes the training phase entirely. Instead of learning a routing matrix via gradient descent, it repurposes the pre-trained expert classification heads ($W_k$) as class-prototype landmarks. By using the maximum cosine similarity to define coordinates, the routing decision is computed directly on the fly. This avoids transductive overfitting and guarantees perfect robustness to OOD tasks (e.g., SVHN).

### 2. Prototypical Networks & Zero-Shot Classification
* **Prior Work:** Prototypical Networks (`Snell2017`) and zero-shot metric learning leverage distance or similarity metrics against semantic class prototypes to classify samples without explicit retraining.
* **The Delta:** While Snell et al. use prototypes in activation space for few-shot classification, PFSR applies this concept directly to the parameter space for *weight-space dynamic model merging*. It bridges metric learning with weight interpolation by using the class-prototype alignments as weight-blending coefficients on-the-fly.

### 3. Systems-ML Request Batching
* **Prior Work:** Advanced serving systems like Orca (`Yu2022`) and vLLM (`Kwon2023`) use dynamic request partitioning and continuous batching to optimize hardware throughput for multiple standalone LLM queries.
* **The Delta:** MBH adapts these scheduling concepts to the *weight-space model merging paradigm*. Rather than managing memory/attention allocations for separate standalone models, MBH groups heterogeneous inputs to eliminate weight-averaging collapse and representation interference within a single dynamically-merged backbone.

### 4. Over-parameterization Deconstructions
* **Prior Work:** Predecessor audits like `PredecessorT5S4` and `PredecessorT5S5` deconstructed wave wavefunction metaphors and unmasked the "Robustness-Accuracy Illusion" of simplex-constrained routers.
* **The Delta:** This paper builds on these deconstructions but takes a radical, minimalist step forward. Rather than trying to optimize or regularize a classical parametric router, it strips routing parameters completely, resolving the underlying stream heterogeneity at the data level.

### 5. Asymmetrical Expert & High-Vocabulary Scaling
* **Prior Work:** Scaling cosine similarity classifiers to high-vocabulary regimes (like LLMs with $C \ge 32,000$) usually incurs severe statistical biases and latency bottlenecks.
* **The Delta:** The paper introduces:
  * *Class-Size Scaling Calibration ($O(\sqrt{\log C_k / d})$):* Corrects the statistical bias over-routing to high-vocabulary experts by normalizing by the expected maximum of random Gaussian similarities.
  * *Sub-Vocabulary Prototype Selection:* A parameter-centric pruning heuristic based on classification weight variance across experts. This achieves a $132.2\times$ speedup and avoids private text data dependency.

## Characterization of Novelty
The novelty of this work is **significant and highly refreshing**. 
Instead of introducing increasingly complex, hyper-parameterized, or metaphor-driven architectures (e.g., "quantum wavefunction" routers) to solve test-time adaptation, the authors apply **Occam's razor** to simplify the model side while shifting the complexity of stream handling to the **systems/serving layer**. 

This co-design of systems-level data scheduling (MBH) and parameter-efficient model merging (PFSR + LoRA) represents a valuable paradigm shift:
* It unmasks the futility of over-engineering layer-wise routers (supported by a rigorous mathematical proof of Layer-Averaging Collapse).
* It offers a highly practical, reproducible, and instantaneous scaling capability (dynamic task addition/deletion without joint calibration) for real-world production model registries.
* It respects physical hardware boundaries (VRAM vs. FLOPs co-design under PEFT/LoRA).

By grounding the solution in both statistical analysis and systems serving, the paper provides a complete and elegant blueprint that successfully navigates the trade-offs of modern deep learning deployments.
