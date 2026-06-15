# Peer Review

## 1. Summary of the Paper
The paper addresses the challenge of post-hoc model merging of specialized neural networks into a single multi-task model. It identifies a hidden vulnerability in existing dynamic routing networks called **heterogeneity collapse**, where hardware constraints average ensembling coefficients across a batch, flattening expert specialization under mixed-task deployment. 

To solve this, the authors propose **Endosymbiotic Holographic Parameter Binding (EHPB)**, a model-merging paradigm based on hyperdimensional computing (HDC). EHPB modulates task-specific parameter updates (task vectors) with random pseudo-orthogonal bipolar carrier keys using the element-wise Hadamard product, superimposing them into a single weight matrix. At test-time, an input-dependent dynamic unbinding operator is used to demodulate and transcribe active expert weights sample-by-sample, bypassing batch-level ensembling constraints. 

While EHPB successfully neutralizes heterogeneity collapse, it suffers from severe weight reconstruction noise (~170% relative error) due to "coordinate isolation" under Hadamard binding. This noise collapses the model's Joint Mean accuracy in a synthetic sandbox from an expert ceiling of 74.6% to **25.4%**—which is **26.9% lower** than simple static **Uniform Merging (52.3%)**. To mitigate this, the authors propose several complex extensions: **Residual-EHPB** (bypassing 5% of critical coordinates uncompressed, achieving 33.7%), **Continuous Cleanup Networks (CCN)** (training 13 separate MLP models at intermediate layers to denoise activations), and **ReLU Post-Hoc Bias Correction** (training extra scale/shift parameters per layer).

---

## 2. Strengths and Weaknesses

### Strengths
- **Exemplary Intellectual Honesty:** The authors deserve substantial credit for their exceptional transparency and scientific honesty. Instead of trying to obfuscate the poor performance of their proposed method, they clearly define and analyze the "Hadamard Dominance Paradox" and the "SVHN Floor Effect Confounder."
- **Rigorous Mathematical Deconstruction:** The paper provides brilliant mathematical insights into why the method fails. The authors successfully prove why Hadamard-based weight reconstruction noise remains scale-invariant (the "Coordinate Isolation Confounder"), how zero-mean noise is systematically rectified into positive bias vectors by ReLU activations, and how LayerNorm exponentially attenuates the semantic signal.
- **Exposure of Heterogeneity Collapse:** Exposing how batch-level hardware ensembling averages out dynamic routing coefficients under streaming mixed-task workloads is a highly practical and valuable observation for the deep learning community.
- **Proactive Exploration of Mitigations:** The paper goes to great lengths to systematically design and evaluate candidate workarounds to handle the noise, such as Residual-EHPB, Continuous Cleanup Networks (CCN), and ReLU bias corrections.

### Weaknesses
- **Extreme, Unjustified Complexity:** The core philosophy of effective machine learning design is that **complexity must only be introduced when justified by massive gains**. EHPB introduces hyperdimensional holographic binding, sample-wise unbinding operators, custom Triton register-level kernels, and a cascade of highly engineered "fixes" (layer-wise Continuous Cleanup Networks, sparse coordinate masks, ReLU bias correction parameters). Despite this overwhelming complexity, the method suffers from a catastrophic performance collapse.
- **Heavily Dominated by Simple, Elegant Baselines:** The proposed framework is completely impractical. Simple, static **Uniform Merging** (simply averaging the specialized models) requires **zero training, zero dynamic memory management, zero extra parameters, and zero inference latency overhead**, yet it achieves **52.3% Joint Mean accuracy**—massively dominating EHPB (25.4%) and Residual-EHPB (33.7%). A highly complex, noisy, and computationally expensive method is fundamentally non-viable if it cannot beat an incredibly simple and elegant baseline.
- **Convoluted and Contradictory "Band-Aid" Pipeline:** Instead of recognizing that the core holographic binding approach is inappropriate for sensitive neural network coordinates, the authors attempt to rescue it by piling on further layers of complexity. For example, training and running **13 extra MLP models** (Continuous Cleanup Networks) at intermediate layers to denoise activations completely violates the premise of "resource-efficient deployment."
- **The Active Memory Paradox:** Under eager-mode deep learning frameworks (like PyTorch), evaluating sample-wise weights dynamically via vectorized mapping (`vmap`) materializes a batch of weight matrices, scaling intermediate memory to $O(B \times P)$—which is far worse than the $O(K \times P)$ RAM footprint of keeping all experts in memory. Relying on custom hardware-level register Triton/CUDA kernels to bypass this represents an enormous, impractical engineering burden that severely limits the method's deployability.
- **Toy Sandbox Restrictions:** The entire empirical evaluation is restricted to a synthetic "Controlled Representation Sandbox" utilizing independent Gaussian-simulated expert weights and prototype evaluations. There is no validation on real-world fine-tuned models on standard model-merging benchmarks (such as GLUE or VTAB), leaving the practical utility of the framework unverified.

---

## 3. Detailed Assessment of Criteria

### Soundness: Poor
The proposed method is fundamentally mathematically flawed for deep neural networks. Neural network parameters are highly sensitive, non-linear, and coordinate-coupled. Modulating task vectors with random sign matrices introduces a destructive cross-talk noise term (approx. 170% relative reconstruction error) that collapses the model's representations. The non-linearities in deep architectures (ReLU and LayerNorm) exacerbate this noise, causing systematic positive bias rectification and exponential signal attenuation. The proposed workarounds (Continuous Cleanup Networks, Residual-EHPB) are convoluted post-hoc patches that fail to overcome these fundamental constraints, and the downstream classification accuracy remains far below a simple static average.

### Presentation: Fair
The writing is highly academic, thorough, and dense. However, the overall narrative is heavily cluttered with unnecessarily complex, flowery, and exotic terminology ("endosymbiotic," "optical holography," "Post-Hoc Model Ensembling Trilemma"). These elaborate physical and biological metaphors act as mathematical obfuscation, dressing up what is actually a simple set of element-wise sign matrix multiplications. The paper would be significantly improved by stripping away this jargon and presenting the core mechanisms directly, clearly, and elegantly.

### Significance: Poor
The practical significance of this work is extremely low. Because the proposed EHPB method introduces a massive performance penalty and requires highly complex execution pipelines (including custom CUDA/Triton kernels to avoid memory bottlenecks and extra MLP networks to denoise representations), it is highly unlikely to be adopted by machine learning practitioners or edge-deployment engineers. Simple, robust, and elegant alternatives (such as static Uniform Merging, TIES-merging, or straightforward parameter-efficient adapter routing) are far superior in practice.

### Originality: Fair
The paper does introduce an original crossover of Vector Symbolic Architecture to deep neural network weights and exposes "heterogeneity collapse" in dynamic routing. However, this originality is over-engineered and fails to result in a functional or practical solution, as it is heavily dominated by extremely simple classical baselines.

---

## 4. Overall Recommendation
**Rating: 2 (Reject)**

### Justification:
The paper proposes an excessively complex, over-engineered model-merging paradigm that is fundamentally flawed and performs poorly. The introduction of hyperdimensional holographic binding destroys the coordinate-wise integrity of specialized weights, resulting in a massive performance collapse. Rather than seeking a simple and elegant solution, the authors attempt to patch this collapse by adding multiple layers of extreme complexity (e.g., training 13 additional MLP cleanup networks, managing sparse coordinate masks, and designing custom Triton kernels). Even with these convoluted workarounds, the method remains heavily dominated by a simple, training-free static average. In machine learning, we must champion elegant, simple, and effective methods, and penalize unnecessary complexity and mathematical obfuscation. Therefore, this paper is recommended for rejection.

---

## 5. Constructive Questions / Suggestions for Authors
1. **Simplify the Framework and Terminology:** Strip away the unnecessary and distracting metaphors of "cellular endosymbiosis" and "optical holography." Describe the method directly as element-wise random sign modulation and superposition. This will make the paper much more readable and honest.
2. **Re-evaluate the Premise of Holographic Superposition:** Since element-wise Hadamard binding destroys weight coordinates and is heavily dominated by simple Uniform Merging, is it worth pursuing this paradigm? Instead of superimposing weights, why not focus on solving the "heterogeneity collapse" issue directly using a simpler, more elegant method (such as vectorized, sample-specific dynamic scaling of lightweight LoRA adapters)?
3. **Validate on Real-World Benchmarks:** Transition away from the synthetic, toy-scale sandbox. Evaluate your proposed methods on actual, standard model-merging benchmarks (such as merging real fine-tuned LLMs on GLUE or real Vision Transformers on VTAB) to prove if your findings generalize to real neural network weight manifolds.
4. **Quantify the Resource Overhead of Cleanup Networks:** Provide a clear, honest comparison of the parameter and compute overhead of training and running the 13 Continuous Cleanup MLP Networks. Does the storage of these extra MLPs exceed the storage saved by superimposing the expert weights?
