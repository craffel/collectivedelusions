# Impact, Presentation, and Qualitative Analysis

## Major Strengths
1. **Outstanding Pragmatic and Systems-ML Completeness:** The paper is exceptionally thorough. Rather than just proposing a mathematical algorithm, it addresses physical framework overhead, CPU thread scaling, memory-bandwidth limits (ARM Cortex-A72 model), cache structures, KV cache sharing for LLMs, and compiler-level C++ CustomOps.
2. **Exhaustive Empirical Validation:** Evaluating both vision (ViT-Tiny) and text (GPT-2) backbones, combined with physical execution benchmarks on actual edge hardware (Raspberry Pi 4), provides an incredibly strong empirical foundation.
3. **Honest Characterization of the "Serving Gap" and Boundary Conditions:** The authors do not hide the PyTorch framework overhead under large batch sizes; instead, they analyze it, propose a vectorized scatter-gather variant (SPS-VSG), and provide compiled loops with physical speedups. They also openly discuss representational overlap as a fundamental boundary condition and propose concrete mitigations (HCC, SHFT) with CUB-200 proof-of-concepts.
4. **Exceptional Writing Quality:** The manuscript is extremely well-structured, mathematically precise, and easy to follow.

## Areas for Improvement (Theorist Lens)
1. **Elevate Theoretical Rigor:** 
   - The paper relies heavily on empirical "discoveries" (e.g., that FSC jumps at Layer 3) and heuristic calibrations (e.g., IDC, Shannon-entropy scaling). It would be significantly stronger if the authors provided a rigorous mathematical framework modeling representation manifolds (e.g., under sub-Gaussian assumptions) to analytically prove routing error bounds.
   - Provide a formal proof or mathematical derivation for the stated sample complexity of Supervised Head Fine-Tuning (SHFT), rather than merely quoting general PAC learning bounds.
2. **Formally Prove Activation Bleeding Behavior:**
   - Instead of qualitatively explaining "activation bleeding" under overlapping domains, the authors should mathematically formalize the relationship between task manifold overlap (e.g., measured via Wasserstein distance or Fisher Separability) and the expected degradation of the blended activation output.
3. **Justify GMM Assumptions Mathematically:**
   - Provide a theoretical analysis justifying why a diagonal GMM is sufficient, and quantify the theoretical information loss compared to a full-covariance GMM in low-dimensional coordinate spaces.

## Overall Presentation Quality
The presentation is **excellent**. The narrative is logical, the figures are high-quality, and the mathematical notation is consistent. The authors have done a superb job of positioning their work relative to prior dynamic routing and model-merging schemes.

## Potential Impact and Significance
The potential impact of this work is **very high**, particularly for the systems-ML and TinyML communities:
- As edge devices continue to serve more modular, specialized adapters (e.g., in LLM and visual assistant pipelines), avoiding weight-switching DRAM overhead and multi-pass sequential backbone execution is critical.
- SPS-ZCA provides an actionable, training-free, and directly deployable blueprint for on-device modular serving.
- The C++ CustomOp benchmarks on Raspberry Pi 4 prove that the proposed latency benefits are immediately achievable on commodity edge CPUs.
