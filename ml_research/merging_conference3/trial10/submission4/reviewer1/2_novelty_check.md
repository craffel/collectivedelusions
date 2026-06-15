# Evaluation Stage 2: Novelty Check

## Key Novel Aspects
1. **Low-Precision Latent Ensembling:** Prior latent ensembling methods (SABLE, ChemMerge, Momentum-Merge) are designed strictly for floating-point precision (Float32). This work is the first to systematically analyze and address their low-precision deployment bottlenecks (INT8/INT4).
2. **Error Feedback in Blending Trajectories:** Adapting error-diffusion feedback concepts (EF-Smooth and AEF) to deep representation routing and activation blending layers is a creative and highly practical application of signal-processing techniques to deep learning pipelines.
3. **Permutation-Invariant Single-Pass Apportionment (PI-SPA):** Proposing a sorting-free, branchless $O(K)$ alternative to Hamilton's apportionment method of simplex projection specifically optimized for edge SIMD pipelines is a valuable and highly practical system-level novelty.

## Delta from Prior Work
- **Prior Work:** SABLE, ChemMerge, and Momentum-Merge show strong dynamic ensembling capabilities under Float32 but collapse to static uniform merging when quantized naively. Standard model quantization techniques (e.g., LLM.int8(), SmoothQuant) address individual network weights and activations but do not address the specific routing dynamics, centroid overlapping, or "small-step quantization bottleneck" of dynamic activation-blending cascades.
- **The 'Delta' of QA-Merge:** It fills the gap between high-precision dynamic ensembling and low-precision edge deployments. It keeps the routing network and blending loops entirely in the integer domain, avoiding expensive float-to-int dynamic conversions, while recovering full-precision ensembling gains.

## Characterization of Novelty
The novelty is characterized as **highly practical and incremental-to-significant**:
- From a theoretical machine learning perspective, the individual pieces (STE, error-feedback noise shaping, and Hamilton's method of apportionment) are established concepts in their respective fields (quantization, signal processing, and social choice theory/resource allocation).
- However, their synthesis into a cohesive, hardware-compatible suite (**QA-Merge**) designed specifically to resolve the "Quantization Collapse" of dynamic ensembling represents a substantial and highly original engineering contribution. The proposed **PI-SPA** algorithm is a novel variant that resolves a critical compile-time and system-level sorting bottleneck for edge hardware.
