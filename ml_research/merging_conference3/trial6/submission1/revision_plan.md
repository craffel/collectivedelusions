# Revision Plan - Addressing Peer Review Critiques (Round 19 / Fourteenth Round)

We have systematically and rigorously addressed all peer review feedback, including the minor presentation gap regarding the visibility and integration of the Appendix-based extensions in the main body text:

## 1. Main-Text Integration of CP/Tucker Tensor Keys (Appendix I / Appendix~\ref{app:tensor_decompositions})
- **Critique:** The reviewer suggested discussing the CP-decomposition and Tucker-decomposition generalizations for 3D/4D weights to extend the framework beyond 2D dense layers.
- **Revision:** We have added an explicit cross-reference in Section 3.1 of `submission/sections/03_method.tex` referencing Appendix~\ref{app:tensor_decompositions}. This directly highlights how CP/Tucker carrier key decompositions prevent the storage complexity from scaling exponentially with dimensions.

## 2. Main-Text Integration of Routing Optimizer Configurations (Appendix J / Appendix~\ref{app:routing_optimizer_details})
- **Critique:** The reviewer requested clarification regarding router parameters, optimizer settings, training protocols (adaptation vs. post-hoc), and calibration sensitivity.
- **Revision:** We have added an explicit cross-reference in Section 3.3 of `submission/sections/03_method.tex` referencing Appendix~\ref{app:routing_optimizer_details}, detailing our use of a 64-sample calibration set, AdamW optimization settings, post-hoc calibration, and the associated calibration set size sweep.

## 3. Main-Text Integration of Quantization Compatibility (Appendix K / Appendix~\ref{app:quantization_boundaries})
- **Critique:** The reviewer asked how EHPB weight-reconstruction noise interacts with low-bit (INT4/INT8) quantization boundaries on on-device accelerators.
- **Revision:** We have added an explicit cross-reference in Section 3.6 of `submission/sections/03_method.tex` referencing Appendix~\ref{app:quantization_boundaries}, detailing the systems-level Precision Underflow Paradox and proposed mitigations.
