# 2. Novelty Check

## Key Novel Aspects and the 'Delta' from Prior Work
The scientific novelty of this paper lies not in the creation of an elaborate, highly complex model architecture, but in a rigorous, refreshing **conceptual deconstruction of complexity creep** in dynamic model merging. 

The primary "delta" from prior work (specifically QWS-Merge, Vance et al., 2025) is two-fold:
1. **Deconstruction of SVHN Collapse:** Prior work asserted that classical linear routing was structurally incapable of robust dynamic parameter blending and collapsed catastrophically on SVHN (reported at $15.30\%$). This paper shows that this collapse is not a structural limitation, but rather an artifact of sub-optimal hyperparameter and design choices (deep task-warped routing, excessive learning rates, and over-optimization). By correcting these choices, a classical unregularized linear router achieves $94.87\%$ on SVHN and $95.46\%$ Joint Mean, rendering complex dynamic fusion frameworks obsolete.
2. **Post-Hoc Routing Regularization:** While routing regularizations (like load-balancing or entropy penalties) are well-established in sparse Mixture-of-Experts (MoE) architectures, their application to the gating networks of post-hoc dynamic model merging was previously overlooked. The paper introduces **Robust Linear Routing (RLR)**, which ports standard $L_2$ weight decay and Softmax Temperature scaling to a tiny 768-parameter classical gating layer, stabilizing its optimization and preventing overconfident gating under OOD representation shifts and heterogeneous batching.

## Characterization of Novelty
The paper's novelty can be characterized as **highly significant and refreshing**. 

Rather than proposing a "highly engineered behemoth," the authors champion the principle of **Occam's razor**: demonstrating that a simpler, elegant, and mathematically transparent method (when properly configured and regularized) can outperform a highly convoluted, multi-stage, quantum-inspired framework. 

In a field often prone to unnecessary mathematical obfuscation and over-engineering, this work represents a critical scientific course correction. It reminds the community of the enduring value of thoroughly understanding and regularizing classical baselines before introducing complex machinery.
