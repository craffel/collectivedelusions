# Novelty and Literature Positioning Check

## Key Novel Aspects of the Submission
The core conceptual novelty of the submission lies in **framing the Overfitting-Optimizer Paradox in test-time model merging as a dimensionality/degrees-of-freedom problem, and resolving it through dynamic, non-parametric gradient sparsification.**

While previous works tried to solve overfitting by wrapping the objective in complex mathematical regularizers (distance penalties, normalization functions) or imposing rigid geometrical trajectory constraints, PG-Merge's novelty is its **uncompromising simplicity and minimalism**. It identifies that merely restricting the optimization search space to a tiny, dynamic subset of high-sensitivity layer-wise merging coefficients (the top-$p\%$) is fully sufficient to prevent transductive overfitting and representation decay on online test streams.

## The "Delta" From Prior Work
1. **Delta from AdaMerging (Yang et al., ICLR 2024):**
   - *AdaMerging* optimizes all $M = L \times K$ merging coefficients unconstrained, which easily overfits on small online calibration sets.
   - *PG-Merge* adds a dynamic, absolute-magnitude-based Top-$k$ sparse gradient filter and a strict parameter projection step to keep $(100 - p)\%$ of the coefficients perfectly frozen at each step, preventing momentum-driven parameter drift.

2. **Delta from RegCalMerge (ICCV 2025):**
   - *RegCalMerge* introduces complex, multi-hyperparameter auxiliary objectives, including Class-Capacity Normalization (CCN), Scale-Normalized Entropy Weighting (SNEW), and Elastic Spatial Regularization (ESR) which penalizes $L_2$ movement of parameters.
   - *PG-Merge* achieves superior or matching performance with **zero** auxiliary loss terms, **zero** additional optimization hyperparameters (aside from the intuitive sparsity ratio $p$), and **zero** computational overhead, confirming that complex spatial regularizers are redundant.

3. **Delta from PolyMerge (NeurIPS 2024):**
   - *PolyMerge* reduces the adaptation search space by forcing merging coefficients to lie along a low-degree (e.g., quadratic, $d = 2$) polynomial trajectory across model layers.
   - *PG-Merge* also reduces search space dimensionality but does so **dynamically and non-parametrically** based on local gradient sensitivity. This allows the network to adapt fine-grained, layer-specific routing patterns instead of being locked into a rigid, predetermined global polynomial curve (which causes PolyMerge to suffer catastrophic performance collapse on tasks like MNIST).

---

## Scholarly Critique: Literature Positioning and Attribution

From a scholarly perspective, while the paper is exceptionally clear and well-written, it exhibits several gaps in placing itself within the broader historical and concurrent literature. Addressing these would significantly elevate its academic rigor:

### 1. Connection to Sparse SGD / Top-k Sparsification in Distributed Optimization
The paper frames PG-Merge's sparse gradient masking as being closely related to PEFT (LoRA, Adapters) in Section 2.4. However, the core mathematical operation—sorting absolute gradient components and selecting the top-$p\%$ elements—is a direct descendant of **Top-$k$ gradient sparsification/compression** from the distributed deep learning optimization literature (e.g., Deep Gradient Compression by Lin et al., 2017; QSGD by Alistarh et al., 2017). 
In distributed training, Top-$k$ sparsification is used to reduce communication bandwidth by only transmitting high-magnitude gradients. PG-Merge repurposes this exact mathematical tool as an *analytical test-time regularizer* to combat the Overfitting-Optimizer Paradox. The paper should explicitly acknowledge and discuss this historical connection, attributing the mathematical formulation of Top-$k$ gradient selection to the distributed optimization field.

### 2. Contextualization within Selective Test-Time Adaptation (TTA)
In Section 2.3 (Test-Time Adaptation), the authors discuss Tent and general TTA instability. However, they miss a critical connection: **restricting parameter updates to a highly specialized subset is a foundational strategy across the entire history of TTA.** 
- For instance, *Tent* (Wang et al., 2021) selectively updates only Batch Normalization parameters to stabilize entropy minimization.
- *Parameter-Selective Mean Teacher (PSMT)* (2024) and CVPR 2026 works on Fisher-driven Selective Adaptation use Fisher Information to select and update only a sparse subset of critical parameters.
PG-Merge does the exact same thing but at the level of *model merging coefficients* instead of *neural network weights*. Situating PG-Merge within this broader context of "selective update strategies for stabilizing TTA" would make the paper's rationale feel much more grounded and historically coherent.

### 3. Missing Citations and References for Stated Concepts
- **"The Overfitting-Optimizer Paradox"**: The authors cite `[regcalmerge, polymerge]` when introducing this concept. They should explicitly clarify whether this exact term was coined by those works or if it is a general term they are formalizing here.
- **"QWS-Merge"**: In Section 1 and Section 2.2, the authors mention "concurrent works like QWS-Merge introduce quantum wave analogies using frozen random projections, normalized phase states, and interference equations..." but fail to provide a formal bibliographic citation for it. Even if QWS-Merge is concurrent or an preprint, a scholarly review expects a proper citation (e.g., arXiv preprint or Anonymous submission).
- **Optimizer State Mismatch (Adam Momentum)**: In Appendix A, the authors provide an excellent discussion on how SGD naturally avoids momentum leakage and state mismatch. This is a highly insightful point that connects deeply to optimization literature (e.g., the interaction of Adam's running moments with sparse/masked gradients, as studied in literature on sparse Adam or masked optimization). Citing relevant literature on sparse optimization with Adam (e.g., how standard Adam behaves poorly with dynamic sparse masks) would strengthen Appendix A's theoretical weight.

## Characterization of Novelty
Overall, the novelty of the paper is **significant but conceptually simple**. It is not a complex algorithmic breakthrough, but rather a **valuable deconstruction of unnecessary complexity** (an "Occam's reality check" for the model merging community). The "delta" from unconstrained AdaMerging is clear and well-motivated. By pointing out that prior works were over-engineering regularizers when a simple Top-$k$ gradient filter performs better, the paper makes a high-impact, refreshing contribution to the field.
