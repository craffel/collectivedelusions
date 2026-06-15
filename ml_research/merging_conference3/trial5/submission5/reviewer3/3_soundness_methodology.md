# Paper Evaluation: 3. Soundness and Methodology

## Clarity of Description and Mathematical Rigor
The methodology is written with high clarity and is mathematically structured. The authors provide explicit equations for:
- The multi-task model merging weight assembly (Section 3.1).
- The low-dimensional input state projection and its normalization (Section 3.2).
- The exact formulation of QWS-Merge and the proposed L3-Router variants (Section 3.3).
- The layer-averaging collapse proof (Section 3.5).
- The gradient backpropagation and chain rule expansion in true layer-by-layer weight merging (Section 11).

The addition of Section 3.2's detailed mitigations for representation drift (Online Incremental PCA and Johnson-Lindenstrauss random projection) is highly rigorous, utilizing classic theorems like the Johnson-Lindenstrauss Lemma to guarantee distance preservation under stream shifts. This demonstrates strong theoretical framing.

## Appropriateness of Methods
- **The Isolating Coordinate Sandbox:** This is an appropriate and elegant methodological tool. By setting the weight-space alignment error $\text{Error}_{alignment} \approx 0$, the authors establish a controlled environment where the routing algorithm's dynamics can be isolated and evaluated without confounding factors.
- **The True Layer-by-Layer weight-merging scheme (Section 11):** By implementing this scheme (fine-tuning experts from a shared base and propagating activations layer-by-layer without averaging), the authors directly answer the objection that classification-head merging is too simplistic. It proves that the findings hold even in deep parameter-space merging.
- **Optimization Audit (Section 9) and Multi-Seed Audit (Section 12):** These are excellent practices for scientific fairness, proving that QWS-Merge's failure is structural and not an artifact of a bad learning rate or a lucky seed.

## Technical Flaws and Ambiguities
Despite its high quality, a rigorous theoretical audit reveals a few discrepancies and minor flaws in the paper:

1. **Incorrect Overclaim of the Universality of Layer-Averaging Collapse:**
   As highlighted in the novelty check, the authors claim that the algebraic collapse proof in Section 3.5 applies "universally to *any* dynamic routing model." This is mathematically incorrect. The proof relies strictly on the linear-algebraic property of linearity under summation. For any non-linear mapping (such as Tanh, Softmax, or the cosine activation in QWS), the sum of layer-wise non-linear functions cannot be simplified to a single-layer function of the same family. It represents a mixture model with $L$ times more capacity. The authors should revise their claims to clarify that:
   - *Strict algebraic collapse* holds only for linear routers (L3-Linear).
   - For non-linear routers, the collapse is *not* algebraic, but rather an *optimization and generalization collapse* driven by high-dimensional optimization noise and parameter overfitting on tiny calibration sets.

2. **Discrepancy in Global Linear Router Parameter Count:**
   There is an inconsistency in how the "global classical Linear Router" is parameterized in different parts of the manuscript:
   - In **Section 1**, the authors state the global Linear Router uses a high-dimensional projection matrix with **768 parameters**, which they note makes it highly susceptible to overfitting.
   - In **Section 3.3**, they formally define the global Linear Router as mapping the high-dimensional representation $z(x)_b \in \mathbb{R}^D$ directly to a $K$-dimensional space:
     $$\boldsymbol{\alpha}_{:, b}^{Global}(l) = \text{Softmax}\left( \mathbf{W}^{Global} z(x)_b + \mathbf{B}^{Global} \right)$$
     For $K=4$ tasks and $D=192$ feature dimensions in the sandbox, this would require $4 \times 192 + 4 = 772$ parameters. For CLIP ($D=768$), it would require $4 \times 768 + 4 = 3076$ parameters.
   - In **Section 11 (True Layer-by-Layer Merging Audit)**, the authors state:
     > "the global classical Linear Router... reduces its trainable parameter count by 14-fold (utilizing only 16 parameters instead of 280)"
     
     If the global router utilizes only **16 parameters**, it cannot be mapping the high-dimensional representation $z(x)_b$ directly. Instead, it must be mapping the *projected low-dimensional representation* $\psi(x)_b \in \mathbb{R}^d$ ($d=4$) to $K=4$ task scores via a single linear layer without bias (i.e., $4 \times 4 = 16$ parameters).
     
     The authors must resolve this contradiction. Is the "global classical Linear Router" mapping the high-dimensional representation $z(x)_b \in \mathbb{R}^D$ directly (as defined in Section 3.3 and evaluated in CLIP) or is it mapping the projected low-dimensional state $\psi(x)_b \in \mathbb{R}^d$ (as implied by the parameter count in Section 11)? Standardizing this definition across all sections is necessary for methodological reproducibility.

## Reproducibility
The reproducibility of this paper is **excellent**. The authors provide:
- Exact dimensions ($L=14, D=192, d=4, B=256, \text{Calibration Split}=64$).
- Specific optimizer and hyperparameter details (AdamW, LR=$10^{-2}$, $\lambda_{wd}=10^{-3}$, 100 epochs, gradient norm clipping threshold of 1.0).
- Detailed initialization ranges for all QWS parameters in Section 9.
- A clear deployment roadmap in Appendix Section 7 with mathematical formulations for unsupervised PCA and zero-shot text-prompt projection.
- FLOP count formulas ($2 K \cdot M$ per layer) and memory bandwidth scale factors ($(1 + K \cdot \gamma) M$) for Triton-based implementations, providing highly actionable deployment details.
