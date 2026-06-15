# Peer Review

## 1. Summary of the Paper
This paper presents a rigorous methodological and theoretical deconstruction of "quantum-inspired" deep learning models in the context of test-time dynamic model merging. Specifically, it targets Quantum Wavefunction Superposition Merging (QWS-Merge), translating its complex wave-inspired superposition framework into classical neural terms (namely, an over-parameterized and unstable non-monotonic cosine activation function). 

To isolate the true drivers of dynamic task routing, the authors introduce the **Isolating Coordinate Sandbox**, which decouples routing performance ($\text{Error}_{routing}$) from weight-space coordinate misalignment ($\text{Error}_{alignment}$). Within this controlled space, they propose a family of simpler **Layer-wise Low-dimensional Classical Routers (L3-Router)** utilizing linear, Tanh, and Softmax channels. 

The empirical and mathematical findings are highly illuminating:
1. Under rigorous sandbox validation, the wave cosine activation of QWS-Merge collapses catastrophically to a Joint Mean of **36.10%** (underperforming uniform static merging at **43.40%**) and yields near-random accuracy (**2.00%**) on out-of-distribution (OOD) SVHN.
2. The proposed classical **L3-Linear** router avoids this collapse, achieving a Joint Mean of **63.10%** (+27.00% absolute improvement over QWS-Merge).
3. The simplest baseline—a global, unregularized classical **Linear Router**—outperforms all multi-layer models, achieving **67.20%** Joint Mean accuracy. This indicates that layer-wise specialized routing is highly over-engineered for shared-head classification merging.
4. During mixed-task deployment streams, batch averaging of routing coefficients triggers **heterogeneity collapse**, severely degrading linear and quantum routers.
5. The paper critically deconstructs its own proposed **L3-Softmax** model, demonstrating that its apparent relative robustness to stream shifts is a **"Robustness-Accuracy Illusion"** created by the Softmax simplex constraint forcing coefficients toward a mediocre, uniform-like average.
6. The authors provide a closed-form algebraic proof of **layer-averaging collapse** to explain why global baselines beat multi-layer routers, back up their findings with a real-scale **CLIP-ViT-B/16** pilot, and outline a compiler-level parallel execution roadmap using Triton kernels and LoRA parameterization.

---

## 2. Strengths and Weaknesses

### Strengths:
1. **Crucial Deflationary Theoretical Contribution:** The paper is a highly valuable correction to a growing trend in deep learning literature that wraps standard neural layers in complex mathematical analogies (like quantum wave mechanics) without rigorous classical justification. By translating QWS-Merge into a classical over-parameterized cosine channel, it exposes the baseline confounders (omission of proper regularization and comparison against crippled classical models) that drove the supposed superiority of the quantum-inspired SOTA.
2. **Exceptional Methodological Rigor and Transparency:** The authors apply the same rigorous scientific skepticism to their own methods as they do to prior work. Rather than celebrating their proposed L3-Softmax router for its relative stability under heterogeneous streams, they mathematically and conceptually expose it as a "Robustness-Accuracy Illusion" driven by simplex constraints forcing dynamic coefficients toward a mediocre average.
3. **Rigorous and Proactive Empirical Audits:** The authors include extensive, proactive audits to address potential confounding variables:
   - **Task Correlation Sweep (Section 13):** Shows that classical linear routing's superiority is robust across varying levels of task overlap ($\rho \in [0, 0.75]$), refuting the idea that orthogonality artificially favors linear projection.
   - **Learning Rate Sensitivity Sweep (Section 9):** Rules out optimization bias for QWS-Merge across a wide range of learning rates ($\eta \in [10^{-2}, 10^{-4}]$).
   - **Multi-Seed Robustness Audit (Section 12):** Evaluates all methods across 5 independent random seeds with complete dataset regeneration.
   - **True Layer-by-Layer Weight Merging Audit (Section 11):** Demonstrates that the findings translate to deep parameter-space weight merging where routing coefficients do not collapse under layer averaging.
4. **Actionable Deployment and Hardware Roadmap:** The compiler-level discussion of Triton-based dynamic weight assembly, low-rank LoRA parameterizations, HBM-to-SRAM memory transfer scaling, and zero-shot CLIP text-prompt projections is highly detailed and of great practical utility for systems engineers.
5. **Outstanding Writing and Clarity:** The manuscript is exceptionally well-structured, precise, and intellectually honest.

### Weaknesses:
1. **Theoretical Overclaim on the Universality of Layer-Averaging Collapse:**
   In Section 5 and Section 8, the authors claim that the proof of layer-averaging collapse (Section 3.5) "applies universally to *any* dynamic routing model." This is mathematically incorrect. The algebraic proof relies strictly on the linear-algebraic property of linearity under summation:
   $$\bar{\alpha}_k = \frac{1}{L} \sum_{l=1}^L \left( \langle \psi(x)_b, W_{l, k} \rangle + B_{l, k} \right) = \left\langle \psi(x)_b, \frac{1}{L} \sum_{l=1}^L W_{l, k} \right\rangle + \frac{1}{L} \sum_{l=1}^L B_{l, k}$$
   For non-linear routing models (such as Tanh, Softmax, or the wave cosine in QWS), the sum/average of $L$ layer-wise functions cannot be simplified to a single-layer function of the exact same family:
   $$\frac{1}{L} \sum_{l=1}^L \tanh\left(\langle \psi(x)_b, W_{l, k} \rangle + B_{l, k}\right) \neq \tanh\left(\langle \psi(x)_b, W_{eff, k} \rangle + B_{eff, k}\right)$$
   A mixture of non-linear functions represents a mixture model with $L$ times more representational capacity, rather than collapsing to a single-layer routing space. The authors must revise this claim to clarify that:
   - *Strict algebraic collapse* holds only for linear routers (L3-Linear).
   - For non-linear layer-wise models, the collapse is an *optimization and generalization collapse* driven by high-dimensional backpropagation noise and parameter overfitting on tiny calibration splits (64 samples).
2. **Ambiguity in Global Linear Router Baseline Definition and Parameter Count:**
   There is a minor contradiction in how the global Linear Router baseline is parameterized and defined in different sections:
   - In **Section 1**, the authors state the global Linear Router uses a high-dimensional projection matrix with **768 parameters**, which they note makes it highly susceptible to overfitting.
   - In **Section 3.3**, they formally define the global Linear Router as mapping the high-dimensional representation $z(x)_b \in \mathbb{R}^D$ directly to a $K$-dimensional space:
     $$\boldsymbol{\alpha}_{:, b}^{Global}(l) = \text{Softmax}\left( \mathbf{W}^{Global} z(x)_b + \mathbf{B}^{Global} \right)$$
     For $K=4$ tasks and $D=192$ feature dimensions in the sandbox, this would require $4 \times 192 + 4 = 772$ parameters. For CLIP ($D=768$), it would require $4 \times 768 + 4 = 3076$ parameters.
   - In **Section 11 (True Layer-by-Layer Merging Audit)**, the authors state:
     > "the global classical Linear Router... reduces its trainable parameter count by 14-fold (utilizing only 16 parameters instead of 280)"
     
     If the global router utilizes only **16 parameters**, it cannot be mapping the high-dimensional representation $z(x)_b$ directly. Instead, it must be mapping the *projected low-dimensional representation* $\psi(x)_b \in \mathbb{R}^d$ ($d=4$) to $K=4$ task scores via a single linear layer without bias (i.e., $4 \times 4 = 16$ parameters).
     
     The authors must resolve this contradiction and standardize the definition and parameter counts of the global Linear Router baseline across all sections.
3. **Hardware Constraints in Triton-Based Feasibility Claims:**
   The compiler-level discussion of Triton kernels is exceptionally detailed. However, loading $K$ distinct task-specific LoRA parameters from High Bandwidth Memory (HBM) to SRAM at runtime can trigger significant synchronization stalls and warp scheduling overheads on modern GPU architectures. The authors should explicitly note these implementation hurdles in Section 7 to temper the feasibility claims.

---

## 3. Soundness

**Rating: Good**

### Justification:
The methodology is exceptionally sound, using appropriate mathematical formalisms and an elegant synthetic sandbox to isolate routing dynamics. The addition of Section 11's true layer-by-layer weight-merging scheme, Section 13's task correlation sweeps, and Section 9's learning rate audits are exemplary practices of rigorous scientific hygiene. 

However, the rating is capped at "Good" due to:
- The mathematically incorrect assertion that the layer-averaging collapse proof applies "universally to any dynamic routing model" (since it only holds strictly for linear models).
- The ambiguity regarding the global Linear Router's parameter counts (768/772 vs. 16 parameters) and representation mappings across Section 1, Section 3.3, and Section 11.
Addressing these minor theoretical and definitional discrepancies will elevate the soundness to "Excellent."

---

## 4. Presentation

**Rating: Excellent**

### Justification:
The paper is written with high clarity and is structured beautifully. The mathematical notation is rigorous and consistent, the definitions are complete, and the figures/tables are highly informative. The authors are incredibly transparent and scientifically honest about the limitations of their own models (specifically auditing L3-Softmax and exposing its mediocrity under stream heterogeneity), which is highly commendable.

---

## 5. Significance

**Rating: Excellent**

### Justification:
This paper addresses a highly important problem in weight-space model merging. By systematically deconstructing QWS-Merge and demonstrating that classical linear projections with standard regularization are highly stable and achieve superior dynamic capacity without auxiliary wave variables, the paper serves as a vital cautionary tale. It is highly likely to influence future research by pushing the community toward rigorous baseline tuning and ablation of structural complexity rather than adopting over-engineered mathematical metaphors. Additionally, the compiler-level implementation roadmap using Triton kernels and LoRA provides immense practical utility for systems engineers.

---

## 6. Originality

**Rating: Excellent**

### Justification:
The work provides exceptional new insights by deconstructing an existing "quantum-inspired" method and demonstrating that its supposed novelty is a baseline-crippled illusion. The design of the Isolating Coordinate Sandbox to decouple routing error from coordinate alignment conflict represents a highly creative and effective combination of theoretical control and empirical evaluation. Exposing the "Robustness-Accuracy Illusion" and analyzing the impact of stream heterogeneity (mixed-task batching) are highly original contributions to the model-merging literature.

---

## 7. Questions and Suggestions for the Authors

1. **Clarify the Generality of Layer-Averaging Collapse:**
   In Section 5 and Section 8, please revise the claim that the layer-averaging collapse proof "applies universally to any dynamic routing model." Under a non-linear mapping (such as Tanh, Softmax, or the cosine activation of QWS), the sum/average of layer-wise functions does not algebraically collapse to a single instance of the same function family. Please clarify that strict algebraic collapse is a property unique to linear routers (L3-Linear). For non-linear layer-wise models, explain that the collapse observed in the sandbox is an *optimization and generalization collapse* driven by high-dimensional backpropagation noise and parameter overfitting on tiny calibration splits, rather than a strict algebraic identity.
2. **Standardize the Global Linear Router Definition:**
   Please resolve the parameter count and input coordinate discrepancy of the "global classical Linear Router" baseline across Section 1, Section 3.3, and Section 11:
   - Does it map the high-dimensional representation $z(x)_b \in \mathbb{R}^D$ directly (as defined in Section 3.3, resulting in 768 parameters in Section 1 or 772 in Section 3.3)?
   - Or does it map the projected low-dimensional representation $\psi(x)_b \in \mathbb{R}^d$ ($d=4$) to $K=4$ task scores via a single linear layer without bias (as implied by the 16 parameters in Section 11)?
   
   Standardizing this definition and resolving the parameter count discrepancy across all sections is necessary to ensure methodological consistency and reproducibility.
3. **Tempering Triton Kernel Feasibility Claims:**
   In Section 7.3, please expand the compiler-level discussion to explicitly acknowledge the physical memory synchronization and warp scheduling bottlenecks on modern GPU architectures (e.g., NVIDIA H100 or A100) when concurrently reading $K$ distinct LoRA parameter matrices from HBM to SRAM at runtime. This will provide a more balanced and hardware-grounded cost-benefit analysis.

---

## 8. Overall Recommendation

**Rating: 5 (Accept)**

### Justification:
This is a technically solid, highly rigorous, and exceptionally well-written paper that makes a significant deflationary contribution to the weight-space model merging literature. By deconstructing "quantum-inspired" metaphors, exposing critical baseline omissions, and introducing the Isolating Coordinate Sandbox to study routing dynamics, the paper provides deep scientific clarity. The thoroughness of its audits (multi-seed, learning rate sensitivity, task correlation, and true layer-by-layer merging) is exemplary. While there is a minor theoretical overclaim regarding the universality of the layer-averaging collapse proof and an ambiguity in the global baseline's parameter counts, these are minor revisions that the authors can easily resolve. The paper meets the bar for acceptance and will serve as an essential cautionary tale for the deep learning community.
