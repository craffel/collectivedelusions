# Novelty and Literature Positioning Assessment

## 1. Delta from Prior Work
The paper positions itself at the intersection of model merging, layer-wise adaptive ensembling, and statistical learning theory. The delta from key prior works is characterized below:

### A. Delta from Static Model Merging (e.g., Task Arithmetic, TIES-Merging, DARE-Merging, Git Re-Basin, ZipIt!)
- **Prior Work**: These methods merge task experts by applying a single, uniform scalar weight or averaging parameters statically across all layers. While simple and zero-parameter at test time, they do not account for the fact that task vector influence and representational alignment vary significantly across network layers.
- **Delta**: This paper uses layer-wise adaptive ensembling, showing that allowing merging coefficients $\alpha_k(l)$ to vary across layers can recover specialized capabilities. More importantly, it shows that unconstrained layer-wise tuning on tiny calibration splits causes severe "transductive overfitting" and representation shearing. The proposed RB-FTM and RB-DCTM constrain this adaptation to a low-dimensional spectral subspace to regularize the weight assembly.

### B. Delta from Unconstrained Layer-Wise Adaptive Ensembling (e.g., AdaMerging, PolyMerge)
- **Prior Work**: These methods optimize independent layer-wise merging coefficients $\alpha_k(l)$ on calibration datasets or use unconstrained curves.
- **Delta**: Unconstrained optimization of independent parameters is highly overparameterized ($K \times L$ dimensions). The proposed methods project the coefficients onto a continuous, low-dimensional harmonic spectral subspace of cutoff frequency $F \ll L$. This acts as a low-pass filter, filtering out high-frequency parameter noise and preventing transductive overfitting on few-shot calibration datasets.

### C. Delta from Rademacher-Bounded Polynomial Merging (RBPM)
- **Prior Work**: RBPM (Chatterjee et al., 2024) is the closest baseline. It projects layer-wise coefficients onto a low-degree polynomial subspace (e.g., $d=2$) and applies an analytical Rademacher penalty to control capacity.
- **Delta**: The paper identifies and addresses a severe architectural limitation of RBPM: **boundary runaway** (Runge's-like phenomenon). Because low-degree quadratic curves have rigid global shape constraints on $z \in [0, 1]$, fitting intermediate representation layers forces the coefficients at the boundary domains ($z=0$ and $z=1$) to extreme, runaway values. In deep networks, the first layers govern crucial low-level feature extraction and the final layers project representations into task logits; boundary runaway at these locations catastrophically degrades classification accuracy. RB-FTM and RB-DCTM resolve this by using bounded sinusoidal/cosinusoidal basis functions. For RB-DCTM, an implicit homogeneous Neumann boundary condition on the derivatives ($h'(0) = h'(1) = 0$) acts as an analytical "boundary buffer" that stabilizes the feature-extraction and classification boundaries.

### D. Delta in Learning-Theoretic Analysis
- **Prior Work**: Prior learning-theoretic bounds for model merging are often vacuous, lack direct bridges to downstream data generalization, or fail to account for the scaling of multiple tasks.
- **Delta**: 
  1. The authors derive a strictly tighter empirical Rademacher complexity bound for the DCT trajectory class than the Fourier alternative (Theorem 3.4), showing a reduction of a factor of 2 inside the logarithm due to the cosine-only basis.
  2. They bridge the trajectory-space depth complexity to the downstream prediction generalization error on unseen data samples via a formal covering-number derivation, establishing an explicit $\widetilde{\mathcal{O}}(1/\sqrt{N})$ decay rate over data samples that is completely independent of the underlying network's parameter count.
  3. They derive multi-task complexity scaling bounds showing that joint multi-task trajectory complexity is either independent of task count $K$ (under scalar projections) or scales strictly logarithmically ($\mathcal{O}(\sqrt{\ln(KF)/L})$ under independent Rademacher variables).

## 2. Characterization of Novelty
The novelty of this work is **significant and highly practical**, moving beyond incremental curve-fitting.
- **Theoretical Novelty**: The derivation of the trajectory-space Rademacher complexity for trigonometric and cosine ensembling classes, the formal proof of the tighter bound for the cosine-only basis, and the covering-number generalization bridge represent a high level of mathematical rigor. Although applying Fourier bases to parameter trajectories is mathematically standard, proving depth-wise complexity bounds and bridging them to data-space covering numbers is novel in the context of deep weight ensembling.
- **Conceptual/Physical Novelty**: Identifying "boundary runaway" as a physical pathology in polynomial weight ensembling, and showing how homogeneous Neumann boundary conditions in the DCT basis act as a "stabilizing buffer" for early feature-extraction and late logit-projection layers, bridges statistical learning theory and the physical architecture of deep networks in a highly intuitive and novel way.
- **Optimization Novelty**: Formulating the **Spectral Lasso** ($L_1$ penalty strictly on the harmonic coefficients, excluding the baseline DC term $a_{k,0}$) is a highly clever optimization choice. Standard weight-decay or L1 penalties on ensembling coefficients shrink the overall weight scale, causing under-scaled activations. Restricting the L1 penalty to non-DC components allows the ensembling weights to adapt while preserving the baseline uniform activation scale and automatically pruning redundant high-frequency harmonics to zero.
