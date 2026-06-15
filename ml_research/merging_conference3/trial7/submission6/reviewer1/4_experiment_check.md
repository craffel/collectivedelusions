# 4. Experiment Check

## Critical Evaluation of the Experimental Setup

### 1. Synthetic Simulator with Injected Representation Entanglement
The continuous weight-merging simulator is a highly artificial, synthetic setup. To justify the use of parametric routing, the authors introduce a non-diagonal, hand-crafted "representation entanglement matrix $M$" to rotatably mix task coordinates, which catastrophically breaks the non-parametric Parameter-Free Subspace Router (PFSR). 

While this rotation matrix is designed to model "representation leakage" in physical networks, it represents an extreme, worst-case scenario specifically tailored to make PFSR fail. A parametric router easily untangles this rotation during calibration simply because it has trainable parameters. Thus, the "catastrophic collapse" of PFSR in Section 4.3 is largely an artifact of the synthetic simulator's design rather than a representative real-world phenomenon.

### 2. Toy Physical MLP Verification
The authors' "empirical validation on physical neural networks" (Section 4.4) uses an extremely small 2-layer Multi-Layer Perceptron (\texttt{TinyMLP}) trained on the scikit-learn \texttt{load\_digits} dataset. This dataset is a toy dataset consisting of low-resolution ($8\times 8$) images. 

While the authors claim this breaks the analytical circularity of the closed-form penalty, this setup is far too small and simple to draw meaningful conclusions about the behavior of modern, high-dimensional foundation models (such as LLMs or large Vision Transformers) where model merging is actually used.

## Failure of Empirical Results to Support the Core Claims

The central claim of the paper is that SR3 is a superior, theoretically optimal regularization strategy for dynamic weight merging. However, a close inspection of the empirical results in Table 1 and Table 3 reveals that the proposed method **fails to demonstrate any meaningful advantage over simple, standard baselines**, and in fact **performance degrades on physical networks.**

### 1. Negligible Performance Gain on Simulator (Table 1)
On the continuous simulator (which should represent the ideal environment for the proposed method):
- **TSAR (Centroid Anchoring)**, a simple, heuristic, complexity-blind method, achieves the highest Joint Mean accuracy of **79.90%**.
- **VR-Router**, another simple heuristic, achieves **79.79%**.
- **SR3-S-Hybrid** (their most complex, patched variant with adaptive gradient tracking) achieves **79.78%**, which fails to outperform the simpler TSAR.
- **SR3-S** (their primary spectral variant) achieves **79.72%**, which is practically identical to standard, simple isotropic $L_2$ weight decay at **79.71%** (a negligible $+0.01\%$ difference).

The extreme complexity of profiling singular value spectra offline and scaling weight decay forces layer-by-layer yields zero practical benefit over applying standard, uniform weight decay.

### 2. Generalization Performance Degradation on Physical MLP (Table 3)
On the physical MLP digit task (evaluated over 10 random seeds), the failure of the proposed method is even more pronounced:
- **Linear Router ($L_2$ Reg.)** achieves a 10-seed mean of **$92.13\% \pm 2.47\%$**.
- **TSAR (Centroid Anchoring)** achieves a 10-seed mean of **$92.13\% \pm 2.92\%$**.
- **SR3-F** (Ours - Frobenius) achieves a 10-seed mean of **$90.50\% \pm 1.36\%$**.
- **SR3-S** (Ours - Spectral) achieves a 10-seed mean of **$90.93\% \pm 1.94\%$**.
- **SR3-H** (Ours - Hybrid) achieves a 10-seed mean of **$91.20\% \pm 1.81\%$**.

In the 10-seed average, **every single variant of SR3 underperforms standard $L_2$ weight decay and TSAR.** 
Standard $L_2$ decay and TSAR outperform SR3-F by $1.63\%$, SR3-S by $1.20\%$, and SR3-H by $0.93\%$ in mean accuracy. 

This is a critical finding that completely invalidates the claim of SR3's practical superiority. When applied to real parameters with no analytical penalty, the asymmetric, geometry-aware weight decay actually degrades classification accuracy, likely because it over-regularizes complex task experts (like SVHN) and starves the router of necessary capacity.

### 3. Analytical Evaluation Circularity in Simulator (A Structurally Biased Evaluation)
The simulator's test-time distance evaluation incorporates an analytical generalization gap penalty directly modeled after the Rademacher bound:
$$\text{Gap}_k = \eta_{\text{noise}} \|W_k\|_2 \|V_k\|_F$$
Because SR3 is mathematically derived to minimize precisely this product ($\|W_k\|_2 \|V_k\|_F$), the simulator's test evaluation is **structurally biased (circular)** in favor of SR3. 

The fact that SR3 *still* cannot outperform the simple TSAR heuristic in Table 1, even when the simulator is literally programmed with a test penalty that directly matches the SR3 objective, is a strong testament to the practical limitations of the method. And as expected, once this closed-form penalty is removed in the physical MLP experiment, the performance of SR3 collapses below that of standard $L_2$ decay and TSAR.
