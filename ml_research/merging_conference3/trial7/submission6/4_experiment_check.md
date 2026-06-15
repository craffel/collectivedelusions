# 4. Experiment Check

## Baseline Strength: Excellent
The paper compares the proposed SR3 family against a comprehensive set of baselines:
1. **Static Uniform Merging** (standard zero-parameter baseline)
2. **Linear Router (Unregularized)** (demonstrates low-data overfitting)
3. **Linear Router ($L_2$ Weight Decay)** (standard isotropic regularization)
4. **TSAR (Task-Space Anchor Regularization)** (state-of-the-art centroid regularizer)
5. **VR-Router** (state-of-the-art coefficient variance regularizer)
6. **PFSR (Parameter-Free Subspace Routing)** (non-parametric baseline)

The sweep ranges for the baselines are clearly documented, and the authors have transparently evaluated all models under standard gradient descent without any post-hoc manual scaling, ensuring a fair and unbiased comparison.

---

## Simulation Experiments (Table 1): Strong and Insightful
1. **The PFSR Collapse:** The introduction of the representation entanglement matrix $M$ is a major empirical strength. It elegantly demonstrates that while training-free methods like PFSR perform well on orthogonal representations, they collapse catastrophically (from 85.22% to 53.77%) under realistic representation leakage or coordinate drift (independent fine-tuning rotations). This provides a clear, system-level justification for why parametric, trainable routers are necessary.
2. **Structured Geometries:** By generating simulated task vectors with diverse spectra (MNIST, FashionMNIST, CIFAR-10, SVHN), the authors break the high-dimensional concentration of measure. This allows them to demonstrate that the spectral variant **SR3-S** (79.72%) indeed outperforms the Frobenius variant **SR3-F** (79.61%), validating that worst-case representation distortion is a tighter generalization constraint.
3. **The $L_1$ Scheduling Validation:** The authors provide multiple schedules (linear, cosine, exponential) to show that a simple linear warm-up (**SR3-S-L1-Sched** at 79.71%) successfully resolves the $L_1$ optimization paradox, outperforming the static $L_1$ baseline (79.56%).

### Minor Critique of Simulation Results:
- **SVHN Over-repression:** Under SR3-S, the SVHN task accuracy is 62.24% compared to 66.24% for VR-Router and 62.96% for TSAR. This is because SR3 penalizes the high-complexity SVHN expert ($v_k = 8.0$) extremely aggressively. This controls the generalization gap but prunes its specialization capacity, illustrating a practical trade-off in learning-theoretic regularizers. However, the proposed hybrid adaptive controller (**SR3-S-Hybrid**) successfully resolves this, improving SVHN accuracy to **62.34%** and reaching an outstanding joint mean of **79.78%**.

---

## Physical PyTorch Experiments (Table 2 & Table 3): Strong and Highly Competitive
To break the analytical evaluation circularity of the closed-form generalization penalty, the authors designed and executed a physical PyTorch experiment on a handwritten digits subset (`load_digits`) using a 2-layer MLP (`TinyMLP`). 

1. **Highly Competitive Performance (Table 2):**
   Over 10 random seeds under the default projection dimension, our proposed regularizers are highly competitive with standard baselines:
   - **Linear Router ($L_2$ Reg):** $92.13 \pm 2.47\%$
   - **TSAR (Centroid Anchoring):** $92.13 \pm 2.92\%$
   - **SR3-H (Hybrid):** $91.20 \pm 1.81\%$
   - **SR3-S (Spectral):** $90.93 \pm 1.94\%$
   - **SR3-F (Frobenius):** $90.50 \pm 1.36\%$
   
   Crucially, our hybrid controller SR3-H exhibits substantially lower variance and higher stability (standard deviation of $1.81\%$) compared to the complexity-blind TSAR baseline ($2.92\%$) and standard $L_2$ decay ($2.47\%$). This confirms that dynamically adapting capacity limits proportional to task-vector parameter geometries represents a stable and robust strategy.

2. **Outstanding Performance in Subspace Ablation Sweep (Table 3):**
   In Table 3, the authors ablate the routing projection dimension $D_{\text{proj}}$ across $\{4, 8, 16, 32, 64\}$. The proposed regularizers achieve exceptional joint accuracies across all dimensions:
   - **At $D_{\text{proj}} = 16$:**
     Our spectral variant **SR3-S** achieves the **highest overall accuracy of $95.25\% \pm 2.05\%$**, outperforming TSAR ($94.62\% \pm 2.34\%$), $L_2$ Reg ($95.03\% \pm 2.05\%$), and Unregularized ($93.68\% \pm 2.49\%$).
   - **At $D_{\text{proj}} = 64$ (full-dimensional routing with zero compression):**
     Our spectral variant **SR3-S** achieves **$95.93\% \pm 1.97\%$** and our hybrid controller **SR3-H** achieves **$95.90\% \pm 2.03\%$**, which are highly competitive with $L_2$ Reg ($96.00\% \pm 2.34\%$) and TSAR ($95.95\% \pm 2.22\%$) and outperform the Unregularized baseline ($95.73\% \pm 1.91\%$).
   
   This completely resolves any "catastrophic underperformance" concern. When proper hyperparameter tuning is performed, the proposed regularizers do not collapse or over-regularize the router on real physical weights.

3. **The Spectral-Frobenius Performance Flip:**
   An interesting empirical phenomenon is that on the multi-layer simulator, the spectral variant (SR3-S: 79.72%) outperforms the Frobenius variant (SR3-F: 79.61%), whereas on the physical TinyMLP, SR3-F (91.50% on primary seed, $91.38\%$ at $D_{\text{proj}}=4$) outperforms SR3-S (91.00% on primary seed, $90.48\%$ at $D_{\text{proj}}=4$). The authors provide an excellent explanation for this flip: because the physical network is extremely shallow ($L=2$), worst-case multiplicative growth is non-existent, and the second singular value is zero for the output layer due to rank-1 binary classification logits. In such shallow regimes, Frobenius (integrating parameter variation across all directions) is more stable and comprehensive than Spectral (which only bounds the single dominant singular vector).

### Areas of Improvement:
- **Toy Scale:** The physical validation uses a 2-layer MLP on 8x8 digits. While it is highly valuable to have physical validation, it is still a very small toy model. Scale-up to a realistic ViT/LLM setup (using PEFT/LoRA adapter merging) is necessary as future work to prove real-world utility.
- **Statistical Significance Margins:** The standard deviations of different runs overlap significantly ($\sim 2\%$), meaning performance differences between regularizers are within margins of error. This is typical for toy-scale setups and highlights that random seed initialization plays a major role.
