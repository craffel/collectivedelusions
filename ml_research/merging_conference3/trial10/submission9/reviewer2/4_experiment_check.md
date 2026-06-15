# 4. Experimental Setup & Results Evaluation

## Evaluation of the Experimental Setup
The experimental setup of this paper is **rigorous, comprehensive, and exceptionally thorough**. 
- **The Analytical Coordinate Sandbox (ACS):** The authors construct a high-fidelity 14-layer, 192-dimensional simulation in PyTorch to model sequential multi-task serving. This isolates routing dynamics from confounding backbone architecture variables.
- **Stream Configurations:** By evaluating under both **Homogeneous Streams** (long blocks with high noise to test noise filtering) and **Heterogeneous Streams** (rapid step-by-step task transitions to test responsiveness), the authors directly target the two extremes of the Jitter-Lag Trade-Off.
- **Manifold Geometries:** The authors evaluate both **Orthogonal Manifolds** (disjoint 48-dimensional blocks) and **Overlapping Manifolds** (sharing a 12-dimensional boundary region to test sensory confusion).
- **Sensory Noise Profiles:** The simulation uses realistic noise distributions matching standard vision datasets (MNIST, FashionMNIST, CIFAR-10, SVHN).

---

## Evaluation of the Baselines
The paper compares AIR against a highly competitive and comprehensive suite of five baselines:
1. **Expert Oracle:** An omniscient router representing the theoretical ceiling of representation alignment.
2. **Uniform Merging:** A simple static baseline representing equal weighting.
3. **SABLE \cite{sable2024}:** A state-of-the-art stateless angular gating router that reacts instantly but suffers from routing jitter.
4. **Momentum-Merge \cite{momentummerge2025}:** A stateful linear filter.
5. **ChemMerge \cite{chemmerge2025}:** A stateful continuous-time biochemical ODE filter.
6. **PAC-Kinetics \cite{packinetics2025}:** An optimized continuous-time gating framework using unrolled gradient descent.

This selection of baselines is excellent, covering the entire spectrum of state-of-the-art stateful, stateless, and optimization-based dynamic routing methods.

---

## Do the Results Support the Claims?
**Yes, the results support the authors' claims comprehensively, with multiple lines of robust empirical evidence:**

1. **Jitter-Lag Trade-Off Resolution (Table 1 and Table 5):**
   - Under stable homogeneous streams, AIR matches SABLE's accuracy ($66.44\%$) while slashing routing jitter by up to **$2.49\times$** (down to $0.0364 \pm 0.0009$).
   - Under rapid heterogeneous transitions, stateful filters (ChemMerge, Momentum-Merge) suffer from representational lag, dropping to $\sim 53$--$54\%$ alignment accuracy. AIR maintains near-oracle tracking speed ($1.4202$ jitter) and near-oracle alignment accuracy ($66.23\%$).
   - This directly and elegantly validates that AIR's precision-weighted prediction errors successfully balance top-down expectations and bottom-up sensory predictions.

2. **Robustness under Model Mismatch (Table 3):**
   - The **High-Dimensional Nonlinear Manifold Stress Test** is a standout experiment. Under non-invertible sinusoidal-quadratic warping and heavy-tailed Student's $t$-distributed noise ($\nu = 3$), stateless SABLE's routing jitter causes its categorical classification accuracy to collapse to $93.99\%$ (compared to Oracle's $99.36\%$). Stateful filters collapse to $\sim 47$--$48\%$ alignment accuracy.
   - AIR is exceptionally robust: it maintains a near-oracle categorical classification accuracy of **$98.83\%$** and a representation alignment accuracy of **$59.38\%$** (directly outperforming PAC-Kinetics) while reducing SABLE's routing noise by over **$3.6\times$**. This confirms that routing stability is crucial for downstream task accuracy in non-linear spaces.

3. **Mechanistic Necessity of Active Inhibition (Section 4.5 & Appendix E):**
   - The ablation study comparing AIR with a non-negative variant ($\mathbf{W} \ge 0$) shows matching sequence-averaged alignment accuracy, but continuous trajectory visualizations (Figure 3) reveal a **15-step transient lag** at task switches.
   - The authors' mathematical deconstruction in Appendix E (explaining that sequence averaging heavily dilutes this lag because boundary steps comprise only a small fraction of the sequence) is intellectually outstanding. It highlights how standard sequence-averaged metrics can mask critical localized bottlenecks, proving the need for excitatory-inhibitory balance in generative mapping.

4. **Registry Scaling and Calibration Generalization (Appendix G & H):**
   - Under $K=16$ experts, dense AIR matches optimal accuracy and reduces SABLE's jitter by **$1.86\times$**.
   - Restricting the generative mapping $\mathbf{W}$ to be diagonal (**AIR-Diagonal**) reduces parameters to linear $\mathcal{O}(K)$ (only 80 parameters). Calibrated on a tiny sequence of $T_{\text{cal}}=32$ steps, it achieves outstanding Homogeneous accuracy ($45.76\%$) and Heterogeneous accuracy ($45.37\%$) with high stability ($0.4198$ jitter). This proves diagonal parameterization is a highly viable, sample-efficient regularizer for large-scale Mixture-of-Experts.
   - Cross-sequence calibration shows that calibrating on stable vs. dynamic streams has negligible impact, indicating that learned precision parameters are highly stable and generalize well.

5. **Avenue of Alternative Projection Spaces (Appendix D, Section 8):**
   - The implementation of **Contractive Autoencoders (CAEs)** as an alternative to linear PCA is a brilliant, creative experiment. CAE-AIR achieves a perfect **$100.00\%$** categorical classification accuracy, a remarkable **$73.26\%$** representation alignment on Orthogonal Manifolds, and a pristine **$0.0000$** routing jitter (completely eliminating high-frequency noise). This shows that non-linear contractive projection flattens localized manifolds, allowing the linear-Gaussian active inference model to achieve absolute stability.
