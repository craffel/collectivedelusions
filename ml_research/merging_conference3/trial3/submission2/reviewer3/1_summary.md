# Peer Review Report: Summary of Submission

## 1. Main Topic and Scope
This paper investigates the weight-space model merging paradigm, focusing specifically on the recently proposed approach of online test-time adaptation (TTA) for dynamically optimizing layer-wise merging coefficients on unlabeled target test streams. The paper critically evaluates the foundational premises of TTA-based model merging, exposing what it describes as two core methodological limitations:
- **The "No-Data" Strawman:** Prior TTA works compare complex, backpropagation-intensive online adaptation exclusively against a naive, completely unoptimized uniform merging baseline. The authors argue that in most real-world scenarios, practitioners have access to a very small labeled validation set (e.g., 5 to 10 samples per task), which can be used to optimize coefficients offline.
- **Vulnerability to Target Distribution Shift:** Unsupervised online TTA methods (relying heavily on entropy minimization) are shown to experience severe transductive noise fitting and catastrophic representational collapse when deployed under realistic, non-idealized stream conditions, such as extreme label shift, temporal task clustering, and tiny batch sizes.

To address these vulnerabilities, the paper proposes **Offline Few-Shot Validation Tuning (OFS-Tune)**, a training-free framework that optimizes merging coefficients offline on a tiny labeled validation set. By parameterizing the layer-wise coefficient search space with low-dimensional models (such as low-degree polynomials across layers, i.e., Poly-Val-Merge) or global task-specific coefficients (GT-Merge), OFS-Tune acts as a structural low-pass filter, preventing validation overfitting and ensuring robust generalization.

---

## 2. Methodology and Approach
The proposed OFS-Tune optimizes the merging parameters $\theta$ of a static multi-task merged model $W_{merged}$.
- **Weight Merging Formulation:** Task vectors $V_k = W_k - W_{base}$ represent task-specific capabilities relative to a pre-trained base model $W_{base}$. The merged weights at layer $l$ are given by $W_{merged}^{(l)}(\theta) = W_{base}^{(l)} + \sum_{k=1}^K \alpha_k(l; \theta) V_k^{(l)}$, where $\alpha_k(l; \theta)$ is the coefficient for task $k$ at layer $l$.
- **Parameterization Search Spaces:**
  - *Global Task-Wise (GT-Merge):* Constrained constant scaling $\alpha_k(l; \theta) = \alpha_k$, resulting in $K$ parameters.
  - *Polynomial Coefficient Profiles (Poly-Val-Merge):* Merging coefficients modeled as a continuous polynomial of normalized layer depth: $\alpha_k(l; \theta) = \sum_{j=0}^d c_{kj} \left(\frac{l}{L}\right)^j$, yielding $K(d+1)$ parameters.
  - *Unconstrained Layer-Wise Search Space:* Independent scaling per layer per task $\alpha_k(l; \theta) = \alpha_{k, l}$, resulting in $K \times L$ parameters (e.g., 48 parameters for a 12-layer model with 4 tasks).
- **Optimization:** OFS-Tune employs derivative-free optimization (specifically the Nelder-Mead simplex algorithm) to minimize the supervised cross-entropy loss over a tiny validation set $D_{val}$ containing $M$ samples per task. For larger task sizes (up to $K=64$), the paper extends the optimization to a gradient-based approach using PyTorch Adam on the weight-space parameters.

The authors evaluate these methods on both a carefully calibrated continuous model-merging simulation landscape (30 independent random seeds, calibrated on Vision Transformer ViT-B/32 statistics across MNIST, FashionMNIST, CIFAR-10, and SVHN) and a physical 5-layer CNN trained on real MNIST and FashionMNIST datasets.

---

## 3. Key Findings
- **Superiority of Few-Shot Static Tuning:** Under standard, clean, i.i.d. streams, OFS-Tune ($d=1, M=10$) achieves an average accuracy of $85.89\%$, outperforming Task Arithmetic ($84.44\%$) and completely dominating Online AdaMerging ($79.72\%$) and RegCalMerge ($80.70\%$) without running any backpropagation steps at test-time.
- **Robustness Under Adversarial Streams:** Under extreme label shift, bursty task streams (temporal task clustering), and small batch sizes, online TTA methods degrade severely (AdaMerging drops down to $77.99\%$). In contrast, OFS-Tune remains perfectly robust and deterministic, maintaining its high performance of $85.89\%$ with zero test-time compute.
- **The Overfitting-Optimizer Paradox:** When validation data is extremely scarce ($M=5$), unconstrained high-dimensional search spaces (48 parameters) overfit severely to sample noise if optimized perfectly using PyTorch Adam (achieving only $80.78\%$). Conversely, constraining the optimization to low-dimensional polynomial parameterizations (e.g., Poly-Val $d=2$) prevents validation overfitting, yielding a superior accuracy of $87.24\%$.
- **Physical Neural Network Validation:** In physical deep CNN experiments, the paper validates these theoretical hypotheses. Online AdaMerging drops below the uniform baseline (collapsing to $42.94\%$ vs. $55.27\%$). High-capacity adaptation baselines like Few-Shot Head-Tuning ($47.97\%$) and Joint Fine-Tuning ($43.77\%$) overfit and generalize poorly, whereas OFS-Tune Poly-Val generalizes stably ($56.31\%$) and shows exceptional immunity to $30\%$ validation label noise ($56.35\%$).

---

## 4. Explicitly Claimed Contributions and Accompanying Evidence
1. **Critical Demystification of TTA Model Merging:** Exposing the "no-data" strawman and the transductive instability of online entropy minimization.
   - *Evidence:* Extensive simulation results (Tables 1, 2, 4) and physical network results (Table 5, Figure 3), demonstrating that online methods degrade under realistic stream distributions, gradient noise, and rugged entropy landscapes.
2. **Offline Few-Shot Validation Tuning (OFS-Tune):** A training-free baseline that optimizes low-dimensional coefficient profiles on tiny validation sets.
   - *Evidence:* Solid performance across multiple seeds in Tables 1, 2, 4, and 5 showing superior accuracy and zero test-time compute.
3. **The Overfitting-Optimizer Paradox:** Revealing the trade-off between optimization capacity and generalization in scarce data regimes.
   - *Evidence:* Quantitative controls in Table 4, showing that a powerful gradient optimizer (PyTorch Adam) on a 48-D layer-wise space yields severe overfitting ($80.78\%$ for $M=5$), while the low-dimensional Poly-Val space maintains high generalization ($87.24\%$).
4. **Task Scalability and Advanced Robustness Analyses:** Addressing optimization scaling up to $K=64$ tasks and sweeping domain diversity and validation selection bias.
   - *Evidence:* Extensive empirical sensitivity sweeps in Appendix C, D, and E (Figures 1, 2, 4), and mathematical formalization of task interference and validation shift.
