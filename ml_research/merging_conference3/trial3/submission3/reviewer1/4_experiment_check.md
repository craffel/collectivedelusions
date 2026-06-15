# Evaluation Task 4: Experimental Evaluation and Claims Check

## 1. Experimental Setup and Baselines
* **Calibrated Simulation Sandbox:** The authors evaluate FlatMerge using a stylized, calibrated continuous simulation sandbox designed to emulate the loss landscape and functional task couplings of a 12-layer CLIP Vision Transformer (ViT-B/32) across a 4-task classification benchmark (MNIST, FashionMNIST, CIFAR-10, SVHN).
* **Statistical Rigor:** To ensure high statistical validity, the simulated evaluations are conducted across **15 independent random seeds** (seeds 42 to 56 inclusive), which is highly commendable and provides solid confidence in the reported standard deviations.
* **Comprehensive Baselines:** The authors compare FlatMerge against a wide range of relevant baselines: Task Arithmetic (static baseline), unconstrained first-order AdaMerging, AdaMerging with $L_2$ and TV regularizers, RegCalMerge, and PolyMerge ($d \le 3$).
* **Physical Validation:** To ground the simulation, the authors conduct physical experiments on a CPU, merging real MLP experts (on MNIST and FashionMNIST) and 5-layer CNN experts (on MNIST, FashionMNIST, and KMNIST).

---

## 2. Critical Evaluation: Claims vs. Results

While the empirical results are extensive, a rigorous comparison of the claims against the actual numbers in the tables reveals several highly critical limitations and gaps:

### A. The Optimization Utility Paradox on Real Models (The "Static Beats Adaptive" Elephant)
The authors claim that FlatMerge is a highly practical and robust solution for edge-deployment test-time model merging. However, looking closely at the real-world validation tables, we observe a major empirical concern: **In physical models, unsupervised test-time optimization (both AdaMerging and FlatMerge) consistently performs worse than the simple static, uniform Task Arithmetic baseline across almost all clean and moderate noise conditions**:

#### 1. Physical 5-layer CNN Validation (Table 4)
* **Clean ($\gamma=0.0$):** Static Task Arithmetic achieves **$58.20\%$** joint average accuracy. Our proposed FlatMerge achieves **$48.57\%$** (a **$9.63\%$ absolute degradation**). First-order AdaMerging collapses completely to $16.67\%$.
* **Moderate Noise ($\gamma=1.0$):** Static Task Arithmetic achieves **$40.67\%$** joint average accuracy. FlatMerge achieves **$29.20\%$** (an **$11.47\%$ absolute degradation**).
* **Heavy Noise ($\gamma=2.0$):** Static Task Arithmetic achieves **$24.60\%$** joint average accuracy. FlatMerge achieves **$19.77\%$** (a **$4.83\%$ absolute degradation**).
* **Extreme Noise ($\gamma=3.0$):** Static Task Arithmetic achieves **$17.77\%$** joint average accuracy. FlatMerge achieves **$16.07\%$** (a **$1.70\%$ absolute degradation**).

#### 2. Physical 3-layer MLP Validation (Table 3)
* **Clean ($\gamma=0.0$):** Static Task Arithmetic achieves **$70.75\%$** joint average accuracy. FlatMerge achieves **$54.71\%$** (a **$16.04\%$ absolute degradation**).
* **Moderate Noise ($\gamma=1.0$):** Static Task Arithmetic achieves **$63.74\%$** joint average accuracy. FlatMerge achieves **$49.11\%$** (a **$14.63\%$ absolute degradation**).
* **Heavy Noise ($\gamma=2.0$):** Static Task Arithmetic achieves **$48.91\%$** joint average accuracy. FlatMerge achieves **$48.88\%$** (nearly identical).
* **Extreme Noise ($\gamma=3.0$):** Static Task Arithmetic achieves **$37.17\%$** joint average. FlatMerge achieves **$41.35\%$** (a **$+4.18\%$ absolute improvement**).

#### Critical Summary:
While FlatMerge successfully avoids the catastrophic "constant-prediction collapse" of unconstrained first-order TTA (AdaMerging and PolyMerge) on physical weights, **it struggles to outperform the zero-overhead, zero-compute static Task Arithmetic baseline under realistic noise scales**. Only under *extreme* noise on the MLP ($\gamma=3.0$) does FlatMerge provide a benefit over Task Arithmetic. 

This raises a serious question about the practical utility of applying on-device coefficient adaptation at all for physical deployments: **If a simple, training-free static uniform merge consistently outperforms adaptive test-time optimization by 5% to 16% on physical models, why should practitioners deploy a complex, latency-heavy zeroth-order optimization loop on edge devices?** The authors should address this critical point directly.

---

### B. Simulation-to-Real Gap
There is an undeniable performance and behavior gap between the simulated Vision Transformer experiments and the physical validations:
* In the simulation (Tables 1 and 2), FlatMerge and other optimization-based methods easily outperform Task Arithmetic on clean and moderate noise (e.g., $85.59\%$ vs. $84.44\%$ at $\gamma=1.5$).
* In the physical experiments, Task Arithmetic is the dominant method, and optimization-based TTA fails to match it.
* This indicates that the continuous simulation landscapes (Model I and Model II) are highly stylized, decoupled from actual weight manifolds, and over-estimate the benefits of unsupervised test-time optimization. While the authors discuss this gap in Section 4.2 by pointing to model capacity and pre-training scales, it remains a major limitation of the paper's empirical claims.

---

### C. Strength: Excellent Zeroth-Order Budget ($B_{\text{zo}}$) Ablation
A major empirical strength of the paper is the ablation study of the perturbation sample size budget $B_{\text{zo}}$ (Figure 7).
* The authors show that FlatMerge is exceptionally sample-efficient: decreasing $B_{\text{zo}}$ from 10 to 4 yields almost identical joint accuracy ($85.70\% \pm 1.15\%$ vs $85.89\% \pm 1.31\%$) while providing a massive **$59\%$ step-latency speedup** ($2.38$ ms/step vs. $5.76$ ms/step).
* This provides highly valuable engineering insight, proving that FlatMerge's search space is highly stable and can function with highly noisy zeroth-order gradient estimates on extreme low-compute hardware budgets.
