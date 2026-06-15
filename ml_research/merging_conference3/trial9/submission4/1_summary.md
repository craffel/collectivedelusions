# Summary of Paper Claims, Methodology, and Contributions

## 1. Overview
The paper **"Momentum-Merge: Deconstructing Biochemical Complexity in Dynamic Model Merging"** focuses on the challenge of serving multiple specialized Parameter-Efficient Fine-Tuning (PEFT) expert adapters (specifically LoRA modules) simultaneously on a heterogeneous, sample-by-sample serving stream where task labels are unavailable. 

Specifically, the paper identifies and addresses **routing jitter**—high-frequency layer-to-layer ensembling weight oscillations caused by representational noise and cascading non-linearities in deep architectures. This jitter causes the network to blend incompatible expert projections in consecutive layers, initiating a cascade of representational drift that severely degrades multi-task accuracy.

---

## 2. Core Thesis
The paper critiques the state-of-the-art stateful routing framework, **ChemMerge**, through the lens of **Occam's razor (conceptual parsimony)**. ChemMerge models ensembling weights as chemical concentrations inside a continuous reactor, governed by non-equilibrium biochemical kinetics, Arrhenius reaction rates, and continuous-time Ordinary Differential Equations (ODEs) integrated via numerical solvers. 

The core thesis of this paper is that **ChemMerge's continuous biochemical metaphor is redundant and can be simplified down to standard, classical mathematical operators**. The authors prove that under standard discretization, ChemMerge's continuous rate equations are mathematically equivalent to a simple Exponential Moving Average (EMA).

---

## 3. Proposed Methodology
Guided by this minimalist perspective, the authors propose **Momentum-Merge**, a training-free, single-parameter ensembling framework that stabilizes routing trajectories using a constant EMA update on ensembling weights across network depth.

### 3.1 Mathematical Formulation
At each adapted layer $l$, the similarity-routing weights $w_k^{(l)}$ are computed using nearest-centroid matching in activation space (Unit-Norm Calibration). Instead of applying these weights directly (stateless routing), Momentum-Merge recursively updates the ensembling coefficients $\alpha_k^{(l)}$ as:
$$\alpha_k^{(l)} = (1 - \beta) w_k^{(l)} + \beta \alpha_k^{(l-1)}$$
where $\beta \in [0, 1]$ is the constant momentum coefficient representing routing inertia, and the recurrence is initialized with a uniform boundary condition:
$$\alpha_k^{(L_{\text{frozen}})} = \frac{1}{K}$$

### 3.2 Key Variants
1. **Momentum-Merge (Base):** Uses the uniform boundary condition and measures cosine similarity to global task centroids pre-computed offline at an early frozen layer.
2. **Momentum-Merge (Advanced):** Features two additional enhancements to improve accuracy and trajectory stability:
   - **Layer-wise Centroid Calibration:** Calibrates and anchors task centroids layer-by-layer ($\mu_k^{(l)}$) across network depth to account for representational transformations.
   - **Raw Boundary Initialization:** Avoids artificial early-layer damping by initializing the recurrence with the raw similarity weight of the first adapted layer:
     $$\alpha_k^{(L_{\text{frozen}})} = w_k^{(L_{\text{frozen}}+1)}$$

---

## 4. Key Empirical Claims & Findings
The authors evaluate Momentum-Merge inside the synthetic **Analytical Coordinate Sandbox (ICS)** on a 1000-sample heterogeneous streaming protocol across 10 independent random seeds.
* **Accuracy Claims:** Basic Momentum-Merge is claimed to achieve **74.85%** accuracy, while the advanced Momentum-Merge variant reaches **74.98%** joint classification accuracy.
* **Jitter Reduction:** Advanced Momentum-Merge drops layer-to-layer routing jitter to **0.000374**, which represents an outstanding reduction in routing oscillations (over 41$\times$ lower than ChemMerge's jitter of 0.015339 and 195$\times$ lower than SABLE's jitter of 0.073198).
* **Interpretability:** The authors sweep $\beta \in [0, 1]$, demonstrating how the momentum parameter controls routing dynamics, smoothly interpolating between stateless routing ($\beta = 0.0$) and static uniform merging ($\beta = 1.0$), peaking at $\beta = 0.60$ in their 5-seed sweep.
