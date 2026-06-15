# Peer Review Report: PAC-Kinetics

## 1. Summary of the Paper
The submission introduces **PAC-Kinetics**, a learning-theoretic and control-theoretic framework designed for test-time stateful model ensembling under sequential, non-i.i.d. query streams. The primary motivation is to resolve the **routing jitter paradox**—where stateless dynamic routers fluctuate rapidly between expert models (like LoRA adapters) in response to query-level noise, leading to representation instability and downstream "cascading representation collapse" in deep networks. 

To address this, the authors model ensembling concentration dynamics as a continuous-time non-equilibrium chemical kinetics system, deriving a discrete-time linear contractive state space recurrence:
$$s_t = \mathbf{A} s_{t-1} + W \mathbf{e}_t$$
This recurrence acts as a robust low-pass filter over early-layer coordinate representations. To manage the stateful-stateless accuracy-stability Pareto frontier, they introduce **Adaptive Online Kinetics**, dynamically scaling state-retention parameters $a_{k, t}$ using a local coordinate cosine similarity metric ($Sim_t$). 

To provide rigorous generalization guarantees under temporally dependent, streaming conditions, the authors assume a stationary $\beta$-mixing stochastic process and derive a novel parameter-space Catoni-type PAC-Bayesian generalization bound. They utilize the **Even/Odd Block Splitting** technique to avoid the exploding Total Variation (TV) penalty that typically arises when coupling dependent sequences inside unbounded exponential moments. The parameters are optimized by directly minimizing this PAC-Bayesian complexity bound on a short calibration stream.

The framework is evaluated across orthogonal and overlapping manifold configurations in an Analytical Coordinates Sandbox (ICS) and validated physically using a real 3-layer PyTorch MLP on MNIST and Fashion-MNIST datasets. The results demonstrate that PAC-Kinetics reduces routing jitter by over **11.2$\times$** (sandbox) and **2.59$\times$** (physical) compared to stateless baselines while matching or exceeding Oracle-level joint accuracy and outperforming static ensembling and heuristic stateful controllers (ChemMerge).

---

## 2. Strengths of the Paper
* **Exceptional Mathematical and Control-Theoretic Rigor:** The mathematical formulation is incredibly thorough. Proving Global Asymptotic Stability (GAS) and Input-to-State Stability (ISS) using quadratic Lyapunov candidate functions for both the autonomous recurrence and the time-varying Adaptive Online Kinetics system provides exceptional guarantees of mathematical safety, ensuring the router is immune to chaotic divergence or representation explosions.
* **Rigorous Learning-Theoretic Bound for Dependent Data:** The derivation of a parameter-space Catoni-type PAC-Bayesian bound under $\beta$-mixing sequences is highly elegant. By employing the **Even/Odd Block Splitting** technique, the authors solve a severe theoretical vulnerability (the exploding TV penalty of direct coupling in exponential moments), keeping the bound mathematically tight and meaningful.
* **Innovative Resolution of the Stateful-Stateless Trade-Off:** The **Adaptive Online Kinetics** mechanism is a brilliantly simple, computationally efficient, and fully differentiable feature. It dynamically turns off state-retention during rapid switches ($Sim_t \approx 0$) to suppress "inertial drag" (routing lag) while maintaining full noise-filtering capacity ($Sim_t \approx 1$) under stable streams.
* **Substantial Simulation-to-Physical Validation:** Unlike many theoretical serving works, the authors validate their method on real PyTorch networks and physical MNIST/Fashion-MNIST datasets, incorporating task-specific classification heads to successfully resolve the "Uniform Merging Paradox." The physical results show a massive **+21.50% absolute accuracy improvement** over static Uniform Merging and a **2.59$\times$ routing jitter reduction** over stateless PAC-ZCA.
* **Exhaustive Sensitivity Sweeps and Systems Profiling:** The submission is highly complete, presenting comprehensive hyperparameter sweeps over prior variance ($\sigma_0^2$), calibration sequence length ($T$), and expert fleet sizes up to $K=16$ (reporting spectral condition numbers to confirm optimization conditioning). Systems-level profiling confirms that PAC-Kinetics executes with negligible wall-clock latency ($\approx 10.4 \mu s$ on CPU, and $<3.35 \mu s$ on GPU under concurrent batching), proving its high compatibility with production serving managers like S-LoRA or Punica.
* **Outstanding Scientific Honesty and Transparency:** The authors transparently document and analyze the performance gap of the randomized router ($R(Q)$) under large prior variances, conducting a detailed follow-up ablation sweep over smaller perturbation variances ($\sigma_{\text{pert}}^2 \le 0.01$) to successfully bridge this gap and validate the served deterministic surrogate ($\Theta_{\text{opt}}$). They also openly discuss and frame the performance of stateful serving under completely uncorrelated streams.

---

## 3. Weaknesses and Detailed Scholarly Criticisms
While the paper is technically stellar, a rigorous scholarly evaluation reveals several critical areas where the historical context, related work, and literature attribution must be expanded and corrected to properly position the contribution:

### A. Contextualization with the Online Learning Literature
The submission claims to present "the first learning-theoretic framework for stateful, sequential model ensembling." This claim is too broad and ignores a massive, foundational body of literature in machine learning: **Online Learning and Prediction with Expert Advice**.
* Stateful, sequential ensembling of experts has been studied extensively for decades. Algorithms such as the **Hedge Algorithm** / **Multiplicative Weights Update (MWU)** (Freund & Schapire 1997, Littlestone & Warmuth 1994) and **Online Convex Optimization** (Zinkevich 2003, Hazan 2016) maintain running history states of expert performances and update them recursively to blend predictions.
* These classical algorithms provide strict, non-asymptotic *regret bounds* under arbitrary, adversarial (non-i.i.d.) streams without requiring any stationarity or mixing assumptions.
* **Requirement for Revision:** The authors must explicitly acknowledge and discuss the Online Learning / Expert Advice literature in Section 2 (Related Work). They should contrast PAC-Kinetics' parameter-space expected risk optimization under mixing processes with classical regret minimization (such as Follow the Regularized Leader or Online Gradient Descent) to provide a nuanced historical context.

### B. Connection to State-Space Models (SSMs) in Deep Learning
The stateful recurrence $s_t = \mathbf{A} s_{t-1} + W \mathbf{e}_t$ is mathematically a multi-dimensional linear State-Space Model (SSM).
* There is a prominent, modern line of research modeling sequential representations in deep learning utilizing linear SSMs, most notably **Structured State Spaces (S4)** (Gu et al. 2021) and **Mamba** (Gu & Dao 2023). 
* Furthermore, modeling neural network modules or controllers as continuous-time dynamical systems (e.g., Neural ODEs, Chen et al. 2018) is a highly related paradigm.
* **Requirement for Revision:** The related work section must be updated to explicitly connect PAC-Kinetics' routing recurrence to the broader literature on linear SSM backbones in deep learning. This will help readers recognize that the routing state functions as a lightweight, localized SSM.

### C. Attribution of Classic Control-Theoretic Concepts
In Section 3.7, the proofs of Global Asymptotic Stability (GAS) and Input-to-State Stability (ISS) are mathematically correct and highly rigorous. However, they represent direct applications of classical linear systems theory and Lyapunov stability analysis, which are presented without any literature citations.
* **Requirement for Revision:** The authors must explicitly cite foundational control theory reference textbooks for these concepts. Specifically, they should cite **Lyapunov's Direct Method / Discrete-Time Lyapunov Equations** (e.g., Khalil's *Nonlinear Systems*, 2002) and Sontag's foundational formulation of **Input-to-State Stability (ISS)** (Sontag's *Input-to-state stability: Basic concepts and results*, 1989).

### D. Scalability Limits of Physical Validation
While the CPU/GPU wall-clock latency profiling demonstrating microsecond-scale execution under large batches is highly compelling, the physical validation on real data is currently limited to a shallow, 3-layer MLP on image data.
* Proving that PAC-Kinetics prevents the "cascading representation collapse" in massive Transformer-based models (such as LLMs or deep Vision Transformers with dozens of layers) under larger physical expert fleets remains a critical open systems-level challenge.
* **Requirement for Revision:** The authors should explicitly frame this simulation-to-physical gap in Section 5.1 as an open systems-level research challenge, acknowledging that physical validation on massive generative transformers remains an active area of future work.

---

## 4. Questions and Suggestions for the Authors
1. **Online Regret vs. Generalization:** How does the PAC-Bayesian out-of-sample risk bound on $\beta$-mixing streams compare conceptually to the regret bounds of classical online learning algorithms (like Hedge or Follow the Regularized Leader) under adversarial streams? Can your stateful recurrence be analyzed through the lens of Online Convex Optimization?
2. **Spectral Norm Regularization:** In your expert fleet scaling sweep (Section 6.2.4), you report the spectral condition number of the learned matrix $W$. Have you considered adding explicit spectral norm regularization (or spectral normalization) during calibration to further constrain the condition number and guarantee contractivity when scaling to $K \ge 32$ experts?
3. **SSM Connection:** Given that the state recurrence is a linear SSM, could the routing state $s_t$ be dynamically updated using a selective mechanism similar to Mamba (making the matrices $A$ and $W$ functions of the input $\mathbf{e}_t$ rather than static learned parameters), while still preserving Lyapunov stability guarantees?
4. **Control References:** Please ensure that the stability proofs in Section 3.7 include proper citations to foundational control literature (e.g., Khalil 2002, Sontag 1989) for the definitions of GAS and ISS.

---

## 5. Evaluation Ratings

* **Soundness:** **Excellent**
  * The proofs for Theorem 3.1, Lemma 3.5, and Lemma 3.6 are mathematically complete, correct, and highly rigorous. The use of Even/Odd Block Splitting is technically flawless. The empirical support is thorough, and the transparent handling of the deterministic surrogate gap is highly commendable.
* **Presentation:** **Excellent**
  * The manuscript is exceptionally well-written, clearly structured, and easy to follow. Visual aids (Figure 3) and summarizing notation tables (Table 1) make the highly technical, multi-disciplinary content highly accessible.
* **Significance:** **Excellent**
  * The work addresses a highly relevant systems bottleneck in PEFT/LoRA serving. By providing a mathematically sound and stable stateful router with flat, microsecond-level latency, this framework has the potential to influence the development of active multi-tenant serving frameworks (e.g., S-LoRA, Punica).
* **Originality:** **Excellent**
  * The paper presents a highly original and creative combination of non-equilibrium chemical kinetics, linear state-space control theory, and PAC-Bayesian theory under mixing processes, representing an outstanding interdisciplinary contribution.

---

## 6. Overall Recommendation
* **Recommendation:** **5: Accept**
  * This is an exceptionally strong, mathematically rigorous, and empirically thorough paper that addresses a critical systems bottleneck in sequential dynamic model ensembling. The control-theoretic stability proofs (GAS and ISS) and the PAC-Bayesian bound on $\beta$-mixing streams are technically sound. The addition of physical validation on real networks and image datasets, along with detailed systems profiling, makes this a complete and high-impact submission.
  * The minor weaknesses identified—such as the need for expanded literature contextualization with Online Learning (Hedge/MWU) and deep State-Space Models (S4/Mamba), and proper control theory citations—can be easily addressed in a minor revision. Once these scholarly connections are integrated, the paper will be an outstanding contribution to the conference.
