# Peer Review of Conference Submission: ThermoMerge (Thermodynamic Model Merging)

---

## 1. Summary of the Paper
This paper introduces **ThermoMerge** (Thermodynamic Model Merging), a framework that reformulates unsupervised test-time adaptive model merging through the lens of statistical mechanics and thermodynamics. 

To merge task-specific expert neural networks sharing a pre-trained ancestor without original training data, the authors map classification logits to negative energies in a canonical Boltzmann ensemble. Under this mapping, model outputs are modeled as thermal probability distributions (isomorphic to temperature-scaled softmax outputs). 

The layer-wise merging coefficients are optimized on streaming unlabeled calibration data by minimizing a proposed **Helmholtz Free Energy Discrepancy (F-Min)** objective, which is shown to be mathematically equivalent to the temperature-scaled Kullback-Leibler (KL) divergence between expert predictions and merged model outputs. 

To navigate rugged, non-convex loss landscapes, the paper introduces a **Thermodynamic Annealing Schedule (TAS)** that decays the global temperature parameter from a high initial value to $1.0$. Furthermore, the framework optimizes task-specific temperatures using trainable local capacities $\tau_k \in [0.2, 5.0]$ (Task-wise Thermal Coupling). 

Empirically, under a sequential streaming test-time adaptation setting on a pre-trained ResNet-18 backbone, the authors report that ThermoMerge achieves an average multi-task accuracy of **29.05%** across MNIST, FashionMNIST, CIFAR-10, and SVHN, outperforming static baselines (Model Soups, Task Arithmetic, TIES-Merging) and adaptive baselines (AdaMerging, SyMerge).

---

## 2. Main Strengths

1. **Exceptional Clarity and Structure:** The paper is written with high structural and rhetorical polish. The authors present their ideas in a highly engaging, eloquent narrative that connects physical thermodynamic concepts to deep learning parameters seamlessly.
2. **Beautiful Mathematical Presentation:** The algebraic formulations and step-by-step mathematical proofs (specifically the link between Free Energy and the KL divergence in Section 3.3 and Appendix A) are written with outstanding clarity and precision.
3. **Intuitive Optimization Heuristic:** Applying a temperature decay schedule (simulated annealing) to smooth out non-convex barriers during test-time adaptive optimization is an intuitive, well-motivated, and elegant optimization heuristic.
4. **Detailed and Transparent Disclosures:** The appendix contains comprehensive details regarding model architectures (Table 2), training and adaptation hyperparameters (Table 3), optimization trajectories, and hyperparameter sensitivity analyses, demonstrating a strong commitment to academic transparency.

---

## 3. Main Weaknesses

### A. Rebranding of Established ML Concepts ("Physics-Washing")
While the thermodynamic vocabulary ("system frustration," "quenched equilibrium," "Helmholtz Free Energy Discrepancy") is ambitious, the actual mathematical and algorithmic contribution is highly incremental. The framework represents a direct relabeling of standard deep learning concepts:
* **Microstate Energy ($E_c \equiv -f_c(x)$):** The negative of classification logits, which is standard in Energy-Based Models (LeCun et al., 2006).
* **Canonical Boltzmann Ensemble:** Mathematically identical to the standard Softmax function with temperature scaling, used widely since the 1980s.
* **F-Min Objective:** The objective minimized (Equation 16) is exactly the temperature-scaled KL divergence between expert predictions (teachers) and merged model predictions (student). The first-principles derivation in Section 3.3 is a textbook algebraic identity of the KL divergence when evaluated over Gibbs distributions.
* **Task-wise Thermal Coupling:** Optimizing trainable task-wise temperatures $\tau_k$ is a minor variation of Platt/temperature scaling (Guo et al., 2017) applied during optimization.

Ultimately, the actual algorithmic "delta" from existing test-time adaptive merging methods (e.g., AdaMerging, SyMerge) is extremely narrow: adding temperature decay (simulated annealing) and trainable task temperatures during the 100 adaptation steps.

### B. Trivial Global Minimization Flaw in Temperature Optimization
In Section 3.5, the task-wise capacities $\tau_k$ scale the temperature: $T_k(t) = \tau_k \cdot T(t)$. These are optimized alongside merging coefficients by minimizing the F-Min objective:
$$\mathcal{L}(\boldsymbol{\Lambda}, \boldsymbol{\tau}) = \sum_{k=1}^K \mathbb{E} [ T_k(t) \cdot \mathcal{D}_{KL}(p^{(k)}(x; T_k(t)) \parallel p^{(MTL, k)}(x; T_k(t))) ]$$

This formulation contains a severe structural defect: the objective has a **trivial global minimum** as $T_k \to \infty$ (and thus $\tau_k \to \infty$). 
* As temperature increases, the Boltzmann distributions $p^{(k)}$ and $p^{(MTL, k)}$ flatten and approach the uniform distribution.
* The KL divergence between uniform distributions is exactly zero ($\lim_{T \to \infty} \mathcal{D}_{KL} = 0$). 
* Even when scaled by $T_k(t)$, the loss term $T_k \cdot \mathcal{D}_{KL}$ vanishes as $\mathcal{O}(1/T_k)$ at high temperatures.

As a result, gradient descent will pathologically push the trainable parameter $\tau_k$ toward its maximum allowed limit to trivially minimize the loss. While the authors clamp $\tau_k \in [0.2, 5.0]$, this means $\tau_k$ does not find a meaningful "equilibrium" or "task capacity"; it simply hits the arbitrary upper constraint boundary of 5.0 because of a structural pathology in the objective. Furthermore, because temperature scaling is positive, $\tau_k$ is mathematically invariant during evaluation and is completely discarded during inference, rendering the learned "crystallized temperature" functionally irrelevant.

### C. Evaluation Scope is Restricted to Toy-Scale convolutional Models
Modern model merging is almost exclusively applied to foundation models (e.g., CLIP, ViT, LLaMA, Mistral) because their overparameterization allows different task representations to merge cleanly. 
In contrast, this paper restricts its entire empirical validation to a compact **ResNet-18 backbone** on toy-scale datasets (**MNIST, FashionMNIST, CIFAR-10, SVHN**). 
Low-capacity convolutional networks have extremely limited representation space in their deeper layers, causing massive destructive interference. This is why the absolute accuracies reported are catastrophically low (averaging **29.05%** overall, and achieving only **20.00% on MNIST** and **32.60% on FashionMNIST**). An average accuracy of 29% on simple visual classification tasks represents near-total representation collapse. A 1.8% average improvement over Task Arithmetic (27.25%) is practically non-functional when the resulting model is completely unusable. While the appendix includes a "Roadmap" to scale to foundation models, no actual foundation model experiments are run.

### D. Glaring Hyperparameter and Procedural Contradictions
There are multiple blatant discrepancies between the main text and the appendix, which severely hurts the paper's scientific rigor and reproducibility:
1. **Starting Temperature ($T_{start}$):** The Abstract and Section 3.4 state that the TAS starts at $T_{start} = 5.0$. However, Appendix Table 3 and Appendix Section 4.5.1 state that $T_{start} = 2.0$.
2. **Cooling Rate ($\beta$):** Section 3.4 states that the cooling rate is $\beta = 0.05$. However, Appendix Table 3 and Appendix Section 4.5.2 report that $\beta = 0.40$.
3. **Optimization Steps:** Section 4.1.4 states that adaptation runs for 100 steps. However, Appendix Table 3 states that adaptation runs for "100 steps (50 steps for ThermoMerge)." 

These contradictions make direct replication of the reported results impossible.

---

## 4. Specific Ratings

* **Soundness:** **Fair** (Due to the mathematical trivial-minimization pathology of $\tau_k$, non-functional absolute performance, and hyperparameter contradictions).
* **Presentation:** **Good** (Exceptionally clear and engaging narrative, but undermined by blatant discrepancies between main text and appendix).
* **Significance:** **Fair** (Restricted entirely to toy models/datasets with non-functional absolute accuracies; lacks foundation model validation).
* **Originality:** **Fair** (Algorithmic delta from prior work is highly incremental; the thermodynamic framing is primarily a rebranding of temperature-scaled KL divergence and simulated annealing).

---

## 5. Overall Recommendation

**Rating:** **3: Weak reject**

**Justification:** 
The paper is exceptionally well-written, mathematically elegant, and explores an interesting connection between simulated thermal cooling and test-time adaptive model merging. However, from a core conceptual and empirical perspective, the contribution falls short of the bar for acceptance:
1. The mathematical framing is largely "physics-washing" of standard, well-known deep learning components (temperature-scaled KL divergence, simulated annealing). The algorithmic novelty compared to prior work (SyMerge/AdaMerging) is highly incremental.
2. The joint optimization of task-specific temperature parameters contains a critical structural pathology (trivial global minimization at $T \to \infty$) causing them to hit the arbitrary clamping wall of 5.0.
3. The empirical evaluation is restricted to a toy-scale convolutional backbone (ResNet-18) on simple datasets, yielding catastrophically low, non-functional absolute accuracies (29% average, 20% on MNIST). 
4. Blatant hyperparameter discrepancies between the main text and the appendix prevent replication.

---

## 6. Questions and Constructive Feedback for the Authors

1. **Resolution of Hyperparameter Discrepancies:** Please clarify the exact hyperparameter configuration used to obtain the results in Table 1. Was $T_{start}$ set to 5.0 or 2.0? Was $\beta$ set to 0.05 or 0.40? Was ThermoMerge run for 50 steps or 100 steps?
2. **Addressing the Trivial $T_k \to \infty$ Minimization:** How do you address the fact that the F-Min loss is minimized by simply driving $T_k$ to infinity? Did you observe $\tau_k$ hitting the upper bound of 5.0 during optimization? To make this parameter meaningful, have you considered decoupling the temperature scaling of the distributions from the outer multiplier, or adding a log-penalty regularizer on $T_k$ (such as the Kullback-Leibler divergence's natural entropy bounds)?
3. **Scale to Foundation Models:** Why was the evaluation restricted to ResNet-18 on simple datasets? Given that modern model merging is designed for foundation models (such as CLIP, ViT, or llama-7b), demonstrating ThermoMerge on at least one CLIP-based multi-task classification benchmark is essential to prove the practical utility and scalability of your framework.
4. **Baseline Hyperparameter Tuning:** Were the learning rates and step sizes of the baselines (especially AdaMerging and SyMerge) individually optimized for the ResNet-18 setting? Unregularized entropy minimization is highly sensitive to learning rates, and using a uniform configuration may have unfairly crippled their performance.
