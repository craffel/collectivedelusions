# 1. Summary of the Paper

This document provides a comprehensive and structured summary of **ThermoMerge (Thermodynamic Model Merging)**, as submitted and compiled in the `submission/` directory.

## 1.1. Core Problem and Context
Model merging is a cost-effective paradigm for fusing task-specific expert weights (sharing a common pre-trained ancestor) into a unified multi-task network without additional training or access to the original training data. However, standard methods suffer from severe limitations:
1. **Parameter-Space System Frustration:** Early static methods (e.g., Task Arithmetic, Model Soups, TIES-Merging) assume a flat Euclidean landscape and perform straight-line parameter interpolation. Since neural loss landscapes are highly non-convex, forcing a linear path across disjoint basins of attraction causes severe representation interference and performance degradation, which the authors characterize as *system frustration*.
2. **The Overfitting-Optimizer Paradox:** Unsupervised test-time adaptation (TTA) methods (e.g., AdaMerging) attempt to dynamically optimize merging coefficients on unlabeled target calibration streams. However, unregularized entropy minimization drives parameters into degenerate local minima with low entropy but zero generalizing representations, causing catastrophic representation collapse.
3. **The Gray-to-Color Bottleneck:** When adapting a shared backbone across heterogeneous grayscale (MNIST, FashionMNIST) and color (CIFAR-10, SVHN) streams, dominant grayscale gradients during unsupervised TTA overwrite fragile color features, causing catastrophic collapse on color tasks under from-scratch training.

---

## 1.2. Proposed Solution: ThermoMerge
The paper proposes **ThermoMerge (Thermodynamic Model Merging)**, a framework that reformulates model merging through the lens of statistical mechanics and thermodynamics. Instead of zero-temperature parameter averaging, ThermoMerge thermalizes model outputs and treats test-time adaptation as a dynamic thermodynamic crystallization process. Its core components are:

### 1.2.1. Canonical Ensemble Mapping
The framework maps downstream expert outputs to microstates in a physical canonical Boltzmann ensemble.
- Task classification logits function as negative microstate energies: $E_c \equiv -f_c(x; \theta)$.
- The canonical probability distribution for class $c$ under expert $k$ at temperature $T$ is modeled as a Boltzmann distribution:
  $$p_c^{(k)}(x; T) = \frac{\exp\left( \frac{f_c(x; \theta_k)}{T} \right)}{Z_k(x; T)}$$
  where $Z_k$ is the canonical partition function.
- The Helmholtz Free Energy is defined as $F_k(x; T) = -T \ln Z_k(x; T)$.

### 1.2.2. Helmholtz Free Energy Discrepancy (F-Min) Minimization
To adapt the layer-wise merging coefficients $\boldsymbol{\Lambda}$ without labels, ThermoMerge minimizes a physically grounded objective, the **Helmholtz Free Energy Discrepancy (F-Min)** objective, defined as:
$$\mathcal{L}(\boldsymbol{\Lambda}, T) = \sum_{k=1}^K \mathbb{E}_{x \in \mathcal{X}_k^{te}} \left[ T \cdot \mathcal{D}_{KL} \left( p^{(k)}(x; T) \parallel p^{(MTL, k)}(x; T) \right) \right]$$
The authors mathematically prove that this objective represents the gap between the *variational free energy* of the expert distribution and the *equilibrium free energy* of the merged multi-task model. It naturally decomposes into:
1. **Expected Energy Difference:** The average energy discrepancy between expert and merged microstates.
2. **Helmholtz Free Energy Difference:** The discrepancy between expert and merged global partition functions.

### 1.2.3. Thermodynamic Annealing Schedule (TAS)
To escape non-convex local minima and navigate over high-loss ridges, ThermoMerge employs simulated physical cooling during TTA. The global temperature decays exponentially:
$$T(t) = T_{end} + (T_{start} - T_{end}) \cdot \exp(-\beta \cdot t)$$
At the optimal starting temperature $T_{start}=2.0$, the free energy landscape is flattened, enabling parameters to slide across disjoint basins. As the system cools to $T_{end}=1.0$ (using a fast quenching cooling rate of $\beta=0.40$ over 50 steps), the multi-task representations crystallize around specialized decision boundaries.

### 1.2.4. Task-wise Thermal Coupling
To handle varying task difficulties, each task is modeled as a subsystem with its own trainable local thermal capacity $\tau_k \in [0.2, 5.0]$ in output logit space, defining task-wise local temperatures:
$$T_k(t) = \tau_k \cdot T(t)$$

---

## 1.3. Experimental Setup and Key Findings
- **Backbone:** ImageNet pre-trained **ResNet-18** with frozen early blocks; only `layer4` and classification heads are trainable/merged.
- **Datasets:** MNIST, FashionMNIST, CIFAR-10, SVHN (Grayscale and Color heterogeneous multi-task benchmark).
- **TTA Setting:** A true sequential streaming protocol (50 steps in code, 100 steps in table description, batch size 128 from each dataset).
- **Core Findings:**
  1. **Outstanding Performance:** ThermoMerge achieves an outstanding multi-task average accuracy of **29.05%**, outperforming Model Soups (27.25%), Task Arithmetic (27.25%), TIES-Merging (26.60%), AdaMerging (26.10%), and the SOTA SyMerge baseline (27.90%).
  2. **Resolving the Gray-to-Color Collapse:** While adaptive merging trained from scratch collapsed on color datasets (SimpleCNN CIFAR-10 accuracy drops to ~10%), pre-trained ancestral connectivity provides robust linear mode connectivity that shields fragile color representations (achieving 33.00% on CIFAR-10 and 30.60% on SVHN).
  3. **Mitigating the Overfitting-Optimizer Paradox:** F-Min minimization acts as a physical regularizer, preventing the complete representation collapse experienced by unregularized AdaMerging (which drops to 16.85% under SimpleCNN and 26.10% under ResNet-18).
  4. **Insights into Grayscale Degradation:** The authors honestly discuss that unsupervised TTA slightly degrades performance on MNIST (20.00% vs 21.40% static) and FashionMNIST (32.60% vs 35.40% static) due to representational drift in the early shared convolutional layers, prioritizing the alignment of complex color texture features.
