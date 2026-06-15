# Summary of the Submission

## Main Topic
The paper addresses the challenge of post-hoc **model merging**, which aims to fuse multiple task-specific expert neural network weights (sharing a common pre-trained ancestor) into a single unified multi-task network without joint training or access to the original training datasets. Specifically, it focuses on unsupervised **test-time adaptation (TTA)** to dynamically optimize layer-wise merging coefficients on unlabeled calibration data streams.

## Proposed Approach: ThermoMerge
The paper introduces **ThermoMerge (Thermodynamic Model Merging)**, which reformulates model merging using concepts from statistical mechanics and thermodynamics. The key components are:
1. **Canonical Ensemble Mapping:** It maps task expert prediction logits to negative microstate energies ($E_c \equiv -f_c(x; \theta_k)$) within a finite-temperature canonical Boltzmann ensemble (effectively a temperature-scaled softmax distribution).
2. **Thermodynamic Annealing Schedule (TAS):** It applies a simulated cooling schedule ($T_{start} = 5.0 \to T_{end} = 1.0$) during test-time adaptation to flatten non-convex optimization barriers and allow merging coefficients to escape local minima.
3. **Helmholtz Free Energy Discrepancy Minimization (F-Min):** It defines a loss objective based on the temperature-scaled Kullback-Leibler (KL) divergence between the expert distributions and the merged model's distribution.
4. **Task-wise Thermal Coupling:** It introduces trainable, task-specific local temperatures (parameterized as trainable thermal capacities $\tau_k \in [0.2, 5.0]$) to handle varying task complexities.

## Key Findings
* Under a sequential streaming test-time adaptation setting on a pre-trained ResNet-18 backbone (using MNIST, FashionMNIST, CIFAR-10, and SVHN), ThermoMerge is reported to achieve a multi-task average accuracy of **29.05%**, outperforming static Task Arithmetic (**27.25%**), AdaMerging (**26.10%**), and SyMerge (**27.90%**).
* The authors claim that ThermoMerge successfully mitigates the "Overfitting-Optimizer Paradox" (where unregularized entropy minimization collapses representations) and resolves the "Gray-to-Color Collapse" by using a pre-trained backbone with ancestral connectivity.

## Explicitly Claimed Contributions
1. **A Thermodynamic Paradigm:** Reformulating model merging through the lens of statistical mechanics, establishing a physical connection between optimization and physical thermodynamics.
2. **First-Principles Regularization:** Deriving the Helmholtz Free Energy Discrepancy (F-Min) objective and proving its mathematical equivalence to the temperature-scaled KL divergence.
3. **Dynamic Annealing & Thermal Coupling:** Proposing a global cooling schedule (TAS) combined with trainable task-wise thermal capacities ($\tau_k$) to balance localized energy and global partition function discrepancies.
4. **Empirical Superiority:** Showing state-of-the-art performance on a heterogeneous 4-task benchmark and demonstrating how ancestral connectivity stabilizes test-time adaptation.
