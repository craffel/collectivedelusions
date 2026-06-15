# Novelty Check and Delta from Prior Work

## Delta from Prior Work
The proposed method, **ThermoMerge**, is positioned as a "radical departure from the static Euclidean paradigm" and a "paradigm-shifting framework." However, when stripped of its heavy thermodynamic terminology, the actual technical delta from prior work is highly incremental:
* **Canonical Ensemble Mapping** is mathematically identical to the standard **temperature-scaled Softmax** function, which is a well-established technique in deep learning (e.g., in calibration and knowledge distillation).
* **Helmholtz Free Energy** ($F_k(x; T) = -T \ln Z_k(x; T)$) is simply the negative temperature-scaled **Log-Sum-Exp** of the logits.
* **Helmholtz Free Energy Discrepancy Minimization (F-Min)** is mathematically identical to **temperature-scaled Kullback-Leibler (KL) divergence** (Knowledge Distillation) between the output distributions of the individual expert models (acting as teachers) and the merged model (acting as the student).
* **Thermodynamic Annealing Schedule (TAS)** is a straightforward application of **Simulated Annealing** (temperature decay), a classical optimization technique proposed in the 1983 and extensively used in various deep learning scheduling contexts.
* **Task-wise Thermal Coupling** ($\tau_k$) is functionally equivalent to optimizing a **per-task logit scaling factor** (temperature) during the adaptation phase.

Thus, the technical delta over existing test-time adaptive merging methods (like AdaMerging and SyMerge) is simply the application of temperature-scaled soft-label knowledge distillation from the experts on unlabeled calibration data, accompanied by a decaying temperature schedule and per-task learnable temperatures.

## Characterization of Novelty
The novelty of this paper is characterized as **highly incremental, disguised by excessive over-theorization**. 

While the physical analogy is intellectually creative, it does not introduce any fundamentally new mathematical or algorithmic primitives. Translating standard machine learning operations (softmax, log-sum-exp, KL divergence, temperature scaling, simulated annealing) into thermodynamic terms (Boltzmann distributions, partition functions, free energy, canonical ensembles, thermal capacities) creates an illusion of high-level physics-grounded novelty. 

In terms of actual machine learning contribution, using soft labels (KL divergence from experts) as an objective to regularize test-time adaptive model merging is a logical and incremental next step to address the overfitting of entropy minimization (AdaMerging), but it does not represent a paradigm shift.
