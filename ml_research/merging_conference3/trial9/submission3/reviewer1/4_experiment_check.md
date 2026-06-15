# Evaluation Part 4: Experimental Design and Empirical Claims

## Critique of Experimental Design and Evaluation Rigor

The empirical section of the paper provides several tables and standard deviations across 10 random seeds. However, an adversarial analysis of the experimental setup reveals severe methodological limitations, weak baselines, and a massive disconnect between the paper's broad claims and its actual toy-scale evaluation.

---

### 1. Toy-Scale and Synthetic Evaluation (No Actual Deep Learning)
The most glaring weakness of the evaluation is that **no actual deep neural networks were trained or evaluated on real-world tasks**.
* **Synthetic Sandbox (Experiments 1 & 2):** The primary testing ground is a 14-layer, 192-dimensional "Analytical Coordinate Sandbox (ICS)" with artificially generated coordinate subspaces. This is a pure mathematical simulation of representation propagation, completely divorced from the complexities of real optimization in deep neural networks.
* **PCA-Reduced Toy Embeddings (Experiment 3):** To claim "real-world validation," the authors extract frozen embeddings from a pre-trained ResNet18. The datasets used are MNIST, Fashion-MNIST, KMNIST, and USPS. These are toy-scale, 2D computer vision datasets that are decades old and have been solved with near-100% accuracy.
* **Dimension Reduction:** They project these toy embeddings into 192 dimensions using Principal Component Analysis (PCA). They then simulate sequential ensembling of linear projection heads. 

There is a massive chasm between this toy setup and the paper's claimed applicability to "modern multi-task language model serving (e.g., routing specialized LoRA adapters on a frozen LLaMA base model)." The paper does not evaluate on **a single real transformer, language model, or large-scale vision model**. It does not perform actual backpropagation through deep feature extractors, nor does it serve actual multi-task benchmarks (such as GLUE or DecaNLP). Serving PCA projections of MNIST embeddings in 2026 is an exceptionally low-standard evaluation for a top-tier ML venue.

---

### 2. Severely Unfavorable Trade-Offs against SABLE / ChemMerge
The authors include non-parametric centroid-based ensembling baselines, SABLE and ChemMerge. A close inspection of the results in **Table 6 (Experiment 3)** reveals that the proposed CR-Router is substantially worse than these baselines:
* **SABLE (Centroid Baseline):** **70.60% ± 2.41%** Accuracy.
* **ChemMerge (Baseline):** **68.90% ± 2.50%** Accuracy.
* **CR-Router (Proposed, $\gamma_{\text{scale}} = 1.0$):** **53.70% ± 2.37%** Accuracy.

This represents an **absolute performance drop of 16.90%** compared to SABLE. Even when the authors employ their post-hoc temperature annealing trick ($\gamma_{\text{scale}} = 0.10$, Table 8) to boost accuracy, CR-Router only reaches **62.45% ± 2.98%**, which is still **8.15% lower** than SABLE.

The authors attempt to justify this massive loss in accuracy by pointing to CPU latency (Table 9), where CR-Router has a forward pass latency of 25.34ms compared to SABLE's 40.06ms (for a batch size of 400). However:
1. A **17% absolute drop in accuracy** (or even an 8% drop) is a catastrophic penalty in almost any real-world machine learning deployment. It is highly unlikely any practitioner would accept such a massive degradation in model capability to save 15 milliseconds of CPU latency.
2. Centroid-based methods like SABLE are extremely simple, have zero training cost, and require no calibration optimization. CR-Router requires formulating a complex joint regularized objective, running gradient descent on scarce calibration sets, and tuning multiple hyperparameters ($\lambda_{\text{spec}}$, $\lambda_{\text{temp}}$, $\gamma_{\text{scale}}$), only to yield a significantly worse model.

---

### 3. Lack of Scaling Analysis (Calibration Size)
The experiments are restricted to a highly constrained data-scarce setting of **16 calibration samples per task**. While data-scarce calibration is an interesting regime, the paper completely fails to provide a scaling analysis. 

What happens when we increase the calibration set size to 64, 256, 1024, or the full training set?
* Does the unregularized "Linear Router" stop overfitting and match/surpass CR-Router?
* If unregularized routing stabilizes with slightly more data, then the complex mathematical constraints of CR-Router are only useful in an extremely narrow, artificially restricted data regime.
* The absence of any dataset-size ablation curves is a major experimental omission.

---

### 4. Unverified Claims regarding Test-Time Annealed Trajectories
In Table 8, the authors show that scaling down the inference temperature to $\gamma_{\text{scale}} = 0.10$ improves accuracy. They assert:
> *"Because the router parameters were trained under strict contraction bounds... the routing trajectories remain stable even when the inference-time temperature is annealed."*

This is an unverified empirical claim. To prove this, the authors must provide Gating Depth-Variance (GDV) and Running Lipschitz Bounds (RLB) *specifically for the annealed models* during inference. 
As argued previously, scaling down the temperature by a factor of 10 or 100 mathematically explodes the Lipschitz constant of the Softmax routing function, which is the exact cause of routing jitter. Without empirical plots showing that the annealed model still maintains a smooth, jitter-free trajectory across depth during inference, this assertion is highly suspicious and likely false. The authors have likely traded away all stability benefits to recover some of SABLE's accuracy, leaving them with an unstable router that still underperforms the non-parametric baseline.
