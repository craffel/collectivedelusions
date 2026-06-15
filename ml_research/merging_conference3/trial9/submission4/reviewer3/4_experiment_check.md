# Intermediate Evaluation 4: Experiment Check

## Experimental Setup & Datasets
The authors use the **Analytical Coordinate Sandbox (ICS)** to simulate a 14-layer deep neural network (e.g., Vision Transformer) with:
- Hidden representation dimension $D = 192$.
- $L_{\text{frozen}} = 3$ shared/frozen layers.
- $L_{\text{adapt}} = 11$ adapted layers target Query and Value matrices via rank-8 LoRA modules.
- $K = 4$ task manifolds (MNIST, Fashion-MNIST, CIFAR-10, SVHN) modeled as orthogonal coordinate blocks of dimension 48 in $\mathbb{R}^{192}$.
- Sequential, online streaming protocol with 1000 heterogeneous, shuffled samples ($B=1$).
- Isotropic layer-wise noise ($\sigma_{\text{layer}} = 0.015$) and calibrated task-specific representation noise scales ($\sigma = [0.05, 0.15, 0.40, 1.20]$).

### Evaluation of Sandbox Design
- **Strengths:** The ICS is highly controlled, allowing for precise tracking of hidden activations, representational noise propagation, and ensembling weight dynamics. It is perfectly synchronized across methods to ensure scientific hygiene.
- **Weaknesses (Ecological Validity):** As the authors openly acknowledge in Section 5.1, the synthetic sandbox does not capture the full complexity of physical language and vision models (e.g., non-orthogonal task overlaps, natural activation scales, and complex token-generation mechanics).

To mitigate this limitation, the authors propose a highly detailed, actionable blueprint for real-world PEFT serving on pre-trained Transformer backbones (e.g., LLaMA-7B) with GLUE/HumanEval/GSM8K tasks in Appendix B. This strongly demonstrates their constructive approach to validation.

## Baselines
The paper compares Momentum-Merge against a robust, representative set of baselines:
1. **Expert Ceiling (Oracle):** Upper-bound standalone execution.
2. **Uniform Merging (Static):** Averaging all expert weights.
3. **SABLE (Stateless):** Sample-wise similarity-based routing (uncalibrated and calibrated via Layer Centroids).
4. **ChemMerge (Stateful SOTA):** Continuous-time biochemical ODE routing (uncalibrated and calibrated).

To ensure a fair baseline, the authors perform a systematic grid sweep over ChemMerge's hyperparameters ($\Delta t \in [0.5, 2.0], k_{\text{decay}} \in [0.1, 0.8]$) across 5 seeds in Appendix A, identifying the optimal configuration ($\Delta t = 1.0, k_{\text{decay}} = 0.3$) for their comparisons. This is a very rigorous baseline alignment.

## Statistical Significance
Unlike many papers that report a single seed, the authors evaluate all methods across **10 independent random seeds**.
- They report both mean joint accuracy and mean routing jitter along with standard deviations.
- They perform a **seed-by-seed pairwise analysis** with a paired $t$-test.
- The results confirm that Momentum-Merge's accuracy improvement over stateless SABLE is statistically significant ($p \approx 0.0212 < 0.05$), and its improvement over SOTA ChemMerge is highly significant ($p \approx 0.0061 < 0.01$). This provides solid empirical grounding.

## Critical Sweeps & Ablations (Appendices)
The paper is exceptionally thorough in its empirical verification, containing several highly high-signal sweeps:
1. **Softmax Temperature Sweep ($\tau$):** In Appendix C, they sweep $\tau \in [0.005, 0.300]$, demonstrating that stateless routing is highly sensitive to temperature (lower temperature causes routing jitter to double from 0.0167 to 0.0732), while Momentum-Merge's temporal smoothing decouples jitter from temperature.
2. **Joint Hyperparameter Interaction Sweep ($\beta \times \tau$):** In Appendix D, they map the joint performance landscape, showing the optimal basin resides in the moderate momentum band ($\beta \in [0.40, 0.80]$).
3. **Depth-wise Momentum Scheduling Sweep:** In Appendix E, they show that a V-shaped depth-wise momentum schedule ($\beta^{(l)}$) achieves an additional **28.8% reduction in routing jitter** while preserving peak classification accuracy.
4. **Calibration Subset Size Sweep ($|\mathcal{C}_k|$):** In Appendix G, they sweep $|\mathcal{C}_k| \in [8, 128]$. They expose **Recurrence Trapping** as a key vulnerability of stateful recurrences under low calibration data ($|\mathcal{C}_k| \le 16$), where initial boundary errors propagate and trap the ensembling coefficients in sub-optimal states, leading to a 4.80% absolute accuracy drop compared to stateless SABLE.
5. **Task-Asymmetric Layer-wise Noise Sweep:** In Appendix H, they construct four noise scenarios. They show that while ChemMerge's task-asymmetric reaction kinetics offer a minor accuracy buffer ($+0.15\%$ to $+0.30\%$) under extreme asymmetry, it incurs a massive surge in routing jitter (surging to 0.0260). Momentum-Merge Advanced maintains near-zero routing jitter (0.002955, an **8.8$\times$ reduction**) with comparable accuracy.
6. **Scalability Sweep ($K=10$):** In Appendix I, they scale the task pool to $K=10$. They show that as distraction noise scales up, the optimal momentum shifts from $\beta = 0.60$ to $\beta = 0.80$ to provide stronger low-pass filtering.

## Do the Results Support the Claims?
Yes, the empirical results strongly support all major claims:
- **Redundancy of Biochemical Metaphor:** Momentum-Merge matches or exceeds the classification accuracy and routing stability of ChemMerge under perfectly synchronized seed comparisons, while being training-free and single-parameter.
- **Accuracy-Stability Trade-off:** SABLE + Layer Centroids achieves the highest joint accuracy (77.24%) but maximum jitter (0.0285). Momentum-Merge Advanced trades a minor fraction of accuracy (74.98%) to virtually eliminate routing oscillations, collapsing jitter to 0.000374 (a 76.2$\times$ reduction over calibrated SABLE).
- **Control of Dynamics via $\beta$:** Sweeping $\beta$ shows a smooth, well-behaved Pareto frontier peaking at $\beta = 0.60$.
- **Boundary Conditions:** Raw Boundary Initialization reduces routing jitter by up to 70.1$\times$ across all layer-wise noise scales, showing high robustness to noise propagation.
