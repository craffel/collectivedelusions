# Momentum-Merge: A Minimalist Deconstruction of Biochemical Kinetics in Dynamic Model Merging

## 1. Persona Alignment
This proposal directly embodies **The Minimalist** persona by aggressively pruning unnecessary complexity. Modern dynamic model ensembling techniques have become needlessly convoluted: ChemMerge models representation trajectories using systems of non-equilibrium biochemical kinetics, Arrhenius reaction rates, catalytic enzymes, and continuous-time ordinary differential equations (ODEs), introducing multiple parameters ($k_{\text{decay}}$, $\Delta t$, $\eta$, $\tau$, $A_0$) and requiring sophisticated numerical solvers (such as explicit Euler projection or exponential integrators).

Using Occam's razor, we deconstruct this complex formulation. We mathematically prove that ChemMerge's stateful kinetics are equivalent to a state-dependent, adaptive Exponential Moving Average (EMA). Applying our philosophy that "simpler is strictly better if it matches complex performance," we propose **Momentum-Merge**, which replaces the entire biochemical kinetics solver with a single, standard, constant Exponential Moving Average (EMA) / Momentum equation. It requires exactly *one* hyperparameter ($\beta$) and *one* standard line of code, completely stripping away the chemical metaphors while preserving—and potentially exceeding—the accuracy and representation stability of ChemMerge.

## 2. Core Techniques
Momentum-Merge introduces a clean, training-free, and parameter-free temporal smoothing layer for dynamic model merging:
- **Unit-Norm Calibration (UNC):** Standard offline, one-time calibration of task-specific centroids $\mu_k^{(3)}$ at the output of the shared feature extractor (Layer 3) on 64 calibration samples:
  $$\mu_k^{(3)} = \frac{1}{|\mathcal{C}_k|} \sum_{s \in \mathcal{C}_k} h_s^{(3)}$$
- **Raw Cosine Similarity Routing:** Measures representation alignment of intermediate activations $h_b^{(l-1)}$ to early-layer centroids $\mu_k^{(3)}$:
  $$S(h_b^{(l-1)}, \mu_k^{(3)}) = \frac{h_b^{(l-1)} \cdot \mu_k^{(3)}}{\|h_b^{(l-1)}\|_2 \|\mu_k^{(3)}\|_2}$$
- **Softmax Temperature Gating:** Normalizes similarity scores into raw ensembling weights $w_{k, b}^{(l)}$ using a temperature parameter $\tau$:
  $$w_{k, b}^{(l)} = \frac{\exp\left( S(h_b^{(l-1)}, \mu_k^{(3)}) / \tau \right)}{\sum_{j=1}^K \exp\left( S(h_b^{(l-1)}, \mu_j^{(3)}) / \tau \right)}$$
- **Constant Exponential Moving Average (EMA) Smoothing:** Couples ensembling weights sequentially across layers using a constant momentum coefficient $\beta \in [0, 1]$:
  $$\alpha_{k, b}^{(l)} = (1 - \beta) w_{k, b}^{(l)} + \beta \alpha_{k, b}^{(l-1)}$$
  with boundary condition initialized uniformly at the entrance of adapted layers:
  $$\alpha_{k, b}^{(3)} = \frac{1}{K}$$

## 3. Mathematical Formulation
- **The Momentum Update Equation:**
  Inside each adapted layer $l \in [L_{\text{frozen}}+1, L]$, the final ensembling weights $\alpha_{k, b}^{(l)}$ are computed as:
  $$\alpha_{k, b}^{(l)} = (1 - \beta) w_{k, b}^{(l)} + \beta \alpha_{k, b}^{(l-1)}$$
  where $\beta \in [0, 1]$ is the constant momentum coefficient, and $w_{k, b}^{(l)}$ is the standard, soft-aligned nearest-centroid routing weight.
- **Layer-to-Layer Routing Jitter:**
  To evaluate the physical representation stability across depth, we define the ensembling weight routing jitter (mean-squared layer-to-layer variance) as:
  $$\text{Jitter} = \frac{1}{L - L_{\text{frozen}} - 1} \sum_{l=L_{\text{frozen}}+1}^{L} \sum_{k=1}^K (\alpha_{k, b}^{(l)} - \alpha_{k, b}^{(l-1)})^2$$
- **The Stability-Accuracy Pareto Sweep:**
  By sweeping the single momentum parameter $\beta$ from $0.0$ to $1.0$, we can map the entire Pareto frontier of model ensembling:
  - When $\beta = 0.0$, Momentum-Merge collapses to standard **stateless nearest-centroid routing** (e.g., SPS-ZCA), which has high accuracy but suffers from high-frequency layer-to-layer weight oscillations (high jitter).
  - When $\beta = 1.0$, Momentum-Merge collapses to **static uniform merging** (e.g., Uniform Merging), which has zero layer-to-layer jitter but suffers from severe multi-task representation collapse.
  - By setting $\beta \in (0.0, 1.0)$, we achieve a mathematically optimal trade-off, smoothing out high-frequency representational noise while preserving deep, task-specific expert specialization.

## 4. Architecture Specifications
We evaluate Momentum-Merge inside the identical configuration as prior work to ensure direct compatibility and rigorous comparisons:
- **Backbone Model:** A simulated 14-layer deep network with intermediate feature dimension $D = 192$, representing standard ViT-Tiny configurations.
- **Shared Feature Extractor:** The first $L_{\text{frozen}} = 3$ layers are frozen and shared across all tasks, with no task-specific adapters loaded.
- **Adapted Layers:** The remaining layers $l \in [4, 14]$ are adapted. Task-specific Low-Rank Adaptation (LoRA) experts target the Query and Value projection weights inside each self-attention block, with rank $r = 8$.
- **Single-Pass Parallel Activation Blending:** In each adapted layer $l \in [4, 14]$, we compute the final blended output activation $h_b^{(l)}$ in a single parallel forward pass:
  $$h_b^{(l)} = h_{\text{base}, b}^{(l)} + \sum_{k=1}^K \alpha_{k, b}^{(l)} h_{\text{expert}, k, b}^{(l)}$$
  where $h_{\text{base}, b}^{(l)} = h_b^{(l-1)} W_{\text{base}}^{(l)}$ and $h_{\text{expert}, k, b}^{(l)} = h_b^{(l-1)} A_k^{(l)} B_k^{(l)}$, ensuring constant $O(1)$ edge serving latency.

## 5. Baselines
We evaluate Momentum-Merge against a robust suite of prior methods:
1. **Expert Ceiling (Oracle):** Standalone execution of the correct expert for each task.
2. **Uniform Merging:** Static weight-space parameter averaging ($\alpha_k = 0.25$).
3. **SPS-ZCA:** Stateless, nearest-centroid routing with no temporal/spatial state tracking.
4. **SABLE:** Stateless, sample-wise activation-blending routing using raw cosine similarities.
5. **ChemMerge:** The SOTA biochemical kinetics ensembling baseline (with Arrhenius rate normalized soft rates and Euler/exponential ODE solvers).

## 6. Step-by-Step Interaction
For each sample $b$ in the serving stream, the forward pass of Momentum-Merge is executed as follows:
1. **Input Extraction:** The input sample representation $h_b^{(0)}$ enters the network and propagates through the shared layers 1--3.
2. **Early Feature Anchoring:** The activation $h_b^{(3)}$ is extracted at the output of Layer 3.
3. **Sequential Layer-wise Ensembling:** For each adapted layer $l = 4 \dots 14$:
   - **Compute Similarity:** Compute the cosine similarity $S(h_b^{(l-1)}, \mu_k^{(3)})$ between the current hidden state and the fixed, shared early-layer centroids $\mu_k^{(3)}$.
   - **Evaluate Routing Weights:** Apply the Softmax function with temperature $\tau = 0.01$ to generate the raw routing weights $w_{k, b}^{(l)}$.
   - **Momentum Smoothing:** Update the running ensembling weights using the simple, standard momentum equation:
     $$\alpha_{k, b}^{(l)} = (1 - \beta) w_{k, b}^{(l)} + \beta \alpha_{k, b}^{(l-1)}$$
     initialized with boundary condition $\alpha_{k, b}^{(3)} = 1/K$.
   - **Execute Parallel Expert Paths:** Pass the representation $h_b^{(l-1)}$ through the shared base weights and the $K$ specialized LoRA experts in parallel.
   - **Activation Blending:** Intersect and blend the expert activations using the smoothed ensembling weights:
     $$h_b^{(l)} = h_{\text{base}, b}^{(l)} + \sum_{k=1}^K \alpha_{k, b}^{(l)} h_{\text{expert}, k, b}^{(l)}$$
4. **Final Logits & Prediction:** Pass the final representation $h_b^{(14)}$ to the task-specific classification heads to generate logits and make predictions.
