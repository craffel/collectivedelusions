# 3. Soundness and Methodology Evaluation

## Clarity of Description
The description of the methodology, experimental design, and optimization formulations is exceptionally clear and transparent. The authors do an outstanding job of defining their variables, outlining simplifying assumptions, and explicitly calling out potential confounding variables (such as optimization capability/budget asymmetry and simulator circularity). 

## Appropriateness of Methods
- **Calibrated Sensitivity Simulator:** The design of the Model II Landscape is highly appropriate and grounded. Rather than using arbitrary synthetic functions, the curvature parameters ($A_k^{(l)}$, $B_k^{(l)}$) and the optimal trajectories are calibrated against empirical layer-wise accuracy-sensitivity statistics of a 12-layer Vision Transformer (ViT-B/32).
- **Correlated Stream Noise Modeling:** The sampling of stream noise as a single transductive offset ($\epsilon_{\text{stream}}$) exactly once per adaptation session is a major improvement over prior models that assume independent, zero-mean noise at each step. This maps much more realistically to the constant selection biases of real-world test streams.
- **Stratified Few-Shot Sampling:** Recognizing that random sampling of $M=10$ instances from multi-class datasets (e.g., 10 classes) carries a $99.96\%$ probability of omitting some classes entirely, the use of stratified sampling to ensure equal class representation is an essential practical detail that ensures validation stability.
- **Symmetrical Baseline Evaluation:** The authors' deconstruction of the optimization asymmetry (running OFS-Tune with a restricted Adam optimizer and Online AdaMerging with L-BFGS-B) is methodologically outstanding, ensuring that the results are not simply artifacts of solver capabilities.

## Potential Technical Flaws and Limitations
While the methodology is highly robust and carefully executed, there are several key limitations and minor gaps that a practitioner must highlight:

1. **Toy-Scale Physical Weight-Space Validation:** 
   The physical weight-space validation is conducted on a custom, 5-layer Convolutional Neural Network (CNN) trained on CPU for simple MNIST and FashionMNIST subsets. This is a very small, toy-scale model. In practical industry settings, model merging is applied to massive foundation models (e.g., 86M ViT, 7B LLaMA, Stable Diffusion). As model size, width, and depth scale up, the nature of weight-space representational clashing, linear mode connectivity, and layer-wise sensitivity changes dramatically. It remains unproven whether the exact numerical advantages (e.g., OFS-Tune outperforming PolyMerge by 3.70% and AdaMerging by 4.20%) hold in massive, high-dimensional parameter spaces. 
2. **Computational Scaling of Solver on Foundation Models:**
   The paper proposes using the Nelder-Mead simplex algorithm to optimize the continuous polynomial trajectory parameters $\theta_S$. While Nelder-Mead is exceptionally fast for low-dimensional spaces ($4$ to $6$ parameters) when evaluation takes less than $1$ millisecond, it requires up to 200 full evaluations of the validation loss. For a massive foundation model (like a 7B LLM), evaluating the validation cross-entropy/perplexity over $M=10$ or $20$ stratified samples across multiple tasks 200 times would be extremely slow, introducing a major pre-deployment latency bottleneck. Although the authors propose "OFS-Adam" (first-order coordinate gradient descent) and "Expert Parameter Offloading" to resolve this scaling challenge, these strategies are presented as future directions and are **not empirically tested** in the current manuscript.
3. **Over-simplification of the TTA Loss in Simulation:**
   In simulation, the online TTA optimization objective ($\mathcal{L}_{\text{TTA}}$) directly tracks the ground-truth optimal parameter profiles perturbed by noise. This is a highly idealized and simplified setting (as acknowledged by the authors in Section 3.7). In actual physical deployments, online TTA has zero access to optimal trajectories or task-specific sensitivity curves and must optimize a highly non-convex, rugged, and potentially misaligned unsupervised prediction entropy surface. Consequently, the simulation results represent an optimistic upper bound, and the actual physical gap between TTA and OFS-Tune in real deployments is likely much wider.

## Reproducibility
The reproducibility of the submission is **excellent**. The paper provides a meticulous breakdown of:
- The DeepCNN architecture, training parameters, learning rates, weight decays, and standalone accuracies.
- The Nelder-Mead solver configurations, reflection/expansion/contraction coefficients, boundary constraint handling, and termination criteria.
- The mathematical calibration equations for early, middle, and late layers in the Model II simulator.
- Full details of the non-smooth zig-zag baseline and experimental setups in the appendix.
An expert reader would easily be able to replicate both the simulation study and the physical weight-space experiments using the provided details.
