# 3. Soundness and Methodology Check

## Soundness of the Mathematical Formulation
The mathematical formulation of Robust Linear Routing (RLR) is exceptionally sound, clean, and elegant. It contains no unnecessary complexity or mathematical obfuscation:

1. **Direct Gating Formulation:** The linear mapping $z = Wx + b$ is the simplest, most transparent way to project input representations to task logits. The total number of parameters is exceptionally small (772 trainable parameters for a 4-task Vision Transformer benchmark), ensuring minimal memory and computational overhead.
2. **Softmax Temperature Scaling:** The introduction of temperature $T \ge 1$ in the softmax formula:
   \begin{equation}
   a_k = \frac{\exp(z_k / T)}{\sum_{j=1}^N \exp(z_j / T)}
   \end{equation}
   is mathematically rigorous. It is a standard and highly effective technique in deep learning for controlling the entropy of a categorical distribution. In the context of weight blending, a higher temperature directly guarantees a smoother, softer mixture of expert models, preventing the blending coefficients from collapsing to a hard, discrete switch (delta distribution) that disrupts the representational alignment of parameter spaces.
3. **Weight Regularization:** Minimizing the Frobenus norm $\|W\|_F^2$ of the routing weights directly bounds the magnitude of $W$. Mathematically, this constrains the Lipschitz constant of the linear mapping, guaranteeing that even under significant out-of-distribution representation shifts (where $x$ has extreme coordinate values), the resulting raw logit variance is bounded. This is a very clean and classic way to secure robust generalization in neural networks.
4. **Uniform Calibration Loss:** The calibration loss:
   \begin{equation}
   \mathcal{L}(W, b) = \sum_{t=1}^N \mathcal{L}_{\text{task}, t}(W, b) + \alpha \|W\|_F^2
   \end{equation}
   is highly parsimonious. It treats all tasks equally and removes any dependency on complex task-difficulty proxies, reinforcement learning-based loss weightings, or heuristic hyperparameter adjustments during calibration. This ensures a clean and objective optimization landscape.

## Methodological Soundness of the Training Loop
The training loop design exhibits high rigor and alignment with the training-free paradigm of model merging:
- **Few-Shot Calibration:** Using a tiny dataset of 16 samples per task (64 samples in total) is a standard practice in the post-hoc merging literature (e.g., OFS-Tune, SuiteMerge, QWS-Merge). This ensures that the calibration process remains extremely data-efficient and takes less than a second on a single GPU.
- **Optimization Strategy:** Using the standard Adam optimizer with a learning rate of $0.01$ and training for exactly 100 steps represents a highly disciplined, parsimonious optimization process. The authors identify that over-optimizing for too many steps (e.g., $>1000$ steps) in the absence of regularization is a primary cause of baseline routing collapse, as it drives unregularized weights to extreme values and saturates the softmax gates. Bounding the optimization to 100 steps is a simple and highly effective methodological choice.
- **Representation Extraction Layer:** The authors default to extracting globally average-pooled representations from deep layers (globally average-pooled representation of Block 11 of the ViT backbone). This choice is ablated comprehensively in Section 4.5 by comparing representations from Early (Patch Embed), Middle (Block 5), and Late (Block 11) layers, demonstrating that routing from any layer achieves successful convergence under their stable, parsimonious training loop. This empirical finding is highly significant as it directly refutes the theoretical claims of prior work that deep representations are fundamentally corrupted.

## Areas for Improvement / Methodological Notes
- **Empirical Superiority vs. Stability Trade-off:** The methodology is honest and scientifically rigorous in highlighting that RLR's weight decay and temperature scaling are empirically redundant in standard homogeneous environments where unregularized routing is already highly effective. RLR acts primarily as a specialized stabilizer to secure robustness under out-of-distribution shifts and mixed heterogeneous streams. This distinction is intellectually honest but means that readers should not expect RLR to provide a significant performance boost in clean, homogeneous in-distribution settings.
- **Sensitivity to Hyperparameters $\alpha$ and $T$:** Since RLR introduces two new hyperparameters—the regularization penalty $\alpha$ and the temperature $T$—it is critical to verify if the method is highly sensitive to their specific values. The authors address this by conducting a thorough 2D hyperparameter sensitivity sweep (Figure 4) over $\alpha \in [0.0, 0.02]$ and $T \in [1.0, 5.0]$, confirming that RLR is exceptionally robust and stable across a wide range of values. This empirical validation makes the methodology highly sound and complete.
