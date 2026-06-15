# Soundness and Methodology Evaluation

## Clarity of Description
The mathematical and procedural descriptions of **Norm-Equalized Task Arithmetic (NETA)** are exceptionally clear, rigorous, and easy to follow. The paper provides explicit formulations for:
*   Task vector extraction (Eq. 1)
*   Layer-wise Frobenius norm and average norm (Eq. 3, Eq. 4)
*   The NETA scaling coefficients (Eq. 5)
*   The final closed-form merging equation (Eq. 6)
*   Extensions like continuous $\alpha$-relaxation (Eq. 7), the noise-damping stabilizer $\beta$ (Section 3.3), the composite Group 0 parameter grouping, and the closed-form scale-compensation factor $\gamma^l$ (Eq. 10).

Algorithm 1 provides clear, step-by-step pseudocode that is highly helpful for reproducibility.

## Appropriateness of Methods
From a physical and geometric perspective, the proposed methodology is highly appropriate:
1.  **Layer-wise vs. Model-wide Normalization**: The authors provide a strong, scientifically grounded justification for performing layer-wise rather than model-wide normalization. Different layers in deep neural networks represent distinct feature granularities, and a model-wide normalization would preserve early-stream representation dominance, leading to destructive interference.
2.  **Composite Input Grouping**: Grouping low-dimensional input-stage parameters (e.g., class token, position embeddings) with the first visual Transformer block avoids unstable scaling of very small updates, which is physically and structurally sound.
3.  **Noise-Damping Stabilizer ($\beta$)**: Introducing $\beta$ prevents the catastrophic amplification of fine-tuning noise in layers with near-zero updates.
4.  **Scale Compensation Factor ($\gamma^l$)**: This mathematically rigorous addition addresses the directional norm contraction of the merged update vector, allowing training-free, automated scale restoration.

## Potential Technical Flaws & Weaknesses
Despite the methodological elegance, there are some potential limitations and inconsistencies:

### 1. The Small Update Inflation Risk
NETA assumes that every task should contribute an update of exactly identical Frobenius norm at every layer. While the noise-damping stabilizer $\beta$ prevents catastrophic division by zero, NETA still scales up extremely small task updates. If a task has a small update, it often means the pre-trained model already possessed highly capable representations for that task, and it only required minimal, local adaptation. Forcing its update scale to equal the layer average might introduce irrelevant noise or interfere with features optimized for more complex tasks. Although the empirical results in Table 2 show that NETA is stable under varying $\beta$, this conceptual risk of "over-scaling" minor updates remains unaddressed theoretically.

### 2. Major Empirical Inconsistency: Standard Deviation of Exactly 0.00%
The most critical soundness concern is a logical and empirical contradiction in the reported standard deviations of Table 1:
*   **Task-Wise AdaMerging** on FashionMNIST reports **$77.54\% \pm 0.00\%$** across 3 seeds.
*   **Layer-Wise AdaMerging** on MNIST reports **$98.44\% \pm 0.00\%$** across 3 seeds.
*   Meanwhile, **standard Task Arithmetic** reports **$96.03\% \pm 0.26\%$** on MNIST and **$82.10\% \pm 0.64\%$** on FashionMNIST.

The authors attempt to explain this in Section 4.2.2:
> "Through rigorous log verification, we confirm that this is not a checkpoint reuse artifact, but is instead driven by physical optimization boundaries... prediction entropy gradients consistently drive the corresponding scaling coefficients to their exact lower or upper parameter constraints (clamping boundaries) across all seeds. Furthermore, because evaluations are conducted on a discretized subset of 1024 test images, these stable clamped weights translate into identical classification counts, yielding a standard deviation of exactly 0.00% while other unconstrained tasks retain standard non-zero variances."

However, this explanation contains a major logical loophole:
1.  If the three seeds represent independent random trials where **different fine-tuned expert checkpoints** are merged, then even if the optimization coefficients are clamped to identical boundaries, the underlying expert weights themselves are different. Merging different expert weights must produce different merged models, yielding non-zero variance in classification accuracy on the test set.
2.  If the three seeds represent trials where **different 1024-image test subsets** are sampled, then evaluating even a completely deterministic and identical merged model on different subsets of 1024 images must result in non-zero variance in accuracy.
3.  The only way standard Task Arithmetic could have non-zero standard deviations ($\pm 0.26\%$ and $\pm 0.64\%$) while AdaMerging has exactly $0.00\%$ standard deviation is if:
    *   The expert checkpoints and the 1024 test images are completely identical across all three seeds (which means standard Task Arithmetic should have $\pm 0.00\%$ standard deviation, but it does not), OR
    *   There is a bug in the evaluation script where AdaMerging evaluations are inadvertently reusing the same model checkpoint or evaluating on a fixed, non-random test subset, failing to apply the seed correctly.

This empirical inconsistency is a significant concern that undermines the rigor of the test-time optimization baseline results.

## Reproducibility
The reproducibility of the proposed NETA method is exceptionally high. It requires no training, is training-free, and has no moving parts except for a simple Frobenius norm calculation and scaling. The paper provides complete mathematical formulations and explicitly lists the PyTorch parameter keys mapped to the composite Group 0 block. Implementing NETA would take fewer than 10 lines of standard PyTorch code, making it highly reproducible.
