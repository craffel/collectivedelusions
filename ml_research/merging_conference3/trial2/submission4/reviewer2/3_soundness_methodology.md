# Soundness and Methodology Evaluation

This evaluation focuses on the technical soundness, mathematical consistency, and empirical reproducibility of the proposed EdgeMerge framework. While the writing is clear, several major methodological flaws and logical inconsistencies severely undermine the paper's scientific credibility.

---

## 1. The Ablation Paradox: EdgeMerge is Functionally Static
The central premise of the paper is that "inter-task interference" in model merging can be resolved by "fine-grained, channel-wise merging coefficients... computed dynamically from activation statistics." 

However, the authors' own ablation studies (Table 5) completely invalidate this claim:
- **Uniform Gating** ($\alpha_k = 1/K = 0.125$ for all tasks) achieves **69.58%** accuracy.
- **Layer-Wise Gating** (LWG, collapsing 512 channels into a single scalar per layer) achieves **69.59%** accuracy.
- **EdgeMerge with CWSG** (the proposed channel-wise softmax gating) achieves **69.58%** accuracy.

Because setting the routing weights to a flat, uniform constant ($1/K$) yields the exact same accuracy (or even $0.01\%$ lower than a single layer scalar), the complex machinery of Forward-Only Activation Sampling (FOAS), Scale-Normalized Delta Activation Salience (SNDAS), and Channel-Wise Softmax Gating (CWSG) is **functionally inert**. 

The performance improvement of Decoupled EdgeMerge (69.58%) over standard Task Arithmetic (68.74%) is **not** driven by adaptive routing or conflict resolution. Instead, it is entirely driven by **Decoupled Scale Routing (DSR)**—specifically, setting a large scale for the transformer layers ($\lambda_{static} = 0.25$) and a highly regularized, small scale for the visual projection layer ($\lambda_{proj} = 0.025$, which is $0.20 / 8$). 

A simple, static model with two scaling parameters ($\lambda_{static} = 0.25, \lambda_{proj} = 0.025$) and *zero* calibration data, *zero* activation forward passes, and *zero* on-device latency achieves the exact same performance of 69.58%. The proposed EdgeMerge framework is an over-engineered implementation of simple static scale tuning.

---

## 2. Suspicious Numerical Invariance & The Division-by-Zero Bug
The paper reports several instances of suspiciously perfect numerical invariance that point to potential bugs or a lack of physical validity in the experiments:

### A. Synthetic Zero Tensors and the "Manifold-Projection Hypothesis"
In Table 2, the authors show that calibrating with **Physical Validation Images**, **Synthetic Gaussian Noise**, and **Synthetic Zero Tensors** all yield the **exact same average accuracy of 68.689%** down to three decimal places. 

If we evaluate the accuracy of 8 tasks on 1024 images each (a total of 8192 test images), an accuracy of $68.689\%$ corresponds to exactly:
$$\frac{5627}{8192} \approx 68.68896\%$$

This means that substituting real, high-resolution physical images with pure zero tensors changed the classification of **exactly zero images** across the entire 8192-image evaluation suite. 

The authors propose a highly sophisticated "manifold-projection hypothesis" in Appendix C to justify this: they claim that upstream positional embeddings, layer norms, and biases pre-condition zero inputs into a structured latent manifold $X_k^{base}$.

However, this explanation is mathematically and logically highly suspect:
1. If the input is a pure zero tensor, the patch embedding projection (typically a convolutional layer) outputs exactly zeros.
2. While positional embeddings and biases do make the downstream representations $X_k^{base}$ non-zero, why would this constant base activation produce *identical* routing coefficients to physical images that actually contain complex spatial semantic structures?
3. More importantly, consider the case where the upstream activations $X_k$ are processed. If there is any bug in the implementation where the calibration forward pass is bypassed or inputs are zeroed, the activation delta $\Delta H_k = X_k (W_k - W_{base})^T$ could evaluate to zero.
4. If $\Delta H_k = 0$, then the Frobenius norm $\|\Delta H_k\|_F = 0$. In Equation 6, computing the normalized delta:
$$\tilde{\Delta} H_k = \frac{\Delta H_k}{\|\Delta H_k\|_F}$$
would result in a **division-by-zero**, yielding `NaN` or `inf` values.
5. If the implementation handles these `NaN`s by replacing them with zeros (using e.g., `torch.nan_to_num`), the salience scores $S_k[j]$ would all evaluate to 0.
6. Passing a vector of zeros through the softmax function (Equation 8) yields:
$$\alpha_k[j] = \frac{\exp(0)}{\sum_{l=1}^K \exp(0)} = \frac{1}{K}$$
which collapses the gating coefficients to a **perfectly uniform distribution** ($\alpha_k = 1/K$).

This mathematical collapse perfectly explains why "Synthetic Zero Tensors" and "Uniform Gating" yield the exact same accuracy. The "manifold-projection hypothesis" appears to be a post-hoc rationalization of a silent division-by-zero bug that collapsed the adaptive routing into static uniform averaging.

### B. Mismatched vs. Correct Calibration (Table 5)
In Table 5, the authors compare Mismatched Calibration (using $X_k^{base}$) and Correct Calibration (using $X_k^{expert}$) across 9 hyperparameter configurations. In 8 out of 9 rows, the average accuracy is **identical to three decimal places**:
- Row 1: 69.580% vs. 69.580%
- Row 2: 69.568% vs. 69.568%
- Row 4: 69.580% vs. 69.580%
- Row 5: 69.568% vs. 69.568%
- Row 7: 69.580% vs. 69.580%
- Row 8: 69.556% vs. 69.556%

For Row 3, the difference is **69.434%** (5688/8192) vs. **69.446%** (5689/8192)—a difference of **exactly one image** out of 8192.

This level of invariance is statistically extraordinary. If the correct calibration uses $K$ different fine-tuned expert encoders, the intermediate features $X_k^{expert}$ must exhibit significant drift from $X_k^{base}$. This drift should alter the computed channel saliences, changing the routing coefficients, which in turn should alter the merged projection layer weights. 

It is virtually impossible for these weight differences to have absolutely zero impact on the predictions of 8192 test images across 8 heterogeneous tasks. This suggests that the gating coefficients $\alpha_k$ either have no functional impact on the final classification decision, or they are identical due to code errors.

---

## 3. Reproducibility
The paper provides a highly detailed appendix listing hyperparameters (Table 7) and entropy statistics, which is commendable. However, without public release of the code, reproducing these exact "invariant" numbers would be highly challenging. 

The fact that the entire performance contribution of the method is driven by two static scaling parameters ($\lambda_{static} = 0.25, \lambda_{proj} = 0.025$) means that a practitioner seeking to reproduce "EdgeMerge" would be far better off simply implementing a 2D grid sweep over these two static parameters rather than implementing the complex, error-prone activation-sampling pipeline.
