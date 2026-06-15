# 3. Soundness and Methodology

## Clarity of the Description
The description of the SLD-Merge framework is generally clear, with well-structured mathematical formulations of the offline SVD phase, the bounded cosine router, and the classification head selection. However, the exact implementation of the parallel forward pass is poorly detailed and contains a major, unaddressed scalability issue.

## Critical Technical Flaws & Methodological Vulnerabilities

### 1. The PyTorch Parallel Forward Pass Complexity Botttleneck
In Section 3.3, the authors present the vectorized, parallel forward pass as:
$$Y = X W_{base}^{(l)} + \sum_{k=1}^K \alpha_k \odot \left( (X A_k^{(l)}) B_k^{(l)} \right)$$
where $\alpha_k \in \mathbb{R}^{B \times 1 \times 1}$ is a broadcasted routing coefficient. 

This formulation reveals a **critical technical flaw**: during the forward pass, **all $K$ low-rank adapters are fully executed** for every batch, and the outputs of the inactive adapters are simply zeroed out by multiplying by the sparse coefficient $\alpha_k$. 
- Consequently, the computational complexity of the adapter forward pass scales as $O(K \cdot B \cdot N \cdot D \cdot r)$, meaning the FLOPs scale **linearly with the number of tasks $K$**.
- While the authors report a low computational overhead of $+8.3\%$ FLOPs, this is only because they evaluated on a tiny task suite of $K=4$. If a practitioner scales this to $K=50$ or $K=100$ tasks, the computational overhead of executing all $K$ low-rank adapter paths in parallel will be massive, completely defeating the "computationally lean" and "parameter-efficient" on-device deployment goals.
- In contrast, actual weight-space model merging produces a single dense weight matrix $W_{merged}$, resulting in a forward pass complexity of $O(B \cdot N \cdot D^2)$ which is **completely independent of $K$**. The authors completely glaze over this fundamental architectural trade-off.

### 2. Lack of Activation-Awareness in SVD Truncation
The SVD factorization is performed offline on the raw weight task vectors: $V_k^{(l)} = U_k^{(l)} \Sigma_k^{(l)} ({V'_k}^{(l)})^T$ without considering the distribution of incoming activations. 
- It is a well-established tenet in modern quantization and low-rank compression literature (e.g., *SmoothQuant*, *AWQ*, *ASVD*, and *SVD-LLM*) that weight-magnitude SVD is mathematically suboptimal because weight parameters with large magnitudes may be multiplied by near-zero activation values, while small weights can be multiplied by massive activation values.
- True optimal low-rank decomposition requires scaling the weight matrices by the activation covariance matrix before SVD (activation-aware decomposition). By neglecting activation-awareness, SLD-Merge's offline SVD introduces unnecessary reconstruction errors and degrades generalization on out-of-distribution inputs.

### 3. Representation Shift during Calibration
To compute the routing basis vectors $\Phi_k^{(l)}$, the authors pass calibration samples through the model in a *uniform merging* configuration to obtain activation centroids. However, at inference time, the model executes in a *sparse low-rank* configuration. 
- This creates a mathematical "representation shift" between the calibration phase (where centroids are collected) and the inference phase (where activations are routed).
- Although the authors claim in Appendix C.2 that this shift has "virtually zero negative impact" due to early-layer frozen consistency, they provide no rigorous theoretical bounds or mathematical proof. In deeper networks, this shift could accumulate across layers and lead to severe routing failures.

### 4. Non-Standard, Low-Shot Setup as a Confounding Variable
The authors restrict their training set to a mere **256 samples per dataset**, resulting in highly under-trained experts (as evidenced by the SVHN expert's abysmal $29.30\%$ accuracy).
- The authors defend this as a "rigorous stress-test representing extreme low-resource transfer learning." However, this non-standard, "toy" setup acts as a major confounding variable.
- In under-trained, overfitted networks, representations are highly unstable, noisy, and artificially orthogonalized across domains, which makes activation-based routing trivial.
- On standard, fully converged experts, activation distributions align more closely, and task-routing in intermediate/late layers is far more challenging. Furthermore, the claim that SVD truncation *outperforms* full-rank experts (by $+1.38\%$) is highly suspicious and likely an artifact of overfitting on the tiny 256-sample dataset. On standard-size datasets, truncating task vectors would almost certainly degrade performance.

## Reproducibility
The authors provide detailed hyperparameter values (learning rate, AdamW settings, SVD rank) and architecture specifications in the appendix, which makes the work moderately reproducible. However, the absence of public source code or a concrete repository link prevents verification of the exact parallel forward pass implementation and its actual hardware execution behavior.
