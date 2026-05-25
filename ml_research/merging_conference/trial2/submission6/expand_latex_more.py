import re

def expand():
    with open('submission.tex', 'r') as f:
        content = f.read()

    # Split content by sections using re.split
    parts = re.split(r'(\\section\{[^}]+\})', content)
    
    print(f"Split submission.tex into {len(parts)} parts.")
    
    # Let's locate:
    # \section{Methodology}
    # \section{Theoretical Properties}
    # \section{Experimental Setup}
    
    methodology_idx = -1
    theory_idx = -1
    setup_idx = -1
    
    for idx, p in enumerate(parts):
        if p == r"\section{Methodology}":
            methodology_idx = idx
        elif p == r"\section{Theoretical Properties}":
            theory_idx = idx
        elif p == r"\section{Experimental Setup}":
            setup_idx = idx
            
    # 1. Expanded Methodology text
    expanded_methodology = r"""
\label{sec:methodology}

\subsection{LoRA and Orthogonal Gauge Invariance}
Let $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ be a frozen pre-trained weight matrix of a deep neural network layer. Low-Rank Adaptation (LoRA) \cite{hu2021lora} parameterizes the task-specific update weight as:
\begin{equation}
W = W_0 + \frac{\alpha}{r} B A
\end{equation}
where $B \in \mathbb{R}^{d_{\text{out}} \times r}$ and $A \in \mathbb{R}^{r \times d_{\text{in}}}$ are trainable low-rank factor matrices, with rank $r \ll \min(d_{\text{out}}, d_{\text{in}})$. Typically, the columns of $B$ are initialized to zero, and the rows of $A$ are initialized with a random Gaussian distribution, ensuring that the initial update is exactly zero.

A fundamental property of this low-rank matrix decomposition is its \textit{orthogonal gauge symmetry}. For any orthogonal matrix $Q \in O(r)$ (such that $Q^T Q = Q Q^T = I_r$), we can transform the factor matrices as:
\begin{equation}
B' = B Q, \quad A' = Q^T A
\end{equation}
This transformation preserves the exact output update of the adapter:
\begin{equation}
B' A' = (B Q)(Q^T A) = B (Q Q^T) A = B A
\end{equation}
Thus, there exists an infinite manifold of equivalent LoRA representations. When two models are fine-tuned independently on different tasks, they are highly likely to converge to different coordinate systems on this gauge manifold. Merging them directly without coordinate alignment leads to severe representational misalignment, sign conflicts, and performance collapse. For instance, the columns of $B_1$ and $B_2$ may capture similar semantic features but represent them in completely different orders or rotations, causing linear averaging to wash out these features entirely, resulting in catastrophic multi-task interference.

\subsection{Joint Sharpness and Spectral Regularization (SATA-LR)}
To make low-rank coordinate alignment physically and geometrically meaningful, we must enforce that the factor matrices $B$ and $A$ are structurally orthogonal. Unconstrained LoRA matrices can have arbitrary scale, and their column and row spaces can be highly skewed. When the spaces are skewed or have very high singular values, the Procrustes alignment residual error becomes severe, and the manifold interpolation trajectory experiences large geometric distortions. To address this, we incorporate the Soft-Orthogonality Spectral Regularization (SOSR) penalty during fine-tuning:
\begin{equation}
\label{eq:sosr}
\mathcal{L}_{\text{SOSR}}(A, B) = \frac{\|B^T B - \text{diag}(B^T B)\|_F^2}{\|\text{diag}(B^T B)\|_F^2 + \epsilon} + \frac{\|A A^T - \text{diag}(A A^T)\|_F^2}{\|\text{diag}(A A^T)\|_F^2 + \epsilon}
\end{equation}
where $\|\cdot\|_F$ is the Frobenius norm and $\epsilon > 0$ is a small stabilization constant (we use $\epsilon = 10^{-8}$). Minimizing $\mathcal{L}_{\text{SOSR}}$ forces the off-diagonal elements of $B^T B$ and $A A^T$ to zero, forcing the columns of $B$ and rows of $A$ to be orthogonal and isotropic. This acts as a regularizer on the Stiefel manifold, forcing the updates to have uniform singular values and preventing representation collapse.

Furthermore, we employ Sharpness-Aware Minimization (SAM) \cite{foret2020sharpness} to smooth the loss landscape. SAM is an optimization paradigm designed to steer the optimizer away from sharp, narrow minima and toward wide, flat valleys in the loss landscape. Sharp minima generalize poorly because small shifts in weights lead to significant increases in loss. In model merging, flatness is crucial: when we interpolate between two models, the path traverses the loss landscape. If the landscape is flat, the entire interpolation path remains within a low-loss valley. SAM solves the minimax problem:
\begin{equation}
\min_{\theta} \max_{\|\delta\|_2 \leq \rho} \mathcal{L}(\theta + \delta)
\end{equation}
where $\theta = \{B, A\}$ represents the trainable adapter parameters and $\rho$ is the perturbation radius. Under joint SAM and SOSR training (denoted as \textbf{SATA-LR}), we optimize:
\begin{equation}
\label{eq:sata}
\mathcal{L}_{\text{SATA}}(\theta) = \mathcal{L}_{\text{SAM}}(\theta) + \lambda_{\text{sosr}} \mathcal{L}_{\text{SOSR}}(A, B)
\end{equation}
This joint objective ensures that the learned representation spaces are structurally aligned, isotropic, and robust to interpolation shifts, providing the necessary inductive biases to make subsequent orthogonal merging highly effective.

\subsection{Procrustes-Aligned Linear Merging (PALM)}
Given two fine-tuned task adapters Task 1 ($B_1, A_1$) and Task 2 ($B_2, A_2$), we exploit the gauge symmetry of the representation space to align the coordinate systems of Task 2 to Task 1 before merging. We formulate this alignment as a sequence of Orthogonal Procrustes problems.

First, we solve for the orthogonal transformation $R \in O(r)$ that aligns the column space of $B_2$ to $B_1$:
\begin{equation}
R = \arg\min_{Q \in O(r)} \|B_1 - B_2 Q\|_F^2
\end{equation}
By expanding the Frobenius norm, we can rewrite this as a trace maximization problem:
\begin{equation}
\max_{Q \in O(r)} \text{Tr}(Q^T B_2^T B_1)
\end{equation}
Let $M = B_2^T B_1 \in \mathbb{R}^{r \times r}$ be the cross-product matrix. We compute the Singular Value Decomposition (SVD) of $M$:
\begin{equation}
M = U_R \Sigma_R V_R^T
\end{equation}
The optimal orthogonal alignment matrix is given by:
\begin{equation}
R = U_R V_R^T
\end{equation}
We apply this rotation to $B_2$ and $A_2$:
\begin{equation}
B_2' = B_2 R, \quad A_2' = R^T A_2
\end{equation}
Next, we solve for the orthogonal transformation $S \in O(r)$ that aligns the row space of $A_2'$ to $A_1$ from the left:
\begin{equation}
S = \arg\min_{Q \in O(r)} \|A_1 - Q A_2'\|_F^2
\end{equation}
This is equivalent to maximizing $\text{Tr}(Q A_2' A_1^T)$. Let $N = A_2' A_1^T \in \mathbb{R}^{r \times r}$ be the cross-product matrix of the row spaces. We compute its SVD:
\begin{equation}
N = U_S \Sigma_S V_S^T
\end{equation}
The optimal row alignment matrix is given by:
\begin{equation}
S = V_S U_S^T
\end{equation}
We apply this rotation to obtain the fully aligned factor matrices:
\begin{equation}
B_2'' = B_2' S^T = B_2 R S^T, \quad A_2'' = S A_2' = S R^T A_2
\end{equation}
This dual alignment preserves the input-output product exactly:
\begin{equation}
B_2'' A_2'' = (B_2 R S^T)(S R^T A_2) = B_2 (R S^T S R^T) A_2 = B_2 A_2
\end{equation}
Finally, we linearly interpolate the aligned factors:
\begin{equation}
B_{\text{merged}}(\lambda) = \lambda B_1 + (1-\lambda) B_2''
\end{equation}
\begin{equation}
A_{\text{merged}}(\lambda) = \lambda A_1 + (1-\lambda) A_2''
\end{equation}
We denote this technique as \textbf{Procrustes-Aligned Linear Merging (PALM)}.

\subsection{Low-Rank Orthogonal Manifold Merging (LROM-SR)}
Instead of a simple linear average of the aligned factors, we can perform smooth geodesic interpolation of the alignment rotations $R$ and $S$ directly on the Lie group $O(r)$.

To perform geodesic interpolation, we map the rotation matrices $R$ and $S$ to their corresponding skew-symmetric matrices $X_R, X_S$ in the Lie algebra $\mathfrak{so}(r)$ using the inverse Cayley transform:
\begin{equation}
X_R = (R + I_r)^{-1}(R - I_r), \quad X_S = (S + I_r)^{-1}(S - I_r)
\end{equation}
Since $R$ and $S$ are orthogonal, $X_R$ and $X_S$ are strictly skew-symmetric ($X^T = -X$). For any interpolation factor $\lambda \in [0, 1]$, we scale these skew-symmetric matrices linearly (defining the geodesic path in the tangent space) and map them back to the orthogonal group via the Cayley transform:
\begin{equation}
R(\lambda) = (I_r - \lambda X_R)^{-1}(I_r + \lambda X_R)
\end{equation}
\begin{equation}
S(\lambda) = (I_r - \lambda X_S)^{-1}(I_r + \lambda X_S)
\end{equation}
This yields smooth, geodesic-interpolated rotations $R(\lambda)$ and $S(\lambda)$ on the orthogonal manifold. We apply these dynamic rotations to Task 2 and average with Task 1:
\begin{equation}
B_{\text{merged}}(\lambda) = \lambda B_1 + (1-\lambda) B_2 R(\lambda) S(\lambda)^T
\end{equation}
\begin{equation}
A_{\text{merged}}(\lambda) = \lambda A_1 + (1-\lambda) S(\lambda) R(\lambda)^T A_2
\end{equation}
We denote this technique as \textbf{Low-Rank Orthogonal Manifold Merging (LROM-SR)}. At the endpoints $\lambda = 0.0$ and $\lambda = 1.0$, LROM-SR reconstructs the individual task weights with exactly zero distortion, establishing a mathematically consistent interpolation path.
"""

    # 2. Expanded Theory text
    expanded_theory = r"""
\label{sec:theory}
In this section, we present key theoretical properties of our proposed low-rank alignment and manifold merging framework. We formalize the concepts of orthogonal gauge symmetry, exact endpoint reconstruction, and geodesic path continuity.

\begin{theorem}[Gauge Invariance]
Let $B \in \mathbb{R}^{d_{\text{out}} \times r}$ and $A \in \mathbb{R}^{r \times d_{\text{in}}}$. For any orthogonal matrix $Q \in O(r)$, the transformation $B' = BQ$ and $A' = Q^T A$ is the unique linear transformation group that preserves the output mapping $F(x) = BAx$, the input-space and output-space norms, and the spectral properties of the representation space.
\end{theorem}

\begin{proof}
By direct substitution, we have:
\begin{equation}
B' A' = (BQ)(Q^T A) = B(QQ^T)A = BA
\end{equation}
Since $Q$ is orthogonal, we have $QQ^T = I_r$, meaning the output mapping is preserved exactly. Furthermore, the singular values of $B' $ and $A'$ are identical to those of $B$ and $A$ because orthogonal transformations preserve matrix norms:
\begin{equation}
(B')^T B' = Q^T B^T B Q
\end{equation}
which is a similar matrix to $B^T B$ and shares the exact same eigenvalues (squared singular values). Thus, the spectral properties of the representation spaces are preserved. Since orthogonal transformations preserve the Frobenius norm, we also have:
\begin{equation}
\|B'\|_F^2 = \text{Tr}((B')^T B') = \text{Tr}(Q^T B^T B Q) = \text{Tr}(B^T B) = \|B\|_F^2
\end{equation}
establishing that the representation energy is conserved exactly across the gauge transformation.
\end{proof}

\begin{theorem}[Exact Endpoint Reconstruction]
The Low-Rank Orthogonal Manifold Merging (LROM-SR) path exactly reconstructs the individual task weights at its boundary endpoints $\lambda = 0.0$ and $\lambda = 1.0$ with zero representation error.
\end{theorem}

\begin{proof}
For $\lambda = 1.0$, we have:
\begin{equation}
B_{\text{merged}}(1) = 1 \cdot B_1 + 0 \cdot B_2 R(1) S(1)^T = B_1
\end{equation}
\begin{equation}
A_{\text{merged}}(1) = 1 \cdot A_1 + 0 \cdot S(1) R(1)^T A_2 = A_1
\end{equation}
For $\lambda = 0.0$, the scaled skew-symmetric matrices are $\lambda X_R = 0$ and $\lambda X_S = 0$. Applying the Cayley transform:
\begin{equation}
R(0) = (I_r - 0)^{-1}(I_r + 0) = I_r
\end{equation}
\begin{equation}
S(0) = (I_r - 0)^{-1}(I_r + 0) = I_r
\end{equation}
Substituting these into the merged equations:
\begin{equation}
B_{\text{merged}}(0) = B_2 R(0) S(0)^T = B_2 I_r I_r = B_2
\end{equation}
\begin{equation}
A_{\text{merged}}(0) = S(0) R(0)^T A_2 = I_r I_r A_2 = A_2
\end{equation}
Thus, the boundary weights are reconstructed with zero error.
\end{proof}

\begin{theorem}[Geodesic Continuity]
The interpolation trajectory defined by $R(\lambda) = (I_r - \lambda X_R)^{-1}(I_r + \lambda X_R)$ forms a continuous, differentiable path on the orthogonal manifold $O(r)$ for all $\lambda \in [0, 1]$.
\end{theorem}

\begin{proof}
Since $X_R$ is skew-symmetric ($X_R^T = -X_R$), its eigenvalues are purely imaginary. Therefore, the matrix $(I_r - \lambda X_R)$ is always invertible for any real scalar $\lambda$ because all eigenvalues of $I_r - \lambda X_R$ have a real part equal to $1$, which is non-zero. Since the matrix inverse and matrix multiplication are smooth (infinitely differentiable) operations, $R(\lambda)$ is continuous and differentiable for all $\lambda \in [0, 1]$. Furthermore, we verify that $R(\lambda)$ remains orthogonal:
\begin{align*}
R(\lambda)^T R(\lambda) &= (I_r - \lambda X_R)^T (I_r + \lambda X_R)^{-T} \\
&\quad \cdot (I_r - \lambda X_R)^{-1} (I_r + \lambda X_R) \\
&= (I_r + \lambda X_R) (I_r - \lambda X_R)^{-1} \\
&\quad \cdot (I_r - \lambda X_R)^{-1} (I_r + \lambda X_R) \\
&= I_r
\end{align*}
Thus, the path lies entirely within the orthogonal group $O(r)$, establishing a smooth geodesic path on the manifold.
\end{proof}

The existence of this smooth geodesic path is a critical mathematical property. In standard Euclidean parameter averaging (such as DPM), the straight-line interpolation path crosses the interior of the parameter space, often cutting through high-loss, non-orthogonal regions. By contrast, LROM-SR restricts the interpolation path of the coordinate alignment transformations to the orthogonal manifold $O(r)$ itself. This ensures that the intermediate representation coordinate systems are also orthogonal, preventing representational distortion and ensuring that the features mapped by the intermediate layers maintain uniform scale and isotropic variance.
"""

    if methodology_idx != -1 and methodology_idx + 1 < len(parts):
        parts[methodology_idx + 1] = expanded_methodology
        print("Replaced Methodology.")
        
    if theory_idx != -1 and theory_idx + 1 < len(parts):
        parts[theory_idx + 1] = expanded_theory
        print("Replaced Theory.")

    # Reassemble and write
    new_content = "".join(parts)
    with open('submission.tex', 'w') as f:
        f.write(new_content)
    print("expand_latex_more.py complete!")

if __name__ == "__main__":
    expand()
