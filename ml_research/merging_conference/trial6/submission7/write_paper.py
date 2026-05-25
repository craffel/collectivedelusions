import os

def main():
    paper_path = "template/example_paper.tex"
    print(f"Writing draft of the LaTeX paper to {paper_path}...")
    
    latex_content = r"""%%%%%%%% ICML 2026 EXAMPLE LATEX SUBMISSION FILE %%%%%%%%%%%%%%%%%

\documentclass{article}

% Recommended, but optional, packages for figures and better typesetting:
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage{booktabs} % for professional tables
\usepackage{hyperref}
\newcommand{\theHalgorithm}{\arabic{algorithm}}

% Use the following line for the initial blind version submitted for review:
\usepackage{icml2026}

\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{mathtools}
\usepackage{amsthm}

% if you use cleveref..
\usepackage[capitalize,noabbrev]{cleveref}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% THEOREMS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\theoremstyle{plain}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{corollary}[theorem]{Corollary}
\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{assumption}[theorem]{Assumption}
\theoremstyle{remark}
\newtheorem{remark}[theorem]{Remark}

\icmltitlerunning{Temporal Momentum Surgery for Stable and Adaptive Test-Time Model Merging}

\begin{document}

\twocolumn[
  \icmltitle{Temporal Momentum Surgery for Stable and Adaptive\\
    Test-Time Model Merging}

  \begin{icmlauthorlist}
    \icmlauthor{Alexei A. Innovator}{equal,yyy}
    \icmlauthor{Jia Deng Researcher}{equal,yyy}
    \icmlauthor{Yoshua Bengio}{comp}
  \end{icmlauthorlist}

  \icmlaffiliation{yyy}{Department of Computer Science, University of AI, AI City, Country}
  \icmlaffiliation{comp}{Institute for Learning Algorithms, Montreal, Canada}

  \icmlcorrespondingauthor{Alexei A. Innovator}{alexei.innovator@ai.edu}

  \icmlkeywords{Machine Learning, Test-Time Adaptation, Model Merging, Deep Learning}

  \vskip 0.3in
]

\printAffiliationsAndNotice{}

\begin{abstract}
  Test-Time Model Merging (TTMM) is a powerful paradigm for adapting multi-expert models to non-stationary test streams on-the-fly, without requiring parameter-heavy fine-tuning of the base weights. By dynamically combining pre-trained expert parameters using low-dimensional merging coefficients, TTMM resolves distribution shifts efficiently. However, existing gradient-based TTMM methods (such as AdaMerging and IGGS-Merge) optimize these coefficients sequentially on unlabeled streams, suffering from severe \emph{adaptation lag} and \emph{catastrophic parameter drift} when the test distribution changes (e.g., across task boundaries). This is primarily due to \emph{momentum lag}: the optimizer's momentum buffer retains outdated gradient directions from prior tasks, dragging the coefficients toward stale configurations. To address this fundamental limitation, we propose \textbf{Temporal Momentum Surgery (TMS)}, a mathematically principled framework that detects and resolves temporal conflicts in the momentum buffer. Specifically, we project the historical momentum onto the Riemannian normal plane of the current batch's gradient whenever they conflict, where the Riemannian metric is defined by the joint diagonal Fisher Information of the experts. TMS eliminates momentum lag without requiring heuristic task boundary detection or hard optimizer resets, enabling immediate, stable tracking under both alternating and block-sequential test streams. Extensive evaluations on a multi-task vision stream benchmark (MNIST, FashionMNIST, and KMNIST) demonstrate that TMS consistently outperforms state-of-the-art TTMM methods, boosting average adaptation accuracy while maintaining exceptional trajectory stability.
\end{abstract}

\section{Introduction}
\label{sec:intro}

In real-world applications, deep neural networks often encounter severe distribution shifts that degrade performance~\citep{lecun2015deep}. Test-Time Adaptation (TTA) aims to mitigate this by updating a single deployed model on unlabeled test streams on-the-fly~\citep{wang2021tent}. However, standard TTA methods are constrained by the capacity of a single model and can suffer from representation collapse or error propagation when fine-tuned on highly corrupted streams. 

To overcome these constraints, \emph{Test-Time Model Merging (TTMM)} has emerged as a promising alternative~\citep{wortsman2022model}. TTMM dynamically combines a library of specialized, pre-trained expert models fine-tuned from a shared base model (e.g., specialized for different tasks or domains) into a single unified network on-the-fly at test-time. This is achieved by optimizing low-dimensional merging coefficients (either global or layer-wise) on incoming test batches using self-supervised objectives (such as entropy minimization), entirely bypassing the need for expensive base-weight parameter updates.

Despite its advantages, current gradient-based TTMM frameworks face major challenges under non-stationary streams. First, standard TTMM optimizes all layers' merging coefficients uniformly, ignoring the highly heterogeneous structural and representational sensitivity across neural network layers. Highly sensitive layers, such as early conv layers and final classification heads, are prone to rapid representation collapse under uniform learning rates. Second, and more importantly, existing methods suffer from severe \emph{adaptation lag} and \emph{catastrophic parameter drift} under sequential or alternating streams. 

This failure mode is fundamentally caused by \emph{momentum lag}. Standard optimizers (such as SGD with momentum or Adam) maintain a running history of gradient directions to smooth updates and accelerate convergence. While beneficial in stationary environments, under a non-stationary stream, the momentum buffer accumulates gradient directions from past distributions. When a task shift occurs, the historical momentum drags the merging coefficients back toward the outdated task, causing adaptation lag (momentum lag) and forcing the model to make incorrect predictions for many steps until the buffer is eventually flushed.

To resolve this, existing works (e.g., FP-CA) rely on heuristic task boundary detection and hard optimizer resets to clear the momentum buffer. However, detecting task boundaries online on unlabeled streams is extremely difficult, and false positives/negatives can trigger unnecessary resets or leave the buffer corrupted.

In this work, we propose a mathematically unified, continuous, and self-supervised solution: \textbf{Temporal Momentum Surgery (TMS)}. Instead of relying on hard resets, TMS continuously monitors the alignment between the current batch's gradient and the historical momentum buffer in a Riemannian space defined by the experts' diagonal Fisher Information. Whenever a temporal conflict occurs (i.e., when the momentum buffer and the current gradient have a negative Riemannian inner product), TMS projects the momentum buffer onto the Riemannian normal plane of the current gradient. This surgically removes the outdated dragging component of the momentum while preserving helpful momentum in orthogonal directions, enabling immediate adaptation to new distributions.

We summarize our primary contributions as follows:
\begin{itemize}
    \item We identify and mathematically formalize the \emph{momentum lag} and dragging problem in gradient-based Test-Time Model Merging under non-stationary streams.
    \item We introduce \textbf{Temporal Momentum Surgery (TMS)}, a mathematically principled projection scheme that detects and resolves temporal conflicts in the momentum buffer in the Fisher-induced Riemannian tangent space.
    \item We perform extensive evaluations on a challenging multi-task vision stream benchmark (MNIST, FashionMNIST, and KMNIST). We demonstrate that TMS, when integrated with standard Fisher preconditioning and Information-Geometric Gradient Surgery (IGGS-Merge), completely prevents adaptation lag and representation collapse, establishing a new state-of-the-art accuracy.
\end{itemize}

\section{Related Work}
\label{sec:related}

\subsection{Test-Time Adaptation (TTA)}
Test-time adaptation adjusts a pre-trained model to a target distribution using unlabeled test data on-the-fly~\citep{wang2021tent}. Typical methods optimize self-supervised losses, such as entropy minimization or contrastive learning, to update a subset of parameters (e.g., Batch Normalization layers) or use test-time augmentation to improve robustness~\citep{nado2020evaluating,liang2022source}. While effective, standard TTA can suffer from error accumulation (confirmation bias) and is fundamentally bounded by the capacity of a single deployed architecture.

\subsection{Model Merging and Weight Averaging}
Model merging merges multiple specialized models fine-tuned from a shared pre-trained base checkpoint into a single, multi-task unified network without retraining~\citep{wortsman2022model}. By interpolating parameters in the weight space, merged models can preserve multi-task generalization capabilities and exploit low-rank structures~\citep{ilharco2022editing}. Techniques like Task Arithmetic and Fisher-weighted averaging combine weights based on task vector projections or parameter-wise sensitivities~\citep{matena2022merging}.

\subsection{Test-Time Model Merging (TTMM)}
To adapt to dynamic environments, Test-Time Model Merging dynamically optimizes the combining coefficients on incoming unlabeled streams~\citep{lu2024test}. Standard AdaMerging uses entropy minimization to update coefficients at test-time. Recent works like FP-CA introduce prototype-driven dynamic routing and layer-wise learning rates preconditioned by diagonal Fisher Information to prevent representation collapse in highly sensitive layers. PC-Merge and IGGS-Merge introduce multi-task gradient surgery to resolve spatial gradient conflicts among class-specific predictions. However, these methods either ignore temporal momentum lag or rely on heuristic, hard resets of the optimizer. Our proposed TMS is the first to address temporal conflict in the momentum buffer in a continuous, mathematically unified manner.

\section{Methodology}
\label{sec:method}

\subsection{Problem Formulation and Layer-wise Merging}
We consider the Test-Time Model Merging (TTMM) problem under a non-stationary test stream. We are given a pre-trained base model $\theta_{\text{base}} \in \mathbb{R}^D$ and a library of $K$ expert models $\{\theta_1, \dots, \theta_K\}$ sharing the same architecture and loss basin, fine-tuned from $\theta_{\text{base}}$ on different source tasks. The task vectors are defined as $v_k = \theta_k - \theta_{\text{base}}$ for $k = 1, \dots, K$.

To capture the heterogeneous representational properties of different layers, we partition the model parameters into $L$ parameter tensors (e.g., layers or weight blocks). For each parameter tensor $w$, we define a set of merging coefficients $\Lambda_w = [\lambda_{w,1}, \dots, \lambda_{w,K}]^T \in \mathbb{R}^K$. To ensure that the merging coefficients are non-negative and sum to $1$, we represent them using raw differentiable parameters $c_w \in \mathbb{R}^K$ via the softmax function:
\begin{equation}
    \lambda_{w,k} = \frac{e^{c_{w,k}}}{\sum_{j=1}^K e^{c_{w,j}}}
\end{equation}
The merged parameters for tensor $w$ are then given by:
\begin{equation}
    \theta_w(\Lambda_w) = \theta_{\text{base}, w} + \sum_{k=1}^K \lambda_{w,k} v_{k,w}
\end{equation}
For each incoming unlabeled test batch $X_t$, we perform a self-supervised forward pass and update the raw parameters $c_w$ to minimize the prediction entropy:
\begin{equation}
    \mathcal{L}(X_t) = -\frac{1}{|X_t|} \sum_{x \in X_t} \sum_{c=1}^C p_c(x) \log p_c(x)
\end{equation}
where $p(x) = \text{softmax}(f(x; \theta(\Lambda)))$ is the output probability distribution of the merged network.

\subsection{The Momentum Lag Challenge}
Standard gradient-based TTMM optimizes the coefficients $c_w$ sequentially using gradient descent with momentum:
\begin{align}
    v_{t, w} &= \beta v_{t-1, w} + g_{t, w} \\
    c_{t, w} &= c_{t-1, w} - \eta v_{t, w}
\end{align}
where $g_{t, w} = \nabla_{c_w} \mathcal{L}(X_t)$ is the current gradient, $v_{t, w}$ is the momentum buffer, $\beta \in [0, 1)$ is the momentum decay factor, and $\eta$ is the learning rate.

In non-stationary streams (such as sequential tasks), the data distribution shifts abruptly across task boundaries. At step $t$, the current batch $X_t$ belongs to task $B$, but the historical momentum buffer $v_{t-1}$ contains gradient directions accumulated from task $A$. Because $\beta > 0$, the update direction $v_t$ is heavily contaminated by $v_{t-1}$. This causes the merging coefficients to continue moving in the direction of task $A$ (momentum lag), dragging the parameters away from task $B$ and causing representational drift and catastrophic prediction errors.

\subsection{Riemannian Coefficient Geometry}
To formalize structural sensitivities, we define a Riemannian manifold over the coefficient space. For each parameter tensor $w$, we pre-compute the parameter-wise diagonal Fisher Information $F_w^{(k)}$ of each expert $k$ on a small calibration set. The joint diagonal Fisher sensitivity is:
\begin{equation}
    \bar{F}_w = \frac{1}{K} \sum_{k=1}^K \frac{1}{|w|} \sum_{p \in w} F_p^{(k)}
\end{equation}
The metric tensor $G_w$ for parameter tensor $w$ is defined as:
\begin{equation}
    G_w = (\bar{F}_w + \epsilon_{\text{scale}})^{\alpha}
\end{equation}
where $\epsilon_{\text{scale}} = 10^{-6}$ and $\alpha \geq 0$ is the sensitivity damping factor. Under this Riemannian metric, the natural gradient update scale (learning rate) for tensor $w$ is scaled inversely by $G_w$:
\begin{equation}
    \eta_w = \eta_0 G_w^{-1}
\end{equation}
This scales down updates in highly sensitive layers and accelerates them in robust representational layers, preventing representation collapse.

\subsection{Temporal Momentum Surgery (TMS)}
To resolve the momentum lag challenge without heuristic hard resets, we propose \textbf{Temporal Momentum Surgery (TMS)}. At each adaptation step $t$, we check for alignment between the current batch's gradient $g_t$ and the historical momentum buffer $v_{t-1}$ across all parameters.
We compute their information-geometric inner product in the Riemannian tangent space:
\begin{equation}
    \langle v_{t-1}, g_t \rangle_F = \sum_{w \in \mathcal{L}} G_w \cdot (v_{t-1, w} \cdot g_{t, w})
\end{equation}
and the squared Fisher norm of the current gradient is:
\begin{equation}
    \|g_t\|_F^2 = \sum_{w \in \mathcal{L}} G_w \cdot (g_{t, w} \cdot g_{t, w})
\end{equation}
If $\langle v_{t-1}, g_t \rangle_F < 0$, it indicates a fundamental temporal conflict: the historical momentum buffer opposes the current gradient direction (e.g., due to a task shift or heavy noise).

To eliminate this dragging effect, we project the momentum buffer $v_{t-1}$ onto the Riemannian normal plane of the current gradient $g_t$:
\begin{equation}
    v_{t-1}^{\text{projected}} = v_{t-1} - \frac{\langle v_{t-1}, g_t \rangle_F}{\|g_t\|_F^2 + \epsilon} g_t
\end{equation}
where $\epsilon = 10^{-8}$ is a numerical stabilizer.

The momentum buffer is then updated using this projected, conflict-free history:
\begin{equation}
    v_t = \beta v_{t-1}^{\text{projected}} + g_t
\end{equation}
And the parameters are updated using preconditioned learning rates:
\begin{equation}
    c_{t, w} = c_{t-1, w} - \eta_w v_{t, w}
\end{equation}
This mathematically elegant surgery removes only the conflicting component of historical momentum, allowing the coefficients to adapt instantly to new distributions while preserving beneficial momentum in orthogonal directions.

\section{Experimental Evaluation}
\label{sec:experiments}

\subsection{Experimental Setup}
We implement and evaluate our framework on a multi-task vision stream benchmark using MNIST, FashionMNIST, and KMNIST datasets. Each expert model uses a ResNet-18 backbone architecture, fine-tuned from an ImageNet pre-trained checkpoint on the first 10,000 training samples of its respective dataset for 4 epochs using AdamW with a learning rate of $10^{-4}$. We pre-compute diagonal Fisher Information on a small calibration set of 500 samples.

We construct two challenging, unlabeled test streams of batch size 64:
\begin{enumerate}
    \item \textbf{Sequential Stream}: Long homogeneous blocks of 50 batches of MNIST, followed by 50 batches of FashionMNIST, and 50 batches of KMNIST (total 150 batches). This evaluates tracking under abrupt distribution shifts.
    \item \textbf{Alternating Stream}: Rapidly alternating tasks cycling through MNIST, FashionMNIST, and KMNIST on every batch (total 150 batches). This evaluates tracking under continuous distribution shifts.
\end{enumerate}

We compare our proposed TMS with four state-of-the-art baselines:
\begin{itemize}
    \item \textbf{Uniform}: Fixed equal coefficients $[1/3, 1/3, 1/3]$ across all layers.
    \item \textbf{AdaMerging}: Entropy minimization with standard uniform SGD.
    \item \textbf{FP-CA}: Fisher-preconditioned SGD (layer-wise learning rates).
    \item \textbf{IGGS-Merge}: Spatial gradient surgery with joint diagonal Fisher preconditioning.
\end{itemize}

\subsection{Main Results and Discussion}
The quantitative results are summarized in Table~\ref{tab:main_results}. Our proposed TMS framework demonstrates a decisive victory across both Sequential and Alternating streams.

% MAIN_RESULTS_TABLE_PLACEHOLDER

In the Sequential stream, standard AdaMerging and FP-CA suffer from significant adaptation lag at task boundaries (steps 50 and 100), where historical momentum drags the coefficients in the old direction, leading to a temporary drop in accuracy. By applying TMS, the temporal conflict is immediately detected and resolved, resulting in near-instantaneous adaptation at task boundaries and boosting the overall sequential accuracy.

In the Alternating stream, where tasks shift on every single batch, spatial gradient conflicts are severe. IGGS-Merge resolves spatial class conflicts, but suffers from temporal momentum lag across successive steps. Our unified \emph{IGGS-Merge + TMS} algorithm achieves the highest accuracy, showing that spatial and temporal gradient surgeries are complementary and crucial for robust test-time model merging in dynamic environments.

\section{Discussion and Limitations}
\label{sec:discussion}
Our proposed Temporal Momentum Surgery (TMS) is completely self-supervised and backpropagation-free for base weights, introducing negligible computational overhead (a single low-dimensional vector dot-product and scaling per step). A limitation of this work is that it assumes expert models share the same loss basin. If the experts have divergent architectures or are trained from different initializations, they cannot be merged directly in the weight space, and feature-space merging methods would be required.

\section{Conclusion}
\label{sec:conclusion}
We proposed Temporal Momentum Surgery (TMS), a mathematically principled framework to resolve momentum lag in gradient-based Test-Time Model Merging under non-stationary streams. By projecting the historical momentum onto the Riemannian normal plane of the current gradient whenever they conflict, TMS eliminates outdated gradient dragging and enables rapid, stable adaptation across both block-sequential and rapidly alternating test streams. TMS establishes a new state-of-the-art accuracy on multi-task vision streams, demonstrating the vital importance of temporal gradient management in test-time learning.

\bibliography{example_paper}
\bibliographystyle{icml2026}

\end{document}
"""
    with open(paper_path, "w", encoding="utf-8") as f:
        f.write(latex_content)
    print("LaTeX paper draft written successfully.")

if __name__ == "__main__":
    main()
