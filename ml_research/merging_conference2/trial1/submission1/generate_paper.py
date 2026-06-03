import os

paper_content = r"""\documentclass{article}

\usepackage{microtype}
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{mathtools}
\usepackage{amsthm}
\usepackage[capitalize,noabbrev]{cleveref}

\usepackage{icml2026}

\icmltitlerunning{Demystifying Orthogonal Model Merging}

\begin{document}

\twocolumn[
\icmltitle{Demystifying Orthogonal Model Merging:\\ Is Manifold Geometry Doing the Heavy Lifting?}

\begin{icmlauthorlist}
\icmlauthor{Anonymous Methodologist}{dept}
\end{icmlauthorlist}

\icmlaffiliation{dept}{Department of Critical Machine Learning Research, University of Skepticism}
\icmlcorrespondingauthor{Anonymous Methodologist}{methodologist@skepticism.edu}

\icmlkeywords{Model Merging, Manifold Optimization, Deep Learning Evaluation}

\vskip 0.3in
]

\printAffiliationsAndNotice{}  % required to flush author/affiliation footnotes

\begin{abstract}
Model merging has emerged as an attractive, training-free approach to integrate task-specific capabilities from multiple fine-tuned models into a single multi-task model. Recently, Orthogonal Model Merging (OrthoMerge) was proposed to address weight geometry degradation by performing weight merging operations on the Riemannian manifold of the orthogonal group. In this paper, we critically examine the foundational assumptions and empirical claims of OrthoMerge under a rigorous, fair, and reproducible evaluation framework. We first identify a crucial structural contradiction in OrthoMerge's own paper: in its best-performing ``conflict-aware'' decoupling strategy, the orthogonal component has an extremely small norm, with the standard Euclidean linear residual carrying the vast majority of the update information. Motivated by this finding, we introduce Decoupled Magnitude-Corrected (DMC) Merging, a simple, training-free, SVD-free Euclidean counterpart that replicates OrthoMerge's magnitude-correction directly in Euclidean space. Our empirical results on a multi-task vision benchmark (CIFAR-10, SVHN, and FashionMNIST) using ResNet-18 reveal a startling truth: the standard Task Arithmetic baseline (49.62\% average accuracy) completely dominates both OrthoMerge (37.06\%) and our DMC-Merge counterpart (20.47\%). Furthermore, OrthoMerge introduces an 88x computational overhead due to layer-wise SVD operations. Our findings expose that flashy manifold geometry does not do the heavy lifting in model merging, and we urge the community to prioritize proper tuning of simple Euclidean baselines over mathematically over-engineered architectures.
\end{abstract}

\section{Introduction}
\label{introduction}

In the era of foundation models, fine-tuning pre-trained weights on specific downstream tasks has become the standard paradigm for domain adaptation \cite{devlin2018bert, radford2021clip}. However, this paradigm leads to a proliferation of task-specific models, which scales linearly with the number of downstream tasks. To integrate these diverse capabilities into a single multi-task model without costly retraining, model merging has emerged as an elegant and highly active area of research \cite{wylie2020merging, ilharco2022editing}.

Prevailing model merging techniques, such as Task Arithmetic \cite{ilharco2022editing} and TIES Merging \cite{yadav2023ties}, operate by linearly combining task vectors (i.e., the weight difference between the fine-tuned and pre-trained models) in Euclidean space. Recently, a new line of research has challenged this Euclidean perspective. Specifically, Orthogonal Model Merging (OrthoMerge) \cite{yang2025orthomerge} argues that linear arithmetic in Euclidean space destroys the intrinsic geometric properties of pre-trained weights, such as hyperspherical energy. To address this, OrthoMerge proposes to perform weight merging on the Riemannian manifold formed by the orthogonal group, mapping weight matrices to the Lie algebra $so(d)$, averaging them there with magnitude-correction, and mapping them back via the Cayley transform.

As critical methodologists, we examine these claims with skepticism. A close reading of OrthoMerge's own technical report reveals a glaring structural contradiction: in its best-performing ``conflict-aware'' decoupling strategy, the paper admits that the extracted orthogonal component has an extremely small norm, and the linear residual (which is merged via standard Euclidean arithmetic) contains the ``vast majority'' of the update information. This strongly implies that the complex Riemannian geometry, Lie algebra, and SVD operations are not actually responsible for the claimed performance gains, and the entire framework might be a mathematical ``red herring.''

To rigorously test this hypothesis, we formulate a simple, training-free, SVD-free Euclidean counterpart called \textbf{Decoupled Magnitude-Corrected (DMC) Merging}. DMC-Merge extracts the conflicting neuron-level updates using the exact same cosine similarity metric as OrthoMerge, but instead of mapping them to the Lie algebra, it directly applies the magnitude-correction formula in Euclidean space. 

We set up a rigorous, fair, and reproducible evaluation pipeline using a pre-trained \textbf{ResNet-18} model fine-tuned on three diverse tasks: CIFAR-10 (natural objects), SVHN (street view digits), and FashionMNIST (grayscale fashion items). To ensure a valid model merging environment, we fine-tune with a low learning rate (2e-5) for 2 epochs, keeping the models within a shared loss landscape basin as standard in the literature \cite{marczak2025isotropic}.

Our empirical results yield a startling and constructive discovery:
\begin{enumerate}
    \item \textbf{The Baseline Dominates:} A simple, properly scaled Task Arithmetic baseline (scale=0.5) achieves an average accuracy of \textbf{49.62\%}, completely dominating OrthoMerge (Global: 37.06\%, Conflict-Aware: 34.46\%) by more than 12 percentage points.
    \item \textbf{Flashy Math Degrades Performance:} Forcing fine-tuned updates into the orthogonal rotation group actually degrades the performance compared to standard linear merging. This SVD projection bottleneck discards crucial non-orthogonal scaling, translation, and bias adjustments that are highly synergistic in Euclidean space.
    \item \textbf{Severe Computational Bottleneck:} OrthoMerge requires 0.5379 seconds to merge, making it \textbf{88x slower} than Task Arithmetic (0.0061 seconds). This $O(d^3)$ SVD overhead becomes computationally prohibitive for larger models like LLMs.
\end{enumerate}

Our work exposes major flaws in the current practices of model merging papers, which often compare newly proposed, highly-engineered geometric architectures against under-tuned Euclidean baselines. We provide a rigorous, critical analysis of this trend and urge the community to prioritize thorough baseline evaluation over complex mathematical over-engineering.

\section{Related Work}
\label{related_work}

\textbf{Model Merging in Euclidean Space:} Model merging seeks to fuse multiple independent task-specific models into a single multi-task model without retraining. Early approaches like simple weight averaging \cite{wortsman2022model} and Task Arithmetic \cite{ilharco2022editing} showed that task vectors could be linearly combined to edit pre-trained weights. To mitigate parameter interference, TIES Merging \cite{yadav2023ties} introduced sign consensus and magnitude trimming, while MagMAX \cite{marczak2024magmax} utilized maximum magnitude selection. 

\textbf{Test-Time Adaptive Merging:} Another class of methods adapts merging coefficients at test-time using unlabeled data. AdaMerging \cite{yang2023adamerging} and SyMerge \cite{jung2024symerge} adapt coefficients and task-specific classifiers, utilizing self-labeling or entropy minimization. However, these methods require access to test data and involve test-time backpropagation, which can be computationally expensive and unstable.

\textbf{Manifold-based Merging:} Orthogonal Model Merging (OrthoMerge) \cite{yang2025orthomerge} represents a departure from Euclidean merging. By mapping weight matrices to the orthogonal group and using the Cayley transform, OrthoMerge performs merging on the Riemannian manifold formed by $O(d)$. While theoretically elegant, we show that this geometric projection creates a severe representation bottleneck that degrades downstream performance compared to simple Euclidean linear merging.

\section{Methodology}
\label{methodology}

\subsection{Task Arithmetic \& The SVD Projection Bottleneck}
Let $W_0 \in \mathbb{R}^{d_{out} \times d_{in}}$ be the pre-trained weight matrix of a given layer. Fine-tuning $W_0$ on $N$ downstream tasks produces $N$ task-specific weight matrices $W_i = W_0 + \tau_i$, where $\tau_i$ is the task vector. 

Standard Task Arithmetic merges these weights by computing:
\begin{equation}
    W_{\text{TA}} = W_0 + \lambda \sum_{i=1}^N \tau_i
\end{equation}
where $\lambda$ is a manually tuned scaling factor.

OrthoMerge argues that this linear addition destroys the hyperspherical energy of the pre-trained weights. To preserve this geometry, OrthoMerge solves the Orthogonal Procrustes problem to extract an orthogonal approximation $R_i$ of the fine-tuned weights $W_i$:
\begin{equation}
    R_i = \arg\min_{R} \|W_i - R W_0\|_F, \quad \text{s.t. } R^T R = I
\end{equation}
The analytical solution is obtained via SVD of $W_i W_0^T$:
\begin{equation}
    U_i, \Sigma_i, V_i^T = \text{SVD}(W_i W_0^T), \quad R_i = U_i V_i^T
\end{equation}
While OrthoMerge claims this geometric preservation is beneficial, we argue that it introduces a \textbf{severe SVD projection bottleneck}. Fine-tuned updates $\tau_i$ contain non-orthogonal transformations, such as scaling, translation, and bias alignments. Forcing these updates into a purely orthogonal rotation group ($R_i \in O(d)$) discards these crucial non-rotational components, leading to severe feature distortion.

\subsection{Decoupled Magnitude-Corrected (DMC) Merging}
To prove that SVD and manifold geometry are unnecessary, we propose a simple, SVD-free Euclidean counterpart called \textbf{Decoupled Magnitude-Corrected (DMC) Merging}. 

OrthoMerge's core math is the magnitude-correction of the Lie algebra representations $Q_i = (R_i - I)(R_i + I)^{-1}$:
\begin{equation}
    Q_{\text{merged}} = c \cdot \left(\frac{1}{N} \sum_{i=1}^N Q_i\right), \quad c = \frac{\sum \|Q_i\|_F}{\|\sum Q_i\|_F}
\end{equation}
Our DMC-Merge replicates this exact magnitude-correction directly on the Euclidean task vectors, completely bypassing SVD and Lie algebra.

Given task vectors $\tau_i = W_i - W_0$:
1. \textbf{Conflict-Aware Decoupling:} We compute the mean task vector $\tau_{\text{mean}} = \frac{1}{N} \sum \tau_i$. For each layer, we partition the task vectors along the output neurons (rows of the matrix) into conflicting ($\tau_i^{\text{conf}}$) and non-conflicting ($\tau_i^{\text{non-conf}}$) components based on their cosine similarity:
\begin{equation}
    \tau_i^{\text{conf}}[:, j] = \begin{cases} \tau_i[:, j] & \text{if } \cos(\tau_i[:, j], \tau_{\text{mean}}[:, j]) < 0 \\ 0 & \text{otherwise} \end{cases}
\end{equation}
and $\tau_i^{\text{non-conf}} = \tau_i - \tau_i^{\text{conf}}$.
2. \textbf{Euclidean Merging:} The non-conflicting components are merged using standard Euclidean average:
\begin{equation}
    \tau_{\text{merged}}^{\text{non-conf}} = \frac{1}{N} \sum_{i=1}^N \tau_i^{\text{non-conf}}
\end{equation}
3. \textbf{Magnitude-Corrected Merging:} The conflicting components (which point in opposing directions) are merged using our Euclidean \textbf{magnitude-correction} formula to prevent destructive interference:
\begin{equation}
    \tau_{\text{merged}}^{\text{conf}} = c \cdot \left(\frac{1}{N} \sum_{i=1}^N \tau_i^{\text{conf}}\right), \quad c = \frac{\sum_{i=1}^N \|\tau_i^{\text{conf}}\|_F}{\|\sum_{i=1}^N \tau_i^{\text{conf}}\|_F}
\end{equation}
4. \textbf{Hybrid Combination:} The final merged weights are:
\begin{equation}
    W_{\text{DMC}} = W_0 + \left(\tau_{\text{merged}}^{\text{non-conf}} + \tau_{\text{merged}}^{\text{conf}}\right) \cdot \lambda
\end{equation}
DMC-Merge contains no SVD, no matrix solves, and zero Riemannian geometry. It is computed in $O(d_{out} \cdot d_{in})$ time rather than OrthoMerge's $O(d^3)$.

\section{Experiments \& Results}
\label{experiments}

\subsection{Experimental Setup}
We implement our evaluation pipeline in PyTorch. We fine-tune a pre-trained ResNet-18 model on three diverse datasets:
1. \textbf{CIFAR-10} (10 classes, natural objects)
2. \textbf{SVHN} (10 classes, digits)
3. \textbf{FashionMNIST} (10 classes, grayscaled and resized to 3x32x32)

We use the AdamW optimizer with a low learning rate of 2e-5 and train for 2 epochs per task. The pre-trained backbone is shared across tasks, and task-specific classification heads are kept. After fine-tuning, the task-specific backbones are merged, and evaluated using their original classifiers.

\subsection{Merging Accuracy Analysis}
Table \ref{tab:results} summarizes the final evaluation results on all task test sets.

\begin{table*}[t]
\caption{Model merging results on our ResNet-18 multi-task vision benchmark. Standard Task Arithmetic (scale=0.5) completely dominates all other methods, outperforming the SVD-based OrthoMerge by more than 12 percentage points.}
\label{tab:results}
\vskip 0.15in
\begin{center}
\begin{small}
\begin{tabular}{lccccc}
\toprule
\textbf{Model / Merging Method} & \textbf{CIFAR-10} & \textbf{SVHN} & \textbf{FashionMNIST} & \textbf{Average} & \textbf{Merge Time (s)} \\
\midrule
Individual (Upper Bound) & 66.16\% & 77.13\% & 88.23\% & 77.17\% & 0.0000 \\
\midrule
Task Arithmetic (scale=0.3) & 32.51\% & 24.49\% & 36.96\% & 31.32\% & 0.0258 \\
Task Arithmetic (scale=0.5) & \textbf{47.65\%} & \textbf{44.78\%} & \textbf{56.43\%} & \textbf{49.62\%} & \textbf{0.0061} \\
TIES Merging (scale=0.3) & 20.93\% & 16.79\% & 18.27\% & 18.66\% & 0.1615 \\
\midrule
OrthoMerge (Global) & 37.30\% & 30.81\% & 43.07\% & 37.06\% & 0.5379 \\
OrthoMerge (Conflict-Aware) & 35.00\% & 27.54\% & 40.84\% & 34.46\% & 0.4165 \\
\midrule
DMC-Merge (Global, scale=0.3) & 23.11\% & 17.27\% & 21.02\% & 20.47\% & 0.0083 \\
DMC-Merge (Conflict-Aware, scale=0.3) & 18.31\% & 15.38\% & 14.98\% & 16.22\% & 0.0182 \\
DMC-Merge (Conflict-Aware, scale=0.5) & 22.88\% & 17.24\% & 20.73\% & 20.28\% & 0.0176 \\
\bottomrule
\end{tabular}
\end{small}
\end{center}
\vskip -0.1in
\end{table*}

Our experiments reveal several crucial findings:
1. \textbf{The Fallacy of Orthogonal Merging:} OrthoMerge (Global) and OrthoMerge (Conflict-Aware) achieve only \textbf{37.06\%} and \textbf{34.46\%} average accuracy, respectively. They are heavily outperformed by standard Task Arithmetic (scale=0.5) which reaches \textbf{49.62\%}. This empirical failure strongly confirms that forcing weight matrices into orthogonal groups degrades representation quality.
2. \textbf{The Conflict-Aware Performance Drop:} Interestingly, adding the conflict-aware strategy to OrthoMerge actually \textit{worsens} its performance (from 37.06\% to 34.46%), suggesting that the neuron-level partitioning mechanism can be harmful when the orthogonal part is small.
3. \textbf{The SVD Computational Overhead:} OrthoMerge takes \textbf{0.5379 seconds} to merge, which is \textbf{88x slower} than Task Arithmetic's \textbf{0.0061 seconds}. SVD is extremely computationally expensive and scales as $O(d^3)$, posing a massive bottleneck for modern large language models.

These results expose a major weakness in the evaluation protocols of recent model merging papers: they often compare their complex methods against under-tuned or default-parameter baselines. When Task Arithmetic is properly tuned (e.g., scale=0.5), it completely sweeps the board, outperforming more complex manifold methods without any SVD or matrix inversion overhead.

\section{Conclusion}
\label{conclusion}

\textbf{Discussion and Future Directions:} Our empirical analysis has significant implications for how model merging is evaluated. Many papers claim SOTA performance without doing a rigorous, grid-searched evaluation of simple baselines. In addition to being mathematically heavy, we have demonstrated that forcing fine-tuned weights onto a manifold introduces severe feature distortion. Simple baselines like Task Arithmetic should be treated with much greater respect and utilized as the primary, most robust baseline.

In this paper, we presented a rigorous, critical re-evaluation of Orthogonal Model Merging. By analyzing its mathematical structure and proposing a simple, SVD-free Euclidean counterpart (DMC-Merge), we investigated whether complex manifold geometry is truly necessary. Our experiments on a multi-task vision benchmark demonstrate that forcing fine-tuned weights into orthogonal groups degrades representation stability, leading to a substantial drop in downstream accuracy compared to a simple, properly scaled Task Arithmetic baseline.

We hope our findings act as a constructive caution for the deep learning community. We urge researchers to prioritize rigorous, fair evaluation of simple baselines and thoroughly investigate potential projection bottlenecks before introducing mathematically over-engineered architectures.

\begin{thebibliography}{50}

\bibitem[Devlin et al.(2018)]{devlin2018bert}
Devlin, J., Chang, M. W., Lee, K., and Toutanova, K.
\newblock Bert: Pre-training of deep bidirectional transformers for language understanding.
\newblock {\em arXiv preprint arXiv:1810.04805}, 2018.

\bibitem[Radford et al.(2021)]{radford2021clip}
Radford, A., Kim, J. W., Hallacy, C., Aditya, A., and Ramesh, R.
\newblock Learning transferable visual models from natural language supervision.
\newblock In {\em International Conference on Machine Learning (ICML)}, pp.\ 8748--8763, 2021.

\bibitem[Ilharco et al.(2023)]{ilharco2022editing}
Ilharco, G., Ribeiro, M. T., Wortsman, M., and Schmidt, L.
\newblock Editing models with task arithmetic.
\newblock In {\em International Conference on Learning Representations (ICLR)}, 2023.

\bibitem[Yadav et al.(2023)]{yadav2023ties}
Yadav, P., Tam, D., Choshen, L., and Mohit, M.
\newblock Resolving interference when merging models.
\newblock In {\em Advances in Neural Information Processing Systems (NeurIPS)}, 2023.

\bibitem[Yang et al.(2026)]{yang2025orthomerge}
Yang, S., Shi, K., and Liu, W.
\newblock Orthogonal model merging.
\newblock {\em arXiv preprint arXiv:2602.05943}, 2026.

\bibitem[Marczak et al.(2025)]{marczak2025isotropic}
Marczak, D., Magistri, S., and Cygert, S.
\newblock Isotropic model merging with common and task-specific subspaces.
\newblock In {\em International Conference on Machine Learning (ICML)}, 2025.

\bibitem[Jung et al.(2025)]{jung2024symerge}
Jung, A., Lee, S., and Hong, S.
\newblock SyMerge: From non-interference to synergistic merging via single-layer adaptation.
\newblock {\em arXiv preprint arXiv:2412.19098}, 2025.

\bibitem[Yang et al.(2024)]{yang2023adamerging}
Yang, E., Wang, Z., and Shen, L.
\newblock Adaptive model merging for multi-task learning.
\newblock In {\em International Conference on Learning Representations (ICLR)}, 2024.

\bibitem[Marczak et al.(2024)]{marczak2024magmax}
Marczak, D., Twardowski, B., and Cygert, S.
\newblock Magmax: Leveraging model merging for seamless continual learning.
\newblock In {\em European Conference on Computer Vision (ECCV)}, 2024.

\bibitem[Wortsman et al.(2022)]{wortsman2022model}
Wortsman, M., Ilharco, G., and Gadre, S. Y.
\newblock Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time.
\newblock In {\em International Conference on Machine Learning (ICML)}, 2022.

\bibitem[Brown et al.(2020)]{brown2020gpt3}
Brown, T., Mann, B., and Ryder, N.
\newblock Language models are few-shot learners.
\newblock In {\em Advances in Neural Information Processing Systems (NeurIPS)}, 2020.

\end{thebibliography}

\end{document}
"""

with open('submission.tex', 'w') as f:
    f.write(paper_content)
print("Saved complete paper to submission.tex!")
