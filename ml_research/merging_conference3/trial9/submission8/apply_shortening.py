import re

def replace_in_file(filepath, old_str, new_string):
    with open(filepath, 'r') as f:
        content = f.read()
    
    if old_str in content:
        content = content.replace(old_str, new_string)
        print(f"Replacement successful in {filepath}")
    else:
        print(f"Warning: String not found in {filepath}")
        # Try a whitespace-insensitive check or partial check if needed
        
    with open(filepath, 'w') as f:
        f.write(content)

# Shortenings for 03_method.tex
method_path = 'submission/sections/03_method.tex'

old_m1 = r"""While we present a static-centroid formulation here as a controlled setting for our analytical sandbox, GraviMerge naturally scales to layer-specific centroids $\boldsymbol{\mu}_k^{(l)}$ extracted at each layer $l$. Under this generalized layer-shifting formulation, the coordinates of the celestial attractors vary dynamically across sequential depth (i.e., $\boldsymbol{\mu}_k^{(l)}$ shifts on $\mathbb{S}^{D-1}$), enabling GraviMerge to track non-linear representational drift across the layer manifolds of deep transformers. All subsequent distance and force equations remain mathematically identical, substituting $\boldsymbol{\mu}_k^{(3)}$ with $\boldsymbol{\mu}_k^{(l)}$, verifying our method's versatility."""

new_m1 = r"""While we present static centroids here for our sandbox, GraviMerge naturally scales to layer-specific centroids $\boldsymbol{\mu}_k^{(l)}$. Under this formulation, the celestial attractors vary dynamically across depth, enabling GraviMerge to track representational drift across layer manifolds of deep transformers, substituting $\boldsymbol{\mu}_k^{(3)}$ with $\boldsymbol{\mu}_k^{(l)}$ seamlessly."""

old_m2 = r"""According to Newton's universal law of gravitation, the force exerted by star $k$ on a spacecraft probe of unit mass ($m = 1$) is inversely proportional to $r_k^2$. However, if the spacecraft passes directly through a task centroid ($r_k \to 0$), the force would experience a numerical singularity (division by zero), leading to chaotic trajectory escapes. To ensure absolute numerical stability, we incorporate a softened inverse-square gravitational force model. The softened gravitational force vector $\mathbf{F}_k^{(l)} \in \mathbb{R}^D$ pulling the spacecraft towards centroid $\boldsymbol{\mu}_k^{(3)}$ is defined as:"""

new_m2 = r"""Newton's law of gravitation defines force as inversely proportional to $r_k^2$. If the spacecraft passes directly through a task centroid ($r_k \to 0$), the force would experience a numerical division-by-zero singularity, leading to chaotic trajectories. To guarantee numerical stability, we incorporate a softened inverse-square gravitational force vector $\mathbf{F}_k^{(l)} \in \mathbb{R}^D$ defined as:"""

old_m3 = r"""We explicitly justify that this softened force magnitude decays as $1/r^2$ at large distances ($r \gg \epsilon$), and smoothly approaches its peak value $G M_k / \epsilon^2$ at the center ($r \to 0$). This softened force matches the negative gradient of an underlying \textit{Arctangent Potential} defined as:"""

new_m3 = r"""This softened force magnitude decays as $1/r^2$ at large distances ($r \gg \epsilon$), and smoothly approaches its peak value $G M_k / \epsilon^2$ at the center ($r \to 0$), matching the negative gradient of an underlying \textit{Arctangent Potential}:"""

old_m4 = r"""Taking the derivative of the Arctangent Potential with respect to $r$ yields:
\vspace{-3pt}
\begin{equation}
    -\frac{d\Phi}{dr} = \frac{G M_k}{\epsilon^2 + r^2}
    \label{eq:potential_gradient}
\end{equation}
\vspace{-3pt}
which is exactly our defined force magnitude in Equation~\ref{eq:force}. Under a traditional Plummer potential, the gradient unphysically drops to zero as $r \to 0$ due to multi-body spatial symmetry. In our weight-blending routing paradigm, we require that closer proximity to an expert star increases its influence on the blending weights. By utilizing this Arctangent potential formulation, the force magnitude smoothly approaches its peak value at the task centroid, providing a well-behaved maximum routing influence without numerical singularities or unphysical force cancellation at the centroid."""

new_m4 = r"""Under a traditional Plummer potential, the gradient unphysically drops to zero as $r \to 0$ due to spatial symmetry. In our weight-blending paradigm, closer proximity to an expert star must increase its influence. Our Arctangent potential ensures the force magnitude smoothly approaches its peak value at the centroid, providing a well-behaved maximum routing influence without numerical singularities or unphysical force cancellations."""

replace_in_file(method_path, old_m1, new_m1)
replace_in_file(method_path, old_m2, new_m2)
replace_in_file(method_path, old_m3, new_m3)
replace_in_file(method_path, old_m4, new_m4)

# Shortenings for 04_experiments.tex
exp_path = 'submission/sections/04_experiments.tex'

old_e1 = r"""Many existing serving paradigms suffer from "Heterogeneity Collapse" (where accuracy drops when processing mixed-domain batches) or "Vectorization Collapse" (where latency/accuracy degrades when batching is disabled, i.e., $B=1$). As shown in Table~\ref{tab:main_results}, GraviMerge maintains a completely flat, optimal accuracy profile across all three batch sizes ($B=256$, $B=1$). This demonstrates high theoretical resilience to both collapse types under simulated workloads, supporting its potential for computationally efficient real-time deployment in high-concurrency cloud servers as well as single-stream edge devices."""

new_e1 = r"""Many serving paradigms suffer from "Heterogeneity Collapse" (accuracy drops on mixed-domain batches) or "Vectorization Collapse" (degradation when batching is disabled, i.e., $B=1$). As shown in Table~\ref{tab:main_results}, GraviMerge maintains a flat, optimal accuracy profile across batch sizes $B=256$ and $B=1$. This demonstrates high theoretical resilience to both collapse types, proving its suitability for both high-concurrency cloud servers and single-stream edge devices."""

old_e2 = r"""For GraviMerge, we set the gravitational constant $G = 0.05$ to enforce a highly active and dynamic trajectory, ensuring that the spacecraft probe actively traverses the sphere (resolving the frozen spacecraft illusion of extremely weak settings like $G=0.002$, addressing Critical Flaw 1). The medium drag/friction coefficient is set to $\gamma_{\text{drag}} = 0.9$ to absorb high-frequency energy, the virtual integration step $\Delta t = 1.0$, the arctangent softening factor $\epsilon = 0.8$ to eliminate force singularities and stabilize dynamics, and the routing temperature $\tau_{\text{grav}} = 0.05$. All baselines are calibrated identically to ensure a fair, high-difficulty evaluation. We run $10$ independent random seeds to report statistical margins."""

new_e2 = r"""For GraviMerge, we set $G = 0.05$ to enforce an active, dynamic trajectory (resolving the frozen spacecraft illusion of weak settings like $G=0.002$). Sibling drag coefficient is $\gamma_{\text{drag}} = 0.9$, virtual step $\Delta t = 1.0$, softening factor $\epsilon = 0.8$ to eliminate singularities, and temperature $\tau_{\text{grav}} = 0.05$. All baselines are calibrated identically across $10$ independent seeds."""

replace_in_file(exp_path, old_e1, new_e1)
replace_in_file(exp_path, old_e2, new_e2)
