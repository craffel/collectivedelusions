# Intermediate Review Step 4: Experimental Evaluation

## Evaluation of the Experimental Setup and Datasets
The authors deserve significant credit for their **exemplary scientific transparency and honesty**. They prominently include a "CRITICAL SCIENTIFIC DISCLOSURE" box at the beginning of Section 4, explicitly declaring that:
1. All results on MNIST, Fashion-MNIST, CIFAR-10, and SVHN are **entirely simulated** inside the Analytical Coordinate Sandbox (ICS) using synthetic, orthogonal coordinate blocks. No actual neural networks or adapters were trained on image pixels.
2. The real-world pre-trained model validation on Vision Transformers ($\text{ViT-B/16}$) is a **routing-only simulation** conducted strictly on offline, frozen activation features, without actual adapter loading or activation blending.

While this transparency is highly commendable, the reliance on simulated environments and routing-only offline validation represents a major limitation of the paper:
- **Synthetic ICS Sandbox:** Although the sandbox is a mathematically tractable, high-fidelity environment designed to isolate ensembling and routing mechanics (Appendix A), it does not capture the complex, non-orthogonal, and highly overlapping task manifolds of real multi-task learning. Deep neural network representations are rarely perfectly orthogonal.
- **Routing-Only ViT Validation:** By evaluating only the routing trajectories on offline activations, the paper fails to prove that ChemMerge's blending mechanism (Catalytic Activation Blending, CAB) actually preserves or improves final model performance when executing actual adapted paths (like LoRAs) end-to-end on real-world multi-task benchmarks (such as VTAB for vision or GLUE for natural language).

---

## Baselines Coverage
The paper compares ChemMerge against an excellent suite of baselines:
- **Static Weight-Space Merging:** Uniform Merging.
- **Parametric / Dynamic Gating:** Linear Router and QWS-Merge.
- **Systems-Level Scheduling:** PFSR + MBH.
- **State-of-the-Art Stateless Dynamic Ensembling:** SABLE and SPS-ZCA.

To isolate the benefits of their state-dependent kinetics, the authors also compare against a series of **Static EMA Routing baselines** (SABLE + Static EMA with $\beta \in [0.1, 0.9]$). This is an exceptionally thorough ablation that demonstrates why static low-pass filtering is insufficient:
- At high smoothing ($\beta = 0.1$), the static EMA introduces severe representational lag, which decays final classification accuracy by **-4.2%** because it cannot adapt fast enough to discriminative deep features.
- At low smoothing ($\beta = 0.9$), routing jitter remains extremely high.
- ChemMerge simultaneously achieves both the highest accuracy and low jitter by allowing the smoothing rate to adapt dynamically based on catalytic similarity.

---

## Alignment of Results with Claims

The empirical results strongly support the paper's core claims:
1. **Jitter Reduction:** On the real pre-trained $\text{ViT-B/16}$, ChemMerge reduces layer-to-layer ensembling weight routing jitter by **9.9$\times$** compared to SPS-ZCA and by over **2.15$\times$** compared to SABLE (under identical sensitivities). This is a highly significant and visually compelling result (Figure 10).
2. **Robustness to Collapse:** ChemMerge achieves flat, stable accuracy across all streaming configurations in the sandbox (Table 1 and Figure 1b). It successfully avoids both *Heterogeneity Collapse* (where Uniform Merging fails) and *Vectorization Collapse* (where QWS-Merge drops to 34.58%).
3. **Computational Efficiency and Scaling:** The scaling analysis up to $K=16$ experts (Table 2) demonstrates that ChemMerge's vectorized numpy implementation scales remarkably well, executing routing updates in 19.9ms (which is **42.1%** faster than SABLE and **49.4%** faster than SPS-ZCA), confirming that its parallel formulation is hardware-friendly.

### Critical Gaps and Limitations
- **Statistical Insignificance of Accuracy Improvements:** On the pre-trained $\text{ViT-B/16}$ routing-only simulation, ChemMerge's routing accuracy is $93.20\% \pm 0.75\%$, which is within one standard deviation of SABLE ($93.00\% \pm 0.60\%$) and SPS-ZCA ($92.80\% \pm 0.80\%$). The absolute classification accuracies are statistically comparable. Thus, ChemMerge's main victory is **not** a massive boost in accuracy, but rather its immense physical stability (jitter reduction). The authors are honest about this, but it must be highlighted.
- **Lack of Physical Hardware Instrumentation:** The latency benchmarks are CPU-bound NumPy evaluations. While they demonstrate vectorized computational efficiency, they do not capture the physical constraints of heterogeneous edge serving hardware (such as NPUs or low-power embedded GPUs), where memory bandwidth, cache capacity, and energy budgets dominate. Oscilloscope-based power profiling and NPU-specific latency measurements are needed to fully validate edge viability.
- **Active Representation Coupling Limitations:** The authors honestly report that the active coupling mechanism ($\eta > 0.0$) degrades accuracy under highly heterogeneous serving due to cascading representational drift. Setting $\eta = 0.0$ (no coupling) is the most robust default, which somewhat undermines the continuous-time dynamical coupling story.
