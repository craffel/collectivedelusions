# Peer Review Report: Soundness and Methodology Evaluation

## 1. Clarity of the Description
The methodology of this paper is exceptionally well-described, mathematically precise, and easy to follow. 
- **Mathematical Formulations:** All equations are clearly laid out. Equation 1 (Task Vectors), Equation 2 (Layer-wise weight merging), Equation 3 (GT-Merge), Equation 4 (Poly-Val-Merge), and Equation 7 (Cross-entropy validation loss) are mathematically sound and standard within the model-merging literature.
- **Problem Setup:** The paper does a superb job of distinguishing between the information-access regimes of supervised few-shot validation tuning (OFS-Tune) and unsupervised zero-shot test-time adaptation (online TTA). Rather than glossing over this difference, the authors address it directly as a pragmatic real-world trade-off, clarifying that in most software engineering deployments, a tiny labeled validation set is trivial to obtain.
- **Algorithms:** Algorithm 1 clearly outlines the step-by-step procedure of OFS-Tune, facilitating rapid understanding and implementation.

---

## 2. Appropriateness of Methods
The methods employed are highly appropriate for the questions the paper aims to answer:
- **Low-Dimensional Search Spaces:** Choosing global task-wise (GT-Merge) and polynomial coefficient profiles (Poly-Val-Merge) is a brilliant, mathematically elegant way to reduce the capacity of the parameter search space. This directly addresses the risk of validation overfitting when data is scarce ($M \le 10$).
- **Optimizers:** Using Scipy's **Nelder-Mead** for low-dimensional, derivative-free optimization is standard and appropriate, as it avoids gradient computations. Extending the optimization to **PyTorch Adam** for high-dimensional task spaces ($K \ge 16$) is equally appropriate, as derivative-free methods suffer from catastrophic dimensionality bottlenecks.
- **Continuous Simulation Landscape:** Utilizing a calibrated continuous simulation environment is a powerful methodological tool. It allows the authors to perform exhaustive multi-seed sweeps (30 random seeds, 5 optimization methods, 4 validation sizes, 3 adversarial conditions) that would be computationally prohibitive on physical networks.
- **Physical CNN Validation:** Running a physical, real-world experiment on a 5-layer CNN on real images (MNIST/FMNIST) is the perfect method to validate the continuous simulation's hypotheses. It ensures the findings are not merely artifacts of the simulation landscape.

---

## 3. Detailed Soundness Evaluation and Potential Flaws
The technical soundness of the paper is **excellent**. 

### Strengths in Soundness:
- **Calibration Verification:** The authors show that when they remove simulation noise and landscape non-convexity (creating "sterile" conditions), they can successfully replicate the SOTA claims of online AdaMerging and RegCalMerge. This confirms that their simulation is mathematically sound and has high fidelity.
- **Empirical Confirmation of Rugged Landscapes:** In Section 5.3, the authors sweep and plot the actual prediction entropy landscape of their physical CNN (Figure 3), demonstrating that it is indeed highly non-convex, rugged, and full of sharp local minima. This provides an incredibly strong, empirical justification for the high-frequency cosine wave surrogate used in their simulation (Equation 9).
- **The Overfitting-Optimizer Paradox:** The comparison of Nelder-Mead vs. PyTorch Adam on the 48-D layer-wise space (Table 4) is a masterclass in soundness. It exposes that Nelder-Mead's apparent resistance to overfitting is simply a failure to optimize (stalling near the uniform baseline), whereas PyTorch Adam successfully minimizes validation loss but causes severe test accuracy degradation ($80.78\%$). This cleanly decouples optimization failure from generalization failure.
- **Mitigation Evaluation:** The authors evaluate standard online stabilizers (Cosine decay and EMA smoothing) under noise, showing they still fail to recover online TTA performance, which solidifies the claim of TTA's fundamental fragility.

### Potential Minor Flaws/Limitations:
- **Scale of the Physical Network:** The physical validation is conducted on a 5-layer CNN on MNIST and FashionMNIST, which is relatively small and uses simple datasets. However, the authors explicitly and transparently acknowledge this as a boundary limitation in Section 6.1 and Appendix F.1. They explain that running full ViT-B/32 or LLM merging sweeps was computationally infeasible in their execution environment, and they discuss how the overparameterized nature of larger models would mathematically amplify the overfitting-optimizer paradox. This level of intellectual honesty and scientific transparency is highly commendable.
- **Heavily Tuned Online Baselines:** To ensure a fair comparison, the authors perform extensive hyperparameter sweeps for the online TTA baselines, confirming that the reported baselines represent their optimal configurations. This prevents any accusation of evaluating unoptimized baselines.

---

## 4. Reproducibility
The reproducibility of the work is **highly outstanding**.
- The paper and appendices provide exact mathematical formulations, network structures, optimizer parameters (iterations, step sizes, tolerances, learning rates), and dataset splits.
- The continuous simulation setup is fully described by equations and parameters.
- The physical validation details (number of samples, training epochs, optimizer, learning rates, layers, functional API usage) are exceptionally detailed.
- Anyone with standard scientific computing tools (NumPy, SciPy, PyTorch) could easily reproduce both the simulated and physical experiments from the provided text.
