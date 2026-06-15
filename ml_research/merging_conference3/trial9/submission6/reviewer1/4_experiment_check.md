# 4. Experiment Check

## Critical Evaluation of the Experimental Setup
1. **The Analytical Coordinate Sandbox**: 
   - The sandbox uses a 14-layer projection stack ($D=192$, $d=8$) to evaluate ensembling under mutual orthogonality and overlap. 
   - While it serves as a controlled environment to study eigenvalue decay, it is highly artificial. In Section 4.1, the authors state that under overlapping manifolds (overlap=12), the task subspaces (which have rank $d=8$) "share 12 of their representation coordinates." Mathematically, it is unclear how two 8-dimensional subspaces can "share 12 coordinates" unless this refers to the coordinates of the ambient $D=192$ space, which is not clearly defined.
2. **The "Simulated GLUE LoRA Benchmark"**: 
   - This simulation is configured to match the scale of RoBERTa-Large ($D=1024$, $r=8$, $L=8$ layers). 
   - However, to demonstrate the "catastrophic coordinate collapse" of flat methods, the authors propagate features through 8 sequential projection layers **without** any residual connections or LayerNorm. This is a highly biased setup: real RoBERTa-Large architectures have residual connections and LayerNorm at every single layer, which are known to act as geometric buffers. By omitting these essential components from their "high-fidelity simulation," the authors create an artificial scenario where flat methods collapse to random guessing ($55.0\%$), allowing C-Lie-MM to show an inflated $+42.0\%$ improvement. In a real, physical RoBERTa-Large model with standard residual paths, SABLE and other parameter-merging methods would perform significantly better, and the actual performance gap would be much smaller.

## Baseline Comparison
- The authors compare their method against a solid set of 11 baselines in the sandbox and 4 standard merging methods (Task Arithmetic, TIES-Merging, SABLE, ZipIt) in the GLUE simulation. 
- However, the comparison reveals that **simpler, flat baselines with optimized routing temperatures are highly competitive**. In the overlapping manifold setting, the flat Temp-Only ERM (UN-PCA) and SABLE (UN-PCA) baselines with optimized temperatures achieve **$70.00\%$** accuracy, which is virtually identical to C-Lie-MM's **$70.30\%$** accuracy. 
- The authors justify their method's complexity by showing that flat models collapse their routing entropy ($\tau \to 0$), whereas C-Lie-MM maintains high, soft routing entropy ($H/H_{\max} \approx 0.90$). While this is mathematically true, the fact remains that soft-cooperative ensembling yields only a **$+0.30\%$** improvement in joint accuracy over the hard-gating flat model. For most practitioners, the extreme engineering and computational complexity of C-Lie-MM cannot be justified for a $0.30\%$ accuracy gain.

## Supporting the Claims
- **Do the results support the claim that flat linear blending causes eigenvalue shrinkage?** Yes, the theoretical proofs and sandbox results confirm this.
- **Do the results support the claim that C-Lie-MM prevents coordinate collapse?** Yes, by strictly enforcing Grassmannian manifold constraints ($\Delta_{\text{idem}} \approx 10^{-7}$), C-Lie-MM prevents norm decay.
- **Do the results support the claim that C-Lie-MM is a superior, practical alternative for real-world model merging?** Only partially. The sandbox ablation study in Section 4.3 (with residuals and LayerNorm) shows that when real-world safeguards are present, the benefit of C-Lie-MM over SABLE shrinks to $+7.7\%$. In addition, the GLUE simulation is artificially stripped of residual paths, meaning its massive $+42\%$ improvement is highly exaggerated and does not reflect actual transformer performance.
