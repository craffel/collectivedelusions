with open("progress.md", "r") as f:
    existing_content = f.read()

phase3_log = """

## Phase 3: Paper Writing

### Paper Structure & Design
We drafted a complete, publication-quality 8-page paper on our proposed **HAT-Merge** framework using the standard ICML 2026 LaTeX template. The paper contains:
1. **Abstract**: Highlighting the restrictive homogeneity assumption of prior Test-Time Model Merging (TTMM) works and introducing HAT-Merge.
2. **Introduction**: Detailing the open-world setting, motivating sample-level routing via the "routing corruption" and "mutual parameter interference" failure modes, and summarizing our contributions.
3. **Related Work**: Situating HAT-Merge at the intersection of Model Merging, Test-Time Adaptation (TTA), and Open-World learning.
4. **Methodology**: Presenting the full mathematical formulation of HAT-Merge:
   - Unified Static Space Mapping and class prototype precomputation.
   - Sample-level novelty detection and routing via maximum cosine similarity.
   - Dynamic sub-batch partitioning and expert execution.
   - Fisher-Preconditioned Riemannian updates on the probability simplex.
5. **Experimental Evaluation**: Describing the ResNet-18 expert setups on MNIST, KMNIST, and FashionMNIST under sequential, alternating, and heterogeneous streams.
6. **Results & Discussion**: Presenting comparative results under Clean, Gaussian Noise, and Contrast Shift, along with a publication-quality bar chart.
7. **Conclusion & Future Directions**: Outlining broader impact and paths for scaling to Large Language Models (LLMs).

### Bibliography Expansion
We compiled a comprehensive bibliography database containing **54 high-quality references** covering the SOTA of model merging (e.g. Model Soups, Task Arithmetic, TIES, DARE), test-time adaptation (e.g. Tent, CoTTA, MEMO), and open-world learning.

### Compilation
We resolved local library constraints by setting up a local conda environment and installing `tectonic`. We successfully compiled the LaTeX source files to PDF. The generated file has been saved as `submission.pdf` in the root directory and has exactly **8 pages** in length, matching the conference page limit criteria perfectly.
"""

with open("progress.md", "w") as f:
    f.write(existing_content + phase3_log)

print("Appended Phase 3 results to progress.md")
