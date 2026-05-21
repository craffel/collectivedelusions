# Research Agent Operating Plan

## Objective
Execute a complete research cycle—from literature review and idea formulation to experimentation and paper writing—within a 6-hour window.
You (this agent) run on a small CPU node (4 CPUs, 16 GB RAM); GPU work happens in separate slurm jobs you submit yourself (see `## Compute`).
The final paper will be submitted to a conference and reviewed based on the criteria in `reviewing_criteria.md`.
Your goal is to write a paper that is likely to be accepted to the conference.
The final submission PDF *must* be saved as `submission.pdf` in the current directory.
Closely follow the general guidelines on doing effective research provided in `research_tips.md`.

## Compute
This agent does **not** have GPUs attached. To run experiments, submit slurm jobs from this CPU node onto the GPU partition.

### Submitting GPU jobs
Write a slurm script (example below) and submit it with `sbatch your_job.slurm`. Each GPU node is a `p5.48xlarge` with 8x H100, 88 CPUs, and ~1.9 TB RAM.

```bash
#!/bin/bash
#SBATCH --job-name=my-experiment
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=88
#SBATCH --time=2:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

# Your training/eval command here.
python train.py --config configs/run.yaml
```

Submit with `sbatch my_experiment.slurm`; check status with `squeue`; cancel with `scancel <jobid>`. Output streams to `<jobname>_<jobid>.{out,err}` in the working directory by default.

### Slurm restrictions enforced for this agent
`sbatch`, `squeue`, and `scancel` are wrappers in front of the real slurm commands. They enforce:

- **One GPU node at a time.** You can have at most **1** node running or pending at any moment. The wrapper inspects `--nodes`/`-N` on the command line and `#SBATCH --nodes=` in the script. If submitting would exceed the cap, you will see:
  ```
  [sbatch-wrapper] Refusing: agent <ID> has N node(s) running/pending; this would add M (cap=1).
  ```
  and the `sbatch` command exits with code 2 and no job is queued. Wait for your current job to finish (or `scancel` it) before submitting the next one.
- **All jobs you submit are auto-tagged** with `--comment=agent-<your_id>`. Do not pass your own `--comment` — it will be stripped.
- **`squeue` only shows your own tagged jobs.** Other users' jobs and any of your own jobs not submitted through this wrapper will be invisible.
- **`scancel` only acts on your own tagged jobs.** Bulk/filter flags are rejected outright:
  ```
  [scancel-wrapper] Refusing: filter flag '-u' not allowed; pass explicit job IDs only.
  ```
  Cancelling a job you do not own:
  ```
  [scancel-wrapper] Refusing: job <id> is not tagged agent-<your_id> (not owned by agent <your_id>).
  ```
  In both cases the command exits with code 2 and no cancellation happens. Only pass job IDs that came from your own `sbatch` calls.

If you hit one of these errors, the fix is procedural (wait for your current job, cancel only your own jobs), not technical — do not try to work around the wrappers (e.g., calling slurm via absolute paths). They are also installed at `/opt/slurm/bin/`, so PATH tricks will not bypass them.

## Runtime Instructions
This agent is invoked every 10 minutes. On each start:
1. **Check Remaining Time:** Run `squeue -h -j $SLURM_JOB_ID -O TimeLeft` to monitor the 6-hour deadline.
2. **Load State:** Read `progress.md` to determine the current status and the next pending task. If all phases (1-3) are marked as complete, transition to Phase 4.
3. **Execute & Persist:** Perform the next step in the workflow and immediately record progress, experimental results, and any state changes to `progress.md`. Retain all notes in progress.md by appending updates.

## Workflow Phases

### Phase 1: Foundation (Read & Formulate)
- **Input:** Read the three PDF papers located in the `papers/` directory.
- **Synthesis:** Identify general themes, core contributions, limitations, and potential extensions.
- **Literature search:** Search for related papers covering related themes and methods, including background work. Use Google Search or the Semantic Scholar API as described in `semantic_scholar.md`. Use the API key stored in the `SEMANTIC_SCHOLAR_API_KEY` environment variable. Download and read relevant PDFs.
- **Idea Generation:** Formulate ten novel research ideas on the identified theme. Record each idea, with expected results and impact, in `progress.md`.
- **Selection:** Choose one of the ten research ideas based on a value provided by a pseudo-random number generator.
- **Iterate:** If possible, improve upon the novelty, feasibility, and importance of the proposed research idea by reconsidering prior work.
- **Commit:** Record the final chosen project hypothesis and rationale in `progress.md`.

### Phase 2: Experimentation
- **Formulation:** Design a set of experiments to test the hypothesis. Reuse an experimental setting (including datasets, baselines, and models) from past work if possible.
- **Implementation:** Write the necessary Python code, reusing code from past research as much as possible to ensure valid results.
- **Environment:** All code, models, and datasets should be written to the current working directory. Do not access any files outside of /fsx/craffel/collectivedelusions/ml_research/. Create and use a local environment with `uv` to manage any new dependencies.
- **Execution:** Run experiments by submitting slurm jobs to the GPU partition as described in `## Compute` above. You are capped at one GPU node at a time, so plan experiment sweeps as a sequence of submissions rather than as a parallel fan-out. The total time budget (this agent + every GPU job it submits) must fit within the 6-hour controller window.
- **LLM API:** Optionally use the Gemini API with the `GEMINI_API_KEY` environment variable.
- **Analysis:** Collect and analyze results. Save key plots and metrics.

### Phase 3: Paper Writing
- **Template:** Use the LaTeX template in the `template/` directory.
- **Constraints:** Exactly 8 pages for the main paper, plus unlimited pages for references and appendix. Follow the formatting instructions in `template/example_paper.pdf`.
- **Sections:** Include an abstract, introduction, and related work section. Describe the proposed method, the experimental setup, and the results.
- **Visuals:** Include diagrams illustrating key ideas. Create plots to visualize results.
- **References:** Typical papers will have at least 50 references, covering the body of related work in detail.
- **Evaluation:** The paper will be judged by the criteria specified in `reviewing_criteria.md`.
- **Compilation:** Ensure the LaTeX document compiles correctly to a PDF.
- **Submission:** The final submission PDF *must* be saved as `submission.pdf` in the current working directory.

### Phase 4: Iterative Refinement
- **Condition:** If Phases 1-3 are complete and time remains.
- **Context:** Re-read `research_tips.md` and `revewing_criteria.md` to understand what makes a good paper and how to improve the project. Your goal is to improve the chances the paper will be accepted based on the criteria in `reviewing_criteria.md`.
- **Action:** Restart the research cycle to refine and strengthen the work.
- **Refinement Strategy:**
  - **Literature:** Build a larger body of related work and expand the set of baselines.
  - **Core hypothesis:** Reconsider, refine, and expand the main hypothesis.
  - **Theory:** Add or expand theoretical justification if it improves the project's core arguments.
  - **Methodology:** Improve the proposed methodology in light of initial experimental results.
  - **Experimentation:** Expand the range of experimental settings, hyperparameters, and ablation studies.
  - **Writing:** Substantially improve the quality, clarity, and technical depth of the written paper.
- **State Management:** Update `progress.md` to begin again at Phase 1, ensuring that previous insights are preserved and used to inform the next iteration.
- **Requirements:** DO NOT declare the paper "finished" if time remains. There are always ways to improve the paper. Find them and take action.

## Critical Requirements
- **Persistence:** Every action and decision MUST be recorded in `progress.md`. If the agent is interrupted, it must be able to resume exactly where it left off. All updates should be appended to progress.md.
- **Deadlines:** All phases—including final paper compilation—must be completed within the 6-hour Slurm job allocation.
- **Efficiency:** Within a single GPU job, parallelize across the node's 8 H100s. Across jobs, you are limited to one node at a time, so prefer fewer, larger, well-scoped jobs over many small ones.
