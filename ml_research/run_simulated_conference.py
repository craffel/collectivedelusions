#!/usr/bin/env python3
"""
Run a simulated conference (research -> review -> metareview), iterated for N trials.

For each trial:
  - 10 submissionZ subdirs are created under trialN/, each seeded with reference
    files from skeleton/ (plans, templates, etc.). Runtime files (sandbox_run.sh,
    bin/, bwrap, format_log.jq, run_*.slurm) are deliberately NOT copied so the
    agent cannot edit the wrappers from inside its sandbox.
  - Trial 1: submissionZ/papers gets the 3 supplied seed PDFs.
  - Trial N>1: submissionZ/papers gets the previous trial's accepted PDFs.
  - 10 research agents run in parallel on hopper-cpu controllers.
  - Reviewing agents run for each submission that produced submission.pdf.
  - One metareview agent runs at the trial root and produces accepted_papers/.

While slurm jobs are in flight a background monitor thread polls every
POLL_INTERVAL_SECONDS and logs notable events: progress.md updates, GPU job
launches/conclusions, controller iteration markers, submission/review file
appearance.

Usage:
    run_simulated_conference.py <output_dir> <num_trials> <seed1.pdf> <seed2.pdf> <seed3.pdf>
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
SKELETON_DIR = REPO_DIR / "skeleton"
NUM_SUBMISSIONS = 10
POLL_INTERVAL_SECONDS = 30

# Items in skeleton/ that must NOT land inside an agent's CWD. (Keeping them
# outside $WORK is what makes the slurm wrappers tamper-proof — see
# research_plan.md and skeleton/sandbox_run.sh.)
RUNTIME_NAMES = {
    "bwrap",
    "sandbox_run.sh",
    "format_log.jq",
    "bin",
    "run_research_agent.slurm",
    "run_reviewing_agent.slurm",
    "run_metareview_agent.slurm",
}


# --- logging --------------------------------------------------------------

_log_lock = threading.Lock()


def log(msg: str) -> None:
    with _log_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --- file staging ---------------------------------------------------------

def seed_reference_files(dest: Path) -> None:
    """Copy reference files from skeleton/ into dest (excluding runtime files)."""
    dest.mkdir(parents=True, exist_ok=True)
    for entry in SKELETON_DIR.iterdir():
        if entry.name in RUNTIME_NAMES:
            continue
        target = dest / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target, dirs_exist_ok=True, symlinks=False)
        else:
            shutil.copy2(entry, target)


# --- slurm helpers --------------------------------------------------------

def submit_with_wait(workdir: Path, jobname: str, slurm_script: Path) -> subprocess.Popen:
    """Submit a slurm job with `sbatch --wait`; return its Popen.

    The first line of `<workdir>/<jobname>.sbatch.log` is the "Submitted batch job N"
    notice that we later scrape to recover the controller's slurm job id.
    """
    log_path = workdir / f"{jobname}.sbatch.log"
    cmd = [
        "sbatch", "--wait",
        f"--chdir={workdir}",
        f"--job-name={jobname}",
        f"--export=ALL,SKELETON_DIR={SKELETON_DIR}",
        str(slurm_script),
    ]
    fp = open(log_path, "w")
    return subprocess.Popen(cmd, stdout=fp, stderr=subprocess.STDOUT)


def read_jobid(sbatch_log: Path) -> str | None:
    if not sbatch_log.exists():
        return None
    try:
        text = sbatch_log.read_text(errors="ignore")
    except OSError:
        return None
    m = re.search(r"Submitted batch job (\d+)", text)
    return m.group(1) if m else None


def squeue_me() -> list[tuple[str, str, str, str]]:
    """Return [(jobid, comment, state, jobname), ...] for current user's jobs."""
    try:
        out = subprocess.run(
            ["squeue", "--me", "-h", "-o", "%i|%k|%t|%j"],
            capture_output=True, text=True, timeout=20, check=False,
        ).stdout
    except subprocess.TimeoutExpired:
        return []
    rows: list[tuple[str, str, str, str]] = []
    for line in out.splitlines():
        parts = line.split("|", 3)
        if len(parts) == 4:
            rows.append((parts[0], parts[1], parts[2], parts[3]))
    return rows


def sacct_final_state(jobid: str) -> str:
    try:
        out = subprocess.run(
            ["sacct", "-X", "-n", "-j", jobid, "-o", "State"],
            capture_output=True, text=True, timeout=20, check=False,
        ).stdout
    except subprocess.TimeoutExpired:
        return "UNKNOWN"
    line = out.strip().splitlines()[0] if out.strip() else ""
    return line.strip() or "UNKNOWN"


# --- per-submission state -------------------------------------------------

@dataclass
class SubmissionState:
    label: str
    workdir: Path
    sbatch_log: Path  # path to the orchestrator's sbatch.log for this controller
    controller_jobid: str | None = None
    progress_mtime: float = 0.0
    iterations_seen: int = 0
    submission_pdf_seen: bool = False
    review_md_seen: bool = False
    accepted_dir_seen: bool = False
    gpu_jobs_active: set[str] = field(default_factory=set)
    gpu_jobs_done: set[str] = field(default_factory=set)


# --- monitor thread -------------------------------------------------------

class Monitor(threading.Thread):
    """Background polling thread that emits events for one phase of one trial."""

    def __init__(self, submissions: list[SubmissionState], stop_event: threading.Event):
        super().__init__(daemon=True, name="trial-monitor")
        self.submissions = submissions
        self.stop_event = stop_event

    def run(self) -> None:
        time.sleep(5)  # let sbatch logs land
        while True:
            try:
                self._poll()
            except Exception as e:  # never let the monitor die
                log(f"[monitor] error: {e}")
            if self.stop_event.wait(POLL_INTERVAL_SECONDS):
                return

    def _poll(self) -> None:
        self._refresh_jobids()
        for s in self.submissions:
            self._poll_progress(s)
            self._poll_artifacts(s)
            self._poll_iterations(s)
        self._poll_gpu_jobs()

    def _refresh_jobids(self) -> None:
        for s in self.submissions:
            if s.controller_jobid:
                continue
            jid = read_jobid(s.sbatch_log)
            if jid:
                s.controller_jobid = jid
                log(f"[{s.label}] controller slurm job id = {jid}")

    def _poll_progress(self, s: SubmissionState) -> None:
        p = s.workdir / "progress.md"
        try:
            st = p.stat()
        except FileNotFoundError:
            return
        if st.st_mtime > s.progress_mtime + 1.0:
            s.progress_mtime = st.st_mtime
            log(f"[{s.label}] progress.md updated (size={st.st_size}B)")

    def _poll_artifacts(self, s: SubmissionState) -> None:
        if not s.submission_pdf_seen and (s.workdir / "submission.pdf").exists():
            s.submission_pdf_seen = True
            size = (s.workdir / "submission.pdf").stat().st_size
            log(f"[{s.label}] submission.pdf appeared ({size}B)")
        if not s.review_md_seen and (s.workdir / "review.md").exists():
            s.review_md_seen = True
            size = (s.workdir / "review.md").stat().st_size
            log(f"[{s.label}] review.md appeared ({size}B)")
        accepted = s.workdir / "accepted_papers"
        if not s.accepted_dir_seen and accepted.is_dir():
            s.accepted_dir_seen = True
            pdfs = list(accepted.glob("*.pdf"))
            log(f"[{s.label}] accepted_papers/ created with {len(pdfs)} PDF(s)")

    def _poll_iterations(self, s: SubmissionState) -> None:
        if not s.controller_jobid:
            return
        candidates = list(s.workdir.glob(f"*_{s.controller_jobid}.out"))
        if not candidates:
            return
        try:
            text = candidates[0].read_text(errors="ignore")
        except OSError:
            return
        sleeps = text.count("Sleeping for 10 minutes")
        if sleeps > s.iterations_seen:
            s.iterations_seen = sleeps
            log(f"[{s.label}] completed iteration #{sleeps} (claude invocation finished)")

    def _poll_gpu_jobs(self) -> None:
        rows = squeue_me()
        ctrl_map: dict[str, SubmissionState] = {
            s.controller_jobid: s for s in self.submissions if s.controller_jobid
        }
        active_now: dict[str, set[str]] = {s.label: set() for s in self.submissions}
        for jobid, comment, state, jobname in rows:
            if not comment.startswith("agent-"):
                continue
            ctrl = comment[len("agent-"):]
            s = ctrl_map.get(ctrl)
            if s is None:
                continue
            if jobid in s.gpu_jobs_done:
                continue
            active_now[s.label].add(jobid)
            if jobid not in s.gpu_jobs_active:
                s.gpu_jobs_active.add(jobid)
                log(f"[{s.label}] launched GPU job {jobid} (name={jobname}, state={state})")
        # Disappearance from squeue = job concluded.
        for s in self.submissions:
            concluded = s.gpu_jobs_active - active_now[s.label]
            for jobid in concluded:
                final = sacct_final_state(jobid)
                log(f"[{s.label}] GPU job {jobid} concluded ({final})")
                s.gpu_jobs_active.discard(jobid)
                s.gpu_jobs_done.add(jobid)


# --- phase runners --------------------------------------------------------

def run_phase(
    phase_label: str,
    submissions_and_procs: list[tuple[SubmissionState, subprocess.Popen]],
) -> None:
    """Wait for all sbatch --wait procs in this phase; run a Monitor concurrently."""
    states = [s for s, _ in submissions_and_procs]
    stop = threading.Event()
    monitor = Monitor(states, stop)
    monitor.start()
    for s, p in submissions_and_procs:
        rc = p.wait()
        if rc != 0:
            log(f"[{s.label}] sbatch --wait exited rc={rc}")
    stop.set()
    monitor.join(timeout=10)
    log(f"===== {phase_label}: done =====")


def run_trial(trial: int, output_dir: Path, input_papers: list[Path]) -> Path:
    trial_dir = output_dir / f"trial{trial}"
    log(f"===== trial {trial}: setting up {trial_dir} =====")
    trial_dir.mkdir(parents=True, exist_ok=True)
    seed_reference_files(trial_dir)  # so metareview can run at trial root

    for z in range(1, NUM_SUBMISSIONS + 1):
        sub_dir = trial_dir / f"submission{z}"
        sub_dir.mkdir(exist_ok=True)
        seed_reference_files(sub_dir)
        papers_dir = sub_dir / "papers"
        papers_dir.mkdir(exist_ok=True)
        for p in input_papers:
            shutil.copy2(p, papers_dir / p.name)
    log(f"trial {trial}: seeded {NUM_SUBMISSIONS} submissions with {len(input_papers)} input paper(s)")

    # ----- Phase 1: research -----
    log(f"===== trial {trial}: launching {NUM_SUBMISSIONS} research agents =====")
    research: list[tuple[SubmissionState, subprocess.Popen]] = []
    for z in range(1, NUM_SUBMISSIONS + 1):
        sub_dir = trial_dir / f"submission{z}"
        jobname = f"research-t{trial}-s{z}"
        state = SubmissionState(
            label=f"t{trial}/sub{z}",
            workdir=sub_dir,
            sbatch_log=sub_dir / f"{jobname}.sbatch.log",
        )
        proc = submit_with_wait(sub_dir, jobname, SKELETON_DIR / "run_research_agent.slurm")
        research.append((state, proc))
    run_phase(f"trial {trial} research", research)

    # ----- Phase 2: reviewing -----
    log(f"===== trial {trial}: launching reviewing agents =====")
    review: list[tuple[SubmissionState, subprocess.Popen]] = []
    skipped = 0
    for z in range(1, NUM_SUBMISSIONS + 1):
        sub_dir = trial_dir / f"submission{z}"
        if not (sub_dir / "submission.pdf").exists():
            log(f"[t{trial}/sub{z}] no submission.pdf — skipping review")
            skipped += 1
            continue
        jobname = f"review-t{trial}-s{z}"
        state = SubmissionState(
            label=f"t{trial}/sub{z}-rev",
            workdir=sub_dir,
            sbatch_log=sub_dir / f"{jobname}.sbatch.log",
        )
        proc = submit_with_wait(sub_dir, jobname, SKELETON_DIR / "run_reviewing_agent.slurm")
        review.append((state, proc))
    log(f"trial {trial}: submitted {len(review)} review jobs (skipped {skipped})")
    if review:
        run_phase(f"trial {trial} review", review)

    # ----- Phase 3: metareview -----
    log(f"===== trial {trial}: launching metareview =====")
    jobname = f"metareview-t{trial}"
    meta_state = SubmissionState(
        label=f"t{trial}/meta",
        workdir=trial_dir,
        sbatch_log=trial_dir / f"{jobname}.sbatch.log",
    )
    proc = submit_with_wait(trial_dir, jobname, SKELETON_DIR / "run_metareview_agent.slurm")
    run_phase(f"trial {trial} metareview", [(meta_state, proc)])

    accepted = trial_dir / "accepted_papers"
    if not accepted.is_dir():
        sys.exit(f"trial {trial}: metareview did not produce {accepted}")
    pdfs = sorted(accepted.glob("*.pdf"))
    log(f"===== trial {trial} complete: {len(pdfs)} accepted paper(s) =====")
    return accepted


# --- main -----------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("output_dir", type=Path, help="Where to write trialN/ directories.")
    ap.add_argument("num_trials", type=int, help="Number of trials to run.")
    ap.add_argument("seed_papers", nargs=3, type=Path, help="Three seed PDFs for trial 1.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_trials < 1:
        sys.exit("num_trials must be positive")
    for p in args.seed_papers:
        if not p.is_file():
            sys.exit(f"seed paper not found: {p}")
    if not SKELETON_DIR.is_dir():
        sys.exit(f"skeleton dir not found: {SKELETON_DIR}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_papers = [p.resolve() for p in args.seed_papers]

    log(f"orchestrator starting; output={output_dir}, trials={args.num_trials}")
    for trial in range(1, args.num_trials + 1):
        accepted = run_trial(trial, output_dir, input_papers)
        pdfs = sorted(accepted.glob("*.pdf"))
        if not pdfs:
            sys.exit(f"trial {trial}: no PDFs in {accepted}")
        input_papers = pdfs
    log(f"all {args.num_trials} trial(s) complete: {output_dir}")


if __name__ == "__main__":
    main()
