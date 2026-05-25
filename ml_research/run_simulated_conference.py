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

hopper-cpu nodes are AWS spot instances, so controllers can be preempted
(exit code 0:10 / NODE_FAIL / PREEMPTED). The orchestrator detects this,
deducts the elapsed time from the remaining budget, cancels any GPU jobs
orphaned by the dead controller, and resubmits the agent until either it
exits cleanly or the budget is exhausted.

A background monitor thread polls every POLL_INTERVAL_SECONDS and logs
notable events across all attempts: progress.md updates, GPU job launches
and conclusions, controller iteration markers, and artifact appearance.

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

# Don't bother resubmitting if less than this many seconds remain after a preemption.
MIN_RESUBMIT_SEC = 600

# Per-submission delay before the first claude invocation. Spaces them out so
# many processes on one node don't all clobber each other writing to ~/.claude.json.
STARTUP_STAGGER_SEC = 15

# Items in skeleton/ that must NOT land inside an agent's CWD. (Keeping them
# outside $WORK is what makes the slurm wrappers tamper-proof.)
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


# --- slurm time parsing ---------------------------------------------------

def parse_slurm_time(spec: str) -> int:
    """Parse a slurm time string to seconds.

    Accepts the formats slurm itself accepts: M, M:S, H:M:S, D-H, D-H:M, D-H:M:S.
    """
    if "-" in spec:
        days_str, rest = spec.split("-", 1)
        days = int(days_str)
    else:
        days, rest = 0, spec
    parts = [int(p) for p in rest.split(":")]
    if len(parts) == 1:
        return days * 86400 + parts[0] * 60
    if len(parts) == 2:
        if days > 0:
            return days * 86400 + parts[0] * 3600 + parts[1] * 60
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return days * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"can't parse slurm time {spec!r}")


def parse_time_limit_seconds(slurm_script: Path) -> int:
    text = slurm_script.read_text()
    m = re.search(r"^#SBATCH\s+(?:--time=|-t\s+)(\S+)", text, re.MULTILINE)
    if not m:
        raise ValueError(f"could not find #SBATCH --time in {slurm_script}")
    return parse_slurm_time(m.group(1))


# --- slurm helpers --------------------------------------------------------

def submit_with_wait(
    workdir: Path,
    jobname: str,
    slurm_script: Path,
    time_limit_minutes: int,
    startup_offset_sec: int = 0,
) -> subprocess.Popen:
    """Submit a slurm job with `sbatch --wait`; return its Popen.

    --time on the CLI overrides the #SBATCH --time directive in the script,
    so we can shrink the budget for retries after a preemption.

    `startup_offset_sec` is plumbed through as the AGENT_STARTUP_OFFSET env
    var; the slurm script sleeps that many seconds before its first claude
    invocation to avoid 10 agents racing for ~/.claude.json on the same node.
    """
    log_path = workdir / f"{jobname}.sbatch.log"
    export = f"ALL,SKELETON_DIR={SKELETON_DIR},AGENT_STARTUP_OFFSET={startup_offset_sec}"
    cmd = [
        "sbatch", "--wait",
        f"--chdir={workdir}",
        f"--job-name={jobname}",
        f"--time={time_limit_minutes}",
        f"--export={export}",
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


def sacct_summary(jobid: str) -> dict[str, object]:
    """Best-effort {state, elapsed_sec, exit_code} for a job from sacct.

    Uses -P (parsable) so the State string is not truncated, e.g. we get the
    full "CANCELLED by 150016" instead of "CANCELLED+".
    """
    try:
        out = subprocess.run(
            ["sacct", "-X", "-n", "-P", "-j", jobid, "-o", "State,ElapsedRaw,ExitCode"],
            capture_output=True, text=True, timeout=20, check=False,
        ).stdout
    except subprocess.TimeoutExpired:
        return {}
    line = out.strip().splitlines()[0] if out.strip() else ""
    parts = line.split("|")
    if len(parts) >= 3:
        elapsed = int(parts[1]) if parts[1].strip().isdigit() else 0
        return {
            "state": parts[0].strip(),
            "elapsed_sec": elapsed,
            "exit_code": parts[2].strip(),
        }
    return {}


def is_preemption(state: str, exit_code: str) -> bool:
    """Detect AWS-spot preemption of a hopper-cpu controller.

    Manifests as ExitCode 0:10 (SIGUSR1 from the slurm prolog/preemption hook)
    or one of the explicit slurm states for involuntary termination.
    """
    if state.startswith(("NODE_FAIL", "PREEMPTED", "BOOT_FAIL")):
        return True
    if exit_code == "0:10":
        return True
    return False


def cancel_orphan_gpu_jobs(controller_jobid: str, label: str) -> None:
    """Cancel any live GPU jobs whose comment ties them to a preempted controller."""
    tag = f"agent-{controller_jobid}"
    try:
        out = subprocess.run(
            ["squeue", "--me", "-h", "-o", "%i|%k|%t"],
            capture_output=True, text=True, timeout=20, check=False,
        ).stdout
    except subprocess.TimeoutExpired:
        return
    to_cancel: list[str] = []
    for line in out.splitlines():
        parts = line.split("|", 2)
        if len(parts) == 3 and parts[1] == tag:
            to_cancel.append(parts[0])
    if not to_cancel:
        return
    log(f"[{label}] cancelling {len(to_cancel)} orphan GPU job(s) from preempted "
        f"controller {controller_jobid}: {','.join(to_cancel)}")
    subprocess.run(["scancel", *to_cancel], capture_output=True, check=False)


# --- per-submission state -------------------------------------------------

@dataclass
class Attempt:
    jobname: str
    sbatch_log: Path
    jobid: str | None = None
    final_state: str = ""
    elapsed_sec: int = 0
    exit_code: str = ""
    rc: int = -1


@dataclass
class SubmissionState:
    label: str
    workdir: Path
    attempts: list[Attempt] = field(default_factory=list)
    progress_mtime: float = 0.0
    iterations_seen: int = 0
    submission_pdf_seen: bool = False
    review_md_seen: bool = False
    accepted_dir_seen: bool = False
    gpu_jobs_active: set[str] = field(default_factory=set)
    gpu_jobs_done: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        # Baseline progress.md mtime now so the seeded copy doesn't read as an update.
        p = self.workdir / "progress.md"
        try:
            self.progress_mtime = p.stat().st_mtime
        except FileNotFoundError:
            pass


# --- managed slot (handles preemption + resubmission) ---------------------

class ManagedSlot:
    """Runs one submission's lifecycle: submit, wait, detect preemption, resubmit."""

    def __init__(
        self,
        state: SubmissionState,
        jobname_base: str,
        slurm_script: Path,
        time_budget_sec: int,
        allow_resubmit: bool = True,
        startup_offset_sec: int = 0,
    ) -> None:
        self.state = state
        self.jobname_base = jobname_base
        self.slurm_script = slurm_script
        self.remaining_sec = time_budget_sec
        self.allow_resubmit = allow_resubmit
        self.startup_offset_sec = startup_offset_sec
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.thread = threading.Thread(
            target=self._lifecycle, daemon=True, name=f"slot-{self.state.label}",
        )
        self.thread.start()

    def join(self, timeout: float | None = None) -> None:
        if self.thread is not None:
            self.thread.join(timeout)

    def _lifecycle(self) -> None:
        while self.remaining_sec >= MIN_RESUBMIT_SEC:
            idx = len(self.state.attempts)
            jobname = self.jobname_base if idx == 0 else f"{self.jobname_base}-r{idx}"
            sbatch_log = self.state.workdir / f"{jobname}.sbatch.log"
            time_min = max(1, self.remaining_sec // 60)
            attempt = Attempt(jobname=jobname, sbatch_log=sbatch_log)
            # Append before submitting so the Monitor can discover the jobid early.
            self.state.attempts.append(attempt)
            # Stagger only the first attempt — retries are already de-synchronized
            # because each happens after a different elapsed time.
            offset = self.startup_offset_sec if idx == 0 else 0
            log(f"[{self.state.label}] submitting attempt #{idx+1} "
                f"({jobname}, --time={time_min}m, startup_offset={offset}s)")
            proc = submit_with_wait(
                self.state.workdir, jobname, self.slurm_script, time_min,
                startup_offset_sec=offset,
            )
            attempt.rc = proc.wait()
            attempt.jobid = read_jobid(sbatch_log)
            if not attempt.jobid:
                log(f"[{self.state.label}] could not read jobid from {sbatch_log}; "
                    "aborting slot")
                return
            summary = sacct_summary(attempt.jobid)
            attempt.final_state = str(summary.get("state", "UNKNOWN"))
            attempt.elapsed_sec = int(summary.get("elapsed_sec", 0))  # type: ignore[arg-type]
            attempt.exit_code = str(summary.get("exit_code", ""))
            log(f"[{self.state.label}] attempt #{idx+1} (job {attempt.jobid}) "
                f"ended: state={attempt.final_state}, elapsed={attempt.elapsed_sec}s, "
                f"exit_code={attempt.exit_code}, rc={attempt.rc}")
            if not self.allow_resubmit:
                return
            if not is_preemption(attempt.final_state, attempt.exit_code):
                return
            # Preempted. Deduct elapsed and try again on a fresh node.
            self.remaining_sec -= attempt.elapsed_sec
            if self.remaining_sec < MIN_RESUBMIT_SEC:
                log(f"[{self.state.label}] preempted, but only "
                    f"{self.remaining_sec}s left; not resubmitting")
                return
            cancel_orphan_gpu_jobs(attempt.jobid, self.state.label)
            log(f"[{self.state.label}] preempted; resubmitting with "
                f"{self.remaining_sec // 60}m remaining")


# --- monitor thread -------------------------------------------------------

class Monitor(threading.Thread):
    """Polls slot states and emits per-submission progress events."""

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
            for a in s.attempts:
                if a.jobid:
                    continue
                jid = read_jobid(a.sbatch_log)
                if jid:
                    a.jobid = jid
                    idx = s.attempts.index(a) + 1
                    log(f"[{s.label}] attempt #{idx} got slurm jobid = {jid}")

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
        total = 0
        for a in s.attempts:
            if not a.jobid:
                continue
            for out_file in s.workdir.glob(f"*_{a.jobid}.out"):
                try:
                    text = out_file.read_text(errors="ignore")
                except OSError:
                    continue
                total += text.count("Sleeping for 10 minutes")
        if total > s.iterations_seen:
            s.iterations_seen = total
            log(f"[{s.label}] completed iteration #{total} "
                "(gemini invocation finished)")

    def _poll_gpu_jobs(self) -> None:
        rows = squeue_me()
        # Map every known controller jobid (any attempt) back to its slot.
        ctrl_map: dict[str, SubmissionState] = {}
        for s in self.submissions:
            for a in s.attempts:
                if a.jobid:
                    ctrl_map[a.jobid] = s
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
                log(f"[{s.label}] launched GPU job {jobid} "
                    f"(name={jobname}, state={state})")
        # Anything missing from this round = concluded.
        for s in self.submissions:
            concluded = s.gpu_jobs_active - active_now[s.label]
            for jobid in concluded:
                summary = sacct_summary(jobid)
                final = summary.get("state", "UNKNOWN")
                log(f"[{s.label}] GPU job {jobid} concluded ({final})")
                s.gpu_jobs_active.discard(jobid)
                s.gpu_jobs_done.add(jobid)


# --- phase runner ---------------------------------------------------------

def run_phase(label: str, slots: list[ManagedSlot]) -> None:
    states = [s.state for s in slots]
    stop = threading.Event()
    monitor = Monitor(states, stop)
    monitor.start()
    for slot in slots:
        slot.start()
    for slot in slots:
        slot.join()
    stop.set()
    monitor.join(timeout=10)
    log(f"===== {label}: done =====")


# --- per-trial flow -------------------------------------------------------

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
    log(f"trial {trial}: seeded {NUM_SUBMISSIONS} submissions with "
        f"{len(input_papers)} input paper(s)")

    # ----- Phase 1: research -----
    research_script = SKELETON_DIR / "run_research_agent.slurm"
    research_budget = parse_time_limit_seconds(research_script)
    log(f"===== trial {trial}: launching {NUM_SUBMISSIONS} research agents "
        f"(budget={research_budget // 60}m each) =====")
    research_slots: list[ManagedSlot] = []
    for z in range(1, NUM_SUBMISSIONS + 1):
        sub_dir = trial_dir / f"submission{z}"
        state = SubmissionState(label=f"t{trial}/sub{z}", workdir=sub_dir)
        slot = ManagedSlot(
            state, f"research-t{trial}-s{z}", research_script, research_budget,
            startup_offset_sec=(z - 1) * STARTUP_STAGGER_SEC,
        )
        research_slots.append(slot)
    run_phase(f"trial {trial} research", research_slots)

    # ----- Phase 2: reviewing -----
    review_script = SKELETON_DIR / "run_reviewing_agent.slurm"
    review_budget = parse_time_limit_seconds(review_script)
    log(f"===== trial {trial}: launching reviewing agents =====")
    review_slots: list[ManagedSlot] = []
    skipped = 0
    review_idx = 0
    for z in range(1, NUM_SUBMISSIONS + 1):
        sub_dir = trial_dir / f"submission{z}"
        if not (sub_dir / "submission.pdf").exists():
            log(f"[t{trial}/sub{z}] no submission.pdf — skipping review")
            skipped += 1
            continue
        state = SubmissionState(label=f"t{trial}/sub{z}-rev", workdir=sub_dir)
        slot = ManagedSlot(
            state, f"review-t{trial}-s{z}", review_script, review_budget,
            startup_offset_sec=review_idx * STARTUP_STAGGER_SEC,
        )
        review_slots.append(slot)
        review_idx += 1
    log(f"trial {trial}: submitted {len(review_slots)} review jobs (skipped {skipped})")
    if review_slots:
        run_phase(f"trial {trial} review", review_slots)

    # ----- Phase 3: metareview -----
    meta_script = SKELETON_DIR / "run_metareview_agent.slurm"
    meta_budget = parse_time_limit_seconds(meta_script)
    log(f"===== trial {trial}: launching metareview =====")
    meta_state = SubmissionState(label=f"t{trial}/meta", workdir=trial_dir)
    meta_slot = ManagedSlot(meta_state, f"metareview-t{trial}", meta_script, meta_budget)
    run_phase(f"trial {trial} metareview", [meta_slot])

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
        existing = output_dir / f"trial{trial}" / "accepted_papers"
        existing_pdfs = sorted(existing.glob("*.pdf")) if existing.is_dir() else []
        if len(existing_pdfs) == 3:
            log(f"===== trial {trial}: skipping — {existing} already has 3 accepted PDFs =====")
            input_papers = existing_pdfs
            continue
        accepted = run_trial(trial, output_dir, input_papers)
        pdfs = sorted(accepted.glob("*.pdf"))
        if not pdfs:
            sys.exit(f"trial {trial}: no PDFs in {accepted}")
        input_papers = pdfs
    log(f"all {args.num_trials} trial(s) complete: {output_dir}")


if __name__ == "__main__":
    main()
