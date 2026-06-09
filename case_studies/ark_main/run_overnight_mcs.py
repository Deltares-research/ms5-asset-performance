"""Overnight supervisor: sequential MCS runs at multiple cr values.

For each cr in ``--cr-values``::

    1. Start ``python -m case_studies.ark_main.analysis.run_mc`` with
       ``--n-samples N``.
    2. Poll the checkpoint CSV every ``--poll`` seconds; once the row count
       reaches ``--target`` the child is terminated.
    3. Re-invoke with ``--postprocess-only`` so ``design_point_lsf_*.json``
       and the convergence plots are produced from whatever made it into the
       checkpoint.

Progress + log lines are persisted to
``<remote>/output/mc_<lsf>/_overnight_state.json`` so a quick ``cat`` from
another shell tells you what's happening overnight without attaching to the
hidden process. The supervisor's own stdout/stderr is mirrored into
``<remote>/output/mc_<lsf>/_supervisor_main.log``.
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_ARK = Path(__file__).resolve().parent
if str(_ARK) not in sys.path:
    sys.path.insert(0, str(_ARK))

from dotenv import load_dotenv
load_dotenv(_ARK / "geolib.env")

from src.io import get_remote_path

_ENV = _ARK / ".env"


def count_samples(ckpt: Path) -> int:
    if not ckpt.exists():
        return 0
    try:
        with open(ckpt) as f:
            n_lines = sum(1 for _ in f)
        return max(0, n_lines - 5)  # 4 meta lines + 1 header
    except OSError:
        return 0


def update_state(state_path: Path, state: dict) -> None:
    state["updated_at"] = datetime.now().isoformat()
    text = json.dumps(state, indent=2, default=str)
    # SMB share occasionally rejects writes/renames with WinError 5; back off
    # and retry. A truncated mid-write read is harmless here — readers can
    # simply look again on the next update.
    last_exc: Exception | None = None
    for delay in (0.0, 0.2, 0.5, 1.0, 2.0, 5.0):
        if delay:
            time.sleep(delay)
        try:
            state_path.write_text(text, encoding="utf-8")
            return
        except PermissionError as e:
            last_exc = e
    if last_exc is not None:
        raise last_exc


def log_line(state: dict, state_path: Path, msg: str) -> None:
    line = f"[{datetime.now().isoformat()}] {msg}"
    print(line, flush=True)
    state.setdefault("log", []).append(line)
    update_state(state_path, state)


def run_one_cr(
    *, lsf: str, cr: float, n_samples: int, target: int, seed: int,
    poll: int, out_root: Path, state: dict, state_path: Path,
) -> None:
    cr_dir = out_root / f"cr_{cr:.4f}"
    cr_dir.mkdir(parents=True, exist_ok=True)
    ckpt = cr_dir / f"checkpoint_cr_{cr:.4f}.csv"
    log_line(state, state_path,
             f"cr={cr}: starting MCS (n_samples={n_samples}, target={target})")

    cmd = [
        sys.executable, "-u", "-m", "case_studies.ark_main.analysis.run_mc",
        "--lsf", lsf, "--n-samples", str(n_samples),
        "--cr", str(cr), "--seed", str(seed),
    ]
    child_log = cr_dir / "_supervisor_child_stdout.log"
    with open(child_log, "w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
    state["current_pid"] = proc.pid
    update_state(state_path, state)

    last_logged_bucket = -1
    while True:
        time.sleep(poll)
        n = count_samples(ckpt)
        rc = proc.poll()
        bucket = n // 100
        if bucket > last_logged_bucket:
            log_line(state, state_path, f"cr={cr}: {n}/{target} samples")
            last_logged_bucket = bucket
        if n >= target:
            log_line(state, state_path,
                     f"cr={cr}: reached target {target} (have {n}), "
                     f"terminating PID {proc.pid}")
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                log_line(state, state_path,
                         f"cr={cr}: terminate timed out, killing")
                proc.kill()
                proc.wait()
            break
        if rc is not None:
            log_line(state, state_path,
                     f"cr={cr}: process exited prematurely with code {rc} "
                     f"at {n} samples")
            break

    log_line(state, state_path, f"cr={cr}: running --postprocess-only")
    pp_log = cr_dir / "_supervisor_postprocess.log"
    with open(pp_log, "w") as logf:
        pp_rc = subprocess.run(
            cmd + ["--postprocess-only"],
            stdout=logf, stderr=subprocess.STDOUT,
        ).returncode
    log_line(state, state_path, f"cr={cr}: postprocess exit code {pp_rc}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lsf", default="lsf_wall_anchor")
    p.add_argument("--n-samples", type=int, default=10000)
    p.add_argument("--target", type=int, default=2000,
                   help="Number of completed samples at which the run is killed.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cr-values", type=float, nargs="+", required=True)
    p.add_argument("--poll", type=int, default=30,
                   help="Checkpoint-poll interval in seconds.")
    args = p.parse_args()

    remote = get_remote_path(_ENV)
    out_root = Path(remote) / "output" / f"mc_{args.lsf}"
    out_root.mkdir(parents=True, exist_ok=True)
    state_path = out_root / "_overnight_state.json"
    main_log = out_root / "_supervisor_main.log"

    # Mirror prints to a persistent log so the supervisor's own output is
    # recoverable even when launched detached via Start-Process -Hidden.
    sys.stdout = open(main_log, "a", buffering=1, encoding="utf-8")
    sys.stderr = sys.stdout

    state: dict = {
        "lsf": args.lsf,
        "n_samples": args.n_samples,
        "target": args.target,
        "seed": args.seed,
        "cr_values": args.cr_values,
        "started_at": datetime.now().isoformat(),
        "completed": [],
        "in_progress": None,
        "current_pid": None,
        "log": [],
    }
    update_state(state_path, state)
    log_line(state, state_path,
             f"overnight supervisor starting (PID {Path(sys.executable).name})")

    for cr in args.cr_values:
        state["in_progress"] = cr
        update_state(state_path, state)
        try:
            run_one_cr(
                lsf=args.lsf, cr=cr, n_samples=args.n_samples,
                target=args.target, seed=args.seed, poll=args.poll,
                out_root=out_root, state=state, state_path=state_path,
            )
        except Exception as e:
            log_line(state, state_path,
                     f"cr={cr}: SUPERVISOR ERROR {type(e).__name__}: {e}")
        state["completed"].append(cr)
        state["in_progress"] = None
        state["current_pid"] = None
        update_state(state_path, state)

    state["finished_at"] = datetime.now().isoformat()
    log_line(state, state_path, "ALL DONE")


if __name__ == "__main__":
    main()
