#!/usr/bin/env python3
"""Run the three required trials (10 g, 100 g, 500 g) and compare them.

  python run_all.py                  # 3 trials in parallel (3 CPU cores) + plots + summary
  python run_all.py --sequential     # one after the other (less RAM / CPU)
  python run_all.py --grip-force 8   # any run_trial.py option is forwarded to every trial

Outputs: results/<trial>/{*.png, summary.json, config.json, run.npz, log.txt}
         results/comparison_*.png, results/summary.md, results/validation.md
"""
import os
import subprocess
import sys
import time

from cupsim.config import TRIAL_MASSES

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    args = sys.argv[1:]
    sequential = "--sequential" in args
    args = [a for a in args if a != "--sequential"]
    out = os.path.join(ROOT, "results")
    if "--out" in args:
        out = args[args.index("--out") + 1]
    os.makedirs(out, exist_ok=True)

    t0 = time.time()
    procs = []
    for name in TRIAL_MASSES:
        os.makedirs(os.path.join(out, name), exist_ok=True)
        log = open(os.path.join(out, name, "log.txt"), "w")
        cmd = [sys.executable, os.path.join(ROOT, "run_trial.py"), "--mass", name] + args
        print("starting:", " ".join(cmd[1:]), f"  (log: {os.path.relpath(log.name, ROOT)})")
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        procs.append((name, p, log))
        if sequential:
            p.wait()
    failed = []
    for name, p, log in procs:
        p.wait()
        log.close()
        status = "ok" if p.returncode == 0 else f"FAILED (exit {p.returncode}) - see {log.name}"
        print(f"  trial {name}: {status}")
        if p.returncode != 0:
            failed.append(name)
    print(f"all trials finished in {time.time() - t0:.0f} s")

    from analyze import compare
    compare(out)
    subprocess.run([sys.executable, os.path.join(ROOT, "validate.py")], cwd=ROOT,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"\nsummary table: {os.path.relpath(os.path.join(out, 'summary.md'), ROOT)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
