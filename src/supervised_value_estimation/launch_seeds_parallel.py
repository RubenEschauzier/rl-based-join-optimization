"""Train several rerun seeds at the same time on one GPU, one child process per seed.

    python -m src.supervised_value_estimation.launch_seeds_parallel 0 1 [-- hydra overrides]

Worth it when a single run leaves the GPU mostly idle (small per-query kernels), which the
job's Usage Graphs on GPULab show. Each child loads the datasets itself, so CPU memory
scales with the number of seeds (~22 GB each).

This process is meant to be the container's PID 1: it forwards GPULab's SIGUSR1 halt
signal to every child and exits with 123 so a restartable job is re-queued. Finished
seeds are skipped on the restart. Otherwise it exits non-zero if any seed failed.
"""
import argparse
import os
import signal
import subprocess
import sys
import threading

HALT_EXIT_CODE = 123


def _stream_with_prefix(stream, prefix):
    for line in iter(stream.readline, ""):
        sys.stdout.write(f"{prefix} {line}")
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("seeds", nargs="+", type=int)
    # Everything after "--" is passed to each seed's hydra command line untouched.
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    arguments = parser.parse_args(argv[:split])
    overrides = argv[split + 1:]

    children = {}
    for seed in arguments.seeds:
        command = [sys.executable, "-m", "src.supervised_value_estimation.rerun_epinet_seeds",
                   f"rerun.seed={seed}", *overrides]
        children[seed] = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        threading.Thread(target=_stream_with_prefix, args=(children[seed].stdout, f"[seed {seed}]"),
                         daemon=True).start()
        print(f"Started seed {seed} (pid {children[seed].pid})", flush=True)

    halted = threading.Event()

    def forward_halt(signum, frame):
        halted.set()
        print("Received SIGUSR1 (GPULab halt): forwarding to all seeds.", flush=True)
        for child in children.values():
            if child.poll() is None:
                child.send_signal(signal.SIGUSR1)

    signal.signal(signal.SIGUSR1, forward_halt)

    exit_codes = {seed: child.wait() for seed, child in children.items()}
    for seed, code in exit_codes.items():
        print(f"Seed {seed} exited with {code}", flush=True)
    if halted.is_set():
        sys.exit(HALT_EXIT_CODE)
    # A child killed by a signal reports a negative code, so collapse failures to 1.
    sys.exit(0 if all(code == 0 for code in exit_codes.values()) else 1)


if __name__ == "__main__":
    main()
