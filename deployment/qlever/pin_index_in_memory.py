#!/usr/bin/env python3
"""Keep a QLever index locked in RAM, so query latencies never include disk reads.

A background process mmaps every index file and mlocks it. mlock reads each page in once
(from disk if it is not cached yet) and then the kernel can neither evict nor swap those
pages until the process exits. The QLever containers bind-mount the same files, so they read
the very same page-cache pages: one locked copy serves every instance on the node.

Usage:
    pin_index_in_memory.py start  <qlever_dir> <index_name>   # lock, wait until resident
    pin_index_in_memory.py status <qlever_dir>                # locked bytes of the pinner
    pin_index_in_memory.py stop   <qlever_dir>                # unlock (kill the pinner)

Locking more than RLIMIT_MEMLOCK needs root; `start` then runs the pinner through `sudo -n`.
"""
import ctypes
import os
import resource
import signal
import subprocess
import sys
import time

PIDFILE = "index_pinner.pid"
LOGFILE = "index_pinner.log"
# Files QLever reads at query time. The input dump (.nt/.ttl), logs, settings and the UI
# database are not needed by the server and are left out.
INDEX_PARTS = (".index.", ".internal.", ".vocabulary.", ".text.")
INDEX_FILES = (".meta-data.json",)
PAGE = os.sysconf("SC_PAGE_SIZE")
PROT_READ, MAP_SHARED = 1, 1


def index_files(qlever_dir, index_name):
    files = []
    for name in sorted(os.listdir(qlever_dir)):
        if not name.startswith(index_name + "."):
            continue
        rest = name[len(index_name):]
        if rest.startswith(INDEX_PARTS) or rest in INDEX_FILES:
            path = os.path.join(qlever_dir, name)
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                files.append(path)
    return files


def pinned_bytes(files):
    """Bytes mlock will hold for these files (whole pages)."""
    return sum(-(-os.path.getsize(f) // PAGE) * PAGE for f in files)


def needs_sudo(n_bytes):
    if os.geteuid() == 0:
        return False
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    return hard != resource.RLIM_INFINITY and hard < n_bytes


def _read_pid(qlever_dir):
    try:
        with open(os.path.join(qlever_dir, PIDFILE)) as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return None


def locked_bytes(pid):
    """VmLck of a process, in bytes (None if it is not running)."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmLck:"):
                    return int(line.split()[1]) * 1024
    except FileNotFoundError:
        return None
    return 0


def stop(qlever_dir, dry_run=False):
    pid = _read_pid(qlever_dir)
    if pid is None or locked_bytes(pid) is None:
        return
    command = ["kill", str(pid)]
    if os.geteuid() != 0 and os.stat(f"/proc/{pid}").st_uid != os.geteuid():
        command = ["sudo", "-n"] + command
    if dry_run:
        print(f"[DRY RUN] Stop old index pinner: {' '.join(command)}")
        return
    subprocess.run(command, check=True)
    for _ in range(100):
        if locked_bytes(pid) is None:
            break
        time.sleep(0.1)
    try:
        os.remove(os.path.join(qlever_dir, PIDFILE))
    except OSError:
        pass
    print(f"Stopped index pinner (pid {pid}).")


def start(qlever_dir, index_name, dry_run=False, timeout_s=1800):
    """Start the pinner and return once every index page is locked; returns the bytes locked."""
    qlever_dir = os.path.abspath(qlever_dir)
    files = index_files(qlever_dir, index_name)
    if not files:
        sys.exit(f"Error: no index files for '{index_name}' in {qlever_dir}.")
    n_bytes = pinned_bytes(files)
    command = [sys.executable, os.path.abspath(__file__), "_daemon", qlever_dir] + files
    if needs_sudo(n_bytes):
        command = ["sudo", "-n"] + command
    if dry_run:
        prefix = "sudo -n " if command[0] == "sudo" else ""
        print(f"[DRY RUN] Pin {len(files)} index files ({n_bytes / 2**30:.2f} GiB) in RAM "
              f"({prefix}{os.path.basename(__file__)} _daemon {qlever_dir} ...):")
        for path in files:
            print(f"  {os.path.basename(path)}")
        return n_bytes
    stop(qlever_dir)
    with open(os.path.join(qlever_dir, LOGFILE), "a") as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                   start_new_session=True)
    print(f"Locking {len(files)} index files ({n_bytes / 2**30:.2f} GiB) in RAM...")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if process.poll() is not None:
            sys.exit(f"Error: index pinner exited with code {process.returncode}; "
                     f"see {os.path.join(qlever_dir, LOGFILE)}. Without root, raise the memlock "
                     f"limit or allow passwordless sudo.")
        pid = _read_pid(qlever_dir)
        locked = locked_bytes(pid) if pid else None
        if locked is not None and locked >= n_bytes:
            print(f"Index locked in RAM: {locked / 2**30:.2f} GiB, pinner pid {pid}.")
            return n_bytes
        time.sleep(0.5)
    sys.exit("Error: timed out waiting for the index to be locked in RAM.")


def _daemon(qlever_dir, files):
    """Runs in the background: mmap + mlock every file, then wait until killed."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                          ctypes.c_int, ctypes.c_long]
    libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    n_bytes = pinned_bytes(files)
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if soft != resource.RLIM_INFINITY and soft < n_bytes:
        resource.setrlimit(resource.RLIMIT_MEMLOCK, (hard, hard))
    keep = []
    for path in files:
        size = os.path.getsize(path)
        fd = os.open(path, os.O_RDONLY)
        address = libc.mmap(None, size, PROT_READ, MAP_SHARED, fd, 0)
        os.close(fd)                          # the mapping keeps the file referenced
        if address in (None, ctypes.c_void_p(-1).value):
            sys.exit(f"mmap failed for {path}: {os.strerror(ctypes.get_errno())}")
        if libc.mlock(ctypes.c_void_p(address), ctypes.c_size_t(size)) != 0:
            sys.exit(f"mlock failed for {path}: {os.strerror(ctypes.get_errno())}")
        keep.append(address)
    with open(os.path.join(qlever_dir, PIDFILE), "w") as f:
        f.write(str(os.getpid()))
    os.chmod(os.path.join(qlever_dir, PIDFILE), 0o644)
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} locked {n_bytes} bytes in {len(files)} files", flush=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    while True:
        signal.pause()


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "_daemon":
        _daemon(sys.argv[2], sys.argv[3:])
    elif len(sys.argv) == 4 and sys.argv[1] == "start":
        start(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3 and sys.argv[1] == "stop":
        stop(os.path.abspath(sys.argv[2]))
    elif len(sys.argv) == 3 and sys.argv[1] == "status":
        pid = _read_pid(os.path.abspath(sys.argv[2]))
        locked = locked_bytes(pid) if pid else None
        print(f"pinner pid {pid}: {locked / 2**30:.2f} GiB locked" if locked is not None else "no index pinner running")
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
