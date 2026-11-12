#!/usr/bin/env python3
import os
import sys
import json
import subprocess

import pin_index_in_memory


def get_total_memory_mb():
    """Read total physical memory from standard Linux procfs."""
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    return int(line.split()[1]) // 1024
    except FileNotFoundError:
        print("Error: /proc/meminfo not found. Linux required.")
        sys.exit(1)
    return 1024


def physical_cores():
    """Logical CPUs of every physical core, as cpuset strings ("0,16"), sorted by first CPU.

    Hyperthread siblings share one core, so two single-threaded QLever instances on siblings
    would slow each other down. One instance per physical core, pinned to all of that core's
    siblings, keeps the second hardware thread free of other QLever work.
    """
    cores = {}
    base = "/sys/devices/system/cpu"
    for name in os.listdir(base):
        if not (name.startswith("cpu") and name[3:].isdigit()):
            continue
        topology = os.path.join(base, name, "topology")
        try:
            with open(os.path.join(topology, "physical_package_id")) as f:
                package = int(f.read())
            with open(os.path.join(topology, "core_id")) as f:
                core = int(f.read())
        except (FileNotFoundError, ValueError):    # offline CPU or no topology information
            continue
        cores.setdefault((package, core), []).append(int(name[3:]))
    if not cores:
        return [str(cpu) for cpu in range(os.cpu_count() or 1)]
    return [",".join(map(str, sorted(cpus))) for cpus in sorted(cores.values(), key=min)]


def load_env(filepath):
    """Safely parse a standard environment file."""
    env_vars = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.split('#')[0].strip()  # Strip inline comments
                if line:
                    key, val = line.split('=', 1)
                    env_vars[key.strip()] = val.strip('\'"')
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        sys.exit(1)
    return env_vars


def main():
    # Detect and extract dry-run flag
    dry_run = "--dry-run" in sys.argv
    no_index = "--no-index" in sys.argv
    no_pin = "--no-pin" in sys.argv

    for flag in ("--dry-run", "--no-index", "--no-pin"):
        if flag in sys.argv:
            sys.argv.remove(flag)

    if len(sys.argv) != 5:
        print(f"Usage: {sys.argv[0]} <qlever_dir> <qleverfile_name> <lan_ip> <env_file> "
              f"[--dry-run] [--no-index] [--no-pin]")
        print("  --no-pin  do not lock the index in RAM (by default it is, so no query reads from disk)")
        print(
            f"Example: {sys.argv[0]} /data/qlever/qlever_yago qleverfile-default 192.168.0.2 qlever_yago.env --dry-run")
        sys.exit(1)

    qlever_dir = os.path.abspath(sys.argv[1])
    qleverfile = sys.argv[2]
    lan_ip = sys.argv[3]
    env_filename = sys.argv[4]
    base_port = 7000

    os.chdir(qlever_dir)

    # A pinner from an earlier deployment holds the old index files; release them before
    # re-indexing, and in any case before re-pinning.
    pin_index_in_memory.stop(qlever_dir, dry_run=dry_run)

    index_cmd = ["qlever", "--qleverfile", qleverfile, "index", "--overwrite-existing"]
    if dry_run:
        print(f"[DRY RUN] Indexing command: {' '.join(index_cmd)}")
    else:
        print("Indexing data...")
        if not no_index:
            subprocess.run(index_cmd, check=True)

    # Load Configuration
    env_vars = load_env(env_filename)
    index_name = env_vars.get("QLEVER_INDEX_NAME")
    internal_port = env_vars.get("QLEVER_INTERNAL_PORT")
    image = env_vars.get("QLEVER_IMAGE")
    token = env_vars.get("QLEVER_ACCESS_TOKEN")
    timeout = env_vars.get("QLEVER_TIMEOUT")
    cache_size = env_vars.get("QLEVER_CACHE_MAX_SIZE")
    cache_entries = env_vars.get("QLEVER_CACHE_MAX_ENTRIES")
    # One worker thread per instance by default: each container is pinned to one core.
    threads = env_vars.get("QLEVER_THREADS", "1")
    # Physical cores left to the OS, Docker and the index pinner (the ones holding CPU 0 first).
    reserved_cores = int(env_vars.get("QLEVER_RESERVED_CORES", "1"))

    if not all([index_name, internal_port, image, token, timeout, cache_size, cache_entries]):
        print(f"Error: Missing required variables in {env_filename}.")
        sys.exit(1)

    base_args = (f"-i {index_name} -p {internal_port} -a {token} -s {timeout} -c {cache_size} "
                 f"-k {cache_entries} -j {threads}")

    # Lock the index in RAM before any server starts: every query then reads from memory,
    # which matters on nodes with a hard disk (one seek is ~10 ms, a typical query ~30 ms).
    pinned_mb = 0
    if not no_pin:
        pinned_mb = pin_index_in_memory.start(qlever_dir, index_name, dry_run=dry_run) // 2**20

    # Calculate Hardware Limits
    all_cores = physical_cores()
    cores = all_cores[min(reserved_cores, len(all_cores) - 1):]
    total_mem_mb = get_total_memory_mb()
    safe_mem_mb = total_mem_mb - 2048 - pinned_mb       # locked index pages are not available

    if safe_mem_mb <= 0:
        safe_mem_mb = total_mem_mb
        print("Warning: Available memory is less than 2048 MB. Host OS may become unstable.")

    # Optional big lane (statistics_based_generation certificate.BackgroundCounter): the LAST
    # QLEVER_BIG_INSTANCES instances (default 1) get QLEVER_BIG_MEMORY_MB each and retry the counts that ran
    # out of memory on the others; the rest share what is left. All limits together stay within safe_mem_mb.
    big_mem_mb = int(env_vars.get("QLEVER_BIG_MEMORY_MB") or 0)
    n_big = int(env_vars.get("QLEVER_BIG_INSTANCES", "1")) if big_mem_mb else 0
    if n_big >= len(cores) or n_big * big_mem_mb >= safe_mem_mb:
        print(f"Error: {n_big} big instance(s) of {big_mem_mb}MB do not fit next to the others "
              f"({len(cores)} instances, {safe_mem_mb}MB).")
        sys.exit(1)
    per_core_mem_mb = (safe_mem_mb - n_big * big_mem_mb) // (len(cores) - n_big)
    # Optional cap, for nodes that also run something memory-hungry (e.g. query generation): instances keep
    # what their heaviest query needed, so 15 instances can otherwise take most of the RAM.
    if env_vars.get("QLEVER_MEMORY_MB"):
        per_core_mem_mb = min(per_core_mem_mb, int(env_vars["QLEVER_MEMORY_MB"]))
    memory_mb = [per_core_mem_mb] * (len(cores) - n_big) + [big_mem_mb] * n_big

    print(f"Hardware detected: {len(all_cores)} physical cores ({os.cpu_count()} logical CPUs), "
          f"{total_mem_mb}MB Total RAM.")
    print(f"Starting {len(cores)} instances ({len(all_cores) - len(cores)} core(s) reserved), "
          f"memory limit {per_core_mem_mb}MB each" +
          (f", the last {n_big} (ports {base_port + len(cores) - n_big}..{base_port + len(cores) - 1}, "
           f"the big lane) {big_mem_mb}MB" if n_big else "") + f"; {sum(memory_mb)}MB in all.")

    # Spawn Isolated Containers
    endpoints = []
    for i, cpuset in enumerate(cores):
        port = base_port + i
        container_name = f"qlever_core_{i}"
        final_server_cmd = f"qlever-server {base_args} -m {memory_mb[i]}M >> qlever_core_{i}.server-log.txt 2>&1"

        docker_cmd = [
            "docker", "run", "-d",
            "--name", container_name,
            "--cpuset-cpus", cpuset,
            "--restart", "unless-stopped",
            "-p", f"{port}:{internal_port}",
            "-v", f"{qlever_dir}:/index",
            "-w", "/index",
            "-u", f"{os.getuid()}:{os.getgid()}",
            "--entrypoint", "bash",
            image,
            "-c", final_server_cmd
        ]

        if dry_run:
            print(f"\n[DRY RUN] Core {i} configuration:")
            print(f"  Remove existing: docker rm -f {container_name}")
            print(f"  Run container:   {' '.join(docker_cmd)}")
        else:
            subprocess.run(["docker", "rm", "-f", container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(docker_cmd, check=True)

        endpoints.append(f"http://{lan_ip}:{port}")

    # Save Endpoints
    endpoints_file = f"endpoints_{lan_ip}.json"
    if dry_run:
        print(f"\n[DRY RUN] JSON payload to be saved to {endpoints_file}:")
        print(json.dumps(endpoints, indent=2))
    else:
        with open(endpoints_file, "w") as f:
            json.dump(endpoints, f, indent=2)
        print(f"Deployment complete. Endpoints saved to {endpoints_file}.")


if __name__ == "__main__":
    main()