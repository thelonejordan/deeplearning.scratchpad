#!/usr/bin/env python3
"""Orchestrator: create a RunPod GPU pod, run pytest over SSH, tear down.

Split into three subcommands so a CI job can attribute failures and timing per phase:

    setup    --pod-name NAME --commit SHA [--test-commands JSON]
    test     --test-commands JSON
    teardown [--pod-name NAME | --pod-id ID]

`setup` exports POD_ID/POD_SSH_KEY/POD_HOST/POD_PORT to $GITHUB_ENV for the later steps. Run
locally it prints them instead, so export them into your shell before calling `test`, and ALWAYS
run `teardown` afterwards -- a surviving pod bills by the minute. `teardown --pod-name` can find
the pod without POD_ID, which is the recovery path if `setup` died before exporting it.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time

import tempfile

import paramiko
import runpod

REPO_URL = "https://github.com/thelonejordan/deeplearning.scratchpad.git"
DEFAULT_IMAGE = "runpod/pytorch:1.3.3-cu1281-torch2130-ubuntu2204"
DEFAULT_GPU = "NVIDIA RTX 4000 Ada Generation"

# Pinned to match uv.lock. Unpinned installs resolve against the image's preinstalled torch,
# which broke CI once already: transformers raised its floor from torch>=2.4 to torch>=2.5, so
# `is_torch_available()` started returning False on an image shipping torch 2.4.1 and every
# test that touches transformers failed with "PyTorch is not installed".
POD_PACKAGES = [
    "transformers==5.4.0",
    "accelerate==1.13.0",
    "huggingface-hub[hf_xet]==1.8.0",
    "hf-xet==1.4.2",
    "hf-transfer==0.1.9",
    "tiktoken==0.12.0",
    "sentencepiece==0.2.1",
    "blobfile==3.2.0",
    "requests==2.33.0",
    "protobuf==7.34.1",
    "pytest==9.0.2",
]

# Xet-backed downloads 404 on `api/models/<repo>/xet-read-token/<rev>` from inside pods even for
# public repos; the plain Hub download path works. Drop this once that is fixed upstream.
POD_TEST_ENV = {"HF_HUB_DISABLE_XET": "1"}
GPU_FALLBACKS = [
    "NVIDIA RTX 4000 Ada Generation",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX A4000",
    "NVIDIA L4",
    "NVIDIA GeForce RTX 3090",
]
# Per-step hard timeouts. Their sum must stay under `timeout-minutes` on the workflow job (30, see
# .github/workflows/test-gpu.yml) with slack left for checkout/uv/teardown, so that these fire
# first and `teardown` runs as a normal step instead of during a job cancellation.
DEFAULT_SETUP_TIMEOUT = 10 * 60
DEFAULT_TEST_TIMEOUT = 15 * 60
# How much of the setup budget may go on waiting for the pod to boot; the rest is for cloning and
# installing, which needs a couple of minutes on a cold host.
SSH_READY_TIMEOUT = 5 * 60
POLL_INTERVAL = 10  # seconds
TERMINATE_ATTEMPTS = 3
TERMINATE_BACKOFF = 5  # seconds


def validate_commit_sha(sha):
    if not re.fullmatch(r'[0-9a-fA-F]{4,40}|HEAD', sha):
        raise ValueError(f"Invalid commit SHA: {sha}")
    return sha


def validate_test_commands(commands):
    if not isinstance(commands, list) or not all(isinstance(c, str) for c in commands):
        raise ValueError("--test-commands must be a JSON array of strings")
    for cmd in commands:
        if not re.fullmatch(r'[a-zA-Z0-9_./ =:\-"]+', cmd):
            raise ValueError(f"Suspicious test command rejected: {cmd}")
    return commands


def generate_ssh_keypair():
    """Generate an ephemeral SSH keypair for this CI run."""
    tmpdir = tempfile.mkdtemp()
    key_path = os.path.join(tmpdir, "id_ed25519")
    subprocess.run(
        ["ssh-keygen", "-t", "ed25519", "-f", key_path, "-N", "", "-q"],
        check=True,
    )
    pub_path = key_path + ".pub"
    with open(pub_path) as f:
        public_key = f.read().strip()
    return key_path, public_key


def create_gpu_pod(name, image, gpu_type, public_key, hf_token=None):
    env = {"PUBLIC_KEY": public_key}
    if hf_token:
        env["HF_TOKEN"] = hf_token

    gpu_types_to_try = [gpu_type] + [g for g in GPU_FALLBACKS if g != gpu_type]
    for gpu in gpu_types_to_try:
        try:
            print(f"Trying GPU: {gpu}")
            pod = runpod.create_pod(
                name=name,
                image_name=image,
                gpu_type_id=gpu,
                gpu_count=1,
                container_disk_in_gb=30,
                volume_in_gb=0,
                ports="22/tcp",
                start_ssh=True,
                env=env,
            )
            print(f"Created pod: {pod['id']} (GPU: {gpu})")
            return pod["id"]
        except (TimeoutError, KeyboardInterrupt):
            # Never swallow the step alarm here: it has already fired, so continuing would retry the
            # remaining GPU types with no timeout left to stop them.
            raise
        except Exception as e:
            print(f"Failed to create pod with {gpu}: {e}")
            continue

    raise RuntimeError("Could not create pod with any available GPU type")


def wait_for_ssh(pod_id, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        pod = runpod.get_pod(pod_id)
        runtime = pod.get("runtime")
        if runtime and runtime.get("ports"):
            for port_info in runtime["ports"]:
                if port_info["privatePort"] == 22:
                    host = port_info["ip"]
                    port = port_info["publicPort"]
                    print(f"SSH available: root@{host}:{port}")
                    return host, port
        status = pod.get("desiredStatus", "UNKNOWN")
        print(f"Waiting for pod... status={status}")
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"Pod {pod_id} did not become ready within {timeout}s")


def ssh_exec(client, cmd, stream=True):
    """Execute a command over SSH. Returns exit code."""
    print(f"$ {cmd}")
    stdin, stdout, stderr = client.exec_command(cmd)
    out = stdout.read().decode()
    err = stderr.read().decode()
    exit_code = stdout.channel.recv_exit_status()
    if out:
        print(out)
    if err:
        print(err, file=sys.stderr)
    return exit_code


def connect_ssh(host, port, ssh_key_path):
    client = paramiko.SSHClient()
    # RunPod pods are ephemeral and host keys are not known in advance
    client.set_missing_host_key_policy(paramiko.WarningPolicy())
    pkey = paramiko.Ed25519Key.from_private_key_file(ssh_key_path)

    for attempt in range(6):
        try:
            client.connect(host, port=port, username="root", pkey=pkey, timeout=30)
            return client
        except Exception as e:
            if attempt == 5:
                raise
            print(f"SSH connect attempt {attempt + 1} failed: {e}, retrying...")
            time.sleep(10)


def prepare_pod(client, commit_sha, hf_token=None):
    # Clone repo at specific commit
    ssh_exec(client, f"git clone {REPO_URL} /workspace/repo")
    ssh_exec(client, f"cd /workspace/repo && git checkout {commit_sha}")

    # Install dependencies (image Python already has torch+CUDA, don't touch those).
    # `python`/`pip` point at the interpreter torch is installed for, `python3` does not.
    packages = " ".join(POD_PACKAGES)
    exit_code = ssh_exec(client, f"python -m pip install --break-system-packages {packages}")
    if exit_code != 0:
        raise RuntimeError(f"Dependency install failed (exit {exit_code})")

    # HuggingFace login via token file (Docker ENV not available in SSH sessions)
    if hf_token:
        sftp = client.open_sftp()
        with sftp.file("/tmp/hf_token", "w") as f:
            f.write(hf_token)
        sftp.close()
        print(f"Uploaded HF token ({len(hf_token)} chars, starts with {hf_token[:4]}...)")
        ssh_exec(client, "cat /tmp/hf_token | wc -c")
        ssh_exec(client, "python -c 'import huggingface_hub; huggingface_hub.login(open(\"/tmp/hf_token\").read().strip())' && rm /tmp/hf_token")
        ssh_exec(client, "python -c 'import huggingface_hub; print(huggingface_hub.whoami())'")
    else:
        print("WARNING: No HF_TOKEN provided, skipping HuggingFace login")

    # Display environment
    ssh_exec(client, "nvidia-smi")
    ssh_exec(client, "cd /workspace/repo && python env.py")
    ssh_exec(client, "python -m pip list | grep -Ei 'torch|transformers|huggingface|tokenizers|accelerate'")


def run_tests(client, test_commands):
    env_prefix = "".join(f"{k}={v} " for k, v in POD_TEST_ENV.items())
    final_exit_code = 0
    for cmd in test_commands:
        full_cmd = f"cd /workspace/repo && {env_prefix}{cmd}"
        exit_code = ssh_exec(client, full_cmd)
        if exit_code != 0:
            print(f"FAILED (exit {exit_code}): {cmd}")
            final_exit_code = exit_code
        else:
            print(f"PASSED: {cmd}")

    return final_exit_code


def set_api_key():
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY environment variable is required", file=sys.stderr)
        sys.exit(1)
    runpod.api_key = api_key


def set_timeout(seconds):
    """Hard timeout for this step, to prevent unbounded cloud spend."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Step timeout of {seconds}s exceeded")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)


def export_pod_state(**values):
    """Hand pod state to later workflow steps via $GITHUB_ENV (printed when run locally).

    The SSH key is passed by path, not by value, which relies on every step sharing the runner
    filesystem -- adding `container:` to a job would break that. Written in the heredoc form so a
    newline inside a value cannot inject further environment entries.
    """
    github_env = os.environ.get("GITHUB_ENV")
    for key, value in values.items():
        print(f"{key}={value}")
    if github_env:
        with open(github_env, "a") as f:
            for key, value in values.items():
                f.write(f"{key}<<__POD_STATE__\n{value}\n__POD_STATE__\n")


def read_pod_state(*keys):
    values = []
    for key in keys:
        value = os.environ.get(key)
        if not value:
            print(f"ERROR: {key} is not set, did the `setup` step run?", file=sys.stderr)
            sys.exit(1)
        values.append(value)
    return values


def find_pod_by_name(name):
    """Look a pod up by name, for when POD_ID never reached $GITHUB_ENV.

    Lets a failed listing propagate: "could not tell" must not be reported as "nothing to clean up".
    """
    for pod in runpod.get_pods():
        if pod.get("name") == name:
            return pod["id"]
    return None


def terminate_pod(pod_id):
    """Terminate a pod and confirm it is gone. Retries: this is the only teardown path there is."""
    for attempt in range(1, TERMINATE_ATTEMPTS + 1):
        try:
            runpod.terminate_pod(pod_id)
        except Exception as e:
            print(f"WARNING: terminate attempt {attempt}/{TERMINATE_ATTEMPTS} failed: {e}", file=sys.stderr)
            if attempt < TERMINATE_ATTEMPTS:
                time.sleep(TERMINATE_BACKOFF)
            continue

        time.sleep(TERMINATE_BACKOFF)  # let the status catch up before believing it
        gone = pod_is_gone(pod_id)
        if gone is not False:
            # `is None` means the call was accepted but we could not read the state back; say so
            # rather than claim success, because the difference shows up on the bill.
            print("Pod terminated." if gone else f"Terminate accepted for {pod_id}, state unconfirmed")
            return True
        print(f"Pod {pod_id} still reports alive after terminate")
        if attempt < TERMINATE_ATTEMPTS:
            time.sleep(TERMINATE_BACKOFF)

    print(f"ERROR: could not terminate pod {pod_id} -- terminate it manually", file=sys.stderr)
    return False


def pod_is_gone(pod_id):
    """True if confirmed gone, False if confirmed alive, None if we could not tell."""
    try:
        pod = runpod.get_pod(pod_id)
    except Exception as e:
        print(f"Could not read back pod state: {e}", file=sys.stderr)
        return None
    if not pod:
        return True
    return pod.get("desiredStatus") in ("TERMINATED", "EXITED")


def cmd_setup(args):
    # Validate before anything can start billing: a typo in the command list should not cost a pod
    # create plus a pip install to discover, since `test` only runs minutes later.
    commit_sha = validate_commit_sha(args.commit)
    if args.test_commands:
        validate_test_commands(json.loads(args.test_commands))
    set_api_key()
    hf_token = os.environ.get("HF_TOKEN")
    set_timeout(args.timeout)

    ssh_key_path, public_key = generate_ssh_keypair()
    print("Generated ephemeral SSH keypair")
    pod_id = create_gpu_pod(args.pod_name, args.image, args.gpu_type, public_key, hf_token)
    # Export before anything else can fail, so `teardown` can always find the pod
    export_pod_state(POD_ID=pod_id, POD_SSH_KEY=ssh_key_path)

    host, port = wait_for_ssh(pod_id, timeout=SSH_READY_TIMEOUT)
    export_pod_state(POD_HOST=host, POD_PORT=port)

    client = connect_ssh(host, port, ssh_key_path)
    try:
        prepare_pod(client, commit_sha, hf_token)
    finally:
        client.close()
        signal.alarm(0)


def cmd_test(args):
    test_commands = validate_test_commands(json.loads(args.test_commands))
    host, port, ssh_key_path = read_pod_state("POD_HOST", "POD_PORT", "POD_SSH_KEY")
    set_timeout(args.timeout)

    client = connect_ssh(host, int(port), ssh_key_path)
    try:
        return run_tests(client, test_commands)
    finally:
        client.close()
        signal.alarm(0)


def cmd_teardown(args):
    pod_id = os.environ.get("POD_ID") or args.pod_id
    if not pod_id and not args.pod_name:
        print("No POD_ID and no --pod-name, nothing to terminate")
        return 0
    set_api_key()
    if not pod_id:
        # setup can be killed between creating the pod and exporting its id (job cancellation is
        # routine here: concurrency.cancel-in-progress is on), so fall back to the pod's name
        print(f"No POD_ID set, looking for a pod named {args.pod_name!r}...")
        try:
            pod_id = find_pod_by_name(args.pod_name)
        except Exception as e:
            print(f"ERROR: could not list pods to look for {args.pod_name!r}: {e}", file=sys.stderr)
            print("Check the RunPod console for an orphaned pod", file=sys.stderr)
            return 1
        if not pod_id:
            print("No matching pod found, nothing to terminate")
            return 0
        print(f"Found orphaned pod {pod_id}")
    print(f"Terminating pod {pod_id}...")
    return 0 if terminate_pod(pod_id) else 1


def main():
    parser = argparse.ArgumentParser(description="Run GPU tests on RunPod")
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup = subparsers.add_parser("setup", help="Create a pod and install dependencies on it")
    setup.add_argument("--pod-name", default="ci-gpu-test", help="Pod name")
    setup.add_argument("--image", default=DEFAULT_IMAGE, help="Docker image")
    setup.add_argument("--gpu-type", default=DEFAULT_GPU, help="GPU type ID")
    setup.add_argument("--commit", default="HEAD", help="Git commit SHA to test")
    setup.add_argument("--test-commands", help="JSON array of pytest commands, validated before the pod is created")
    setup.add_argument("--timeout", type=int, default=DEFAULT_SETUP_TIMEOUT, help="Max seconds before the step gives up")

    test = subparsers.add_parser("test", help="Run test commands on the pod created by `setup`")
    test.add_argument("--test-commands", required=True, help="JSON array of pytest commands")
    test.add_argument("--timeout", type=int, default=DEFAULT_TEST_TIMEOUT, help="Max seconds before the step gives up")

    teardown = subparsers.add_parser("teardown", help="Terminate the pod created by `setup`")
    teardown.add_argument("--pod-name", help="Pod name to look up if POD_ID was never exported")
    teardown.add_argument("--pod-id", help="Pod to terminate, when POD_ID is not in the environment")

    args = parser.parse_args()
    handlers = {"setup": cmd_setup, "test": cmd_test, "teardown": cmd_teardown}

    try:
        sys.exit(handlers[args.command](args) or 0)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
