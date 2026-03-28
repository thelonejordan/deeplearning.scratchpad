#!/usr/bin/env python3
"""Orchestrator: create a RunPod GPU pod, run pytest over SSH, tear down."""

import argparse
import json
import os
import re
import signal
import sys
import time

import paramiko
import runpod

REPO_URL = "https://github.com/thelonejordan/deeplearning.scratchpad.git"
DEFAULT_IMAGE = "runpod/pytorch:1.0.3-cu1281-torch260-ubuntu2204"
DEFAULT_GPU = "NVIDIA GeForce RTX 4090"
DEFAULT_TIMEOUT = 20 * 60  # 20 minutes
POLL_INTERVAL = 10  # seconds


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


def create_gpu_pod(name, image, gpu_type, hf_token=None):
    env = {}
    if hf_token:
        env["HF_TOKEN"] = hf_token
    pod = runpod.create_pod(
        name=name,
        image_name=image,
        gpu_type_id=gpu_type,
        gpu_count=1,
        container_disk_in_gb=30,
        volume_in_gb=0,
        ports="22/tcp",
        start_ssh=True,
        env=env if env else None,
    )
    print(f"Created pod: {pod['id']}")
    return pod["id"]


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
    if stream:
        for line in iter(stdout.readline, ""):
            print(line, end="")
    exit_code = stdout.channel.recv_exit_status()
    if not stream:
        out = stdout.read().decode()
        err = stderr.read().decode()
        if out:
            print(out)
        if err:
            print(err, file=sys.stderr)
    return exit_code


def run_tests_on_pod(host, port, commit_sha, test_commands):
    client = paramiko.SSHClient()
    # RunPod pods are ephemeral and host keys are not known in advance
    client.set_missing_host_key_policy(paramiko.WarningPolicy())

    for attempt in range(6):
        try:
            client.connect(host, port=port, username="root", timeout=30)
            break
        except Exception as e:
            if attempt == 5:
                raise
            print(f"SSH connect attempt {attempt + 1} failed: {e}, retrying...")
            time.sleep(10)

    try:
        # Clone repo at specific commit
        ssh_exec(client, f"git clone {REPO_URL} /workspace/repo")
        ssh_exec(client, f"cd /workspace/repo && git checkout {commit_sha}")

        # Install uv and dependencies
        ssh_exec(client, "curl -LsSf https://astral.sh/uv/install.sh | sh")
        ssh_exec(client, "cd /workspace/repo && $HOME/.local/bin/uv python install 3.11")
        ssh_exec(client, "cd /workspace/repo && $HOME/.local/bin/uv sync --group dev")

        # HuggingFace login using the HF_TOKEN env var injected into the pod
        ssh_exec(client, 'cd /workspace/repo && $HOME/.local/bin/uv run huggingface-cli login --token "$HF_TOKEN"')

        # Display environment
        ssh_exec(client, "nvidia-smi")
        ssh_exec(client, "cd /workspace/repo && $HOME/.local/bin/uv run python env.py")

        # Run test commands
        final_exit_code = 0
        for cmd in test_commands:
            full_cmd = f"cd /workspace/repo && $HOME/.local/bin/uv run {cmd}"
            exit_code = ssh_exec(client, full_cmd)
            if exit_code != 0:
                print(f"FAILED (exit {exit_code}): {cmd}")
                final_exit_code = exit_code
            else:
                print(f"PASSED: {cmd}")

        return final_exit_code
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description="Run GPU tests on RunPod")
    parser.add_argument("--pod-name", default="ci-gpu-test", help="Pod name")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Docker image")
    parser.add_argument("--gpu-type", default=DEFAULT_GPU, help="GPU type ID")
    parser.add_argument("--commit", default="HEAD", help="Git commit SHA to test")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Max seconds before force-terminating pod")
    parser.add_argument("--test-commands", required=True, help="JSON array of pytest commands")
    args = parser.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY environment variable is required", file=sys.stderr)
        sys.exit(1)
    runpod.api_key = api_key

    commit_sha = validate_commit_sha(args.commit)
    test_commands = validate_test_commands(json.loads(args.test_commands))
    pod_id = None

    # Hard timeout for the entire execution to prevent unbounded cloud spend
    def timeout_handler(signum, frame):
        raise TimeoutError("Overall execution timeout exceeded")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.timeout)

    try:
        hf_token = os.environ.get("HF_TOKEN")
        pod_id = create_gpu_pod(args.pod_name, args.image, args.gpu_type, hf_token)
        host, port = wait_for_ssh(pod_id, timeout=min(args.timeout, 300))
        exit_code = run_tests_on_pod(host, port, commit_sha, test_commands)
        sys.exit(exit_code)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        signal.alarm(0)  # cancel alarm
        if pod_id:
            print(f"Terminating pod {pod_id}...")
            try:
                runpod.terminate_pod(pod_id)
                print("Pod terminated.")
            except Exception as e:
                print(f"WARNING: Failed to terminate pod {pod_id}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
