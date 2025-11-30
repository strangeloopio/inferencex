# Multi-GPU vLLM inference deployments for benchmarking
# Creates separate endpoints for 1-GPU, 2-GPU, and 4-GPU configurations
# Includes real-time GPU power monitoring via nvidia-smi
#
# Configuration via environment variables (set before deploying):
#   MODEL_NAME: HuggingFace model name (default: Qwen/Qwen3-8B-FP8)
#   MODEL_REVISION: Model revision/commit (default: 220b46e3b2180893580a4454f21f22d3ebb187d3)
#   GPU_TYPE: GPU type - H100, H200, B200, A100 (default: H100)

import os
import modal

# =============================================================================
# CONFIGURATION - Modify these values or set via environment variables
# =============================================================================
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B-FP8")
MODEL_REVISION = os.environ.get("MODEL_REVISION", "220b46e3b2180893580a4454f21f22d3ebb187d3")
GPU_TYPE = os.environ.get("GPU_TYPE", "H100")

# =============================================================================
# INFRASTRUCTURE SETUP
# =============================================================================
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .pip_install(
        "vllm==0.11.2",
        "huggingface-hub==0.36.0",
        "flashinfer-python==0.5.2",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

# Cache volumes
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
power_logs_vol = modal.Volume.from_name("power-logs", create_if_missing=True)

FAST_BOOT = True
MINUTES = 60
VLLM_PORT = 8000
POWER_LOG_DIR = "/power_logs"

app = modal.App("vllm-multi-gpu-benchmark")


def get_vllm_cmd(n_gpu: int) -> list:
    """Generate vLLM serve command"""
    cmd = [
        "vllm", "serve", "--uvicorn-log-level=info",
        MODEL_NAME, "--revision", MODEL_REVISION,
        "--served-model-name", MODEL_NAME, "llm",
        "--host", "0.0.0.0", "--port", str(VLLM_PORT),
        "--enforce-eager" if FAST_BOOT else "--no-enforce-eager",
        "--tensor-parallel-size", str(n_gpu),
    ]
    return cmd


def start_power_monitor(n_gpu: int, interval: float = 1.0):
    """Start background power monitoring using nvidia-smi"""
    import subprocess
    import threading
    from datetime import datetime

    log_file = f"{POWER_LOG_DIR}/power_log_{n_gpu}gpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    def monitor():
        # Write header
        with open(log_file, "w") as f:
            f.write("timestamp,gpu_id,power_draw_w,temperature_c,gpu_util_pct,mem_util_pct,mem_used_mb\n")
        power_logs_vol.commit()

        while True:
            try:
                result = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=index,power.draw,temperature.gpu,utilization.gpu,utilization.memory,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    timestamp = datetime.now().isoformat()
                    with open(log_file, "a") as f:
                        for line in result.stdout.strip().split("\n"):
                            f.write(f"{timestamp},{line.strip()}\n")
                    # Commit to persist to volume (every 10 seconds)
                    power_logs_vol.commit()
            except Exception as e:
                print(f"Power monitoring error: {e}")

            import time
            time.sleep(interval)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    print(f"Power monitoring started, logging to {log_file}")


# =============================================================================
# 1-GPU ENDPOINT
# =============================================================================
@app.function(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:1",
    scaledown_window=15 * MINUTES,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        POWER_LOG_DIR: power_logs_vol,
    },
)
@modal.concurrent(max_inputs=128)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve_1gpu():
    import subprocess
    start_power_monitor(n_gpu=1)
    cmd = get_vllm_cmd(n_gpu=1)
    print(f"Model: {MODEL_NAME} (revision: {MODEL_REVISION})")
    print(f"GPU: {GPU_TYPE} x 1")
    print(f"Command: {' '.join(cmd)}")
    subprocess.Popen(" ".join(cmd), shell=True)


# =============================================================================
# 2-GPU ENDPOINT
# =============================================================================
@app.function(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:2",
    scaledown_window=15 * MINUTES,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        POWER_LOG_DIR: power_logs_vol,
    },
)
@modal.concurrent(max_inputs=128)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve_2gpu():
    import subprocess
    start_power_monitor(n_gpu=2)
    cmd = get_vllm_cmd(n_gpu=2)
    print(f"Model: {MODEL_NAME} (revision: {MODEL_REVISION})")
    print(f"GPU: {GPU_TYPE} x 2")
    print(f"Command: {' '.join(cmd)}")
    subprocess.Popen(" ".join(cmd), shell=True)


# =============================================================================
# 4-GPU ENDPOINT
# =============================================================================
@app.function(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:4",
    scaledown_window=15 * MINUTES,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        POWER_LOG_DIR: power_logs_vol,
    },
)
@modal.concurrent(max_inputs=128)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve_4gpu():
    import subprocess
    start_power_monitor(n_gpu=4)
    cmd = get_vllm_cmd(n_gpu=4)
    print(f"Model: {MODEL_NAME} (revision: {MODEL_REVISION})")
    print(f"GPU: {GPU_TYPE} x 4")
    print(f"Command: {' '.join(cmd)}")
    subprocess.Popen(" ".join(cmd), shell=True)


# =============================================================================
# POWER LOGS DOWNLOAD
# =============================================================================
@app.function(
    volumes={POWER_LOG_DIR: power_logs_vol},
)
def list_power_logs():
    """List all power log files"""
    import os
    logs = []
    for f in os.listdir(POWER_LOG_DIR):
        if f.endswith(".csv"):
            path = os.path.join(POWER_LOG_DIR, f)
            size = os.path.getsize(path)
            logs.append({"name": f, "size_bytes": size})
    return logs


@app.function(
    volumes={POWER_LOG_DIR: power_logs_vol},
)
def get_power_log(filename: str):
    """Get contents of a power log file"""
    path = os.path.join(POWER_LOG_DIR, filename)
    with open(path, "r") as f:
        return f.read()


@app.function(
    volumes={POWER_LOG_DIR: power_logs_vol},
)
def clear_power_logs():
    """Clear all power log files"""
    import os
    count = 0
    for f in os.listdir(POWER_LOG_DIR):
        if f.endswith(".csv"):
            os.remove(os.path.join(POWER_LOG_DIR, f))
            count += 1
    return f"Deleted {count} log files"


# =============================================================================
# LOCAL ENTRYPOINT
# =============================================================================
@app.local_entrypoint()
def main(action: str = "info"):
    """
    Manage vLLM deployment and power logs.

    Actions:
        info    - Show configuration and endpoints (default)
        logs    - List power log files
        download <filename> - Download a specific power log
        clear   - Clear all power logs
    """
    import sys

    if action == "info":
        print("="*60)
        print("MULTI-GPU vLLM DEPLOYMENT")
        print("="*60)
        print(f"Model: {MODEL_NAME}")
        print(f"Revision: {MODEL_REVISION}")
        print(f"GPU Type: {GPU_TYPE}")
        print("="*60)
        print("\nEndpoints:")
        print(f"  1 GPU: {serve_1gpu.web_url}")
        print(f"  2 GPU: {serve_2gpu.web_url}")
        print(f"  4 GPU: {serve_4gpu.web_url}")
        print("\nPower logs are saved to the 'power-logs' volume.")
        print("Use 'modal run vllm_multi_gpu.py logs' to list them.")

    elif action == "logs":
        print("Power log files:")
        logs = list_power_logs.remote()
        if not logs:
            print("  No logs found")
        else:
            for log in logs:
                print(f"  {log['name']} ({log['size_bytes']} bytes)")

    elif action == "clear":
        result = clear_power_logs.remote()
        print(result)

    elif action.startswith("download"):
        parts = action.split(maxsplit=1)
        if len(parts) < 2:
            print("Usage: modal run vllm_multi_gpu.py 'download <filename>'")
            return
        filename = parts[1]
        content = get_power_log.remote(filename)
        # Save locally
        with open(filename, "w") as f:
            f.write(content)
        print(f"Downloaded {filename}")

    else:
        print(f"Unknown action: {action}")
        print("Available actions: info, logs, download <filename>, clear")
