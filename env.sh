# Source this file before running anything in this repo:  `source env.sh`
#
# It sets up the three isolated pieces of the setup:
#   * LIBERO_CONFIG_PATH - avoids the interactive prompt on `import libero`
#   * PYTHONPATH         - exposes third_party/libero to the client
#   * MUJOCO_GL          - offscreen rendering (no libOSMesa on this box; use `glx` if EGL fails)
#
# The server (Python 3.11) and the LIBERO client (Python 3.8) are separate venvs and must
# be activated explicitly:
#   source .venv/bin/activate                  # server / policy
#   source examples/libero/.venv/bin/activate  # LIBERO simulation client

export STREAMINGVLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LIBERO_CONFIG_PATH=/tmp/libero
export PYTHONPATH="$PYTHONPATH:$STREAMINGVLA_ROOT/third_party/libero"
export MUJOCO_GL=egl

# Hugging Face is not reachable directly from this network; use the mirror.
export HF_ENDPOINT=https://hf-mirror.com

# Pick the GPU that actually has free memory (SVLA server needs ~12-15 GiB).
# export CUDA_VISIBLE_DEVICES=0

# Keep JAX (used by the pi05_libero baseline) from pre-allocating the whole card.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6

echo "StreamingVLA env loaded (root=$STREAMINGVLA_ROOT)"
