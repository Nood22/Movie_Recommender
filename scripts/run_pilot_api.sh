#!/usr/bin/env bash
# Some compute nodes no longer provide the system Python used by .venv.
set -euo pipefail
cd /home/mila/a/adls/tears_project_final
# CPU partitions can use spare CPU capacity on nodes that also contain GPUs.
# Keep inference on the resources allocated to this job.
if [[ "${SLURM_JOB_PARTITION:-}" == *cpu* ]]; then
  export CUDA_VISIBLE_DEVICES=""
fi
if [[ -x .venv/bin/python ]]; then
  exec .venv/bin/python -m uvicorn pilot_api:app --host 0.0.0.0 --port 8010
fi
pilot_python=/cvmfs/ai.mila.quebec/apps/arch/distro/python/3.10/bin/python3
# The fallback interpreter needs its matching libraries on newer compute nodes.
pilot_distro=/cvmfs/ai.mila.quebec/apps/arch/distro
export LD_LIBRARY_PATH="$pilot_distro/OpenSSL/1.1/lib:$pilot_distro/libffi/3.2.1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PWD/.venv/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
exec "$pilot_python" -m uvicorn pilot_api:app --host 0.0.0.0 --port 8010
