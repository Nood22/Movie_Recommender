#!/usr/bin/env bash
#SBATCH --job-name=v10_full_cohort
#SBATCH --partition=long
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --output=/network/scratch/a/adls/FullTrainingTEARS/full_ml32m_emiliano_prompt/full_cohort/v010_20260816_evidence_gated_emiliano_frozen/logs/slurm-v10-full-cohort-%j.out
#SBATCH --error=/network/scratch/a/adls/FullTrainingTEARS/full_ml32m_emiliano_prompt/full_cohort/v010_20260816_evidence_gated_emiliano_frozen/logs/slurm-v10-full-cohort-%j.err

set -euo pipefail

project_root=/home/mila/a/adls/tears_project_final
production_root=/network/scratch/a/adls/FullTrainingTEARS/full_ml32m_emiliano_prompt/full_cohort/v010_20260816_evidence_gated_emiliano_frozen

cd "$project_root"
export PYTHONPATH="$project_root"
export UV_CACHE_DIR=/tmp/tears-v10-production-uv-cache
export WANDB_SILENT=true
export PYTHONUNBUFFERED=1

uv run python scripts/run_frozen_v10_remaining_shards.py \
  --root "$production_root" \
  --config "$project_root/configs/ml32m.toml" \
  --auto-retry-objective-failures
