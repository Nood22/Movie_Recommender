#!/usr/bin/env bash
# Same discovery name: the gateway can serve the actual models on CPU.
#SBATCH --job-name=tears_pilot_api
#SBATCH --partition=long-cpu,long-cpu-eek
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=/home/mila/a/adls/tears_project_final/logs/tears-pilot-api-cpu-%j.out
#SBATCH --error=/home/mila/a/adls/tears_project_final/logs/tears-pilot-api-cpu-%j.err

# Reuse the process supervision and requeue handling of GPU serving.
source /home/mila/a/adls/tears_project_final/slurm/tears_pilot_api.sbatch
