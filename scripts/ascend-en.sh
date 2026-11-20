#!/usr/bin/env bash
#SBATCH --job-name=manga-en
#SBATCH --cluster=ascend
#SBATCH --partition=nextgen
#SBATCH --account=PAS2836
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=manga-%j.log

# English chapter on the A100, all stages, with the environments on scratch:
#   cd ~/manga-translation-pipeline && sbatch scripts/ascend-en.sh
# Time: a 98-page chapter measured 37 min plus about 5 for the server to load, around
# 23 s per page. For a longer chapter: sbatch scripts/ascend-en.sh --time=02:00:00
# Slurm reads #SBATCH lines only from the submitted file, so they are repeated here;
# ascend.sh does the work. A MANGA_* variable set at submit time still wins.
envs=/fs/scratch/PAS2836/drink36/manga-envs
export MANGA_CONFIG="${MANGA_CONFIG:-configs/ascend-en.yaml}"
export MANGA_PYTHON="${MANGA_PYTHON:-$envs/models/bin/python}"
export MANGA_SERVER_PYTHON="${MANGA_SERVER_PYTHON:-$envs/vllm/bin/python}"
# Last command, so its exit code is the job's (and reaches a submit wrapper's notification).
bash "${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}/scripts/ascend.sh"
