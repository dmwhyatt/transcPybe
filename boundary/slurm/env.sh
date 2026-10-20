# Sourced by every job: the A100 node environment, the project's venv, and this repository on the path.
# REPO and VENV can be overridden before sbatch (sbatch --export=ALL,REPO=...,VENV=...).
module purge
module load rhel8/default-amp
REPO=${REPO:-$HOME/transcPybe}
VENV=${VENV:-$HOME/venvs/boundary}
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
DATA=${DATA:-/rds/user/$USER/hpc-work/boundary}
cd "$REPO"
echo "$(date -u +%FT%TZ) $(hostname) job=${SLURM_JOB_ID:-none} repo=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)"
