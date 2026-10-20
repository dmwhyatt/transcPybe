# Sourced by every job: the A100 node environment, the project's venv, and this repository on the path.
# REPO, VENV and DATA can be overridden before sbatch (sbatch --export=ALL,REPO=...,VENV=...).
# The jobs stay in the directory they were submitted from, so relative paths mean what they meant there;
# every stage also writes absolute paths into its manifest.
REPO=${REPO:-$HOME/transcPybe}
VENV=${VENV:-$HOME/venvs/boundary}
DATA=${DATA:-/rds/user/$USER/hpc-work/boundary}
# The module system and venv activate scripts read variables that may be unset; the jobs run with
# set -u, which would turn that into a failed job before Python starts. Relax it just for them.
case $- in *u*) _boundary_nounset=1 ;; *) _boundary_nounset=0 ;; esac
set +u
module purge
module load rhel8/default-amp
source "$VENV/bin/activate"
[ "$_boundary_nounset" = 1 ] && set -u
export PYTHONPATH="$REPO:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
echo "$(date -u +%FT%TZ) $(hostname) job=${SLURM_JOB_ID:-none} repo=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null) cwd=$PWD"
