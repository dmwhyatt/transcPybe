# Sourced by every job: the node's module environment, the project's venv, this repository on the path, and the ledger.
# REPO, VENV and DATA can be overridden (submit.sh passes them). Jobs stay in the directory they were submitted from,
# so relative paths mean what they meant there; every stage also writes absolute paths into its manifest.
REPO=${REPO:-$HOME/transcPybe}
VENV=${VENV:-$HOME/venvs/boundary}
DATA=${DATA:-/rds/user/$USER/hpc-work/boundary}
# The module system and venv activate scripts read variables that may be unset; the jobs run with
# set -u, which would turn that into a failed job before Python starts. Relax it just for them.
_boundary_t0=$(date +%s)
case $- in *u*) _boundary_nounset=1 ;; *) _boundary_nounset=0 ;; esac
set +u
module purge
# The GPU (Ampere) set brings CUDA and its libraries; Ice Lake and other CPU nodes take their own default set. On
# 22 Oct an Ice Lake job spent about 2 minutes here loading the Ampere set; a GPU job took 7 s.
case "${SLURM_JOB_PARTITION:-}" in
    ampere*) module load rhel8/default-amp ;;
    *) module load rhel8/default-icl ;;
esac
source "$VENV/bin/activate"
[ "$_boundary_nounset" = 1 ] && set -u
export PYTHONPATH="$REPO:$REPO/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
# One thread per allocated core, as CSD3's PyTorch guide sets. CSD3's login environment sets OMP_NUM_THREADS=1 and
# sbatch carries it into the job, and nproc obeys OMP_NUM_THREADS, so count the job's cores without it: on 21 Oct
# every job ran single-threaded because of this (nproc said 1; the real count on a login node was 76).
export OMP_NUM_THREADS=$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)
export MKL_NUM_THREADS=$OMP_NUM_THREADS

# The ledger: one line when a job starts and one when it ends (any exit, including a time-out's SIGTERM),
# so status.sh can show what ran, for how long, and how many GPU hours it used.
mkdir -p "$DATA/logs"
case "${CUDA_VISIBLE_DEVICES:-}" in         # awk prints nothing for empty input, so count only when there is a list
    "") _boundary_gpus=0 ;;
    *) _boundary_gpus=$(printf '%s\n' "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}') ;;
esac
_boundary_ledger() {
    printf '%s\t%s\t%s\t%s\t%s\tgpus=%s\t%s\n' "$(date -u +%FT%TZ)" "${SLURM_JOB_ID:-none}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}" \
        "${SLURM_JOB_NAME:-interactive}" "$(hostname -s)" "$1" "$_boundary_gpus" "$2" >> "$DATA/ledger.tsv"
}
_boundary_ledger start "repo=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null) cwd=$PWD"
trap '_boundary_rc=$?; _boundary_ledger end "exit=$_boundary_rc elapsed_s=$(( $(date +%s) - _boundary_t0 ))"' EXIT
trap 'exit 143' TERM
echo "$(date -u +%FT%TZ) $(hostname) job=${SLURM_JOB_ID:-none} repo=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null) cwd=$PWD gpus=$_boundary_gpus cores=$OMP_NUM_THREADS setup=$(( $(date +%s) - _boundary_t0 ))s"
[ "$_boundary_gpus" != 0 ] && command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
