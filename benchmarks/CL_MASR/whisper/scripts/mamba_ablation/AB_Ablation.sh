#!/bin/bash
#SBATCH --job-name=AB_Ablation
#SBATCH --time=24:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --account=def-ravanelm
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --array=0-24
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -euo pipefail

cp $HOME/projects/def-ravanelm/datasets/CL-MASR.tar.gz $SLURM_TMPDIR/
cd $SLURM_TMPDIR
mkdir CL_MASR && tar -zxf CL-MASR.tar.gz -C CL_MASR
ls -l CL_MASR  # Check if files exist

module load StdEnv/2023  gcc/12.3 intel/2023.2.1 gcccore/.12.3 ucc/1.2.0 ucx/1.14.1 openmpi/4.1.5 arrow/17.0.0 cuda/11.8
source $HOME/projects/def-ravanelm/salmanhu/benchmarks/venv/bin/activate

# ---------- Slice indices ----------
read -r START_IDX END_IDX <<< "$(python - <<'PY'
import os
tid = int(os.environ.get("SLURM_ARRAY_TASK_ID","0"))
start = tid * 3
end = min(start + 3, 75)
print(start, end)
PY
)"

# ---------- Build grid ----------
GRID_JSON=$(python - <<'PY'
import itertools, json
kern = [4, 8, 16, 24, 32]
exp  = [2, 3, 4]
dst  = [8, 16, 24, 32, 64]
grid = list(itertools.product(dst, exp, kern))
print(json.dumps(grid))
PY
)

RANK=104
LOCATION="encoder"
PROJ_DIR="$HOME/projects/def-ravanelm/salmanhu/benchmarks/benchmarks/CL_MASR/whisper"
DATA_DIR="$SLURM_TMPDIR/CL_MASR/CL-MASR"
SCRATCH_DIR="$SCRATCH/whisper"
LOGDIR="$SCRATCH/whisper_logs/mamba_ablation_25x3"
mkdir -p "$LOGDIR"

cd "$PROJ_DIR"

echo "Task $SLURM_ARRAY_TASK_ID handles combo indices [$START_IDX, $END_IDX)"

# Helper: extract tuple by index
get_tuple () {
  local idx="$1"
  python - <<PY
import json, sys, os
grid = json.loads(os.environ["GRID_JSON"])
d,e,k = grid[int(sys.argv[1])]
print(d); print(e); print(k)
PY
  "$idx"
}

# ---------- Loop over 3 combos sequentially ----------
for (( IDX="$START_IDX"; IDX<"$END_IDX"; IDX++ )); do
  read -r D_STATE EXPAND KERNEL_SIZE < <(get_tuple "$IDX")

  # Random seed (torch.randint)
  SEED=$(python - <<'EOF'
import torch
print(torch.randint(0, 2**32-1, (1,)).item())
EOF
)

  echo "=== [$IDX] d_state=${D_STATE} expand=${EXPAND} kernel=${KERNEL_SIZE} seed=${SEED} ==="

  # --- AB_S4A ---
  tee "$LOGDIR/AB_idx${IDX}_d${D_STATE}_e${EXPAND}_k${KERNEL_SIZE}.log" < <(
    python train_ft_adapters.py hparams/mamba_ablation/AB_S4A.yaml \
      --data_folder "$DATA_DIR" \
      --scratch_folder "$SCRATCH_DIR" \
      --seed "$SEED" \
      --projection_size "$RANK" \
      --location "$LOCATION" \
      --d_state "$D_STATE" \
      --kernel_size "$KERNEL_SIZE" \
      --expand "$EXPAND"
  )

done