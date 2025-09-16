#!/bin/bash
#SBATCH --job-name=KAB_S4A_encoder_lrgrid
#SBATCH --time=2:20:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --account=def-ravanelm
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --array=0-8
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
set -euo pipefail

# -------------------- Load Dataset --------------------
cp "$HOME/projects/def-ravanelm/datasets/CL-MASR.tar.gz" "$SLURM_TMPDIR/"
cd "$SLURM_TMPDIR"
mkdir -p CL_MASR && tar -zxf CL-MASR.tar.gz -C CL_MASR
ls -l CL_MASR  # Check if files exist

# -------------------- Activate Environment --------------------
module load StdEnv/2023 gcc/12.3 intel/2023.2.1 gcccore/.12.3 ucc/1.2.0 ucx/1.14.1 openmpi/4.1.5 arrow/17.0.0 cuda/11.8
source "$HOME/projects/def-ravanelm/salmanhu/benchmarks/venv/bin/activate"

rank=104

# -------------------- LR Grid (around lr=1e-3, final=5e-5) --------------------
LR_LIST=(6e-4 1e-3 1.6e-3)     # initial learning rates
FINAL_LIST=(2e-5 5e-5 1e-4)    # scheduler floor (final_value)

NUM_LR=${#LR_LIST[@]}
NUM_FINAL=${#FINAL_LIST[@]}
combo_idx=$(( SLURM_ARRAY_TASK_ID % (NUM_LR * NUM_FINAL) ))
lr_idx=$(( combo_idx / NUM_FINAL ))
final_idx=$(( combo_idx % NUM_FINAL ))

LR0=${LR_LIST[$lr_idx]}
LRF=${FINAL_LIST[$final_idx]}

echo "===> Combo ${combo_idx}: lr0=${LR0}, final_value=${LRF}"

# -------------------- Fixed seeds (reproducible) --------------------
SEEDS=(138946508 40873422 290384992 90188442 3311940)

# -------------------- Train --------------------
cd "$HOME/projects/def-ravanelm/salmanhu/benchmarks/benchmarks/CL_MASR/whisper"

for s in "${SEEDS[@]}"; do
  echo "==> Training | seed=${s} | lr0=${LR0} | final_value=${LRF}"
  python train_ft_adapters.py hparams/KAB/encoder/KAB_S4A.yaml \
    --data_folder "$SLURM_TMPDIR/CL_MASR/CL-MASR" \
    --scratch_folder "$SCRATCH/whisper" \
    --seed "$s" \
    --projection_size "$rank" \
    --location "encoder" \
    --learning_rate "$LR0" \
    --final_value "$LRF" \
    --run_name "enc_lr${LR0}_final${LRF}_seed${s}"

  # If your script doesn't accept --learning_rate/--final_value,
  # use HyperPyYAML override instead (uncomment below):
  # python train_ft_adapters.py hparams/KAB/encoder/KAB_S4A.yaml \
  #   --data_folder "$SLURM_TMPDIR/CL_MASR/CL-MASR" \
  #   --scratch_folder "$SCRATCH/whisper" \
  #   --seed "$s" \
  #   --projection_size "$rank" \
  #   --location "encoder" \
  #   --override "learning_rate: ${LR0}, final_value: ${LRF}, run_name: enc_lr${LR0}_final${LRF}_seed${s}"
done