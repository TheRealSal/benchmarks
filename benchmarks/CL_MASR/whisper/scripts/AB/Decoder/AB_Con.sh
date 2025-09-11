#!/bin/bash
#SBATCH --job-name=AB_Con_decoder
#SBATCH --time=2:20:0
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --account=def-ravanelm
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1

# Load Dataset
cp $HOME/projects/def-ravanelm/datasets/CL-MASR.tar.gz $SLURM_TMPDIR/
cd $SLURM_TMPDIR
mkdir CL_MASR && tar -zxf CL-MASR.tar.gz -C CL_MASR
ls -l CL_MASR  # Check if files exist

# Activate Environment
module load StdEnv/2023  gcc/12.3 intel/2023.2.1 gcccore/.12.3 ucc/1.2.0 ucx/1.14.1 openmpi/4.1.5 arrow/17.0.0 cuda/11.8
source $HOME/projects/def-ravanelm/salmanhu/benchmarks/venv/bin/activate

rank=48

# Train with 5 random seeds
cd $HOME/projects/def-ravanelm/salmanhu/benchmarks/benchmarks/CL_MASR/whisper
for i in {1..5}; do
    seed=$(python - <<EOF
import torch
print(torch.randint(0,2**32-1,(1,)).item())
EOF
)
    echo "Training with seed $seed"
    python train_ft_adapters.py hparams/AB/decoder/AB_Con.yaml --data_folder $SLURM_TMPDIR/CL_MASR/CL-MASR --scratch_folder $SCRATCH/whisper --seed $seed --projection_size $rank --location "decoder"
done