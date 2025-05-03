#!/bin/bash
#SBATCH --job-name=AB_S4A_proj_sweep
#SBATCH --array=0-10
#SBATCH --time=2:30:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --account=def-ravanelm
#SBATCH --gres=gpu:v100l:1
#SBATCH --nodes=1
#SBATCH --output=logs/AB/Scaling/array_%A_%a.out

# Load Dataset
cp $HOME/projects/def-ravanelm/datasets/CL-MASR.tar.gz $SLURM_TMPDIR/
cd $SLURM_TMPDIR
mkdir CL_MASR && tar -zxf CL-MASR.tar.gz -C CL_MASR
ls -l CL_MASR  # Check if files exist

# Activate Environment
module load StdEnv/2023  gcc/12.3 intel/2023.2.1 gcccore/.12.3 ucc/1.2.0 ucx/1.14.1 openmpi/4.1.5 arrow/17.0.0 cuda/11.8
source $HOME/projects/def-ravanelm/salmanhu/benchmarks/venv/bin/activate

# Train with 5 random seeds
cd $HOME/projects/def-ravanelm/salmanhu/benchmarks/benchmarks/CL_MASR/whisper

projection_sizes=(128 112 104 96 80 72 64 48 36 32 24 16)
rank=${projection_sizes[$SLURM_ARRAY_TASK_ID]}

python train_ft_adapters.py hparams/KAB/Encoder/KAB_S4A.yaml --data_folder $SLURM_TMPDIR/CL_MASR/CL-MASR --seed 0 --projection_size $rank --location "encoder"