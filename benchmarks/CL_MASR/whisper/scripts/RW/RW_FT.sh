#!/bin/bash
#SBATCH --mail-user=s_ssaina@live.concordia.ca
#SBATCH --mail-type=ALL

# Load Dataset
cp $HOME/projects/def-ravanelm/datasets/CL-MASR.tar.gz $SLURM_TMPDIR/
cd $SLURM_TMPDIR
mkdir CL_MASR && tar -zxf CL-MASR.tar.gz -C CL_MASR
ls -l CL_MASR  # Check if files exist

# Activate Environment
module load StdEnv/2023  gcc/12.3 intel/2023.2.1 gcccore/.12.3 ucc/1.2.0 ucx/1.14.1 openmpi/4.1.5 arrow/17.0.0 cuda/11.8
source $HOME/projects/def-ravanelm/salmanhu/benchmarks/venv/bin/activate

# Train
cd $HOME/projects/def-ravanelm/salmanhu/benchmarks/benchmarks/CL_MASR/whisper
python train_ft.py hparams/RW/RW_FT.yaml --data_folder $SLURM_TMPDIR/CL_MASR/CL-MASR