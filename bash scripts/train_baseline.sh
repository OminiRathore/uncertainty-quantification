#!/bin/bash

#SBATCH --job-name=semantic_seg
#SBATCH --output=/p/project1/delia-mp/omini/systemOUTTest
#SBATCH --error=/p/project1/delia-mp/omini/systemERR
#SBATCH --time=03:30:00
#SBATCH --account=delia-mp
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1

export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

export MASTER_PORT=54312
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"

# Allow communication over InfiniBand cells.

MASTER_ADDR="${MASTER_ADDR}i"

# Get IP for hostname.

export MASTER_ADDR="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')"

echo $MASTER_ADDR

export NCCL_DEBUG=INFO

export CUDA_VISIBLE_DEVICES=0

source /p/project1/delia-mp/omini/sc_venv_template/activate.sh

echo 'running python script'

srun python3 /p/project1/delia-mp/omini/uncertainty_estimaion_in_segmentation/baseline.py --data_path /p/scratch/delia-mp/rathore1/datasets/COSTA_dataset_v1/COSTA-Dataset-v1/IXI-Guys --save_path /p/scratch/delia-mp/rathore1/baseline/output-L4-finalConn/