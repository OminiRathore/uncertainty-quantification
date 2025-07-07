#!/bin/bash

#SBATCH --job-name=semantic_seg
#SBATCH --output=/p/project1/delia-mp/omini/systemOUTTest_en
#SBATCH --error=/p/project1/delia-mp/omini/systemERR_en
#SBATCH --time=20:00:00
#SBATCH --account=delia-mp
#SBATCH --partition=dc-gpu

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2  
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2

export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

export MASTER_PORT=54312
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"

# Allow communication over InfiniBand cells.

MASTER_ADDR="${MASTER_ADDR}i"

# Get IP for hostname.

export MASTER_ADDR="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')"

echo $MASTER_ADDR

export NCCL_DEBUG=INFO

export CUDA_VISIBLE_DEVICES=0,1

source /p/project1/delia-mp/omini/sc_venv_template/activate.sh


echo 'running python script'
srun python3 /p/project1/delia-mp/omini/uncertainty_estimaion_in_segmentation/ensemble.py --data_path /p/scratch/delia-mp/rathore1/datasets/COSTA_dataset_v1/COSTA-Dataset-v1/IXI-Guys/ --save_path /p/scratch/delia-mp/rathore1/ensemble/output-L7-M3/ --train_step 1 --samples 20 --num_submodels 3 --num_layers 7 --ckpt /p/scratch/delia-mp/rathore1/trained_models/MRA_ensemble1-lr=0.001-epoch=200-val_loss=0.02-L=7-M=3-T20-W10.0
