#!/bin/bash
#SBATCH -p gpu17
#SBATCH --gres gpu:2
#SBATCH --nodes 8
#SBATCH --ntasks-per-node 2
#SBATCH -t 0-08:00:0
#SBATCH -o train_vae_output_%j.log
#SBATCH -e train_vae_error_%j.log

# activate Conda environment
source ~/.bashrc
conda activate ldm-new  

cd ../vavae

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# print start time
echo $(date)
echo $MASTER_ADDR
echo $MASTER_PORT

# start command
srun python3 main.py \
    --train \
    --base configs/f16d32_vfdinov2.yaml \
    --logdir logs \
    --num_nodes $SLURM_NNODES
