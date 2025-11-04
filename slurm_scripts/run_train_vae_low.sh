#!/bin/bash
#SBATCH -p <partition_name> 
#SBATCH --gres gpu:<num_gpus>
#SBATCH --nodes <num_nodes> 
#SBATCH --ntasks-per-node <tasks_per_node>
#SBATCH -t <days>-<hours>:<minutes>:<seconds>
#SBATCH -o train_vae_low_output_%j.log
#SBATCH -e train_vae_low_error_%j.log

# activate Conda environment
source ~/.bashrc
conda activate ldm

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
    --base configs/f16d32_vfdinov2_low.yaml \
    --logdir logs \
    --num_nodes $SLURM_NNODES
