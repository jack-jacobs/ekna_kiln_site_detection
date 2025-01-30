#!/bin/bash
#SBATCH -J ekna_inference
#SBATCH --partition=low_gpu_tesla,low_gpu_a40,low_gpu_titan
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH -o logs/inf_output.log
#SBATCH -e logs/inf_error.log
#SBATCH --time=4:00:00

# Run this script from ~/ekna_kiln_detect

# Activate conda environment for tiling
source ~/miniconda3/etc/profile.d/conda.sh
conda activate detectron2_tesla

python ~/ekna_kiln_detect/test_2024/run_inference.py $1
