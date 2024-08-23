#!/bin/bash

#SBATCH --job-name=mnist_train
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --time=20:00:00
#SBATCH --mem=32000M
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=prakhar.ganesh@mila.quebec

source activate ffb
export CUBLAS_WORKSPACE_CONFIG=:16:8

for seed1 in {0..100}
do
    python mnist-train.py --seed $seed1 --dataset mnist-noise-high --save-model
    python mnist-train.py --seed $seed1 --dataset mnist-mixup --save-model
done