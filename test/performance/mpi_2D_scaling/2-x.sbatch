#!/bin/bash

#SBATCH -J 2Q-XYC-2
#SBATCH -N 1
#SBATCH -p zen
#SBATCH -t 1-0
#SBATCH -x zen1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=2

./run.sh 2 1
./run.sh 2 2
