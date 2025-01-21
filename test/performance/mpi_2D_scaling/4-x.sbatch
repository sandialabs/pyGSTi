#!/bin/bash

#SBATCH -J 2Q-XYC-4
#SBATCH -N 1
#SBATCH -p zen
#SBATCH -t 1-0
#SBATCH -x zen1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=4

./run.sh 4 1
./run.sh 4 2
./run.sh 4 4
