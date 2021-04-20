#!/bin/bash

#SBATCH -J 2Q-XYC-32
#SBATCH -N 1
#SBATCH -p zen
#SBATCH -t 1-0
#SBATCH -x zen1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=32

./run.sh 32 1
./run.sh 32 2
./run.sh 32 4
./run.sh 32 8
./run.sh 32 16
./run.sh 32 32
