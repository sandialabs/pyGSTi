#!/bin/bash

#SBATCH -J 2Q-XYC-16
#SBATCH -N 1
#SBATCH -p zen
#SBATCH -t 1-0
#SBATCH -x zen1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=16

./run.sh 16 1
./run.sh 16 2
./run.sh 16 4
./run.sh 16 8
./run.sh 16 16
