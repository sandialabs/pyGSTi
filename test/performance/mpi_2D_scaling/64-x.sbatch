#!/bin/bash

#SBATCH -J 2Q-XYC-64
#SBATCH -N 1
#SBATCH -p zen
#SBATCH -t 1-0
#SBATCH -x zen1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=64

./run.sh 64 1
./run.sh 64 2
./run.sh 64 4
./run.sh 64 8
./run.sh 64 16
./run.sh 64 32
./run.sh 64 64
