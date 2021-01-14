#!/bin/sh
#
#SBATCH --job-name=train005_l4st0_hs50nl5np10_seq100nk10_h10sU10_mixed
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:v100:2
#SBATCH --time=48:00:00
#SBATCH --mem=20GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dq271@nyu.edu

module purge
module load python3/intel/3.6.3
module load cudnn/10.1v7.6.5.32
source ~/machine_learning/py3.6.3/bin/activate

cd /scratch/dq271/machine_learning/code
python3 -u train_model_para.py
