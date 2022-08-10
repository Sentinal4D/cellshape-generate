#!/bin/bash
#SBATCH --job-name=cell_vae16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH --time=96:00:00
#SBATCH --output=/data/scratch/DBI/DUDBI/DYNCESYS/mvries/cellshape-generate/outputs/output_%j.out
#SBATCH --error=/data/scratch/DBI/DUDBI/DYNCESYS/mvries/cellshape-generate/errors/error_%j.err
#SBATCH --partition=gpuhm
module load anaconda/3
source /opt/software/applications/anaconda/3/etc/profile.d/conda.sh

conda activate cs


python ../cellshape_generate/main.py --cloud_dataset_path '/data/scratch/DBI/DUDBI/DYNCESYS/mvries/SingleCellFromNathan_17122021/' --dataframe_path '/data/scratch/DBI/DUDBI/DYNCESYS/mvries/SingleCellFromNathan_17122021/all_data_removedwrong_ori_removedTwo.csv' --output_dir '/data/scratch/DBI/DUDBI/DYNCESYS/mvries/cellshape-generate/' --num_features 16 --learning_rate 0.00001 --num_epochs 250 --pretained_path "/data/scratch/DBI/DUDBI/DYNCESYS/mvries/cellshape-generate/nets/dgcnn_foldingnetbasic_16_pretrained_001.pt" 

