#!/usr/bin/bash

#SBATCH -J mipnerf-blender-lego20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -t 3-0
#SBATCH -o logs/slurm-%A.out

squeue --job $SLURM_JOBID
cat $0

pwd
which python
hostname

# python train.py --dataset_name llff --log_dir log/llff/flower --scene flower 
# python visualize.py --dataset_name llff --model_weight_path log/llff/flower/model.pt --log_dir log/llff/flower --scene flower 
# python extract_mesh.py --model_weight_path log/llff/flower/model.pt --log_dir log/llff/flower

# # dataset 
# python train.py --dataset_name blender --log_dir log/blender/lego20 --max_steps '40_000'
python train.py --dataset_name blender --log_dir log/blender/lego40 --max_steps '80_000'
# python train.py --dataset_name blender --log_dir log/blender/lego60 --max_steps '120_000'
# python train.py --dataset_name blender --log_dir log/blender/lego80 --max_steps '160_000'

# dataset gaussian
# python train.py --dataset_name blender --log_dir log/blender/lego_gau_25_2 --scene lego_gau_25_2 
# python visualize.py --dataset_name blender --model_weight_path log/blender/lego_gau_25_2/model.pt --log_dir log/blender/lego_gau_25_2 --scene lego 
# python extract_mesh.py --model_weight_path log/blender/lego_gau_25_2/model.pt --log_dir log/blender/lego_gau_25_2

# python visualize.py --dataset_name blender --model_weight_path log/blender/lego40/model.pt --log_dir log/blender/lego40 --scene lego 
# python extract_mesh.py --model_weight_path log/blender/lego40/model.pt --log_dir log/blender/lego40

# python visualize.py --dataset_name blender --model_weight_path log/blender/lego60/model.pt --log_dir log/blender/lego60 --scene lego 
# python extract_mesh.py --model_weight_path log/blender/lego60/model.pt --log_dir log/blender/lego60

# python visualize.py --dataset_name blender --model_weight_path log/blender/lego80/model.pt --log_dir log/blender/lego80 --scene lego 
# python extract_mesh.py --model_weight_path log/blender/lego80/model.pt --log_dir log/blender/lego80

exit 0
