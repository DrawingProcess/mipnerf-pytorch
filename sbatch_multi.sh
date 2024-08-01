#!/usr/bin/bash

#SBATCH -J mipnerf-multicam-multi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=48G
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

# python train.py --dataset_name blender --log_dir log/blender/lego40
# python visualize.py --dataset_name blender --model_weight_path log/blender/lego40/model.pt --log_dir log/blender/lego40 --scene lego 
# python extract_mesh.py --model_weight_path log/blender/lego40/model.pt --log_dir log/blender/lego40

# python visualize.py --dataset_name blender --model_weight_path log/blender/lego60/model.pt --log_dir log/blender/lego60 --scene lego 
# python extract_mesh.py --model_weight_path log/blender/lego60/model.pt --log_dir log/blender/lego60

# python visualize.py --dataset_name blender --model_weight_path log/blender/lego80/model.pt --log_dir log/blender/lego80 --scene lego 
# python extract_mesh.py --model_weight_path log/blender/lego80/model.pt --log_dir log/blender/lego80

python train.py --dataset_name multicam --log_dir log/multicam/lego --continue_training
python visualize.py --dataset_name multicam --model_weight_path log/multicam/lego/model.pt --log_dir log/multicam/lego --scene lego 
python extract_mesh.py --model_weight_path log/multicam/lego/model.pt --log_dir log/multicam/lego

exit 0
