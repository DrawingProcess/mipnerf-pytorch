#!/usr/bin/bash

#SBATCH -J mipnerf-custom-train-240516_classroom1_inpainting
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

# python train.py --dataset_name custom --log_dir log/custom/240516_classroom1_inpainting --scene 240516_classroom1_inpainting
# python train.py --dataset_name llff --log_dir log/llff/flower --scene flower 
# python visualize.py --dataset_name llff --model_weight_path log/llff/flower/model.pt --log_dir log/llff/flower --scene flower 
# python extract_mesh.py --model_weight_path log/llff/flower/model.pt --log_dir log/llff/flower

# # dataset 
# python train.py --dataset_name blender --log_dir log/blender/lego_top --scene lego --max_steps '40_000'
# python train.py --dataset_name blender --log_dir log/blender/lego
# python train.py --dataset_name blender --log_dir log/blender/lego_gau_25_2 --scene lego_gau_25_2
# python train.py --dataset_name blender --log_dir log/blender/lego_gau_25_2_80 --scene lego_gau_25_2 --max_steps '160_000'
# python train.py --dataset_name blender --log_dir log/blender/lego20 --max_steps '40_000'
# python train.py --dataset_name blender --log_dir log/blender/lego40 --max_steps '80_000'
# python train.py --dataset_name blender --log_dir log/blender/lego60 --max_steps '120_000'
# python train.py --dataset_name blender --log_dir log/blender/lego80 --max_steps '160_000'

# python eval.py --dataset_name blender --log_dir log/blender/lego_gau_25_2_80 --scene lego_gau_25_2 --continue_training
# python eval_depth.py --dataset_name blender --log_dir log/blender/lego20 --scene lego --continue_training
# python eval_depth.py --dataset_name blender --log_dir log/blender/lego_front --scene lego --continue_training
# python eval_depth.py --dataset_name blender --log_dir log/blender/lego_back --scene lego --continue_training
# python eval_depth.py --dataset_name blender --log_dir log/blender/lego_right --scene lego --continue_training
# python eval_depth.py --dataset_name blender --log_dir log/blender/lego_left --scene lego --continue_training
# python eval_depth.py --dataset_name blender --log_dir log/blender/lego_top --scene lego --continue_training

python visualize.py --dataset_name custom --model_weight_path log/custom/240516_classroom1_inpainting/model.pt --log_dir log/custom/240516_classroom1_inpainting --scene 240516_classroom1_inpainting --visualize_depth --visualize_normals
# python visualize.py --dataset_name blender --model_weight_path log/blender/lego20/model.pt --log_dir log/blender/lego20 --scene lego --visualize_depth --visualize_normals
# python visualize.py --dataset_name blender --model_weight_path log/blender/lego40/model.pt --log_dir log/blender/lego40 --scene lego --visualize_depth --visualize_normals
# python visualize.py --dataset_name blender --model_weight_path log/blender/lego60/model.pt --log_dir log/blender/lego60 --scene lego --visualize_depth --visualize_normals
# python visualize.py --dataset_name blender --model_weight_path log/blender/lego80/model.pt --log_dir log/blender/lego80 --scene lego --visualize_depth --visualize_normals
# python visualize.py --dataset_name blender --model_weight_path log/blender/lego/model.pt --log_dir log/blender/lego --scene lego --visualize_depth --visualize_normals
# python visualize.py --dataset_name multicam --model_weight_path log/multicam/lego/model.pt --log_dir log/multicam/lego --scene lego --visualize_depth --visualize_normals
# python visualize.py --dataset_name blender --model_weight_path log/blender/lego_gau_25_2/model.pt --log_dir log/blender/lego_gau_25_2 --scene lego 

# python visualize.py --dataset_name blender --model_weight_path log/blender/lego_front/model.pt --log_dir log/blender/lego_front --scene lego --visualize_depth --visualize_normals
# python visualize.py --dataset_name blender --model_weight_path log/blender/lego_back/model.pt --log_dir log/blender/lego_back --scene lego --visualize_depth --visualize_normals
# python visualize.py --dataset_name blender --model_weight_path log/blender/lego_right/model.pt --log_dir log/blender/lego_right --scene lego --visualize_depth --visualize_normals
# python visualize.py --dataset_name blender --model_weight_path log/blender/lego_left/model.pt --log_dir log/blender/lego_left --scene lego --visualize_depth --visualize_normals
# python visualize.py --dataset_name blender --model_weight_path log/blender/lego_top/model.pt --log_dir log/blender/lego_top --scene lego --visualize_depth --visualize_normals
# python extract_mesh.py --model_weight_path log/blender/lego_front/model.pt --log_dir log/blender/lego_front --scene lego 
# python extract_mesh.py --model_weight_path log/blender/lego_back/model.pt --log_dir log/blender/lego_back --scene lego 
# python extract_mesh.py --model_weight_path log/blender/lego_right/model.pt --log_dir log/blender/lego_right --scene lego 
# python extract_mesh.py --model_weight_path log/blender/lego_left/model.pt --log_dir log/blender/lego_left --scene lego 
# python extract_mesh.py --model_weight_path log/blender/lego_top/model.pt --log_dir log/blender/lego_top --scene lego 

# python extract_mesh.py --model_weight_path log/blender/lego/model.pt --log_dir log/blender/lego --scene lego 
# python extract_mesh.py --model_weight_path log/blender/lego_gau_25_2/model.pt --log_dir log/blender/lego_gau_25_2 
# python extract_mesh.py --model_weight_path log/blender/lego20/model.pt --log_dir log/blender/lego20 --scene lego 
# python extract_mesh.py --model_weight_path log/blender/lego40/model.pt --log_dir log/blender/lego40 --scene lego 
# python extract_mesh.py --model_weight_path log/blender/lego60/model.pt --log_dir log/blender/lego60 --scene lego 
# python extract_mesh.py --model_weight_path log/blender/lego80/model.pt --log_dir log/blender/lego80 --scene lego 
# python extract_mesh.py --model_weight_path log/multicam/lego/model.pt --log_dir log/multicam/lego --scene lego 
# dataset gaussian
# python train.py --dataset_name blender --log_dir log/blender/lego_gau_25_2 --scene lego_gau_25_2 
# python visualize.py --dataset_name blender --model_weight_path log/blender/lego_gau_25_2/model.pt --log_dir log/blender/lego_gau_25_2 --scene lego 
# python extract_mesh.py --model_weight_path log/blender/lego_gau_25_2/model.pt --log_dir log/blender/lego_gau_25_2

# python visualize.py --dataset_name blender --model_weight_path log/blender/lego20/model.pt --log_dir log/blender/lego20 --scene lego --visualize_depth --visualize_normals
# python extract_mesh.py --model_weight_path log/blender/lego20/model.pt --log_dir log/blender/lego20

# python visualize.py --dataset_name blender --model_weight_path log/blender/lego40/model.pt --log_dir log/blender/lego40 --scene lego 
# python extract_mesh.py --model_weight_path log/blender/lego40/model.pt --log_dir log/blender/lego40

# python visualize.py --dataset_name blender --model_weight_path log/blender/lego60/model.pt --log_dir log/blender/lego60 --scene lego 
# python extract_mesh.py --model_weight_path log/blender/lego60/model.pt --log_dir log/blender/lego60

# python visualize.py --dataset_name blender --model_weight_path log/blender/lego80/model.pt --log_dir log/blender/lego80 --scene lego 
# python extract_mesh.py --model_weight_path log/blender/lego80/model.pt --log_dir log/blender/lego80

# python visualize.py --dataset_name multicam --model_weight_path log/multicam/lego/model.pt --log_dir log/multicam/lego --scene lego 
# python extract_mesh.py --model_weight_path log/multicam/lego/model.pt --log_dir log/multicam/lego

exit 0
