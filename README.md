# TLDR: Temporal Distance-aware Representations for Unsupervised Goal-Conditioned RL

This repository contains the official implementation of [TLDR: Unsupervised Goal-Conditioned RL via Temporal Distance-Aware Representations](https://heatz123.github.io/tldr/) by [Junik Bae](https://heatz123.github.io/), [Kwanyoung Park](https://kwanyoungpark.github.io/) and [Youngwoon Lee](https://youngwoon.github.io/).

## Requirements
- Python 3.8

## Installation

```
conda create --name tldr python=3.8
conda activate tldr
conda install pytorch==2.0.1 pytorch-cuda=11.8 patchelf -c pytorch -c nvidia
pip install -r requirements.txt --no-deps
pip install -e .
pip install -e garaged
pip install -e d4rl
```

## Commands to run experiments

### Ant
```
# TLDR:
python tests/main.py --env ant --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type preset --eval_plot_axis -50 50 -50 50 --trans_optimization_epochs 50 --n_epochs_per_log 100 --n_epochs_per_eval 1000 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --sac_max_buffer_size 1000000 --algo tldr --discrete 0 --dim_option 4 --goal_reaching 1 --eval_plot_axis -80 80 -80 80 --trans_minibatch_size 1024 --num_video_repeats 24 --dual_lam 3000 --q_layer_normalization 1 --exp_q_layer_normalization 1 --lr_te 5e-4 --sac_discount 0.97 --exploration_sac_discount 0.99 --description "tldr" --n_epochs 40002
# METRA:
python tests/main.py --env ant --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type preset --eval_plot_axis -50 50 -50 50 --trans_optimization_epochs 50 --n_epochs_per_log 100 --n_epochs_per_eval 1000 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --sac_max_buffer_size 1000000 --algo metra --goal_reaching 0 --discrete 0 --trans_minibatch_size 1024 --dim_option 2 --description "metra"
```

### HalfCheetah
```
# TLDR:
python tests/main.py --env half_cheetah --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type preset --trans_optimization_epochs 50 --n_epochs_per_log 100 --n_epochs_per_eval 1000 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --sac_max_buffer_size 1000000 --algo tldr --discrete 0 --dim_option 4 --trans_minibatch_size 1024 --goal_reaching 1 --num_video_repeats 24 --dual_lam 3000 --q_layer_normalization 1 --exp_q_layer_normalization 1 --lr_te 5e-4 --sac_discount 0.97 --exploration_sac_discount 0.99 --description "tldr" --n_epochs 40002
# METRA:
python tests/main.py --env half_cheetah --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type preset --trans_optimization_epochs 50 --n_epochs_per_log 100 --n_epochs_per_eval 1000 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --sac_max_buffer_size 1000000 --algo metra --goal_reaching 0 --discrete 1 --dim_option 16 --trans_minibatch_size 1024 --description "metra"
```

### AntMaze-Large
```
# TLDR:
python tests/main.py --env antmaze-large-play --max_path_length 300 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --trans_optimization_epochs 75 --n_epochs_per_log 100 --n_epochs_per_eval 1000 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --sac_max_buffer_size 1000000 --algo tldr --discrete 0 --dim_option 4 --trans_minibatch_size 1024 --goal_reaching 1 --num_video_repeats 24 --dual_lam 3000 --q_layer_normalization 1 --exp_q_layer_normalization 1 --lr_te 5e-4 --sac_discount 0.97 --exploration_sac_discount 0.99 --description "tldr" --n_epochs 40002
# METRA:
python tests/main.py --env antmaze-large-play --max_path_length 300 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --trans_optimization_epochs 75 --n_epochs_per_log 100 --n_epochs_per_eval 1000 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --sac_max_buffer_size 1000000 --algo metra --goal_reaching 0 --discrete 0 --dim_option 4 --trans_minibatch_size 1024 --description "metra"
```

### AntMaze-Ultra
```
# TLDR:
python tests/main.py --env antmaze-ultra-play --max_path_length 600 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --trans_optimization_epochs 150 --n_epochs_per_log 100 --n_epochs_per_eval 500 --n_epochs_per_save 500 --n_epochs_per_pt_save 500 --sac_max_buffer_size 1000000 --algo tldr --goal_reaching 1 --discrete 0 --dim_option 4 --trans_minibatch_size 1024 --goal_reaching 1 --num_video_repeats 24 --dual_lam 3000 --q_layer_normalization 1 --exp_q_layer_normalization 1 --lr_te 5e-4 --sac_discount 0.97 --exploration_sac_discount 0.99 --description "tldr" --n_epochs 40002
# METRA:
python tests/main.py --env antmaze-ultra-play --max_path_length 600 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --trans_optimization_epochs 150 --n_epochs_per_log 100 --n_epochs_per_eval 500 --n_epochs_per_save 500 --n_epochs_per_pt_save 500 --sac_max_buffer_size 1000000 --algo metra --goal_reaching 0 --discrete 0 --dim_option 4 --trans_minibatch_size 1024 --description "metra"
```

### Quadruped-Escape
```
# TLDR:
python tests/main.py --env dmc_quadruped_state_escape --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --trans_optimization_epochs 50 --n_epochs_per_log 100 --n_epochs_per_eval 1000 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --sac_max_buffer_size 1000000 --algo tldr --goal_reaching 1 --discrete 0 --dim_option 4 --trans_minibatch_size 1024 --eval_plot_axis -15 15 -15 15 --goal_reaching 1 --num_video_repeats 24 --dual_lam 3000 --q_layer_normalization 1 --exp_q_layer_normalization 1 --lr_te 5e-4 --sac_discount 0.97 --exploration_sac_discount 0.99 --description "tldr" --n_epochs 40002
# METRA:
python tests/main.py --env dmc_quadruped_state_escape --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --trans_optimization_epochs 50 --n_epochs_per_log 100 --n_epochs_per_eval 1000 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --sac_max_buffer_size 1000000 --algo metra --goal_reaching 0 --discrete 0 --dim_option 4 --trans_minibatch_size 1024 --eval_plot_axis -15 15 -15 15 --description "metra"
```

### Humanoid-Run
```
# TLDR:
python tests/main.py --env dmc_humanoid_state --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --trans_optimization_epochs 50 --n_epochs_per_log 100 --n_epochs_per_eval 1000 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --sac_max_buffer_size 1000000 --algo tldr --goal_reaching 1 --discrete 0 --dim_option 4 --trans_minibatch_size 1024 --eval_plot_axis -60 60 -60 60 --goal_reaching 1 --num_video_repeats 24 --dual_lam 3000 --q_layer_normalization 1 --exp_q_layer_normalization 1 --lr_te 5e-4 --sac_discount 0.97 --exploration_sac_discount 0.99 --description "tldr" --n_epochs 40002
# METRA:
python tests/main.py --env dmc_humanoid_state --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --trans_optimization_epochs 50 --n_epochs_per_log 100 --n_epochs_per_eval 1000 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --sac_max_buffer_size 1000000 --algo metra --goal_reaching 0 --discrete 0 --dim_option 4 --trans_minibatch_size 1024 --eval_plot_axis -60 60 -60 60 --description "metra"
```

### Quadruped (Pixel)
```
# TLDR:
python tests/main.py --env dmc_quadruped --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --video_skip_frames 2 --frame_stack 3 --sac_max_buffer_size 300000 --eval_plot_axis -15 15 -15 15 --algo tldr --goal_reaching 1 --discrete 0 --dim_option 4 --trans_optimization_epochs 200 --n_epochs_per_log 25 --n_epochs_per_eval 125 --n_epochs_per_save 125 --n_epochs_per_pt_save 125 --encoder 1 --sample_cpu 0 --goal_reaching 1 --num_video_repeats 24 --dual_lam 3000 --q_layer_normalization 1 --exp_q_layer_normalization 1 --lr_te 5e-4 --sac_discount 0.97 --exploration_sac_discount 0.99 --description "tldr" --n_epochs 40002
# METRA:
python tests/main.py --env dmc_quadruped --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --video_skip_frames 2 --frame_stack 3 --sac_max_buffer_size 300000 --eval_plot_axis -15 15 -15 15 --algo metra --goal_reaching 0 --discrete 0 --dim_option 4 --trans_optimization_epochs 200 --n_epochs_per_log 25 --n_epochs_per_eval 125 --n_epochs_per_save 125 --n_epochs_per_pt_save 125 --encoder 1 --sample_cpu 0 --description "metra"
```

### Kitchen (Pixel)
```
# TLDR:
python tests/main.py --env kitchen --max_path_length 50 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --num_video_repeats 1 --frame_stack 3 --sac_max_buffer_size 100000 --algo tldr --goal_reaching 1 --discrete 0 --dim_option 4 --sac_lr_a -1 --trans_optimization_epochs 100 --n_epochs_per_log 25 --n_epochs_per_eval 250 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --encoder 1 --sample_cpu 0 --goal_reaching 1 --num_video_repeats 24 --dual_lam 3000 --q_layer_normalization 1 --exp_q_layer_normalization 1 --lr_te 5e-4 --sac_discount 0.97 --exploration_sac_discount 0.99 --description "tldr" --n_epochs 40002
# METRA:
python tests/main.py --env kitchen --max_path_length 50 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --num_video_repeats 1 --frame_stack 3 --sac_max_buffer_size 100000 --algo metra --goal_reaching 0 --discrete 1 --dim_option 24 --sac_lr_a -1 --trans_optimization_epochs 100 --n_epochs_per_log 25 --n_epochs_per_eval 250 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --encoder 1 --sample_cpu 0 --description "metra"
```


If you use this code for your research, please consider citing our paper:
```
@article{bae2024tldr,
  title={TLDR: Unsupervised Goal-Conditioned RL via Temporal Distance-Aware Representations},
  author={Junik Bae and Kwanyoung Park and Youngwoon Lee},
  journal={arXiv preprint arXiv:2407.08464},
  year={2024}
}
```
