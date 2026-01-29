#!/bin/bash
# Training script for LHBL/SSBL
# Run directly in terminal

# Common variables
save_folder="saved_models/reproduction"
seed=15
max_update_steps=50  # Change this for different LHBL horizons

# =============================================================================
# Rubik's Cube 3x3 (cube3)
# =============================================================================

# SSBL (Single-Step Bellman Learning)
python ctg_approx/avi.py \
  --env cube3 \
  --nnet_name $save_folder \
  --states_per_update 50000000 \
  --batch_size 10000 \
  --max_itrs 1000000 \
  --back_max 30 \
  --num_update_procs 5 \
  --train_method bellman \
  --update_method astar \
  --seed $seed \
  --max_update_steps 1 \
  --update_target_threshold 5 \
  --update_batch_size 10000 \
  --astar_batch_size 10000 \
  --reach_max_steps

# LHBL(50) (Limited-Horizon Bellman Learning with horizon=50)
python ctg_approx/avi.py \
  --env cube3 \
  --nnet_name $save_folder \
  --states_per_update 50000000 \
  --batch_size 10000 \
  --max_itrs 1000000 \
  --back_max 30 \
  --num_update_procs 5 \
  --train_method bellman \
  --update_method astar \
  --seed $seed \
  --max_update_steps $max_update_steps \
  --update_target_threshold 5 \
  --update_batch_size 10000 \
  --astar_batch_size 10000 \
  --reach_max_steps \
  --limited_horizon_update

# =============================================================================
# 35-Puzzle (puzzle35)
# =============================================================================

# SSBL (Single-Step Bellman Learning)
python ctg_approx/avi.py \
  --env puzzle35 \
  --nnet_name $save_folder \
  --states_per_update 50000000 \
  --batch_size 10000 \
  --max_itrs 1000000 \
  --back_max 1000 \
  --num_update_procs 5 \
  --train_method bellman \
  --update_method astar \
  --seed $seed \
  --max_update_steps 1 \
  --update_target_threshold 1 \
  --update_batch_size 10000 \
  --astar_batch_size 1 \
  --reach_max_steps

# LHBL(50) (Limited-Horizon Bellman Learning with horizon=50)
python ctg_approx/avi.py \
  --env puzzle35 \
  --nnet_name $save_folder \
  --states_per_update 50000000 \
  --batch_size 10000 \
  --max_itrs 1000000 \
  --back_max 1000 \
  --num_update_procs 5 \
  --train_method bellman \
  --update_method astar \
  --seed $seed \
  --max_update_steps $max_update_steps \
  --update_target_threshold 1 \
  --update_batch_size 10000 \
  --astar_batch_size 1 \
  --reach_max_steps \
  --limited_horizon_update

# =============================================================================
# Lights Out 7x7 (lightsout7)
# =============================================================================

# SSBL (Single-Step Bellman Learning)
python ctg_approx/avi.py \
  --env lightsout7 \
  --nnet_name $save_folder \
  --states_per_update 500000 \
  --batch_size 1000 \
  --max_itrs 1000000 \
  --back_max 50 \
  --num_update_procs 5 \
  --train_method bellman \
  --update_method astar \
  --seed $seed \
  --max_update_steps 1 \
  --update_target_threshold 1 \
  --update_batch_size 10000 \
  --astar_batch_size 1 \
  --num_test 1000 \
  --reach_max_steps

# LHBL(50) (Limited-Horizon Bellman Learning with horizon=50)
python ctg_approx/avi.py \
  --env lightsout7 \
  --nnet_name $save_folder \
  --states_per_update 500000 \
  --batch_size 1000 \
  --max_itrs 1000000 \
  --back_max 50 \
  --num_update_procs 5 \
  --train_method bellman \
  --update_method astar \
  --seed $seed \
  --max_update_steps $max_update_steps \
  --update_target_threshold 1 \
  --update_batch_size 10000 \
  --astar_batch_size 1 \
  --num_test 1000 \
  --reach_max_steps \
  --limited_horizon_update
