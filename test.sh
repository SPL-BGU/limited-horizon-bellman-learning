#!/bin/bash
# A* Search testing script
# Run directly in terminal

# Common variables
model_dir="saved_models/reproduction"
results_dir="results/reproduction"
start_idx=0
end_idx=200
solution_time_cap=10

# =============================================================================
# Rubik's Cube 3x3 (cube3)
# =============================================================================

python search_methods/astar.py \
  --env cube3 \
  --states data/cube3/test/data_0.pkl \
  --model_dir $model_dir/cube3/current/ \
  --weight 0.6 \
  --batch_size 10000 \
  --results_dir $results_dir/cube3/ \
  --nnet_batch_size 10000 \
  --start_idx $start_idx \
  --end_idx $end_idx \
  --solution_time_cap $solution_time_cap \
  --language cpp

# =============================================================================
# 35-Puzzle (puzzle35)
# =============================================================================

python search_methods/astar.py \
  --env puzzle35 \
  --states data/puzzle35/tree/data_0.pkl \
  --model_dir $model_dir/puzzle35/current/ \
  --weight 0.8 \
  --batch_size 20000 \
  --results_dir $results_dir/puzzle35/ \
  --nnet_batch_size 10000 \
  --start_idx $start_idx \
  --end_idx $end_idx \
  --solution_time_cap $solution_time_cap \
  --language cpp

# =============================================================================
# Lights Out 7x7 (lightsout7)
# =============================================================================

python search_methods/astar.py \
  --env lightsout7 \
  --states data/lightsout7/test/data_0.pkl \
  --model_dir $model_dir/lightsout7/current/ \
  --weight 0.2 \
  --batch_size 1000 \
  --results_dir $results_dir/lightsout7/ \
  --nnet_batch_size 10000 \
  --start_idx $start_idx \
  --end_idx $end_idx \
  --solution_time_cap $solution_time_cap \
  --language cpp
