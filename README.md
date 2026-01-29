# Limited-Horizon Bellman Learning (LHBL)
This is the code for [LHBL](https://arxiv.org/abs/2511.10264) for python3 and PyTorch.

This currently contains the code for using LHBL and DeepCubeA to solve the Rubik's cube, 15-puzzle, 24-puzzle, 35-puzzle, 48-puzzle, and Lights Out.

This is a version of the [Original DeepCubeA](https://github.com/forestagostinelli/DeepCubeA) repository.

# Setup
For required python packages, please see requirements.txt OR cube.yml.
You should be able to install these packages with pip or conda.

Python version used: 3.7.2

# Training and A* Search

Training uses Approximate Value Iteration (AVI) to learn a cost-to-go (heuristic) function. The learned heuristic is then used with A* search to solve puzzle instances.

There are two training methods available:
- **SSBL (Single-step Bellman Learning)**: Single-step Bellman backup updates
- **LHBL (Limited-Horizon Bellman Learning)**: Multi-step updates using graph-based Bellman backups within a search horizon, which can improve heuristic quality

There are pre-trained models in the `saved_models/` directory as well as `output.txt` files to let you know what output to expect.

There are pre-computed results of A* search in the `results/` directory.

### Commands to train LHBL to solve the 15-puzzle

###### Train cost-to-go function (LHBL(10) - Limited-Horizon Bellman Learning with horizon=50)
`python ctg_approx/avi.py --env puzzle15 --states_per_update 50000000 --batch_size 10000 --nnet_name puzzle15 --max_itrs 1000000 --loss_thresh 0.1 --back_max 500 --num_update_procs 30 --max_update_steps 50 --update_method ASTAR --limited_horizon_update`

###### Train cost-to-go function (SSBL)
`python ctg_approx/avi.py --env puzzle15 --states_per_update 50000000 --batch_size 10000 --nnet_name puzzle15 --max_itrs 1000000 --loss_thresh 0.1 --back_max 500 --num_update_procs 30`

The key additional flags for LHBL are:
- `--limited_horizon_update`: Enable limited-horizon Bellman backup
- `--max_update_steps`: Number of search steps to take (search horizon)
- `--update_method ASTAR`: Use A* search during updates (alternative: GBFS)

###### Solve with A* search, use --verbose for more information
`python search_methods/astar.py --states data/puzzle15/test/data_0.pkl --model_dir saved_models/puzzle15/current/ --env puzzle15 --weight 0.8 --batch_size 20000 --results_dir results/puzzle15/ --language cpp --nnet_batch_size 10000`

### Improving Results
During approximate value iteration (AVI), one can get better results by increasing the batch size (`--batch_size`) and number of states per update (`--states_per_update`).
Decreasing the threshold before the target network is updated (`--loss_thresh`) can also help.

One can also add additional states to training set by doing greedy best-first search (GBFS) during the update stage and adding the states encountered during GBFS to the states used for approximate value iteration (`--max_update_steps`). Setting `--max_update_steps` to 1 is the same as doing approximate value iteration.

During A* search, increasing the weight on the path cost (`--weight`, range should be [0,1]) and the batch size (`--batch_size`) generally improves results.

These improvements often come at the expense of time.

# Parallelism
Training and solving can be easily parallelized across multiple CPUs and GPUs.

When training with `ctg_approx/avi.py`, set the number of workers for doing approximate value iteration with `--num_update_procs`
During the update process, the target DNN is spawned on each available GPU and they work in parallel during the udpate step.

The number of GPUs used can be controlled by setting the `CUDA_VISIBLE_DEVICES` environment variable.

i.e. `export CUDA_VISIBLE_DEVICES="0,1,2,3"`

# Memory
When obtaining training data with approximate value iteration and solving using A* search, the batch size of the data 
given to the DNN can be controlled with `--update_nnet_batch_size` for the `avi.py` file and `--nnet_batch_size` for
the `astar.py` file. Reduce this value if your GPUs are running out of memory during approximate value iteration or 
during A* search.

# Compiling C++ for A* Search
`cd cpp/`

`make`

If you are not able to get the C++ version working on your computer, you can change the `--language` switch for
`search_methods/astar.py` from `--language cpp` to `--language python`.
Note that the C++ version is generally faster.

# Citation
To cite this project, please use

```
@misc{hadar2025singlestepupdatesreinforcementlearning,
      title={Beyond Single-Step Updates: Reinforcement Learning of Heuristics with Limited-Horizon Search}, 
      author={Gal Hadar, Forest Agostinelli, Shahaf S. Shperberg},
      year={2025},
      eprint={2511.10264},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2511.10264}, 
}
```

