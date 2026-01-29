import random
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import data_utils, nnet_utils, env_utils
from typing import Dict, List, Tuple, Any

from environments.environment_abstract import Environment
from updaters.updater import Updater
from search_methods.gbfs import gbfs_test
import torch

import torch.nn as nn
import os
import pickle

from argparse import ArgumentParser
import numpy as np
import time

import sys
import shutil


def parse_arguments(parser: ArgumentParser) -> Dict[str, Any]:
    # Environment
    parser.add_argument('--env', type=str, required=True, help="Environment")

    # Debug
    parser.add_argument('--debug', action='store_true', default=False, help="")

    # Gradient Descent
    parser.add_argument('--lr', type=float, default=0.001, help="Initial learning rate")
    parser.add_argument('--lr_d', type=float, default=0.9999993, help="Learning rate decay for every iteration. "
                                                                      "Learning rate is decayed according to: "
                                                                      "lr * (lr_d ^ itr)")

    # Training
    parser.add_argument('--max_itrs', type=int, default=1500000, help="Maxmimum number of iterations")
    parser.add_argument('--batch_size', type=int, default=1000, help="Batch size")
    parser.add_argument('--single_gpu_training', action='store_true',
                        default=False, help="If set, train only on one GPU. Update step will still use "
                                            "all GPUs given by CUDA_VISIBLE_DEVICES")
    parser.add_argument('--seed', type=int, help="Seed for random generation")
    parser.add_argument('--train_method', type=str, default='bellman',
                        help="The train method of the algorithm (bellman)")

    # Update
    parser.add_argument('--loss_thresh', type=float, default=None, help="When the loss falls below this value, "
                                                                        "the target network is updated to the current "
                                                                        "network.")
    parser.add_argument('--states_per_update', type=int, default=1000, help="How many states to train on before "
                                                                            "checking if target network should be "
                                                                            "updated")
    parser.add_argument('--epochs_per_update', type=int, default=1, help="How many epochs to train for. "
                                                                         "Making this greater than 1 could increase "
                                                                         "risk of overfitting, however, one can train "
                                                                         "for more iterations without having to "
                                                                         "generate more data.")
    parser.add_argument('--num_update_procs', type=int, default=1, help="Number of parallel workers used to "
                                                                        "compute updated cost-to-go function")
    parser.add_argument('--update_nnet_batch_size', type=int, default=10000, help="Batch size of each nnet used for "
                                                                                  "each process update. "
                                                                                  "Make smaller if running out of "
                                                                                  "memory.")
    parser.add_argument('--max_update_steps', type=int, default=1, help="Number of steps to take when trying to "
                                                                        "solve training states with "
                                                                        "greedy best-first search (GBFS) or A* search. "
                                                                        "Each state "
                                                                        "encountered when solving is added to the "
                                                                        "training set. Number of steps starts at "
                                                                        "1 and is increased every update until "
                                                                        "the maximum number is reached. "
                                                                        "Value of 1 is the same as doing "
                                                                        "value iteration on only given training "
                                                                        "states. Increasing this number "
                                                                        "can make the cost-to-go function more "
                                                                        "robust by exploring more of the "
                                                                        "state space.")

    parser.add_argument('--update_method', type=str, default="GBFS", help="GBFS or ASTAR. If max_update_steps is 1 "
                                                                          "then either one is the same as doing value "
                                                                          "iteration")

    parser.add_argument('--eps_max', type=float, default=0, help="When addings training states with GBFS, each "
                                                                 "instance will have an eps that is distributed "
                                                                 "randomly between 0 and epx_max.")

    parser.add_argument('--update_batch_size', type=int, default=10000, help="Size of batch to update with")
    parser.add_argument('--astar_batch_size', type=int, default=1,
                        help="Size of batch to pop from open list in ASTAR")
    parser.add_argument('--reach_max_steps', action='store_true', default=False,
                        help="If true we search until max_update_steps")

    # Testing
    parser.add_argument('--num_test', type=int, default=10000, help="Number of test states.")

    # data
    parser.add_argument('--back_max', type=int, required=True, help="Maximum number of backwards steps from goal")
    parser.add_argument('--limited_horizon_update', action='store_true', default=False,
                        help="Use limited horizon Bellman backup instead of regular one step Bellman backup")

    # model
    parser.add_argument('--nnet_name', type=str, required=True, help="Name of neural network")
    parser.add_argument('--update_num', type=int, default=0, help="Update number")
    parser.add_argument('--save_dir', type=str, default="saved_models", help="Director to which to save model")
    parser.add_argument('--save_interval', type=str, default="20,30,40,50,60,70,80,90,100",
                        help="Save model snapshot at each specified step")
    parser.add_argument('--update_target_threshold', type=int, default=10,
                        help="Maximum number of current network updates before updating the target network")

    # parse arguments
    args = parser.parse_args()

    args_dict: Dict[str, Any] = vars(args)

    # make save directory
    prefix: str = "lhu_" if args_dict["limited_horizon_update"] else ""
    suffix: str = "_lt" if args_dict["loss_thresh"] else ""
    folder_name: str = f"{prefix}{args_dict['train_method']}_{args_dict['seed']}_{args_dict['max_update_steps']}{suffix}"
    model_dir: str = f"{args_dict['save_dir']}/{args_dict['nnet_name']}/{args_dict['env']}/{folder_name}/"
    args_dict['targ_dir'] = "%s/%s/" % (model_dir, f"target")
    args_dict['curr_dir'] = "%s/%s/" % (model_dir, f"current")
    args_dict['history_dir'] = "%s/%s/" % (model_dir, f"history")
    args_dict['model_dir'] = model_dir

    if not os.path.exists(args_dict['targ_dir']):
        os.makedirs(args_dict['targ_dir'])

    if not os.path.exists(args_dict['curr_dir']):
        os.makedirs(args_dict['curr_dir'])

    if not os.path.exists(args_dict['history_dir']):
        os.makedirs(args_dict['history_dir'])
    args_dict["output_save_loc"] = "%s/output.txt" % (model_dir)

    # save args
    args_save_loc = "%s/args.pkl" % model_dir
    print("Saving arguments to %s" % args_save_loc)
    with open(args_save_loc, "wb") as f:
        pickle.dump(args, f, protocol=-1)

    print("Batch size: %i" % args_dict['batch_size'])

    return args_dict


def copy_files(src_dir: str, dest_dir: str):
    src_files: List[str] = os.listdir(src_dir)
    for file_name in src_files:
        full_file_name: str = os.path.join(src_dir, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, dest_dir)


def do_update(back_max: int, update_num: int, env: Environment, max_update_steps: int, update_method: str,
              update_batch_size: int, num_states: int, eps_max: float, heur_fn_i_q, heur_fn_o_qs, rng_proc,
              limited_horizon_update: bool, reach_max_steps: bool, train_method: str = "",
              astar_batch_size: int = 100, log_output: str = "output.txt") -> Tuple[List[np.ndarray], np.ndarray]:
    print("Using limited horizon update" if limited_horizon_update else "Using neighbor backup")
    if reach_max_steps:
        update_steps: int = max_update_steps
    else:
        update_steps: int = min(update_num + 1, max_update_steps)

    # Do updates
    output_time_start = time.time()

    num_states: int = int(np.ceil(num_states / update_steps))

    print(f"Updating cost-to-go with value iteration with method: {train_method}")
    if max_update_steps >= 1:
        print("Using %s with %i step(s) to add extra states to training set" % (update_method.upper(), update_steps))

    print(f"Generating states with a random number within range [0, {back_max}] as random back steps from the goal")
    updater: Updater = Updater(env, num_states, back_max, heur_fn_i_q, heur_fn_o_qs, update_steps, update_method,
                               rng_proc, limited_horizon_update, update_batch_size=update_batch_size, eps_max=eps_max,
                               train_method=train_method, astar_batch_size=astar_batch_size, log_output=log_output)

    states_update_nnet: List[np.ndarray]
    output_update: np.ndarray
    states_update_nnet, output_update, is_solved = updater.update()
    # Print stats
    if max_update_steps >= 1:
        print("%s produced %s states, %.2f%% solved (%.2f seconds)" % (update_method.upper(),
                                                                       format(output_update.shape[0], ","),
                                                                       100.0 * np.mean(is_solved),
                                                                       time.time() - output_time_start))

    mean_ctg = output_update[:, 0].mean()
    min_ctg = output_update[:, 0].min()
    max_ctg = output_update[:, 0].max()
    print("Cost-to-go (mean/min/max): %.2f/%.2f/%.2f" % (mean_ctg, min_ctg, max_ctg))

    return states_update_nnet, output_update


def load_nnet(nnet_dir: str, env: Environment, device) -> Tuple[nn.Module, int, int, int]:
    nnet_file: str = "%s/model_state_dict.pt" % nnet_dir
    if os.path.isfile(nnet_file):
        nnet = nnet_utils.load_nnet(nnet_file, env.get_nnet_model(), device)
        itr: int = pickle.load(open("%s/train_itr.pkl" % nnet_dir, "rb"))
        update_num: int = pickle.load(open("%s/update_num.pkl" % nnet_dir, "rb"))
        main_network_updates_counter: int = pickle.load(open("%s/main_network_updates_counter.pkl" % nnet_dir, "rb"))
    else:
        nnet: nn.Module = env.get_nnet_model()
        itr: int = 0
        update_num: int = 0
        main_network_updates_counter: int = 0

    return nnet, itr, update_num, main_network_updates_counter


def set_seed(seed: int):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def save_random_states(save_path: str, rng):
    # Save random module state
    random_state = random.getstate()

    # Save numpy random state
    np_state = np.random.get_state()

    # Save the current state of the rng
    rng_states_generation = rng.bit_generator.state

    # Save PyTorch random states (for both CPU and GPU)
    torch_cpu_rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        torch_cuda_rng_state = torch.cuda.get_rng_state()

    # Save to a file
    with open(save_path, 'wb') as file:
        pickle.dump({
            'random_state': random_state,
            'np_random_state': np_state,
            'rng_states_generation': rng_states_generation,
            'torch_cpu_rng_state': torch_cpu_rng_state,
            'torch_cuda_rng_state': torch_cuda_rng_state if torch.cuda.is_available() else None,
        }, file)


def load_random_states(file_path: str):
    # Load from a file
    with open(file_path, 'rb') as file:
        states = pickle.load(file)

    # Restore random module state
    random.setstate(states['random_state'])

    # Restore numpy random state
    np.random.set_state(states['np_random_state'])

    # Restore the rng saved state for generating states
    rng_proc_state = states['rng_states_generation']

    # Restore PyTorch random states (for both CPU and GPU)
    torch.set_rng_state(states['torch_cpu_rng_state'])
    if torch.cuda.is_available() and states['torch_cuda_rng_state'] is not None:
        torch.cuda.set_rng_state(states['torch_cuda_rng_state'])

    return rng_proc_state


def update_target_network(curr_dir: str, target_dir: str, last_update_num: int) -> int:
    print("Updating target network")
    copy_files(curr_dir, target_dir)
    update_num = last_update_num + 1

    pickle.dump(update_num, open("%s/update_num.pkl" % curr_dir, "wb"), protocol=-1)
    pickle.dump(0, open("%s/main_network_updates_counter.pkl" % target_dir, "wb"), protocol=-1)

    return update_num


def main():
    # arguments
    parser: ArgumentParser = ArgumentParser()
    args_dict: Dict[str, Any] = parse_arguments(parser)

    if not args_dict["debug"]:
        sys.stdout = data_utils.Logger(args_dict["output_save_loc"], "a")
    print(args_dict)

    random_generators_state_path = "%s/random_generators_state.pkl" % args_dict['model_dir']
    if os.path.exists(random_generators_state_path):
        print("Loading random generators states from file: %s" % random_generators_state_path)
        rng_proc_state = load_random_states(random_generators_state_path)
        rng_proc = np.random.default_rng()
        rng_proc.bit_generator.state = rng_proc_state
    else:
        if args_dict['seed']:
            print("Running with seed %i" % args_dict['seed'])
            set_seed(args_dict['seed'])
            rng_proc_seed = args_dict['seed']
        else:
            print("No seed provided, running with default settings")
            rng_proc_seed = np.random.randint(0, 2 ** 31 - 1)

        rng_proc = np.random.default_rng(rng_proc_seed)

    save_intervals: List[int] = list(map(int, args_dict['save_interval'].split(',')))

    # environment
    env: Environment = env_utils.get_environment(args_dict['env'])

    # get device
    on_gpu: bool
    device: torch.device
    device, devices, on_gpu = nnet_utils.get_device()

    print("device: %s, devices: %s, on_gpu: %s" % (device, devices, on_gpu))

    # load nnet
    nnet: nn.Module
    itr: int
    update_num: int
    update_target_threshold: int = args_dict["update_target_threshold"]
    loss_threshold: float = args_dict["loss_thresh"]

    nnet, itr, update_num, main_network_updates_counter = load_nnet(args_dict['curr_dir'], env, device)

    nnet.to(device)

    if on_gpu and (not args_dict['single_gpu_training']):
        nnet = nn.DataParallel(nnet)

    # training
    num_statets = 0
    while itr < args_dict['max_itrs']:
        # update
        targ_file: str = "%s/model_state_dict.pt" % args_dict['targ_dir']
        all_zeros: bool = not os.path.isfile(targ_file)
        if not all_zeros:
            print("Loading target network")
        heur_fn_i_q, heur_fn_o_qs, heur_procs = nnet_utils.start_heur_fn_runners(args_dict['num_update_procs'],
                                                                                 args_dict['targ_dir'],
                                                                                 device, on_gpu, env,
                                                                                 all_zeros=all_zeros,
                                                                                 clip_zero=True,
                                                                                 batch_size=args_dict["update_nnet_batch_size"])

        states_nnet: List[np.ndarray]
        outputs: np.ndarray
        states_nnet, outputs = do_update(args_dict['back_max'], update_num, env,
                                         args_dict['max_update_steps'], args_dict['update_method'],
                                         args_dict['update_batch_size'], args_dict['states_per_update'],
                                         args_dict['eps_max'], heur_fn_i_q, heur_fn_o_qs, rng_proc,
                                         args_dict['limited_horizon_update'], args_dict["reach_max_steps"],
                                         train_method=args_dict['train_method'], astar_batch_size=args_dict['astar_batch_size'],
                                         log_output=args_dict["output_save_loc"])
        print(f"CTG backup: {outputs.mean()}/{outputs.min()}/{outputs.max()}(Mean/Min/Max)")
        num_statets += states_nnet[0].shape[0]

        nnet_utils.stop_heuristic_fn_runners(heur_procs, heur_fn_i_q)

        # train nnet
        num_train_itrs: int = args_dict['epochs_per_update'] * np.ceil(outputs.shape[0] / args_dict['batch_size'])
        print("Training model for update number %i for %i iterations" % (update_num, num_train_itrs))
        last_loss = nnet_utils.train_nnet(nnet, states_nnet, outputs, device, args_dict['batch_size'], num_train_itrs,
                                          itr, args_dict['lr'], args_dict['lr_d'])
        itr += num_train_itrs
        main_network_updates_counter += 1

        # save nnet
        torch.save(nnet.state_dict(), "%s/model_state_dict.pt" % args_dict['curr_dir'])
        pickle.dump(itr, open("%s/train_itr.pkl" % args_dict['curr_dir'], "wb"), protocol=-1)
        pickle.dump(update_num, open("%s/update_num.pkl" % args_dict['curr_dir'], "wb"), protocol=-1)
        pickle.dump(main_network_updates_counter,
                    open("%s/main_network_updates_counter.pkl" % args_dict['curr_dir'], "wb"), protocol=-1)

        save_random_states(random_generators_state_path, rng_proc)  # Save random states for reproducibility

        # test
        start_time = time.time()
        heuristic_fn = nnet_utils.get_heuristic_fn(nnet, device, env, batch_size=args_dict['update_nnet_batch_size'])

        max_solve_steps: int = min(update_num + 1, args_dict['back_max'])
        gbfs_test(args_dict['num_test'], args_dict['back_max'], env, heuristic_fn, max_solve_steps=max_solve_steps,
                  train_method="bellman")

        print("Test time: %.2f" % (time.time() - start_time))

        # clear cuda memory
        torch.cuda.empty_cache()

        print("Last loss was %f" % last_loss)
        if loss_threshold:
            if last_loss <= loss_threshold:
                update_num = update_target_network(args_dict['curr_dir'], args_dict['targ_dir'], update_num)
                main_network_updates_counter = 0
        elif main_network_updates_counter >= update_target_threshold:
            update_num = update_target_network(args_dict['curr_dir'], args_dict['targ_dir'], update_num)
            main_network_updates_counter = 0

        snapshot_iter = int(itr / 10000)
        if snapshot_iter in save_intervals:
            # Save model snapshot
            print(f"Saving model snapshot for iteration {snapshot_iter}")
            torch.save(
                nnet.state_dict(),
                f"{os.path.join(args_dict['history_dir'], f'model_state_dict_{snapshot_iter}.pt')}"
            )

    print(f"Done, trained with {num_statets} statets")


if __name__ == "__main__":
    main()
