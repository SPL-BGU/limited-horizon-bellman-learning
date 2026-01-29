from typing import List, Tuple, Callable
import numpy as np
from utils import nnet_utils, misc_utils
from environments.environment_abstract import Environment, State
from search_methods.gbfs import GBFS
from search_methods.astar import AStar, Node
from torch.multiprocessing import Queue, get_context
import time
import random
from utils.search_utils import run_limited_horizon_update


def gbfs_update(states: List[State], env: Environment, num_steps: int, heuristic_fn: Callable,
                eps_max: float, train_method: str = ""):
    eps: List[float] = list(np.random.rand(len(states)) * eps_max)

    gbfs = GBFS(states, env, eps=eps, train_method=train_method)
    for _ in range(num_steps):
        gbfs.step(heuristic_fn)

    trajs: List[List[Tuple[State, float]]] = gbfs.get_trajs()

    trajs_flat: List[Tuple[State, float]]
    trajs_flat, _ = misc_utils.flatten(trajs)

    is_solved: np.ndarray = np.array(gbfs.get_is_solved())

    states_update: List = []
    cost_to_go_update_l: List[float] = []
    for traj in trajs_flat:
        states_update.append(traj[0])
        cost_to_go_update_l.append(traj[1])

    cost_to_go_update = np.array(cost_to_go_update_l)

    return states_update, cost_to_go_update, is_solved


def astar_update(states: List[State], env: Environment, num_steps: int, heuristic_fn: Callable,
                 limited_horizon_update: bool, back_max: int, batch_size: int = 1,
                 train_method: str = ""):
    weights: List[float] = [1.0] * len(states)
    astar = AStar(states, env, heuristic_fn, weights)
    for i in range(num_steps):
        astar.step(heuristic_fn, batch_size, fill_new_instances=False, back_max=back_max)

    nodes_popped: List[List[Node]] = astar.get_popped_nodes()
    nodes_popped_flat: List[Node]
    nodes_popped_flat, _ = misc_utils.flatten(nodes_popped)

    cost_to_go_update: np.array = np.array([])
    if limited_horizon_update:
        for instance, tree in zip(astar.instances, nodes_popped):
            leaves = [node[2] for node in instance.open_set]
            leaves.extend(instance.goal_nodes)
            run_limited_horizon_update(leaves)
            cost_to_go_update = np.append(cost_to_go_update,
                                          np.array([node.lhu_val for node in tree]))
            del instance.closed_dict
            del instance.open_set
    else:
        nodes_backup: List[float] = []
        for node in nodes_popped_flat:
            node.compute_bellman(train_method)
            nodes_backup.append(node.bellman)
        cost_to_go_update = np.append(cost_to_go_update, np.array(nodes_backup))

    states_update: List[State] = [node.state for node in nodes_popped_flat]
    is_solved: np.array = np.array([node.is_solved for node in nodes_popped_flat])

    return states_update, cost_to_go_update, is_solved


def update_runner(num_states: int, back_max: int, update_batch_size: int, heur_fn_i_q, heur_fn_o_q,
                  proc_id: int, env: Environment, result_queue: Queue, num_steps: int, update_method: str,
                  eps_max: float, seed: int, limited_horizon_update: bool, train_method: str = "", astar_batch_size: int = 1):
    heuristic_fn = nnet_utils.heuristic_fn_queue(heur_fn_i_q, heur_fn_o_q, proc_id, env)

    np.random.seed(seed)
    random.seed(seed)

    start_idx: int = 0
    while start_idx < num_states:
        end_idx: int = min(start_idx + update_batch_size, num_states)
        states_itr, _ = env.generate_states(end_idx - start_idx, (0, back_max))

        if update_method.upper() == "GBFS":
            states_update, cost_to_go_update, is_solved = gbfs_update(
                states_itr, env, num_steps, heuristic_fn, eps_max, train_method=train_method
            )
        elif update_method.upper() == "ASTAR":
            states_update, cost_to_go_update, is_solved = astar_update(
                states_itr, env, num_steps, heuristic_fn,
                limited_horizon_update, back_max, astar_batch_size, train_method=train_method
            )
        else:
            raise ValueError("Unknown update method %s" % update_method)

        states_update_nnet: List[np.ndarray] = env.state_to_nnet_input(states_update)

        result_queue.put((states_update_nnet, cost_to_go_update, is_solved, proc_id))

        start_idx = end_idx

    result_queue.put(None)


class Updater:
    def __init__(self, env: Environment, num_states: int, back_max: int, heur_fn_i_q, heur_fn_o_qs,
                 num_steps: int, update_method: str, rng_proc, limited_horizon_update: bool, update_batch_size: int = 1000,
                 eps_max: float = 0.0, train_method: str = "", astar_batch_size: int = 1, log_output: str = "output.txt"):
        super().__init__()
        ctx = get_context("spawn")
        self.num_steps = num_steps
        num_procs = len(heur_fn_o_qs)

        # initialize queues
        self.result_queue: ctx.Queue = ctx.Queue()

        if num_steps > 1:
            num_states = int(np.ceil(num_states / astar_batch_size))
            update_batch_size = int(np.ceil(update_batch_size / astar_batch_size))
        else:
            astar_batch_size = 1

        # num states per process
        num_states_per_proc: List[int] = misc_utils.split_evenly(num_states, num_procs)

        self.num_batches: int = int(np.ceil(np.array(num_states_per_proc) / update_batch_size).sum())
        print(f"Number of states per process: {num_states_per_proc}")
        print(f"Number of batches: {self.num_batches}")
        print(f"Update batch size: {update_batch_size}")

        # initialize processes
        self.procs: List[ctx.Process] = []

        for proc_id in range(len(heur_fn_o_qs)):
            num_states_proc: int = num_states_per_proc[proc_id]
            if num_states_proc == 0:
                continue

            # Generate a unique seed for each process
            seed = int(rng_proc.integers(0, 2 ** 31 - 1))

            proc = ctx.Process(target=update_runner, args=(num_states_proc, back_max, update_batch_size,
                                                           heur_fn_i_q, heur_fn_o_qs[proc_id], proc_id, env,
                                                           self.result_queue, num_steps, update_method, eps_max, seed,
                                                           limited_horizon_update, train_method, astar_batch_size))
            proc.daemon = True
            proc.start()
            self.procs.append(proc)

    def update(self):
        states_update_nnet: List[np.ndarray]
        cost_to_go_update: np.ndarray
        is_solved: np.ndarray
        states_update_nnet, cost_to_go_update, is_solved = self._update()

        output_update = np.expand_dims(cost_to_go_update, 1)

        return states_update_nnet, output_update, is_solved

    def _update(self) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        # process results
        states_update_nnet_l: List[List[np.ndarray]] = []
        cost_to_go_update_l: List = []
        is_solved_l: List = []

        none_count: int = 0
        result_count: int = 0
        display_counts: List[int] = list(np.linspace(1, self.num_batches, 10, dtype=np.int64))

        start_time = time.time()

        while none_count < len(self.procs):
            result = self.result_queue.get()
            if result is None:
                none_count += 1
                continue
            result_count += 1

            states_nnet_q: List[np.ndarray]
            states_nnet_q, cost_to_go_q, is_solved_q, proc_id = result
            states_update_nnet_l.append([states_nnet_q, proc_id])

            cost_to_go_update_l.append([cost_to_go_q, proc_id])
            is_solved_l.append([is_solved_q, proc_id])

            if result_count in display_counts:
                print("%.2f%% (Total time: %.2f)" % (100 * result_count / self.num_batches, time.time() - start_time))

        # Sort the results based on the process id
        states_update_nnet_l.sort(key=lambda s: s[1])
        states_update_nnet_l = [i[0] for i in states_update_nnet_l]
        cost_to_go_update_l.sort(key=lambda s: s[1])
        cost_to_go_update_l = [i[0] for i in cost_to_go_update_l]
        is_solved_l.sort(key=lambda s: s[1])
        is_solved_l = [i[0] for i in is_solved_l]

        # Get the sorted indices based on the first state of each ndarray returned by the processes
        sorted_indices = sorted(range(len(states_update_nnet_l)), key=lambda i: states_update_nnet_l[i][0][0, 0])

        # Sort the lists according to the first state of each process results
        states_update_nnet_l_sorted = [states_update_nnet_l[i] for i in sorted_indices]
        cost_to_go_update_l_sorted = [cost_to_go_update_l[i] for i in sorted_indices]
        is_solved_l_sorted = [is_solved_l[i] for i in sorted_indices]

        num_states_nnet_np: int = len(states_update_nnet_l[0])
        states_update_nnet: List[np.ndarray] = []
        for np_idx in range(num_states_nnet_np):
            states_nnet_idx: np.ndarray = np.concatenate([x[np_idx] for x in states_update_nnet_l_sorted], axis=0)
            states_update_nnet.append(states_nnet_idx)

        cost_to_go_update: np.ndarray = np.concatenate(cost_to_go_update_l_sorted, axis=0)
        is_solved: np.ndarray = np.concatenate(is_solved_l_sorted, axis=0)

        for proc in self.procs:
            proc.join()

        return states_update_nnet, cost_to_go_update, is_solved
