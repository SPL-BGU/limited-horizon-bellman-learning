import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List, Tuple, Dict, Callable, Optional, Any
from environments.environment_abstract import Environment, State
import numpy as np
from heapq import heappush, heappop
from subprocess import Popen, PIPE

from argparse import ArgumentParser
import torch
from utils import env_utils, nnet_utils, search_utils, misc_utils, data_utils
import pickle
import time
import sys
import os
import socket
from torch.multiprocessing import Process
import logging


class Node:
    __slots__ = ['state', 'path_cost', 'heuristic', 'cost', 'is_solved', 'parent_move', 'parent',
                 'transition_costs', 'children', 'parents', 'parents_tcs', 'bellman', 'child_index', 'lhu_val']

    def __init__(self, state: Optional[State], path_cost: float, is_solved: bool,
                 parent_move: Optional[int], parent: Optional[Any], heuristic: float = None):
        self.state: State = state
        self.path_cost: float = path_cost
        self.heuristic: Optional[float] = heuristic
        self.cost: Optional[float] = None
        self.is_solved: bool = is_solved
        self.parent_move: Optional[int] = parent_move
        self.parent: Optional[Node] = parent

        self.transition_costs: List[float] = []
        self.children: List[Node] = []
        self.parents: List[Node] = []
        self.parents_tcs: List[float] = []

        # Fields for backup
        self.bellman: float = np.inf
        self.child_index: Optional[int] = None  # Store position in parent's children list
        self.lhu_val: float = np.inf

    def compute_bellman(self, method: str):
        if self.is_solved:
            self.bellman = 0.0
        elif len(self.children) == 0:
            self.bellman = self.heuristic
        else:
            if method == "bellman":
                for node_c, tc in zip(self.children, self.transition_costs):
                    self.bellman = min(self.bellman, tc + node_c.heuristic)
            else:
                print(f"Unknown update method {method}")
                raise f"Unknown update method {method}"

    def __eq__(self, other):
        return self.state == other.state

    def __hash__(self):
        # Assuming self.state is hashable
        return hash(self.state)


OpenSetElem = Tuple[float, int, Node]


class Instance:

    def __init__(self, root_node: Node):
        self.open_set: List[OpenSetElem] = []
        self.heappush_count: int = 0
        self.closed_dict: Dict[State, Node] = dict()
        self.popped_nodes: List[Node] = []
        self.goal_nodes: List[Node] = []
        self.num_nodes_generated: int = 0

        self.root_node: Node = root_node

        self.push_to_open([self.root_node])

        self.closed_dict[self.root_node.state] = self.root_node

    def push_to_open(self, nodes: List[Node]):
        for node in nodes:
            heappush(self.open_set, (node.cost, self.heappush_count, node))
            self.heappush_count += 1

    def pop_from_open(self, num_nodes: int) -> List[Node]:
        num_to_pop: int = min(num_nodes, len(self.open_set))

        popped_nodes = [heappop(self.open_set)[2] for _ in range(num_to_pop)]
        self.goal_nodes.extend([node for node in popped_nodes if node.is_solved])
        self.popped_nodes.extend(popped_nodes)

        popped_nodes = [node for node in popped_nodes if not node.is_solved]
        return popped_nodes

    def remove_in_closed(self, nodes: List[Node]) -> List[Node]:
        nodes_not_in_closed: List[Node] = []

        for node in nodes:
            prev_node: Optional[Node] = self.closed_dict.get(node.state)
            if prev_node is None:
                nodes_not_in_closed.append(node)
                self.closed_dict[node.state] = node
            elif prev_node.path_cost > node.path_cost:
                nodes_not_in_closed.append(node)
                self.closed_dict[node.state] = node
                replace_node_in_parent(prev_node, node)
            else:
                # Parent should point to the node with the better path cost
                replace_node_in_parent(node, prev_node)

        return nodes_not_in_closed


def replace_node_in_parent(node_to_remove: Node, node_to_add: Node):
    parent = node_to_remove.parent
    if parent and node_to_remove.child_index is not None:
        parent.children[node_to_remove.child_index] = node_to_add
        node_to_add.parents.append(parent)
        node_to_add.parents_tcs.append(1.0)
        if not node_to_add.parent:
            node_to_add.child_index = node_to_remove.child_index  # Maintain index for future replacements
        elif parent.path_cost < node_to_add.parent.path_cost:
            node_to_add.child_index = node_to_remove.child_index  # Maintain index for future replacements
            node_to_add.parent = parent


def pop_from_open(instances: List[Instance], batch_size: int) -> List[List[Node]]:
    popped_nodes_all: List[List[Node]] = [instance.pop_from_open(batch_size) for instance in instances]

    return popped_nodes_all


def expand_nodes(instances: List[Instance], popped_nodes_all: List[List[Node]], env: Environment):
    # Get children of all nodes at once (for speed)
    popped_nodes_flat: List[Node]
    split_idxs: List[int]
    popped_nodes_flat, split_idxs = misc_utils.flatten(popped_nodes_all)

    if len(popped_nodes_flat) == 0:
        return [[]]

    states: List[State] = [x.state for x in popped_nodes_flat]

    states_c_by_node: List[List[State]]
    tcs_np: List[np.ndarray]

    states_c_by_node, tcs_np = env.expand(states)

    tcs_by_node: List[List[float]] = [list(x) for x in tcs_np]

    # Get is_solved on all states at once (for speed)
    states_c: List[State]

    states_c, split_idxs_c = misc_utils.flatten(states_c_by_node)
    is_solved_c: List[bool] = list(env.is_solved(states_c))
    is_solved_c_by_node: List[List[bool]] = misc_utils.unflatten(is_solved_c, split_idxs_c)

    # Update path costs for all states at once (for speed)
    parent_path_costs = np.expand_dims(np.array([node.path_cost for node in popped_nodes_flat]), 1)
    path_costs_c: List[float] = (parent_path_costs + np.array(tcs_by_node)).flatten().tolist()

    path_costs_c_by_node: List[List[float]] = misc_utils.unflatten(path_costs_c, split_idxs_c)

    # Reshape lists
    tcs_by_inst_node: List[List[List[float]]] = misc_utils.unflatten(tcs_by_node, split_idxs)
    patch_costs_c_by_inst_node: List[List[List[float]]] = misc_utils.unflatten(path_costs_c_by_node,
                                                                               split_idxs)
    states_c_by_inst_node: List[List[List[State]]] = misc_utils.unflatten(states_c_by_node, split_idxs)
    is_solved_c_by_inst_node: List[List[List[bool]]] = misc_utils.unflatten(is_solved_c_by_node, split_idxs)

    # Get child nodes
    instance: Instance
    nodes_c_by_inst: List[List[Node]] = []
    for inst_idx, instance in enumerate(instances):
        nodes_c_by_inst.append([])
        parent_nodes: List[Node] = popped_nodes_all[inst_idx]
        tcs_by_node: List[List[float]] = tcs_by_inst_node[inst_idx]
        path_costs_c_by_node: List[List[float]] = patch_costs_c_by_inst_node[inst_idx]
        states_c_by_node: List[List[State]] = states_c_by_inst_node[inst_idx]

        is_solved_c_by_node: List[List[bool]] = is_solved_c_by_inst_node[inst_idx]

        parent_node: Node
        tcs_node: List[float]
        states_c: List[State]
        str_reps_c: List[str]
        for parent_node, tcs_node, path_costs_c, states_c, is_solved_c in zip(parent_nodes, tcs_by_node,
                                                                              path_costs_c_by_node, states_c_by_node,
                                                                              is_solved_c_by_node):
            state: State
            for move_idx, state in enumerate(states_c):
                path_cost: float = path_costs_c[move_idx]
                is_solved: bool = is_solved_c[move_idx]
                node_c: Node = Node(state, path_cost, is_solved, move_idx, parent_node)
                node_c.child_index = move_idx
                nodes_c_by_inst[inst_idx].append(node_c)

                parent_node.children.append(node_c)
                node_c.parents.append(parent_node)
                node_c.parents_tcs.append(1.0)

            parent_node.transition_costs.extend(tcs_node)

        instance.num_nodes_generated += len(nodes_c_by_inst[inst_idx])

    return nodes_c_by_inst


def remove_in_closed(instances: List[Instance], nodes_c_all: List[List[Node]]) -> List[List[Node]]:
    for inst_idx, instance in enumerate(instances):
        nodes_c_all[inst_idx] = instance.remove_in_closed(nodes_c_all[inst_idx])

    return nodes_c_all


def add_heuristic_and_cost(nodes: List[Node], heuristic_fn: Callable,
                           weights: List[float], bfs: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    # flatten nodes
    nodes: List[Node]

    if len(nodes) == 0:
        return np.zeros(0), np.zeros(0)

    # get heuristic
    states: List[State] = [node.state for node in nodes]

    # compute node cost
    heuristics = heuristic_fn(states)
    path_costs: np.ndarray = np.array([node.path_cost for node in nodes])
    is_solved: np.ndarray = np.array([node.is_solved for node in nodes])

    if bfs:
        costs: np.ndarray = np.array(weights) * path_costs
    else:
        costs: np.ndarray = np.array(weights) * path_costs + heuristics * np.logical_not(is_solved)

    # add cost to node
    for node, heuristic, cost in zip(nodes, heuristics, costs):
        node.heuristic = heuristic
        node.cost = cost

    return path_costs, heuristics


def add_to_open(instances: List[Instance], nodes: List[List[Node]]) -> None:
    nodes_inst: List[Node]
    instance: Instance
    for instance, nodes_inst in zip(instances, nodes):
        instance.push_to_open(nodes_inst)


def get_path(node: Node) -> Tuple[List[State], List[int], float]:
    path: List[State] = []
    moves: List[int] = []

    parent_node: Node = node
    while parent_node.parent is not None:
        path.append(parent_node.state)

        moves.append(parent_node.parent_move)
        parent_node = parent_node.parent

    path.append(parent_node.state)

    path = path[::-1]
    moves = moves[::-1]

    return path, moves, node.path_cost


class AStar:

    def __init__(self, states: List[State], env: Environment, heuristic_fn: Callable, weights: List[float]):
        self.env: Environment = env
        self.weights: List[float] = weights
        self.step_num: int = 0

        self.timings: Dict[str, float] = {"pop": 0.0, "expand": 0.0, "check": 0.0, "heur": 0.0,
                                          "add": 0.0, "itr": 0.0}

        # compute starting costs
        root_nodes: List[Node] = []
        is_solved_states: np.ndarray = self.env.is_solved(states)
        for state, is_solved in zip(states, is_solved_states):
            root_node: Node = Node(state, 0.0, is_solved, None, None)
            root_nodes.append(root_node)

        add_heuristic_and_cost(root_nodes, heuristic_fn, self.weights)

        # initialize instances
        self.instances: List[Instance] = []
        for root_node in root_nodes:
            self.instances.append(Instance(root_node))

        self.finished_instances: List[Instance] = []
        self.num_of_finished_instances: int = 0

    def step(self, heuristic_fn: Callable, batch_size: int,
             include_solved: bool = False, verbose: bool = False, fill_new_instances: bool = False, back_max: int = 30):
        start_time_itr = time.time()
        instances: List[Instance]
        if include_solved:
            instances = self.instances
        else:
            if not fill_new_instances:
                instances = [instance for instance in self.instances if len(instance.goal_nodes) == 0]
            else:
                instances, weights = self.fill_instances(heuristic_fn, back_max)

        # Pop from open
        start_time = time.time()
        popped_nodes_all: List[List[Node]] = pop_from_open(instances, batch_size)
        pop_time = time.time() - start_time

        # Expand nodes
        start_time = time.time()
        nodes_c_all: List[List[Node]] = expand_nodes(instances, popped_nodes_all, self.env)
        expand_time = time.time() - start_time

        # Get heuristic of children, do heur before check so we can do backup
        start_time = time.time()
        nodes_c_all_flat, _ = misc_utils.flatten(nodes_c_all)
        weights = self.weights if not fill_new_instances else weights
        weights_flat, _ = misc_utils.flatten([[weight] * len(nodes_c) for weight, nodes_c in zip(weights, nodes_c_all)])
        path_costs, heuristics = add_heuristic_and_cost(nodes_c_all_flat, heuristic_fn, weights_flat)
        heur_time = time.time() - start_time

        # Check if children are in closed
        start_time = time.time()
        nodes_c_all = remove_in_closed(instances, nodes_c_all)
        check_time = time.time() - start_time

        # Add to open
        start_time = time.time()
        add_to_open(instances, nodes_c_all)
        add_time = time.time() - start_time

        itr_time = time.time() - start_time_itr

        # Print to screen
        if verbose:
            if heuristics.shape[0] > 0:
                min_heur = np.min(heuristics)
                min_heur_pc = path_costs[np.argmin(heuristics)]
                max_heur = np.max(heuristics)
                max_heur_pc = path_costs[np.argmax(heuristics)]

                print("Itr: %i, Added to OPEN - Min/Max Heur(PathCost): "
                      "%.2f(%.2f)/%.2f(%.2f) " % (self.step_num, min_heur, min_heur_pc, max_heur, max_heur_pc))

            print("Times - pop: %.2f, expand: %.2f, check: %.2f, heur: %.2f, "
                  "add: %.2f, itr: %.2f" % (pop_time, expand_time, check_time, heur_time, add_time, itr_time))

            print("")

        # Update timings
        self.timings['pop'] += pop_time
        self.timings['expand'] += expand_time
        self.timings['check'] += check_time
        self.timings['heur'] += heur_time
        self.timings['add'] += add_time
        self.timings['itr'] += itr_time

        logging.info("End of step")

        self.step_num += 1

    def fill_instances(self, heuristic_fn: Callable, back_max: int) -> Tuple[
        List[Instance], List[float]]:
        num_of_finished_instances = len(self.finished_instances)
        finished_instances_set = set(self.finished_instances)  # Convert to set for fast lookups

        # Filter unfinished instances
        unfinished_instances = [
            (instance, weight) for instance, weight in zip(self.instances, self.weights)
            if instance not in finished_instances_set and len(instance.goal_nodes) == 0
        ]

        instances, weights = (list(t) for t in zip(*unfinished_instances)) if unfinished_instances else ([], [])

        # Update finished instances in bulk
        self.finished_instances.extend([
            instance for instance in self.instances
            if instance not in finished_instances_set and len(instance.goal_nodes) > 0
        ])

        num_solved = len(self.instances) - num_of_finished_instances - len(instances)
        if num_solved > 0:
            new_states, _ = self.env.generate_states(num_solved, (0, back_max))

            # compute starting costs
            new_states_weights: List[float] = [1.0] * len(new_states)
            is_solved_states: np.ndarray = self.env.is_solved(new_states)

            root_nodes: List[Node] = [
                Node(state, 0.0, is_solved, None, None)
                for state, is_solved in zip(new_states, is_solved_states)
            ]

            add_heuristic_and_cost(root_nodes, heuristic_fn, new_states_weights)

            # initialize instances
            new_instances: List[Instance] = [Instance(root_node) for root_node in root_nodes]

            # Extend the list of unfinished instances with the new instances
            instances.extend(new_instances)
            weights.extend(new_states_weights)
            self.instances.extend(new_instances)
            self.weights.extend(new_states_weights)

        return instances, weights

    def has_found_goal(self) -> List[bool]:
        goal_found: List[bool] = [len(self.get_goal_nodes(idx)) > 0 for idx in range(len(self.instances))]

        return goal_found

    def get_goal_nodes(self, inst_idx) -> List[Node]:
        return self.instances[inst_idx].goal_nodes

    def get_goal_node_smallest_path_cost(self, inst_idx) -> Node:
        goal_nodes: List[Node] = self.get_goal_nodes(inst_idx)
        path_costs: List[float] = [node.path_cost for node in goal_nodes]

        goal_node: Node = goal_nodes[int(np.argmin(path_costs))]

        return goal_node

    def get_num_nodes_generated(self, inst_idx: int) -> int:
        return self.instances[inst_idx].num_nodes_generated

    def get_popped_nodes(self) -> List[List[Node]]:
        popped_nodes_all: List[List[Node]] = [instance.popped_nodes for instance in self.instances]
        return popped_nodes_all


def main():
    # parse arguments
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--states', type=str, required=True, help="File containing states to solve")
    parser.add_argument('--model_dir', type=str, required=True, help="Directory of nnet model")
    parser.add_argument('--env', type=str, required=True, help="Environment: cube3, 15-puzzle, 24-puzzle")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for BWAS")
    parser.add_argument('--weight', type=float, default=1.0, help="Weight of path cost")
    parser.add_argument('--language', type=str, default="python", help="python or cpp")

    parser.add_argument('--results_dir', type=str, required=True, help="Directory to save results")
    parser.add_argument('--start_idx', type=int, default=0, help="The first state index in data file")
    parser.add_argument('--end_idx', type=int, default=1000, help="The last state index in data file")
    parser.add_argument('--nnet_batch_size', type=int, default=None, help="Set to control how many states per GPU are "
                                                                          "evaluated by the neural network at a time. "
                                                                          "Does not affect final results, "
                                                                          "but will help if nnet is running out of "
                                                                          "memory.")

    parser.add_argument('--generate_plots', action='store_true', default=False,
                        help="Set to run A* on all checkpoints in model_dir and generate summary plots")
    parser.add_argument('--verbose', action='store_true', default=False, help="Set for verbose")
    parser.add_argument('--debug', action='store_true', default=False, help="Set when debugging")
    parser.add_argument('--checkpoint_file', type=str, default="model_state_dict.pt",
                        help="The .pt file name to load the model from")
    parser.add_argument('--solution_time_cap', type=int, default=10, help="Time cap for each solution (minutes)")

    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(), "logs"), exist_ok=True)

    solution_time_cap: int = args.solution_time_cap

    checkpoint_file_num = args.checkpoint_file.split(".")[0].split("_")[
        -1]  # extract checkpoint number, from file name format "model_state_dict_<XX>.pt"
    start_idx, end_idx = args.start_idx, args.end_idx

    results_file: str = f"%s/results-cp{checkpoint_file_num}-s{start_idx}-e{end_idx}-tc{solution_time_cap}.pkl" % args.results_dir
    output_file: str = f"%s/output-cp{checkpoint_file_num}-s{start_idx}-e{end_idx}-tc{solution_time_cap}.txt" % args.results_dir

    results: Dict[str, Any] = dict()
    rerun = os.path.exists(results_file)

    if rerun:
        output_file = output_file.replace(".txt", "-rerun.txt")
        results = pickle.load(open(results_file, "rb"))
        start_idx = len(results["states"])
    else:
        results["states"] = []
        results["solutions"] = []
        results["paths"] = []
        results["times"] = []
        results["num_nodes_generated"] = []
        results["solved_states"] = []
        results["popped_nodes_info"] = []

    if not args.debug:
        sys.stdout = data_utils.Logger(output_file, "w")

    if rerun:
        print(f"Results file {results_file} already exists, recovering from previous run")
        print(f"Loaded results from {results_file}, starting from state {start_idx}")

    results["solution_time_cap"] = solution_time_cap

    # get data
    input_data = pickle.load(open(args.states, "rb"))
    states: List[State] = input_data['states'][start_idx:end_idx]

    # environment
    env: Environment = env_utils.get_environment(args.env)

    if args.language == "python":
        bwas_python(args, env, states, solution_time_cap, results, results_file, start_idx)
    elif args.language == "cpp":
        bwas_cpp(args, env, states, solution_time_cap, results, results_file, start_idx)
    else:
        raise ValueError("Unknown language %s" % args.language)

    pickle.dump(results, open(results_file, "wb"), protocol=-1)

    print("Done")


def bwas_python(args, env: Environment, states: List[State], solution_time_cap: int, results: Dict[str, Any],
                results_file: str, start_idx: int):
    # get device
    on_gpu: bool
    device: torch.device
    device, devices, on_gpu = nnet_utils.get_device()

    print("device: %s, devices: %s, on_gpu: %s" % (device, devices, on_gpu))

    heuristic_fn = nnet_utils.load_heuristic_fn(args.model_dir, device, on_gpu, env.get_nnet_model(),
                                                env, clip_zero=True, batch_size=args.nnet_batch_size,
                                                model_checkpoint=args.checkpoint_file)

    print("Loaded heuristic function from %s" % args.model_dir)

    time_cap = 60 * solution_time_cap  # minutes to seconds

    for state_idx, state in enumerate(states):
        state_idx += start_idx
        start_time = time.time()

        num_itrs: int = 0
        astar = AStar([state], env, heuristic_fn, [args.weight])
        reached_time_cap = False

        while not min(astar.has_found_goal()):
            if time.time() - start_time > time_cap:
                reached_time_cap = True
                break
            astar.step(heuristic_fn, args.batch_size, verbose=args.verbose)
            num_itrs += 1

        if reached_time_cap:
            results["states"].append(state)
            results["solutions"].append(None)
            results["paths"].append(None)
            results["times"].append(None)
            results["num_nodes_generated"].append(None)
            results["solved_states"].append(None)
            results["popped_nodes_info"].append(None)

            print(f"Time cap of {time_cap / 60} minutes reached for state {state_idx}, continuing to the next state")
            continue

        path: List[State]
        soln: List[int]
        path_cost: float
        num_nodes_gen_idx: int
        goal_node: Node = astar.get_goal_node_smallest_path_cost(0)
        path, soln, path_cost = get_path(goal_node)

        num_nodes_gen_idx: int = astar.get_num_nodes_generated(0)

        solve_time = time.time() - start_time

        # record solution information
        results["states"].append(state)
        results["solutions"].append(soln)
        results["paths"].append(path)
        results["times"].append(solve_time)
        results["num_nodes_generated"].append(num_nodes_gen_idx)
        results["solved_states"].append(state_idx)

        nodes_popped: List[List[Node]] = astar.get_popped_nodes()
        nodes_popped_flat: List[Node]
        nodes_popped_flat, _ = misc_utils.flatten(nodes_popped)

        popped_nodes_info = {
            "state": [],
            "heuristic": [],
            "cost": [],
        }
        for node in nodes_popped_flat:
            popped_nodes_info["state"].append(node.state)
            popped_nodes_info["heuristic"].append(node.heuristic)
            popped_nodes_info["cost"].append(node.cost)

        results["popped_nodes_info"].append(popped_nodes_info)

        pickle.dump(results, open(results_file, "wb"), protocol=-1)  # save for reproducibility

        # check soln
        assert search_utils.is_valid_soln(state, soln, env)

        # print to screen
        timing_str = ", ".join(["%s: %.2f" % (key, val) for key, val in astar.timings.items()])
        print("Times - %s, num_itrs: %i" % (timing_str, num_itrs))

        print("State: %i, SolnCost: %.2f, # Moves: %i, "
              "# Nodes Gen: %s, Time: %.2f" % (state_idx, path_cost, len(soln),
                                               format(num_nodes_gen_idx, ","),
                                               solve_time))

        print("Finished running python bwas on all states")


def bwas_cpp(args, env: Environment, states: List[State], solution_time_cap: int, results: Dict[str, Any],
             results_file: str, start_idx: int):
    assert (args.env.upper() in ['CUBE3', 'CUBE4', 'PUZZLE15', 'PUZZLE24', 'PUZZLE35', 'PUZZLE48', 'LIGHTSOUT7'])

    # Make c++ socket
    socket_name: str = results_file.split(".")[0]

    try:
        os.unlink(socket_name)
    except OSError:
        if os.path.exists(socket_name):
            raise

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(socket_name)

    # Get state dimension
    if args.env.upper() == 'CUBE3':
        state_dim: int = 54
    elif args.env.upper() == 'PUZZLE15':
        state_dim: int = 16
    elif args.env.upper() == 'PUZZLE24':
        state_dim: int = 25
    elif args.env.upper() == 'PUZZLE35':
        state_dim: int = 36
    elif args.env.upper() == 'PUZZLE48':
        state_dim: int = 49
    elif args.env.upper() == 'LIGHTSOUT7':
        state_dim: int = 49
    else:
        raise ValueError("Unknown c++ environment: %s" % args.env)

    # start heuristic proc
    num_parallel: int = len(os.environ['CUDA_VISIBLE_DEVICES'].split(","))
    device, devices, on_gpu = nnet_utils.get_device()
    heur_fn_i_q, heur_fn_o_qs, heur_procs = nnet_utils.start_heur_fn_runners(num_parallel, [args.model_dir], device,
                                                                             on_gpu, env, all_zeros=False,
                                                                             clip_zero=True,
                                                                             batch_size=args.nnet_batch_size,
                                                                             model_checkpoint=args.checkpoint_file)
    nnet_utils.heuristic_fn_par(states, env, heur_fn_i_q, heur_fn_o_qs)  # initialize

    heur_proc = Process(target=cpp_listener, args=(sock, args, env, state_dim, heur_fn_i_q, heur_fn_o_qs))
    heur_proc.daemon = True
    heur_proc.start()

    time.sleep(2)  # give socket time to intialize

    time_cap = 60 * solution_time_cap

    for state_idx, state in enumerate(states):
        state_idx += start_idx

        # Get string rep of state
        if args.env.upper() == "CUBE3":
            state_str: str = " ".join([str(x) for x in state.colors])
        elif args.env.upper() in ["PUZZLE15", "PUZZLE24", "PUZZLE35", "PUZZLE48"]:
            state_str: str = " ".join([str(x) for x in state.tiles])
        elif args.env.upper() in ["LIGHTSOUT7"]:
            state_str: str = " ".join([str(x) for x in state.tiles])
        else:
            raise ValueError("Unknown c++ environment: %s" % args.env)

        popen = Popen(['./cpp/parallel_weighted_astar', state_str, str(args.weight), str(args.batch_size),
                       socket_name, args.env, str(time_cap), "0"], stdout=PIPE, stderr=PIPE, bufsize=1,
                      universal_newlines=True)
        lines = []
        for stdout_line in iter(popen.stdout.readline, ""):
            stdout_line = stdout_line.strip('\n')
            lines.append(stdout_line)
            if args.verbose:
                sys.stdout.write("%s\n" % stdout_line)
                sys.stdout.flush()

        reached_time_cap = int(lines[-1])
        if reached_time_cap:
            results["states"].append(state)
            results["solutions"].append(None)
            results["paths"].append(None)
            results["times"].append(None)
            results["num_nodes_generated"].append(None)
            results["solved_states"].append(None)
            results["popped_nodes_info"].append(None)
            print(f"Time cap of {time_cap / 60} minutes reached for state {state_idx}, continuing to the next state")
            continue

        popped_nodes_info = get_h_g_from_cpp(lines)

        moves = [int(x) for x in lines[-7].split(" ")[:-1]]
        soln = [x for x in moves][::-1]
        num_nodes_gen_idx = int(lines[-5])
        solve_time = float(lines[-3])

        # record solution information
        path: List[State] = [state]
        next_state: State = state
        transition_costs: List[float] = []

        for move in soln:
            next_states, tcs = env.next_state([next_state], move)

            next_state = next_states[0]
            tc = tcs[0]

            path.append(next_state)
            transition_costs.append(tc)

        path_cost: float = sum(transition_costs)

        # record solution information
        results["states"].append(state)
        results["solutions"].append(soln)
        results["paths"].append(path)
        results["times"].append(solve_time)
        results["num_nodes_generated"].append(num_nodes_gen_idx)
        results["solved_states"].append(state_idx)
        results["popped_nodes_info"].append(popped_nodes_info)

        pickle.dump(results, open(results_file, "wb"), protocol=-1)  # save for reproducibility

        # check soln
        assert search_utils.is_valid_soln(state, soln, env)

        # print to screen
        print("State: %i, SolnCost: %.2f, # Moves: %i, "
              "# Nodes Gen: %s, Time: %.2f" % (state_idx, path_cost, len(soln),
                                               format(num_nodes_gen_idx, ","),
                                               solve_time))

    os.unlink(socket_name)

    nnet_utils.stop_heuristic_fn_runners(heur_procs, heur_fn_i_q)

    print("Finished running cpp bwas on all states")


def cpp_listener(sock, args, env: Environment, state_dim: int, heur_fn_i_q, heur_fn_o_qs):
    sock.listen(1)
    connection, client_address = sock.accept()

    # device, devices, on_gpu = nnet_utils.get_device()
    # heuristic_fn = nnet_utils.load_heuristic_fn(args.model_dir, device, on_gpu, env.get_nnet_model(),
    #                                             env, clip_zero=True, batch_size=args.nnet_batch_size)

    max_bytes: int = 4096
    while True:
        data_rec = connection.recv(8)
        while not data_rec:
            connection, client_address = sock.accept()
            data_rec = connection.recv(8)

        num_bytes_recv = np.frombuffer(data_rec, dtype=np.int64)[0]

        num_bytes_seen = 0
        data_rec = b""
        while num_bytes_seen < num_bytes_recv:
            con_rec = connection.recv(max_bytes)
            data_rec = data_rec + con_rec
            num_bytes_seen = num_bytes_seen + len(con_rec)

        states_np = np.frombuffer(data_rec, dtype=env.dtype)
        states_np = states_np.reshape(int(len(states_np) / state_dim), state_dim)

        # Get nnet representation of state
        if args.env.upper() == "CUBE3":
            states_np = states_np / 9
            states_np = states_np.astype(env.dtype)
            states_nnet: List[np.ndarray] = [states_np]
        elif args.env.upper() in ["PUZZLE15", "PUZZLE24", "PUZZLE35", "PUZZLE48"]:
            states_np = states_np.astype(env.dtype)
            states_nnet: List[np.ndarray] = [states_np]
        elif args.env.upper() in ["LIGHTSOUT7"]:
            states_np = states_np.astype(env.dtype)
            states_nnet: List[np.ndarray] = [states_np]
        else:
            raise ValueError("Unknown c++ environment %s" % args.env)

        # get heuristic
        results = heuristic_fn_par(states_nnet, heur_fn_i_q, heur_fn_o_qs)

        # send results
        connection.sendall(results.astype(np.float32))


def get_h_g_from_cpp(lines):
    popped_nodes_info = {
        "heuristic": [],
        "cost": [],
    }
    try:
        start_idx = lines.index("ALL POPPED NODES INFO:") + 1
        end_idx = lines.index("Move nums:")
        for line in lines[start_idx:end_idx]:
            # Expected format: "cost=5.00,heuristic=2.00"
            parts = line.split(',')
            cost = float(parts[0].split('=')[1].strip())
            heuristic = float(parts[1].split('=')[1].strip())
            popped_nodes_info["heuristic"].append(heuristic)
            popped_nodes_info["cost"].append(cost)
    except ValueError:
        pass

    return popped_nodes_info


def heuristic_fn_par(states_nnet: List[np.ndarray], heur_fn_i_q, heur_fn_o_qs):
    num_parallel: int = len(heur_fn_o_qs)

    num_states: int = states_nnet[0].shape[0]

    parallel_nums = range(min(num_parallel, num_states))
    split_idxs = np.array_split(np.arange(num_states), len(parallel_nums))
    for idx in parallel_nums:
        states_nnet_idx = [x[split_idxs[idx]] for x in states_nnet]
        heur_fn_i_q.put((idx, states_nnet_idx, True))

    # Check until all data is obtaied
    results = [None] * len(parallel_nums)
    for idx in parallel_nums:
        results[idx] = heur_fn_o_qs[idx].get()

    results = np.concatenate(results, axis=0)

    return results


if __name__ == "__main__":
    main()
