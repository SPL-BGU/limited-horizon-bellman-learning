import heapq
from typing import List, Tuple, Callable
import numpy as np
from environments.environment_abstract import Environment, State
from search_methods.astar import Node
from utils import misc_utils


def is_valid_soln(state: State, soln: List[int], env: Environment) -> bool:
    soln_state: State = state
    move: int
    for move in soln:
        soln_state = env.next_state([soln_state], move)[0][0]

    return env.is_solved([soln_state])[0]


def bellman(states: List, target_heuristic_fn: Callable, env: Environment,
            method: str) -> Tuple[np.ndarray, List[np.ndarray], List[List[State]]]:
    # expand states
    states_exp, tc_l = env.expand(states)
    tc = np.concatenate(tc_l, axis=0)

    # get cost-to-go of expanded states
    states_exp_flat, split_idxs = misc_utils.flatten(states_exp)
    ctg_next: np.ndarray = target_heuristic_fn(states_exp_flat)

    # backup cost-to-go
    ctg_next_p_tc = tc + ctg_next
    ctg_next_p_tc_l = np.split(ctg_next_p_tc, split_idxs)

    is_solved = env.is_solved(states)

    if method == "bellman":
        ctg_backup = np.array([np.min(x) for x in ctg_next_p_tc_l]) * np.logical_not(is_solved)
    else:
        print(f"Unknown update method {method}")
        raise f"Unknown update method {method}"

    return ctg_backup, ctg_next_p_tc_l, states_exp


def run_dijkstra_reverse(root: Node):
    root.lhu_val = 0.0
    pq = [(root.lhu_val, id(root), root)]

    while pq:
        current_cost, _, current_node = heapq.heappop(pq)
        if current_cost != current_node.lhu_val:
            continue

        # Relax every edge from current_node
        for parent, edge_cost in zip(current_node.parents, current_node.parents_tcs):
            new_cost = current_cost + edge_cost
            if parent.lhu_val is None or new_cost < parent.lhu_val:
                parent.lhu_val = new_cost
                heapq.heappush(pq, (new_cost, id(parent), parent))


def run_limited_horizon_update(leaves: List[Node]):
    dummy_node = Node(state=None, path_cost=0.0, is_solved=False, parent_move=None, parent=None, heuristic=0.0)
    for leaf in leaves:
        if leaf.is_solved:
            leaf.lhu_val = 0.0
        for parent in leaf.parents:
            dummy_node.parents.append(parent)
            dummy_node.parents_tcs.append(1.0 if leaf.is_solved else max(leaf.heuristic, 0.0) + 1.0)

    run_dijkstra_reverse(dummy_node)
