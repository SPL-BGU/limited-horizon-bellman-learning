import pickle
from collections import defaultdict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from matplotlib import pyplot as plt
import os
from matplotlib.ticker import ScalarFormatter
from PIL import Image
from itertools import cycle
from argparse import ArgumentParser

marker_styles = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'x', '+']
marker_cycle = cycle(marker_styles)

MAX_STATES = 200


def extract_method_name(method: str):
    algo_name = method.replace("_", "")
    algo_name = algo_name.replace("{}", "@")
    split = algo_name.split("@")
    prefix, steps = split[0], split[1]
    if steps == "1":
        algo_name = "$SSBL$"
    elif "tree" in prefix.lower():
        algo_name = f"$LHBL({steps})$"
    else:
        algo_name = fr"$LHBL_S({steps})$"

    return algo_name


def load_pkl_file(filepath, checkpoint=10):
    with open(filepath, 'rb') as file:
        data = pickle.load(file)

    return data


def plot_results_folder_combined(config, seeds, sub_folders, algo_names, same_states=False):
    main_dir = "results/RLC/"
    plot_results = {domain: {} for domain in config.keys()}

    for domain, conf in config.items():
        weight = conf["weight"]
        batch_size = conf["batch_size"]

        print("Starting plots for domain: ", domain)
        par_dir = os.path.join(main_dir, domain, weight, batch_size)
        for seed in seeds:
            directories = []
            for sub_f in sub_folders:
                dir = sub_f.format(seed)
                directories.append(f"{par_dir}/{dir}")
            res = get_same_states_results_for_plot(directories, algo_names) if same_states else get_results_for_plots(directories)
            plot_results[domain][seed] = res

    save_dir = os.path.join(main_dir, "combined_plots")
    plt_std_mean = len(seeds) != 1
    plot_metrics_combined(plot_results, save_dir, algo_names, plt_std_mean, same_states=same_states)


def plot_results_folder_per_domain(domain, seeds, sub_folders, algo_names, weight=None, batch_size=None, test_name=None, same_states=False):
    """
    Plot results from 2 folders containing pkl files of 2 algorithms
    :param domain: domain name (for locating the result folder)
    :param seeds: seed numbers (for locating the result folder)
    :param same_states: if True, plots on states that both checkpoints solved successfully
    :return:
    """
    main_dir = "results/"
    print("Starting plots for domain: ", domain)

    par_dir = f"{main_dir}/{domain}"
    save_dir = os.path.join(par_dir, f"seed{seeds}_plots")

    # par_dir = os.path.join(par_dir, test_name)

    # same_states = False
    plot_results = {}
    for seed in seeds:
        directories = []
        for sub_f in sub_folders:
            d = sub_f.format(seed)
            directories.append(f"{par_dir}/{d}")
        res = get_same_states_results_for_plot(directories, algo_names) if same_states else get_results_for_plots(
            directories)
        plot_results[seed] = res

    plt_std_mean = len(seeds) != 1
    plot_metrics_per_domain(plot_results, save_dir, algo_names, test_name, plt_std_mean, domain, same_states=same_states)


def plot_metrics_per_domain(metrics_dict, save_dir, algo_names, test_name, plt_std_mean, domain, same_states=False,
                 close_plots=False,
                 solved_all=False, log_scale=False):
    if close_plots:
        save_dir = os.path.join(save_dir, "close_plots")

    if same_states:
        save_dir = os.path.join(save_dir, test_name)

    if solved_all:
        save_dir = os.path.join(save_dir, "solved_all")

    if log_scale:
        save_dir = os.path.join(save_dir, "log_scale")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    domain_names = {
        "cube3": "Rubik's Cube",
        "puzzle15": "15-Tile Puzzle",
        "puzzle35": "35-Tile Puzzle",
        "lightsout7": "Lightsout 7x7",
    }

    title_dict = {
        "steps": r"Iteration ($\times 10^4$)",
        "average_generated_nodes": [r"Avg. Generated Nodes",
                                    {"cube3": "upper right", "lightsout7": "upper left", "puzzle35": "lower right", "puzzle15": "lower right"}],
        "average_solution_length": ["Avg. Solution Length (moves)",
                                    {"cube3": "lower right", "lightsout7": "lower right", "puzzle35": "lower right", "puzzle15": "lower right"}],
        # "average_time": ["Avg. Solution Time (sec)",
        #                  {"cube3": "upper right", "lightsout7": "upper left", "puzzle35": "lower right"}],
        "percentage_solved": ["Problems Solved (%)",
                              {"cube3": "lower right", "lightsout7": "upper left", "puzzle35": "lower right", "puzzle15": "lower right"}],
    }

    plt.rcParams.update({'font.size': 18, 'axes.titlesize': 26, 'axes.labelsize': 24, 'legend.fontsize': 14})

    plots = [key for key in title_dict.keys() if key != 'steps']

    seeds = list(metrics_dict.keys())

    for plot in plots:
        fig, ax = plt.subplots(figsize=(8.1, 6.5))

        if plt_std_mean:
            for algo_name in algo_names:
                # Aggregate metrics for the given key across all seeds
                mean_values, std_values = calculate_mean_std(metrics_dict, algo_name, plot)
                steps = list(mean_values.keys())
                mean_values, std_values = list(mean_values.values()), list(std_values.values())

                divider = 1e6 if plot == 'average_generated_nodes' else 1
                to_print = [float(round(num / divider, 2)) for num in mean_values]
                formatted_string = " & ".join(f"{num:.2f}" for num in to_print)

                print(f"{algo_name}: {formatted_string}")
                marker = algo_markers[algo_name]

                if domain == "cube3":
                    if plot != "percentage_solved" and algo_name == "Bellman@1":
                        steps = metrics_dict[10][algo_name]["steps"]
                        mean_values = [None] + list(mean_values)
                        ax.plot(steps, mean_values, marker=marker, markersize=8, label=algo_name)
                        continue

                ax.plot(steps, mean_values, marker=marker, markersize=8, label=algo_name)
                ax.fill_between(steps,
                                np.array(mean_values) - np.array(std_values),
                                np.array(mean_values) + np.array(std_values),
                                alpha=0.3)
        else:
            for algo_name in algo_names:
                algo_dict = metrics_dict[seeds[0]][algo_name]
                algo_dict = dict(sorted(algo_dict.items()))
                values = metrics_dict[seeds[0]][algo_name][plot]
                steps = metrics_dict[seeds[0]][algo_name]['steps']
                marker = algo_markers[algo_name]

                divider = 1e6 if plot == 'average_generated_nodes' else 1
                to_print = [float(round(num / divider, 2)) for num in values]
                formatted_string = " & ".join(f"{num:.2f}" for num in to_print)
                print(f"{algo_name}: {formatted_string}")

                ax.plot(steps, values, marker=marker, markersize=8, label=algo_name)

        title = domain_names[domain]
        # ax.set_title(f"{title[:title.index('(')].replace('Avg.', 'Average') if 'Avg.' in title else title}")
        # ax.set_yticks([0, 5, 10, 15, 20])
        ax.set_title(f"{title.replace('Avg.', 'Average') if 'Avg.' in title else title}")
        ax.set_xlabel(title_dict['steps'])
        ax.set_ylabel(title_dict[plot][0].replace('Avg. ', ''))

        if log_scale and plot == 'average_generated_nodes':
            ax.set_yscale('log')

        legend_loc = title_dict[plot][1][domain]
        ax.legend(loc=legend_loc, markerscale=1.5)
        ax.legend(loc=legend_loc, markerscale=1.5, fontsize='small')

        if not close_plots:
            all_steps = []
            all_values = []
            for seed, seed_res in metrics_dict.items():
                for plot_label, values in seed_res.items():
                    all_steps.extend(values.get('steps', []))
                    all_values.extend(values.get(plot, []))

            if all_steps and all_values:
                max_step = max(all_steps)
                min_step = min(all_steps)
                max_value = max(all_values)
                ax.set_xlim(min_step, max_step)
                ax.set_ylim(0, max_value * 1.01)

        # Save the plot
        save_file_location = os.path.join(save_dir, f'{plot}.png')
        plt.savefig(save_file_location)
        print(f"Saved {title} vs Steps comparison plot in {save_file_location}")

        plt.close(fig)

        image = Image.open(save_file_location)

        # Crop the image (example: crop 100 pixels from the right side)
        width, height = image.size
        left = 0
        top = 30
        right = width - 50  # Crop 100 pixels from the right
        bottom = height

        # Crop the image
        cropped_image = image.crop((left, top, right, bottom))

        # Save the cropped image
        cropped_image.save(save_file_location)


def plot_metrics_combined(metrics_dict,
                          save_dir,
                          algo_names,
                          plt_std_mean,
                          close_plots=False,
                          same_states=False):
    """
    Combined side-by-side percentage_solved plots with a single legend underneath.
    """

    if close_plots:
        save_dir = os.path.join(save_dir, "close_plots")
    os.makedirs(save_dir, exist_ok=True)

    # ——— bump up all the font sizes ———
    plt.rcParams.update({
        'font.size': 20,
        'axes.titlesize': 24,
        'axes.labelsize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 20,
    })

    domain_names = {
        "puzzle35": "35-Tile Puzzle",
        "puzzle15": "15-Tile Puzzle",
        "cube3":    "Rubik's Cube",
        "lightsout7":"Lightsout 7×7",
    }
    x_label = "Training Progress (%)"
    y_label = "Problems Solved (%)"

    domains = list(metrics_dict.keys())
    fig, axes = plt.subplots(1, len(domains), figsize=(18, 5))

    for ax, domain in zip(axes, domains):
        for algo in algo_names:
            display_name = extract_method_name(algo)
            mean_dict, std_dict = calculate_mean_std(metrics_dict[domain], algo, "percentage_solved")
            steps = np.array(list(mean_dict.keys()))
            means = np.array(list(mean_dict.values()))
            stds  = np.array(list(std_dict.values()))

            ax.plot(steps, means,
                    marker=algo_markers[algo],
                    markersize=8,
                    label=display_name)
            if plt_std_mean:
                ax.fill_between(steps,
                                means - stds,
                                means + stds,
                                alpha=0.18)

        ax.set_title(domain_names[domain])
        ax.set_xlabel(x_label)
        # ax.set_ylabel(y_label)

        ax.set_xlim(steps.min(), steps.max())
        ax.set_ylim(0, 100)

    axes[0].set_ylabel(y_label)

    # single legend below
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='lower center',
               ncol=len(labels),
               frameon=False,
               bbox_to_anchor=(0.5, -0.12))

    # tighten up spacing: less space between plots, room at bottom for legend
    fig.subplots_adjust(left=0.05,
                        right=0.98,
                        top=0.88,
                        bottom=0.18,
                        wspace=0.18)    # was 0.25 → now much tighter

    out_path = os.path.join(save_dir, "percentage_solved_side_by_side.png")
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"Saved combined figure to {out_path}")


def calculate_mean_std(metrics_dict, algo_name, plot):
    step_values = {}

    for seed, algo_data in metrics_dict.items():
        if algo_name in algo_data:
            steps = algo_data[algo_name]['steps']
            avg_generated_nodes = algo_data[algo_name][plot]

            for step, avg_gen_nodes in zip(steps, avg_generated_nodes):
                if step not in step_values:
                    step_values[step] = []
                step_values[step].append(avg_gen_nodes)

    # Calculate the mean for each step
    sorted_steps_values = dict(sorted(step_values.items()))
    mean_per_step = {step: np.mean(values) for step, values in sorted_steps_values.items()}
    std_per_step = {step: np.std(values) for step, values in sorted_steps_values.items()}
    return mean_per_step, std_per_step


def get_same_states_results_for_plot(directories, algo_names):
    sorted_files = []
    for directory in directories:
        if os.path.exists(directory):
            sorted_files.append(
                sorted([(f, int(f.split('-')[1][2:])) for f in os.listdir(directory) if f.endswith('.pkl')],
                       key=lambda x: x[1]))

    aggregated_dict = defaultdict(list)

    for sublist in sorted_files:
        for item in sublist:
            aggregated_dict[item[1]].append(item[0])

    plot_results_dicts = {
        algo_name: {"steps": [], "average_solution_length": [], "average_time_taken": [], "average_generated_nodes": [],
                    "percentage_solved": []}
        for algo_name in algo_names
    }

    aggregated_dict = dict(sorted(aggregated_dict.items()))

    for cp_num, files in aggregated_dict.items():
        filename = files[0]
        results_dicts = {algo_name: load_pkl_file(os.path.join(directory, filename)) for directory, algo_name in
                         zip(directories, algo_names) if os.path.exists(os.path.join(directory, filename))}
        get_same_states_results(plot_results_dicts, results_dicts, cp_num)

    return plot_results_dicts


def get_results_for_plots(directories):
    res = {}

    for directory in directories:
        print(f"Processing {directory}")
        plot_results = defaultdict(list)
        sorted_files = sorted([f for f in os.listdir(directory) if f.endswith('.pkl')],
                              key=lambda x: int(x.split('-')[1][2:]))

        for filename in sorted_files:
            if filename.endswith('.pkl'):
                print(f"Processing {filename}")
                file_path = os.path.join(directory, filename)
                checkpoint_iter_n = filename.split('-')[1][2:]

                # Open and load the pkl file
                with open(file_path, 'rb') as file:
                    results = pickle.load(file)

                results["states"] = results["states"][:MAX_STATES]

                for key in ['solutions', 'times', 'num_nodes_generated']:
                    results[key] = results[key][:MAX_STATES]
                    results[key] = [x for x in results[key] if x is not None]

                solved = len(results["solutions"]) != 0

                if solved:
                    plot_results["steps"].append(int(checkpoint_iter_n))
                    plot_results["average_solution_length"].append(np.mean([len(sol) for sol in results["solutions"]]))
                    plot_results["average_time_taken"].append(np.mean(results["times"]))
                    plot_results["average_generated_nodes"].append(np.mean(results["num_nodes_generated"]))
                    plot_results["percentage_solved"].append(
                        100 * len(results["num_nodes_generated"]) / len(results["states"])
                    )

        directory_method = directory.split('/')[-1].split('_')[0]
        # directory_label = dir_plot_labels[directory_method]
        # res[directory_label] = plot_results

    return res


def pad_results(results_dict):
    for results in results_dict.values():
        results['states'] = results['states'][:MAX_STATES]

        indices = results['solved_states']
        if len(indices) == MAX_STATES or len(indices) == 0:
            continue

        result_indices = [None] * MAX_STATES
        for key in ['solutions', 'times', 'num_nodes_generated']:
            results[key] = results[key][:MAX_STATES]
            elements = results[key]
            result_elements = [None] * MAX_STATES

            for idx, elem in zip(indices, elements):
                if idx is None:
                    print()
                result_elements[idx] = elem
                result_indices[idx] = idx

            results[key] = result_elements

        results['solved_states'] = result_indices


def get_same_states_results(plot_results_dict, results_dict, checkpoint_iter_n):
    # Step 1: Find solved states that are common across all result dicts
    for algo_name, results in results_dict.items():
        for key, lst in results.items():
            if isinstance(lst, list):
                results_dict[algo_name][key] = lst[:MAX_STATES]

    pad_results(results_dict)

    first_algo_name = list(results_dict.keys())[0]
    results_dict[first_algo_name]["solved_states"] = [x for x in results_dict[first_algo_name]["solved_states"] if
                                                      x is not None]
    solved_states = set(results_dict[first_algo_name]["solved_states"])
    algo_to_delete = []  # if one algo did not solve anything, we want the states of the other algos
    for algo_name, results in results_dict.items():
        results['solved_states'] = [x for x in results['solved_states'] if x is not None]
        if len(results['solved_states']) == 0:
            plot_results_dict[algo_name]["steps"].append(int(checkpoint_iter_n))
            plot_results_dict[algo_name]["average_solution_length"].append(None)
            plot_results_dict[algo_name]["average_time_taken"].append(None)
            plot_results_dict[algo_name]["average_generated_nodes"].append(None)
            plot_results_dict[algo_name]["percentage_solved"].append(0.0)
            algo_to_delete.append(algo_name)
            continue
        solved_states.intersection_update(set(results['solved_states']))

    for algo in algo_to_delete:
        del results_dict[algo]

    total_solved = len(solved_states)
    if total_solved == 0:
        return

    solved_states = list(solved_states)

    # Step 2: For each plot results dict, append the data based on common solved states
    for algo_name, results in results_dict.items():
        plot_results_dict[algo_name]["steps"].append(int(checkpoint_iter_n))
        plot_results_dict[algo_name]["average_solution_length"].append(
            sum([len(results["solutions"][i]) for i in solved_states]) / total_solved
        )
        plot_results_dict[algo_name]["average_time_taken"].append(
            sum([results["times"][i] for i in solved_states]) / total_solved
        )
        plot_results_dict[algo_name]["average_generated_nodes"].append(
            sum([results["num_nodes_generated"][i] for i in solved_states]) / total_solved
        )
        plot_results_dict[algo_name]["percentage_solved"].append(
            100 * len(results["solved_states"]) / len(results["states"])
        )


def scatter_plot(domain, seeds, result_filenames, sub_folders, algo_names, weight=None, batch_size=None,
                 output_filename="scatter.png"):
    par_dir = f"results/Dobell/{domain}"
    results_dir = os.path.join(par_dir, f"seed{seeds}_plots")

    # par_dir = os.path.join(par_dir, f"{weight}/{batch_size}")

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    results = {}

    # Load the result files
    for sub_folder, result_filename, algo_name in zip(sub_folders, result_filenames, algo_names):
        seed_res = []
        for seed in seeds:
            file_path = os.path.join(par_dir, sub_folder.format(seed), result_filename)
            if os.path.isfile(file_path):
                print(f"Loading {file_path}")
                with open(file_path, 'rb') as file:
                    data = pickle.load(file)
                    seed_res.append(data['num_nodes_generated'])
        mean_res = [sum(values) / len(values) for values in zip(*seed_res)]
        results[algo_name] = mean_res

    for algo_name in algo_names:
        results[algo_name] = [value / 1e6 for value in results[algo_name] if value is not None]

    combined = list(zip(*[results[algo_name] for algo_name in algo_names]))
    combined_sorted = sorted(combined, key=lambda x: x[1])

    sorted_results = list(zip(*combined_sorted))

    # Plotting
    plt.rcParams.update({'font.size': 18, 'axes.titlesize': 26, 'axes.labelsize': 24, 'legend.fontsize': 26})
    fig, ax = plt.subplots(figsize=(8.1, 6.5))
    # fig, ax = plt.subplots(figsize=(7.6, 6))

    # Scatter plot for each result
    for i, algo_name in enumerate(algo_names):
        ax.scatter(range(len(sorted_results[i])), sorted_results[i], label=algo_name, marker=algo_markers[algo_name])

    plt.xlabel('Problem Instance')
    plt.ylabel(r'Generated Nodes ($\times 10^6$)')
    plt.title('Generated Nodes Per Instance')

    max_value = max(max(results[algo_name][:200]) for algo_name in algo_names)
    ax.set_ylim(0, max_value * 1.01)
    ax.set_xlim(0, 200)

    # Remove scientific notation from y-axis
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.ticklabel_format(style='plain', axis='y')

    # plt.legend(loc='lower left', markerscale=1.7)
    plt.legend(loc='upper left', markerscale=1.7)

    # Save the plot
    save_file_location = os.path.join(results_dir, output_filename)
    plt.savefig(save_file_location)
    print(f"Saved scatter plot in {save_file_location}")

    plt.close(fig)

    image = Image.open(save_file_location)

    # Crop the image (example: crop 100 pixels from the right side)
    width, height = image.size
    left = 0
    top = 30
    right = width - 50  # Crop 100 pixels from the right
    bottom = height

    # Crop the image
    cropped_image = image.crop((left, top, right, bottom))

    # Save the cropped image
    cropped_image.save(save_file_location)


def parse_arguments(parser: ArgumentParser):
    parser.add_argument('--run_underestimation', action='store_true', default=False,
                        help="Run the underestimation test of cube")
    parser.add_argument('--domain', type=str, help="Update number")
    parser.add_argument('--seeds', type=str, default="5,10,15", help="Results seeds to plot")
    parser.add_argument('--scatter_plot', action='store_true', default=False, help="Run the scatter plot")

    # parse arguments
    return parser.parse_args()


def get_algo_names(folders):
    algo_names = []
    for folder in folders:
        # Remove the placeholder '{}' from the folder name
        base_name = folder.replace('_{}', '')

        # Split the remaining string by underscores
        parts = base_name.split('_')

        if len(parts) < 2:
            # If there's no number part, just capitalize the name
            algo_name = base_name.capitalize()
        else:
            # The last part is assumed to be the number
            *name_parts, number = parts

            # Capitalize each part of the algorithm name and join them
            capitalized_name = ''.join(part.capitalize() for part in name_parts)

            # Combine the name with the number using '@'
            algo_name = f"{capitalized_name}@{number}"

        # Append the formatted name to the list
        algo_names.append(algo_name)

    return algo_names


def check_dist(data):
    opt = data['optimal_lens']
    pred = data['heuristics']
    labels = data['labels']
    bellman = data['bellman']

    sorted_opt = sorted(enumerate(opt), key=lambda x: x[1])

    sorted_opt_l = [item[1] for item in sorted_opt]
    sorted_indices = [item[0] for item in sorted_opt]

    sorted_pred = pred[sorted_indices]
    sorted_labels = labels[sorted_indices]

    # plot_line_comparison(sorted_opt_l, sorted_pred, sorted_labels)
    plot_line_comparison(opt, pred, labels, bellman)
    # plot_scatter_with_labels(sorted_opt_l, sorted_pred, sorted_labels)
    plot_scatter_with_labels(opt, pred, labels)


def plot_line_comparison(opt, pred, labels, bellman):
    plt.figure(figsize=(14, 7))
    plt.plot(opt, label='Optimal', color='blue', linewidth=2)
    plt.plot(pred, label='Heuristics', color='orange', linewidth=2)
    plt.plot(labels, label='Tree Labels', color='red', linewidth=2)
    plt.plot(bellman, label='Bellman', color='yellow', linewidth=2)
    plt.xlabel('State', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Comparison of Optimal Lenses, Heuristics and Labels', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    # plt.show()
    plt.savefig('overestimation_lines.png')
    plt.close()


def plot_scatter_with_labels(sorted_opt, sorted_pred, sorted_labels):
    fig, ax = plt.subplots(figsize=(8.1, 6.5))

    # Scatter plot for each result
    ax.scatter(range(len(sorted_opt)), sorted_opt, label='Optimal Lenses', marker="o")
    ax.scatter(range(len(sorted_pred)), sorted_pred, label='Heuristics', marker="^")
    ax.scatter(range(len(sorted_labels)), sorted_labels, label='Tree Labels', marker="x")

    plt.xlabel('State', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Comparison of Optimal Lenses, Heuristics and Labels', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    # plt.show()
    plt.savefig('overestimation_scatter.png')
    plt.close(fig)


if __name__ == "__main__":
    global algo_markers
    parser: ArgumentParser = ArgumentParser()
    args = parse_arguments(parser)

    base_str = "{}bellman_{}_{}"
    prefix = ["tree_", ""]
    update_steps = ["10", "50", "100"]
    sub_folders = []
    for pref in prefix:
        for up_step in update_steps:
            sub_folders.append(base_str.format(pref, "{}", up_step))

    sub_folders.append("bellman_{}_1")
    # sub_folders.remove("bellman_{}_10")
    algo_names = get_algo_names(sub_folders)
    algo_markers = {algo: next(marker_cycle) for algo in algo_names}

    seeds = list(map(int, args.seeds.split(',')))

    # ======== FOR AAAI PLOTS ========
    config = {
        'puzzle35': {
            "weight": "0_8",
            "batch_size": "20000"
        },
        'lightsout7': {
            "weight": "0_2",
            "batch_size": "1000"
        },
        'cube3': {
            "weight": "0_6",
            "batch_size": "10000"
        }
    }
    plot_results_folder_combined(config=config, seeds=seeds, sub_folders=sub_folders, algo_names=algo_names, same_states=True)

    for domain, conf in config.items():
        weight = conf["weight"]
        batch_size = conf["batch_size"]
        test_name = f"{weight.replace('_', '.')}_{batch_size}_plots"
        plot_results_folder_per_domain(domain=domain, seeds=seeds, weight=weight, batch_size=batch_size, sub_folders=sub_folders,
                            algo_names=algo_names, test_name=test_name, same_states=True)
