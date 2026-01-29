import os.path
import pickle
import sys
import numpy as np
from scipy.ndimage import gaussian_filter1d

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the base path to sys.path to resolve environment imports when loading pickle files
sys.path.append(BASE_PATH)

from matplotlib import pyplot as plt
from scripts.generate_heatmaps import extract_method_name

start_end_idx = {
    "puzzle35": [117, 120],
    "cube3": [13, 15],
    "lightsout7": [84, 86],
}

# State indices for the combined plot (one per domain)
combined_state_indices = {
    "puzzle35": 117,
    "cube3": 14,
    "lightsout7": 85,
}

domain_mapping = {
    'puzzle35': '35-Tile Puzzle',
    'lightsout7': 'Lightsout 7x7',
    'cube3': "Rubik's Cube"
}


def load_pkl_file(pkl_path, domain, filename="results-cp100-s{}-e{}-tc10.pkl"):
    start_end = start_end_idx[domain]
    filename = filename.format(start_end[0], start_end[1])
    with open(os.path.join(pkl_path, filename), 'rb') as file:
        data = pickle.load(file)

    return data


def smooth_values(values, method='moving_avg', window_size=100, sigma=10):
    """
    Smooth the values using different methods.
    
    Args:
        values: List of values to smooth
        method: Smoothing method ('moving_avg' or 'gaussian')
        window_size: Window size for moving average
        sigma: Standard deviation for Gaussian filter
        
    Returns:
        Smoothed values
    """
    if len(values) <= window_size:
        return values
        
    if method == 'moving_avg':
        # Simple moving average
        smoothed = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
        # Pad the beginning to match original length
        padding = np.full(window_size-1, smoothed[0])
        return np.concatenate((padding, smoothed))
    elif method == 'gaussian':
        # Gaussian filter
        return gaussian_filter1d(values, sigma=sigma)
    else:
        return values


def plot_h(values_per_run, save_path, state_index, seed, smooth=True, smooth_method='gaussian', window_size=10, sigma=2):
    # Plot the graph
    plt.rcParams.update(
        {
            'font.size': 22, 
            'axes.titlesize': 30, 
            'axes.labelsize': 48, 
            'legend.fontsize': 18, 
            'xtick.labelsize': 20, 
            'ytick.labelsize': 20
        }
    )
    plt.figure(figsize=(8.5, 7))

    for run, values in values_per_run.items():
        x_values = list(range(len(values)))
        run_name = extract_method_name(run)
        
        if smooth:
            smoothed_values = smooth_values(values, method=smooth_method, window_size=window_size, sigma=sigma)
            plt.plot(x_values, smoothed_values, linestyle='-', linewidth=2, label=run_name)
            # Optionally plot original values with lower alpha
            # plt.plot(x_values, values, linestyle='-', linewidth=1, alpha=0.3, color=plt.gca().lines[-1].get_color())
        else:
            plt.plot(x_values, values, linestyle='-', linewidth=2, label=run_name)

    plt.xscale("log")
    plt.xlabel("Expansions")
    plt.ylabel("Heuristic")
    plt.title(domain_mapping[domain])
    plt.legend(loc="upper right", markerscale=1.5, fontsize="medium")
    
    smooth_suffix = f"_smooth_{smooth_method}" if smooth else ""
    filename = f"{domain}_h_expansions_{seed}_{state_index}{smooth_suffix}.png"
    plt.savefig(os.path.join(save_path, filename))
    print(f"Saved {os.path.join(save_path, filename)}")


def main(results_folder, domain, runs_to_plot, seed=15, save_path="", smooth=True, smooth_method='gaussian'):
    print(f"Plotting {results_folder} for {domain}")
    values_per_run = {}
    for state_index in [0, 1, 2]:
        if state_index == 2 and domain != 'puzzle35':
            continue
        for run in runs_to_plot:
            pkl_path = os.path.join(results_folder, run.format(seed))
            pkl_data = load_pkl_file(pkl_path, domain)
            h_vals = pkl_data["popped_nodes_info"][state_index]["heuristic"]
            values_per_run[run] = h_vals[1:]

        plot_h(values_per_run, save_path, state_index=start_end_idx[domain][0] + state_index,
               seed=seed, smooth=smooth, smooth_method=smooth_method)


def plot_h_combined(all_domain_data, save_path, smooth=False, smooth_method='gaussian', window_size=10, sigma=2):
    """
    Create a combined figure with 3 subplots side-by-side and a shared legend at the bottom.

    Args:
        all_domain_data: Dict with structure {domain: {"values_per_run": {...}, "seed": int, "state_index": int}}
        save_path: Directory to save the combined plot
        smooth: Whether to apply smoothing
        smooth_method: Smoothing method ('moving_avg' or 'gaussian')
        window_size: Window size for moving average
        sigma: Standard deviation for Gaussian filter
    """
    # Font settings matching plot_metrics_combined
    plt.rcParams.update({
        'font.size': 20,
        'axes.titlesize': 24,
        'axes.labelsize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
    })

    # Order: puzzle35, lightsout7, cube3
    domains_order = ["puzzle35", "lightsout7", "cube3"]

    # Custom colors: SSBL=blue, LHBL_S=oranges/reds, LHBL=greens/browns
    method_colors = {
        "bellman_{}_1": "#1f77b4",        # blue for SSBL
        "bellman_{}_10": "#ff7f0e",       # orange for LHBL_S(10)
        "bellman_{}_50": "#d62728",       # red for LHBL_S(50)
        "bellman_{}_100": "#c51b7d",      # magenta-rose (replaces pink)
        "tree_bellman_{}_10": "#2ca02c",  # green for LHBL(10)
        "tree_bellman_{}_50": "#8c564b",  # brown for LHBL(50)
        "tree_bellman_{}_100": "#1b5e20", # dark green for LHBL(100)
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, domain in zip(axes, domains_order):
        domain_data = all_domain_data[domain]
        values_per_run = domain_data["values_per_run"]

        for run, values in values_per_run.items():
            x_values = list(range(len(values)))
            run_name = extract_method_name(run)
            color = method_colors.get(run, None)

            # Apply smoothing: cube3 always uses moving_avg, others based on smooth parameter
            if domain == "cube3":
                plot_values = smooth_values(values, method='moving_avg', window_size=window_size, sigma=sigma)
            elif smooth:
                plot_values = smooth_values(values, method=smooth_method, window_size=window_size, sigma=sigma)
            else:
                plot_values = values

            ax.plot(x_values, plot_values, linestyle='-', linewidth=2, label=run_name, color=color)

        # Only use log scale for lightsout7
        if domain == "lightsout7":
            ax.set_xscale("log")
        ax.set_xlabel("Expansions")
        ax.set_title(domain_mapping[domain])

    # Only set ylabel on leftmost plot
    axes[0].set_ylabel("Heuristic")

    # Single legend below all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    # Create copies of handles with thicker lines for legend only
    from copy import copy
    legend_handles = []
    for handle in handles:
        legend_handle = copy(handle)
        legend_handle.set_linewidth(4)
        legend_handles.append(legend_handle)
    fig.legend(legend_handles, labels,
               loc='lower center',
               ncol=len(labels),
               frameon=False,
               bbox_to_anchor=(0.5, -0.12))

    # Adjust spacing
    fig.subplots_adjust(left=0.05,
                        right=0.98,
                        top=0.88,
                        bottom=0.18,
                        wspace=0.18)

    smooth_suffix = f"_smooth_{smooth_method}" if smooth else ""
    filename = f"h_expansions_combined{smooth_suffix}.png"
    out_path = os.path.join(save_path, filename)
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"Saved combined figure to {out_path}")


def main_combined(results_base_dir, runs_to_plot, save_path, smooth=False, smooth_method='gaussian'):
    """
    Generate a combined plot with all 3 domains side-by-side.
    """
    # Domain configuration: domain -> (weight, seed, state_index)
    domain_config = {
        "puzzle35": {"weight": "0_4", "seed": 10, "state_index": 0},  # state 117
        "lightsout7": {"weight": "0_2", "seed": 10, "state_index": 1},  # state 85
        "cube3": {"weight": "0_2", "seed": 15, "state_index": 1},  # state 14
    }
    batch_size = "1"

    all_domain_data = {}

    for domain, config in domain_config.items():
        weight = config["weight"]
        seed = config["seed"]
        state_index = config["state_index"]

        folder = os.path.join(results_base_dir, domain, weight, batch_size)
        values_per_run = {}

        for run in runs_to_plot:
            pkl_path = os.path.join(folder, run.format(seed))
            pkl_data = load_pkl_file(pkl_path, domain)
            h_vals = pkl_data["popped_nodes_info"][state_index]["heuristic"]
            values_per_run[run] = h_vals[1:]

        all_domain_data[domain] = {
            "values_per_run": values_per_run,
            "seed": seed,
            "state_index": combined_state_indices[domain]
        }

    os.makedirs(save_path, exist_ok=True)
    plot_h_combined(all_domain_data, save_path, smooth=smooth, smooth_method=smooth_method)


if __name__ == "__main__":
    results_base_dir = "results/"
    runs_to_plot = ["bellman_{}_1", "bellman_{}_10", "bellman_{}_50", "bellman_{}_100", "tree_bellman_{}_10", "tree_bellman_{}_50", "tree_bellman_{}_100"]

    # Smoothing configuration
    smooth = False
    # smooth_method = 'moving_avg'  # 'moving_avg' or 'gaussian'

    # Generate combined plot (3 domains side-by-side with shared legend)
    combined_save_path = os.path.join(results_base_dir, "plots/combined")
    main_combined(results_base_dir, runs_to_plot, combined_save_path, smooth=smooth)
