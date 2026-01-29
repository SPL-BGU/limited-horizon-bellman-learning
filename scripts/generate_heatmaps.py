import os
import pickle
from statistics import mean
from typing import Dict, Optional, List
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import re
import sys

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the base path to sys.path to resolve environment imports when loading pickle files
sys.path.append(BASE_PATH)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

title_dict = {
    "steps": r"Iteration ($\times 10^4$)",
    "average_generated_nodes": [r"Avg. Generated Nodes",
                                {"cube3": "upper right", "lightsout7": "upper left", "puzzle35": "lower right"}],
    "average_solution_length": ["Avg. Solution Length (moves)",
                                {"cube3": "lower right", "lightsout7": "lower right", "puzzle35": "lower right"}],
    "average_time": ["Avg. Solution Time (sec)",
                     {"cube3": "upper right", "lightsout7": "upper left", "puzzle35": "lower right"}],
    "percentage_solved": ["Problems Solved (%)",
                            {"cube3": "lower right", "lightsout7": "upper left", "puzzle35": "lower right"}],
}

NUM_OF_TEST_INSTANCES = 200
WEIGHT = "0_2"


def extract_method_name(method: str):
    algo_name = method.replace("_", "")
    algo_name = algo_name.replace("{}", "@")
    split = algo_name.split("@")
    prefix, steps = split[0], split[1]
    if steps == "1":
        algo_name = "$SSBL$"
    elif "tree" in prefix:
        algo_name = f"$LHBL({steps})$"
    else:
        algo_name = fr"$LHBL_S({steps})$"

    return algo_name


def compute_mean_average_generated_nodes(
        root_dir: str,
        seeds: List[int] = [5, 10, 15],
        checkpoint: Optional[int] = None,
        weights: Optional[list] = None,
        batch_sizes: Optional[list] = None,
        methods: Optional[list] = None,
) -> Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]]:
    """
    Traverses the directory structure and computes the mean of 'average_generated_nodes'
    from .pkl files for each weight, batch_size, and method.

    Parameters:
        root_dir (str): The root directory containing the weights folders.
        checkpoint (int): Model checkpoint number
        weights (list, optional): List of weight folder names. If None, all subdirectories in root_dir are used.
        batch_sizes (list, optional): List of batch_size folder names. If None, all subdirectories in weight folders are used.
        methods (list, optional): List of method folder names. If None, all subdirectories in batch_size folders are used.

    Returns:
        Dict[str, Dict[str, Dict[str, Optional[float]]]]: Nested dictionary with mean values.
    """
    result: Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]] = {}

    # If weights not provided, list all subdirectories in root_dir
    if weights is None:
        weights = [
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ]
        logging.info(f"Discovered weights: {weights}")

    for weight in weights:
        weight_path = os.path.join(root_dir, weight)
        if not os.path.isdir(weight_path):
            logging.warning(f"Weight directory '{weight_path}' does not exist. Skipping.")
            continue

        result[weight] = {}

        # If batch_sizes not provided, list all subdirectories in weight_path
        if batch_sizes is None:
            current_batch_sizes = [
                d for d in os.listdir(weight_path)
                if os.path.isdir(os.path.join(weight_path, d))
            ]
            logging.info(f"Discovered batch_sizes for weight '{weight}': {current_batch_sizes}")
        else:
            current_batch_sizes = batch_sizes

        for batch_size in current_batch_sizes:
            batch_path = os.path.join(BASE_PATH, weight_path, batch_size)
            if not os.path.isdir(batch_path):
                logging.warning(f"Batch size directory '{batch_path}' does not exist. Skipping.")
                continue

            result[weight][batch_size] = {}

            # If methods not provided, list all subdirectories in batch_path
            if methods is None:
                current_methods = [
                    d for d in os.listdir(batch_path)
                    if os.path.isdir(os.path.join(batch_path, d))
                ]
                logging.info(f"Discovered methods for weight '{weight}', batch_size '{batch_size}': {current_methods}")
            else:
                current_methods = methods

            for method in current_methods:
                result[weight][batch_size][method] = {}
                average_nodes = []
                problems_solved = []
                average_solution_length = []
                average_time = []

                for seed in seeds:
                    method_name = method.format(seed)
                    method_path = os.path.join(batch_path, method_name)
                    if not os.path.isdir(method_path):
                        logging.warning(f"Method directory '{method_path}' does not exist. Skipping.")
                        continue

                    # Iterate over all .pkl files in the method directory
                    for file_name in os.listdir(method_path):
                        if file_name.endswith('.pkl') and "cp100" in file_name:
                            file_path = os.path.join(method_path, file_name)
                            try:
                                with open(file_path, 'rb') as f:
                                    data = pickle.load(f)
                                    if 'num_nodes_generated' in data:
                                        generated_nodes = data['num_nodes_generated']
                                        if len(generated_nodes) < NUM_OF_TEST_INSTANCES:
                                            logging.warning(
                                                f"Method {method.format(seed)} does not have {NUM_OF_TEST_INSTANCES} instances")
                                            # break
                                        solved_generated_nodes = [val for val in generated_nodes if val is not None]
                                        problems_solved.append(
                                            len(solved_generated_nodes) * 100.0 / len(generated_nodes))
                                        average_nodes.append(np.mean(solved_generated_nodes))
                                        times = [val for val in data['times'] if val is not None]
                                        average_time.append(np.mean(times))
                                    else:
                                        logging.warning(f"Key 'num_nodes_generated' not found in file '{file_path}'.")

                                    if 'solutions' in data:
                                        solutions = data['solutions']
                                        solved_solutions = [val for val in solutions if val is not None]
                                        average_solution_length.append(np.mean([len(sol) for sol in solved_solutions]))
                                    else:
                                        logging.warning(f"Key 'num_nodes_generated' not found in file '{file_path}'.")
                            except Exception as e:
                                logging.error(f"Error reading file '{file_path}': {e}")

                if average_nodes and average_solution_length:
                    result[weight][batch_size][method]['average_generated_nodes'] = (
                        np.mean(average_nodes), np.std(average_nodes)
                    )
                    result[weight][batch_size][method]['percentage_solved'] = (
                        np.mean(problems_solved), np.std(problems_solved)
                    )
                    result[weight][batch_size][method]['average_solution_length'] = (
                        np.mean(average_solution_length), np.std(average_solution_length)
                    )
                    result[weight][batch_size][method]['average_time'] = (np.mean(average_time), np.std(average_time))

                    logging.info(
                        f"Computed mean for weight '{weight}', batch_size '{batch_size}', method '{method}': {np.mean(average_nodes)}")
                else:
                    result[weight][batch_size][method]['average_generated_nodes'] = None
                    result[weight][batch_size][method]['percentage_solved'] = None
                    # result[weight][batch_size][method]['percentage_solved'] = np.mean(problems_solved)

                    logging.warning(
                        f"No valid 'average_generated_nodes' found for weight '{weight}', batch_size '{batch_size}', method '{method}'.")

    return result


# Custom formatter function to make standard deviation appear smaller
def format_cell(mean, std, is_int=True, precision=1, include_std=True):
    if is_int:
        if mean >= 1000000:
            # Get the exponent (power of 10)
            exponent = int(np.floor(np.log10(mean)))
            # Get the mantissa (coefficient)
            mantissa = mean / (10 ** exponent)
            mean_str = f"${mantissa:.1f}×10^{exponent}$"
        else:
            mean_str = f"{mean:,.0f}"
        std_str = f"(±{std:,.0f})" if std >= 1000 else f"(±{std:,.1f})"
    else:
        # Format std with scientific notation if more than 4 digits
        if mean >= 1000000:
            # Get the exponent (power of 10)
            exponent = int(np.floor(np.log10(mean)))
            # Get the mantissa (coefficient)
            mantissa = mean / (10 ** exponent)
            mean_str = f"${mantissa:.1f}×10^{exponent}$"
        else:
            mean_str = f"{mean:.1f}" if precision == 1 else f"{mean:.2f}"
        std_str = f"(±{std:,.0f})" if std >= 1000 else f"(±{std:,.1f})"
    
    if include_std:
        return f"{mean_str}\n{std_str}"
    else:
        return f"{mean_str}"
    

def create_heatmap_batch_size_by_method(mean_dict, save_path="results/RLC/batch_heatmap_plots", weight_to_use=WEIGHT):
    methods = list(next(iter(next(iter(mean_dict.values())).values())).keys())
    os.makedirs(save_path, exist_ok=True)
    metrics = ["average_generated_nodes", "average_solution_length", 'average_time', "percentage_solved"]

    batch_data = mean_dict[weight_to_use]
    
    for metric in metrics:
        data = []

        # Gather data from each batch_size, for each method
        for batch_size, method_dict in batch_data.items():
            for method in methods:
                algo_name = extract_method_name(method)
                metric_values = method_dict.get(method, {}).get(metric, None)
                if metric_values:
                    mean_val, std_val = metric_values  # Unpack (mean, std)
                    data.append([algo_name, batch_size, mean_val, std_val])

        # Create a DataFrame
        df = pd.DataFrame(data, columns=["Method", "Batch Size", "Mean", "Std"])

        # If df is empty (no data), skip
        if df.empty:
            print(f"No data found for weight={weight_to_use}, metric={metric}")
            continue

        # Format the annotations based on metric type
        if metric == "average_generated_nodes":
            df["Formatted"] = df.apply(lambda row: format_cell(row['Mean'], row['Std'], is_int=True, include_std=False), axis=1)
        elif metric == "percentage_solved":
            df["Formatted"] = df.apply(lambda row: format_cell(row['Mean'], row['Std'], is_int=False, precision=1, include_std=row['Mean'] < 100), axis=1)
        else:
            df["Formatted"] = df.apply(lambda row: format_cell(row['Mean'], row['Std'], is_int=False, precision=2), axis=1)

        # Pivot so that columns=Weight, rows=Method, values=Mean
        df_pivot = df.pivot(index="Method", columns="Batch Size", values="Mean")

        # Custom sort key
        def custom_sort_key(val):
            if val.startswith("$LHBL("):
                group = 0
            elif val.startswith("$LHBL_S("):
                group = 1
            else:
                group = 2  # put $SSBL$ last

            match = re.search(r'\((\d+)\)', val)
            number = int(match.group(1)) if match else float('inf')

            return (group, number)

        # Sort using the key
        df_pivot_sorted = df_pivot.sort_values(by="Method", key=lambda x: x.map(custom_sort_key))

        # Create a "Formatted" column to show "mean ± std" as a string
        formatted_pivot = df.pivot(index="Method", columns="Batch Size", values="Formatted")
        df_formatted_pivot_sorted = formatted_pivot.sort_values(by="Method", key=lambda x: x.map(custom_sort_key))

        # Plot heatmap
        plt.rcParams.update({'font.size': 16, 'axes.titlesize': 20, 'axes.labelsize': 20})
        plt.figure(figsize=(9, 6))

        # Choose colormap based on metric "direction"
        if metric in ["percentage_solved"]:
            cmap = "RdYlGn"  # Green = good (high), Red = bad (low)
        else:
            cmap = "RdYlGn_r"  # Reversed: Green = good (low), Red = bad (high)

        ax = sns.heatmap(
            df_pivot_sorted,
            annot=df_formatted_pivot_sorted,  # Use our formatted strings
            fmt="",  # Avoid numerical formatting since we have strings
            cmap=cmap,
            linewidths=0.5,
            cbar=True,  # Disable color bar (optional)
        )

        plt.xlabel("Batch Size")
        # if metric == "average_generated_nodes":
        plt.ylabel("Method")
        # else:
        #     plt.ylabel("")
        #     ax.set_yticks([])  # Removes tick marks
        #     ax.set_yticklabels([])  # Removes tick labels
        # plt.title(f"Heatmap for '{metric.replace('_', ' ').title()}'")
        plt.title(title_dict[metric][0])
        plt.tight_layout()

        # Save the figure
        os.makedirs(save_path, exist_ok=True)
        output_file = os.path.join(save_path, f"{metric}.png")
        plt.savefig(output_file, dpi=300)
        plt.close()

        print(f"Saved heatmap for '{metric}' at: {output_file}")


def create_table_plots_by_method(mean_dict, save_path="results/RLC/heatmap_plots"):
    # Separate data by method
    methods = list(next(iter(next(iter(mean_dict.values())).values())).keys())
    os.makedirs(save_path, exist_ok=True)

    metrics = ["average_generated_nodes", "average_solution_length", "times"]
    for method in methods:
        for metric in metrics:
            # Prepare data for the current method
            data = []
            for weight, batch_data in mean_dict.items():
                for batch_size, method_data in batch_data.items():
                    values = method_data.get(method, None)
                    if values:
                        percentage_solved = values.get("problems_solved", None)
                        values = values.get(metric, None)
                        mean, std = values
                        # if percentage_solved == 100.0:
                        data.append([weight, batch_size, mean, std])

            # Create a DataFrame for the current method
            df = pd.DataFrame(data, columns=["Weight", "Batch Size", "Mean", "Std"])
            df_pivot = df.pivot(index="Weight", columns="Batch Size", values="Mean")
            if metric == "average_generated_nodes":
                # formatted_df = df.map(lambda row: f"{row['Mean']:,.0f}\n±{row['Std']:.1f}" if pd.notnull(row[['Mean']]) and pd.notnull(row[['Std']]) else "", axis=1)
                formatted_df = df.set_index(["Weight", "Batch Size"]).apply(
                    lambda row: f"{row['Mean']:,.0f}\n(±{row['Std']:.1f})", axis=1
                ).unstack()
            else:
                # formatted_df = df.map(lambda row: f"{row['Mean']:.2f}\n±{row['Std']:.1f}" if pd.notnull(row[['Mean']]) and pd.notnull(row[['Std']]) else "", axis=1)
                formatted_df = df.set_index(["Weight", "Batch Size"]).apply(
                    lambda row: f"{row['Mean']:.2f}\n(±{row['Std']:.1f})", axis=1
                ).unstack()

            # formatted_df = formatted_df.pivot(index="Weight", columns="Batch Size")
            
            # Choose colormap based on metric "direction"
            if metric in ["percentage_solved"]:
                cmap = "RdYlGn"  # Green = good (high), Red = bad (low)
            else:
                cmap = "RdYlGn_r"  # Reversed: Green = good (low), Red = bad (high)

            # Plot heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                df_pivot,
                annot=formatted_df,
                fmt="",
                cmap=cmap,
                linewidths=0.5,
                cbar=False,  # Disable color bar
            )
            algo_name = method.replace("_", "")
            algo_name = algo_name.replace("{}", "@")

            plt.title(f"Table Plot for Method '{algo_name}'")
            plt.xlabel("Batch Size")
            plt.ylabel("Weight")
            plt.tight_layout()

            # Save the plot
            save_folder = os.path.join(save_path, metric)
            os.makedirs(save_folder, exist_ok=True)
            output_file = os.path.join(save_folder, f"{algo_name}.png")
            plt.savefig(output_file, dpi=300)
            plt.close()  # Close the figure to avoid overlaps in subsequent plots

            print(f"Saved plot for method '{method}' at: {output_file}")


def create_multi_domain_heatmaps(all_domain_data, save_path="results/RLC/batch_heatmap_plots", weight_to_use=WEIGHT):
    """
    Create heatmaps where each figure represents one metric with 3 subplots (one per domain).
    
    Parameters:
        all_domain_data (dict): Dictionary with domain -> mean_dict structure
        save_path (str): Path to save the figures
        weight_to_use (str): Which weight to use for plotting
    """
    os.makedirs(save_path, exist_ok=True)
    metrics = ["percentage_solved", "average_generated_nodes", 'average_time', "average_solution_length"]
    # metrics = ["average_time", "average_solution_length"]#, 'average_time', "percentage_solved"]
    # domains = list(all_domain_data.keys())
    # domains = ['puzzle35', 'lightsout7', 'cube3']
    domains = ['cube3']
    # domain_mapping = {
    #     'puzzle35': '35-Tile Puzzle',
    #     'lightsout7': 'Lightsout 7x7',
    #     'cube3': "Rubik's Cube"  # Changed from 'Rubik\'s Cube' to "Rubik's Cube"
    # }
    domain_mapping = {'cube3': "Rubik's Cube"}  # Changed from 'Rubik\'s Cube' to "Rubik's Cube"
    
    # Set up the plot style
    plt.rcParams.update({'font.size': 18, 'axes.titlesize': 22, 'axes.labelsize': 18})
    
    for metric in metrics:
        # Create figure with 3 subplots
        fig, ax = plt.subplots(1, 1, figsize=(20, 6))  # Increased figure size for bigger squares
        # fig.suptitle(title_dict[metric][0], fontsize=20, y=0.95)
        
        # Store vmin and vmax for consistent color scaling across domains
        min_vals = []
        max_vals = []
        
        # First pass: collect min and max values for this metric across all domains
        for domain in domains:
            if domain not in all_domain_data:
                continue
                
            mean_dict = all_domain_data[domain]
            
            if weight_to_use not in mean_dict:
                continue
                
            batch_data = mean_dict[weight_to_use]
            methods = list(next(iter(batch_data.values())).keys())
            
            # Gather data from each batch_size, for each method
            for batch_size, method_dict in batch_data.items():
                for method in methods:
                    metric_values = method_dict.get(method, {}).get(metric, None)
                    if metric_values:
                        mean_val, _ = metric_values  # Unpack (mean, std)
                        min_vals.append(mean_val)
                        max_vals.append(mean_val)
        
        vmin = min(min_vals) if min_vals else 0
        vmax = max(max_vals) if max_vals else 1
        
        # Choose colormap based on metric "direction"
        if metric in ["percentage_solved"]:
            cmap = "RdYlGn"  # Green = good (high), Red = bad (low)
        else:
            cmap = "RdYlGn_r"  # Reversed: Green = good (low), Red = bad (high)
        
        # Create a list to store heatmap objects
        heatmaps = []
        
        for idx, domain in enumerate(domains):
            # ax = axes[idx]
            
            if domain not in all_domain_data:
                print(f"Domain {domain} not found in data")
                ax.text(0.5, 0.5, f'No Data for {domain}', ha='center', va='center', transform=ax.transAxes)
                continue
                
            mean_dict = all_domain_data[domain]
            
            if weight_to_use not in mean_dict:
                print(f"Weight {weight_to_use} not found for domain {domain}")
                continue
                
            batch_data = mean_dict[weight_to_use]
            methods = list(next(iter(batch_data.values())).keys())
            
            data = []
            
            # Gather data from each batch_size, for each method
            for batch_size, method_dict in batch_data.items():
                for method in methods:
                    algo_name = extract_method_name(method)
                    metric_values = method_dict.get(method, {}).get(metric, None)
                    if metric_values:
                        mean_val, std_val = metric_values  # Unpack (mean, std)
                        data.append([algo_name, batch_size, mean_val, std_val])
            
            # Create a DataFrame
            df = pd.DataFrame(data, columns=["Method", "Batch Size", "Mean", "Std"])
            
            # If df is empty (no data), skip this subplot
            if df.empty:
                print(f"No data found for domain={domain}, weight={weight_to_use}, metric={metric}")
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(domain_mapping[domain].replace('_', ' ').title().replace("S", "s"))
                continue
            
            # Format the annotations based on metric type
            if metric == "average_generated_nodes":
                df["Formatted"] = df.apply(lambda row: format_cell(row['Mean'], row['Std'], is_int=True, include_std=False), axis=1)
            elif metric == "percentage_solved":
                df["Formatted"] = df.apply(lambda row: format_cell(row['Mean'], row['Std'], is_int=False, precision=1, include_std=row['Mean'] < 100), axis=1)
            else:
                df["Formatted"] = df.apply(lambda row: format_cell(row['Mean'], row['Std'], is_int=False, precision=2), axis=1)
            
            # Pivot for heatmap
            df_pivot = df.pivot(index="Method", columns="Batch Size", values="Mean")
            
            # Custom sort key
            def custom_sort_key(val):
                if val.startswith("$LHBL("):
                    group = 0
                elif val.startswith("$LHBL_S("):
                    group = 1
                else:
                    group = 2  # put $SSBL$ last
                
                match = re.search(r'\((\d+)\)', val)
                number = int(match.group(1)) if match else float('inf')
                
                return (group, number)
            
            # Sort using the key
            df_pivot_sorted = df_pivot.sort_values(by="Method", key=lambda x: x.map(custom_sort_key))
            
            # Create formatted pivot for annotations
            formatted_pivot = df.pivot(index="Method", columns="Batch Size", values="Formatted")
            df_formatted_pivot_sorted = formatted_pivot.sort_values(by="Method", key=lambda x: x.map(custom_sort_key))
            
            # Create heatmap with consistent vmin/vmax across all domains
            hm = sns.heatmap(
                df_pivot_sorted,
                annot=df_formatted_pivot_sorted,  # Use our formatted strings
                fmt="",  # Avoid numerical formatting since we have strings
                cmap=cmap,
                linewidths=0.55,  # Increased linewidth for better separation
                cbar=False,  # No individual colorbars
                ax=ax,
                vmin=vmin,
                vmax=vmax
            )
            
            heatmaps.append(hm)
            
            # Set labels and title
            ax.set_xlabel("Batch Size")
            if idx == 0:  # Only show y-label on leftmost plot
                ax.set_ylabel(metric.replace('_', ' ').title())
            else:
                ax.set_ylabel("")
            
            ax.set_title(domain_mapping[domain].replace('_', ' ').title().replace("S", "s"))
        
        # Add a single colorbar for all domains if we have at least one heatmap
        if heatmaps:
            # Create space for the colorbar
            fig.subplots_adjust(right=0.9)
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
            fig.colorbar(heatmaps[0].get_children()[0], cax=cbar_ax)
        
        # plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make room for the colorbar
        
        # Save the figure in both PNG and PDF formats
        base_output_file = os.path.join(save_path, f"{metric}_combined")
        plt.savefig(f"{base_output_file}.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{base_output_file}.pdf", format='pdf', bbox_inches='tight')
        plt.close()
        
        print(f"Saved combined heatmap for '{metric}' as PNG and PDF")


def create_heatmap_weight_by_method(mean_dict, domain, output_dir, batch_size="100"):
    """
    Create a single figure with 2 side-by-side heatmaps showing average_generated_nodes and percentage_solved.
    X-axis: Weights (0_0, 0_2, 0_4, 0_8, 1_0)
    Y-axis: Methods (algorithms)

    Parameters:
        mean_dict (dict): Nested dictionary with weight -> batch_size -> method -> metrics structure
        domain (str): Domain name (for title)
        output_dir (str): Directory to save the plot
        batch_size (str): Fixed batch size to use (default: "100")
    """
    os.makedirs(output_dir, exist_ok=True)

    # Define the metrics and weights to use
    # metrics = ["average_generated_nodes", "average_solution_length"]
    metrics = ["average_generated_nodes", "percentage_solved"]
    weights = ["0_0", "0_2", "0_4", "0_6", "0_8", "1_0"]

    # Domain mapping for title
    domain_mapping = {
        'puzzle35': '35-Tile Puzzle',
        'lightsout7': 'Lightsout 7x7',
        'cube3': "Rubik's Cube"
    }

    # Set up the plot style
    plt.rcParams.update({'font.size': 18, 'axes.titlesize': 22, 'axes.labelsize': 18})

    # Create figure with 2 subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Process each metric
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        data = []

        # Gather data for each weight and method
        for weight in weights:
            if weight not in mean_dict:
                logging.warning(f"Weight {weight} not found in data")
                continue

            if batch_size not in mean_dict[weight]:
                logging.warning(f"Batch size {batch_size} not found for weight {weight}")
                continue

            method_dict = mean_dict[weight][batch_size]

            for method in method_dict.keys():
                algo_name = extract_method_name(method)
                metric_values = method_dict[method].get(metric, None)
                if metric_values:
                    mean_val, std_val = metric_values  # Unpack (mean, std)
                    data.append([algo_name, weight, mean_val, std_val])

        # Create a DataFrame
        df = pd.DataFrame(data, columns=["Method", "Weight", "Mean", "Std"])

        # If df is empty (no data), skip
        if df.empty:
            print(f"No data found for domain={domain}, batch_size={batch_size}, metric={metric}")
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            continue

        # Format the annotations based on metric type
        if metric == "average_generated_nodes":
            df["Formatted"] = df.apply(lambda row: format_cell(row['Mean'], row['Std'], is_int=True, include_std=False), axis=1)
        elif metric == "percentage_solved":
            df["Formatted"] = df.apply(lambda row: format_cell(row['Mean'], row['Std'], is_int=False, precision=1, include_std=row['Mean'] < 100), axis=1)
        else:
            df["Formatted"] = df.apply(lambda row: format_cell(row['Mean'], row['Std'], is_int=False, precision=2), axis=1)

        # Pivot so that columns=Weight, rows=Method, values=Mean
        df_pivot = df.pivot(index="Method", columns="Weight", values="Mean")

        # Custom sort key for methods
        def custom_sort_key(val):
            if val.startswith("$LHBL("):
                group = 0
            elif val.startswith("$LHBL_S("):
                group = 1
            else:
                group = 2  # put $SSBL$ last

            match = re.search(r'\((\d+)\)', val)
            number = int(match.group(1)) if match else float('inf')

            return (group, number)

        # Sort using the key
        df_pivot_sorted = df_pivot.sort_values(by="Method", key=lambda x: x.map(custom_sort_key))

        # Reorder columns to ensure weights are in order
        df_pivot_sorted = df_pivot_sorted[weights]

        # Create a "Formatted" column to show "mean ± std" as a string
        formatted_pivot = df.pivot(index="Method", columns="Weight", values="Formatted")
        df_formatted_pivot_sorted = formatted_pivot.sort_values(by="Method", key=lambda x: x.map(custom_sort_key))
        df_formatted_pivot_sorted = df_formatted_pivot_sorted[weights]

        # Convert weight labels from underscore format (0_8) to decimal format (0.8) for display
        weight_labels_display = [w.replace('_', '.') for w in weights]
        df_pivot_sorted.columns = weight_labels_display
        df_formatted_pivot_sorted.columns = weight_labels_display

        # Choose colormap based on metric "direction"
        if metric in ["percentage_solved"]:
            cmap = "RdYlGn"  # Green = good (high), Red = bad (low)
        else:
            cmap = "RdYlGn_r"  # Reversed: Green = good (low), Red = bad (high)

        # Create heatmap
        sns.heatmap(
            df_pivot_sorted,
            annot=df_formatted_pivot_sorted,  # Use our formatted strings
            fmt="",  # Avoid numerical formatting since we have strings
            cmap=cmap,
            linewidths=0.55,
            cbar=False,
            ax=ax
        )

        # Set labels
        ax.set_xlabel("Weight (λ)")
        if idx == 0:  # Only show y-label on leftmost plot
            ax.set_ylabel("Method")
        else:
            ax.set_ylabel("")

        # Set title based on metric
        if metric == "average_generated_nodes":
            ax.set_title("Avg. Generated Nodes")
        elif metric == "percentage_solved":
            ax.set_title("Problems Solved (%)")

    # Add overall figure title
    # fig.suptitle(f"{domain_mapping.get(domain, domain)} - Batch Size {batch_size}", fontsize=24, y=1.02)

    plt.tight_layout()

    # Save the figure
    output_file = os.path.join(output_dir, f"{domain}_weight_by_method_batch{batch_size}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved weight-by-method heatmap for domain '{domain}' at: {output_file}")


def create_multi_domain_multi_metric_heatmaps(all_domain_data, save_path="results/RLC/batch_heatmap_plots", weight_to_use=WEIGHT):
    """
    Create a single figure with multiple metrics in a grid layout.
    Each row represents a metric, and each column represents a domain.
    
    Parameters:
        all_domain_data (dict): Dictionary with domain -> mean_dict structure
        save_path (str): Path to save the figures
        weight_to_use (str): Which weight to use for plotting
    """
    os.makedirs(save_path, exist_ok=True)
    metrics = ["percentage_solved", "average_generated_nodes"]
    # metrics = ["average_time", "average_solution_length"]
    # domains = list(all_domain_data.keys())
    domains = ['puzzle35', 'lightsout7', 'cube3']
    domain_mapping = {
        'puzzle35': '35-Tile Puzzle',
        'lightsout7': 'Lightsout 7x7',
        'cube3': "Rubik's Cube"  # Changed from 'Rubik\'s Cube' to "Rubik's Cube"
    }
    
    # Set up the plot style
    plt.rcParams.update({'font.size': 18, 'axes.titlesize': 22, 'axes.labelsize': 18})
    
    # Create a single figure with a grid of subplots (2 rows for metrics, 3 columns for domains)
    fig, axes = plt.subplots(len(metrics), len(domains), figsize=(24, 12))
    
    # Store vmin and vmax for each metric to ensure consistent color scaling across domains
    vmin_vmax = {}
    
    # First pass: collect min and max values for each metric across all domains
    for metric in metrics:
        min_vals = []
        max_vals = []
        
        for domain in domains:
            if domain not in all_domain_data:
                continue
                
            mean_dict = all_domain_data[domain]
            
            if weight_to_use not in mean_dict:
                continue
                
            batch_data = mean_dict[weight_to_use]
            methods = list(next(iter(batch_data.values())).keys())
            
            # Gather data from each batch_size, for each method
            for batch_size, method_dict in batch_data.items():
                for method in methods:
                    metric_values = method_dict.get(method, {}).get(metric, None)
                    if metric_values:
                        mean_val, _ = metric_values  # Unpack (mean, std)
                        min_vals.append(mean_val)
                        max_vals.append(mean_val)
        
        if min_vals and max_vals:
            vmin_vmax[metric] = (min(min_vals), max(max_vals))
        else:
            vmin_vmax[metric] = (0, 1)  # Default if no data
    
    # Process each metric (row)
    for row_idx, metric in enumerate(metrics):
        # Create a list to store heatmap objects for this row
        heatmaps = []
        
        # Choose colormap based on metric "direction"
        if metric in ["percentage_solved"]:
            cmap = "RdYlGn"  # Green = good (high), Red = bad (low)
        else:
            cmap = "RdYlGn_r"  # Reversed: Green = good (low), Red = bad (high)
        
        # # Get vmin and vmax for consistent color scaling
        # vmin, vmax = vmin_vmax[metric]
        
        # Process each domain (column)
        for col_idx, domain in enumerate(domains):
            ax = axes[row_idx, col_idx]
            
            if domain not in all_domain_data:
                print(f"Domain {domain} not found in data")
                ax.text(0.5, 0.5, f'No Data for {domain}', ha='center', va='center', transform=ax.transAxes)
                continue
                
            mean_dict = all_domain_data[domain]
            
            if weight_to_use not in mean_dict:
                print(f"Weight {weight_to_use} not found for domain {domain}")
                ax.text(0.5, 0.5, f'No Data for weight {weight_to_use}', ha='center', va='center', transform=ax.transAxes)
                continue
                
            batch_data = mean_dict[weight_to_use]
            methods = list(next(iter(batch_data.values())).keys())
            
            data = []
            
            # Gather data from each batch_size, for each method
            for batch_size, method_dict in batch_data.items():
                for method in methods:
                    algo_name = extract_method_name(method)
                    metric_values = method_dict.get(method, {}).get(metric, None)
                    if metric_values:
                        mean_val, std_val = metric_values  # Unpack (mean, std)
                        data.append([algo_name, batch_size, mean_val, std_val])
            
            # Create a DataFrame
            df = pd.DataFrame(data, columns=["Method", "Batch Size", "Mean", "Std"])
            
            # If df is empty (no data), skip this subplot
            if df.empty:
                print(f"No data found for domain={domain}, weight={weight_to_use}, metric={metric}")
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Format the annotations based on metric type
            if metric == "average_generated_nodes":
                df["Formatted"] = df.apply(lambda row: format_cell(row['Mean'], row['Std'], is_int=True, include_std=False), axis=1)
            elif metric == "percentage_solved":
                df["Formatted"] = df.apply(lambda row: format_cell(row['Mean'], row['Std'], is_int=False, precision=1, include_std=row['Mean'] < 100), axis=1)
            else:
                df["Formatted"] = df.apply(lambda row: format_cell(row['Mean'], row['Std'], is_int=False, precision=2), axis=1)
            
            # Pivot for heatmap
            df_pivot = df.pivot(index="Method", columns="Batch Size", values="Mean")
            
            # Custom sort key
            def custom_sort_key(val):
                if val.startswith("$LHBL("):
                    group = 0
                elif val.startswith("$LHBL_S("):
                    group = 1
                else:
                    group = 2  # put $SSBL$ last
                
                match = re.search(r'\((\d+)\)', val)
                number = int(match.group(1)) if match else float('inf')
                
                return (group, number)
            
            # Sort using the key
            df_pivot_sorted = df_pivot.sort_values(by="Method", key=lambda x: x.map(custom_sort_key))
            
            # Create formatted pivot for annotations
            formatted_pivot = df.pivot(index="Method", columns="Batch Size", values="Formatted")
            df_formatted_pivot_sorted = formatted_pivot.sort_values(by="Method", key=lambda x: x.map(custom_sort_key))
            
            # Create heatmap with consistent vmin/vmax across the row
            hm = sns.heatmap(
                df_pivot_sorted,
                annot=df_formatted_pivot_sorted,  # Use our formatted strings
                fmt="",  # Avoid numerical formatting since we have strings
                cmap=cmap,
                linewidths=0.55,  # Increased linewidth for better separation
                cbar=False,  # No individual colorbars - we'll add one for the whole row
                ax=ax,
                # vmin=vmin,
                # vmax=vmax
            )
            
            heatmaps.append(hm)
            
            # Set labels and title
            if row_idx == len(metrics) - 1:  # Only show x-label on bottom row
                ax.set_xlabel("Batch Size")
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])  # Hide x-tick labels for non-bottom rows
            
            if col_idx == 0:  # Only show y-label on leftmost column
                if metric == "percentage_solved":
                    ax.set_ylabel("Problems Solved (%)")
                else:
                    ax.set_ylabel(metric.replace('_', ' ').title())
            else:
                ax.set_ylabel("")
            
            # Set subplot title: domain name for top row, metric name for leftmost column
            if row_idx == 0:
                ax.set_title(domain_mapping[domain].replace('_', ' ').title().replace("S", "s"))
        
        # Add a single colorbar for the entire row if we have at least one heatmap
        # if heatmaps:
            # Create space for the colorbar
            # fig.subplots_adjust(right=0.9)
            # cbar_ax = fig.add_axes([0.92, 0.11 + row_idx * 0.45, 0.02, 0.35])  # [left, bottom, width, height]
            # fig.colorbar(heatmaps[0].get_children()[0], cax=cbar_ax)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make room for the colorbar
    
    # Save the figure in both PNG and PDF formats
    base_output_file = os.path.join(save_path, "combined_metrics_heatmap")
    plt.savefig(f"{base_output_file}.png", dpi=300, bbox_inches='tight')
    # plt.savefig(f"{base_output_file}.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Saved combined metrics heatmap as PNG and PDF")


def generate_tables():
    # Define the root directory
    # domains = ['puzzle35', 'cube3', 'lightsout7']
    domains = ['cube3']
    test_name = 'RLC'

    # batch_sizes = ["1", "100", "1000", "10000"]
    weights = [WEIGHT]
    batch_sizes = ["100"]
    methods = ['bellman_{}_1', "bellman_{}_10", "bellman_{}_50", "bellman_{}_100",
               "tree_bellman_{}_10", "tree_bellman_{}_50", "tree_bellman_{}_100"]

    # Collect data for all domains first
    all_domain_data = {}

    for domain in domains:
        main_dir = f"results/{test_name}/{domain}/"

        # Compute the mean values
        mean_dict = compute_mean_average_generated_nodes(
            root_dir=main_dir,
            weights=weights,
            batch_sizes=batch_sizes,
            methods=methods
        )

        all_domain_data[domain] = mean_dict

    # Create multi-domain heatmaps (new functionality)
    save_dir = f"results/aaai/batch_heatmap_plots"

    # Create the new combined multi-metric, multi-domain heatmap
    create_multi_domain_multi_metric_heatmaps(all_domain_data, save_dir)

    # Also keep the original separate heatmaps
    create_multi_domain_heatmaps(all_domain_data, save_dir)


def generate_weight_plots():
    """
    Generate heatmaps with fixed batch size (100) and varying weights.
    Creates plots for each domain showing average_generated_nodes and percentage_solved.
    """
    # Define domains and test settings
    # domains = ['puzzle35', 'cube3', 'lightsout7']
    domains = ['puzzle35']
    test_name = 'RLC'

    # Fixed batch size, varying weights
    weights = ["0_0", "0_2", "0_4", "0_6", "0_8", "1_0"]
    batch_sizes = ["100"]
    methods = ['bellman_{}_1', "bellman_{}_10", "bellman_{}_50", "bellman_{}_100",
               "tree_bellman_{}_10", "tree_bellman_{}_50", "tree_bellman_{}_100"]

    save_dir = f"results/aaai/weight_heatmap_plots"

    for domain in domains:
        main_dir = f"results/{test_name}/{domain}/"

        # Compute the mean values across all weights
        mean_dict = compute_mean_average_generated_nodes(
            root_dir=main_dir,
            weights=weights,
            batch_sizes=batch_sizes,
            methods=methods
        )

        # Create weight-by-method heatmap for this domain
        create_heatmap_weight_by_method(mean_dict, domain, save_dir, batch_size="100")


if __name__ == "__main__":
    # generate_tables()
    # Uncomment the line below to generate weight variation plots
    generate_weight_plots()
