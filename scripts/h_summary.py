import os
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
print(sys.path)

from typing import List
from environments.environment_abstract import Environment, State
import numpy as np

from argparse import ArgumentParser
from utils import env_utils, nnet_utils
import pickle
import os
import pandas as pd

points = [20,30,40,50,60,70,80,90,100]
scrambles = [1, 5, 7, 10, 13, 15, 17, 20, 30]  
seeds = [5,10,15]
models = ["bellman_{}_1", "bellman_{}_10", "bellman_{}_50", "bellman_{}_100", "tree_bellman_{}_10", "tree_bellman_{}_50", "tree_bellman_{}_100"]

models_names= {
    "bellman_{}_1": "$SSBL$",
    "bellman_{}_10": "$LHBL_S(10)$",
    "bellman_{}_50": "$LHBL_S(50)$",
    "bellman_{}_100": "$LHBL_S(100)$",
    "tree_bellman_{}_10": "$LHBL(10)$",
    "tree_bellman_{}_50": "$LHBL(50)$",
    "tree_bellman_{}_100": "$LHBL(100)$",
}


def plot_path_lens(args,h,n,error_metric):
    env: Environment = env_utils.get_environment(args.env)

    all_hioristics = []
    all_len = []
    all_std = []
    all_errors = []
    # error_metric = lambda real,estimation : estimation - real

    # error_metric = lambda real,estimation : real - estimation
    # error_metric = lambda real,estimation : abs(real - estimation)

    if args.data_mode == 'optimal':
        # Load optimal.pkl once for all evaluations
        dataPath = "data/cube3/expended_data/optimal.pkl"
        input_data = pickle.load(open(dataPath, "rb"))
        states: List[State] = input_data['states']
        lens = input_data['optimal_lens']
        device, _, on_gpu = nnet_utils.get_device()

        checkpoint_hioristics = []
        std_checkpoint_hioristics = []
        error_hioristics = []

        for checkpoint_file_n in points:
            seeds_h = []
            errors_h = []
            for seed in seeds:
                checkpoint_file = f"model_state_dict_{checkpoint_file_n}.pt"
                h_path = f"saved_models/RLC/cube3/{h.format(seed)}/history/"
                if not os.path.isfile(f"{h_path}/{checkpoint_file}"):
                    print(f"no file {h_path}/{checkpoint_file}")
                    break
                heuristic_fn = nnet_utils.load_heuristic_fn(h_path, device, on_gpu, env.get_nnet_model(),
                                                            env, clip_zero=True, batch_size=args.nnet_batch_size,
                                                            model_checkpoint=checkpoint_file)
                heuristics = heuristic_fn(states)
                e = error_metric(lens, heuristics)
                seeds_h.append(sum(heuristics) / len(heuristics))
                errors_h.append(sum(e) / len(e))

            avrage_seed_h = np.mean(seeds_h)
            std_seed_h = np.std(seeds_h)
            checkpoint_hioristics.append(avrage_seed_h)
            std_checkpoint_hioristics.append(std_seed_h)
            error_hioristics.append(np.mean(errors_h))

        # Store as single entry (treated as single "scramble" for plotting compatibility)
        all_hioristics.append(checkpoint_hioristics)
        all_len.append([(sum(lens) / len(lens))] * len(points))
        all_std.append(std_checkpoint_hioristics)
        all_errors.append(error_hioristics)

    else:  # scramble mode (original behavior)
        for scramble in scrambles:
            dataPath=f"data/cube3/expended_data/{scramble}_{scramble}/data_extended_33_bar.pkl"

            input_data = pickle.load(open(dataPath, "rb"))
            states: List[State] = input_data['states']
            lens = input_data['optimal_lens'] #lens  optimal_lens
            device, _, on_gpu = nnet_utils.get_device()
            scrumble_hioristics = []
            std_scrumble_hioristics = []
            error_hioristics = []
            for checkpoint_file_n in points:
                seeds_h = []
                errors_h = []
                for seed in seeds:

                    checkpoint_file = f"model_state_dict_{checkpoint_file_n}.pt"
                    h_path = f"saved_models/RLC/cube3/{h.format(seed)}/history/"
                    if not os.path.isfile(f"{h_path}/{checkpoint_file}"):
                        print(f"no file {h_path}/{checkpoint_file}")
                        break
                    heuristic_fn = nnet_utils.load_heuristic_fn(h_path, device, on_gpu, env.get_nnet_model(),
                                                                env, clip_zero=True, batch_size=args.nnet_batch_size,
                                                                model_checkpoint=checkpoint_file)
                    heuristics = heuristic_fn(states)
                    e = error_metric(lens,heuristics)
                    seeds_h.append(sum(heuristics) / len(heuristics))
                    errors_h.append(sum(e) / len(e))

                avrage_seed_h = np.mean(seeds_h)
                std_seed_h = np.std(seeds_h)
                scrumble_hioristics.append(avrage_seed_h)
                std_scrumble_hioristics.append(std_seed_h)
                error_hioristics.append(np.mean(errors_h))


            all_hioristics.append(scrumble_hioristics)
            all_len.append( [(sum(lens) / len(lens) )] * len(points))
            all_std.append(std_scrumble_hioristics)
            all_errors.append(error_hioristics)

    plt.figure()

    colors = ["green", "red", "blue", "gold", "black", "sienna", "tan", "indigo", "pink"]

    if args.data_mode == 'optimal':
        # Single line plot for optimal mode
        print(f"Optimal instances (mean={np.mean(lens):.2f})")
        print(f"{n} : {[round(all_hioristics[0][i],2) for i in range(len(all_hioristics[0]))]}")
        print(f" shortest : {[round(all_len[0][i],2) for i in range(len(all_len[0]))]}")
        print(f" error : {[round(all_errors[0][i],2) for i in range(len(all_errors[0]))]}")

        plt.plot(points, all_len[0], color='black', linestyle='dashed', label='Optimal Cost')
        plt.plot(points, all_hioristics[0], color='blue', label='Heuristic Estimate')
        plt.fill_between(points, np.array(all_hioristics[0]) - np.array(all_std[0]),
                        np.array(all_hioristics[0]) + np.array(all_std[0]), alpha=0.3, color='blue')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), ncol=2, shadow=True)
    else:
        # Multi-line plot for scramble mode
        for e, name in enumerate(scrambles):
            print(f"{name} scrambles")
            print(f"{n} : {[round(all_hioristics[e][i],2) for i in range(len(all_hioristics[e]))]}")
            print(f" shortest : {[round(all_len[e][i],2) for i in range(len(all_len[e]))]}")
            print(f" error : {[round(all_errors[e][i],2) for i in range(len(all_errors[e]))]}")

            c = colors[e]
            plt.plot(points, all_len[e], color=c, linestyle='dashed')
            plt.plot(points, all_hioristics[e], color=c, label=name)
            plt.fill_between(points, np.array(all_hioristics[e]) - np.array(all_std[e]),
                           np.array(all_hioristics[e]) + np.array(all_std[e]), alpha=0.3, color=c)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), ncol=3, shadow=True)

    plt.title(f'{models_names[n]}',loc='right')
    plt.xlabel('Training Progress (%)', fontsize=14)
    plt.ylabel('Cost-to-go', fontsize=14)
    plt.savefig(f"{args.results_dir}/{models_names[n]}")
    plt.savefig(f"{args.results_dir}/{models_names[n]}.pdf", format="pdf", bbox_inches="tight")

    plt.figure()

    if args.data_mode == 'optimal':
        # Single error line for optimal mode
        plt.plot(points, all_errors[0], color='blue', label='Error')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), ncol=1, shadow=True)
    else:
        # Multi-line error plot for scramble mode
        for e, name in enumerate(scrambles):
            c = colors[e]
            plt.plot(points, all_errors[e], color=c, label=name)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), ncol=3, shadow=True)

    # Add horizontal dashed line at y=0.0
    plt.axhline(y=0.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    plt.title(f'{models_names[n]}',loc='right')
    plt.xlabel('Training Progress (%)', fontsize=14)
    plt.ylabel('Cost-to-go', fontsize=14)
    plt.savefig(f"{args.results_dir}/{models_names[n]}_e")
    plt.savefig(f"{args.results_dir}/{models_names[n]}_e.pdf", format="pdf", bbox_inches="tight")

    return all_errors


if __name__ == "__main__":
      # parse arguments
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--env', type=str, required=True, help="Environment: cube3, 15-puzzle, 24-puzzle")
    parser.add_argument('--results_dir', type=str, required=True, help="Directory to save results")
    parser.add_argument('--metric', type=str, required=True, help="error_metric")
    parser.add_argument('--data_mode', type=str, default='scramble', choices=['scramble', 'optimal'],
                        help="Data mode: 'scramble' for per-scramble analysis, 'optimal' for optimal.pkl")

    parser.add_argument('--nnet_batch_size', type=int, default=None, help="Set to control how many states per GPU are "
                                                                          "evaluated by the neural network at a time. "
                                                                          "Does not affect final results, "
                                                                          "but will help if nnet is running out of "
                                                                          "memory.")


    args = parser.parse_args()

    metric_name = args.metric
    print(f"metric_name : {metric_name}")
    metric_dict = {
        "real_minus_est": {
            "func" : lambda real,estimation : (real - estimation),
            "y_title" : "(real - estimation)"
        },
        "est_minus_real": {
            "func" : lambda real,estimation : (estimation - real),
            "y_title" : "(estimation - real)"
        },
        "abs": {
            "func" : lambda real,estimation : abs(estimation - real),
            "y_title" : "abs(estimation - real)"
        },
        "mse": {
            "func" : lambda real,estimation : (estimation - real)**2,
            "y_title" : "(estimation - real)**2"
        },
        "msrele": {
            "func" :  lambda real,estimation : abs(estimation - real)/real,
            "y_title" : "abs(estimation - real)/real"
        },
    }
    error_metric = metric_dict[metric_name]["func"]

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    df_list = []

    for h in models:
        res = plot_path_lens(args, h, h, error_metric)
        print(res)
        if args.data_mode == 'optimal':
            # For optimal mode, only one "scramble" (the optimal dataset)
            for e2, p in enumerate(points):
                value = res[0][e2]
                df_list.append([models_names[h], 'optimal', p, value])
        else:
            # For scramble mode, iterate through all scrambles
            for e1, s in enumerate(scrambles):
                for e2, p in enumerate(points):
                    value = res[e1][e2]
                    df_list.append([models_names[h], s, p, value])

    print(f"{df_list=}")
    df = pd.DataFrame(df_list, columns=["Model", "scramble", "Step", "Value"])
    print(df)
    
    plt.figure()
    for _, name in enumerate(models):
            first_8 = df.loc[df['Model'] == models_names[name]]
            grp = first_8.groupby(by=["Step"])
            steps_cal_meam = []
            print(f"{name=}")
            for steo_num, group in grp:
                values = group['Value'].values
                print(f"{steo_num=} {values=}")
                print(f"avrage : {sum(values)/len(values)}")
                steps_cal_meam.append(sum(values)/len(values))
            print(f"{steps_cal_meam=}")
            plt.plot(points, steps_cal_meam, label=models_names[name])

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), ncol=3, fancybox=True, shadow=True, loc="center")
    plt.title(f'Heuristic accuracy throughout training', y=1.1)
    plt.xlabel('Training Progress (%)', fontsize=14)
    plt.ylabel(f'Cost-to-go Estimation Error', fontsize=14)
    plt.axhline(y=0.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    plt.savefig(f"{args.results_dir}/prosses_acc.png")
    plt.savefig(f"{args.results_dir}/prosses_acc.pdf", format="pdf", bbox_inches="tight")

    plt.figure()
    last_step_acc = []

    if args.data_mode == 'optimal':
        # For optimal mode, create a bar chart showing final error for each model
        for _, name in enumerate(models):
            final_value = df.loc[df['Model'] == models_names[name]].loc[df['Step'] == 100]['Value'].values[0]
            print(f"{name=} final value={final_value}")
            last_step_acc.append(final_value)

        plt.bar(range(len(models)), last_step_acc)
        plt.xticks(range(len(models)), [models_names[m] for m in models], rotation=45, ha='right')
        plt.title(f'Heuristic accuracy at the end of training (Optimal dataset)', y=1.1)
        plt.ylabel(f'Cost-to-go Estimation Error', fontsize=14)
        plt.tight_layout()
    else:
        # For scramble mode, plot error vs scrambles for each model
        for _, name in enumerate(models):
            first_8 = df.loc[df['Model'] == models_names[name]].loc[df['Step'] == 100]['Value'].values
            print(f"{name=} final {first_8=}")
            last_step_acc.append([first_8])
            plt.plot(scrambles, first_8, label=models_names[name])
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), ncol=3, fancybox=True, shadow=True, loc="center")
        plt.title(f'Heuristic accuracy at the end of the training', y=1.1)
        plt.xlabel('Scrambles', fontsize=14)
        plt.ylabel(f'Cost-to-go Estimation Error', fontsize=14)

    plt.axhline(y=0.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    plt.savefig(f"{args.results_dir}/100_acc.png")
    plt.savefig(f"{args.results_dir}/100.pdf", format="pdf", bbox_inches="tight")
