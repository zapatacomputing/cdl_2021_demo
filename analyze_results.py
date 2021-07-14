import numpy as np
import os
import glob
import re
from zquantum.core.serialization import load_optimization_results


def analyze_results(dir_name):
    result_files = glob.glob(os.path.join(dir_name, "opt_results_*"))

    graph_ids = [
        int(re.search(r"opt_results_graph_(\d+)", file_name).group(1))
        for file_name in result_files
    ]

    max_graph_id = max(graph_ids)
    layers_ids = [
        int(re.search(r"opt_results_graph_(\d+)_layers_(\d+)", file_name).group(2))
        for file_name in result_files
    ]
    min_layer = min(layers_ids)
    max_layer = max(layers_ids)

    for n_layers in range(min_layer, max_layer + 1):
        print("Number of layers", n_layers)
        optimal_params = []
        optimal_costs = []

        for graph_id in range(max_graph_id + 1):
            opt_results_files = glob.glob(
                os.path.join(
                    dir_name, f"opt_results_graph_{graph_id}_layers_{n_layers}_*"
                )
            )
            best_result = min(
                [
                    load_optimization_results(opt_result_file)
                    for opt_result_file in opt_results_files
                ],
                key=lambda result: result.opt_value,
            )
            optimal_params.append(best_result.opt_params)
            optimal_costs.append(best_result.opt_value)

        print("Mean optimal cost:", np.mean(optimal_costs))
        print("Cost std:", np.std(optimal_costs))


if __name__ == "__main__":
    analyze_results("2021_07_14_13_11_29")
