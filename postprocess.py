from collections import defaultdict
from typing import DefaultDict
import tutorialworkflowresult
from pprint import pprint
from openfermion import count_qubits
import statistics
import pandas as pd
import numpy as np


def count_tasks(result):
    task_names = list(result.tasks.keys())

    task_counts = {}

    for name in task_names:
        if name[-1].isdigit():
            base_name = name.rsplit("-", 1)[0]
        else:
            base_name = name
        if base_name in task_counts:
            task_counts[base_name] += 1
        else:
            task_counts[base_name] = 1
    return task_counts


def build_table(result):
    # First Sort out the tasks by mode from find-appropriate-params tasks
    reference_tasks = defaultdict(dict)
    eval_tasks = []

    for name, task in result.tasks.items():
        if name.startswith("find-appropriate-params"):
            n_layers = len(task.inputs["initial_params"]) / 2
            reference_tasks[n_layers][task.inputs["mode"]] = task

        if name.startswith("evaluate-params"):
            eval_tasks.append(task)

    buckets = defaultdict(list)

    for task in eval_tasks:
        layer_count = task.inputs["number_of_layers"]
        qubit_count = task.inputs["graph_size"]
        task_mode = None
        # check the task inputs against outputs from our reference tasks
        task_params = task.inputs["selected_results"]["opt_params"]

        for ref_mode, ref_task in reference_tasks[layer_count].items():
            ref_params = ref_task.result.opt_params
            if len(task_params) == len(ref_params) and np.allclose(
                task_params,
                ref_params,
            ):
                task_mode = ref_mode
        buckets[(qubit_count, layer_count, task_mode)] = task.result

    processed_outputs = {}
    for condition, values in buckets.items():
        processed_outputs[condition + ("mean",)] = statistics.mean(values)
        if len(values) > 1:
            processed_outputs[condition + ("stdev",)] = statistics.stdev(values)
        else:
            processed_outputs[condition + ("stdev",)] = None
        processed_outputs[condition + ("samples",)] = len(values)

    return pd.Series(processed_outputs)


if __name__ == "__main__":
    result = tutorialworkflowresult.load_cached_workflowresult(
        "results/qaoa-concentration-ccb37456-4211-43a9-949f-8f0de428e68d_workflow_result.json"
    )
    pprint(count_tasks(result))
    df = build_table(result)
    # A little pandas magic to match the table format of the paper
    df = df.reorder_levels([1, 3, 2, 0])  # set the index order to match paper
    print(
        df.unstack(fill_value=0).unstack(fill_value=0).unstack(fill_value=0)
    )  # do some unstacking to move index "above" our table
    import pdb

    pdb.set_trace()
