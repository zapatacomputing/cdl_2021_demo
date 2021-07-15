from collections import defaultdict
from typing import DefaultDict
import tutorialworkflowresult
from pprint import pprint 
from openfermion import count_qubits
import statistics
import pandas as pd

def count_tasks(result):
    task_names = list(result.tasks.keys())

    task_counts = {}

    for name in task_names:
        if name[-1].isdigit():
            base_name = name.rsplit('-',1)[0]
        else:
            base_name = name
        if base_name in task_counts:
            task_counts[base_name] += 1
        else:
            task_counts[base_name] = 1
    return task_counts


def build_table(result):
    # First Sort out the tasks by mode from find-appropriate-params tasks
    reference_tasks = {}
    eval_tasks = []

    for name, task in result.tasks.items():
        if name.startswith("find-appropriate-params"):
            if task.inputs['mode'] in ["high", "low", "random"]:
                reference_tasks[task.inputs['mode']] = task
            elif task.inputs['mode'] == "evaluate":
                eval_tasks.append(task)

    buckets = defaultdict(list)

    for task in eval_tasks:
        layer_count = len(task.result.opt_params) / 2
        qubit_count = count_qubits(task.inputs["cost_hamiltonian"])
        task_mode = None
        # check the task inputs against outputs from our reference tasks
        for ref_mode, ref_task in reference_tasks.items():
            if (ref_task.result.opt_params == task.inputs['initial_params']).all():
                task_mode = ref_mode

        buckets[(qubit_count, layer_count, task_mode)].append(task.result.opt_value)

    processed_outputs = {}
    for condition, values in buckets.items():
        processed_outputs[condition + ("mean",)] = statistics.mean(values)
        if len(values) > 1:
            processed_outputs[condition + ("stdev",)] = statistics.stdev(values)
        else:
            processed_outputs[condition + ("stdev",)] =  None
        processed_outputs[condition + ("samples",)] = len(values)
        

    return pd.Series(processed_outputs)


if __name__ == "__main__":
    result = tutorialworkflowresult.load_cached_workflowresult("results/qaoa-opt-6a7c59a1-6f4c-42c0-95a4-e333a8d0b685_workflow_result.json")
    pprint(count_tasks(result))
    df = build_table(result)
    # A little pandas magic to match the table format of the paper
    df = df.reorder_levels([1,3,2,0])  # set the index order to match paper
    print(df.unstack(fill_value=0).unstack(fill_value=0).unstack(fill_value=0))  # do some unstacking to move index "above" our table