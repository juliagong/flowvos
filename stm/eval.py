import os
import sys
from time import time
import argparse

import numpy as np
import pandas as pd
from davis17eval.davis2017.evaluation import DAVISEvaluation


def eval_results(results_path, davis_path, dataset_split='val', task='semi-supervised'):
    csv_name_global = f'global_results-{dataset_split}.csv'
    csv_name_per_sequence = f'per-sequence_results-{dataset_split}.csv'
    csv_name_global_path = os.path.join(results_path, csv_name_global)
    csv_name_per_sequence_path = os.path.join(results_path, csv_name_per_sequence)

    time_start = time()

    # Create dataset and evaluate
    dataset_eval = DAVISEvaluation(davis_root=davis_path, task=task, gt_set=dataset_split)
    metrics_res = dataset_eval.evaluate(results_path)
    J, F = metrics_res['J'], metrics_res['F']

    # Generate dataframe for the general results
    g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']
    final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
    g_res = np.array([final_mean, np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(F["M"]), np.mean(F["R"]),
                      np.mean(F["D"])])
    g_res = np.reshape(g_res, [1, len(g_res)])
    table_g = pd.DataFrame(data=g_res, columns=g_measures)
    with open(csv_name_global_path, 'w') as f:
        table_g.to_csv(f, index=False, float_format="%.3f")

    # Generate a dataframe for the per sequence results
    seq_names = list(J['M_per_object'].keys())
    seq_measures = ['Sequence', 'J-Mean', 'F-Mean']
    J_per_object = [J['M_per_object'][x] for x in seq_names]
    F_per_object = [F['M_per_object'][x] for x in seq_names]
    table_seq = pd.DataFrame(data=list(zip(seq_names, J_per_object, F_per_object)), columns=seq_measures)
    with open(csv_name_per_sequence_path, 'w') as f:
        table_seq.to_csv(f, index=False, float_format="%.3f")

    total_time = time() - time_start
    return final_mean, total_time
