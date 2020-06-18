"""
select_instances.py

This file randomly chooses the appropriate number of files from each
group to be used in the training and validation sets.
"""

import numpy as np

def by_numbers(log_list,flags):
    """
    by_numbers modifies and returns the log of training
    or validation set files based on the group breakdown
    specified by instance_proportion.

    param: log_list
    param: instance_proportion
    return: new_log_list
    """

    group = list()
    for row in log_list:
        group.append(row[3])
    group = np.array(group)

    unique_group = set(group)
    unique_group = np.array(list(unique_group))

    n_group = np.zeros(len(unique_group))
    proportion = np.zeros(len(unique_group))
    for i, n in enumerate(unique_group):
        n_group[i] = np.sum(group == n)
        proportion[i] = flags['instance_proportion'][np.int(unique_group[i])]

    a = np.min( n_group*np.max(proportion) / proportion )

    n_instance = np.floor(a * proportion / np.max(proportion)).astype('int')

    # random selection
    use = np.zeros(len(log_list), dtype = np.int)
    for i in range(len(unique_group)):
        population_idx = [idx for idx in range(len(log_list)) if log_list[idx][3] == unique_group[i]]
        selected_idx = np.random.choice(population_idx, n_instance[i], replace = False)

        for j in selected_idx:
            use[j] = 1

    new_log_list = list()
    for i, row in enumerate(log_list):
        row = row[0:4]
        row.append(use[i])
        new_log_list.append(row)

    return new_log_list



