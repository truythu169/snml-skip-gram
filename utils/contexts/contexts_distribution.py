import argparse
import numpy as np
from collections import Counter
import utils.tools as utils
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../../data/processed data/split/', type=str)
    args = parser.parse_args()

    print('Reading file...')
    contexts = []
    iteration = 0
    for file in os.listdir(args.data_path):
        iteration += 1
        if iteration % 1000 == 0:
            print('Importing ', file)
        data = np.genfromtxt(args.data_path + file, delimiter=',').astype(int)
        contexts.extend(data[:, 1])

    context_counts = Counter(contexts)
    n_context = len(context_counts)
    n_data = len(contexts)

    print('Making distribution...')
    context_distribution = np.zeros(n_context)
    for i in range(n_context):
        context_distribution[i] = context_counts[i] / n_data

    print('Saving file...')
    utils.save_pkl(context_distribution, 'context_distribution.pkl')

    print('Finished!')
    print('Saved: {} contexts / {} records'.format(n_context, n_data))


