import argparse
import numpy as np
from collections import Counter
import utils.tools as utils
from utils.settings import config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../../data/wiki/', type=str)
    args = parser.parse_args()

    print('Reading file...')
    contexts = []

    with open(args.data_path + config['TRAIN']['train_data']) as fp:
        line = fp.readline()
        iteration = 0
        for line in fp:
            iteration += 1
            try:
                context = int(line.split(',')[1])
                contexts.append(context)
            except:
                print('Failed {}th line: {}'.format(iteration, line))
            finally:
                if iteration % 10000000 == 0:
                    print('Processed: {} lines'.format(iteration))

    context_counts = Counter(contexts)
    n_context = len(context_counts)
    n_data = len(contexts)

    print('Making distribution...')
    context_distribution = np.zeros(n_context)
    for i in range(n_context):
        context_distribution[i] = context_counts[i] / n_data

    print('Saving file...')
    utils.save_pkl(context_distribution, args.data_path + 'contexts/context_distribution.pkl')

    print('Finished!')
    print('Saved: {} contexts / {} records'.format(n_context, n_data))


