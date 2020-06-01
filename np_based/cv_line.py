from np_based.snml_model import Model
from utils.tools import save_pkl
import argparse
import numpy as np


def get_loss_list(m, d):
    loss_list = []
    count = 0
    for datum in d:
        w, c = int(datum[0]), int(datum[1])
        prob = m.validation_loss(w, c)
        loss_list.append(prob)

        count += 1
        if count % 100000 == 0:
            print('Processing step: ', count)

    return loss_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='../notebooks/output/100-context-500000-data-38-questions/1/300k/{}dim/', type=str)
    parser.add_argument('--scope_file', default='../notebooks/output/100-context-500000-data-38-questions/scope.csv', type=str)
    parser.add_argument('--n_sample', default=200000, type=int)
    args = parser.parse_args()

    dims = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 130, 140, 150,
                160, 170, 180, 190, 200]
    data = np.genfromtxt(args.scope_file, delimiter=',').astype(int)
    sum_loss_list = []

    for dim in dims:
        print("Loading model dim: ", dim)
        # full data
        model = Model(args.model_path.format(dim), '../notebooks/output/100-context-500000-data-38-questions/', learning_rate=0.1)

        print("Compute loss...")
        loss_list = get_loss_list(model, data[:args.n_sample])
        print("Computed loss of {} data records.".format(len(loss_list)))
        # save_pkl(loss_list, '../../output/sgns/cv/text8/cv_{}_dim.pkl'.format(dim), local=True)
        sum_loss_list.append(sum(loss_list))

    for sum_loss in sum_loss_list:
        print(sum_loss)

