from np_based.snml_model import Model
from utils.tools import save_pkl
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
    dims = [50, 90, 100, 110, 120]
    n_sample = 2692279
    # read snml train file
    data = np.genfromtxt('../../data/text8/shufle/1/scope.csv', delimiter=',').astype(int)
    print(len(data))

    for dim in dims:
        print("Loading model dim: ", dim)
        # full data
        model = Model('../../output/sgns/text8/2/train1/{}dim/'.format(dim),
                      '../../data/text8/', learning_rate=0.1)

        print("Compute loss...")
        loss_list = get_loss_list(model, data[:n_sample])
        print("Computed loss of {} data records.".format(len(loss_list)))
        save_pkl(loss_list, 'C:\\Users/hungp/Downloads/information criteria on sg/text8 ns/snml/cv_{}_dim.pkl'.format(dim), local=True)
