from np_based.snml_model import Model
from utils.tools import save_pkl
import numpy as np


def get_loss_list(m, d):
    loss_list = []
    for datum in d:
        w, c = int(datum[0]), int(datum[1])
        prob = m.get_prob(w, c)
        loss_list.append(prob)

    return - np.log(loss_list)


def get_loss_list_batch(m, d, batch_size=1000):
    loss_list = []
    d = d.reshape(d.shape[0] // batch_size, -1, d.shape[1])

    for batch in d:
        w = batch[:, 0]
        c = batch[:, 1]
        loss = np.log(m.get_prob(w, c))
        print(loss[0][0])
        loss_list.extend(loss)

    return loss_list


if __name__ == "__main__":
    dims = [50, 60, 70, 80, 90, 100, 110, 120]
    n_sample = 100000
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
