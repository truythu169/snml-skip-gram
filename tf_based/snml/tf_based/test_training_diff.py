from tf_based.snml.tf_based.model import Model
import utils.tools as utils
import numpy as np


if __name__ == "__main__":
    epochs = 16
    dim = '200'

    # read snml train file
    data = np.genfromtxt('../../../data/text8/scope.csv', delimiter=',').astype(int)

    loss_list = []
    n_sample = 2000
    model = Model('../../../output/text8/momentum/snml/1/' + dim + 'dim/',
                  '../../../data/text8/contexts/', n_context_sample=3000, learning_rate=0.0004)

    for i in range(n_sample):
        datum = data[i]
        w, c = data[i][0], data[i][1]
        w = int(w)
        c = int(c)

        ps_a = -np.log(model.train_one_sample(w, c, epochs=epochs, update_weight=True))
        loss_list.append(ps_a)

        if i % 100 == 0:
            print('{} th loop'.format(i))

    utils.save_pkl(loss_list, '../../../output/text8/momentum/test/4.pkl', local=True)
