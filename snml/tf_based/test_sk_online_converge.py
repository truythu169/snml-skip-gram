from snml.tf_based.model import Model
from sklearn.metrics import mean_absolute_error
import utils.tools as utils
from matplotlib import pyplot as plt
import time
import numpy as np


if __name__ == "__main__":
    # data
    data = [[11469, 1659], [1153, 1465], [357, 92], [260, 77], [56, 216],
            [5, 60], [18, 4], [21, 274], [912, 1818], [33, 2543]]

    # full data
    model_full = Model('../../../output/convergence_test/35epochs/full/100dim/',
                       '../../../data/processed data/split/',
                       '../models/100dim/output/',
                       '../context_distribution.pkl',
                       n_train_sample=10000,
                       n_context_sample=600,
                       n_neg_sample=3000)

    # model_snml = Model('../../../output/convergence_test/35epochs/snml/100dim/',
    #                    '../../../data/processed data/split/',
    #                    '../models/100dim/output/',
    #                    '../context_distribution.pkl',
    #                    n_train_sample=10000,
    #                    n_context_sample=600,
    #                    n_neg_sample=3000)

    p_full = []
    p_snml = []
    for datum in data:
        # ps = model_snml.train(datum[0], datum[1], epochs=35, update_weight=True)
        pf = model_full.get_neg_prob(datum[0], datum[1])
        # print(pf, ps)
        p_full.append(pf)
        # p_snml.append(ps)

    # print('MAE: ', mean_absolute_error(p_snml, p_full))
    print(p_full)

    # for i in range(1000):
    #     p, losses = model.train_neg_adam(8229, 9023)
    #     p_sum.append(p)
    # end = time.time()
    # print("100 loop in {:.4f} sec".format(end - start))
    # plt.hist(p_sum, bins=20)
    # plt.show()

    # print('Mean: {} \nMin: {} \nMax: {} \nstd: {}'.format(np.mean(p_sum), min(p_sum), max(p_sum), np.std(p_sum)))
    #
    # model1 = Model('../../../output/convergence_test/20epochs/50dim/1/')
    # model2 = Model('../../../output/convergence_test/20epochs/50dim/2/')
    # model3 = Model('../../../output/convergence_test/20epochs/50dim/3/')
