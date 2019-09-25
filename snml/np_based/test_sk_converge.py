from snml.np_based.model import Model
import utils.tools as utils
from matplotlib import pyplot as plt
import time
import numpy as np


if __name__ == "__main__":
    # model = Model('../models/150dim/')
    # p_sum = []
    # start = time.time()
    # for i in range(100):
    #     model.reset()
    #     p, losses = model.train_neg_adam(8229, 9023)
    #     p_sum.append(p)
    # end = time.time()
    # print("100 loop in {:.4f} sec".format(end - start))
    # print('Mean: {} \nMin: {} \nMax: {} \nstd: {}'.format(np.mean(p_sum), min(p_sum), max(p_sum), np.std(p_sum)))

    model1 = Model('../../../output/convergence_test/20epochs/50dim/1/')
    model2 = Model('../../../output/convergence_test/20epochs/50dim/2/')
    model3 = Model('../../../output/convergence_test/20epochs/50dim/3/')

    words = []
    contexts = []

    for i in range(10):
        ws, cs = utils.sample_learning_data('../../../data/processed data/split/', 12802, 100)
        words.extend(ws)
        contexts.extend(cs)

    result = []
    for i in range(len(words)):
        w = words[i]
        c = contexts[i]

        p1 = model1.get_prob(w, c)
        p2 = model2.get_prob(w, c)
        p3 = model3.get_prob(w, c)

        std = np.std([p1, p2, p3])
        mean_std_percent = std / np.mean([p1, p2, p3]) * 100

        if (mean_std_percent > 100.):
            print(w, c)

            print(p1, p2, p3)

        result.append(mean_std_percent)

    print(np.mean(result))
    print(min(result))
    print(max(result))
    plt.hist(result, bins=100)
    plt.show()
