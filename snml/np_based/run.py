from snml.np_based.model import Model
import time
import numpy as np


if __name__ == "__main__":
    model = Model('../models/100dim/',
                  '../context_distribution.pkl', n_context_sample=600)

    p_sum = []
    start = time.time()
    for i in range(100):
        p, losses = model.train_neg_adam(8229, 9023)
        p_sum.append(p)
    end = time.time()
    print("100 loop in {:.4f} sec".format(end - start))
    print('Mean: {} \nMin: {} \nMax: {} \nstd: {}'.format(np.mean(p_sum), min(p_sum), max(p_sum), np.std(p_sum)))

