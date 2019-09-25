from snml.np_based.model import Model
import utils.tools as utils
from matplotlib import pyplot as plt
import time
import numpy as np
from matplotlib import pyplot as plt


if __name__ == "__main__":
    # get samples from training data
    words = []
    contexts = []
    loop_no = 100
    for i in range(loop_no):
        ws, ctx = utils.sample_learning_data('../../../data/processed data/split/', 12802, 100)
        words.extend(ws)
        contexts.extend(ctx)

    model1 = Model('../models/31epochs/50dim/', '../context_distribution.pkl')
    model2 = Model('../models/31epochs/100dim/', '../context_distribution.pkl')
    model3 = Model('../models/31epochs/150dim/', '../context_distribution.pkl')
    model4 = Model('../models/31epochs/200dim/', '../context_distribution.pkl')
    models = [model1, model2, model3, model4]

    losses = [[], [], [], []]
    for i in range(len(words)):
        w = words[i]
        c = contexts[i]

        for j in range(4):
            neg_log = -np.log(models[j].train(w, c, epochs=31, neg_size=3000, update_weights=True))
            losses[j].append(neg_log)

    print(np.sum(losses[0]))
    print(np.sum(losses[1]))
    print(np.sum(losses[2]))
    print(np.sum(losses[3]))
