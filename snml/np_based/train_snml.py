from snml.np_based.model import Model
import time
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='../models/31epochs/50dim/', type=str)
    parser.add_argument('--snml_train_file', default='../../../data/processed data/scope.csv', type=str)
    parser.add_argument('--scope', default=100, type=int)
    args = parser.parse_args()

    # read snml train file
    data = np.genfromtxt(args.snml_train_file, delimiter=',').astype(int)

    # Run snml
    model = Model(args.model)
    snml_lengths = []
    print_step = 5
    start = time.time()
    for i in range(args.scope):
        w = data[i][0]
        c = data[i][1]

        length, probs = model.snml_length_sampling_multiprocess(w, c, epochs=31, neg_size=3000, n_context_sample=600)
        snml_lengths.append(length)

        # print process
        if (i + 1) % print_step == 0:
            end = time.time()
            print('Run {} step in: {:.4f} sec'.format(i + 1, (end - start)))
            start = time.time()

    print('{} scope snml length: {}'.format(args.scope, sum(snml_lengths)))

    # Save result to file
    output = open(args.model + 'scope-{}-snml_length.txt'.format(args.scope), 'w')
    for i in snml_lengths:
        output.write(str(i) + '\n')
    output.close()
