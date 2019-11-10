from snml.tf_based.model import Model
import time
import numpy as np
import argparse
import utils.tools as utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='../../../output/text8/momentum/snml/1/150dim/', type=str)
    parser.add_argument('--context_path', default='../../../data/text8/contexts/', type=str)
    parser.add_argument('--snml_train_file', default='../../../data/text8/scope.csv', type=str)
    parser.add_argument('--scope', default=5, type=int)
    parser.add_argument('--epochs', default=4, type=int)
    args = parser.parse_args()

    # read snml train file
    utils.download_from_gcs(args.snml_train_file)
    data = np.genfromtxt(args.snml_train_file, delimiter=',').astype(int)

    # Run snml
    model = Model(args.model, args.context_path, n_neg_sample=3000, n_context_sample=3000, learning_rate=0.0004)
    snml_lengths = []
    print_step = 5
    start = time.time()
    for i in range(args.scope):
        w = data[i][0]
        c = data[i][1]

        length = model.snml_length_sampling(w, c, epochs=args.epochs)
        snml_lengths.append(length)

        # print process
        if (i + 1) % print_step == 0:
            end = time.time()
            print('Run {} step in: {:.4f} sec, snml length: {}'.format(i + 1, (end - start), sum(snml_lengths)))
            start = time.time()

        # save steps
        if (i + 1) % 100 == 0:
            step_path = args.model + '{}-step/'.format(i + 1)
            filename = step_path + 'scope-{}-snml_length.pkl'.format(args.scope)
            utils.save_pkl(snml_lengths, filename)

    print('{} scope snml length: {}'.format(args.scope, sum(snml_lengths)))

    # Save result to file
    filename = args.model + 'scope-{}-snml_length.txt'.format(args.scope)
    output = open(filename, 'w')
    for i in snml_lengths:
        output.write(str(i) + '\n')
    output.close()

    # upload to gcs
    utils.upload_to_gcs(filename, force_update=True)
