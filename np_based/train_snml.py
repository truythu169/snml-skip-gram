from np_based.snml_model import Model
import time
import numpy as np
import argparse
import utils.tools as utils
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='../notebooks/output/100-context-500000-data-38-questions/1/300k/5dim/', type=str)
    parser.add_argument('--context_file', default='../notebooks/output/100-context-500000-data-38-questions/', type=str)
    parser.add_argument('--snml_train_file', default='../notebooks/output/100-context-500000-data-38-questions/scope.csv', type=str)
    parser.add_argument('--negative_sample', default=3, type=int)
    parser.add_argument('--scope', default=200000, type=int)
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--learning_rate', default=0.008, type=float)
    parser.add_argument('--continue_from', default=0, type=int)
    parser.add_argument('--continue_scope', default=0, type=int)
    args = parser.parse_args()

    # Set up parameters
    if args.continue_scope == 0:
        args.continue_scope = args.scope

    # read snml train file
    # utils.download_from_gcs(args.snml_train_file)
    data = np.genfromtxt(args.snml_train_file, delimiter=',').astype(int)

    # Initialize model
    model = Model(args.model, args.context_file, args.learning_rate, args.negative_sample)

    # Continue from previous
    previous_file = args.model + '{}-step/scope-{}-snml_length.pkl'.format(args.continue_from, args.continue_scope)
    if os.path.isfile(previous_file):
        snml_lengths = utils.load_pkl(previous_file)
    else:
        snml_lengths = []

    for i in range(args.continue_from):
        model.train(data[i][0], data[i][1], epochs=args.epochs)
    print('Continue step: {}, from file: {}'.format(args.continue_from, previous_file))

    # Run snml
    print_step = 50000
    start = time.time()
    for i in range(args.continue_from, args.scope):
        w = data[i][0]
        c = data[i][1]

        length = model.snml_length(w, c, epochs=args.epochs)
        snml_lengths.append(length)

        # print process
        if (i + 1) % print_step == 0:
            end = time.time()
            print('Run {} step in: {:.4f} sec, snml length: {}'.format(i + 1, (end - start), sum(snml_lengths)))
            start = time.time()

        # save steps
        # if (i + 1) % 500000 == 0:
        #     step_path = args.model + '{}-step/'.format(i + 1)
        #     filename = step_path + 'scope-{}-snml_length.pkl'.format(args.scope)
        #     utils.save_pkl(snml_lengths, filename)

    print('{} scope snml length: {}'.format(args.scope, sum(snml_lengths)))

    # Save result to file
    filename = args.model + 'scope-{}-snml_length.pkl'.format(args.scope)
    utils.save_pkl(snml_lengths, filename, local=True)

    # upload to gcs
    # utils.upload_to_gcs(filename, force_update=True)
