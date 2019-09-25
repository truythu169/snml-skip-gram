from snml.tf_based.model import Model
import time
import numpy as np
import argparse


def test_train(model, word, context):
    p_sum = []
    start = time.time()
    for i in range(100):
        p = model.train(word, context, epochs=20, update_weight=False)
        p_sum.append(p)
    end = time.time()
    print("100 loop in {:.4f} sec".format(end - start))
    print('Mean: {} \nMin: {} \nMax: {} \nstd: {}'.format(np.mean(p_sum), min(p_sum), max(p_sum), np.std(p_sum)))

    p_sum = []
    start = time.time()
    for i in range(100):
        p = model.train(word, context, epochs=20, update_weight=False, train_one=True)
        p_sum.append(p)
    end = time.time()
    print("100 loop in {:.4f} sec".format(end - start))
    print('Mean: {} \nMin: {} \nMax: {} \nstd: {}'.format(np.mean(p_sum), min(p_sum), max(p_sum), np.std(p_sum)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='../models/100dim/', type=str)
    parser.add_argument('--sample_path', default='../../../data/processed data/split/', type=str)
    parser.add_argument('--snml_train_file', default='../../../data/processed data/scope.csv', type=str)
    parser.add_argument('--output_path', default='models/100dim/output/', type=str)
    parser.add_argument('--context_distribution_file', default='../context_distribution.pkl', type=str)
    args = parser.parse_args()

    # read snml train file
    data = np.genfromtxt(args.snml_train_file, delimiter=',').astype(int)

    model = Model(args.model_path, args.sample_path, args.output_path, args.context_distribution_file,
                  n_train_sample=10000, n_context_sample=400)
    # snml_length = model.snml_length(data[10][0], data[10][1], epochs=10)
    # print(snml_length)
    # snml_length = model.snml_length(data[10][0], data[10][1], epochs=10)
    # print(snml_length)
    # snml_length = model.snml_length(data[10][0], data[10][1], epochs=10)
    # print(snml_length)

    # 50 dim
    print('50 dim: ')
    test_train(model, 8229, 9023)

    # 100 dim
    # print('100 dim: ')
    # model.change_model('models/100dim/')
    # p = model.train(data[10][0], data[10][1], epochs=10, update_weigh=False, train_one=True)
    # print(p)
    # p = model.train(data[10][0], data[10][1], epochs=10, update_weigh=False, train_one=False)
    # print(p)
    #
    # # 150 dim
    # print('150 dim: ')
    # model.change_model('models/150dim/')
    # p = model.train(data[10][0], data[10][1], epochs=10, update_weigh=False, train_one=True)
    # print(p)
    # p = model.train(data[10][0], data[10][1], epochs=10, update_weigh=False, train_one=False)
    # print(p)
    #
    # # 200 dim
    # print('200 dim: ')
    # model.change_model('models/200dim/')
    # p = model.train(data[10][0], data[10][1], epochs=10, update_weigh=False, train_one=True)
    # print(p)
    # p = model.train(data[10][0], data[10][1], epochs=10, update_weigh=False, train_one=False)
    # print(p)
