from np_based.bic.model import BICModel
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='../../../output/sgns/text8/2/full/{}dim/', type=str)
    parser.add_argument('--data_file', default='../../../data/text8/shufle/1/data.csv', type=str)
    parser.add_argument('--context_path', default='../../../data/text8/', type=str)
    parser.add_argument('--n_negative_sample', default=15, type=int)
    args = parser.parse_args()

    dims = [40, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 130, 140, 150, 160, 170, 200]
    aic_list = []
    bic_list = []

    for dim in dims:
        print("Loading model dim: ", dim)
        model = BICModel(args.model_path.format(dim), args.data_file, args.context_path, args.n_negative_sample)
        aic_list.append(model.aic())
        bic_list.append(model.bic())

    print('AIC: ')
    for aic in aic_list:
        print(aic)

    print('BIC: ')
    for bic in bic_list:
        print(bic)
