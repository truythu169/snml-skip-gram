from np_based.bic.model import BICModel
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='../notebooks/output/100-context-500000-data-38-questions/1/full/{}dim/', type=str)
    parser.add_argument('--data_file', default='../notebooks/output/100-context-500000-data-38-questions/data.csv', type=str)
    parser.add_argument('--context_path', default='../notebooks/output/100-context-500000-data-38-questions/', type=str)
    parser.add_argument('--n_negative_sample', default=3, type=int)
    args = parser.parse_args()

    dims = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 130, 140, 150,
                160, 170, 180, 190, 200]
    aic_list = []
    bic_list = []

    for dim in dims:
        print("Loading model dim: ", dim)
        model = BICModel(args.model_path.format(dim), args.data_file, args.context_path, args.n_negative_sample)
        aic = model.aic()
        bic = model.bic()
        print('aic: {}, bic: {}'.format(aic, bic))
        aic_list.append(aic)
        bic_list.append(bic)

    print('AIC: ')
    for aic in aic_list:
        print(aic)

    print('BIC: ')
    for bic in bic_list:
        print(bic)
