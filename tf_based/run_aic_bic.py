from tf_based.bic.model import BICModel
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='output/text8-oSG/5dim/', type=str)
    parser.add_argument('--data_file', default='data/text8/data.csv', type=str)
    args = parser.parse_args()

    dims = [5]
    aic_list = []
    bic_list = []

    for dim in dims:
        print("Loading model dim: ", dim)
        model = BICModel(args.model_path.format(dim), args.data_file)
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
