from tf_based.bic import BICModel
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='../notebooks/output/50-context-500000-data-18-questions/485000/model/5dim/', type=str)
    parser.add_argument('--data', default='../notebooks/output/50-context-500000-data-18-questions/485000/scope_full.csv', type=str)
    args = parser.parse_args()

    model = BICModel(args.model, args.data)
    result = - model.log_likelihood()
    print('Minus log likelihood: ', result)
