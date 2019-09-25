from tf_based.skip_gram import SkipGram
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='../../data/processed data/', type=str)
    parser.add_argument('--output_path', default='../output/', type=str)
    parser.add_argument('--n_embedding', default=50, type=int)
    parser.add_argument('--n_sampled', default=6000, type=int)
    parser.add_argument('--epochs', default=35, type=int)
    parser.add_argument('--batch_size', default=10000, type=int)
    parser.add_argument('--print_step', default=1000, type=int)
    args = parser.parse_args()

    skip_gram = SkipGram(args.input_path, args.output_path, n_embedding=args.n_embedding)
    skip_gram.train(n_sampled=args.n_sampled, epochs=args.epochs,
                    batch_size=args.batch_size, print_step=args.print_step)
    skip_gram.export_embedding()
    skip_gram.export_model()
