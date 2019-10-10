from tf_based.skip_gram import SkipGram
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='../../data/wiki/', type=str)
    parser.add_argument('--output_path', default='../../output/wiki/full/', type=str)
    parser.add_argument('--n_embedding', default=50, type=int)
    parser.add_argument('--n_sampled', default=3000, type=int)
    parser.add_argument('--epochs', default=16, type=int)
    parser.add_argument('--batch_size', default=10000, type=int)
    parser.add_argument('--print_step', default=1000, type=int)
    parser.add_argument('--snml_dir', default='../../output/wiki/snml/', type=str)
    args = parser.parse_args()
    snml = args.snml_dir != ''

    skip_gram = SkipGram(args.input_path, args.output_path, n_embedding=args.n_embedding,
                         n_sampled=args.n_sampled, epochs=args.epochs, batch_size=args.batch_size,
                         snml=snml, snml_dir=args.snml_dir)
    skip_gram.train(print_step=args.print_step)
    skip_gram.export_embedding()
    skip_gram.export_model()
