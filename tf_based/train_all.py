from tf_based.skip_gram import SkipGram

if __name__ == "__main__":
    input_path = '../../data/text8/'
    output_path = '../../output/text8/20200117/snml/1/train1/'

    dimension_list = [100, 50]

    for dimension in dimension_list:
        skip_gram = SkipGram(input_path, output_path, n_embedding=dimension,
                             n_sampled=3000, epochs=150, batch_size=1000,
                             snml=False, snml_dir='')
        skip_gram.train(print_step=0, stop_threshold=0.000001)
        skip_gram.export_embedding()
        skip_gram.export_model()
