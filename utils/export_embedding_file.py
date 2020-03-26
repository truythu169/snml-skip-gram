from utils.tools import load_pkl


def export_embedding(embedding, filename):
    # write embedding result to file
    output = open(filename, 'w', encoding='utf-8')
    for i in range(embedding.shape[0]):
        text = int_to_vocab[i]
        for j in embedding[i]:
            text += ' %f' % j
        text += '\n'
        try:
            output.write(text)
        except:
            print(text)

    output.close()


if __name__ == "__main__":
    dims = [50, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 300]
    path = '../../output/wiki/20200126/1/train1/{}dim/step-90/'

    int_to_vocab = load_pkl('../../data/wiki/dict/int_to_vocab.dict', local=True)

    for dim in dims:
        embedding = load_pkl(path.format(dim) + 'embedding.pkl', local=True)
        export_embedding(embedding, path.format(dim) + 'embedding.txt')
