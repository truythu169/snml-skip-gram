from evaluation.wordsim import Wordsim
from evaluation.word_analogy import WordAnalogy
from utils.embedding import Embedding


if __name__ == "__main__":
    wordsim = Wordsim()
    word_analogy = WordAnalogy()
    word_analogy.set_top_words('../../data/text8/top_30000_words.txt')

    suffix = ''
    dimension_list = [50, 75, 100, 125, 150, 175, 200, 250, 300, 400, 500]
    wa_list = []
    ws_list = []

    for dimension in dimension_list:
        filename = '../../output/wiki/full/16epochs/1/{}dim/embedding-e={}-n_sampled=3000-epochs=16-batch_size=10000{}.txt'.format(dimension, dimension, suffix)
        print('Reading: ', filename)
        embedding = Embedding.from_file(filename)

        # wa_result = word_analogy.evaluate(embedding, high_level_category=False, restrict_top_words=False)
        ws_result = wordsim.evaluate(embedding)
        wordsim.pprint(ws_result)

        # wa_list.append(wa_result['all'])
        ws_list.append(ws_result['EN-WS-353-ALL'][2])

    print('Word analogy: ')
    for wa in wa_list:
        print(wa)
    print('Word sim: ')
    ws_list.reverse()
    for ws in ws_list:
        print(ws)
