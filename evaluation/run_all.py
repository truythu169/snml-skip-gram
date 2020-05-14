from evaluation.wordsim import Wordsim
from evaluation.word_analogy import WordAnalogy
from utils.embedding import Embedding


if __name__ == "__main__":
    wordsim = Wordsim()
    word_analogy = WordAnalogy()
    word_analogy.set_top_words('../../data/text8/top_30000_words.txt')

    suffix = ''
    dimension_list = [40, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 130, 140, 150, 160, 170, 200]
    wa_list = []
    ws_list = []
    men_tr = []
    mturk771 = []

    for dimension in dimension_list:
        filename = '../../output/sgns/text8_ng4/2/full/{}dim/embedding-e={}-n_sampled=15-epochs=15-batch_size=1.txt'.format(dimension, dimension)
        print('Reading: ', filename)
        embedding = Embedding.from_file(filename)

        # wa_result = word_analogy.evaluate(embedding, high_level_category=False, restrict_top_words=False)
        ws_result = wordsim.evaluate(embedding)
        wordsim.pprint(ws_result)

        # wa_list.append(wa_result['all'])
        ws_list.append(ws_result['EN-WS-353-ALL'][2])
        men_tr.append(ws_result['EN-MEN-TR-3k'][2])
        mturk771.append(ws_result['EN-MTurk-771'][2])

    print('Word sim: ')
    for ws in ws_list:
        print(ws)

    print('EN-MEN-TR-3k')
    for ws in men_tr:
        print(ws)

    print('EN-MTurk-771')
    for ws in mturk771:
        print(ws)
