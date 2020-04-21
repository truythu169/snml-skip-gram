from evaluation.wordsim import Wordsim
from evaluation.word_analogy import WordAnalogy
from utils.embedding import Embedding


if __name__ == "__main__":
    wordsim = Wordsim()
    word_analogy = WordAnalogy()
    word_analogy.set_top_words('../../data/wiki/top_30000_words.txt')

    suffix = ''
    dimension_list = [50, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 300]
    epochs = [90]
    wa_list = []
    ws_list = []

    for dimension in dimension_list:
        filename = '../../output/wiki/20200126/1/train1/{}dim/step-90/embedding.txt'.format(dimension)
        print('Reading: ', filename)
        embedding = Embedding.from_file(filename)

        # wa_result = word_analogy.evaluate(embedding, high_level_category=False, restrict_top_words=False)
        ws_result = wordsim.evaluate(embedding)
        wordsim.pprint(ws_result)

        # wa_list.append(wa_result['all'])
        # ws_list.append(ws_result['EN-WS-353-ALL'][2])

    # print('Word analogy: ')
    # for wa in wa_list:
    #     print(wa)
    print('Word sim: ')
    ws_list.reverse()
    for ws in ws_list:
        print(ws)
