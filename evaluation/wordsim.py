import os
from collections import defaultdict
import utils.math as umath
from utils.embedding import Embedding


class Wordsim:

    def __init__(self, data_path='datasets/wordsim'):
        full_path = os.path.dirname(os.path.abspath(__file__)) + '\\' + data_path
        self.files = [file_name.replace(".txt", "") for file_name in os.listdir(full_path) if ".txt" in file_name]
        self.dataset = defaultdict(list)
        for file_name in self.files:
            for line in open(data_path + "/" + file_name + ".txt"):
                self.dataset[file_name].append([float(w) if i == 2 else w for i, w in enumerate(line.strip().split())])

    @staticmethod
    def pprint(result):
        from prettytable import PrettyTable
        x = PrettyTable(["Dataset", "Found", "Not Found", "Score (rho)"])
        x.align["Dataset"] = "l"
        for k, v in result.items():
            x.add_row([k,v[0],v[1],v[2]])
        print(x)

    def evaluate(self, embedding):
        result = {}
        for file_name, data in self.dataset.items():
            pred, label, found, notfound = [] ,[], 0, 0
            for datum in data:
                if embedding.in_vocab(datum[0]) and embedding.in_vocab(datum[1]):
                    found += 1
                    pred.append(umath.cos(embedding.vector(datum[0]), embedding.vector(datum[1])))
                    label.append(datum[2])
                else:
                    notfound += 1
            result[file_name] = (found, notfound, umath.rho(label,pred)*100)
        return result


if __name__ == "__main__":
    wordsim = Wordsim()
    embedding = Embedding.from_file('../../output/100dim/embedding-e=100-n_sampled=200-epochs=10-batch_size=10000.txt')
    result = wordsim.evaluate(embedding)
    # wordsim.pprint(result)
    print(result['EN-WS-353-ALL'][2])
