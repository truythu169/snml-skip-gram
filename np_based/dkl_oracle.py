import numpy as np
import utils.tools as utils
from sklearn.metrics import mean_absolute_error
from np_based.snml_model import Model
from scipy import stats


def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def cos(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    return vec1.dot(vec2) / (norm1 * norm2)


def rho(vec1,vec2):
    return stats.stats.spearmanr(vec1, vec2)[0]


def evaludate_model(dim):
    kl_list = []
    mae_list = []
    cos_list = []
    rho_list = []
    parent_dir = '../notebooks/output/100-context-500000-data-38-questions/'

    model = Model(parent_dir + '1/full/{}dim/'.format(dim),
                  '../../data/text8/', learning_rate=0.1)

    context_distribution = utils.load_pkl(parent_dir + 'context_distribution.dict')

    for i in range(len(context_distribution)):
        true_dis = context_distribution[i]
        pred_dis = model.get_context_dis(i)

        kl_list.append(kl_divergence(pred_dis, true_dis))
        mae_list.append(mean_absolute_error(pred_dis, true_dis))
        cos_list.append(cos(pred_dis, true_dis))
        rho_list.append(rho(pred_dis, true_dis))

    return np.sum(rho_list)


if __name__ == "__main__":
    dim_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 130, 140, 150,
                160, 170, 180, 190, 200]
    n_data = 500000
    for dim in dim_list:
        # print('{}dim dkl: {}'.format(dim, evaludate_model(dim)))
        print(evaludate_model(dim))
