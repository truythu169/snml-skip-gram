from aic.model import Model
import utils.tools as utils
import numpy as np


class BICModel(Model):

    def __init__(self, model_path, data_path):
        super(BICModel, self).__init__(model_path, data_path)
        self.n_datums = utils.count_line(self.filename)

    def bic(self):
        return self.k * np.log(self.n_datums) - 2 * self.log_likelihood()
