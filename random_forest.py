from sklearn.ensemble import RandomForestClassifier
import numpy as np
from learn import Learn

class RandomForest(Learn):
    def __init__(self, **kwargs):
        Learn.__init__(self, **kwargs)
        self.algo = RandomForestClassifier()

    def _train_routine(self, train_X, train_Y):
        return self.algo.fit(train_X, train_Y)

    def predict(self, test_data=[]):
        return self.algo.predict(test_data)

    def set_parameters(self, parameters):
        for key, value in parameters.iteritems():
            setattr(self.algo, key, value)
        Learn.set_parameters(self, parameters)
