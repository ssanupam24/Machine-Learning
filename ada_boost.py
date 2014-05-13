from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from learn import Learn

class AdaBoost(Learn):
    def __init__(self, **kwargs):
        Learn.__init__(self, **kwargs)
        self.algo = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 1))

    def _train_routine(self, train_X, train_Y):
        return self.algo.fit(train_X, train_Y)

    def predict(self, test_data=[]):
        return self.algo.predict(test_data)

    def set_parameters(self, parameters):
        for key, value in parameters.iteritems():
            setattr(self.algo, key, value)
        Learn.set_parameters(self, parameters)