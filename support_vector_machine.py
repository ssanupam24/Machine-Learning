from learn import Learn
import numpy
from sklearn.svm import SVC

class SupportVectorMachine(Learn):
    def __init__(self, **kwargs):
        Learn.__init__(self, **kwargs)
        self.algo = SVC();

    def _train_routine(self, train_X, train_Y):
        # define training routine for svm
        self.algo.fit(train_X, train_Y)


    def predict(self, test_X):
        # predict routine for svm
        return self.algo.predict(test_X)

    def set_parameters(self, parameters={}):
        for key, value in parameters.iteritems():
            setattr(self.algo, key, value)
        Learn.set_parameters(self, parameters)
