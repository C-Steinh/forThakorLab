import numpy as np
from sklearn.svm import SVC

class SVM:
    def __init__(self, X, y, C = 1.0, kernel = 'linear'):
        self.__model = SVC( C = C, kernel = kernel, probability = True )
        self.train( X, y )

    def train(self, X, y):
        self.__model = self.__model.fit( X, y )

    def predict(self, X, prob = False):
        yprob = None
        if prob:
            yprob = self.__model.predict_proba( X )
            yhat = np.argmax( yprob, axis=1 )
        else: yhat = self.__model.predict( X )        
        return yhat.astype( np.int ), yprob

if __name__ == '__main__':
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from ..utils.Metrics import confusion_matrix

    import time

    data = load_digits()
    Xtrain, Xtest, ytrain, ytest = train_test_split( data.data, data.target, test_size = 0.33 )

    t1 = time.clock()
    mdl = SVM( Xtrain, ytrain )
    t2 = time.clock()
    yhat, _ = mdl.predict( Xtest, prob = True )
    t3 = time.clock()

    print( 'Training time:', t2 - t1, 's' )
    print( 'Testing time:', t3 - t2, 's' )

    cm = confusion_matrix( ytest, yhat, labels = data.target_names )