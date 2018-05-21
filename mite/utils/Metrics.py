import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

def confusion_matrix(ytest, yhat, labels = [], cmap = 'viridis', show = True):
    cm = sk_confusion_matrix( ytest, yhat )
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    if show:
        plt.imshow( cm, interpolation = 'nearest', vmin = 0.0, vmax = 1.0 )
        try:
            plt.set_cmap( cmap )
        except ValueError as err: cmap = 'viridis'
        plt.colorbar()
    
        if len( labels ):
            tick_marks = np.arange( len( labels ) )
            plt.xticks( tick_marks, labels, rotation=0 )
            plt.yticks( tick_marks, labels )
    
        thresh = 0.5 # cm.max() / 2.
        colors = mpl.cm.get_cmap( cmap )
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            r,g,b,a = colors(cm[i,j])
            br = np.sqrt( r*r*0.241 + g*g*0.691 + b*b*0.068 )
            plt.text(j, i, format(cm[i, j], '.2f'),
                     horizontalalignment = "center",
                     verticalalignment = 'center',
                     color = "black" if br > thresh else "white")
    
        plt.tight_layout()
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
    
        plt.show( block = True )
    return cm