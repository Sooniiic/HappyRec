import numpy as np

def get_auto_embedding_dim(num_classes):
    '''
    reference: Deep & Cross Network for Ad Click Predictions.(ADKDD'17)
    why?
    '''
    return int(np.floor(6 * np.pow(num_classes, 0.25)))