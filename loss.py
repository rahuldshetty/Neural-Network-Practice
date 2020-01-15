import numpy as np 

def loss_L2(pred,target):
    return np.sum(np.square(pred-target))/pred.shape[0]

def d_loss_L2(pred,target):
    return 2*(pred-target)
