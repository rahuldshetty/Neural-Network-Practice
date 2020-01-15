import numpy as np
class FullyConnectedLayer(object):

    def __init__(self,num_inputs,layer_size,activation_fn,d_activation_fn):
        super().__init__()
        self.W = np.random.standard_normal((num_inputs,layer_size))
        self.b = np.random.standard_normal(layer_size)
        self.activation_fn = activation_fn
        self.d_activation_fn = d_activation_fn
        self.x,self.y,self.dL_dW,self.dL_db = 0,0,0,0

    
    def forward(self,x):
        z = np.dot(x,self.W) + self.b 
        self.y = self.activation_fn(z)
        self.x = x
        return self.y

    def backward(self,dL_dy):
        dy_dz = self.d_activation_fn(self.y)
        dL_dz = (dL_dy*dy_dz)
        dz_dW =self.x.T 
        dz_dx = self.W.T 

        dz_db = np.ones(dL_dy.shape[0])

        self.dL_dW = np.dot(dz_dW,dL_dz)
        self.dL_db = np.dot(dz_db,dL_dz)

        dL_dx = np.dot(dL_dz,dz_dx)
        return dL_dx

    def optimize(self,epsilon):
        self.W -= epsilon*self.dL_dW
        self.b -= epsilon*self.dL_db