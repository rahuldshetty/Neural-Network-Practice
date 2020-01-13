import numpy as np
class FullyConnectedLayer(object):

    def __init__(self,num_inputs,layer_size,activation_fn):
        super().__init__()
        self.W = np.random.standard_normal((num_inputs,layer_size))
        self.b = np.random.standard_normal(layer_size)
        self.activation_fn = activation_fn
    
    def forward(self,x):
        z = np.dot(x,self.W) + self.b 
        return self.activation_fn(z)

if __name__ == "__main__":
    x1 = np.random.uniform(-1,1,2).reshape(1,2)
    x2 = np.random.uniform(-1,1,2).reshape(1,2)
    print("Inputs:",x1,x2)

    relu_fn = lambda x: np.maximum(0,x)
    layer = FullyConnectedLayer(2,3,relu_fn)

    out1 = layer.forward(x1)
    out2 = layer.forward(x2)

    x12 = np.concatenate((x1,x2))
    out12 = layer.forward(x12)
    print("Results:",out12)