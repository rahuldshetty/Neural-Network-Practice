import numpy as np

class Neuron(object):
    def __init__(self,num_inputs,activation_fn):
        super().__init__()
        self.W = np.random.rand(num_inputs)
        self.b = np.random.rand(1)
        self.activation_fn = activation_fn
    
    def forward(self,x):
        z = np.dot(x,self.W) + self.b
        return self.activation_fn(z)

if __name__ == "__main__":
    x = np.random.rand(3).reshape(1,3)
    print("Input:",x)
    step_fn = lambda x: 1 if x > 0 else 0 
    perceptron = Neuron(x.size,step_fn)
    out = perceptron.forward(x)
    print("Output",out)    
