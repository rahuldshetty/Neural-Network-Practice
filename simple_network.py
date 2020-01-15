import numpy as np 
from activation import *
from loss import *
from fully_conneted import *
class SimpleNetwork(object):

    def __init__(self,num_inputs,num_outputs,hidden_layers_sizes=(64,32),loss_fn=loss_L2,d_loss_fn=d_loss_L2):
        super().__init__()
        sizes = [num_inputs,*hidden_layers_sizes,num_outputs]
        self.layers = [
            FullyConnectedLayer(sizes[i],sizes[i+1],sigmoid,d_sigmoid)
            for i in range(len(sizes)-1)
        ]

        self.loss_fn,self.d_loss_fn = loss_fn,d_loss_fn

    def backward(self,dL_dy):
        for layer in reversed(self.layers):
            dL_dy = layer.backward(dL_dy)
        return dL_dy

    def optimize(self,epsilon):
        for layer in self.layers:
            layer.optimize(epsilon)
        
    def train(self,X_train,Y_train,X_val,Y_val,batch_size = 32,epochs = 5,lr = 5e-3):
        num_batches_per_epoch = len(X_train)
        loss,accuracy = [],[]
        for i in range(epochs):
            epoch_loss = 0
            for b in range(num_batches_per_epoch):
                b_idx = b*batch_size
                b_idx_e = b_idx + batch_size
                x,y_true = X_train[b_idx:b_idx_e],Y_train[b_idx:b_idx_e]
                y = self.forward(x)
                epoch_loss += self.loss_fn(y,y_true)
                dL_dy = self.d_loss_fn(y,y_true)
                
                self.backward(dL_dy)

                self.optimize(lr)

            loss.append(epoch_loss/num_batches_per_epoch)
            accuracy.append(self.evaluate_accuracy(X_val,Y_val))
            print("Epoch:{:4d}: training loss = {:.6f} | Validation Accuracy = {:.2f}%".format(i,loss[i],accuracy[i]*100))

    def forward(self,x):
        for layer in self.layers:
            x = layer.forward(x)
        return x 
    
    def predict(self,x):
        estimations = self.forward(x)
        best_class = np.argmax(estimations)
        return best_class
    
    def evaluate_accuracy(self,X_val,Y_val):
        num_corrects = 0
        for i in range(len(X_val)):
            if self.predict(X_val) == Y_val[i]:
                num_corrects += 1
        return num_corrects/len(X_val)