import numpy as np
import load_data as ld # load Binary AlphaDigits or Minst
import torch 
import matplotlib.pyplot as plt
from principal_DBM_alpha import DBN # load class DBN
from principal_RBM_alpha import RBM

class DNN:
    
    def __init__(self, nb_layer, dim_ima, dims, dim_out):
        self.dnn = DBN()
        self.dnn.init_DBN(nb_layer, dim_ima = dim_ima, dims = dims)
        self.dim_ima = dim_ima
        self.dims = dims
        self.dim_out = dim_out
        self.classifier =  RBM(self.dims[-1], self.dim_out)
        self.classifier.init_RBM()
        
    def calcul_softma(self, x, rbm):
        W = self.classifier.W
        b = self.classifier.b
        h = x @ W + b
        softmax = torch.nn.Softmax(dim=1)
        p = softmax(h)
        return h,p
    
    def entree_sortie_reseau(self, x):
        hidden_layers = []
        if type(x) is np.ndarray:
            x_ = torch.from_numpy(x).float()
        else:
            x_ = x.clone()
        nb_layer = len(self.dnn.layers)
        h = x_
        hidden_layers.append(h)
        for i in range(nb_layer):
            h = self.dnn.layers[i].entree_sortie_RBM(h)
            hidden_layers.append(h)
        h, p =  self.calcul_softma(h, self.classifier)
        hidden_layers.append(h)
        return hidden_layers,p
    
    def retropropagation(self, x, y, epochs, lr, batch_size, x_test, y_test):
        n, _ = x.shape
        nb_layer = len(self.dnn.layers) + 1
        for i in range(0, epochs):
            if type(x) is np.ndarray:
                x_ = torch.from_numpy(x).float()
                y_ = torch.from_numpy(y).float()
            else:
                x_ = x.clone()
                y_ = y.clone()
            idx = torch.randperm(n)
            x_ = x_[idx]
            y_ = y_[idx]
            for batch in range(0,n,batch_size):
                dW_ = []
                db_ = []
                data_batch = x_[batch:min(batch+batch_size,n)]
                y_batch = y_[batch:min(batch+batch_size,n)]
                taille_batch = data_batch.shape[0]
                h, p = self.entree_sortie_reseau(data_batch)
                for j in range(nb_layer-1, -1,-1):
                    if j == nb_layer-1:
                        C =  p - y_batch
                    elif j == nb_layer-2:
                        C_ = C
                        C = C_ @ self.classifier.W.t() * (h[j+1] * (1-h[j+1]))
                    else:
                        C_ = C
                        C = C_ @ self.dnn.layers[j+1].W.t() * (h[j+1] * (1-h[j+1]))
                    dW_.append((h[j].t() @ C)/taille_batch)
                    db_.append(torch.mean(C,dim=0))
                #gradient decent
                self.classifier.W -= lr * dW_[0]
                self.classifier.b -= lr * db_[0].reshape((1,-1))
                for k in range(0,nb_layer-1):
                    self.dnn.layers[k].W -= lr * dW_[nb_layer-1-k]
                    self.dnn.layers[k].b -= lr * db_[nb_layer-1-k].reshape((1,-1))
            # train loss and accuracy
            _, p = self.entree_sortie_reseau(torch.from_numpy(x).float())
            Error = -torch.sum(torch.from_numpy(y).float()*torch.log(p))/n
            idx0 = torch.argmax(p, dim=1).numpy().astype(int)
            idx1 = torch.argmax(torch.from_numpy(y).float(), dim=1).numpy().astype(int)
            accuracy = (np.array(idx0) == np.array(idx1)).mean()
            # test loss and accuracy
            _, p = self.entree_sortie_reseau(torch.from_numpy(x_test).float())
            Error_ = -torch.sum(torch.from_numpy(y_test).float()*torch.log(p))/n
            idx0 = torch.argmax(p, dim=1).numpy().astype(int)
            idx1 = torch.argmax(torch.from_numpy(y_test).float(), dim=1).numpy().astype(int)
            accuracy_ = (np.array(idx0) == np.array(idx1)).mean()
            print("Epochs {iteration}, The train_loss is {error}, The train_accuracy is {accuracy}, The test_accuracy is {accuracy_}".format(iteration = i, error = Error, accuracy = accuracy, error_ = Error_, accuracy_ = accuracy_))
    
    def test_DNN(self, x, y):
        if type(x) is np.ndarray:
            x_ = torch.from_numpy(x).float()
            y_ = torch.from_numpy(y).float()
        else:
            x_ = x.clone()
            y_ = y.clone()
        _, p = self.entree_sortie_reseau(x_)
        idx0 = torch.argmax(p, dim=1).numpy().astype(int)
        idx1 = torch.argmax(y_, dim=1).numpy().astype(int)
        accuracy = (np.array(idx0) == np.array(idx1)).mean()
        loss = -torch.sum(y_*torch.log(p))/x.shape[0]
        return loss,accuracy
    
    
    
    
    
    
    
    
    
    