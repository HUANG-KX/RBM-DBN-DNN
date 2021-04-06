import numpy as np
import load_data as ld # load Binary AlphaDigits or Minst
import torch 
import matplotlib.pyplot as plt


class RBM:
    
    def __init__(self, v_dim, h_dim):
        self.v_dim = v_dim
        self.h_dim = h_dim
        
    def init_RBM(self):
        self.a = torch.zeros(self.v_dim).reshape((1,-1)).float()
        self.b = torch.zeros(self.h_dim).reshape((1,-1)).float()
        self.W =  torch.normal(0, 0.01, size=(self.v_dim,self.h_dim)).float()
        
    def entree_sortie_RBM(self, v):
        h = torch.sigmoid(self.b + v.mm(self.W))
        return h
        
    def sortie_entree_RBM(self, h):
        v = torch.sigmoid(self.a + h.mm(self.W.t()))
        return v
    
    def train_RBM(self, x, epochs, lr, batch_size):     # x: input data, lr: learning rate
        n, p = x.shape
        for i in range(0, epochs):
            if type(x) is np.ndarray:
                x_ = torch.from_numpy(x).float()
            else:
                x_ = x.clone()
            x_ = x_[torch.randperm(n)]
            for batch in range(0,n,batch_size):
                data_batch = x_[batch:min(batch+batch_size,n)]
                taille_batch = data_batch.shape[0]
                v_0 = data_batch
                p_h_v0 = self.entree_sortie_RBM(v_0)
                h_0 = ((torch.rand((taille_batch,self.h_dim)) < p_h_v0)*1).float()
                p_v_h0 = self.sortie_entree_RBM(h_0)
                v_1 = ((torch.rand((taille_batch,p)) < p_v_h0)*1).float().float()
                p_h_v_1 =  self.entree_sortie_RBM(v_1)
                da = torch.sum(v_0-v_1, axis=0).reshape((1,-1))
                db = torch.sum(p_h_v0 - p_h_v_1, axis=0).reshape((1,-1))
                dW = v_0.t().mm(p_h_v0) - v_1.t().mm(p_h_v_1)
                self.W += lr/taille_batch *dW
                self.b += lr/taille_batch *db
                self.a += lr/taille_batch *da
            h = self.entree_sortie_RBM(x_)
            x_reconstruire = self.sortie_entree_RBM(h)
            Error = torch.sum(torch.norm(x_reconstruire-x_,dim=1))/(n)
            print("Epochs {iteration}, The reconstrue error is {error}".format(iteration = i, error = Error))
    
    def generer_image_RBM(self, nb_images, iter_gibbs):
        ncols = 20
        nrows = int(np.ceil(nb_images/ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20,10))
        axes = axes.ravel()
        for i in range(0, nb_images):
            v = ((torch.rand(self.v_dim) < 1/2)*1).float().reshape((1,-1))
            for j in range(0, iter_gibbs):
                p_h = self.entree_sortie_RBM(v)
                h = ((torch.rand(self.h_dim) < p_h)*1).float().reshape((1,-1))
                p_v = self.sortie_entree_RBM(h)
                v = ((torch.rand(self.v_dim) < p_v)*1).float().reshape((1,-1))
            # reshape 
            v = v.reshape((20,16))
            axes[i].imshow(v,cmap="gray")           