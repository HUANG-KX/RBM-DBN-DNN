import numpy as np
import load_data as ld # load Binary AlphaDigits or Minst
import torch 
import matplotlib.pyplot as plt
from principal_RBM_alpha import RBM # load class RBM


class DBN:
    def __init__(self):
        pass
    
    def init_DBN(self, nb_layer,dim_ima,dims):
        self.layers = []
        for i in range(nb_layer):
            if i  == 0:
                a = RBM(dim_ima,dims[i])
                a.init_RBM()
                self.layers.append(a)
            else:
                a = RBM(dims[i-1],dims[i])
                a.init_RBM()
                self.layers.append(a)
                
                        
    def pretrain_DBN(self, x, epochs, lr, batch_size):
        x_ = torch.from_numpy(x).float()
        nb_layer = len(self.layers)
        for i in range(nb_layer):
            self.layers[i].train_RBM(x_, epochs, lr, batch_size)
            x_ = self.layers[i].entree_sortie_RBM(x_)
    
    def generer_image_DBN(self, nb_images, iter_gibbs):
        ncols = 20
        nrows = int(np.ceil(nb_images/ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20,5))
        fig.tight_layout()
        fig.subplots_adjust(wspace =0.03, hspace =0.03)
        axes = axes.ravel()
        nb_layer = len(self.layers)
        fin_layer = self.layers[-1]
        # Gibbs sample
        for i in range(0, nb_images):
            v = ((torch.rand(fin_layer.v_dim) < 1/2)*1).float().reshape((1,-1))
            for j in range(0, iter_gibbs):
                p_h = fin_layer.entree_sortie_RBM(v)
                h = ((torch.rand(fin_layer.h_dim) < p_h)*1).float().reshape((1,-1))
                p_v = fin_layer.sortie_entree_RBM(h)
                v = ((torch.rand(fin_layer.v_dim) < p_v)*1).float().reshape((1,-1))
            # pass through layers
            for k in range(nb_layer-2, -1,-1):
                v = self.layers[k].sortie_entree_RBM(v)
            #v = v.reshape((20,16))
            v = v.reshape((28,28))
            axes[i].imshow(v,cmap="gray")
            axes[i].set_axis_off()