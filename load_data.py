import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical

def lire_alpha_digit(path="./Binary AlphaDigits"): # path is where you store the .mat dataset
    # file parh
    filename = "binaryalphadigs"
    filepath = path + "/" + filename
    data = scio.loadmat(filepath)
    # number of class is 36 and per class have 39 images, so the total number of image is 1404
    num_class = int(data["numclass"])
    num_per_class = int(data["classcounts"][0][0])
    # dataset has dimension (1404, 320), and label has dimension (1404,)
    dataset = []
    label = []
    for i in range(num_class):
        for j in range(num_per_class):
            dataset.append(data["dat"][i][j].reshape((1,-1))[0])
            label.append(data["classlabels"][0][i])
    dataset = np.array(dataset)
    label = np.array(label).reshape(-1)
    # return a dictonary with key: data and label
    dic ={"data": dataset,"label":label}
    return dic


def lire_Minst_BW():
    # download from keras.datasets then convert to black and white
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    n1 = train_X.shape[0]
    n2 = test_X.shape[0]
    m = train_X.shape[1] * train_X.shape[2]
    train_X = np.ceil(train_X/255).reshape((n1,m))
    test_X = np.ceil(test_X/255).reshape((n2,m))
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    return (train_X, train_y), (test_X, test_y)