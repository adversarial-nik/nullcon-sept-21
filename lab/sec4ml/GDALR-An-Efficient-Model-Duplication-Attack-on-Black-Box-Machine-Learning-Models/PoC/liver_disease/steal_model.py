import pickle
import torch
from torch.autograd import Variable
import torch.nn as nn
from keras.models import load_model
import math
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from pandas import read_csv
from keras.utils import to_categorical

data = read_csv('dataset/liver_disease.csv', header=None).values

dataX = data[:,:-1]
dataY = data[:,-1]-1


dataY = to_categorical(dataY, num_classes=2)

x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.1, random_state=1)


model = load_model('../pretrained_models/liver_disease_model')

x_data = x_train
y_data = model.predict(x_data)
print(y_data.shape)
print(x_data.shape)


# Labelize data into binary
_ = np.zeros_like(y_data)
_[np.arange(len(y_data)), y_data.argmax(1)] = 1
y_data = _

plt_labels = []

def cprint(name, d):
    print("########## {} ##########".format(name))
    print(d)

class Net(nn.Module):
    def __init__(self, seed=1337):
        self.seed = seed
        torch.manual_seed(self.seed)

        super(Net, self).__init__()
        self.fc1 = nn.Linear(6, 16)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc3(x))
        return x
    
    def loss(self, x, y):
        y_pred = self.forward(x)
        return torch.mean((y - y_pred)*(y - y_pred))
    
class NNet:
    def __init__(self, plot=True, equation=None, epoch=1000, lr=0.01):
        self.new_model = Net()

        self.equation = equation
        self.lr = lr
        self.epoch = epoch
        self.plot = plot
        self.losses = []
    
    def get_label_name(self):
        label_name = [k for k,v in globals().items() if v is self][0]
        cprint("label name", label_name)
        self.label_name = label_name + "_" + str(self.lr)
        plt_labels.append(self.label_name)

    def exec_main_algo(self, v):
        g = v.grad.data

        if self.equation:
            g1 = torch.tanh(g)
            g1[g1 == 0] = 1e-7

            lr_current = self.equation(g1)*self.lr
            v.data = v.data - lr_current * g
        else:
            v.data = v.data - self.lr * g

        # Manually zero the gradients after updating weights
        v.grad.data.zero_()

    def train(self):
        self.get_label_name()
        for _ in range(self.epoch):
            for x_val, y_val in zip(x_data, y_data):
                l = self.new_model.loss(torch.Tensor([x_val]), torch.Tensor([y_val]))
                l.backward()

                self.exec_main_algo(self.new_model.fc1.weight)
                self.exec_main_algo(self.new_model.fc1.bias)
                self.exec_main_algo(self.new_model.fc3.weight)
                self.exec_main_algo(self.new_model.fc3.bias)
            
            if _ % 10 == 0:
                print("Epoch ", _, "loss ", l)
            
            self.losses.append(float(l.data))

    def run(self):
        self.train()
        print("Epoch ", self.epoch, "loss ", self.losses[-1])
        if self.plot:
            # pickle.dump(self.losses, open('$YOUR_PATH/' + self.label_name, 'wb'))
            plt.plot(self.losses)

epochs = [40]
lrs = [1e-5]

for _e in epochs:
    for _l in lrs:
        original = NNet(epoch = _e, lr=_l)
        original.run()

logfn = lambda x: abs(x*torch.log10(abs(x))*2*math.pi)
for _e in epochs:
    for _l in lrs:
        log = NNet(equation=logfn, epoch = _e, lr=_l)
        log.run()

plt.gca().legend(tuple(plt_labels))
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
