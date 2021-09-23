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
from torch import optim
from pandas import read_csv
import keras

torch.set_printoptions(precision=15)

# GPU stuff
gpu_exists = False
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if device.type == 'cuda':
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')
#     torch.backends.cudnn.benchmark=True
#     gpu_exists = True

actual_model = load_model('../pretrained_models/land_satellite_model')

data_test = read_csv('dataset/sat.trn', header=None, delimiter=' ').values
x_data = data_test[:,:-1][:200]
x_data = x_data.reshape(x_data.shape[0],1,4,9)
y_data = actual_model.predict(x_data.reshape((x_data.shape[0],4,9,1)))

# Labelize data into binary
_ = np.zeros_like(y_data)
_[np.arange(len(y_data)), y_data.argmax(1)] = 1
y_data = _

plt_labels = []

def cprint(name, d):
    print("########## {} ##########".format(name))
    print(d)

class Net(nn.Module):
    def __init__(self, seed=1234):
        self.seed = seed
        torch.manual_seed(self.seed)

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, (2,2))
        self.pool1 = nn.AvgPool2d(1, 2)
        self.conv2 = nn.Conv2d(64, 128, (2,2))
        self.pool2 = nn.AvgPool2d(1,4)
        self.fc1 = nn.Linear(128, 7)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 128)
        x = F.softmax(self.fc1(x))
        return x

    def loss(self, x, y):
        y_pred = self.forward(x)

        # logloss / cross entropy implementation
        index = y.argmax()
        ans = -torch.log(y_pred[0][int(index)])
        return ans

class NNet:
    def __init__(self, plot=True, equation=None, epoch=1000, lr=0.01):
        if gpu_exists:
            self.model = Net().to(device)
        else:
            self.model = Net()

        self.equation = equation
        self.l = None
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
        g1 = torch.tanh(g)
        g1[g1 == 0] = math.pow(10, -7)

        if self.equation:
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
                l = self.model.loss(torch.Tensor([x_val]), torch.Tensor([y_val]))
                l.backward()

                self.exec_main_algo(self.model.conv1.weight)
                self.exec_main_algo(self.model.conv1.bias)

                self.exec_main_algo(self.model.conv2.weight)
                self.exec_main_algo(self.model.conv2.bias)

                self.exec_main_algo(self.model.fc1.weight)
                self.exec_main_algo(self.model.fc1.bias)
            
            if _ % 10 == 0:
                print("Epoch ", _, "loss ", l)
            
            self.losses.append(float(l.data))

    def run(self):
        self.train()
        print("Epoch ", self.epoch, "loss ", self.losses[-1])
        if self.plot:
            # pickle.dump(self.losses, open('$YOUR_PATH/' + self.label_name, 'wb'))
            plt.plot(self.losses)

epochs = [30]
lrs = [0.0001]

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

