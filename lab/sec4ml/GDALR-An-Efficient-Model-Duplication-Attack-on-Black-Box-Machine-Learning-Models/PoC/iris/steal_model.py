import pickle
import torch
from torch.autograd import Variable
import math
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state=0)

model = pickle.load(open('../pretrained_models/iris_model', 'rb'))

x_data = X_test
y_data = model.predict(x_data)

lb = preprocessing.LabelBinarizer()
y_data = lb.fit_transform(y_data)

plt_labels = []

def cprint(name, d):
    print("########## {} ##########".format(name))
    print(d)

class Obj:
    def __init__(self, plot=True, equation=False, lr=0.01, epoch=10, seed=1234):
        self.lr = lr
        self.epoch = epoch
        self.seed = seed
        torch.manual_seed(self.seed)

        self.w = Variable(torch.rand(4,3),  requires_grad=True)  # Any random value
        self.b = Variable(torch.rand(1,3),  requires_grad=True)  # Any random value
        
        self.losses = []
        self.equation = equation
        self.plot = plot

    def get_label_name(self):
        label_name = [k for k,v in globals().items() if v is self][0]
        cprint("label name", label_name)
        self.label_name = label_name + "_" + str(self.lr)
        plt_labels.append(self.label_name)

    def forward(self, x):
        return torch.sigmoid(torch.mm(x,self.w)+self.b)

    def loss(self, x, y):
        y_pred = self.forward(x)
        return torch.mean((y - y_pred)*(y - y_pred))
    
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
                l = self.loss(torch.Tensor([x_val]), torch.Tensor([y_val]))
                l.backward()

                self.exec_main_algo(self.w)
                self.exec_main_algo(self.b)

            if _ % 100 == 0:
                print("Epoch ", _, "loss ", l)
            
            self.losses.append(float(l.data))        
    
    def run(self):
        self.train()
        print("Epoch ", self.epoch, "loss ", self.losses[-1])
        if self.plot:
            # pickle.dump(self.losses, open('$YOUR_PATH/' + self.label_name, 'wb'))
            plt.plot(self.losses)
            
epochs = [400]
lrs = [0.01]

# Traditional method
for _e in epochs:
    for _l in lrs:
        original = Obj(epoch = _e, lr=_l)
        original.run()

# Proposed method
logfn = lambda x: abs(x*torch.log10(abs(x))*2*math.pi)
for _e in epochs:
    for _l in lrs:
        log = Obj(equation=logfn, epoch = _e, lr=_l)
        log.run()

plt.gca().legend(tuple(plt_labels))
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
