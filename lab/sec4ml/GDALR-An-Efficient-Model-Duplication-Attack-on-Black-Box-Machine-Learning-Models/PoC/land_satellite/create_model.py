import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras import Sequential
from keras.layers import Dense, Dropout, Conv2D, AveragePooling2D, Flatten
from keras.callbacks import History
import pickle
import keras
import random
from pandas import read_csv
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np

np.random.seed(1337)
history = History()
data_train = read_csv('dataset/sat.trn', header=None, delimiter=' ').values
data_test = read_csv('dataset/sat.tst', header=None, delimiter=' ').values

x_train = data_train[:,:-1]
x_train = x_train.reshape(4435,4,9,1)
y_train = data_train[:,-1]-1

x_test = data_test[:,:-1]
x_test = x_test.reshape(2000,4,9,1)
y_test = data_test[:,-1]-1

y_train = keras.utils.to_categorical(y_train, num_classes=7)
y_test = keras.utils.to_categorical(y_test, num_classes=7)

model = Sequential()
model.add(Conv2D(64, kernel_size=(2,2), activation='sigmoid', input_shape=(4,9,1)))
model.add(AveragePooling2D(pool_size=(1,2)))
model.add(Conv2D(128, (2,2), activation='sigmoid'))
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(7, activation='softmax'))

opt = keras.optimizers.Adam(lr=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_data=(x_test,y_test),
          epochs=50,
          batch_size=32, callbacks=[history])

model.save('/tmp/nn_sat_keras_model')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.gca().legend(['loss', 'val_loss'])
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
