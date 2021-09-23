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
# from keras.datasets import mnist
import pickle
import keras
import random
from pandas import read_csv

np.random.seed(1337)
history = History()
data = read_csv('dataset/liver_disease.csv', header=None).values

dataX = data[:,:-1]
dataY = data[:,-1]-1


dataY = keras.utils.to_categorical(dataY, num_classes=2)

x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.3, random_state=1)

model = Sequential()
model.add(Dense(16, input_dim=6, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))

opt = keras.optimizers.Adam(lr=0.0005)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_data=(x_test,y_test),
          epochs=300,
          batch_size=8, callbacks=[history])

# model.save('$YOUR_PATH/nn_liver_disease_model')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.gca().legend(['loss', 'val_loss'])
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()