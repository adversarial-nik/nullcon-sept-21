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

# load the iris datasets
dataset = datasets.load_iris()

X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.33)

history = History()

lb = LabelBinarizer()
lb.fit(dataset.target)

y_test = lb.transform(y_test)
y_train = lb.transform(y_train)

model = Sequential()
model.add(Dense(32, input_dim=(4), activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,validation_data=(X_test,y_test),
          epochs=200,
          batch_size=128, callbacks=[history])

# model.save('$YOUR_PATH/iris_model')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.gca().legend(['loss', 'val_loss'])
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()