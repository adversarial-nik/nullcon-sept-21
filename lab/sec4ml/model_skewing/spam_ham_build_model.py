import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from keras import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import History

history = History()

data = pd.read_csv("model/spam.csv", encoding='latin-1')
data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
data = data.rename(columns={"v1":"label", "v2":"text"})
data['label_num'] = data.label.map({'ham':0, 'spam':1})

X_train,X_test,y_train,y_test = train_test_split(data["text"],data["label_num"], test_size = 0.2, random_state = 10)

vect = CountVectorizer()
vect.fit(X_train)
fn = vect.get_feature_names()
X_train_df = vect.transform(X_train)
X_test_df = vect.transform(X_test)

print(X_train_df.shape)
print(X_test_df.shape)

model = Sequential()
model.add(Dense(128, input_dim=len(fn), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mse',
              optimizer='rmsprop',
              metrics=['accuracy'])

history = model.fit(X_train_df, y_train,
          epochs=20,
          batch_size=128, callbacks=[history])

score = model.evaluate(X_test_df, y_test, batch_size=128)
print(score)
print('saving model')
model.save('/tmp/trained_model')
print('saving vect')
with open('./vect', 'wb') as f:
	pickle.dump(vect, f)

plt.plot(history.history['loss'], c='r')
plt.xlabel('Loss')
plt.ylabel('Iteration')
plt.grid()
plt.show()

plt.plot(history.history['acc'], c='r')
plt.xlabel('Accuracy')
plt.ylabel('Iteration')
plt.grid()
plt.show()
