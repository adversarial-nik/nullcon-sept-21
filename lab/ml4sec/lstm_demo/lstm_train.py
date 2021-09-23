# dataset: https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt
# numbe of lines: 10500

from keras.callbacks import LambdaCallback, CSVLogger, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
import numpy as np
import pickle
import os

def train_lstm(path='./', lr = 0.01, batch_size = 128, epochs = 50, units = 256):
    # path: target profile dir path
    # create required dirs
    model_path = path+'/lstm_models/'
    trainable_text = './t8.shakespeare.txt'
    # num_lines = 1000

    for p in [model_path]:
        if not os.path.isdir(p):
            os.mkdir(p)

    csv_logger = CSVLogger(path+'/')
    
    # load text
    with open(trainable_text) as f:
        # head = [next(f) for x in range(num_lines)]
        head = f.read().split('\n')[10000:10500]
    
    text = ' '.join(head).split(' ')
    print('\033[96m[+] Trainable corpus length:\033[0m', len(text))


    chars = sorted(list(set(text)))
    print('\033[96m[+] Unique Chars:\033[0m', len(chars))     
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # pickle char_indices and indices_char for futire use
    pickle.dump(char_indices, open(path+'char_indices','wb'))
    pickle.dump(indices_char, open(path+'indices_char','wb'))

    # generate dataset of semi-redundant sequences of maxlen characters
    maxlen = 70
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])

    # one hot encoding 
    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1


    # build model
    print('\033[96m[+] Building LSTM model\033[0m')
    model = Sequential()
    model.add(LSTM(units, input_shape=(maxlen, len(chars))))
    # model.add(LSTM(256))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    # print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    checkpointer = ModelCheckpoint(filepath=model_path+'/weights_{epoch:02d}_{loss:.4f}.hdf5')
    csv_logger = CSVLogger(path+'/keras_lstm_training.log')
    model.fit( x, y,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=[csv_logger, checkpointer])

    # on_epoch_end(19, 'garbge')
    print('\033[96m[+] Saved trained models on every iteration at {}.\033[0m'.format(model_path))
    return model_path

train_lstm(path='./', lr = 0.005, batch_size = 512, epochs = 3, units = 512)
