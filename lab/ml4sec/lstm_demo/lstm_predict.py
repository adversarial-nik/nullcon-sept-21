from keras.models import load_model
import numpy as np
import pickle
import os
import sys

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def get_best_model(model_path):
    models = os.listdir(model_path)
    d = dict([(float(model.split('_')[-1][:-5]),model) for model in models])
    best_model = d[min(d.keys())]
    print('\033[96m[+] Best model: '+best_model+'\033[0m')
    return model_path+best_model

def generate_text(target_path, num_text=10, diversity=0.8):
    char_indices = pickle.load(open(target_path+'/char_indices','rb'))
    indices_char = pickle.load(open(target_path+'/indices_char','rb'))
    maxlen = 70
    print('\033[96m[+] Randomness: '+str(diversity)+'\033[0m')

    # load trained model
    print('\033[96m[+] Searching the best model for target'+'\033[0m')
    model = load_model(get_best_model(target_path+'/lstm_models/'))
    gen_text = []
    generated = []
    sentence = ['the']
    generated += sentence
    sentence_length = 40

    while (len(gen_text)!=num_text):
        x_pred = np.zeros((1, maxlen, len(char_indices)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        # next_index = np.argmax(preds)
        next_char = indices_char[next_index]
        generated += [next_char]
        sentence = sentence[1:] + [next_char]

        if len(generated) == sentence_length:
            gen_text.append(' '.join(generated))
            # reset inits
            generated = []
            sentence = ['the']
            generated += sentence
    return gen_text

for div in [1.2]:
    for i,_ in enumerate(generate_text(target_path='./', num_text=3, diversity=div)):
        print('\033[92m'+str(i)+': '+_+'\033[0m \n')
        print()
