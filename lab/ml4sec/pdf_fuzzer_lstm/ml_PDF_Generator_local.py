from keras.models import load_model
import numpy as np
import pickle
import time
import re
import gc

class mlPDFGenerator():

    _name = "ML PDF Generator"

    def __init__(self, cache=None, lock=None):
        """
        Constructor

        :param cache: global worker cache
        :param lock: global worker lock
        """
        # super(mlPDFGenerator, self).__init__(cache=cache, lock=lock)
        self.maxlen = 50
        self.char_indices = pickle.load(open('./mlPDFModel/char_indices','rb'), encoding='latin1')
        self.indices_char = pickle.load(open('./mlPDFModel/indices_char','rb'), encoding='latin1')
        # load trained model
        self.model = load_model('./mlPDFModel/weights_06_0.64.hdf5')


    def _sample(self, preds, temperature=1.2):
        """
        Helper function to leverage probability to generate distinct output

        :param preds: prediction
        :param temperature: randomness factor
        """
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)
    
    def _strip_objs(self, objs):
        # objs: list of objects
        # remove "xx 0 obj" and "endobj" from object
        for i in range(len(objs)):
            objs[i] = re.sub('[0-9]+ 0 obj\n*\r*', '', objs[i])
            objs[i] = re.sub('\n*endobj', '', objs[i])
        return objs

    def _generate_objects(self, length, seed):
        """
        Generate PDF objects

        :return: generated objects
        :rtype: list
        """
        generated = ''
        sentence = seed
        generated += sentence
        objs=[]
        for i in range(length):
            x_pred = np.zeros((1, self.maxlen, len(self.char_indices)))
            for t, char in enumerate(sentence):
                x_pred[0, t, self.char_indices[char]] = 1.
            preds = self.model.predict(x_pred, verbose=0)[0]
            next_index = self._sample(preds)
            # next_index = np.argmax(preds)
            next_char = self.indices_char[next_index]
            generated += next_char
            sentence = sentence[1:] + next_char


        # Extract objects from generated raw data
        while len(generated):
            try:
                objs.append(generated[generated.index('obj')+3:generated.index('endobj')])
                generated = generated[generated.index('endobj')+6:]
            except ValueError as e:
                return objs

    def get_objects(self, min_obj_count=10):
        objs_list=[]

        # generate min objects
        while len(objs_list) < min_obj_count:
            # generate raw pdf objects using seed
            seed = '1 0 obj\r\n<</Type/Catalog/Pages 2 0 R/Lang(en-US) /'
            objs_list += self._generate_objects(length=600, seed=seed)

        return objs_list

# generate pdf objects
s_time = time.time()
a = mlPDFGenerator()
print('generating objects')

objects = a.get_objects()
print('\n'.join(objects))
print(time.time()-s_time)