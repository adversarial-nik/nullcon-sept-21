import pickle
import numpy as np

labels = ['Good query','Bad query']
query = '/<script>alert(1)</script>' + '/home'*20
# query = '/examples/query'

# load model and tfidf vectorizer
lgs = pickle.load(open('pickled_lgs', 'rb'))
vectorizer = pickle.load(open('pickled_vectorizer','rb'))

# vectorize
# query = input('Query: ')
query_vectorized = vectorizer.transform([query])

# predict
proba = lgs.predict_proba(query_vectorized)

print('Predicted probabilities: ',proba)
print('score: ', proba[0, 0] / len(query))
print('Predicted class: ',labels[np.argmax(proba)])
print()

