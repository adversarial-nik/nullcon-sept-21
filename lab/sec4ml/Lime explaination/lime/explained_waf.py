import pickle

class_names = ['Good query','Bad query']
lgs = pickle.load(open('pickled_lgs', 'rb'))
vectorizer = pickle.load(open('pickled_vectorizer','rb'))


# Explaining predictions using lime
# Lime explainers assume that classifiers act on raw text, but sklearn classifiers act on vectorized representation of texts. 
# For this purpose, we use sklearn's pipeline, and implements predict_proba on raw_text lists.
from lime import lime_text
from sklearn.pipeline import make_pipeline
prediction_pipeline = make_pipeline(vectorizer, lgs)



# Now we create an explainer object. We pass the class_names as an argument for prettier display.
from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer(class_names=class_names)


# We then generate an explanation
# query = '/<script>alert(1)</script>/'
query = '/index/../../home.html'
exp = explainer.explain_instance(query, prediction_pipeline.predict_proba, num_features=6)
print('Probability =', prediction_pipeline.predict_proba([query])[0,1])


# The explanation is presented below as a list of weighted features. 
# print(exp.as_list())

# we can save the fully contained html page to a file:
exp.save_to_file('/tmp/explaination.html')
