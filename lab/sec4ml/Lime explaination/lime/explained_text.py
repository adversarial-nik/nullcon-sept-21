#!/usr/bin/env python
# coding: utf-8

# In[1]:

import lime
import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics

# ## Fetching data, training a classifier

# For this tutorial, we'll be using the [20 newsgroups dataset](http://scikit-learn.org/stable/datasets/#the-20-newsgroups-text-dataset). In particular, for simplicity, we'll use a 2-class subset: atheism and christianity.

# In[2]:


from sklearn.datasets import fetch_20newsgroups
categories = ['alt.atheism', 'soc.religion.christian']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
class_names = ['atheism', 'christian']


# Let's use the tfidf vectorizer, commonly used for text.
# In[3]:
vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(newsgroups_train.data)
test_vectors = vectorizer.transform(newsgroups_test.data)


# Now, let's say we want to use random forests for classification. It's usually hard to understand what random forests are doing, especially with many trees.
# In[4]:
rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(train_vectors, newsgroups_train.target)


# In[5]:
pred = rf.predict(test_vectors)
sklearn.metrics.f1_score(newsgroups_test.target, pred, average='binary')


# We see that this classifier achieves a very high F score. [The sklearn guide to 20 newsgroups](http://scikit-learn.org/stable/datasets/#filtering-text-for-more-realistic-training) indicates that Multinomial Naive Bayes overfits this dataset by learning irrelevant stuff, such as headers. Let's see if random forests do the same.

# ## Explaining predictions using lime

# Lime explainers assume that classifiers act on raw text, but sklearn classifiers act on vectorized representation of texts. For this purpose, we use sklearn's pipeline, and implements predict_proba on raw_text lists.

# In[6]:


from lime import lime_text
from sklearn.pipeline import make_pipeline
c = make_pipeline(vectorizer, rf)


# In[7]:
print(c.predict_proba([newsgroups_test.data[0]]))


# Now we create an explainer object. We pass the class_names as an argument for prettier display.

# In[8]:


from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer(class_names=class_names)


# We then generate an explanation with at most 6 features for an arbitrary document in the test set.

# In[9]:


idx = 100
exp = explainer.explain_instance(newsgroups_test.data[idx], c.predict_proba, num_features=6)
print('Document id: %d' % idx)
print('Probability(christian) =', c.predict_proba([newsgroups_test.data[idx]])[0,1])
print('True class: %s' % class_names[newsgroups_test.target[idx]])


# The classifier got this example right (it predicted atheism).  
# The explanation is presented below as a list of weighted features. 

# In[10]:


exp.as_list()


# These weighted features are a linear model, which approximates the behaviour of the random forest classifier in the vicinity of the test example. Roughly, if we remove 'Posting' and 'Host' from the document , the prediction should move towards the opposite class (Christianity) by about 0.27 (the sum of the weights for both features). Let's see if this is the case.

# In[11]:


print('Original prediction:', rf.predict_proba(test_vectors[idx])[0,1])
tmp = test_vectors[idx].copy()
tmp[0,vectorizer.vocabulary_['Posting']] = 0
tmp[0,vectorizer.vocabulary_['Host']] = 0
print('Prediction removing some features:', rf.predict_proba(tmp)[0,1])
print('Difference:', rf.predict_proba(tmp)[0,1] - rf.predict_proba(test_vectors[idx])[0,1])


# Pretty close!  
# The words that explain the model around this document seem very arbitrary - not much to do with either Christianity or Atheism.  
# In fact, these are words that appear in the email headers (you will see this clearly soon), which make distinguishing between the classes much easier.

# ## Visualizing explanations

# Alternatively, we can save the fully contained html page to a file:

# In[14]:
exp.save_to_file('/tmp/explaination.html')
