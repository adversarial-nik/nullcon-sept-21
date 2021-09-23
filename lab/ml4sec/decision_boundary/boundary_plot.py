import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# get dataset
# x, y = make_moons(noise=0.3, random_state=0)
x, y = make_circles(noise=0.2, factor=0.5, random_state=1)

# split into training and testing set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=1337)


# meshgrid for search space
h = 0.1 # mesh step size
x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# train classifier
# clf = KNeighborsClassifier(3)
# clf = DecisionTreeClassifier(max_depth=5)
clf = RandomForestClassifier(max_depth=5, n_estimators=8, max_features=1)
# clf = LogisticRegression()
# clf = MLPClassifier(alpha=0.1, max_iter=1000)

clf.fit(x_train, y_train)
print('classifier: ', type(clf).__name__)
print('score: ', clf.score(x_test, y_test))

# get predictions for points in mesh
pred = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
pred = pred.reshape(xx.shape)


# plot the predictions 
plt.contourf(xx, yy, pred,  alpha=0.8)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train,  edgecolors='k')
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test,  edgecolors='k')
plt.show()
# plt.savefig('/tmp/'+str(type(clf).__name__)+'.png')
