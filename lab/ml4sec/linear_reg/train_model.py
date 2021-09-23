import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# load dataset
data = pd.read_csv('/tmp/linear_reg_dataset.csv', header=None).values
X = data[:,0].reshape(-1,1)
Y = data[:,1]

# split dataset as train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=1337)

# build and train model
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, Y_train)

# get predictions on test set
y_pred = lin_reg_model.predict(X_test)

# calculate error on test set
mse = mean_squared_error(y_pred, Y_test)

print('MSE: ',mse)
print('coef: ',lin_reg_model.coef_)
print('intercept: ',lin_reg_model.intercept_)

# plot line
plt.scatter(X,Y, marker='.')

x = range(11)
y = [lin_reg_model.intercept_+(lin_reg_model.coef_[0]*i) for i in x]

plt.grid()
plt.plot(y, c='r')
plt.show()

# predict for new input
ip = 50
ip_pred = lin_reg_model.predict([[ip]])
print('When X = {}, Y is {}'.format(ip, ip_pred))
