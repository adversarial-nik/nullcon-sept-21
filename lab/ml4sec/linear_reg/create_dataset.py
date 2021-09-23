import numpy as np
import matplotlib.pyplot as plt

def f(n): 
    return 4 + 2.5 * n + np.random.rand()

    
x = 10*np.random.rand(100, 1)                                              
y = np.array([f(n) for n in x])  

dataset = np.append(x.reshape(100,1),y.reshape(100,1),axis=1)         
np.savetxt('/tmp/linear_reg_dataset.csv', dataset, delimiter=',')

plt.scatter(x,y, marker='.')
plt.grid()
plt.xlabel('X')	
plt.ylabel('Y')
plt.show()