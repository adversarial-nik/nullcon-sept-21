# Efficient Black Box Model Duplication with Dynamic Learning Rate

The following graphs shows the difference between loss functions of traditional and our proposed model duplication attacks.

**Classifier: Linear Regression, Dataset: Iris, Epochs: 400, Loss function: Mean Squared Error (MSE)**

*Learning rate: 0.01*

![iris_400_0.01.png](PoC/results/iris_400_0.01.png)

*Learning rate: 0.05*

![iris_400_0.05.png](PoC/results/iris_400_0.05.png)

**Classifier: Multi Layer Perceptron (MLP), Dataset: Liver disease, Epochs: 400, Loss function: Mean Squared Error (MSE)**

*Learning rate: 0.00001*

![liver_400_0.00001.png](PoC/results/liver_400_1e-05.png)

*Learning rate: 0.0001*

![liver_400_0.0001.png](PoC/results/liver_400_0.0001.png)

*Learning rate: 0.001*

![liver_400_0.001.png](PoC/results/liver_400_0.001.png)

Since both the lines converge closely at 400 epochs, lets visualize and analyze the graph after 50 epochs to see the results of the proposed approach.

*Learning rate: 0.001, epochs = 50*

![liver_50_0.001.png](PoC/results/liver_50_0.001.png)

**Classifier: Convolutional Neural Networks (CNNs) , Dataset: Land satellite disease, Epochs: 400, Loss function: Cross Entropy or Log Loss**

*Learning rate: 0.00001*

![satellite_400_0.00001.png](PoC/results/satellite_400_1e-05.png)

*Learning rate: 0.0001*

![satellite_400_0.0001.png](PoC/results/satellite_400_0.0001.png)

*Learning rate: 0.001*

![satellite_400_0.001.png](PoC/results/satellite_400_0.001.png)

# Installation
```
pip -r install requirements.txt
```

# Execution
To create the models
```
python PoC/<folder>/create_model.py
```

To steal the created models
```
python PoC/<folder>/steal_model.py
```

# Paper link 
TBD

# Authors
Nikhil Joshi, Rewanth Cool

# Conclusion
The above graphs or statistics clearly proves the ability of our proposed method to steal the black box machine learning models much more efficiently. Our proposed approach's loss function is much less fluctuating than the traditional approach.