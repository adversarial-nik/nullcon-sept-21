from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

# load model
model = InceptionV3()
print(model.summary())

# load image
img_path = './street_sign.jpg'
img = image.load_img(img_path, target_size=(299, 299))

# preprocess image for Inception
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img /= 255.


# select output layer
layer_index = 10
op_layer = model.get_layer(index=layer_index)
print('layer name: ', op_layer.name)

# model that outputs 
layer_model = Model(inputs=model.input, outputs=op_layer.output)
layer_op = layer_model.predict(img)

# remove single-dimensional entries
layer_op = layer_op.reshape(layer_op.shape[1:])  
layer_op = np.moveaxis(layer_op, 2, 0)


# done
fig = plt.figure()

cols = 8
rows = 8
for i in range(layer_op.shape[0]):
    fig.add_subplot(rows, cols, i+1)
    plt.imshow(layer_op[i])
plt.show()
# plt.savefig('/tmp/cnn_vis.png')