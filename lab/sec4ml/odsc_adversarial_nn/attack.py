## Copyright (C) 2017, Nicholas Carlini and Nicolas Papernot.
## All rights reserved.
from __future__ import print_function


# Start off by just disbaling tensorflow's warnings.
try:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
except:
    # it's okay if it fails
    pass

import keras.preprocessing.image as img
# First, we import tensorflow (the ML library we are going to use)
import tensorflow as tf

# Next we'll import numpy and scipy, two common libraries
import numpy as np  # useful to manipulate matrices
import scipy.misc  # useful to load and save images in our case

# Then we import the Inception file, for classifying our images
import inception

# Tensorflow works by maintaining a "session" of the current
# environment. This is where we instantiate it.
# The session contains all of the elements of the graph,
# which is how TensorFlow defines a neural network.
sess = tf.Session()

# First, we set up the Inception model. If this is the first
# time this file is being run, this will also download and
# extract the Inception weights into a temporary folder.
model = inception.setup(sess)

# Now we're going to attack it.
# Begin by importing the cleverhans library, which we use to
# generate adversarial examples
import cleverhans.attacks

# Cleverhans has it's own model interface, so we're going to
# need to wrap the model we have with the callable wrapper.
# This lets us tell cleverhans that we are passing it the
# logits layer with Inception (as opposed to the probs).
from cleverhans.model import CallableModelWrapper

cleverhans_model = CallableModelWrapper(model, 'logits')

# We are going to use a very simple attack to start off.
# Begin by constructing an instance of the attack object.
fgsm = cleverhans.attacks.FastGradientMethod(cleverhans_model, sess=sess)
# lbfgs = cleverhans.attacks.LBFGS(cleverhans_model, sess=sess)


# We're now done defining the graph. It's time to actually
# load up some images and generate adversarial examples.

# The first image, of a panda, is already in the images folder.
# We load it with scipy
image = img.load_img("images/panda.png", target_size=(299, 299))
image = img.img_to_array(image)

# And finally, convert each pixel to the range [0,1].
image = (image/255.0)

# The fast gradient (sign) method constructs adversarial
# examples by slightly moving the image in the direction
# of the gradient, to maximize the loss of the network.
target = np.zeros((1,1000))
target[0,820] = 1
adversarial_example = fgsm.generate_np(np.array([image]),
                                       y_target=target,
                                       eps=.2)

# adversarial_example = lbfgs.generate_np(np.array([image]),
#                                        y_target=target)

img.save_img("/tmp/adversarial_panda_FGSM.png", adversarial_example[0])

print("Done generating adversarial example.")
