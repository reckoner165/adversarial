# __author__ = 'Srinivasan'

# from __future__ import print_function
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import model_from_json
from keras import backend as K

import numpy
import scipy.misc
from scipy import optimize as spo

import logging as log

num_classes = 10

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Load json and create model
json_file = open('mnist_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)


# Load weights into new model
loaded_model.load_weights("mnist_model.h5")
print("Loaded model from disk")

# # evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


pred = loaded_model.predict(numpy.expand_dims(x_test[0], axis=0))
print('prediction',pred,'label',y_test[0])

print('cross entropy')
cross = numpy.sum(y_test[0]*numpy.log2(pred))
print(cross)

# Let's do L2 norm because it's easy (ffs!)

epsilon = 5
b_init = numpy.zeros(input_shape)

b = b_init
pred_x = loaded_model.predict(numpy.expand_dims(x_test[0], axis=0))
pred_perturb = loaded_model.predict(numpy.expand_dims((x_test[0] + b), axis=0))
J_init = numpy.sum(pred_x*numpy.log2(pred_perturb))
print('J_init: ', J_init)

# test_range = x_test.shape[0]
test_range = 300

J = numpy.zeros(test_range)
b = numpy.random.random(input_shape)
b = epsilon*b/numpy.linalg.norm(b)

# Let's try scipy optimize minimize

def cross_ent(b):
    b = numpy.reshape(b,input_shape)
    J = 0
    for iter in range(0,test_range):
        pred_x = loaded_model.predict(numpy.expand_dims(x_test[iter], axis=0))
        pred_perturb = loaded_model.predict(numpy.expand_dims((x_test[iter] + b), axis=0))

        # Noise term to avoid zero error in the log function
        avoid_zeros = 0.0000000001*numpy.ones(pred_perturb.shape)

        y_log_p = pred_x*numpy.log2(pred_perturb + avoid_zeros)
        one_minus = (1-pred_x)*numpy.log2((1-pred_perturb) + avoid_zeros)

        J = J + numpy.sum(y_log_p + one_minus)
    return J/test_range


def constraint(b):
    return numpy.linalg.norm(b) - epsilon

cons = ({'type': 'ineq', 'fun': constraint })

x_mini_test = x_test[0:test_range]
x_mini_perturbed = x_test[0:test_range]
y_mini_test = y_test[0:test_range]

print('Performance for unperturbed data:')
new_score = loaded_model.evaluate(x_mini_test, y_mini_test, verbose=1)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], new_score[1]*100))





epochs = 10
for i in range(0, epochs):
    print('Starting ',i)
    sol = spo.minimize(cross_ent, b_init, constraints=cons)
    b_init = epsilon*sol.x/numpy.linalg.norm(sol.x)

    print('Iteration ',i, ' completed.')
    for k in range(0,test_range):
        # y_mini_test[i] = loaded_model.predict(numpy.expand_dims(x_mini_test[i], axis = 0))
        x_mini_perturbed[k] = x_mini_perturbed[k] + numpy.reshape(b, input_shape)

    # Testing the perturbation in the network
    print('Performance for perturbed data:')
    new_score = loaded_model.evaluate(x_mini_perturbed, y_mini_test, verbose=1)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], new_score[1]*100))

    print('Norm of perturbation', numpy.linalg.norm(b))
    print('----')
