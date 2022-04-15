import tensorflow as tf
import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense


print("\n\nInstalled Tensorflow Version : {}\n\n".format(tf.__version__))

myNNLayer = Dense(units=1, input_shape=[1])

model = Sequential([myNNLayer])
print("\n\n")
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500)

inputX = 10.0
outputY = model.predict([inputX])
print("\n\n [Optimizer: 'Stochastic Gradient Descent', Loss Function: 'Mean Squared Error'] ** Calculated Weights based on model training: ' {} '".format(myNNLayer.get_weights()))
print("\n\nFor input X: '{}', My Model Predicted Y: '{}'\n\n".format(inputX, outputY))
