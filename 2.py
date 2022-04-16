import tensorflow as tf
import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense


print("\n\nNum GPUs Available: {}\n\n".format(tf.config.list_physical_devices('GPU')))

