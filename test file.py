import tensorflow as tf
import numpy as np


y = x.reshape(-1,1)

w = tf.Variable([[0],[0],[0],[0]], trainable=True, dtype=tf.float64)