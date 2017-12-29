import tensorflow as tf
import numpy as np

x = tf.constant(5)
y = tf.constant(6)

z = x * y

with tf.Session() as sess:
    result = sess.run(z)

print("result: {}".format(result))

a = np.zeros((2, 2), np.uint64)
b = np.copy(a)
a[1, 1] = 10
print("a: {}".format(a))
print("b: {}".format(b))
