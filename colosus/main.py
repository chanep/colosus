import tensorflow as tf
import numpy as np

v = np.random.random_integers(1,9,(3,1,1,2))
x = tf.cast(v,tf.float32)
w = tf.ones((1,1,2,1))

conv = tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')


with tf.Session() as sess:
    result = sess.run(conv)
    x_res = sess.run(x)

print("result: {}".format(result))
print("x: {}".format(x_res))
print("shape: {}".format(result.shape))

b = np.zeros((8,8), np.uint8)
b[1:4, 3:6] = 1
c = np.zeros((8,8), np.uint8)
c[6,4] = 1
print(c)
d = np.sum(c, axis=0)
print(d)
e = np.sum(c, axis=1)
print(e)
print(np.arange(8))




