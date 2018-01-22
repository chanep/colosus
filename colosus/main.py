import tensorflow as tf
import numpy as np
import time

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

# p = np.array([0.02, 0.9, 0.02, 0.02, 0.02, 0.02])
#
# print(np.random.choice(6, 1, p=p))
# print(np.random.normal(np.ones(10), 0.01))


# p = np.random.normal(np.ones(10), 0.01)
# print(p/np.sum(p))

# print("dirichlettttt")
# p = np.random.dirichlet([0.3] * 10)
# print(p is None)
# print(type(p))
# print(np.std(p))

x = np.array([15], dtype=">u2")
# x = np.array([2], np.uint8)

# print(x)
# print(type(x))
#
# y = x.view(np.uint8)
#
# print(y)
# print(type(y))
# print(y.shape)
# print(y.dtype)
# print(np.flip(np.unpackbits(y), axis=0))
#
# print("\n\n")
#
# x = 15
# a = np.zeros((16,), np.uint8)
#
# start = time.time()
#
# for n in range(100000):
#     a = np.zeros((16,), np.uint8)
#     for i in range(16):
#         a[i] = (x & (1 << i)) >> i
#
# print(a)
# print(str(time.time() - start))
#
# start = time.time()
#
# for n in range(100000):
#     y = np.array([x], dtype=">u2")
#     z = y.view(np.uint8)
#     a = np.flip(np.unpackbits(z), axis=0)
#
# print(a)
# print(str(time.time() - start))
x = np.zeros((3, 3))
x[0, :] = np.array([1, 2, 3])
print(x)