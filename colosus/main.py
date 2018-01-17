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

# p = np.array([0.02, 0.9, 0.02, 0.02, 0.02, 0.02])
#
# print(np.random.choice(6, 1, p=p))
# print(np.random.normal(np.ones(10), 0.01))


# p = np.random.normal(np.ones(10), 0.01)
# print(p/np.sum(p))

print("dirichlettttt")
p = np.random.dirichlet([0.3] * 10)
print(p)
print(sum(p))
print(np.std(p))
