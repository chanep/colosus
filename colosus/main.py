import tensorflow as tf

x = tf.constant(5)
y = tf.constant(6)

z = x * y

with tf.Session() as sess:
    result = sess.run(z)

print("result: {}".format(result))
