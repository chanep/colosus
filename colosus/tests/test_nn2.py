import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, Activation, Flatten, Dense, Add, Multiply, Lambda, Concatenate
from tensorflow.python.layers.normalization import BatchNormalization
from tensorflow.python.keras.regularizers import l2


def build_model():
    in_x = x = Input((4,))
    # x = Flatten(name="flatten")(x)
    x = Dense(6, activation="relu", name="value_dense")(x)
    y = Dense(1, activation="sigmoid", name="y")(x)

    model = Model(in_x, [y, x], name="model")

    # opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # opt = tf.keras.optimizers.SGD(lr=0.01, decay=0.0, momentum=0.0, nesterov=False)
    # opt = tf.keras.optimizers.SGD(lr=0.01, decay=0.0, momentum=0.0, nesterov=False)
    opt = tf.keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    model.compile(loss=['mean_squared_error', None], optimizer=opt, metrics=['accuracy'])

    return model


def train(xs, ys, model, epochs):
    xs = np.stack(xs)

    ys = np.array(ys)

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs", write_grads=True, write_images=True)

    model.fit(xs, ys,
              batch_size=6,
              epochs=epochs,
              shuffle=True,
              validation_split=0.0,
              verbose=2,
              callbacks=None)


def predict(x, model):
    x_in = np.stack(x)
    output = model.predict_on_batch(x_in)
    return output


xs = []
ys = []

xs = [np.zeros((4,), np.uint8)] * 6
ys = [0.0] * 6

xs[0] = np.array([1, 1, 0, 0])
xs[1] = np.array([1, 0, 1, 0])
xs[2] = np.array([1, 0, 0, 1])
xs[3] = np.array([0, 1, 1, 0])
xs[4] = np.array([0, 1, 0, 1])
xs[5] = np.array([0, 0, 1, 1])
ys = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0]

model = build_model()
train(xs, ys, model, 100)

xs2 = reversed(xs)

out = predict([[0, 1, 1, 0]], model)
# out = predict(xs2, model)
print(out[0])
print(out[1][0])
# out = predict(3, 3, 3, 4, model)
# print(out)
# out = predict(1, 6, 4, 6, model)
# print(out)