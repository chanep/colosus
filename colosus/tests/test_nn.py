import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, Activation, Flatten, Dense, Add, Multiply, Lambda, Concatenate
from tensorflow.python.layers.normalization import BatchNormalization
from tensorflow.python.keras.regularizers import l2


def to_rank_file(square):
    return int(square // 8), square % 8


def build_dataset():
    xs = []
    ys = []
    for s1 in range(64):
        s1_r, s1_f = to_rank_file(s1)
        for s2 in range(64):
            if s1 != s2:
                s2_r, s2_f = to_rank_file(s2)
                x = np.zeros((8, 8, 2), np.uint8)
                x[s1_r, s1_f, 0] = 1
                x[s2_r, s2_f, 1] = 1
                y = 1.0 if (s1_r == s2_r or s1_f == s2_f) else 0.0
                xs.append(x)
                ys.append(y)
    return xs, ys



def build_model():
    # reg = l2(1e-4)
    reg = None

    def build_residual_block(x, index):
        in_x = x
        res_name = "res" + str(index)
        x = Conv2D(filters=64, kernel_size=3, padding="same",
                   data_format="channels_last", use_bias=False, kernel_regularizer=reg,
                   name=res_name + "_1")(x)
        x = BatchNormalization(axis=3, name=res_name + "_bn1")(x)
        x = Activation("relu", name=res_name + "_relu1")(x)
        x = Conv2D(filters=64, kernel_size=3, padding="same",
                   data_format="channels_last", use_bias=False, kernel_regularizer=reg,
                   name=res_name + "_2")(x)
        x = BatchNormalization(axis=3, name="res" + str(index) + "_bn2")(x)
        x = Add(name=res_name + "_add")([in_x, x])
        x = Activation("relu", name=res_name + "_relu2")(x)
        return x

    in_x = x = Input((8, 8, 2))

    x = Conv2D(filters=128, kernel_size=3, padding="same",
               data_format="channels_last", use_bias=False, kernel_regularizer=reg,
               name="c1")(x)
    x = BatchNormalization(axis=3, name="bn1")(x)
    x = Activation("relu", name="relu1")(x)

    x = Conv2D(filters=128, kernel_size=5, padding="same",
               data_format="channels_last", use_bias=False, kernel_regularizer=reg,
               name="c2")(x)
    x = BatchNormalization(axis=3, name="bn2")(x)
    x = Activation("relu", name="relu2")(x)
    #
    # x = Conv2D(filters=128, kernel_size=3, padding="same",
    #            data_format="channels_last", use_bias=False, kernel_regularizer=reg,
    #            name="c3")(x)
    # x = BatchNormalization(axis=3, name="bn3")(x)
    # x = Activation("relu", name="relu3")(x)

    for i in range(0):
        x = build_residual_block(x, i + 1)

    res_out = x
    x = Flatten(name="flatten")(x)
    # x = Dense(512, kernel_regularizer=reg, activation="relu", name="value_dense")(x)
    y = Dense(1, kernel_regularizer=reg, activation="sigmoid", name="y")(x)

    model = Model(in_x, y, name="model")

    # opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # opt = tf.keras.optimizers.SGD(lr=0.1, decay=0.0, momentum=0.9, nesterov=False)
    opt = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    # opt = tf.keras.optimizers.Adamax(lr=0.002)
    # opt = tf.keras.optimizers.Nadam(lr=0.0001)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

    return model


def train(xs, ys, model, epochs):
    xs = np.stack(xs)

    ys = np.array(ys)

    model.fit(xs, ys,
              batch_size=32,
              epochs=epochs,
              shuffle=True,
              validation_split=0.0,
              verbose=2,
              callbacks=None)


def predict(s1_r, s1_f, s2_r, s2_f, model):
    x = np.zeros((8, 8, 2), np.uint8)
    x[s1_r, s1_f, 0] = 1
    x[s2_r, s2_f, 1] = 1

    x_in = np.stack([x])

    output = model.predict_on_batch(x_in)
    return output[0][0]


xs, ys = build_dataset()

# for i in range(20):
#     index = np.random.random_integers(0, len(ys) - 1)
#     xt = np.transpose(xs[index], [2, 0, 1])
#     print(ys[index])
#     print(xt[0] + xt[1])

model = build_model()
train(xs, ys, model, 10)

out = predict(3, 3, 4, 4, model)
print(out)
out = predict(3, 3, 3, 4, model)
print(out)
out = predict(1, 6, 5, 6, model)
print(out)




