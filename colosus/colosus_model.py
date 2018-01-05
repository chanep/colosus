import numpy as np
from tensorflow.python.keras import Input
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, Activation, Flatten, Dense, Add
from tensorflow.python.layers.normalization import BatchNormalization
from tensorflow.python.keras.regularizers import l2

class ColosusModel:
    def __init__(self):
        self.model = None  # type: Model

    def build(self):
        in_x = x = Input((8, 8, 4))

        # (batch, channels, height, width)
        x = Conv2D(filters=256, kernel_size=5, padding="same",
                   data_format="channels_last", use_bias=False, kernel_regularizer=l2(1e-4),
                   name="input_conv-ini")(x)
        x = BatchNormalization(axis=3, name="input_batchnorm")(x)
        x = Activation("relu", name="input_relu")(x)

        # for i in range(2):
        #     x = self._build_residual_block(x, i + 1)

        res_out = x

        # for policy output
        x = Conv2D(filters=2, kernel_size=1, data_format="channels_last", use_bias=False,
                   kernel_regularizer=l2(1e-4),
                   name="policy_conv-1-2")(res_out)
        x = BatchNormalization(axis=3, name="policy_batchnorm")(x)
        x = Activation("relu", name="policy_relu")(x)
        x = Flatten(name="policy_flatten")(x)
        # no output for 'pass'
        policy_out = Dense(64 * 64, kernel_regularizer=l2(1e-4), activation="softmax",
                           name="policy_out")(x)

        # for value output
        x = Conv2D(filters=4, kernel_size=1, data_format="channels_last", use_bias=False,
                   kernel_regularizer=l2(1e-4),
                   name="value_conv-1-4")(res_out)
        x = BatchNormalization(axis=3, name="value_batchnorm")(x)
        x = Activation("relu", name="value_relu")(x)
        x = Flatten(name="value_flatten")(x)
        x = Dense(256, kernel_regularizer=l2(1e-4), activation="relu", name="value_dense")(x)
        value_out = Dense(1, kernel_regularizer=l2(1e-4), activation="tanh", name="value_out")(x)

        self.model = Model(in_x, [policy_out, value_out], name="colosus_model")

        opt = Adam()
        losses = ['categorical_crossentropy', 'mean_squared_error']  # avoid overfit for supervised
        self.model.compile(optimizer=opt, loss=losses, loss_weights=[1.25, 1.0])

    def _build_residual_block(self, x, index):
        mc = self.config.model
        in_x = x
        res_name = "res" + str(index)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_last", use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                   name=res_name + "_conv1-" + str(mc.cnn_filter_size) + "-" + str(mc.cnn_filter_num))(x)
        x = BatchNormalization(axis=3, name=res_name + "_batchnorm1")(x)
        x = Activation("relu", name=res_name + "_relu1")(x)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_last", use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                   name=res_name + "_conv2-" + str(mc.cnn_filter_size) + "-" + str(mc.cnn_filter_num))(x)
        x = BatchNormalization(axis=3, name="res" + str(index) + "_batchnorm2")(x)
        x = Add(name=res_name + "_add")([in_x, x])
        x = Activation("relu", name=res_name + "_relu2")(x)
        return x

    def predict(self, board) -> (float, np.ndarray):
        board_t = np.transpose(board, [1, 2, 0])
        input = np.expand_dims(board_t, axis=0)
        output = self.model.predict_on_batch(input)
        value = output[1][0][0]
        policy = output[0][0]
        return value, policy

    def train(self, boards, policies, values):
        self.model.fit(boards, [policies, values],
                             batch_size=32,
                             epochs=100,
                             shuffle=True,
                             validation_split=0,
                             callbacks=None)

