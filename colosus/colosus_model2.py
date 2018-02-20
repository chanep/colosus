from colosus.colosus_model import ColosusModel
from typing import List

import numpy as np
import tensorflow as tf
from threading import Lock
from tensorflow.python.keras import Input
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, Activation, Flatten, Dense, Add, Multiply, Lambda, Concatenate
from tensorflow.python.layers.normalization import BatchNormalization
from tensorflow.python.keras.regularizers import l2

from colosus.config import ColosusConfig
from colosus.game import model_position
from colosus.game.model_position import ModelPosition
from colosus.game.position import Position


class ColosusModel2(ColosusModel):
    def build(self):
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.reg = l2(1e-5)
            # self.reg = None
            self.conv_size = 80
            data_format = "channels_last" if self.config.data_format_channel_last else "channels_first"
            bn_axis = 3 if self.config.data_format_channel_last else 1

            if self.config.data_format_channel_last:
                in_x = x = Input((self.B_SIZE, self.B_SIZE, 2))
            else:
                in_x = x = Input((2, self.B_SIZE, self.B_SIZE))

            # (batch, channels, height, width)
            x = Conv2D(filters=self.conv_size, kernel_size=4, padding="same",
                       data_format=data_format, use_bias=False, kernel_regularizer=self.reg,
                       name="input_conv-ini")(x)

            # x2 = Conv2D(filters=self.conv_size, kernel_size=(4, 1), padding="same", data_format=data_format,
            #             use_bias=False, kernel_regularizer=self.reg,
            #             name="input_conv-ini2")(x)
            #
            # x3 = Conv2D(filters=self.conv_size, kernel_size=(1, 4), padding="same", data_format=data_format,
            #             use_bias=False, kernel_regularizer=self.reg,
            #             name="input_conv-ini3")(x)
            #
            # x = Add(name="in_conv_add")([x, x2, x3])

            x = BatchNormalization(axis=bn_axis, name="input_batchnorm")(x)
            x = Activation("relu", name="input_relu")(x)

            for i in range(2):
                x = self._build_residual_block(x, i + 1)

            res_out = x

            # for policy output
            x = Conv2D(filters=16, kernel_size=1, padding="same", data_format=data_format, use_bias=False,
                       kernel_regularizer=self.reg,
                       name="policy_conv-1-2")(res_out)
            x = BatchNormalization(axis=bn_axis, name="policy_batchnorm")(x)
            x = Activation("relu", name="policy_relu")(x)
            x = Flatten(name="policy_flatten")(x)
            policy_out = Dense(self.B_SIZE * self.B_SIZE, kernel_regularizer=self.reg, activation="softmax",
                               name="policy_out")(x)

            # for value output
            x = Conv2D(filters=4, kernel_size=1, data_format=data_format, use_bias=False,
                       kernel_regularizer=self.reg,
                       name="value_conv-1-4")(res_out)
            x = BatchNormalization(axis=bn_axis, name="value_batchnorm")(x)
            x = Activation("relu", name="value_relu")(x)
            x = Flatten(name="value_flatten")(x)
            x = Dense(256, kernel_regularizer=self.reg, activation="relu", name="value_dense")(x)

            value_out = Dense(1, kernel_regularizer=self.reg, activation="tanh", name="value_out")(x)

            self.model = Model(in_x, [policy_out, value_out], name="colosus_model")

            opt = Adam(lr=self.config.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
            losses = ['categorical_crossentropy', 'mean_squared_error']  # avoid overfit for supervised

            self.model.compile(optimizer=opt, loss=losses, loss_weights=[1.25, 1.0])
            self.model._make_predict_function()  # for multithread