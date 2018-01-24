import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, Activation, Flatten, Dense, Add, Multiply, Lambda, Concatenate
from tensorflow.python.layers.normalization import BatchNormalization
from tensorflow.python.keras.regularizers import l2

from colosus.game import model_position
from colosus.game.model_position import ModelPosition
from colosus.game.position import Position


class ColosusModel:
    B_SIZE = Position.B_SIZE

    def __init__(self):
        self.model = None  # type: Model
        self.reg = None
        self.graph = None
        self.session = None

    def __del__(self):
        if self.session is not None:
            self.session.close()

    def build(self):
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)
        self.session.as_default()

        self.reg = l2(1e-4)
        # reg = None

        in_x = x = Input((self.B_SIZE, self.B_SIZE, 2))

        # (batch, channels, height, width)
        x = Conv2D(filters=256, kernel_size=6, padding="same",
                   data_format="channels_last", use_bias=False, kernel_regularizer=self.reg,
                   name="input_conv-ini")(x)
        x = BatchNormalization(axis=3, name="input_batchnorm")(x)
        x = Activation("relu", name="input_relu")(x)

        # for i in range(1):
        #     x = self._build_residual_block(x, i + 1)

        res_out = x

        # for policy output
        x = Conv2D(filters=4, kernel_size=1, data_format="channels_last", use_bias=False,
                   kernel_regularizer=self.reg,
                   name="policy_conv-1-2")(res_out)
        x = BatchNormalization(axis=3, name="policy_batchnorm")(x)
        x = Activation("relu", name="policy_relu")(x)
        x = Flatten(name="policy_flatten")(x)
        policy_out = Dense(self.B_SIZE * self.B_SIZE, kernel_regularizer=self.reg, activation="softmax",
                           name="policy_out")(x)

        # for value output
        x = Conv2D(filters=4, kernel_size=1, data_format="channels_last", use_bias=False,
                   kernel_regularizer=self.reg,
                   name="value_conv-1-4")(res_out)
        x = BatchNormalization(axis=3, name="value_batchnorm")(x)
        x = Activation("relu", name="value_relu")(x)
        x = Flatten(name="value_flatten")(x)
        x = Dense(256, kernel_regularizer=self.reg, activation="relu", name="value_dense")(x)

        value_out = Dense(1, kernel_regularizer=self.reg, activation="tanh", name="value_out")(x)

        self.model = Model(in_x, [policy_out, value_out], name="colosus_model")

        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        losses = ['categorical_crossentropy', 'mean_squared_error']  # avoid overfit for supervised
        self.model.compile(optimizer=opt, loss=losses, loss_weights=[1.25, 1.0])

    def _build_residual_block(self, x, index):
        in_x = x
        res_name = "res" + str(index)
        x = Conv2D(filters=256, kernel_size=3, padding="same",
                   data_format="channels_last", use_bias=False, kernel_regularizer=self.reg,
                   name=res_name + "_conv1-3-256")(x)
        x = BatchNormalization(axis=3, name=res_name + "_batchnorm1")(x)
        x = Activation("relu", name=res_name + "_relu1")(x)
        x = Conv2D(filters=256, kernel_size=3, padding="same",
                   data_format="channels_last", use_bias=False, kernel_regularizer=self.reg,
                   name=res_name + "_conv2-3-256")(x)
        x = BatchNormalization(axis=3, name="res" + str(index) + "_batchnorm2")(x)
        x = Add(name=res_name + "_add")([in_x, x])
        x = Activation("relu", name=res_name + "_relu2")(x)
        return x

    def predict(self, position: ModelPosition) -> (np.ndarray, float):
        self.session.as_default()

        board = self._positions_to_inputs(position)
        output = self.model.predict_on_batch(board)
        value = output[1][0][0]
        policy = output[0][0]
        return policy, value

    def train(self, positions, policies, values, epochs):
        self.session.as_default()

        boards = self._positions_to_inputs(positions)
        policies = np.stack(policies)
        values = np.array(values)

        self.model.fit(boards, [policies, values],
                       batch_size=256,
                       epochs=epochs,
                       shuffle=True,
                       validation_split=0.02,
                       verbose=2,
                       callbacks=None)

    @staticmethod
    def legal_policy(policy, legal_moves):
        legal_policy = np.zeros_like(policy)
        for m in legal_moves:
            legal_policy[m] = policy[m]
        return legal_policy / np.sum(legal_policy)

    def _positions_to_inputs(self, positions):
        if isinstance(positions, list):
            boards = map(lambda p: np.transpose(p.board, [1, 2, 0]), positions)
            boards = np.stack(boards)
            return boards
        else:
            positions = [positions]
            return self._positions_to_inputs(positions)
