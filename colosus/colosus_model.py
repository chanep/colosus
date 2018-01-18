import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, Activation, Flatten, Dense, Add, Multiply, Lambda, Concatenate
from tensorflow.python.layers.normalization import BatchNormalization
from tensorflow.python.keras.regularizers import l2

from colosus.game import model_position
from colosus.train_record_set import TrainRecordSet


class ColosusModel:
    policy_len = 4096

    def __init__(self):
        self.model = None  # type: Model

    def build(self):
        in_x = x = Input((8, 8, 4))

        move_count_factor = Input((1,))

        # (batch, channels, height, width)
        x = Conv2D(filters=256, kernel_size=2, padding="same",
                   data_format="channels_last", use_bias=False, kernel_regularizer=l2(1e-4),
                   name="input_conv-ini")(x)
        x = BatchNormalization(axis=3, name="input_batchnorm")(x)
        x = Activation("relu", name="input_relu")(x)

        # for i in range(3):
        #     x = self._build_residual_block(x, i + 1)

        res_out = x

        # for policy output
        x = Conv2D(filters=4, kernel_size=1, data_format="channels_last", use_bias=False,
                   kernel_regularizer=l2(1e-4),
                   name="policy_conv-1-2")(res_out)
        x = BatchNormalization(axis=3, name="policy_batchnorm")(x)
        x = Activation("relu", name="policy_relu")(x)
        x = Flatten(name="policy_flatten")(x)
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

        x = Dense(1, kernel_regularizer=l2(1e-4), activation="tanh", name="value_out_womcf")(x)
        value_out = Multiply(name="value_out")([x, move_count_factor])

        self.model = Model([in_x, move_count_factor], [policy_out, value_out], name="colosus_model")

        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        losses = ['categorical_crossentropy', 'mean_squared_error']  # avoid overfit for supervised
        self.model.compile(optimizer=opt, loss=losses, loss_weights=[1.25, 1.0])

    def _build_residual_block(self, x, index):
        in_x = x
        res_name = "res" + str(index)
        x = Conv2D(filters=256, kernel_size=2, padding="same",
                   data_format="channels_last", use_bias=False, kernel_regularizer=l2(1e-4),
                   name=res_name + "_conv1-3-256")(x)
        x = BatchNormalization(axis=3, name=res_name + "_batchnorm1")(x)
        x = Activation("relu", name=res_name + "_relu1")(x)
        x = Conv2D(filters=256, kernel_size=2, padding="same",
                   data_format="channels_last", use_bias=False, kernel_regularizer=l2(1e-4),
                   name=res_name + "_conv2-3-256")(x)
        x = BatchNormalization(axis=3, name="res" + str(index) + "_batchnorm2")(x)
        x = Add(name=res_name + "_add")([in_x, x])
        x = Activation("relu", name=res_name + "_relu2")(x)
        return x

    def predict(self, position: model_position) -> (np.ndarray, float):
        board, move_count_factor = ColosusModel._positions_to_inputs(position)
        output = self.model.predict_on_batch([board, move_count_factor])
        value = output[1][0][0]
        policy = output[0][0]
        # p = np.random.normal(np.ones(ColosusModel.policy_len), 0.01)
        # policy = p / np.sum(p)
        # value = np.random.normal(0.0, 0.01, None)
        return policy, value

    def train(self, positions, policies, values, epochs):
        boards, move_count_factors = ColosusModel._positions_to_inputs(positions)

        policies = np.stack(policies)

        values = np.array(values)

        self.model.fit([boards, move_count_factors], [policies, values],
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

    @staticmethod
    def _positions_to_inputs(positions):
        if isinstance(positions, list):
            boards = map(lambda p: np.transpose(p.board, [1, 2, 0]), positions)
            boards = np.stack(boards)

            move_count_factors = map(lambda p: np.array([1.0]) if p.move_count < 100 else np.array([0.0]), positions)
            move_count_factors = np.stack(move_count_factors)

            return boards, move_count_factors
        else:
            positions = [positions]
            return ColosusModel._positions_to_inputs(positions)

