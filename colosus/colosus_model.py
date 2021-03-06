import math
from typing import List

import numpy as np
import tensorflow as tf
from threading import Lock

from tensorflow.python.keras.utils import Sequence
from tensorflow.python.keras import Input
from tensorflow.python.keras.optimizers import Adam, SGD
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv2D, Activation, Flatten, Dense, Add, Multiply, Lambda, Concatenate
from tensorflow.python.layers.normalization import BatchNormalization
from tensorflow.python.keras.regularizers import l2

from colosus.config import ColosusConfig
from colosus.game import model_position
from colosus.game.model_position import ModelPosition
from colosus.game.position import Position
from .cyclical import CyclicLR
from .batch_renormalization import BatchRenormalization


class PositionSequence(Sequence):

    def __init__(self, boards, policies, values, batch_size):
        self.boards = boards
        self.policies = policies
        self.values = values
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.boards) / self.batch_size)

    def __getitem__(self, idx):
        # print("idx: " + str(idx))
        batch_boards = self.boards[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_policies = self.policies[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_values = self.values[idx * self.batch_size: (idx + 1) * self.batch_size]

        return batch_boards, [batch_policies, batch_values]

    def on_epoch_end(self):
        # print("on_epoch_end")
        pass


class ColosusModel:
    B_SIZE = Position.B_SIZE

    def __init__(self, config: ColosusConfig):
        self.config = config
        self.model = None  # type: Model
        self.reg = l2(config.regularizer)
        self.graph = None
        self.session = None
        self.lock = Lock()

    def __del__(self):
        if self.session is not None:
            self.session.close()

    def build(self):
        self.graph = tf.Graph()
        session_config = None
        if self.config.half_memory:
            session_config = tf.ConfigProto()
            session_config.gpu_options.per_process_gpu_memory_fraction = 0.30
        # session_config = tf.ConfigProto(log_device_placement=True)
        self.session = tf.Session(graph=self.graph, config=session_config)

        with self.graph.as_default():
            data_format = "channels_last" if self.config.data_format_channel_last else "channels_first"
            bn_axis = -1 if self.config.data_format_channel_last else 1

            if self.config.data_format_channel_last:
                in_x = x = Input((self.B_SIZE, self.B_SIZE, 2))
            else:
                in_x = x = Input((2, self.B_SIZE, self.B_SIZE))

            # (batch, channels, height, width)
            x = Conv2D(filters=self.config.conv_size, kernel_size=4, padding="same",
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
            # x = BatchRenormalization(axis=bn_axis, name="input_batchnorm")(x)
            x = Activation("relu", name="input_relu")(x)

            for i in range(self.config.residual_blocks):
                x = self._build_residual_block(x, i + 1)

            res_out = x

            # for policy output
            x = Conv2D(filters=self.config.policy_conv_size, kernel_size=1, padding="same", data_format=data_format,
                       use_bias=False,
                       kernel_regularizer=self.reg,
                       name="policy_conv-1-2")(res_out)
            x = BatchNormalization(axis=bn_axis, name="policy_batchnorm")(x)
            # x = BatchRenormalization(axis=bn_axis, name="policy_batchnorm")(x)
            x = Activation("relu", name="policy_relu")(x)
            x = Flatten(name="policy_flatten")(x)
            policy_out = Dense(self.B_SIZE * self.B_SIZE, kernel_regularizer=self.reg, activation="softmax",
                               name="policy")(x)

            # for value output
            x = Conv2D(filters=4, kernel_size=1, data_format=data_format, use_bias=False,
                       kernel_regularizer=self.reg,
                       name="value_conv-1-4")(res_out)
            x = BatchNormalization(axis=bn_axis, name="value_batchnorm")(x)
            # x = BatchRenormalization(axis=bn_axis, name="value_batchnorm")(x)
            x = Activation("relu", name="value_relu")(x)
            x = Flatten(name="value_flatten")(x)
            x = Dense(256, kernel_regularizer=self.reg, activation="relu", name="value_dense")(x)

            value_out = Dense(1, kernel_regularizer=self.reg, activation="tanh", name="value")(x)

            self.model = Model(in_x, [policy_out, value_out], name="colosus_model")

            opt = Adam(lr=self.config.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
            # opt2 = SGD(lr=self.config.lr, momentum=0.85, nesterov=True)
            losses = ['categorical_crossentropy', 'mean_squared_error']
            metrics = {"policy": 'acc'}
            # metrics = ['accuracy']

            self.model.compile(optimizer=opt, loss=losses, metrics=metrics, loss_weights=[1.25, 1.0])
            self.model._make_predict_function()  # for multithread

    def _build_residual_block(self, x, index):
        in_x = x
        res_name = "res" + str(index)

        data_format = "channels_last" if self.config.data_format_channel_last else "channels_first"
        bn_axis = 3 if self.config.data_format_channel_last else 1

        x = Conv2D(filters=self.config.conv_size, kernel_size=3, padding="same",
                   data_format=data_format, use_bias=False, kernel_regularizer=self.reg,
                   name=res_name + "_conv1-3-256")(x)
        x = BatchNormalization(axis=bn_axis, name=res_name + "_batchnorm1")(x)
        # x = BatchRenormalization(axis=bn_axis, name=res_name + "_batchnorm1")(x)
        x = Activation("relu", name=res_name + "_relu1")(x)
        x = Conv2D(filters=self.config.conv_size, kernel_size=3, padding="same",
                   data_format=data_format, use_bias=False, kernel_regularizer=self.reg,
                   name=res_name + "_conv2-3-256")(x)
        x = BatchNormalization(axis=bn_axis, name="res" + str(index) + "_batchnorm2")(x)
        # x = BatchRenormalization(axis=bn_axis, name="res" + str(index) + "_batchnorm2")(x)
        x = Add(name=res_name + "_add")([in_x, x])
        x = Activation("relu", name=res_name + "_relu2")(x)
        return x

    def load_weights(self, filename):
        with self.graph.as_default():
            with self.session.as_default():
                self.model.load_weights(filename)

    def save_weights(self, filename):
        with self.graph.as_default():
            with self.session.as_default():
                self.model.save_weights(filename)

    def predict(self, position: ModelPosition) -> (np.ndarray, float):
        board = self._positions_to_inputs(position)

        if self.config.thread_safe:
            self.lock.acquire()

        with self.session.as_default():
            output = self.model.predict_on_batch(board)
        value = output[1][0][0]
        policy = output[0][0]
        # policy = np.random.dirichlet([100] * (self.B_SIZE * self.B_SIZE))
        # value = np.random.normal(scale=0.01)

        if self.config.thread_safe:
            self.lock.release()

        return policy, value

    def predict_on_batch(self, positions: List[ModelPosition]) -> (np.ndarray, np.ndarray):
        board = self._positions_to_inputs(positions)

        if self.config.thread_safe:
            self.lock.acquire()

        with self.session.as_default():
            output = self.model.predict_on_batch(board)
        values = np.squeeze(output[1], axis=1)
        policies = output[0]

        if self.config.thread_safe:
            self.lock.release()

        return policies, values

    def train(self, positions, policies, values, epochs):

        boards = self._positions_to_inputs(positions)
        policies = np.stack(policies)
        values = np.array(values)

        with self.graph.as_default():
            with self.session.as_default():
                self.model.fit(boards, [policies, values],
                               batch_size=384,
                               epochs=epochs,
                               shuffle=True,
                               validation_split=0.02,
                               verbose=2,
                               callbacks=None)

    def train_clr(self, positions, policies, values, epochs, base_lr, max_lr, step_size, mode='triangular'):

        boards = self._positions_to_inputs(positions)
        policies = np.stack(policies)
        values = np.array(values)

        clr = CyclicLR(base_lr=base_lr, max_lr=max_lr, step_size=step_size, mode=mode)

        with self.graph.as_default():
            with self.session.as_default():
                self.model.fit(boards, [policies, values],
                               batch_size=384,
                               epochs=epochs,
                               shuffle=True,
                               validation_split=0.02,
                               verbose=2,
                               callbacks=[clr])

    def train_generator(self, positions, policies, values, epochs):
        batch_size = 512
        validation_split = 0.02

        boards = self._positions_to_inputs(positions)
        policies = np.stack(policies)
        values = np.array(values)

        data_last = len(boards) - 1
        train_last = math.ceil(len(boards) * (1 - validation_split) - 1)
        boards_train = boards[0: train_last]
        policies_train = policies[0: train_last]
        values_train = values[0: train_last]

        boards_val = boards[train_last + 1: data_last]
        policies_val = policies[train_last + 1: data_last]
        values_val = values[train_last + 1: data_last]

        seq = PositionSequence(boards_train, policies_train, values_train, batch_size)
        seq_val = PositionSequence(boards_val, policies_val, values_val, batch_size)

        with self.graph.as_default():
            with self.session.as_default():
                self.model.fit_generator(seq,
                                         epochs=epochs,
                                         shuffle=True,
                                         validation_data=seq_val,  # (boards_val, [policies_val, values_val]),
                                         workers=4,
                                         use_multiprocessing=True,
                                         verbose=2,
                                         callbacks=None)

    def _positions_to_inputs(self, positions):
        if isinstance(positions, list):
            if self.config.data_format_channel_last:
                positions = map(lambda p: np.transpose(p.board, [1, 2, 0]), positions)
            else:
                positions = map(lambda p: p.board, positions)
            boards = np.stack(positions)
            return boards
        else:
            positions = [positions]
            return self._positions_to_inputs(positions)
