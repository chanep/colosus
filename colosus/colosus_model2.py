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
    pass