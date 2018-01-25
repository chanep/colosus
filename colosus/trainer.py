import numpy as np

from colosus.colosus_model import ColosusModel
from colosus.config import TrainerConfig
from .game.position import Position
from .state import State
from .searcher import Searcher
from .train_record import TrainRecord
from .train_record_set import TrainRecordSet


class Trainer:
    def __init__(self, config: TrainerConfig):
        self.config = config

    def train(self, train_filename, weights_filename, epochs, prev_weights_filename=None):
        colosus = ColosusModel(self.config.colosus_config)
        colosus.build()
        if prev_weights_filename is not None:
            colosus.load_weights(prev_weights_filename)

        train_record_set = TrainRecordSet.load_from_file(train_filename)
        records = train_record_set.records
        positions = list(map(lambda r: r.position, records))
        policies = list(map(lambda r: r.policy, records))
        values = list(map(lambda r: r.value, records))

        print("training...")
        colosus.train(positions, policies, values, epochs)
        print("training finished!")
        colosus.save_weights(weights_filename)