import numpy as np

from colosus.colosus_model import ColosusModel
from .game.position import Position
from .state import State
from .searcher import Searcher
from .train_record import TrainRecord
from .train_record_set import TrainRecordSet


class Trainer:
    def train(self, train_filename, weights_filename, epochs):
        colosus = ColosusModel()
        colosus.build()

        train_record_set = TrainRecordSet.load_from_file(train_filename)
        records = train_record_set.records
        positions = list(map(lambda r: r.position, records))
        policies = list(map(lambda r: r.policy, records))
        values = list(map(lambda r: r.value, records))

        print("training...")
        colosus.train(positions, policies, values, epochs)
        print("training finished!")
        colosus.model.save_weights(weights_filename)