import unittest
import cProfile, pstats, io

import numpy as np
import time

from colosus.colosus_model import ColosusModel
from colosus.config import SelfPlayConfig, ColosusConfig, SelfPlayMpConfig, TrainerConfig
from colosus.game.square import Square
from colosus.self_play import SelfPlay
from colosus.game.position import Position
from colosus.game.side import Side
from colosus.self_play_mp import SelfPlayMp
from colosus.train_record_set import TrainRecordSet
from colosus.trainer import Trainer


class GenerationsTestCase(unittest.TestCase):

    def test_generations(self):
        pos_ini = Position()
        prev_weights_filename = "c_11_800_1600.h5"
        file_prefix = "cg"
        games_per_gen = 200
        iterations = 1600
        gen_count = 4
        gen_ini = 12
        epochs = 6

        trainer_config = TrainerConfig()
        trainer_config.colosus_config.lr = 0.0001
        trainer = Trainer(trainer_config)

        self_play = SelfPlayMp(SelfPlayMpConfig())

        start_time = time.time()
        for g in range(gen_count):
            pos = pos_ini.clone()
            print(f"Self playing gen {g}...")
            recordset_filename = self.get_filename(file_prefix, gen_ini, g, games_per_gen, iterations, "dat")
            self_play.play(games_per_gen, iterations, pos, recordset_filename, 16, prev_weights_filename)
            print(f"Self play finished gen {g}!")

            TrainRecordSet.merge_and_rotate(recordset_filename, 16)
            print(f"record sets merged gen {g}!")

            print(f"Training gen {g}...")
            weights_filename = self.get_filename(file_prefix, gen_ini, g, games_per_gen, iterations, "h5")
            trainer.train(recordset_filename, weights_filename, epochs, prev_weights_filename)
            print(f"Training finished gen {g}!")

            prev_weights_filename = weights_filename

        print("fin. time: " + str(time.time() - start_time))

    def get_filename(self, prefix, gen_ini, gen, games_per_gen, iterations, extension):
        return f"{prefix}_{gen_ini + gen}_{games_per_gen}_{iterations}.{extension}"


if __name__ == '__main__':
    unittest.main()