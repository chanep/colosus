from enum import Enum

class ColosusConfig:
    def __init__(self):
        self.thread_safe = False
        self.lr = 0.0005
        self.conv_size = 120
        self.policy_conv_size = 32
        self.residual_blocks = 6
        self.regularizer = 2e-5
        self.data_format_channel_last = True
        self.half_memory = False


class StateConfig:
    def __init__(self):
        self.cpuct = 2
        self.noise_alpha = 0.3
        self.noise_factor = 0.25
        self.fpuRoot = 0.5
        self.backup_factor = 0.999
        self.policy_offset = -1


class SearchConfig:
    def __init__(self):
        self.cpuct = 2
        self.noise_alpha = 0.3
        self.noise_factor = 0.25
        self.fpuRoot = 0.5
        self.move_count_temp0 = 44
        self.temp0 = 1.0
        self.tempf = 0.1
        self.mb_size = 64
        self.max_collisions = 16
        self.smart_pruning_factor = 1


class TrainerConfig:
    def __init__(self):
        self.colosus_config = ColosusConfig()


class SelfPlayConfig:
    def __init__(self):
        self.z_factor = 0
        self.colosus_config = ColosusConfig()
        self.state_config = StateConfig()
        self.search_config = SearchConfig()
        self.search_config.cpuct = 3
        self.search_config.mb_size = 16
        self.search_config.max_collisions = 1
        self.search_config.smart_pruning_factor = 0


class PlayerType(Enum):
    player = 1
    player2 = 2
    player_mb = 3
    player_mb2 = 4


class EvaluatorConfig:
    def __init__(self):
        self.colosus_config = ColosusConfig()
        self.colosus2_config = ColosusConfig()
        self.player_config = PlayerConfig()
        self.player2_config = PlayerConfig()
        self.player_config.search_config.move_count_temp0 = 38
        self.player2_config.search_config.move_count_temp0 = 38
        self.player_config.search_config.temp0 = 0.75
        self.player2_config.search_config.temp0 = 0.75
        self.player_type = PlayerType.player_mb
        self.player2_type = PlayerType.player2_mb


class PlayerConfig:
    def __init__(self):
        self.state_config = StateConfig()
        self.search_config = SearchConfig()
        self.state_config.noise_factor = 0.0
        self.search_config.move_count_temp0 = 25
        self.search_config.temp0 = 0.5
        self.search_config.tempf = 0.2


class MatchConfig:
    def __init__(self):
        self.weights_filename = "./colosus/tests/e_14_2000_800.h5"
        self.colosus_config = ColosusConfig()
        self.player_config = PlayerConfig()



