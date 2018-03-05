class ColosusConfig:
    def __init__(self):
        self.thread_safe = False
        self.lr = 0.0005
        self.conv_size = 80
        self.residual_blocks = 3
        self.regularizer = 2e-5
        self.data_format_channel_last = True


class StateConfig:
    def __init__(self):
        self.cpuct = 1.41
        self.noise_alpha = 0.3
        self.noise_factor = 0.25
        self.backup_factor = 0.9999


class SearchConfig:
    def __init__(self):
        self.move_count_temp0 = 22
        self.workers = 8
        self.mp_cpuct_factor = 0.5
        self.mp_cpuct0 = 0.5
        self.mp_main_worker_id = 1
        self.mp_temp_factor = 0.38


class TrainerConfig:
    def __init__(self):
        self.colosus_config = ColosusConfig()


class SelfPlayConfig:
    def __init__(self):
        self.colosus_config = ColosusConfig()
        self.state_config = StateConfig()
        self.state_config.cpuct = 1.41 * 3
        self.search_config = SearchConfig()


class SelfPlayMpConfig:
    def __init__(self):
        self.colosus_config = ColosusConfig()
        self.state_config = StateConfig()
        self.state_config.cpuct = 1.41 * 3
        self.search_config = SearchConfig()


class EvaluatorConfig:
    def __init__(self):
        self.colosus_config = ColosusConfig()
        self.colosus2_config = ColosusConfig()
        self.player_config = PlayerConfig()
        self.player2_config = PlayerConfig()
        self.player_config.search_config.move_count_temp0 = 16
        self.player2_config.search_config.move_count_temp0 = 16
        self.player2_is_mp = False
        self.iterations_mp_factor = 0.4


class PlayerConfig:
    def __init__(self):
        self.state_config = StateConfig()
        self.search_config = SearchConfig()
        self.state_config.noise_factor = 0.0
        self.search_config.move_count_temp0 = 6


class MatchConfig:
    def __init__(self):
        self.colosus_config = ColosusConfig()
        self.player_config = PlayerConfig()
        self.mp = False



