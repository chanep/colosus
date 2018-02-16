class ColosusConfig:
    def __init__(self):
        self.thread_safe = False
        self.lr = 0.0005
        self.data_format_channel_last = True
        # self.data_format = "channels_first"


class StateConfig:
    def __init__(self):
        self.cpuct = 1.41
        self.noise_alpha = 1000
        self.noise_factor = 0.25


class SearchConfig:
    def __init__(self):
        self.move_count_temp0 = 16
        self.workers = 4
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
        self.search_config = SearchConfig()


class SelfPlayMpConfig:
    def __init__(self):
        self.colosus_config = ColosusConfig()
        self.state_config = StateConfig()
        self.search_config = SearchConfig()


class EvaluatorConfig:
    def __init__(self):
        self.colosus_config = ColosusConfig()
        self.state_config = StateConfig()
        self.search_config = SearchConfig()
        self.state_config.noise_factor = 0.0
        self.search_config.move_count_temp0 = 8


class PlayerConfig:
    def __init__(self):
        self.colosus_config = ColosusConfig()
        self.state_config = StateConfig()
        self.search_config = SearchConfig()
        self.state_config.noise_factor = 0.0
        self.search_config.move_count_temp0 = 8

