class StateConfig:
    def __init__(self):
        self.cpuct = 1.41
        self.noise_alpha = 0.3
        self.noise_factor = 0.25


class SearchConfig:
    def __init__(self):
        self.move_count_temp0 = 10


class SelfPlayConfig:
    def __init__(self):
        self.state_config = StateConfig()
        self.search_config = SearchConfig()

class EvaluatorConfig:
    def __init__(self):
        self.state_config = StateConfig()
        self.search_config = SearchConfig()
        self.state_config.noise_factor = 0.0

