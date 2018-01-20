class StateConfig:
    def __init__(self):
        self.cpuct = 1.41
        self.value_decay = 0.999


class SearchConfig:
    def __init__(self):
        self.move_count_temp0 = 10


class EvaluatorConfig:
    def __init__(self):
        self.state_config = StateConfig()
        self.search_config = SearchConfig()

