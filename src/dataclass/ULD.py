import numpy as np


# Define the ULD class
class ULD:
    def __init__(
        self, id: str, length: int, width: int, height: int, weight_limit: int
    ):
        self.id = id
        self.dimensions = np.array([length, width, height])
        self.weight_limit = weight_limit
        self.current_weight = 0
