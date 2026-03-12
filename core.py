import numpy as np
from objetos import normalize


class Ray:
    """representa um raio com origem e direção."""
    def __init__(self, origin, direction):
        self.origin = np.array(origin, dtype=float)
        self.direction = normalize(np.array(direction, dtype=float))


EPSILON = 1e-4
