import numpy as np

class PointLight:
    def __init__(self, position, intensity):
        self.position = np.array(position)
        self.intensity = np.array(intensity)
        self.type = "POINT"


class DirectionalLight:
    def __init__(self, direction, intensity):
        self.direction = np.array(direction) / np.linalg.norm(direction)
        self.intensity = np.array(intensity)
        self.type = "DIRECTIONAL"


class SpotLight:
    def __init__(self, position, direction, angle_degrees, intensity):
        self.position = np.array(position)
        self.direction = np.array(direction) / np.linalg.norm(direction)
        self.cutoff_cos = np.cos(np.deg2rad(angle_degrees)) # Cosseno do ângulo de abertura
        self.intensity = np.array(intensity)
        self.type = "SPOT"