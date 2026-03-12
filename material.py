import numpy as np
import sys
from PIL import Image


class Material:
    """
    representa as propriedades do material e carregamento de textura.
    """
    def __init__(self, ka, kd, ke, shininess=32.0, texture_path=None, is_procedural_texture=False):
        self.ka = np.array(ka, dtype=float)
        self.kd = np.array(kd, dtype=float)
        self.ke = np.array(ke, dtype=float)
        self.shininess = shininess
        self.is_procedural_texture = is_procedural_texture

        self.texture_data = None
        self.texture_width = 0
        self.texture_height = 0
        self.has_image_texture = False

        if texture_path:
            try:
                img = Image.open(texture_path)
                self.texture_data = np.array(img) / 255.0
                self.texture_width = img.width
                self.texture_height = img.height
                self.has_image_texture = True
                print(f"textura '{texture_path}' carregada ({self.texture_width}x{self.texture_height})")
            except IOError as e:
                print(f"!!! erro ao carregar textura '{texture_path}': {e}")
                print("!!! usando cor sólida (Ka/Kd) como fallback.")

    def get_diffuse_color(self, uv):
            """
            Retorna a cor difusa (Kd) baseada nas coordenadas UV.
            Se tiver textura, busca o pixel. Se não, retorna o Kd base.
            """
            if self.has_image_texture:
                # Garante que UV esteja entre 0 e 1
                u = uv[0] - np.floor(uv[0])
                v = uv[1] - np.floor(uv[1])
                
                # Converte UV para coordenadas de pixel (X, Y)
                x = int(u * (self.texture_width - 1))
                y = int((1 - v) * (self.texture_height - 1))
                
                # Pega a cor do pixel (RGB)
                pixel = self.texture_data[y, x]
                
                # Se a imagem tiver Alpha (RGBA), pega só os 3 primeiros canais
                if len(pixel) > 3:
                    return pixel[:3]
                return pixel
                
            # --- Textura procedural de xadrez ---    
            elif self.is_procedural_texture:
                scale = 4.0
                check = (int(uv[0] * scale) + int(uv[1] * scale)) % 2
                if check == 0:
                    return np.array([0.0, 0.0, 0.0]) # Preto
                else:
                    return np.array([1.0, 1.0, 1.0]) # Branco
                    
            else:
                return self.kd