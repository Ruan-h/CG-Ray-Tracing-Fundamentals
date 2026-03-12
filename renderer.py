import numpy as np
from core import Ray 
from shading import compute_shading

def normalize(v):
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm

def render(width, height, camera_pos, look_at, up_vector, scene, lights, ambient_light, xmin, xmax, ymin, ymax, dist=1.0, projection_type="PERSPECTIVE"):
    image = np.zeros((height, width, 3))
    
    # Vetores da base da câmera (para referência, mas no espaço da câmera a geometria já está alinhada)
    # transformamos a cena inteira com M_wc antes de chamar o render,
    # câmera está na origem (0,0,0) olhando para -Z.
    
    # Pré-cálculo da janela
    window_width = xmax - xmin
    window_height = ymax - ymin
    
    print(f"Renderizando {projection_type}...")
    
    for j in range(height):
        print(f"  Linha {j+1}/{height}")
        for i in range(width):
            # Coordenadas normalizadas [0, 1]
            u_norm = (i + 0.5) / width
            v_norm = (j + 0.5) / height
            
            # Posição do pixel no plano de projeção
            px_x = xmin + u_norm * window_width
            px_y = ymax - v_norm * window_height
            
            # --- DEFINIÇÃO DO RAIO BASEADO NO TIPO DE PROJEÇÃO ---
            
            if projection_type == "PERSPECTIVE":
                # Origem: O olho da câmera (0, 0, 0)
                # Direção: Do olho até o pixel (px_x, px_y, -dist)
                ray_origin = np.array([0.0, 0.0, 0.0])
                ray_dir = np.array([px_x, px_y, -dist])
                ray_dir = ray_dir / np.linalg.norm(ray_dir)
            
            elif projection_type == "ORTHOGRAPHIC":
                # Origem: O próprio pixel no plano (px_x, px_y, 0)
                # Direção: Sempre para frente (-Z), ou seja, (0, 0, -1)
                ray_origin = np.array([px_x, px_y, 0.0]) 
                ray_dir = np.array([0.0, 0.0, -1.0])
                
            elif projection_type == "OBLIQUE":
                # Origem: O pixel (px_x, px_y, 0)
                # Direção: Inclinada. (dx, dy, -1). 
                slant = 0.5 # Fator de cisalhamento
                ray_origin = np.array([px_x, px_y, 0.0])
                ray_dir = np.array([slant, slant, -1.0])
                ray_dir = ray_dir / np.linalg.norm(ray_dir)
            
            # Cria o raio
            ray = Ray(ray_origin, ray_dir)

            # --- INTERSEÇÃO E SHADING ---
            closest_hit = None
            min_t = float('inf')

            for obj in scene:
                hit = obj.intersect(ray)
                if hit is not None and hit.t < min_t:
                    min_t = hit.t
                    closest_hit = hit

            if closest_hit:
                color = compute_shading(closest_hit, ray, lights, ambient_light, scene)
                image[j, i] = np.clip(color, 0.0, 1.0)
            else:
                # Skybox simples
                if projection_type == "PERSPECTIVE":
                    unit_dir = ray.direction
                    t = 0.5 * (unit_dir[1] + 1.0)
                else:
                    # Em ortográfica, o vetor diretor é constante, então o skybox ficarar cor sólida.
                    t = v_norm 
                
                white = np.array([1.0, 1.0, 1.0])
                blue  = np.array([0.5, 0.7, 1.0])
                image[j, i] = (1.0 - t) * white + t * blue

    return image