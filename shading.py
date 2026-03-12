from utils import normalize
import numpy as np
from core import Ray 

def compute_shading(hit, ray, lights, ambient_light, scene):
    # -----------------------------------------------------------
    # 1. RECUPERAR TEXTURA E COR BASE
    # -----------------------------------------------------------
    uv = getattr(hit, 'uv', np.array([0.0, 0.0]))
    kd = hit.material.get_diffuse_color(uv)
    
    # Usa a cor da textura (kd) também para o ambiente, para a sombra ficar colorida
    current_ka = kd 
    final_color = current_ka * ambient_light

    # -----------------------------------------------------------
    # CÁLCULOS GEOMÉTRICOS
    # -----------------------------------------------------------
    view_dir = normalize(ray.origin - hit.point)
    normal = hit.normal
    if np.dot(view_dir, normal) < 0:
        normal = -normal

    EPSILON = 1e-4
    shadow_origin = hit.point + normal * EPSILON

    # -----------------------------------------------------------
    # 2. LOOP DE LUZES
    # -----------------------------------------------------------
    for light in lights:
        L = None
        dist_to_light = float('inf')
        intensity = light.intensity
        
        # --- CÁLCULO DA DIREÇÃO ---
        if light.type == "POINT":
            vec_to_light = light.position - hit.point
            dist_to_light = np.linalg.norm(vec_to_light)
            L = vec_to_light / dist_to_light
            
        elif light.type == "DIRECTIONAL":
            L = normalize(-light.direction)
            
        elif light.type == "SPOT":
            vec_to_light = light.position - hit.point
            dist_to_light = np.linalg.norm(vec_to_light)
            L = vec_to_light / dist_to_light
            spot_effect = np.dot(-L, light.direction)
            if spot_effect > light.cutoff_cos:
                intensity = intensity
            else:
                intensity = np.array([0.0, 0.0, 0.0])

        # --- TESTE DE SOMBRA
        in_shadow = False
        if np.any(intensity > 0): 
            shadow_ray = Ray(shadow_origin, L)
            
            for obj in scene:
                # Se o objeto testado for o Sol, pulamos ele.
                if getattr(obj, 'name', '') == "Sol Visual":
                    continue 

                shadow_hit = obj.intersect(shadow_ray)
                if shadow_hit and shadow_hit.t < dist_to_light:
                    in_shadow = True
                    break
        
        # --- ILUMINAÇÃO (PHONG) ---
        if not in_shadow:
            # Difusa
            diff = max(np.dot(normal, L), 0.0)
            diffuse = kd * diff * intensity
            
            # Especular
            reflect_dir = normalize(2.0 * np.dot(normal, L) * normal - L)
            spec = max(np.dot(view_dir, reflect_dir), 0.0)
            specular = hit.material.ke * (spec ** hit.material.shininess) * intensity
            
            final_color += diffuse + specular

    return final_color