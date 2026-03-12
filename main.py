import numpy as np
from objetos import Sphere, Plane, Cylinder, Cone, Mesh
from material import Material
from lights import PointLight, DirectionalLight, SpotLight
from renderer import render
from io_utils import save_ppm
from core import Ray 

from utils import (
    lookat_matrix, transform_point, normalize,
    translation_matrix, scale_matrix, 
    rotation_quaternion_matrix, shear_matrix, reflection_matrix
)

def pick_pixel(x, y, width, height, camera_pos, look_at, up_vec, scene, xmin, xmax, ymin, ymax, dist=1.0):
    """
    Simula um clique do mouse na posição pixel (x, y).
    Retorna o nome do objeto atingido e a distância t.
    """
    # 1. Base da Câmera (Mesma do Render)
    w = normalize(camera_pos - look_at)
    u = normalize(np.cross(up_vec, w))
    v = np.cross(w, u)
    
    # 2. Coordenadas do Pixel na Janela
    
    u_norm = (x + 0.5) / width
    v_norm = (y + 0.5) / height
    
    window_width = xmax - xmin
    window_height = ymax - ymin
    
    # Posição no plano de projeção
    px_x = xmin + u_norm * window_width
    px_y = ymax - v_norm * window_height
    
    # 3. Direção do Raio (Perspectiva)
    # Origem = Câmera
    # Alvo = Ponto na janela
    ray_dir = normalize(px_x * u + px_y * v - dist * w)
    ray = Ray(camera_pos, ray_dir)
    
    # 4. Interseção
    min_t = float('inf')
    picked_obj = None
    
    for obj in scene:
        hit = obj.intersect(ray)
        if hit and hit.t < min_t:
            min_t = hit.t
            picked_obj = obj
            
    if picked_obj:
        # Se o objeto tiver .name, usa ele, senão usa o nome da classe
        name = getattr(picked_obj, 'name', type(picked_obj).__name__)
        print(f"[PICK] Pixel ({x}, {y}) -> Atingiu: '{name}' na distância {min_t:.2f}")
        return picked_obj
    else:
        print(f"[PICK] Pixel ({x}, {y}) -> Não atingiu nada (Céu)")
        return None
    
# ==========================================================
# CONSTRUTORES AUXILIARES
# ==========================================================
def create_pyramid_mesh_origin(size, height, material):
    s = size / 2.0
    vertices = [[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0], [0, 0, height]]
    indices = [[0, 2, 1], [0, 3, 2], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]
    mesh = Mesh(vertices, indices, material)
    mesh.name = "Pirâmide de Gizé"
    return mesh

def create_hat_cone_origin(radius, height, material):
    cone = Cone(center_base=[0,0,0], axis=[0,0,1], radius=radius, height=height, material=material)
    cone.name = "Touca Vermelha"
    return cone

def rodar_cenario(cenario_id):
    width, height = 500, 500 
    
    print(f"\n{'='*60}")
    print(f"INICIANDO RENDERIZAÇÃO DO CENÁRIO: {cenario_id}")
    print(f"{'='*60}")

    # --- OFFSET PARA 1º OCTANTE ---
    OFF_X = 100.0
    OFF_Y = 100.0
    
    # --- Configurações Padrão ---
    # Posição padrão (Visto de quina e de cima)
    eye = np.array([24.0 + OFF_X, -14.0 + OFF_Y, 20.0]) 
    at  = np.array([10.0 + OFF_X, 10.0 + OFF_Y, 5.0]) 
    up  = np.array([0.0, 0.0, 1.0])

    # Parâmetros de Projeção
    projection_type = "PERSPECTIVE"
    dist_focal = 1.0        
    fov_degrees = 60.0      
    ortho_height = 20.0     

    # --- Lógica do Seletor ---
    if cenario_id == 1: # Perspectiva Padrão
        print(">> Perspectiva Padrão (Opção 2: Janela via FOV 60)")
        fov_degrees = 60.0

    elif cenario_id == 2: # Zoom Out (Perspectiva)
        print(">> Zoom Out Perspectiva (Opção 2: Aumentar Janela via FOV)")
        fov_degrees = 90.0 

    elif cenario_id == 3: # Zoom In (Perspectiva)
        print(">> Zoom In Perspectiva (Opção 2: Diminuir Janela via FOV)")
        fov_degrees = 30.0 

    elif cenario_id == 4: # 1 Ponto de Fuga
        print(">> Config: 1 Ponto de Fuga (Olhando reto numa face)")
        eye = np.array([10.0 + OFF_X, -40.0 + OFF_Y, 5.0]) 
        at  = np.array([10.0 + OFF_X, 0.0 + OFF_Y, 5.0])

    elif cenario_id == 5: # 2 Pontos de Fuga
        print(">> Config: 2 Pontos de Fuga")
        eye = np.array([40.0 + OFF_X, -40.0 + OFF_Y, 5.0]) 
        at  = np.array([10.0 + OFF_X, 10.0 + OFF_Y, 5.0])

    elif cenario_id == 6: # 3 Pontos de Fuga (Padrão)
        print(">> Config: 3 Pontos de Fuga (Visto de cima/quina)")
        # Mantém eye/at padrões definidos no início
        pass

    elif cenario_id == 7: # Ortográfica Padrão
        print(">> Projeção Ortográfica")
        projection_type = "ORTHOGRAPHIC"
        ortho_height = 20.0
        eye = np.array([40.0 + OFF_X, -40.0 + OFF_Y, 40.0]) 
        at  = np.array([10.0 + OFF_X, 10.0 + OFF_Y, 0.0])

    elif cenario_id == 71: # Ortográfica Zoom In
        print(">> Ortográfica ZOOM IN (Diminuir Janela)")
        projection_type = "ORTHOGRAPHIC"
        ortho_height = 10.0 
        eye = np.array([40.0 + OFF_X, -40.0 + OFF_Y, 40.0]) 
        at  = np.array([10.0 + OFF_X, 10.0 + OFF_Y, 0.0])

    elif cenario_id == 72: # Ortográfica Zoom Out
        print(">> Ortográfica ZOOM OUT (Aumentar Janela)")
        projection_type = "ORTHOGRAPHIC"
        ortho_height = 40.0 
        eye = np.array([40.0 + OFF_X, -40.0 + OFF_Y, 40.0]) 
        at  = np.array([10.0 + OFF_X, 10.0 + OFF_Y, 0.0])

    elif cenario_id == 8: # Oblíqua
        print(">> Projeção Oblíqua")
        projection_type = "OBLIQUE"
        ortho_height = 20.0 
        eye = np.array([10.0 + OFF_X, 0.0 + OFF_Y, 5.0])
        at  = np.array([10.0 + OFF_X, 10.0 + OFF_Y, 5.0])

    # --- CÁLCULO DA JANELA ---
    aspect_ratio = width / height

    if projection_type == "PERSPECTIVE":
        half_height = dist_focal * np.tan(np.deg2rad(fov_degrees / 2))
    else:
        half_height = ortho_height / 2.0

    half_width  = half_height * aspect_ratio
    ymax, ymin = half_height, -half_height
    xmax, xmin = half_width, -half_width
    
    # ==========================================================
    # LUZES E CENA
    # ==========================================================
    sun_pos_original = np.array([-30.0, 400.0, 240.0])
    sun_position = sun_pos_original + np.array([OFF_X, OFF_Y, 0.0])
    raio_sol = 70.0
    posicao_luz_offset = sun_position - np.array([0.0, raio_sol+1.0, 0.0])
    
    lights = []
    lights.append(PointLight(position=posicao_luz_offset, intensity=[1.5, 1.4, 1.0]))
    ambient_light = np.array([0.7, 0.7, 0.6])

    direcao_sol = np.array([40.0, -390.0, -240.0]) 
    direcao_sol = direcao_sol / np.linalg.norm(direcao_sol)
    lights.append(DirectionalLight(direction=direcao_sol, intensity=[0.8, 0.7, 0.5]))

    torch_pos_obj = np.array([6.0 + OFF_X, 8.0 + OFF_Y, 3.0]) 
    target_sandman = np.array([10.0 + OFF_X, 10.0 + OFF_Y, 6.8])
    spot_dir = target_sandman - torch_pos_obj
    spot_dir = spot_dir / np.linalg.norm(spot_dir)
    posicao_luz_offset_spot = torch_pos_obj + (spot_dir * 0.6)

    lights.append(SpotLight(position=posicao_luz_offset_spot,direction=spot_dir, angle_degrees=25.0, intensity=[4.0, 0.1, 0.1]))

    # --- MATERIAIS E OBJETOS ---
    mat_sand = Material(ka=[0.4,0.3,0.1], kd=[0.9,0.7,0.2], ke=[0,0,0], shininess=1, texture_path="sand.jpeg")
    mat_body = Material(
        ka=[0.9, 0.7, 0.2], 
        kd=[0.9, 0.7, 0.2], 
        ke=[0,0,0], 
        shininess=2, 
        texture_path="sand.jpeg"
    )
    mat_orange = Material(ka=[0.9, 0.4, 0.0], kd=[1.0, 0.5, 0.0], ke=[0.1, 0.1, 0.1], shininess=10)
    mat_green  = Material(ka=[0.2, 0.7, 0.2], kd=[0.1, 0.6, 0.1], ke=[0.1, 0.1, 0.1], shininess=10)
    mat_stone  = Material(ka=[0.3,0.25,0.15], kd=[0.7,0.55,0.2], ke=[0.4,0.4,0.2], shininess=32) 
    mat_sun = Material(ka=[1.0, 0.8, 0.0], kd=[1.0, 0.8, 0.0], ke=[0,0,0], shininess=1)
    mat_red_flame = Material(ka=[0.8, 0.0, 0.0], kd=[1.0, 0.1, 0.1], ke=[0.2, 0, 0], shininess=100)
    mat_black_shiny = Material(ka=[0.1, 0.1, 0.1], kd=[0.1, 0.1, 0.1], ke=[0.9, 0.9, 0.9], shininess=100)
    mat_wood = Material(ka=[0.3, 0.2, 0.1], kd=[0.5, 0.35, 0.1], ke=[0,0,0], shininess=1)
    mat_red_fabric = Material(ka=[0.5, 0.1, 0.1], kd=[0.9, 0.1, 0.1], ke=[0,0,0], shininess=1)
    mat_white_fur = Material(ka=[0.7, 0.7, 0.7], kd=[0.9, 0.9, 0.9], ke=[0,0,0], shininess=1)

    scene = []

    # 1. Visual da Tocha
    torch_handle = Cylinder(center_base=[torch_pos_obj[0], torch_pos_obj[1], 0.0], 
                            axis=[0,0,1], radius=0.3, height=3.0, material=mat_wood)
    scene.append(torch_handle)
    
    torch_flame = Sphere(center=torch_pos_obj, radius=0.45, material=mat_red_flame)
    scene.append(torch_flame)

    # 2. Chão
    scene.append(Plane(point=[0,0,0], normal=[0,0,1], material=mat_sand))

    # 3. Sandman
    sandman_parts = []
    # Corpo
    sandman_parts.append(Sphere(center=[0,0,2.0], radius=2.0, material=mat_body))
    sandman_parts.append(Sphere(center=[0,0,4.8], radius=1.4, material=mat_body))
    sandman_parts.append(Sphere(center=[0,0,6.8], radius=1.0, material=mat_body))
    # Nariz
    sandman_parts.append(Cone(center_base=[0.8,0,6.8], axis=[1,0,0], radius=0.25, height=0.8, material=mat_orange))
    # Óculos
    sandman_parts.append(Sphere(center=[0.8, -0.35, 7.0], radius=0.25, material=mat_black_shiny))
    sandman_parts.append(Sphere(center=[0.8, 0.35, 7.0], radius=0.25, material=mat_black_shiny))
    sandman_parts.append(Cylinder(center_base=[1.05, -0.35, 7.0], axis=[0,1,0], radius=0.08, height=0.7, material=mat_black_shiny, has_top=True))
    # Braços
    sandman_parts.append(Cylinder(center_base=[0, -1.2, 5.0], axis=[0.3, -1.0, 0.4], radius=0.25, height=2.2, material=mat_wood))
    sandman_parts.append(Cylinder(center_base=[0, 1.2, 5.0], axis=[0.3, 1.0, 0.4], radius=0.25, height=2.2, material=mat_wood))
    # Touca
    hat_cone = create_hat_cone_origin(radius=1.0, height=2.5, material=mat_red_fabric)
    hat_cone.transform(rotation_quaternion_matrix([0,1,0], np.deg2rad(-25))) 
    hat_cone.transform(translation_matrix(0, 0, 7.7)) 
    sandman_parts.append(hat_cone)
    
    sandman_parts.append(Cylinder(center_base=[0, 0, 7.5], axis=[0,0,1], radius=1.05, height=0.3, material=mat_white_fur, has_top=True))
    sandman_parts.append(Sphere(center=[-1.0, 0, 9.8], radius=0.35, material=mat_white_fur))

    # Posiciona o Sandman
    M_rot_sandman = rotation_quaternion_matrix([0,0,1], np.deg2rad(-50))
    M_pos_sandman = translation_matrix(10.0 + OFF_X, 10.0 + OFF_Y, 0)
    
    #  ROTAÇÃO DOIDA Teste de Quaternion
    # aponta para a direita (1), para trás (-0.5) e para cima (0.5)
    axis_doido = np.array([1.0, -0.5, 0.5])
    axis_doido = axis_doido / np.linalg.norm(axis_doido)
    angle_doido = np.deg2rad(75)

    # 3. Cria a matriz a partir do quaternion
    #M_rot_sandman = rotation_quaternion_matrix(axis_doido, angle_doido)
    for part in sandman_parts:
        part.transform(M_rot_sandman) 
        part.transform(M_pos_sandman) 
        scene.append(part)

    # 4. Cactos (Cisalhamento)
    cx, cy = 13.0 + OFF_X, 8.0 + OFF_Y
    M_shear_wind = shear_matrix(0, 0.2, 0, 0, 0, 0) # Vento
    
    cacto_tronco = Cylinder(center_base=[0, 0, 0], axis=[0,0,1], radius=0.8, height=4.0, material=mat_green, has_top=True)
    cacto_tronco.transform(M_shear_wind)
    cacto_tronco.transform(translation_matrix(cx, cy, 0))
    scene.append(cacto_tronco)

    cacto_braco = Cylinder(center_base=[0, 0, 0], axis=[1,0,0.5], radius=0.4, height=2.5, material=mat_green)
    cacto_braco.transform(M_shear_wind)
    cacto_braco.transform(translation_matrix(cx, cy, 2.0))
    scene.append(cacto_braco)

    # 5. Cacto Reflexo (Espelho)
    plane_point = np.array([cx + 3.0, cy + 3.0, 0]) 
    plane_normal = np.array([-1.0, -0.2, 0.0]) 
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    M_mirror = reflection_matrix(plane_point, plane_normal)

    cacto_reflexo = Cylinder(center_base=[0, 0, 0], axis=[0,0,1], radius=0.8, height=4.0, material=mat_green, has_top=True)
    cacto_reflexo.transform(M_shear_wind)
    cacto_reflexo.transform(translation_matrix(cx, cy, 0))
    cacto_reflexo.transform(M_mirror)
    scene.append(cacto_reflexo)

    braço_reflexo = Cylinder(center_base=[0, 0, 0], axis=[1,0,0.5], radius=0.4, height=2.5, material=mat_green)
    braço_reflexo.transform(M_shear_wind)              
    braço_reflexo.transform(translation_matrix(cx, cy, 2.0)) 
    braço_reflexo.transform(M_mirror)                  
    scene.append(braço_reflexo)

    # 6. Pirâmide
    pyramid = create_pyramid_mesh_origin(size=25, height=22, material=mat_stone)
    pyramid.transform(translation_matrix(0 + OFF_X, 25 + OFF_Y, 0))
    scene.append(pyramid)
    
    # 7. Sol Visual
    sun_visual = Sphere(center=[0,0,0], radius=1.0, material=mat_sun)
    sun_visual.name = "Sol Visual"
    sun_visual.transform(scale_matrix(70, 70, 70))
    sun_visual.transform(translation_matrix(sun_position[0], sun_position[1], sun_position[2]))
    scene.append(sun_visual)

    # ==========================================================
    # TRANSFORMACÕES MUNDO -> CÂMERA
    # ==========================================================
    M_wc = lookat_matrix(eye, at, up)
    # Transforma luzes e objetos para o espaço da câmera
    for l in lights:
        if hasattr(l, 'position'): 
            l.position = transform_point(l.position, M_wc)
        if hasattr(l, 'direction'):
            dir_vec_4 = np.array([l.direction[0], l.direction[1], l.direction[2], 0.0])
            res_vec = M_wc @ dir_vec_4
            new_dir = res_vec[0:3]
            l.direction = new_dir / np.linalg.norm(new_dir)

    for obj in scene:
        obj.transform(M_wc)

    # ==========================================================
    # RENDER E PICK
    # ==========================================================
    img = render(
        width, height, 
        np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0]), 
        scene, lights, ambient_light, 
        xmin, xmax, ymin, ymax,
        dist=dist_focal, projection_type=projection_type
    ) 
    
    filename = f'cenario_final_{cenario_id}.ppm'
    save_ppm(filename, img, width, height)
    print(f"Salvo: {filename}")

    # PICK TEST (Só no centro para validar)
    mid_x, mid_y = width // 2, height // 2
    camera_local_pos = np.array([0.0, 0.0, 0.0])
    camera_local_at  = np.array([0.0, 0.0, -1.0])
    camera_local_up  = np.array([0.0, 1.0, 0.0])
    
    pick_pixel(
        mid_x, mid_y, width, height, 
        camera_local_pos, camera_local_at, camera_local_up,
        scene, xmin, xmax, ymin, ymax, dist=dist_focal
    )

# ==========================================================
# MAIN LOOP
# ==========================================================
def main():
    cenarios_para_rodar = [5]

    for id_teste in cenarios_para_rodar:
        rodar_cenario(id_teste)

if __name__ == '__main__':
    main()