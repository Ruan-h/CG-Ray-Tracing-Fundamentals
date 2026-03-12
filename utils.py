import numpy as np

def normalize(v):
    """Normaliza um vetor."""
    v = np.array(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def lookat_matrix(eye, at, up):
    """Gera a matriz de transformação World -> Camera."""
    forward = (at - eye)
    forward = normalize(forward) # Eixo -Z da câmera
    
 
    z_axis = -forward 
    
    right = np.cross(up, z_axis)
    right = normalize(right) # Eixo X da câmera
    
    true_up = np.cross(z_axis, right) # Eixo Y da câmera
    
    # Matriz 4x4
    mat = np.eye(4)
    mat[0, :3] = right
    mat[1, :3] = true_up
    mat[2, :3] = z_axis
    mat[:3, 3] = -mat[:3, :3] @ eye
    return mat

def transform_point(p, matrix):
    """Aplica matriz 4x4 em ponto 3D (w=1)."""
    p_h = np.array([p[0], p[1], p[2], 1.0])
    return (matrix @ p_h)[:3]

def transform_vec(v, matrix):
    """Aplica matriz 4x4 em vetor direção 3D (w=0)."""
    v_h = np.array([v[0], v[1], v[2], 0.0])
    return (matrix @ v_h)[:3]


def _apply_point(mat, p):
    return (mat @ np.append(p, 1.0))[:3]

def _apply_vec(mat, v):
    return (mat @ np.append(v, 0.0))[:3]


def shear_matrix(h_xy=0, h_xz=0, h_yx=0, h_yz=0, h_zx=0, h_zy=0):
    """
    Matriz de Cisalhamento (Shear).
    Deforma o objeto deslizando camadas.
    """
    m = np.eye(4)
    m[0, 1] = h_xy
    m[0, 2] = h_xz
    m[1, 0] = h_yx
    m[1, 2] = h_yz
    m[2, 0] = h_zx
    m[2, 1] = h_zy
    return m

def reflection_matrix(point, normal):
    """
    Matriz de Espelho em plano arbitrário.
    Reflete um objeto através de um plano definido por ponto e normal.
    """
    n = normalize(np.array(normal))
    # Equação do plano: ax + by + cz + d = 0 -> d = -dot(p, n)
    d = -np.dot(point, n)
    
    m = np.eye(4)
    
    # Coluna 0
    m[0,0] = 1 - 2 * n[0] * n[0]
    m[1,0] =   - 2 * n[1] * n[0]
    m[2,0] =   - 2 * n[2] * n[0]
    m[3,0] = 0
    
    # Coluna 1
    m[0,1] =   - 2 * n[0] * n[1]
    m[1,1] = 1 - 2 * n[1] * n[1]
    m[2,1] =   - 2 * n[2] * n[1]
    m[3,1] = 0
    
    # Coluna 2
    m[0,2] =   - 2 * n[0] * n[2]
    m[1,2] =   - 2 * n[1] * n[2]
    m[2,2] = 1 - 2 * n[2] * n[2]
    m[3,2] = 0
    
    # Coluna 3 (Translação da reflexão)
    m[0,3] = - 2 * d * n[0]
    m[1,3] = - 2 * d * n[1]
    m[2,3] = - 2 * d * n[2]
    m[3,3] = 1
    
    return m

def translation_matrix(dx, dy, dz):
    """
    Matriz de Translação.
    Move o objeto somando (dx, dy, dz) às suas coordenadas.
    """
    m = np.eye(4)
    m[0, 3] = dx
    m[1, 3] = dy
    m[2, 3] = dz
    return m

def rotation_quaternion_matrix(axis, angle_rad):
    """
    Gera matriz de rotação a partir de um quatérnio (q = w + xi + yj + zk).
    """
    # 1. Criar o Quatérnio a partir do Eixo e Ângulo
    u = normalize(axis)
    ux, uy, uz = u
    
    # Metade do ângulo
    half_angle = angle_rad / 2.0
    sin_a = np.sin(half_angle)
    
    # Componentes do Quatérnio (w, x, y, z)
    w = np.cos(half_angle)
    x = ux * sin_a
    y = uy * sin_a
    z = uz * sin_a
    
    # 2. Converter Quatérnio para Matriz 4x4
    m = np.eye(4)
    
    # Fórmulas de conversão padrão Quatérnio -> Matriz
    m[0,0] = 1 - 2*(y**2 + z**2)
    m[0,1] = 2*(x*y - z*w)
    m[0,2] = 2*(x*z + y*w)
    
    m[1,0] = 2*(x*y + z*w)
    m[1,1] = 1 - 2*(x**2 + z**2)
    m[1,2] = 2*(y*z - x*w)
    
    m[2,0] = 2*(x*z - y*w)
    m[2,1] = 2*(y*z + x*w)
    m[2,2] = 1 - 2*(x**2 + y**2)
    
    return m

def scale_matrix(sx, sy, sz):
    """
    Matriz de Escala.
    Altera o tamanho do objeto nos eixos X, Y e Z.
    """
    m = np.eye(4)
    m[0, 0] = sx
    m[1, 1] = sy
    m[2, 2] = sz
    return m