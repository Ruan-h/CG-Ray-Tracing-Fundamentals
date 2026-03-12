import numpy as np
from utils import _apply_point, _apply_vec
# --- FUNÇÕES AUXILIARES ---

def normalize(v):
    """Normaliza um vetor, evitando divisão por zero."""
    v = np.array(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


class HitInfo:
    """Armazena informações sobre uma interseção de raio."""
    def __init__(self, t, point, normal, material, uv=np.array([0,0], dtype=float)):
        self.t = t
        self.point = point
        self.normal = normal
        self.material = material
        self.uv = uv


# --- OBJETOS GEOMÉTRICOS ---


class Sphere:
    def __init__(self, center, radius, material):
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)
        self.material = material

    def transform(self, matrix):
        # 1. Move o centro (como você já fazia)
        self.center = _apply_point(matrix, self.center)

        # 2. Extrai o fator de escala da matriz (Requisito 1.4.3)
        # Pegamos a norma da primeira coluna da matriz 3x3 superior
        # Isso nos diz o quanto o eixo X foi esticado.
        scale_factor = np.linalg.norm(matrix[:3, 0])
        
        # 3. Atualiza o raio
        self.radius *= scale_factor

    def intersect(self, ray):
        oc = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius**2
        discriminant = b*b - 4*a*c

        if discriminant < 0:
            return None

        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2.0*a)
        t2 = (-b + sqrt_disc) / (2.0*a)

        t = -1.0
        if t1 > 1e-4: t = t1
        elif t2 > 1e-4: t = t2
        else: return None

        point = ray.origin + t * ray.direction
        normal = normalize(point - self.center)

        # Mapeamento UV esférico
        phi = np.arctan2(normal[2], normal[0])
        theta = np.arcsin(normal[1])
        u = 1 - (phi + np.pi) / (2 * np.pi)
        v = (theta + np.pi / 2) / np.pi
        
        return HitInfo(t, point, normal, self.material, np.array([u, v]))


class Plane:
    def __init__(self, point, normal, material, u_axis=None, v_axis=None, uv_scale=1.0):
        self.point = np.array(point, dtype=float)
        self.normal = normalize(np.array(normal, dtype=float))
        self.material = material
        self.uv_scale = uv_scale

        # Define eixos para textura se não forem fornecidos
        if u_axis is None or v_axis is None:
            if abs(np.dot(self.normal, np.array([0, 1, 0]))) > 0.9:
                temp_up = np.array([1, 0, 0])
            else:
                temp_up = np.array([0, 1, 0])
            self.u_axis = normalize(np.cross(temp_up, self.normal))
            self.v_axis = normalize(np.cross(self.normal, self.u_axis))
        else:
            self.u_axis = normalize(np.array(u_axis, dtype=float))
            self.v_axis = normalize(np.array(v_axis, dtype=float))

    def transform(self, matrix):
        self.point = _apply_point(matrix, self.point)
        self.normal = normalize(_apply_vec(matrix, self.normal))
        self.u_axis = normalize(_apply_vec(matrix, self.u_axis))
        self.v_axis = normalize(_apply_vec(matrix, self.v_axis))

    def intersect(self, ray):
        denom = np.dot(self.normal, ray.direction)
        if abs(denom) > 1e-6:
            t = np.dot(self.point - ray.origin, self.normal) / denom
            if t > 1e-4:
                point = ray.origin + t * ray.direction
                vec = point - self.point
                u = np.dot(vec, self.u_axis) * self.uv_scale
                v = np.dot(vec, self.v_axis) * self.uv_scale
                uv = np.array([u - np.floor(u), v - np.floor(v)])
                return HitInfo(t, point, self.normal, self.material, uv)
        return None


class Cylinder:
    """Cilindro com opção de tampas."""
    def __init__(self, center_base, axis, radius, height, material, has_bottom=True, has_top=True):
        self.cb = np.array(center_base, dtype=float)
        self.axis = normalize(np.array(axis, dtype=float))
        self.radius = float(radius)
        self.height = float(height)
        self.material = material
        self.has_bottom = has_bottom
        self.has_top = has_top
        
        # Calcula topo
        self.ct = self.cb + self.height * self.axis

    def transform(self, matrix):
        self.cb = _apply_point(matrix, self.cb)
        self.axis = normalize(_apply_vec(matrix, self.axis))
        # Recalcula topo com base na nova posição/orientação
        self.ct = self.cb + self.height * self.axis

    def intersect(self, ray):
        hits = []

        # 1. LATERAL
        oc = ray.origin - self.cb
        rd = ray.direction
        
        oc_proj = np.dot(oc, self.axis)
        rd_proj = np.dot(rd, self.axis)
        
        oc_perp = oc - oc_proj * self.axis
        rd_perp = rd - rd_proj * self.axis
        
        a = np.dot(rd_perp, rd_perp)
        b = 2.0 * np.dot(oc_perp, rd_perp)
        c = np.dot(oc_perp, oc_perp) - self.radius**2

        if abs(a) > 1e-6:
            discriminant = b**2 - 4*a*c
            if discriminant >= 0:
                sqrt_disc = np.sqrt(discriminant)
                t_options = [(-b - sqrt_disc)/(2*a), (-b + sqrt_disc)/(2*a)]
                
                for t in t_options:
                    if t > 1e-4:
                        p = ray.origin + t * rd
                        h = np.dot(p - self.cb, self.axis)
                        if 0 <= h <= self.height:
                            normal = normalize(p - self.cb - h * self.axis)
                            if np.dot(normal, rd) > 0: normal = -normal # Backface check
                            hits.append((t, p, normal))

        # 2. TAMPAS
        # Base (h=0) e Topo (h=height)
        planes = []
        if self.has_bottom: planes.append((self.cb, -self.axis))
        if self.has_top:    planes.append((self.ct, self.axis))

        for center, normal_plane in planes:
            denom = np.dot(normal_plane, ray.direction)
            if abs(denom) > 1e-6:
                t = np.dot(center - ray.origin, normal_plane) / denom
                if t > 1e-4:
                    p = ray.origin + t * ray.direction
                    if np.linalg.norm(p - center)**2 <= self.radius**2:
                        hits.append((t, p, normal_plane))

        if not hits: return None
        best_t, best_p, best_n = min(hits, key=lambda x: x[0])
        return HitInfo(best_t, best_p, best_n, self.material)


class Cone:
    def __init__(self, center_base, axis, radius, height, material, has_base=True):
        self.cb = np.array(center_base, dtype=float)
        self.axis = normalize(np.array(axis, dtype=float))
        self.radius = float(radius)
        self.height = float(height)
        self.material = material
        self.has_base = has_base
        
        self.tip = self.cb + self.height * self.axis
        self.k = (self.radius / self.height)**2

    def transform(self, matrix):
        self.cb = _apply_point(matrix, self.cb)
        self.axis = normalize(_apply_vec(matrix, self.axis))
        # Recalcula vértice
        self.tip = self.cb + self.height * self.axis

    def intersect(self, ray):
        hits = []

        # 1. LATERAL
        co = ray.origin - self.tip
        rd = ray.direction
        
        dy = np.dot(rd, self.axis)
        oy = np.dot(co, self.axis)
        
        rd_perp = rd - dy * self.axis
        co_perp = co - oy * self.axis
        
        a = np.dot(rd_perp, rd_perp) - self.k * dy**2
        b = 2 * (np.dot(rd_perp, co_perp) - self.k * dy * oy)
        c = np.dot(co_perp, co_perp) - self.k * oy**2
        
        if abs(a) > 1e-6:
            discriminant = b**2 - 4*a*c
            if discriminant >= 0:
                sqrt_disc = np.sqrt(discriminant)
                for t in [(-b - sqrt_disc)/(2*a), (-b + sqrt_disc)/(2*a)]:
                    if t > 1e-4:
                        p = ray.origin + t * rd
                        h = np.dot(p - self.cb, self.axis)
                        if 0 <= h <= self.height:
                            cp = p - self.cb
                            h_p = np.dot(cp, self.axis)
                            axis_pt = self.cb + h_p * self.axis
                            radial = normalize(p - axis_pt)
                            
                            hyp = np.sqrt(self.radius**2 + self.height**2)
                            nr = self.height / hyp
                            nh = self.radius / hyp
                            normal = radial * nr + self.axis * nh
                            
                            if np.dot(normal, rd) > 0: normal = -normal
                            hits.append((t, p, normal))

        # 2. BASE
        if self.has_base:
            normal_base = -self.axis
            denom = np.dot(normal_base, ray.direction)
            if abs(denom) > 1e-6:
                t = np.dot(self.cb - ray.origin, normal_base) / denom
                if t > 1e-4:
                    p = ray.origin + t * ray.direction
                    if np.linalg.norm(p - self.cb)**2 <= self.radius**2:
                        hits.append((t, p, normal_base))

        if not hits: return None
        best_t, best_p, best_n = min(hits, key=lambda x: x[0])
        return HitInfo(best_t, best_p, best_n, self.material)


class Triangle:
    """Triângulo simples."""
    def __init__(self, v0, v1, v2, material, uv0=np.array([0,0]), uv1=np.array([0,1]), uv2=np.array([1,0])):
        self.v0 = np.array(v0, dtype=float)
        self.v1 = np.array(v1, dtype=float)
        self.v2 = np.array(v2, dtype=float)
        self.material = material
        self.uv0 = uv0
        self.uv1 = uv1
        self.uv2 = uv2
        
        self.update_normal()

    def update_normal(self):
        edge1 = self.v1 - self.v0
        edge2 = self.v2 - self.v0
        self.normal = normalize(np.cross(edge1, edge2))

    def transform(self, matrix):
        self.v0 = _apply_point(matrix, self.v0)
        self.v1 = _apply_point(matrix, self.v1)
        self.v2 = _apply_point(matrix, self.v2)
        self.update_normal()

    def intersect(self, ray):
        epsilon = 1e-6
        edge1 = self.v1 - self.v0
        edge2 = self.v2 - self.v0
        pvec = np.cross(ray.direction, edge2)
        det = np.dot(edge1, pvec)

        if abs(det) < epsilon: return None

        inv_det = 1.0 / det
        tvec = ray.origin - self.v0
        u = np.dot(tvec, pvec) * inv_det

        if u < 0.0 or u > 1.0: return None

        qvec = np.cross(tvec, edge1)
        v = np.dot(ray.direction, qvec) * inv_det

        if v < 0.0 or u + v > 1.0: return None

        t = np.dot(edge2, qvec) * inv_det

        if t > 1e-4:
            point = ray.origin + t * ray.direction
            # Interpola UV
            w = 1.0 - u - v
            uv = w * self.uv0 + u * self.uv1 + v * self.uv2
            return HitInfo(t, point, self.normal, self.material, uv)
        return None


class Mesh:
    """Agrupamento de triângulos."""
    def __init__(self, vertices, indices, material):
        self.vertices = np.array(vertices, dtype=float)
        self.triangles = []
        
        for idx in indices:
            v0 = self.vertices[idx[0]]
            v1 = self.vertices[idx[1]]
            v2 = self.vertices[idx[2]]
            tri = Triangle(v0, v1, v2, material)
            self.triangles.append(tri)

    def transform(self, matrix):
        # Transforma a lista de vértices original (opcional, mas bom para debug)
        new_verts = []
        for v in self.vertices:
            new_verts.append(_apply_point(matrix, v))
        self.vertices = np.array(new_verts)
        
        # Transforma cada triângulo individualmente
        for tri in self.triangles:
            tri.transform(matrix)

    def intersect(self, ray):
        closest_hit = None
        for tri in self.triangles:
            hit = tri.intersect(ray)
            if hit:
                if closest_hit is None or hit.t < closest_hit.t:
                    closest_hit = hit
        return closest_hit


class Instance:
    """Transformação de objeto (Instância)."""
    def __init__(self, object_ref, matrix):
        self.object = object_ref
        self.matrix = np.array(matrix, dtype=float)
        self._update_inverses()

    def _update_inverses(self):
        self.inv_matrix = np.linalg.inv(self.matrix)
        self.inv_transpose = self.inv_matrix.T 

    def transform(self, matrix):
        # Acumula a transformação
        self.matrix = matrix @ self.matrix
        self._update_inverses()

    def intersect(self, ray):
        # 1. Mundo -> Local
        orig_w = np.append(ray.origin, 1.0)
        orig_obj = self.inv_matrix @ orig_w
        
        dir_w = np.append(ray.direction, 0.0)
        dir_obj = self.inv_matrix @ dir_w
        
        # Normaliza direção local
        dir_obj_norm = normalize(dir_obj[0:3])
        
        # Cria raio local
        class RayLocal:
            def __init__(self, o, d):
                self.origin = o
                self.direction = d
        
        ray_local = RayLocal(orig_obj[0:3], dir_obj_norm)
        
        # 2. Intersect
        hit = self.object.intersect(ray_local)
        
        # 3. Local -> Mundo
        if hit:
            p_h = np.append(hit.point, 1.0)
            point_w = self.matrix @ p_h
            
            n_h = np.append(hit.normal, 0.0)
            normal_w = self.inv_transpose @ n_h
            
            hit.point = point_w[0:3]
            hit.normal = normalize(normal_w[0:3])
            
            # Recalcula T correto no mundo
            hit.t = np.linalg.norm(hit.point - ray.origin)
            return hit
            
        return None