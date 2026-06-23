import glm
import math

class Ray:
    def __init__(self, origin: glm.vec3, direction: glm.vec3):
        self.origin = glm.vec3(origin)
        self.direction = glm.normalize(glm.vec3(direction))

    def at(self, t: float) -> glm.vec3:
        return self.origin + t * self.direction

class HitRecord:
    def __init__(self, t: float = 0.0, p: glm.vec3 = None, normal: glm.vec3 = None, u: float = 0.0, v: float = 0.0, material = None):
        self.t = t
        self.p = p if p is not None else glm.vec3(0)
        self.normal = normal if normal is not None else glm.vec3(0)
        self.u = u
        self.v = v
        self.material = material
        self.front_face = True

    def set_face_normal(self, ray: Ray, outward_normal: glm.vec3):
        self.front_face = glm.dot(ray.direction, outward_normal) < 0.0
        self.normal = outward_normal if self.front_face else -outward_normal

class Sphere:
    def __init__(self, center: glm.vec3, radius: float, material):
        self.center = glm.vec3(center)
        self.radius = radius
        self.material = material

    def intersect(self, ray: Ray, t_min: float, t_max: float) -> HitRecord:
        oc = ray.origin - self.center
        a = glm.dot(ray.direction, ray.direction) # should be 1.0
        b = 2.0 * glm.dot(ray.direction, oc)
        c = glm.dot(oc, oc) - self.radius * self.radius
        discriminant = b * b - 4.0 * a * c
        
        if discriminant < 0.0:
            return None
            
        sqrtd = math.sqrt(discriminant)
        
        # Find the nearest root that lies in the acceptable range
        t = (-b - sqrtd) / (2.0 * a)
        if t <= t_min or t >= t_max:
            t = (-b + sqrtd) / (2.0 * a)
            if t <= t_min or t >= t_max:
                return None
                
        p = ray.at(t)
        outward_normal = (p - self.center) / self.radius
        
        rec = HitRecord(t=t, p=p, material=self.material)
        rec.set_face_normal(ray, outward_normal)
        
        # Simple spherical coordinates mapping for u, v
        phi = math.atan2(outward_normal.z, outward_normal.x)
        theta = math.acos(outward_normal.y)
        rec.u = 1.0 - (phi + math.pi) / (2.0 * math.pi)
        rec.v = theta / math.pi
        
        return rec

class Quad:
    def __init__(self, Q: glm.vec3, u: glm.vec3, v: glm.vec3, material):
        self.Q = glm.vec3(Q)
        self.u = glm.vec3(u)
        self.v = glm.vec3(v)
        self.material = material
        
        # Precompute plane equations and intersection helper variables
        n = glm.cross(self.u, self.v)
        self.normal = glm.normalize(n)
        self.D = glm.dot(self.normal, self.Q)
        self.n_raw = n
        self.denom = glm.dot(n, n)
        
        # Compute area
        self.area = glm.length(n)

    def intersect(self, ray: Ray, t_min: float, t_max: float) -> HitRecord:
        denom_ray = glm.dot(self.normal, ray.direction)
        if abs(denom_ray) < 1e-8:
            return None
            
        t = (self.D - glm.dot(self.normal, ray.origin)) / denom_ray
        if t <= t_min or t >= t_max:
            return None
            
        p = ray.at(t)
        planar_vector = p - self.Q
        
        # Project onto the coordinate axes of the quad
        alpha = glm.dot(glm.cross(planar_vector, self.v), self.n_raw) / self.denom
        beta = glm.dot(glm.cross(self.u, planar_vector), self.n_raw) / self.denom
        
        if alpha < 0.0 or alpha > 1.0 or beta < 0.0 or beta > 1.0:
            return None
            
        rec = HitRecord(t=t, p=p, u=alpha, v=beta, material=self.material)
        rec.set_face_normal(ray, self.normal)
        return rec

    def sample(self) -> tuple[glm.vec3, glm.vec3, float]:
        # Returns: (point, normal, pdf)
        # Random uniform sample on the quad surface
        import random
        r1 = random.random()
        r2 = random.random()
        point = self.Q + r1 * self.u + r2 * self.v
        return point, self.normal, 1.0 / self.area

def create_box(center: glm.vec3, size: glm.vec3, angle_y_degrees: float, material) -> list[Quad]:
    dx_local = glm.vec3(size.x, 0, 0)
    dy_local = glm.vec3(0, size.y, 0)
    dz_local = glm.vec3(0, 0, size.z)
    
    rad = glm.radians(angle_y_degrees)
    cos_t = glm.cos(rad)
    sin_t = glm.sin(rad)
    
    def rotate_y(v):
        return glm.vec3(
            v.x * cos_t + v.z * sin_t,
            v.y,
            -v.x * sin_t + v.z * cos_t
        )
        
    dx = rotate_y(dx_local)
    dy = dy_local
    dz = rotate_y(dz_local)
    
    # Bottom-back-left corner of the box
    min_corner = center - 0.5 * dx - 0.5 * dy - 0.5 * dz
    
    quads = []
    # Front face (+Z'): outward normal points to +dz'
    quads.append(Quad(min_corner + dz, dx, dy, material))
    # Back face (-Z'): outward normal points to -dz'
    quads.append(Quad(min_corner + dx, -dx, dy, material))
    # Right face (+X'): outward normal points to +dx'
    quads.append(Quad(min_corner + dx + dz, -dz, dy, material))
    # Left face (-X'): outward normal points to -dx'
    quads.append(Quad(min_corner, dz, dy, material))
    # Top face (+Y'): outward normal points to +dy'
    quads.append(Quad(min_corner + dy + dz, dx, -dz, material))
    # Bottom face (-Y'): outward normal points to -dy'
    quads.append(Quad(min_corner, dx, dz, material))
    
    return quads

class Scene:
    def __init__(self, ambient_light: glm.vec3 = glm.vec3(0.02)):
        self.objects = []
        self.lights = []
        self.ambient_light = glm.vec3(ambient_light)

    def add(self, obj):
        if isinstance(obj, list):
            for sub_obj in obj:
                self.add(sub_obj)
        else:
            self.objects.append(obj)
            # Check if object is emissive
            if hasattr(obj, 'material') and obj.material is not None:
                if glm.length(obj.material.emission) > 0.0:
                    self.lights.append(obj)

    def intersect(self, ray: Ray, t_min: float = 0.001, t_max: float = float('inf')) -> tuple[HitRecord, any]:
        closest_rec = None
        closest_obj = None
        closest_so_far = t_max
        
        for obj in self.objects:
            rec = obj.intersect(ray, t_min, closest_so_far)
            if rec is not None:
                closest_so_far = rec.t
                closest_rec = rec
                closest_obj = obj
                
        return closest_rec, closest_obj

def build_cornell_box(materials_dict) -> Scene:
    scene = Scene(ambient_light=glm.vec3(0.01))
    
    # Cornell Box dimensions: 0 to 555
    red = materials_dict['red']
    green = materials_dict['green']
    white = materials_dict['white']
    light_mat = materials_dict['light']
    
    # Walls
    # Floor: White
    scene.add(Quad(glm.vec3(0, 0, 0), glm.vec3(555, 0, 0), glm.vec3(0, 0, 555), white))
    # Ceiling: White
    scene.add(Quad(glm.vec3(0, 555, 0), glm.vec3(555, 0, 0), glm.vec3(0, 0, 555), white))
    # Back Wall: White
    scene.add(Quad(glm.vec3(0, 0, 555), glm.vec3(555, 0, 0), glm.vec3(0, 555, 0), white))
    # Left Wall: Red
    scene.add(Quad(glm.vec3(0, 0, 0), glm.vec3(0, 0, 555), glm.vec3(0, 555, 0), red))
    # Right Wall: Green
    scene.add(Quad(glm.vec3(555, 0, 0), glm.vec3(0, 0, 555), glm.vec3(0, 555, 0), green))
    
    # Area Light at Ceiling (smaller than whole ceiling)
    # Positioned at y = 554 (just below ceiling), centered in x and z
    scene.add(Quad(glm.vec3(213, 554, 227), glm.vec3(130, 0, 0), glm.vec3(0, 0, 105), light_mat))
    
    # Objects inside
    # Tall Box: size (165, 330, 165), center of base at (368.5, 0, 351.25) => 3D center at (368.5, 165, 351.25)
    # Rotated 18 degrees around Y axis
    box_mat = materials_dict.get('box', white)
    scene.add(create_box(glm.vec3(368.5, 165, 351.25), glm.vec3(165, 330, 165), 18.0, box_mat))
    
    # Sphere (instead of short box): radius 82, center of base at (186, 0, 168.5) => center at (186, 82, 168.5)
    sphere_mat = materials_dict.get('sphere', white)
    scene.add(Sphere(glm.vec3(186, 82, 168.5), 82.0, sphere_mat))
    
    return scene
