import glm
import math
import random
from scene import Ray

class Camera:
    def __init__(self, lookfrom: glm.vec3 = glm.vec3(278, 278, -800), 
                 lookat: glm.vec3 = glm.vec3(278, 278, 278), 
                 vup: glm.vec3 = glm.vec3(0, 1, 0), 
                 vfov_degrees: float = 40.0, 
                 aspect_ratio: float = 1.0):
        self.lookfrom = glm.vec3(lookfrom)
        self.lookat = glm.vec3(lookat)
        self.vup = glm.vec3(vup)
        self.vfov = vfov_degrees
        
        theta = glm.radians(vfov_degrees)
        h = math.tan(theta / 2.0)
        viewport_height = 2.0 * h
        viewport_width = aspect_ratio * viewport_height
        
        # Orthonormal basis vectors for camera frame
        self.w = glm.normalize(self.lookfrom - self.lookat)
        self.u = glm.normalize(glm.cross(self.vup, self.w))
        self.v = glm.cross(self.w, self.u)
        
        # Viewport directions
        self.viewport_u = viewport_width * self.u
        self.viewport_v = viewport_height * self.v
        
        # Bottom-left corner of the viewport
        # Image plane is 1 unit away from eye in direction -w
        self.lower_left_corner = self.lookfrom - self.w - 0.5 * self.viewport_u - 0.5 * self.viewport_v

    def get_ray(self, pixel_x: float, pixel_y: float, width: int, height: int, jitter: bool = True) -> Ray:
        # Offset for anti-aliasing
        offset_x = random.random() if jitter else 0.5
        offset_y = random.random() if jitter else 0.5
        
        s = (pixel_x + offset_x) / width
        t = 1.0 - (pixel_y + offset_y) / height # invert y so 0 is top
        
        pixel_point = self.lower_left_corner + s * self.viewport_u + t * self.viewport_v
        direction = pixel_point - self.lookfrom
        return Ray(self.lookfrom, direction)
