import glm
import math
import random

def get_onb(normal: glm.vec3) -> tuple[glm.vec3, glm.vec3, glm.vec3]:
    N = glm.normalize(normal)
    # Find perpendicular vector
    A = glm.vec3(1.0, 0.0, 0.0) if abs(N.x) < 0.9 else glm.vec3(0.0, 0.0, 1.0)
    U = glm.normalize(glm.cross(A, N))
    V = glm.cross(N, U)
    return U, V, N

class Material:
    def __init__(self, albedo: glm.vec3 = glm.vec3(0.8), emission: glm.vec3 = glm.vec3(0.0)):
        self.albedo = glm.vec3(albedo)
        self.emission = glm.vec3(emission)

    def eval(self, wo: glm.vec3, wi: glm.vec3, normal: glm.vec3) -> glm.vec3:
        raise NotImplementedError

    def sample(self, wo: glm.vec3, normal: glm.vec3) -> tuple[glm.vec3, float]:
        # Returns: (wi, pdf) or (None, 0.0) if sampling fails
        raise NotImplementedError

    def pdf(self, wo: glm.vec3, wi: glm.vec3, normal: glm.vec3) -> float:
        raise NotImplementedError

class Lambertian(Material):
    def eval(self, wo: glm.vec3, wi: glm.vec3, normal: glm.vec3) -> glm.vec3:
        n_dot_wi = glm.dot(normal, wi)
        n_dot_wo = glm.dot(normal, wo)
        if n_dot_wi <= 0.0 or n_dot_wo <= 0.0:
            return glm.vec3(0.0)
        return self.albedo / math.pi

    def sample(self, wo: glm.vec3, normal: glm.vec3) -> tuple[glm.vec3, float]:
        r1 = random.random()
        r2 = random.random()
        
        # Cosine-weighted hemisphere sampling
        phi = 2.0 * math.pi * r1
        z = math.sqrt(r2)
        r = math.sqrt(1.0 - r2)
        
        x = r * math.cos(phi)
        y = r * math.sin(phi)
        
        U, V, N = get_onb(normal)
        wi = x * U + y * V + z * N
        wi = glm.normalize(wi)
        
        pdf_val = self.pdf(wo, wi, normal)
        return wi, pdf_val

    def pdf(self, wo: glm.vec3, wi: glm.vec3, normal: glm.vec3) -> float:
        n_dot_wi = glm.dot(normal, wi)
        if n_dot_wi <= 0.0:
            return 0.0
        return n_dot_wi / math.pi

class Emissive(Material):
    def eval(self, wo: glm.vec3, wi: glm.vec3, normal: glm.vec3) -> glm.vec3:
        return glm.vec3(0.0)

    def sample(self, wo: glm.vec3, normal: glm.vec3) -> tuple[glm.vec3, float]:
        return None, 0.0

    def pdf(self, wo: glm.vec3, wi: glm.vec3, normal: glm.vec3) -> float:
        return 0.0

class CookTorrance(Material):
    def __init__(self, albedo: glm.vec3 = glm.vec3(0.8), roughness: float = 0.5, metalness: float = 0.0, emission: glm.vec3 = glm.vec3(0.0)):
        super().__init__(albedo, emission)
        self.roughness = roughness
        self.metalness = metalness

    def _get_alpha(self) -> float:
        # Prevent division by zero with tiny roughness
        return max(self.roughness * self.roughness, 0.001)

    def D_ggx(self, n_dot_h: float) -> float:
        alpha = self._get_alpha()
        a2 = alpha * alpha
        denom = (n_dot_h * n_dot_h * (a2 - 1.0) + 1.0)
        return a2 / (math.pi * denom * denom)

    def F_schlick(self, v_dot_h: float) -> glm.vec3:
        # Dielectric standard F0 is 0.04, metal standard is the albedo
        F0 = glm.mix(glm.vec3(0.04), self.albedo, self.metalness)
        return F0 + (glm.vec3(1.0) - F0) * math.pow(1.0 - v_dot_h, 5.0)

    def G1_schlick_ggx(self, n_dot_x: float) -> float:
        alpha = self._get_alpha()
        k = alpha / 2.0
        return n_dot_x / (n_dot_x * (1.0 - k) + k)

    def G_smith(self, n_dot_v: float, n_dot_l: float) -> float:
        return self.G1_schlick_ggx(n_dot_v) * self.G1_schlick_ggx(n_dot_l)

    def eval(self, wo: glm.vec3, wi: glm.vec3, normal: glm.vec3) -> glm.vec3:
        n_dot_wi = glm.dot(normal, wi)
        n_dot_wo = glm.dot(normal, wo)
        if n_dot_wi <= 0.0 or n_dot_wo <= 0.0:
            return glm.vec3(0.0)

        h = glm.normalize(wo + wi)
        n_dot_h = max(glm.dot(normal, h), 0.0)
        v_dot_h = max(glm.dot(wo, h), 0.0)

        # Cook-Torrance specular term
        D = self.D_ggx(n_dot_h)
        F = self.F_schlick(v_dot_h)
        G = self.G_smith(n_dot_wo, n_dot_wi)
        
        specular_num = D * G * F
        specular_den = 4.0 * n_dot_wo * n_dot_wi
        specular = specular_num / max(specular_den, 1e-8)

        # Diffuse term: energy conservation
        # ks = Fresnel term (specular fraction)
        # kd = 1 - ks, and also 1 - metalness (metals don't have diffuse reflection)
        ks = F
        kd = (glm.vec3(1.0) - ks) * (1.0 - self.metalness)
        
        diffuse = kd * self.albedo / math.pi

        return diffuse + specular

    def sample(self, wo: glm.vec3, normal: glm.vec3) -> tuple[glm.vec3, float]:
        n_dot_wo = glm.dot(normal, wo)
        if n_dot_wo <= 0.0:
            return None, 0.0
            
        r1 = random.random()
        r2 = random.random()
        
        # Specular choice probability (split 50/50 for robust sampling of both lobes)
        p_spec_choice = 0.5
        
        if random.random() < p_spec_choice:
            # Specular: sample GGX distribution of half vectors
            alpha = self._get_alpha()
            phi_h = 2.0 * math.pi * r1
            
            # GGX sampling formulas for theta
            cos_theta_h_2 = (1.0 - r2) / (r2 * (alpha * alpha - 1.0) + 1.0)
            cos_theta_h = math.sqrt(cos_theta_h_2)
            sin_theta_h = math.sqrt(max(0.0, 1.0 - cos_theta_h_2))
            
            x_h = sin_theta_h * math.cos(phi_h)
            y_h = sin_theta_h * math.sin(phi_h)
            z_h = cos_theta_h
            
            U, V, N = get_onb(normal)
            h = x_h * U + y_h * V + z_h * N
            h = glm.normalize(h)
            
            # Reflect outgoing ray to get incoming ray
            wi = glm.reflect(-wo, h)
            wi = glm.normalize(wi)
        else:
            # Diffuse: sample cosine-weighted hemisphere
            phi = 2.0 * math.pi * r1
            z = math.sqrt(r2)
            r = math.sqrt(1.0 - r2)
            
            x = r * math.cos(phi)
            y = r * math.sin(phi)
            
            U, V, N = get_onb(normal)
            wi = x * U + y * V + z * N
            wi = glm.normalize(wi)
            
        pdf_val = self.pdf(wo, wi, normal)
        if pdf_val <= 0.0:
            return None, 0.0
            
        return wi, pdf_val

    def pdf(self, wo: glm.vec3, wi: glm.vec3, normal: glm.vec3) -> float:
        n_dot_wi = glm.dot(normal, wi)
        n_dot_wo = glm.dot(normal, wo)
        if n_dot_wi <= 0.0 or n_dot_wo <= 0.0:
            return 0.0
            
        # Diffuse component PDF
        p_diff = n_dot_wi / math.pi
        
        # Specular component PDF
        h = glm.normalize(wo + wi)
        n_dot_h = glm.dot(normal, h)
        wo_dot_h = glm.dot(wo, h)
        
        if wo_dot_h <= 0.0 or n_dot_h <= 0.0:
            p_spec = 0.0
        else:
            D = self.D_ggx(n_dot_h)
            # Jacobian of reflection mapping (h -> wi)
            p_spec = (D * n_dot_h) / (4.0 * wo_dot_h)
            
        # Linear combination
        return 0.5 * p_diff + 0.5 * p_spec
