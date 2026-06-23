import glm
import random
import math
from PIL import Image
from scene import Ray, HitRecord, build_cornell_box
from camera import Camera

def clamp_color(color: glm.vec3) -> glm.vec3:
    return glm.clamp(color, 0.0, 1.0)

def tone_map_reinhard(color: glm.vec3) -> glm.vec3:
    # Reinhard tone mapping
    return color / (color + glm.vec3(1.0))

def gamma_correct(color: glm.vec3) -> glm.vec3:
    # Gamma 2.2
    return glm.pow(color, glm.vec3(1.0 / 2.2))

def to_ldr(color: glm.vec3) -> tuple[int, int, int]:
    mapped = tone_map_reinhard(color)
    gammaed = gamma_correct(mapped)
    r = int(glm.clamp(gammaed.x * 255.99, 0.0, 255.0))
    g = int(glm.clamp(gammaed.y * 255.99, 0.0, 255.0))
    b = int(glm.clamp(gammaed.z * 255.99, 0.0, 255.0))
    return r, g, b

# ----------------- PASSO 1: Path Tracing Básico -----------------
def run_passo_1(width: int, height: int, spp: int, d_max: int, scene, camera) -> Image.Image:
    img = Image.new("RGB", (width, height))
    pixels = img.load()
    
    for y in range(height):
        # Progress indication
        if (y + 1) % 10 == 0 or y == height - 1:
            print(f"Passo 1: Renderizando linha {y+1}/{height} ({(y+1)/height*100:.1f}%)", end="\r")
            
        for x in range(width):
            pixel_color = glm.vec3(0.0)
            
            for s in range(spp):
                ray = camera.get_ray(x, y, width, height, jitter=True)
                pixel_color += trace_path_passo1(ray, scene, d_max)
                
            pixel_color /= spp
            pixels[x, y] = to_ldr(pixel_color)
            
    print() # newline after progress loop
    return img

def trace_path_passo1(ray: Ray, scene, d_max: int) -> glm.vec3:
    beta = glm.vec3(1.0)
    L = glm.vec3(0.0)
    current_ray = ray
    
    for depth in range(d_max):
        rec, obj = scene.intersect(current_ray)
        if rec is None:
            L += beta * scene.ambient_light
            break
            
        # Last path closure rule
        if depth == d_max - 1:
            # If hit emissive, add it
            if glm.length(rec.material.emission) > 0.0:
                L += beta * rec.material.emission
            else:
                # Force sampling of the area light
                if scene.lights:
                    light_obj = random.choice(scene.lights)
                    p_light, n_light, pdf_light = light_obj.sample()
                    
                    d_light = p_light - rec.p
                    r = glm.length(d_light)
                    if r > 1e-4:
                        wi = glm.normalize(d_light)
                        # Shadow ray
                        shadow_ray = Ray(rec.p + rec.normal * 0.001, wi)
                        shadow_rec, shadow_obj = scene.intersect(shadow_ray)
                        
                        # Check visibility
                        if shadow_rec is not None and shadow_obj == light_obj and shadow_rec.t >= r - 0.05:
                            fr = rec.material.eval(-current_ray.direction, wi, rec.normal)
                            cos_hit = max(0.0, glm.dot(wi, rec.normal))
                            cos_light = max(0.0, glm.dot(-wi, n_light))
                            
                            # Add radiance contribution
                            L += beta * fr * light_obj.material.emission * (cos_hit * cos_light * light_obj.area / (r * r))
            break
            
        # If we hit light source directly before depth limit
        if glm.length(rec.material.emission) > 0.0:
            L += beta * rec.material.emission
            break
            
        # Sample BRDF
        wi, pdf_val = rec.material.sample(-current_ray.direction, rec.normal)
        if wi is None or pdf_val <= 0.0:
            break
            
        fr = rec.material.eval(-current_ray.direction, wi, rec.normal)
        cos_theta = max(0.0, glm.dot(wi, rec.normal))
        
        beta *= (fr * cos_theta) / pdf_val
        current_ray = Ray(rec.p + rec.normal * 0.001, wi)
        
    return L

# ----------------- PASSO 2: Roleta Russa -----------------
def run_passo_2(width: int, height: int, spp: int, d_max: int, scene, camera) -> Image.Image:
    img = Image.new("RGB", (width, height))
    pixels = img.load()
    
    for y in range(height):
        if (y + 1) % 10 == 0 or y == height - 1:
            print(f"Passo 2: Renderizando linha {y+1}/{height} ({(y+1)/height*100:.1f}%)", end="\r")
            
        for x in range(width):
            pixel_color = glm.vec3(0.0)
            
            for s in range(spp):
                ray = camera.get_ray(x, y, width, height, jitter=True)
                pixel_color += trace_path_passo2(ray, scene, d_max)
                
            pixel_color /= spp
            pixels[x, y] = to_ldr(pixel_color)
            
    print()
    return img

def trace_path_passo2(ray: Ray, scene, d_max: int) -> glm.vec3:
    beta = glm.vec3(1.0)
    L = glm.vec3(0.0)
    current_ray = ray
    
    for depth in range(d_max):
        # Russian Roulette termination (start at depth 2)
        if depth >= 2:
            q = 0.2 # probability of termination (20%)
            if random.random() < q:
                break
            else:
                beta /= (1.0 - q)
                
        rec, obj = scene.intersect(current_ray)
        if rec is None:
            L += beta * scene.ambient_light
            break
            
        # Last path closure rule
        if depth == d_max - 1:
            if glm.length(rec.material.emission) > 0.0:
                L += beta * rec.material.emission
            else:
                if scene.lights:
                    light_obj = random.choice(scene.lights)
                    p_light, n_light, pdf_light = light_obj.sample()
                    d_light = p_light - rec.p
                    r = glm.length(d_light)
                    if r > 1e-4:
                        wi = glm.normalize(d_light)
                        shadow_ray = Ray(rec.p + rec.normal * 0.001, wi)
                        shadow_rec, shadow_obj = scene.intersect(shadow_ray)
                        if shadow_rec is not None and shadow_obj == light_obj and shadow_rec.t >= r - 0.05:
                            fr = rec.material.eval(-current_ray.direction, wi, rec.normal)
                            cos_hit = max(0.0, glm.dot(wi, rec.normal))
                            cos_light = max(0.0, glm.dot(-wi, n_light))
                            L += beta * fr * light_obj.material.emission * (cos_hit * cos_light * light_obj.area / (r * r))
            break
            
        if glm.length(rec.material.emission) > 0.0:
            L += beta * rec.material.emission
            break
            
        wi, pdf_val = rec.material.sample(-current_ray.direction, rec.normal)
        if wi is None or pdf_val <= 0.0:
            break
            
        fr = rec.material.eval(-current_ray.direction, wi, rec.normal)
        cos_theta = max(0.0, glm.dot(wi, rec.normal))
        
        beta *= (fr * cos_theta) / pdf_val
        current_ray = Ray(rec.p + rec.normal * 0.001, wi)
        
    return L

# ----------------- PASSO 3 & 4: Multiple Importance Sampling (MIS) -----------------
# Passo 3 uses MIS with Lambertian, Passo 4 uses MIS with Cook-Torrance.
# The algorithm is exactly the same, as the materials handle their evaluation/sampling.
def run_passo_3(width: int, height: int, spp: int, d_max: int, scene, camera) -> Image.Image:
    img = Image.new("RGB", (width, height))
    pixels = img.load()
    
    for y in range(height):
        if (y + 1) % 10 == 0 or y == height - 1:
            print(f"Passo 3: Renderizando linha {y+1}/{height} ({(y+1)/height*100:.1f}%)", end="\r")
            
        for x in range(width):
            pixel_color = glm.vec3(0.0)
            
            for s in range(spp):
                ray = camera.get_ray(x, y, width, height, jitter=True)
                pixel_color += trace_path_mis(ray, scene, d_max)
                
            pixel_color /= spp
            pixels[x, y] = to_ldr(pixel_color)
            
    print()
    return img

def run_passo_4(width: int, height: int, spp: int, d_max: int, scene, camera) -> Image.Image:
    img = Image.new("RGB", (width, height))
    pixels = img.load()
    
    for y in range(height):
        if (y + 1) % 10 == 0 or y == height - 1:
            print(f"Passo 4: Renderizando linha {y+1}/{height} ({(y+1)/height*100:.1f}%)", end="\r")
            
        for x in range(width):
            pixel_color = glm.vec3(0.0)
            
            for s in range(spp):
                ray = camera.get_ray(x, y, width, height, jitter=True)
                pixel_color += trace_path_mis(ray, scene, d_max)
                
            pixel_color /= spp
            pixels[x, y] = to_ldr(pixel_color)
            
    print()
    return img

def trace_path_mis(ray: Ray, scene, d_max: int) -> glm.vec3:
    beta = glm.vec3(1.0)
    L = glm.vec3(0.0)
    current_ray = ray
    
    # Track pdf of the last bounce ray (for MIS weight computation when hitting a light)
    last_pdf_brdf = 1.0
    was_specular_bounce = True # Set to true initially so we don't weight the first hit (camera ray)
    
    for depth in range(d_max):
        # Russian Roulette termination (start at depth 2)
        if depth >= 2:
            q = 0.2
            if random.random() < q:
                break
            else:
                beta /= (1.0 - q)
                
        rec, obj = scene.intersect(current_ray)
        if rec is None:
            L += beta * scene.ambient_light
            break
            
        # If we hit light source directly
        if glm.length(rec.material.emission) > 0.0:
            if was_specular_bounce:
                # Add full emission (unweighted)
                L += beta * rec.material.emission
            else:
                # Calculate MIS weight for hitting light source directly via BRDF sampling
                # Find the light object
                light_obj = obj
                r = rec.t
                pdf_light_a = 1.0 / light_obj.area
                cos_light = max(0.0, glm.dot(-current_ray.direction, rec.normal))
                
                # Conversion to solid angle PDF
                pdf_light_w = pdf_light_a * r * r / max(cos_light, 1e-8)
                
                # Balanced Heuristic
                weight = last_pdf_brdf / (last_pdf_brdf + pdf_light_w)
                L += beta * weight * rec.material.emission
            break
            
        # Last path closure rule
        if depth == d_max - 1:
            # Under MIS, the closure is just the direct light sampling (since path terminates)
            if scene.lights:
                light_obj = random.choice(scene.lights)
                p_light, n_light, pdf_light_a = light_obj.sample()
                d_light = p_light - rec.p
                r = glm.length(d_light)
                
                if r > 1e-4:
                    wi = glm.normalize(d_light)
                    shadow_ray = Ray(rec.p + rec.normal * 0.001, wi)
                    shadow_rec, shadow_obj = scene.intersect(shadow_ray)
                    
                    if shadow_rec is not None and shadow_obj == light_obj and shadow_rec.t >= r - 0.05:
                        fr = rec.material.eval(-current_ray.direction, wi, rec.normal)
                        cos_hit = max(0.0, glm.dot(wi, rec.normal))
                        cos_light = max(0.0, glm.dot(-wi, n_light))
                        
                        # Solid angle PDF of sampling this direction on the light
                        pdf_light_w = pdf_light_a * r * r / max(cos_light, 1e-8)
                        # BRDF PDF in this direction
                        pdf_brdf_w = rec.material.pdf(-current_ray.direction, wi, rec.normal)
                        
                        weight = pdf_light_w / (pdf_light_w + pdf_brdf_w)
                        L += beta * weight * fr * light_obj.material.emission * cos_hit / max(pdf_light_w, 1e-8)
            break
            
        # 1. Direct Light Sampling (Next Event Estimation with MIS)
        if scene.lights:
            light_obj = random.choice(scene.lights)
            p_light, n_light, pdf_light_a = light_obj.sample()
            d_light = p_light - rec.p
            r = glm.length(d_light)
            
            if r > 1e-4:
                wi = glm.normalize(d_light)
                shadow_ray = Ray(rec.p + rec.normal * 0.001, wi)
                shadow_rec, shadow_obj = scene.intersect(shadow_ray)
                
                if shadow_rec is not None and shadow_obj == light_obj and shadow_rec.t >= r - 0.05:
                    fr = rec.material.eval(-current_ray.direction, wi, rec.normal)
                    cos_hit = max(0.0, glm.dot(wi, rec.normal))
                    cos_light = max(0.0, glm.dot(-wi, n_light))
                    
                    pdf_light_w = pdf_light_a * r * r / max(cos_light, 1e-8)
                    pdf_brdf_w = rec.material.pdf(-current_ray.direction, wi, rec.normal)
                    
                    weight = pdf_light_w / (pdf_light_w + pdf_brdf_w)
                    L += beta * weight * fr * light_obj.material.emission * cos_hit / max(pdf_light_w, 1e-8)
                    
        # 2. Path Continuation (BRDF Sampling)
        wi, pdf_val = rec.material.sample(-current_ray.direction, rec.normal)
        if wi is None or pdf_val <= 0.0:
            break
            
        fr = rec.material.eval(-current_ray.direction, wi, rec.normal)
        cos_theta = max(0.0, glm.dot(wi, rec.normal))
        
        beta *= (fr * cos_theta) / pdf_val
        last_pdf_brdf = pdf_val
        
        # Check if it was a specular bounce (strictly, we check if the PDF has a specular component)
        # For simplicity, if material is CookTorrance, we can check if it sampled the specular lobe
        # but to keep it simple, we treat it as diffuse/specular mixture and apply MIS.
        was_specular_bounce = False
        
        current_ray = Ray(rec.p + rec.normal * 0.001, wi)
        
    return L

# ----------------- PASSO 5: Bidirectional Path Tracing (BDPT) -----------------
class PathVertex:
    def __init__(self, p: glm.vec3, normal: glm.vec3, throughput: glm.vec3, material, wi_L: glm.vec3 = None, wo: glm.vec3 = None):
        self.p = glm.vec3(p)
        self.normal = glm.vec3(normal)
        self.throughput = glm.vec3(throughput)
        self.material = material
        self.wi_L = wi_L # incoming direction from previous light vertex
        self.wo = wo # outgoing direction towards camera for camera vertex

def run_passo_5(width: int, height: int, spp: int, d_max: int, scene, camera) -> Image.Image:
    img = Image.new("RGB", (width, height))
    pixels = img.load()
    
    for y in range(height):
        if (y + 1) % 10 == 0 or y == height - 1:
            print(f"Passo 5: Renderizando linha {y+1}/{height} ({(y+1)/height*100:.1f}%)", end="\r")
            
        for x in range(width):
            pixel_color = glm.vec3(0.0)
            
            for s in range(spp):
                pixel_color += trace_path_bdpt(x, y, width, height, scene, camera, d_max)
                
            pixel_color /= spp
            pixels[x, y] = to_ldr(pixel_color)
            
    print()
    return img

def trace_path_bdpt(x: int, y: int, width: int, height: int, scene, camera, d_max: int) -> glm.vec3:
    # 1. Generate Camera Subpath
    camera_subpath = []
    cam_ray = camera.get_ray(x, y, width, height, jitter=True)
    
    beta_C = glm.vec3(1.0)
    current_ray = cam_ray
    
    for depth in range(d_max):
        rec, obj = scene.intersect(current_ray)
        if rec is None:
            break
            
        # Store camera vertex
        camera_subpath.append(PathVertex(
            p=rec.p,
            normal=rec.normal,
            throughput=beta_C,
            material=rec.material,
            wo=-current_ray.direction
        ))
        
        # If we hit light, stop camera subpath tracing
        if glm.length(rec.material.emission) > 0.0:
            break
            
        # Sample next ray
        wi, pdf_val = rec.material.sample(-current_ray.direction, rec.normal)
        if wi is None or pdf_val <= 0.0:
            break
            
        fr = rec.material.eval(-current_ray.direction, wi, rec.normal)
        cos_theta = max(0.0, glm.dot(wi, rec.normal))
        
        beta_C *= (fr * cos_theta) / pdf_val
        current_ray = Ray(rec.p + rec.normal * 0.001, wi)

    # 2. Generate Light Subpath
    light_subpath = []
    if scene.lights:
        light_obj = random.choice(scene.lights)
        p_light, n_light, pdf_light_a = light_obj.sample()
        
        # Cosine weighted emission direction
        r1 = random.random()
        r2 = random.random()
        phi = 2.0 * math.pi * r1
        z = math.sqrt(r2)
        r = math.sqrt(1.0 - r2)
        
        from materials import get_onb
        U, V, N = get_onb(n_light)
        wi_L = x_l = r * math.cos(phi) * U + r * math.sin(phi) * V + z * N
        wi_L = glm.normalize(wi_L)
        
        # Initial light throughput
        # L_init = Le * area * pi
        beta_L = light_obj.material.emission * light_obj.area * math.pi
        
        # Store light vertex y0
        light_subpath.append(PathVertex(
            p=p_light,
            normal=n_light,
            throughput=beta_L,
            material=light_obj.material
        ))
        
        # Trace light path
        current_ray = Ray(p_light + n_light * 0.001, wi_L)
        
        for depth in range(1, d_max):
            rec, obj = scene.intersect(current_ray)
            if rec is None or glm.length(rec.material.emission) > 0.0:
                break
                
            light_subpath.append(PathVertex(
                p=rec.p,
                normal=rec.normal,
                throughput=beta_L,
                material=rec.material,
                wi_L=-current_ray.direction
            ))
            
            # Sample next bounce
            wi, pdf_val = rec.material.sample(-current_ray.direction, rec.normal)
            if wi is None or pdf_val <= 0.0:
                break
                
            fr = rec.material.eval(-current_ray.direction, wi, rec.normal)
            cos_theta = max(0.0, glm.dot(wi, rec.normal))
            
            beta_L *= (fr * cos_theta) / pdf_val
            current_ray = Ray(rec.p + rec.normal * 0.001, wi)

    # 3. Connection and Radiance accumulation
    L = glm.vec3(0.0)
    
    # Strategy: uniform heuristic weight = 1.0 / k for path of length k = i + j + 1
    
    # 3a. Direct hit on light from camera path (j = -1, length k = i)
    for i, x_vertex in enumerate(camera_subpath):
        if glm.length(x_vertex.material.emission) > 0.0:
            k = i + 1
            weight = 1.0 / k
            L += x_vertex.throughput * x_vertex.material.emission * weight
            
    # 3b. Connect camera vertices x_i (i >= 1) to light vertices y_j (j >= 0)
    # Total path length: k = i + j + 1
    for i, x_vertex in enumerate(camera_subpath):
        # Camera path index is 1-based (i + 1 vertices)
        # vertex index is i
        camera_len = i + 1
        
        # We can't connect if camera vertex is on the light
        if glm.length(x_vertex.material.emission) > 0.0:
            continue
            
        for j, y_vertex in enumerate(light_subpath):
            light_len = j + 1
            path_len = camera_len + light_len
            
            if path_len > d_max:
                continue
                
            # Test visibility between x_vertex and y_vertex
            d_conn = y_vertex.p - x_vertex.p
            r = glm.length(d_conn)
            if r < 1e-4:
                continue
                
            wi_C = d_conn / r
            wi_L = -wi_C
            
            # Shadow ray
            shadow_ray = Ray(x_vertex.p + x_vertex.normal * 0.001, wi_C)
            shadow_rec, shadow_obj = scene.intersect(shadow_ray)
            
            # Check visibility
            visible = False
            if shadow_rec is None or shadow_rec.t >= r - 0.05:
                visible = True
            elif j == 0 and shadow_obj == scene.lights[0] and shadow_rec.t >= r - 0.05:
                visible = True
                
            if visible:
                # Camera BRDF
                fr_C = x_vertex.material.eval(x_vertex.wo, wi_C, x_vertex.normal)
                
                # Light BRDF (or emission if j == 0)
                if j == 0:
                    fr_L = glm.vec3(1.0) # Already integrated in y_vertex.throughput
                else:
                    fr_L = y_vertex.material.eval(y_vertex.wi_L, wi_L, y_vertex.normal)
                    
                cos_C = max(0.0, glm.dot(wi_C, x_vertex.normal))
                cos_L = max(0.0, glm.dot(wi_L, y_vertex.normal))
                G = (cos_C * cos_L) / (r * r)
                
                L_conn = x_vertex.throughput * fr_C * G * fr_L * y_vertex.throughput
                
                # Weighting: Uniform Heuristic
                weight = 1.0 / path_len
                
                L += L_conn * weight
                
    return L
