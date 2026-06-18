import argparse
import os
import glm
from PIL import Image, ImageFilter

from scene import build_cornell_box
from camera import Camera
from materials import Lambertian, Emissive, CookTorrance
import integrators

def parse_args():
    parser = argparse.ArgumentParser(description="Motor de Renderização Path Tracing (Cornell Box)")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4, 5], default=None,
                        help="Passo específico a ser executado (1 a 5). Se não for passado, executa todos.")
    parser.add_argument("--width", type=int, default=800, help="Largura da imagem gerada")
    parser.add_argument("--height", type=int, default=600, help="Altura da imagem gerada")
    parser.add_argument("--spp", type=int, default=25, help="Quantidade de amostras por píxel")
    parser.add_argument("--d_max", type=int, default=4, help="Profundidade máxima do caminho")
    parser.add_argument("--use_filter", action="store_true", help="Ativa a flag de pós-processamento/suavização")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Garantir que a pasta de output existe
    os.makedirs("output", exist_ok=True)
    
    # Dicionário mapeando passos aos nomes de ficheiros correspondentes
    step_filenames = {
        1: "passo1_path_tracing_basico.png",
        2: "passo2_roleta_russa.png",
        3: "passo3_mis.png",
        4: "passo4_microfacets.png",
        5: "passo5_bdpt.png"
    }
    
    steps_to_run = [args.step] if args.step else [1, 2, 3, 4, 5]
    
    # Inicialização da câmara
    aspect_ratio = args.width / args.height
    camera = Camera(
        lookfrom=glm.vec3(278, 278, -800),
        lookat=glm.vec3(278, 278, 278),
        vup=glm.vec3(0, 1, 0),
        vfov_degrees=40.0,
        aspect_ratio=aspect_ratio
    )
    
    # Definir materiais base comuns
    mat_red = Lambertian(albedo=glm.vec3(0.65, 0.05, 0.05))
    mat_green = Lambertian(albedo=glm.vec3(0.12, 0.45, 0.15))
    mat_white = Lambertian(albedo=glm.vec3(0.73, 0.73, 0.73))
    mat_light = Emissive(emission=glm.vec3(15.0, 15.0, 15.0))
    
    for current_step in steps_to_run:
        print(f"\n========================================")
        print(f"A executar o Passo {current_step}...")
        print(f"Configuração: {args.width}x{args.height} | {args.spp} SPP | d_max: {args.d_max} | Filtro: {args.use_filter}")
        print(f"========================================")
        
        # Configurar materiais baseados no passo
        # No Passo 4, usamos materiais Cook-Torrance (PBR)
        if current_step == 4:
            # Esfera metálica lisa
            sphere_mat = CookTorrance(albedo=glm.vec3(0.95, 0.95, 0.95), roughness=0.05, metalness=1.0)
            # Caixa plástica azulada rugosa
            box_mat = CookTorrance(albedo=glm.vec3(0.1, 0.5, 0.9), roughness=0.4, metalness=0.0)
        else:
            # Passos 1, 2, 3 e 5 usam materiais puramente Lambertianos (difusos)
            sphere_mat = mat_white
            box_mat = mat_white
            
        materials_dict = {
            'red': mat_red,
            'green': mat_green,
            'white': mat_white,
            'light': mat_light,
            'sphere': sphere_mat,
            'box': box_mat
        }
        
        # Construir cena
        scene = build_cornell_box(materials_dict)
        
        # Escolher integrador correspondente
        if current_step == 1:
            image = integrators.run_passo_1(args.width, args.height, args.spp, args.d_max, scene, camera)
        elif current_step == 2:
            image = integrators.run_passo_2(args.width, args.height, args.spp, args.d_max, scene, camera)
        elif current_step == 3:
            image = integrators.run_passo_3(args.width, args.height, args.spp, args.d_max, scene, camera)
        elif current_step == 4:
            image = integrators.run_passo_4(args.width, args.height, args.spp, args.d_max, scene, camera)
        elif current_step == 5:
            image = integrators.run_passo_5(args.width, args.height, args.spp, args.d_max, scene, camera)
            
        # Pós-processamento: filtro Gaussiano
        if args.use_filter:
            print("A aplicar filtro de suavização Gaussian Blur...")
            image = image.filter(ImageFilter.GaussianBlur(radius=1.0))
            
        # Guardar imagem no diretório output/
        filename = step_filenames[current_step]
        output_path = os.path.join("output", filename)
        image.save(output_path)
        print(f"Passo {current_step} concluído! Imagem guardada em: {output_path}")

if __name__ == "__main__":
    main()
