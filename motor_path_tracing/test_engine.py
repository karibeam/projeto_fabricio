import subprocess
import os
import sys

def run_step_test(step):
    print(f"\n--- Testando Passo {step} ---")
    cmd = [
        "python3", "main.py",
        "--step", str(step),
        "--width", "40",
        "--height", "30",
        "--spp", "2",
        "--d_max", "3",
        "--use_filter"
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Passo {step} executado com sucesso.")
        print(res.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar o Passo {step}:")
        print(e.stderr)
        return False

def main():
    # Mudar diretório de execução para o diretório deste script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("Iniciando testes de fumaça (smoke tests)...")
    
    # Limpar pasta de output antiga se houver
    if os.path.exists("output"):
        for f in os.listdir("output"):
            os.remove(os.path.join("output", f))
            
    success = True
    for step in range(1, 6):
        if not run_step_test(step):
            success = False
            break
            
    if not success:
        print("\n[FALHA] Alguns passos falharam nos testes rápidos.")
        sys.exit(1)
        
    # Verificar se as imagens foram criadas
    expected_files = [
        "passo1_path_tracing_basico.png",
        "passo2_roleta_russa.png",
        "passo3_mis.png",
        "passo4_microfacets.png",
        "passo5_bdpt.png"
    ]
    
    missing_files = []
    for f in expected_files:
        path = os.path.join("output", f)
        if not os.path.exists(path):
            missing_files.append(f)
            
    if missing_files:
        print(f"\n[FALHA] Ficheiros em falta: {missing_files}")
        sys.exit(1)
        
    print("\n[SUCESSO] Todos os passos rodaram sem erros e geraram os ficheiros PNG esperados!")

if __name__ == "__main__":
    main()
