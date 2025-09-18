"""
SCRIPT DE EXECUÇÃO SIMPLIFICADO
Sistema Multi-Agente para Automação VR/VA
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Instala as dependências necessárias"""
    print("📦 Instalando dependências...")
    requirements = [
        'pandas>=2.0.0',
        'openpyxl>=3.1.0',
        'numpy>=1.24.0',
        'python-dateutil>=2.8.0'
    ]
    
    for req in requirements:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', req])
    
    print("✅ Dependências instaladas com sucesso!\n")

def check_files():
    """Verifica se os arquivos necessários estão presentes"""
    required_files = [
        'ATIVOS.xlsx',
        'FÉRIAS.xlsx',
        'ADMISSÃO ABRIL.xlsx',
        'DESLIGADOS.xlsx',
        'AFASTAMENTOS.xlsx',
        'APRENDIZ.xlsx',
        'ESTÁGIO.xlsx',
        'EXTERIOR.xlsx',
        'Base dias uteis.xlsx',
        'Base sindicato x valor.xlsx'
    ]
    
    missing_files = []
    input_dir = Path('input')
    for file in required_files:
        if not (input_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("⚠️  ATENÇÃO: Os seguintes arquivos não foram encontrados:")
        for file in missing_files:
            print(f"   • {file}")
        print("\n🔍 Certifique-se de que todos os arquivos estejam na pasta atual.")
        return False
    
    print("✅ Todos os arquivos necessários foram encontrados!\n")
    return True

def create_output_folder():
    """Cria a pasta de saída se não existir"""
    output_path = Path("output")
    output_path.mkdir(exist_ok=True)
    (output_path / "logs").mkdir(exist_ok=True)
    print("📁 Pasta de saída criada/verificada: ./output/\n")

def run_main_system():
    """Executa o sistema principal"""
    print("🚀 Iniciando o Sistema Multi-Agente...\n")
    
    # Importar e executar o sistema principal
    try:
        # Importar o módulo principal (assumindo que está salvo como vr_automation.py)
        from vr_automation_agent import main
        main()
    except ImportError:
        print("❌ Erro: O arquivo 'vr_automation_agent.py' não foi encontrado.")
        print("   Por favor, salve o código principal como 'vr_automation_agent.py'")
        return False
    except Exception as e:
        print(f"❌ Erro durante a execução: {e}")
        return False
    
    return True

def display_instructions():
    """Exibe instruções de uso"""
    print("""
╔════════════════════════════════════════════════════════════════════╗
║           SISTEMA MULTI-AGENTE PARA AUTOMAÇÃO VR/VA               ║
╠════════════════════════════════════════════════════════════════════╣
║  INSTRUÇÕES DE USO:                                                ║
║                                                                     ║
║  1. Certifique-se de que todos os arquivos Excel estão na pasta   ║
║     atual junto com este script                                    ║
║                                                                     ║
║  2. O sistema irá:                                                 ║
║     • Instalar as dependências necessárias                         ║
║     • Verificar a presença dos arquivos                           ║
║     • Executar o processamento multi-agente                       ║
║     • Gerar o arquivo 'VR MENSAL 05.2025.xlsx' em ./output/       ║
║                                                                     ║
║  3. Logs de execução serão salvos em ./output/logs/              ║
║                                                                     ║
╚════════════════════════════════════════════════════════════════════╝
    """)

if __name__ == "__main__":
    print("""
    ============================================================
                  AUTOMAÇÃO VR/VA - SETUP & EXECUÇÃO
    ============================================================
    Grupo: Agentes em Treinamento
    Componente: Nicolas França
    ============================================================
    """)
    
    # Exibir instruções
    display_instructions()
    
    input("\n🔵 Pressione ENTER para iniciar o processo...")
    print("\n" + "="*60 + "\n")
    
    # 1. Instalar dependências
    try:
        install_requirements()
    except Exception as e:
        print(f"❌ Erro ao instalar dependências: {e}")
        sys.exit(1)
    
    # 2. Verificar arquivos
    if not check_files():
        print("\n⚠️  Por favor, adicione os arquivos faltantes e execute novamente.")
        sys.exit(1)
    
    # 3. Criar pasta de saída
    create_output_folder()
    
    # 4. Executar sistema principal
    if run_main_system():
        print("\n" + "="*60)
        print("✅ PROCESSO CONCLUÍDO COM SUCESSO!")
        print("📄 Arquivo gerado: ./output/VR MENSAL 05.2025.xlsx")
        print("📊 Logs disponíveis em: ./output/logs/")
        print("="*60)
    else:
        print("\n❌ O processo não foi concluído. Verifique os logs para mais detalhes.")
        sys.exit(1)
